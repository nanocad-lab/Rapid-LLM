#!/usr/bin/env python3
# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightweight overlap tuner using golden-section search (no SciPy).

Global knobs (edit below):
  - RUN_INFERENCE / RUN_TRAINING: enable which suites to optimize
  - INF_DEVICES / TRAIN_DEVICES: which devices to run from validation CSVs
  - GOLD_TOL / GOLD_MAX_ITER: search tolerance and iteration cap

Current behavior:
  - Inference: optimize network.overlap.tp_overlap using validation_scripts/nvidia_inf.py
  - Training:  optimize network.overlap.tp_sp_overlap using validation_scripts/nvidia_train.py

Extensible: add more evaluators that return an average pct error for a target overlap key.
"""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_inf, nvidia_train  # type: ignore
from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite  # type: ignore
from tools import mod_field

# ======= Global config =======
RUN_INFERENCE = True
RUN_TRAINING = True
# Set to True to write best overlaps back to hardware config files (rounded to 2 decimals).
APPLY_BEST = True

INF_DEVICES: Sequence[str] = ("A100",)
TRAIN_DEVICES: Sequence[str] = ("A100_korthi",)

GOLD_TOL = 0.01  # stop when interval width < GOLD_TOL
GOLD_MAX_ITER = 20

# =============================


def _merge_dict(target: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, val in updates.items():
        if isinstance(val, Mapping):
            current = target.get(key, {})
            if not isinstance(current, dict):
                current = {}
            target[key] = _merge_dict(dict(current), val)
        else:
            target[key] = val
    return target


def _with_overlap(spec: ValidationSpec, overlap_key: str, overlap_value: float) -> ValidationSpec:
    base_overrides: Dict[str, Any] = {}
    if spec.hardware_overrides and isinstance(spec.hardware_overrides, Mapping):
        base_overrides = _merge_dict({}, spec.hardware_overrides)
    base_overrides = _merge_dict(base_overrides, {"network": {"overlap": {overlap_key: float(overlap_value)}}})
    return ValidationSpec(
        label=spec.label,
        model_overrides=spec.model_overrides,
        hardware_overrides=base_overrides,
        metadata=spec.metadata,
        model_config_path=spec.model_config_path,
        hardware_config_path=spec.hardware_config_path,
        order=spec.order,
    )


def _avg_pct_error(errors: Iterable[float]) -> float:
    vals = [e for e in errors if not math.isnan(e)]
    if not vals:
        return float("inf")
    return sum(vals) / len(vals)


def _all_hardware_configs() -> List[Path]:
    roots = [
        PROJECT_ROOT / "configs" / "hardware-config",
        PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config",
    ]
    paths: List[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*.yaml")))
    return paths


def eval_inference(tp_overlap: float) -> float:
    """Average pct error across INF_DEVICES for given tp_overlap."""
    all_errors: List[float] = []
    for device in INF_DEVICES:
        specs, actual_lookup, base_model_path, base_hw_path = nvidia_inf.build_specs_for_device(
            device, network_ignored=True
        )
        specs = [_with_overlap(spec, "tp_overlap", tp_overlap) for spec in specs]
        with io.StringIO() as buf, redirect_stdout(buf):
            results = run_validation_suite(
                specs,
                base_model_config_path=base_model_path,
                base_hardware_config_path=base_hw_path,
                result_parser=nvidia_inf.parse_inference_time,
                run_perf_path=nvidia_inf.RUN_PERF,
            )
        rows = nvidia_inf.compute_pct_errors(results, actual_lookup)
        all_errors.extend(float(row["pct_error"]) for row in rows)
    return _avg_pct_error(all_errors)


def eval_training(tp_sp_overlap: float) -> float:
    """Average pct error across TRAIN_DEVICES for given tp_sp_overlap."""
    all_errors: List[float] = []
    for device in TRAIN_DEVICES:
        specs, actual_lookup, base_model_path, base_hw_path = nvidia_train.build_specs_for_device(device)
        specs = [_with_overlap(spec, "tp_sp_overlap", tp_sp_overlap) for spec in specs]
        with io.StringIO() as buf, redirect_stdout(buf):
            results = run_validation_suite(
                specs,
                base_model_config_path=base_model_path,
                base_hardware_config_path=base_hw_path,
                result_parser=nvidia_train.parse_training_time,
                run_perf_path=nvidia_train.RUN_PERF,
            )
        rows = nvidia_train.compute_pct_errors(results, actual_lookup)
        all_errors.extend(float(row["pct_error"]) for row in rows)
    return _avg_pct_error(all_errors)


@dataclass
class SearchResult:
    best_x: float
    best_y: float
    evaluations: List[Tuple[float, float]]


def golden_section_search(
    fn: Callable[[float], float],
    lo: float = 0.0,
    hi: float = 1.0,
    tol: float = GOLD_TOL,
    max_iter: int = GOLD_MAX_ITER,
    log_fn: Optional[Callable[[int, int, float, float], None]] = None,
) -> SearchResult:
    phi = (1 + 5 ** 0.5) / 2
    inv_phi = 1 / phi
    inv_phi_sq = inv_phi ** 2

    a, b = lo, hi
    h = b - a
    if h <= tol:
        mid = (a + b) / 2
        val = fn(mid)
        return SearchResult(best_x=mid, best_y=val, evaluations=[(mid, val)])

    n = int(math.ceil(math.log(tol / h) / math.log(inv_phi)))
    c = a + inv_phi_sq * h
    d = a + inv_phi * h
    yc = fn(c)
    yd = fn(d)
    if log_fn:
        log_fn(1, max_iter, c, yc)
        log_fn(2, max_iter, d, yd)
    evals: List[Tuple[float, float]] = [(c, yc), (d, yd)]

    iter_idx = 2
    for _ in range(min(n, max_iter)):
        if yc < yd:
            b = d
            d, yd = c, yc
            h = inv_phi * h
            c = a + inv_phi_sq * h
            yc = fn(c)
            evals.append((c, yc))
        else:
            a = c
            c, yc = d, yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = fn(d)
            evals.append((d, yd))
        iter_idx += 1
        if log_fn:
            log_fn(iter_idx, max_iter, evals[-1][0], evals[-1][1])
        if h < tol:
            break

    best_x, best_y = min(evals, key=lambda t: t[1])
    return SearchResult(best_x=best_x, best_y=best_y, evaluations=evals)


def main() -> None:
    if RUN_INFERENCE:
        print("=== Optimizing tp_overlap for inference ===")
        res = golden_section_search(
            eval_inference,
            log_fn=lambda i, m, x, y: print(f"[{i}/{m}] Tried {x:.5f} --> {y:.2f}"),
        )
        print(f"Best tp_overlap={res.best_x:.4f}, avg pct error={res.best_y:.4f}")
        for x, y in res.evaluations:
            print(f"  tried {x:.4f} -> {y:.4f}")
        err0 = eval_inference(0.0)
        err1 = eval_inference(1.0)
        print(f"  error at 0.0: {err0:.4f}")
        print(f"  error at 1.0: {err1:.4f}")
        if APPLY_BEST:
            best_val = round(res.best_x, 2)
            print(f"Applying tp_overlap={best_val:.2f} to all hardware configs")
            for cfg_path in _all_hardware_configs():
                msg = mod_field.set_field(cfg_path, "network.overlap.tp_overlap", best_val, dry_run=False)
                print(f"  {msg}")

    if RUN_TRAINING:
        print("\n=== Optimizing tp_sp_overlap for training ===")
        res = golden_section_search(
            eval_training,
            log_fn=lambda i, m, x, y: print(f"[{i}/{m}] Tried {x:.5f} --> {y:.2f}"),
        )
        print(f"Best tp_sp_overlap={res.best_x:.4f}, avg pct error={res.best_y:.4f}")
        for x, y in res.evaluations:
            print(f"  tried {x:.4f} -> {y:.4f}")
        err0 = eval_training(0.0)
        err1 = eval_training(1.0)
        print(f"  error at 0.0: {err0:.4f}")
        print(f"  error at 1.0: {err1:.4f}")
        if APPLY_BEST:
            best_val = round(res.best_x, 2)
            print(f"Applying tp_sp_overlap={best_val:.2f} to all hardware configs")
            for cfg_path in _all_hardware_configs():
                msg = mod_field.set_field(cfg_path, "network.overlap.tp_sp_overlap", best_val, dry_run=False)
                print(f"  {msg}")


if __name__ == "__main__":
    main()
