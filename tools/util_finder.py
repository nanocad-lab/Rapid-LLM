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
Utility tuner (core util, HBM bandwidth util) for training validation.

Modeled after overlap_finder.py but searches in two dimensions using a
coordinate-wise golden-section search. Objective is average absolute percent
error across Korthi+Selene training validation suites (nvidia_train.py).

Global knobs (edit below):
  - DEVICES: which training devices to validate (defaults: Korthi + Selene)
  - APPLY_BEST: write the best utils back to all hardware configs via mod_field
  - RANGES/TOL/ITER: search bounds and golden search controls
  - COORD_STEPS: number of coordinate-descent passes (keep small; each eval is slow)
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_train, nvidia_inf, uci_train  # type: ignore
from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite  # type: ignore
from tools import mod_field

# ======= Global config =======
MODE: str = "train"  # default; can be overridden via CLI
DEVICES_TRAIN: Sequence[str] = ("A100_korthi", "A100_selene")
DEVICES_INF_DEFAULT: Sequence[str] = ("A100",)  # inference should optimize one device at a time
DEVICES: Sequence[str] = DEVICES_TRAIN
APPLY_BEST = False
IGNORE_NETWORK = True  # for inference mode
OBJECTIVE: str = "rms"  # "rms", "max", or "avg"

# Search bounds for utilities (floats in (0,1]). Set to None to skip tuning that axis.
CORE_UTIL_RANGE: Optional[Tuple[float, float]] = (0.8, 1.0)
HBM_UTIL_RANGE: Optional[Tuple[float, float]] = None
GOLD_TOL = 0.02       # stop when interval width < GOLD_TOL
GOLD_MAX_ITER = 6     # cap per-axis golden iterations
COORD_STEPS = 2       # how many coordinate-descent passes (keep small)
# =============================

# Exclude specific heavy/slow validation cases by metadata match.
EXCLUDE_TESTS_TRAIN: Tuple[Dict[str, Any], ...] = (
    # Skip 3k-GPU Selene GPT-1T case (dp=6, tp=8, pp=64, mb=512, batch=3072).
    {"device": "A100_selene", "model": "GPT 1T", "pp": 64},
    {"device": "A100_korthi", "model": "GPT 22B", "pp": 1},
    {"variant": "DDP", "cp": 4},
    {"variant": "DDP", "tp": 4},
    {"variant": "DDP", "cp": 2, "tp": 2},
    {"variant": "FSDP"},
)

EXCLUDE_TESTS_INF: Tuple[Dict[str, Any], ...] = (
    # Example: no exclusions yet for inference; keep empty or add as needed.
)


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


def _with_utils(spec: ValidationSpec, core_util: Optional[float], hbm_util: Optional[float]) -> ValidationSpec:
    base_overrides: Dict[str, Any] = {}
    if spec.hardware_overrides and isinstance(spec.hardware_overrides, Mapping):
        base_overrides = _merge_dict({}, spec.hardware_overrides)
    util_overrides: Dict[str, Any] = {"tech_param": {}}
    if core_util is not None:
        util_overrides["tech_param"]["core"] = {"util": float(core_util)}
    if hbm_util is not None:
        util_overrides["tech_param"]["DRAM"] = {"util": float(hbm_util)}
    if not util_overrides["tech_param"]:
        return ValidationSpec(
            label=spec.label,
            model_overrides=spec.model_overrides,
            hardware_overrides=base_overrides,
            metadata=spec.metadata,
            model_config_path=spec.model_config_path,
            hardware_config_path=spec.hardware_config_path,
            order=spec.order,
        )
    base_overrides = _merge_dict(base_overrides, util_overrides)
    return ValidationSpec(
        label=spec.label,
        model_overrides=spec.model_overrides,
        hardware_overrides=base_overrides,
        metadata=spec.metadata,
        model_config_path=spec.model_config_path,
        hardware_config_path=spec.hardware_config_path,
        order=spec.order,
    )


def _max_abs_pct_error(errors: Iterable[float]) -> float:
    vals = [e for e in errors if not math.isnan(e)]
    if not vals:
        return float("inf")
    return max(vals)


def _rms_pct_error(errors: Iterable[float]) -> float:
    vals = [e for e in errors if not math.isnan(e)]
    if not vals:
        return float("inf")
    squared_sum = sum(e * e for e in vals)
    mean_squared = squared_sum / len(vals)
    return math.sqrt(mean_squared)


def _avg_abs_pct_error(errors: Iterable[float]) -> float:
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


def _should_exclude(meta: Mapping[str, Any], mode: str) -> bool:
    rules = EXCLUDE_TESTS_TRAIN if mode == "train" else EXCLUDE_TESTS_INF
    for rule in rules:
        matches = True
        for key, val in rule.items():
            if str(meta.get(key)) != str(val):
                matches = False
                break
        if matches:
            return True
    return False


def _filter_specs(
    specs: Sequence[ValidationSpec],
    actual_lookup: Mapping[Any, float],
    *,
    mode: str,
) -> Tuple[List[ValidationSpec], Dict[Any, float]]:
    kept: List[ValidationSpec] = []
    kept_lookup: Dict[Any, float] = {}
    for spec in specs:
        meta = spec.metadata or {}
        if _should_exclude(meta, mode):
            continue
        if mode == "train":
            if "variant" in meta:
                key = (
                    str(meta.get("variant")),
                    int(meta.get("tp")),
                    int(meta.get("pp")),
                    int(meta.get("cp")),
                    int(meta.get("dp")),
                )
            else:
                key = (
                    meta.get("model"),
                    int(meta.get("batch")),
                    int(meta.get("mb")),
                    int(meta.get("dp")),
                    int(meta.get("tp")),
                    int(meta.get("pp")),
                    int(meta.get("cp")),
                    bool(meta.get("tp_sp")),
                    str(meta.get("recomputation")),
                )
        else:
            key = (
                meta.get("model"),
                int(meta.get("tp")),
            )
        if key in actual_lookup:
            kept_lookup[key] = actual_lookup[key]
        kept.append(spec)
    return kept, kept_lookup


@dataclass
class SearchResult:
    best_x: float
    best_y: float
    evaluations: List[Tuple[float, float]]


def golden_section_search(
    fn: Callable[[float], float],
    lo: float,
    hi: float,
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


class UtilTuner:
    def __init__(self) -> None:
        self._eval_cache: Dict[Tuple[float, float], float] = {}

    def eval_pair(self, core_util: float, hbm_util: float) -> float:
        key = (round(core_util, 6), round(hbm_util, 6))
        if key in self._eval_cache:
            print(f"[CACHE HIT] core.util={core_util:.4f}, DRAM.util={hbm_util:.4f} -> {self._eval_cache[key]:.3f}")
            return self._eval_cache[key]

        all_errors: List[float] = []
        if MODE == "train":
            try:
                uci_model_cfg = str(uci_train.DEFAULT_MODEL_CONFIG)
                uci_hw_cfg = str(uci_train.DEFAULT_HW_CONFIG)
                uci_specs, uci_lookup = uci_train.build_specs(("ddp",), uci_model_cfg, uci_hw_cfg)
                uci_specs, uci_lookup = _filter_specs(uci_specs, uci_lookup, mode=MODE)
                uci_specs = [_with_utils(spec, core_util, hbm_util) for spec in uci_specs]
                print(f"[EVAL] mode=train, UCI specs={len(uci_specs)}, core.util={core_util:.4f}, DRAM.util={hbm_util:.4f}")
                with io.StringIO() as buf, redirect_stdout(buf):
                    uci_results = run_validation_suite(
                        uci_specs,
                        base_model_config_path=uci_model_cfg,
                        base_hardware_config_path=uci_hw_cfg,
                        result_parser=uci_train.parse_training_time,
                        run_perf_path=uci_train.RUN_PERF,
                    )
                uci_rows = uci_train.compute_pct_errors(uci_results, uci_lookup)
                all_errors.extend(float(r.get("pct_error", float("nan"))) for r in uci_rows)
                print(f"[RESULTS] mode=train, UCI pct_errors={ [float(r.get('pct_error', float('nan'))) for r in uci_rows] }")
            except Exception as e:
                print(f"[WARN] UCI training validation failed: {e}")

        module = nvidia_train if MODE == "train" else nvidia_inf
        parser = nvidia_train.parse_training_time if MODE == "train" else nvidia_inf.parse_inference_time
        for device in DEVICES:
            if MODE == "train":
                specs, actual_lookup, base_model_path, base_hw_path = module.build_specs_for_device(device)
            else:
                specs, actual_lookup, base_model_path, base_hw_path = module.build_specs_for_device(
                    device, network_ignored=IGNORE_NETWORK
                )
            specs, actual_lookup = _filter_specs(specs, actual_lookup, mode=MODE)
            if not specs:
                print(f"[SKIP DEVICE] {device}: no specs after filtering.")
                continue
            specs = [_with_utils(spec, core_util, hbm_util) for spec in specs]
            print(f"[EVAL] mode={MODE}, device={device}, specs={len(specs)}, core.util={core_util:.4f}, DRAM.util={hbm_util:.4f}")
            with io.StringIO() as buf, redirect_stdout(buf):
                results = run_validation_suite(
                    specs,
                    base_model_config_path=base_model_path,
                    base_hardware_config_path=base_hw_path,
                    result_parser=parser,
                    run_perf_path=module.RUN_PERF,
                )
            rows = module.compute_pct_errors(results, actual_lookup)
            if MODE == "train":
                avg_error = _avg_abs_pct_error(float(row["pct_error"]) for row in rows)
                all_errors.extend([avg_error])
            else:
                all_errors.extend(float(row["pct_error"]) for row in rows)
            print(f"[RESULTS] mode={MODE}, device={device}, pct_errors={ [float(row['pct_error']) for row in rows] }")

        if OBJECTIVE == "max":
            score = _max_abs_pct_error(all_errors)
        elif OBJECTIVE == "avg":
            score = _avg_abs_pct_error(all_errors)
        else:
            score = _rms_pct_error(all_errors)
        self._eval_cache[key] = score
        label_map = {"max": "MAX", "rms": "RMS", "avg": "AVG"}
        label = label_map.get(OBJECTIVE, "RMS")
        print(f"[EVAL DONE] core.util={core_util:.4f}, DRAM.util={hbm_util:.4f}, {label} error={score:.3f}")
        return score

    def optimize(self) -> Tuple[Optional[float], Optional[float], float]:
        core_util = None if CORE_UTIL_RANGE is None else (CORE_UTIL_RANGE[0] + CORE_UTIL_RANGE[1]) / 2
        if HBM_UTIL_RANGE is None:
            hbm_util = 0.5931
        else:
            hbm_util = (HBM_UTIL_RANGE[0] + HBM_UTIL_RANGE[1]) / 2

        if CORE_UTIL_RANGE is None and HBM_UTIL_RANGE is None:
            raise ValueError("At least one of CORE_UTIL_RANGE or HBM_UTIL_RANGE must be set.")

        # If only one axis is active, do a single golden search on that axis.
        if CORE_UTIL_RANGE is None and HBM_UTIL_RANGE is not None:
            h_lo, h_hi = HBM_UTIL_RANGE
            print("Tuning only DRAM.util")
            res_hbm = golden_section_search(
                lambda y: self.eval_pair(core_util or 1.0, y),
                lo=h_lo,
                hi=h_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, y, v: print(f"  [{i}/{m}] hbm_util={y:.4f} -> {v:.3f}"),
            )
            best_val = res_hbm.best_y
            return core_util, res_hbm.best_x, best_val

        if HBM_UTIL_RANGE is None and CORE_UTIL_RANGE is not None:
            c_lo, c_hi = CORE_UTIL_RANGE
            print("Tuning only core.util")
            res_core = golden_section_search(
                lambda x: self.eval_pair(x, hbm_util or 1.0),
                lo=c_lo,
                hi=c_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, x, y: print(f"  [{i}/{m}] core_util={x:.4f} -> {y:.3f}"),
            )
            best_val = res_core.best_y
            return res_core.best_x, hbm_util, best_val

        # Both axes active: coordinate descent.
        core_lo, core_hi = CORE_UTIL_RANGE  # type: ignore[assignment]
        hbm_lo, hbm_hi = HBM_UTIL_RANGE    # type: ignore[assignment]
        core = float(core_util)
        hbm = float(hbm_util)
        best_val = self.eval_pair(core, hbm)

        for step in range(COORD_STEPS):
            print(f"\n--- Coordinate step {step + 1}/{COORD_STEPS} ---")
            # Optimize core util with HBM fixed
            print(f"Optimizing core util (hbm_util={hbm:.4f})")
            res_core = golden_section_search(
                lambda x: self.eval_pair(x, hbm),
                lo=core_lo,
                hi=core_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, x, y: print(f"  [{i}/{m}] core_util={x:.4f} -> {y:.3f}"),
            )
            core = res_core.best_x
            best_val = res_core.best_y

            # Optimize HBM util with core fixed
            print(f"Optimizing HBM util (core_util={core:.4f})")
            res_hbm = golden_section_search(
                lambda y: self.eval_pair(core, y),
                lo=hbm_lo,
                hi=hbm_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, y, v: print(f"  [{i}/{m}] hbm_util={y:.4f} -> {v:.3f}"),
            )
            hbm = res_hbm.best_x
            best_val = min(best_val, res_hbm.best_y)

        best_val = self.eval_pair(core, hbm)
        return core, hbm, best_val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune core/HBM utils to minimize validation RMS error.")
    p.add_argument("--mode", choices=["train", "inf"], default="train", help="Optimization target: training or inference")
    p.add_argument("--objective", choices=["rms", "max", "avg"], default="rms", help="Error objective: RMS, max absolute, or average absolute percent error")
    p.add_argument(
        "--devices",
        nargs="+",
        help="Devices to validate. Training: multiple allowed. Inference: exactly one (e.g., A100 or H100).",
    )
    p.add_argument("--include-network", action="store_true", help="Consider network bandwidth in validation")
    p.add_argument("--apply-best", action="store_true", help="Apply tuned utils to all hardware configs")
    p.add_argument("--core-range", nargs=2, type=float, metavar=("LO", "HI"), help="Search range for core.util (e.g., 0.8 1.0)")
    p.add_argument("--hbm-range", nargs=2, type=float, metavar=("LO", "HI"), help="Search range for DRAM.util (e.g., 0.4 0.6)")
    p.add_argument("--coord-steps", type=int, default=COORD_STEPS, help="Number of coordinate-descent passes")
    p.add_argument("--gold-tol", type=float, default=GOLD_TOL, help="Golden-section tolerance")
    p.add_argument("--gold-max-iter", type=int, default=GOLD_MAX_ITER, help="Golden-section max iterations per axis")
    return p.parse_args()


def main() -> None:
    global MODE, DEVICES, APPLY_BEST, CORE_UTIL_RANGE, HBM_UTIL_RANGE, COORD_STEPS, GOLD_TOL, GOLD_MAX_ITER, IGNORE_NETWORK, OBJECTIVE

    args = _parse_args()
    MODE = args.mode
    OBJECTIVE = args.objective
    APPLY_BEST = bool(args.apply_best)
    COORD_STEPS = int(args.coord_steps)
    GOLD_TOL = float(args.gold_tol)
    GOLD_MAX_ITER = int(args.gold_max_iter)
    IGNORE_NETWORK = not args.include_network

    if args.core_range:
        CORE_UTIL_RANGE = (float(args.core_range[0]), float(args.core_range[1]))
    if args.hbm_range:
        HBM_UTIL_RANGE = (float(args.hbm_range[0]), float(args.hbm_range[1]))

    if args.devices and len(args.devices) > 0:
        DEVICES = tuple(args.devices)
    else:
        DEVICES = DEVICES_TRAIN if MODE == "train" else DEVICES_INF_DEFAULT

    if MODE == "inf":
        if len(DEVICES) != 1:
            print("[ERROR] Inference mode requires exactly one device (e.g., --devices A100 or --devices H100).")
            sys.exit(2)

    print(f"[CONFIG] mode={MODE}, objective={OBJECTIVE}, devices={list(DEVICES)}, core_range={CORE_UTIL_RANGE}, hbm_range={HBM_UTIL_RANGE}, coord_steps={COORD_STEPS}")

    tuner = UtilTuner()
    best_core, best_hbm, best_err = tuner.optimize()
    print("\n=== Best utils ===")
    print(f"  core.util = {best_core if best_core is None else f'{best_core:.4f}'}")
    print(f"  DRAM.util = {best_hbm if best_hbm is None else f'{best_hbm:.4f}'}")
    label_map = {"max": "MAX", "rms": "RMS", "avg": "AVG"}
    print(f"  {label_map.get(OBJECTIVE, 'RMS')} error = {best_err:.3f}")

    if APPLY_BEST:
        core_val = None if best_core is None else round(best_core, 2)
        hbm_val = None if best_hbm is None else round(best_hbm, 2)
        print("\nApplying tuned utils to all hardware configs")
        for cfg_path in _all_hardware_configs():
            if core_val is not None:
                msg1 = mod_field.set_field(cfg_path, "tech_param.core.util", core_val, dry_run=False)
                print(f"  {msg1}")
            if hbm_val is not None:
                msg2 = mod_field.set_field(cfg_path, "tech_param.DRAM.util", hbm_val, dry_run=False)
                print(f"  {msg2}")


if __name__ == "__main__":
    main()
