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
GMap sweep utility.

This tool explores tensor/context/pipeline parallelism combinations that live in
the first network dimension (TP/CP/PP) for a square, power-of-two GPU count.
Each configuration is evaluated by running RAPID-LLM with the
RAPID_GMAP_ONLY=1 environment variable so execution stops immediately after
SCOTCH emits the mapping artefacts.  The script parses the resulting
`first_dim_comm.metrics` file (and the `[GMapDebug]` printouts) to recover the
before/after CommExpan values, computes the percentage delta, and produces both
 a tab-separated report and a scatter plot coloured by % change.

"""

from __future__ import annotations

import argparse
import copy
import io
import math
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from train_timing import TimeCalculationLLM  # noqa: E402
from parallelism_sweep import set_astrasim_cache_mode  # type: ignore

# -----------------------------------------------------------------------------#
# Global configuration                                                         #
# -----------------------------------------------------------------------------#

# Total GPU count in the first dimension (must be a square, power-of-two).
NUM_GPUS = 256
NUM_WORKERS = 95
ASTRA_CACHE_MODE = "NO_CACHE"
TOPOLOGY_VARIANTS = ("Mesh2D", "Torus2D")
# TOPOLOGY_VARIANTS = ("Mesh2D",)
# Hard bounds (inclusive) on each parallelism axis.
TP_BOUNDS = (1, 128)
CP_BOUNDS = (1, 128)
PP_BOUNDS = (1, 128)

# -----------------------------------------------------------------------------#
# Helpers lifted / adapted from existing sweep tools                           #
# -----------------------------------------------------------------------------#


def read_yaml(path: str) -> Dict[str, object]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def determine_model_mode(model_config_path: str) -> str:
    model_dict = read_yaml(model_config_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError(f"model_param.mode must be defined in {model_config_path}")
    return str(mode)


def make_temp_hw_config(
    base_hw_dict: Dict[str, object],
    parallel_settings: Dict[str, int],
    hw_mutator=None,
):
    updated = copy.deepcopy(base_hw_dict)
    parallel_block = updated.setdefault("parallelism", {})
    for key, value in parallel_settings.items():
        if key in {"tp", "cp", "pp", "mb", "tp_sp"}:
            parallel_block[key] = int(value)

    train_block = parallel_block.setdefault("train", {})
    train_block["dp"] = int(parallel_settings.get("dp", train_block.get("dp", 1)) or 1)
    train_block.setdefault("ep", 1)
    train_block.setdefault("tp_ep", True)

    inference_block = parallel_block.setdefault("inference", {})
    inference_block.setdefault("replica_count", 1)
    inference_block.setdefault("moe_dp", 1)

    if hw_mutator:
        hw_mutator(updated)

    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(updated, tmp_file, default_flow_style=False, sort_keys=False)
        tmp_file.flush()
        tmp_file.close()
        hw_config = config.parse_config(tmp_file.name, config_type="hardware")
        return hw_config
    finally:
        try:
            tmp_file.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


# -----------------------------------------------------------------------------#
# Core sweep logic                                                             #
# -----------------------------------------------------------------------------#


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def is_perfect_square(value: int) -> bool:
    root = math.isqrt(value)
    return root * root == value


def enumerate_parallelisms(num_gpus: int) -> List[Dict[str, int]]:
    limit = max(1, num_gpus // 2)
    def _powers_of_two_between(lo: int, hi: int) -> List[int]:
        lo = max(1, lo)
        if hi < lo:
            return []
        vals: List[int] = []
        val = 1
        while val < lo:
            val <<= 1
        while val <= hi:
            vals.append(val)
            val <<= 1
        return vals

    tp_min, tp_max = TP_BOUNDS
    cp_min, cp_max = CP_BOUNDS
    pp_min, pp_max = PP_BOUNDS
    tp_values = _powers_of_two_between(tp_min, min(tp_max, limit))
    cp_values = _powers_of_two_between(cp_min, min(cp_max, limit))
    pp_values = _powers_of_two_between(pp_min, min(pp_max, limit))
    combos: List[Dict[str, int]] = []
    for tp in tp_values:
        for cp in cp_values:
            for pp in pp_values:
                if tp * cp * pp == num_gpus:
                    combos.append({"tp": tp, "cp": cp, "pp": pp})
    combos.sort(key=lambda item: (item["tp"], item["cp"], item["pp"]))
    return combos


def update_first_dimension_size(hw_dict: Dict[str, object], num_gpus: int, topology: str) -> None:
    network = hw_dict.get("network")
    if not isinstance(network, dict):
        return
    dimensions = network.get("dimensions")
    if isinstance(dimensions, list) and dimensions:
        first = dimensions[0]
        if isinstance(first, dict):
            first["size"] = int(num_gpus)
            topology_block = first.setdefault("topology", {})
            topology_block["type"] = topology


def parse_metrics_file(path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(path):
        return None, None
    section: Optional[str] = None
    before = after = None
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Before"):
                section = "before"
                continue
            if line.startswith("After"):
                section = "after"
                continue
            if "CommExpan=" in line:
                try:
                    value = float(line.split("CommExpan=")[1].split("\t", 1)[0])
                except (ValueError, IndexError):
                    value = None
                if value is None:
                    continue
                if section == "before" and before is None:
                    before = value
                elif section == "after" and after is None:
                    after = value
    return before, after


_COMMEXPAN_REGEX = re.compile(
    r"CommExpan before\s+([0-9eE+\-\.]+)\s+after\s+([0-9eE+\-\.]+)"
)


def parse_stdout_for_commexpan(stdout_text: str) -> Tuple[Optional[float], Optional[float]]:
    before = after = None
    for line in stdout_text.splitlines():
        if "CommExpan" not in line:
            continue
        match = _COMMEXPAN_REGEX.search(line)
        if not match:
            continue
        try:
            before = float(match.group(1))
            after = float(match.group(2))
        except ValueError:
            continue
    return before, after


def compute_delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before * 100.0


def _format_delta(delta: Optional[float]) -> str:
    if delta is None or delta != delta:
        return "nan"
    return f"{delta:+.2f}"


def _format_float(value: Optional[float]) -> str:
    if value is None or value != value:
        return "nan"
    return f"{value:.6f}"


def run_single_configuration(
    base_hw_dict: Dict[str, object],
    model_config,
    mode: str,
    parallel_settings: Dict[str, int],
    num_gpus: int,
    keep_artifacts: bool,
    topology: str,
    capture_output: bool = True,
) -> Dict[str, object]:
    hw_config = make_temp_hw_config(
        base_hw_dict,
        parallel_settings,
        hw_mutator=lambda cfg: update_first_dimension_size(cfg, num_gpus, topology),
    )
    if keep_artifacts:
        base_dir = os.path.join("tools", "gmap_artifacts")
        os.makedirs(base_dir, exist_ok=True)
        dir_name = "gmap_{topology}_tp{tp}_cp{cp}_pp{pp}".format(
            topology=topology.lower(),
            tp=parallel_settings["tp"],
            cp=parallel_settings["cp"],
            pp=parallel_settings["pp"],
        )
        temp_dir = os.path.join(base_dir, dir_name)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="gmap_sweep_")
    prev_flag = os.environ.get("RAPID_GMAP_ONLY")
    prev_persist = os.environ.get("RAPID_PERSIST_ASTRASIM_ARTIFACTS")    
    os.environ["RAPID_GMAP_ONLY"] = "1"
    if keep_artifacts:
        os.environ["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = "1"
    else:
        os.environ.pop("RAPID_PERSIST_ASTRASIM_ARTIFACTS", None)
    stdout_buffer = io.StringIO()
    try:
        calculator = TimeCalculationLLM(hw_config, model_config, mode, output_dir=temp_dir)
        if capture_output:
            with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
                try:
                    calculator.calc_time_llm()
                except SystemExit as exc:
                    if exc.code not in (0, None):
                        raise
        else:
            try:
                calculator.calc_time_llm()
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
    finally:
        if prev_flag is None:
            os.environ.pop("RAPID_GMAP_ONLY", None)
        else:
            os.environ["RAPID_GMAP_ONLY"] = prev_flag
        if prev_persist is None:
            os.environ.pop("RAPID_PERSIST_ASTRASIM_ARTIFACTS", None)
        else:
            os.environ["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = prev_persist
    stdout_text = stdout_buffer.getvalue() if capture_output else ""
    metrics_path = os.path.join(temp_dir, "first_dim_comm.metrics")
    before, after = parse_metrics_file(metrics_path)
    if before is None or after is None:
        alt_before, alt_after = parse_stdout_for_commexpan(stdout_text)
        if before is None:
            before = alt_before
        if after is None:
            after = alt_after
    delta_pct = compute_delta(before, after)
    applied = bool(
        before is not None and after is not None and before > 0 and after < before
    )
    artifact_dir = temp_dir if keep_artifacts else None
    if not keep_artifacts:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"Saved artifacts to {temp_dir}")
    return {
        "parallelism": dict(parallel_settings),
        "before": before,
        "after": after,
        "delta_pct": delta_pct,
        "applied": applied,
        "stdout": stdout_text,
        "artifact_dir": artifact_dir,
        "topology": topology,
    }


# -----------------------------------------------------------------------------#
# Multiprocessing glue                                                         #
# -----------------------------------------------------------------------------#

_GLOBAL_HW_DICT = None
_GLOBAL_MODEL_CONFIG = None
_GLOBAL_MODE = None
_GLOBAL_NUM_GPUS = None
_GLOBAL_KEEP_ARTIFACTS = False


def _worker_init(hw_dict, model_config_path, mode, num_gpus, keep_artifacts):
    global _GLOBAL_HW_DICT, _GLOBAL_MODEL_CONFIG, _GLOBAL_MODE, _GLOBAL_NUM_GPUS, _GLOBAL_KEEP_ARTIFACTS
    _GLOBAL_HW_DICT = hw_dict
    _GLOBAL_MODE = mode
    _GLOBAL_NUM_GPUS = num_gpus
    _GLOBAL_KEEP_ARTIFACTS = bool(keep_artifacts)
    _GLOBAL_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)


def _worker_task(parallel_items: Tuple[Tuple[str, int], ...]) -> Dict[str, object]:
    parallel_settings: Dict[str, int] = {}
    topology: Optional[str] = None
    for key, value in parallel_items:
        if key == "__topology":
            topology = str(value)
        else:
            parallel_settings[key] = int(value)
    if topology is None:
        topology = TOPOLOGY_VARIANTS[0]
    try:
        result = run_single_configuration(
            _GLOBAL_HW_DICT,
            _GLOBAL_MODEL_CONFIG,
            _GLOBAL_MODE,
            parallel_settings,
            _GLOBAL_NUM_GPUS,
            _GLOBAL_KEEP_ARTIFACTS,
            topology,
        )
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "parallelism": parallel_settings,
            "topology": topology,
        }


# -----------------------------------------------------------------------------#
# Plotting / reporting                                                         #
# -----------------------------------------------------------------------------#


def plot_results(results: Sequence[Dict[str, object]], num_gpus: int, topology: str, path: str) -> None:
    if not results:
        return
    xs = np.arange(len(results))
    ys = np.array(
        [
            entry["delta_pct"] if entry["delta_pct"] == entry["delta_pct"] else float("nan")
            for entry in results
        ],
        dtype=float,
    )
    labels = [
        "tp={tp}, cp={cp}, pp={pp}".format(
            tp=entry["parallelism"]["tp"],
            cp=entry["parallelism"]["cp"],
            pp=entry["parallelism"]["pp"],
        )
        for entry in results
    ]
    applied_flags = [bool(entry.get("applied")) for entry in results]
    applied_x = [x for x, flag in zip(xs, applied_flags) if flag]
    applied_y = [y for y, flag in zip(ys, applied_flags) if flag]
    skipped_x = [x for x, flag in zip(xs, applied_flags) if not flag]
    skipped_y = [y for y, flag in zip(ys, applied_flags) if not flag]

    plt.figure(figsize=(12, 6))
    if applied_x:
        plt.scatter(applied_x, applied_y, c="tab:green", s=80, label="Applied", edgecolors="black", linewidths=0.5)
    if skipped_x:
        plt.scatter(skipped_x, skipped_y, c="tab:red", s=80, label="Skipped", edgecolors="black", linewidths=0.5)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Δ CommExpan (%) (negative is better)")
    plt.xlabel("Parallelism configuration")
    plt.title(f"GMap sweep ({topology}, num_gpus={num_gpus})")
    plt.grid(alpha=0.3, axis="y")
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {path}")


def write_report(results: Sequence[Dict[str, object]], path: str, topology: str) -> None:
    header = [
        "tp",
        "cp",
        "pp",
        "before_commexpan",
        "after_commexpan",
        "delta_percent",
        "applied",
        "artifact_dir",
        "topology",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for entry in results:
            parallelism = entry["parallelism"]
            row = [
                str(parallelism["tp"]),
                str(parallelism["cp"]),
                str(parallelism["pp"]),
                _format_float(entry.get("before")),
                _format_float(entry.get("after")),
                _format_delta(entry.get("delta_pct")),
                "yes" if entry.get("applied") else "no",
                entry.get("artifact_dir") or "",
                topology,
            ]
            handle.write("\t".join(row) + "\n")
    print(f"Wrote report to {path}")


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep TP/CP/PP combinations for GMap cost deltas.")
    parser.add_argument("--hardware-config", default="configs/hardware-config/gmap_mesh2d_demo.yaml", help="Base hardware config path.")
    parser.add_argument("--model-config", default="configs/model-config/Llama3.1-405B.yaml", help="Model config path.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent worker processes.")
    parser.add_argument("--plot-output", default="tools/gmap_sweep.png", help="Path for the scatter plot.")
    parser.add_argument("--report-output", default="tools/gmap_sweep.tsv", help="Path for the TSV report.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Preserve per-run artefacts (first_dim_comm.*) on disk.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)
    if args.max_workers is None or args.max_workers <= 0:
        args.max_workers = max(1, (os.cpu_count() or 2) - 1)
    num_gpus = int(NUM_GPUS)
    if not is_power_of_two(num_gpus):
        raise ValueError("num_gpus must be a power of two.")
    if not is_perfect_square(num_gpus):
        raise ValueError("num_gpus must be a perfect square to satisfy the Mesh2D constraint.")

    base_hw_dict = read_yaml(args.hardware_config)
    mode = determine_model_mode(args.model_config)
    cases = enumerate_parallelisms(num_gpus)
    if not cases:
        print("No TP/CP/PP combinations satisfy the constraints.")
        return
    print(f"Evaluating {len(cases)} configuration(s) for num_gpus={num_gpus}")

    def _suffix_path(base: str, suffix: str) -> str:
        root, ext = os.path.splitext(base)
        return f"{root}_{suffix.lower()}{ext or ''}"

    task_specs: List[Dict[str, object]] = []
    for topology in TOPOLOGY_VARIANTS:
        for combo in cases:
            entry = dict(combo)
            entry["__topology"] = topology
            task_specs.append(entry)

    results_by_topology: Dict[str, List[Dict[str, object]]] = {topo: [] for topo in TOPOLOGY_VARIANTS}
    errors_by_topology: Dict[str, List[str]] = {topo: [] for topo in TOPOLOGY_VARIANTS}

    if len(task_specs) == 1:
        spec = task_specs[0]
        topology = spec["__topology"]
        combo = {k: v for k, v in spec.items() if k != "__topology"}
        model_config_obj = config.parse_config(args.model_config, config_type=mode)
        try:
            result = run_single_configuration(
                base_hw_dict,
                model_config_obj,
                mode,
                combo,
                num_gpus,
                args.keep_artifacts,
                topology,
                capture_output=False,
            )
            results_by_topology[topology].append(result)
        except Exception as exc:
            errors_by_topology[topology].append(f"{combo}: {exc}")
    else:
        args.max_workers = NUM_WORKERS
        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else 1
        worker_count = min(max_workers, max(1, (os.cpu_count() or 1) - 1), len(task_specs))
        task_items = [tuple(sorted(spec.items())) for spec in task_specs]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=(base_hw_dict, args.model_config, mode, num_gpus, args.keep_artifacts),
        ) as executor:
            futures = {executor.submit(_worker_task, items): items for items in task_items}
            with tqdm(total=len(task_items), desc="Evaluating", unit="config") as progress:
                for future in as_completed(futures):
                    progress.update(1)
                    outcome = future.result()
                    topo = outcome.get("result", {}).get("topology") if outcome.get("status") == "ok" else outcome.get("topology")
                    if topo is None:
                        topo = TOPOLOGY_VARIANTS[0]
                    if outcome["status"] == "ok":
                        results_by_topology[topo].append(outcome["result"])
                    else:
                        errors_by_topology[topo].append(f"{outcome['parallelism']}: {outcome.get('error')}")

    for topology in TOPOLOGY_VARIANTS:
        print(f"\n=== Topology: {topology} ===")
        results = results_by_topology[topology]
        errors = errors_by_topology[topology]
        if errors:
            print("Encountered errors:")
            for msg in errors:
                print("  -", msg)
        if not results:
            print("No successful evaluations for this topology; skipping.")
            continue
        results.sort(key=lambda entry: (entry["parallelism"]["tp"], entry["parallelism"]["cp"], entry["parallelism"]["pp"]))
        best = min(
            (entry for entry in results if entry.get("delta_pct") == entry.get("delta_pct")),
            key=lambda e: e.get("delta_pct"),
            default=None,
        )
        if best:
            best_delta = best.get("delta_pct")
            print(
                "Best improvement: tp={tp}, cp={cp}, pp={pp}, Δ={delta}".format(
                    tp=best["parallelism"]["tp"],
                    cp=best["parallelism"]["cp"],
                    pp=best["parallelism"]["pp"],
                    delta=_format_delta(best_delta),
                )
            )
        report_path = _suffix_path(args.report_output, topology)
        plot_path = _suffix_path(args.plot_output, topology)
        write_report(results, report_path, topology)
        plot_results(results, num_gpus, topology, plot_path)


if __name__ == "__main__":
    main()
