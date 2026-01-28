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
Coordinate 1D sweep using the latest SPSA-tuned config as a starting point.
"""

import argparse
import copy
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import huggingface_bench_validation as hf  # noqa: E402
from validation_scripts import hf_meta_sweep as ms  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "hf_1d_sweep"
INPUT_CONFIG_PATH = ms.TUNED_HW_CONFIG_PATH
OUTPUT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "hardware-config"
    / "H100_SXM5_80GB_calibrated_tuned_1d.yaml"
)

DEFAULT_PASSES = 1
GRID_POINTS = 5
GOLDEN = True
GOLDEN_MAX_ITERS = 25
GOLDEN_STEP_FRAC = 0.1
GOLDEN_EXPAND = 2.0
GOLDEN_SHRINK = 0.5
SIGMOID_STEP_MULT = 2.5

DISABLED_PARAMS = {
    "u_flops",
    "u_mem",
    "u_net_tp",
    "u_net_lp",
    "u_net_dp",
    "tp_sp_overlap",
    "net_latency",
    "kernel_launch_overhead",
}

PARAM_BOUNDS = {
    "u_flops": (ms.UTIL_MIN, ms.UTIL_MAX),
    "u_mem": (ms.UTIL_MIN, ms.UTIL_MAX),
    "u_net_tp": (ms.NET_UTIL_MIN, ms.NET_UTIL_MAX),
    "u_net_lp": (ms.NET_UTIL_MIN, ms.NET_UTIL_MAX),
    "u_net_dp": (ms.NET_UTIL_MIN, ms.NET_UTIL_MAX),
    "tp_sp_overlap": (ms.OVERLAP_MIN, ms.OVERLAP_MAX),
    "net_latency": (0.0, ms.NET_LAT_MAX),
    "kernel_launch_overhead": (ms.KERNEL_LAUNCH_MIN, ms.KERNEL_LAUNCH_MAX),
    "grad_acc_overhead": (ms.GRAD_ACC_MIN, ms.GRAD_ACC_MAX),
}

INIT_DEFAULTS = {
    "u_flops": ms.INIT_UTIL_FLOPS,
    "u_mem": ms.INIT_UTIL_MEM,
    "u_net_tp": ms.INIT_UTIL_NET_TP,
    "u_net_lp": ms.INIT_UTIL_NET_LP,
    "u_net_dp": ms.INIT_UTIL_NET_DP,
    "tp_sp_overlap": 0.164,
    "net_latency": 3.3e-7,
    "kernel_launch_overhead": 5e-5,
    "grad_acc_overhead": ms.INIT_GRAD_ACC,
}

PARAM_TRANSFORMS = {
    "u_flops": (ms._util_s0, ms._util_from_s),
    "u_mem": (ms._util_s0, ms._util_from_s),
    "u_net_tp": (ms._net_util_s0, ms._net_util_from_s),
    "u_net_lp": (ms._net_util_s0, ms._net_util_from_s),
    "u_net_dp": (ms._net_util_s0, ms._net_util_from_s),
    "tp_sp_overlap": (ms._overlap_s0, ms._overlap_from_s),
    "net_latency": (ms._net_lat_s0, ms._net_lat_from_s),
    "kernel_launch_overhead": (ms._kernel_launch_s0, ms._kernel_launch_from_s),
    "grad_acc_overhead": (ms._grad_acc_s0, ms._grad_acc_from_s),
}

SIGMOID_PARAMS = {
    "net_latency",
    "kernel_launch_overhead",
    "grad_acc_overhead",
}

# FIXED_PARAMS = getattr(ms, "FIXED_PARAMS", {})
FIXED_PARAMS = {}


def _load_hw_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _to_search_space(name: str, value: float) -> float:
    transform = PARAM_TRANSFORMS.get(name)
    if transform is None:
        return float(value)
    return float(transform[0](float(value)))


def _from_search_space(name: str, value: float) -> float:
    transform = PARAM_TRANSFORMS.get(name)
    if transform is None:
        return float(value)
    return float(transform[1](float(value)))


def _search_bounds(name: str) -> Tuple[float, float]:
    low, high = PARAM_BOUNDS[name]
    if name not in PARAM_TRANSFORMS:
        return float(low), float(high)
    return _to_search_space(name, low), _to_search_space(name, high)


def _search_step(
    name: str,
    current: float,
    grid_points: int,
    low: float,
    high: float,
    current_s: float,
) -> float:
    search_range = abs(high - low)
    if search_range < 1e-12:
        return 0.0
    if name in SIGMOID_PARAMS:
        real_low, real_high = PARAM_BOUNDS[name]
        up = _clamp(current * SIGMOID_STEP_MULT, real_low, real_high)
        down = _clamp(current / SIGMOID_STEP_MULT, real_low, real_high)
        step_up = abs(_to_search_space(name, up) - current_s)
        step_down = abs(current_s - _to_search_space(name, down))
        step = max(step_up, step_down)
        if step < 1e-12:
            step = search_range * GOLDEN_STEP_FRAC
        return step
    step = search_range * GOLDEN_STEP_FRAC
    step = max(step, search_range / max(1.0, float(grid_points)))
    return step


def _extract_dim_value(dim: Dict[str, Any], key: str, default: float) -> float:
    topo = dim.get("topology", {}) if dim else {}
    val = topo.get(key, default)
    if isinstance(val, (list, tuple)):
        val = val[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _extract_params(hw_cfg: Dict[str, Any]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    tech = hw_cfg.get("tech_param") or hw_cfg.get("tech_config") or {}
    core = tech.get("core", {})
    params["u_flops"] = float(core.get("util", INIT_DEFAULTS["u_flops"]))

    dram = tech.get("DRAM") or tech.get("dram") or {}
    params["u_mem"] = float(dram.get("util", INIT_DEFAULTS["u_mem"]))

    net = hw_cfg.get("network", {})
    dims = net.get("dimensions", [])

    params["u_net_tp"] = _extract_dim_value(dims[0] if len(dims) > 0 else {}, "util", INIT_DEFAULTS["u_net_tp"])
    params["u_net_lp"] = _extract_dim_value(dims[1] if len(dims) > 1 else {}, "util", INIT_DEFAULTS["u_net_lp"])
    params["u_net_dp"] = _extract_dim_value(dims[2] if len(dims) > 2 else {}, "util", INIT_DEFAULTS["u_net_dp"])

    overlap = net.get("overlap", {})
    params["tp_sp_overlap"] = float(overlap.get("tp_sp_overlap", INIT_DEFAULTS["tp_sp_overlap"]))

    params["net_latency"] = _extract_dim_value(dims[0] if len(dims) > 0 else {}, "latency", INIT_DEFAULTS["net_latency"])

    sw = hw_cfg.get("sw_param", {})
    params["kernel_launch_overhead"] = float(sw.get("kernel_launch_overhead", INIT_DEFAULTS["kernel_launch_overhead"]))
    params["grad_acc_overhead"] = float(sw.get("grad_acc_overhead", INIT_DEFAULTS["grad_acc_overhead"]))

    for key, value in FIXED_PARAMS.items():
        params[key] = float(value)

    for key, default in INIT_DEFAULTS.items():
        if key not in params or params[key] is None:
            params[key] = float(default)

    for key, bounds in PARAM_BOUNDS.items():
        if key in params:
            params[key] = _clamp(params[key], bounds[0], bounds[1])

    return params


def _candidate_values(name: str, current: float, points: int) -> List[float]:
    low, high = _search_bounds(name)
    if points <= 1 or abs(high - low) < 1e-12:
        return [_from_search_space(name, _to_search_space(name, current))]
    current_s = _to_search_space(name, current)
    step = (high - low) / float(points - 1)
    s_values = [low + idx * step for idx in range(points)]
    s_values.append(float(current_s))
    values = [_from_search_space(name, val) for val in s_values]
    dedup = sorted({float(v) for v in values})
    return dedup


def _format_pct(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.2f}"


def _format_val(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    value = float(value)
    if value == 0.0:
        return "0.0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _stats_summary(stats: Dict[str, float]) -> str:
    return (
        "fail={fail}%|mean_abs={mean}%|median_abs={median}%|p90_abs={p90}%|p95_abs={p95}%|mean_signed={signed}%"
    ).format(
        fail=_format_pct(stats.get("fail_rate", float("nan")) * 100.0),
        mean=_format_pct(stats.get("mean_abs_pct", float("nan")) * 100.0),
        median=_format_pct(stats.get("median_abs_pct", float("nan")) * 100.0),
        p90=_format_pct(stats.get("p90_abs_pct", float("nan")) * 100.0),
        p95=_format_pct(stats.get("p95_abs_pct", float("nan")) * 100.0),
        signed=_format_pct(stats.get("mean_signed_pct", float("nan")) * 100.0),
    )


def _log(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def _active_params() -> List[str]:
    # Use BASE_PARAM_SPECS so 1D sweeps can include params fixed in SPSA.
    names = [spec.name for spec in ms.BASE_PARAM_SPECS]
    return [name for name in names if name not in DISABLED_PARAMS]


def _load_rows() -> Tuple[List[hf.BenchmarkRow], List[hf.BenchmarkRow], List[hf.BenchmarkRow]]:
    rows_all = hf.load_success_rows(
        hf.CSV_PATH,
        assume_bf16=hf.ASSUME_BF16,
        filter_pp_fix=hf.FILTER_PP_FIX,
        max_rows=None,
        mode=ms.MODE,
    )
    rows_all = [row for row in rows_all if row.tok_s_gpu and row.tok_s_gpu > 0]
    if ms.RESTRICT_TO_PP1:
        rows_all = [row for row in rows_all if row.pp == 1]
        print(f"[1d] Restricting to category pp=1: {len(rows_all)} rows")

    if not rows_all:
        raise RuntimeError("No Success rows with tok/s/gpu available.")

    rng = random.Random(ms.RNG_SEED + 7)
    indices = list(range(len(rows_all)))
    rng.shuffle(indices)

    if ms.USE_VAL_SPLIT:
        val_count = int(round(len(indices) * ms.VAL_FRACTION))
        val_count = min(max(1, val_count), max(1, len(indices) - 1))
        val_idx = set(indices[:val_count])
        train_idx = set(indices[val_count:])
    else:
        val_idx = set()
        train_idx = set(indices)

    rows_train = [rows_all[i] for i in train_idx]
    rows_val = [rows_all[i] for i in val_idx]
    return rows_all, rows_train, rows_val


def _eval_param(
    name: str,
    value: float,
    params: Dict[str, float],
    rows_train: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
    cache: Dict[float, Tuple[float, Dict[str, float], float, float, float]],
) -> Tuple[float, Dict[str, float], float, float, float]:
    if value in cache:
        return cache[value]

    trial = dict(params)
    trial[name] = float(value)
    for key, fixed in FIXED_PARAMS.items():
        trial[key] = float(fixed)

    loss, stats = ms._evaluate_loss(rows_train, base_hw, trial)
    mean_abs_pct = stats.get("mean_abs_pct", float("nan")) * 100.0
    fail_pct = stats.get("fail_rate", float("nan")) * 100.0
    loss_pct = mean_abs_pct + (ms.LOSS_FAIL_LAMBDA * fail_pct)
    cache[value] = (loss, stats, mean_abs_pct, fail_pct, loss_pct)
    return cache[value]


def _golden_search(
    name: str,
    current: float,
    params: Dict[str, float],
    rows_train: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
    grid_points: int,
) -> Tuple[float, float, float, Dict[str, float]]:
    low, high = _search_bounds(name)
    if abs(high - low) < 1e-12:
        return current, float("inf"), float("nan"), {}

    cache: Dict[float, Tuple[float, Dict[str, float], float, float, float]] = {}
    current_s = _clamp(_to_search_space(name, current), low, high)
    current_val = _from_search_space(name, current_s)
    step = _search_step(name, current_val, grid_points, low, high, current_s)

    loss0, stats0, _, _, _ = _eval_param(
        name, current_val, params, rows_train, base_hw, cache
    )

    left_s = _clamp(current_s - step, low, high)
    right_s = _clamp(current_s + step, low, high)
    left_val = _from_search_space(name, left_s)
    right_val = _from_search_space(name, right_s)

    loss_left, stats_left, _, _, _ = _eval_param(
        name, left_val, params, rows_train, base_hw, cache
    )
    loss_right, stats_right, _, _, _ = _eval_param(
        name, right_val, params, rows_train, base_hw, cache
    )

    _log(f"bracket|{name}|v={_format_val(current_val)}|{_stats_summary(stats0)}")
    _log(f"bracket|{name}|v={_format_val(left_val)}|{_stats_summary(stats_left)}")
    _log(f"bracket|{name}|v={_format_val(right_val)}|{_stats_summary(stats_right)}")

    best_value = current_val
    best_loss = loss0
    best_stats = stats0
    best_loss_pct = stats0.get("mean_abs_pct", float("nan")) * 100.0

    if loss_left >= loss0 and loss_right >= loss0:
        a, b = (left_s, right_s) if left_s < right_s else (right_s, left_s)
    else:
        direction = -1.0 if loss_left < loss_right else 1.0
        x_prev, f_prev = current_s, loss0
        x_best = left_s if direction < 0 else right_s
        f_best = loss_left if direction < 0 else loss_right
        step_exp = step
        a, b = None, None
        for _ in range(GOLDEN_MAX_ITERS):
            step_exp *= GOLDEN_EXPAND
            x_next = _clamp(x_best + direction * step_exp, low, high)
            if x_next == x_best:
                break
            f_next, _, _, _, _ = _eval_param(
                name,
                _from_search_space(name, x_next),
                params,
                rows_train,
                base_hw,
                cache,
            )
            if f_next >= f_best:
                a, b = (x_prev, x_next) if x_prev < x_next else (x_next, x_prev)
                best_value = _from_search_space(name, x_best)
                best_loss = f_best
                best_stats = cache[best_value][1]
                best_loss_pct = cache[best_value][4]
                break
            x_prev, f_prev = x_best, f_best
            x_best, f_best = x_next, f_next
        if a is None or b is None:
            a, b = (x_prev, x_best) if x_prev < x_best else (x_best, x_prev)
            best_value = _from_search_space(name, x_best)
            best_loss = f_best
            best_stats = cache[best_value][1]
            best_loss_pct = cache[best_value][4]

    if a == b:
        return best_value, best_loss, best_loss_pct, best_stats

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)

    cand_progress = tqdm(total=GOLDEN_MAX_ITERS, desc=f"{name} golden", leave=False) if tqdm else None
    for _ in range(GOLDEN_MAX_ITERS):
        if abs(b - a) < 1e-12:
            break
        c_val = _from_search_space(name, c)
        d_val = _from_search_space(name, d)
        loss_c, stats_c, _, _, loss_c_pct = _eval_param(
            name, c_val, params, rows_train, base_hw, cache
        )
        loss_d, stats_d, _, _, loss_d_pct = _eval_param(
            name, d_val, params, rows_train, base_hw, cache
        )
        if loss_c < best_loss:
            best_loss = loss_c
            best_value = c_val
            best_stats = stats_c
            best_loss_pct = loss_c_pct
        if loss_d < best_loss:
            best_loss = loss_d
            best_value = d_val
            best_stats = stats_d
            best_loss_pct = loss_d_pct

        if loss_c < loss_d:
            b = d
            d = c
            c = b - phi * (b - a)
        else:
            a = c
            c = d
            d = a + phi * (b - a)
        if cand_progress is not None:
            cand_progress.update(1)

    if cand_progress is not None:
        cand_progress.close()

    return best_value, best_loss, best_loss_pct, best_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="1D coordinate sweep after SPSA.")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Number of passes over all params.")
    parser.add_argument("--grid", type=int, default=GRID_POINTS, help="Grid points per parameter.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_CONFIG_PATH),
        help="Starting hardware config (tuned SPSA output).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CONFIG_PATH),
        help="Output hardware config path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        base_hw = _load_hw_config(input_path)
        print(f"[1d] Loaded base config: {input_path}")
    else:
        base_hw = hf.load_base_hw_config(ms.BASE_HW_CONFIG_PATH)
        print(f"[1d] Base config missing, using: {ms.BASE_HW_CONFIG_PATH}")

    params = _extract_params(base_hw)

    rows_all, rows_train, rows_val = _load_rows()
    if not rows_train:
        raise RuntimeError("Train split is empty after filtering.")

    phase0_comparisons, hard_fail_ids = ms._phase0_eval(
        rows_all,
        base_hw,
        params,
        ms.PHASE0_TIMEOUT_S,
    )

    if hard_fail_ids:
        print(
            "Phase0 hard-fail filter (>={:.0f}s): {}/{}".format(
                ms.PHASE0_TIMEOUT_S,
                len(hard_fail_ids),
                len(rows_all),
            )
        )

    phase0_by_id = {comp.row.row_index: comp for comp in phase0_comparisons}
    phase0_train_ids = {
        row.row_index for row in rows_train if row.row_index not in hard_fail_ids
    }
    phase0_val_ids = {row.row_index for row in rows_val}

    phase0_train_stats = (
        ms._error_stats([phase0_by_id[i] for i in phase0_train_ids if i in phase0_by_id])
        if phase0_train_ids
        else {}
    )
    phase0_val_stats = (
        ms._error_stats([phase0_by_id[i] for i in phase0_val_ids if i in phase0_by_id])
        if phase0_val_ids
        else {}
    )
    phase0_all_stats = ms._error_stats(phase0_comparisons)

    print("\n=== Full Eval (start, phase0 timeout) ===")
    ms._print_eval_table(phase0_train_stats, phase0_val_stats, phase0_all_stats)

    rows_train = [row for row in rows_train if row.row_index not in hard_fail_ids]
    if not rows_train:
        raise RuntimeError("Train split is empty after hard-fail filtering.")

    active_params = _active_params()
    if not active_params:
        raise RuntimeError("No active parameters to sweep (check FIXED_PARAMS/DISABLED_PARAMS).")

    best_params = copy.deepcopy(params)
    best_eval_stats: Optional[Dict[str, Dict[str, float]]] = None
    train_ids = {row.row_index for row in rows_train}
    val_ids = {row.row_index for row in rows_val}

    pass_iter = range(int(args.passes))
    if tqdm is not None:
        pass_iter = tqdm(pass_iter, desc="passes")

    for pass_idx in pass_iter:
        _log(f"\n=== Pass {pass_idx + 1}/{args.passes} ===")
        param_iter = active_params
        if tqdm is not None:
            param_iter = tqdm(param_iter, desc=f"pass {pass_idx + 1} params", leave=False)

        for name in param_iter:
            current = params[name]
            best_loss = float("inf")
            best_value = current
            best_loss_pct = float("nan")
            best_stats: Optional[Dict[str, float]] = None

            if GOLDEN:
                best_value, best_loss, best_loss_pct, best_stats = _golden_search(
                    name,
                    current,
                    params,
                    rows_train,
                    base_hw,
                    int(args.grid),
                )
            else:
                candidates = _candidate_values(name, current, int(args.grid))
                cand_iter = candidates
                if tqdm is not None:
                    cand_iter = tqdm(candidates, desc=f"{name} grid", leave=False)
                cache: Dict[float, Tuple[float, Dict[str, float], float, float, float]] = {}

                for cand in cand_iter:
                    loss, stats, _, _, _ = _eval_param(
                        name,
                        cand,
                        params,
                        rows_train,
                        base_hw,
                        cache,
                    )
                    _log(
                        "p{p}/{pt}|{name}|v={val}|{stats}".format(
                            p=pass_idx + 1,
                            pt=args.passes,
                            name=name,
                            val=_format_val(cand),
                            stats=_stats_summary(stats),
                        )
                    )

                    if loss < best_loss:
                        best_loss = loss
                        best_value = float(cand)
                        best_loss_pct = cache[float(cand)][4]
                        best_stats = stats

            params[name] = float(best_value)
            stats_blob = _stats_summary(best_stats) if best_stats else "stats=nan"
            _log(
                "choose|{name}|v={val}|{stats}".format(
                    name=name,
                    val=_format_val(best_value),
                    stats=stats_blob,
                )
            )

        stats = ms._run_full_eval("pass_end", rows_all, train_ids, val_ids, base_hw, params)
        val_stats = stats.get("val", {}) if ms.USE_VAL_SPLIT else stats.get("all", {})
        val_loss = ms._loss_from_stats(val_stats)
        if best_eval_stats is None:
            best_eval_stats = stats
            best_params = copy.deepcopy(params)
        elif not math.isnan(val_loss):
            prev_stats = best_eval_stats.get("val", val_stats) if ms.USE_VAL_SPLIT else best_eval_stats.get("all", val_stats)
            if val_loss < ms._loss_from_stats(prev_stats):
                best_eval_stats = stats
                best_params = copy.deepcopy(params)

    tuned_cfg = ms._apply_params_to_hw_config(base_hw, best_params)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tuned_cfg, handle, sort_keys=False)

    print(f"\nSaved 1D tuned config: {output_path}")


if __name__ == "__main__":
    main()
