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
SPSA calibration for HF benchmark tok/s/gpu.
"""

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import huggingface_bench_validation as hf  # noqa: E402


# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================

BASE_HW_CONFIG_PATH = hf.CALIBRATED_HW_CONFIG_PATH
TUNED_HW_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB_calibrated_tuned.yaml"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "hf_spsa"

MODE = "perf_only"
NUM_WORKERS = 120

USE_VAL_SPLIT = False
VAL_FRACTION = 0.1

SPSA_ITERS = 40
BATCH_SIZE_EARLY = 40
BATCH_SIZE_LATE = 80

SPSA_DIRECTIONS = 2
SPSA_DECAY_C = False
SPSA_C_DECAY_POWER = 0.1

LOSS_EPS = 1e-3
LOSS_FAIL_VALUE = 4.0
LOSS_FAIL_LAMBDA = 1.0 

ALPHA_MULT = 3
ADAM_ALPHA = 0.2 * ALPHA_MULT
ADAM_ALPHA_LAT = 0.08 * ALPHA_MULT
ADAM_ALPHA_KL = 0.05 * ALPHA_MULT
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
GRAD_CLIP = 10.0

RNG_SEED = 8675311

MIN_STRATUM_ROWS = 10

FAST_MODE = True
PHASE0_TIMEOUT_S = 45.0
PHASE1_TIMEOUT_S = 25.0

UTIL_TEST = False # tests if configs work properly. don't need to run ever again
UTIL_TEST_BATCH = 32
UTIL_TEST_DIFF_PCT = 1.0
UTIL_TEST_UTIL_LOW = 0.50

RESTRICT_TO_PP1 = True
RESTRICT_TO_TP_RANGE = (2, 16)
RESTRICT_TO_DP_RANGE = (8, 64)

UTIL_MIN = 0.5
UTIL_MAX = 1.0
NET_UTIL_MIN = UTIL_MIN
NET_UTIL_MAX = 1.2
NET_LAT_MAX = 1e-5
KERNEL_LAUNCH_MIN = 1e-5
KERNEL_LAUNCH_MAX = 1e-3
GRAD_ACC_MIN = 1e-6
GRAD_ACC_MAX = 1e-2
OVERLAP_MIN = 0.0
OVERLAP_MAX = 1.0

INIT_UTIL_FLOPS = 0.93
INIT_UTIL_MEM = 0.9
INIT_UTIL_NET_TP = 0.9
INIT_UTIL_NET_LP = 0.9
INIT_UTIL_NET_DP = 0.6
INIT_NET_LAT = 3e-7
INIT_KERNEL_LAUNCH = 6e-5
INIT_GRAD_ACC = 1e-4
INIT_TP_SP_OVERLAP = 0.2

C_UTIL = 0.15
C_LAT = 1.5
C_OVER = 0.5

# Params listed here are fixed in real-space (disabled from SPSA).
FIXED_PARAMS = {
    "net_latency": 3.3e-7,
    "kernel_launch_overhead": 5e-5,
    "grad_acc_overhead": 1e-4,
}


# ==============================================================================
# PARAMETERIZATION
# ==============================================================================


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def _bounded_sigmoid(s: float, low: float, high: float) -> float:
    return low + (high - low) * _sigmoid(s)


def _util_from_s(s: float) -> float:
    return UTIL_MIN + (UTIL_MAX - UTIL_MIN) * _sigmoid(s)


def _util_s0(u0: float) -> float:
    return _logit((u0 - UTIL_MIN) / (UTIL_MAX - UTIL_MIN))


def _net_util_from_s(s: float) -> float:
    return NET_UTIL_MIN + (NET_UTIL_MAX - NET_UTIL_MIN) * _sigmoid(s)


def _net_util_s0(u0: float) -> float:
    return _logit((u0 - NET_UTIL_MIN) / (NET_UTIL_MAX - NET_UTIL_MIN))


def _overlap_from_s(s: float) -> float:
    return OVERLAP_MIN + (OVERLAP_MAX - OVERLAP_MIN) * _sigmoid(s)


def _overlap_s0(u0: float) -> float:
    return _logit((u0 - OVERLAP_MIN) / (OVERLAP_MAX - OVERLAP_MIN))


def _net_lat_from_s(s: float) -> float:
    return NET_LAT_MAX * _sigmoid(s)


def _net_lat_s0(lat0: float) -> float:
    return _logit(lat0 / NET_LAT_MAX)


def _kernel_launch_from_s(s: float) -> float:
    return _bounded_sigmoid(s, KERNEL_LAUNCH_MIN, KERNEL_LAUNCH_MAX)


def _kernel_launch_s0(val: float) -> float:
    return _logit((val - KERNEL_LAUNCH_MIN) / (KERNEL_LAUNCH_MAX - KERNEL_LAUNCH_MIN))


def _grad_acc_from_s(s: float) -> float:
    return _bounded_sigmoid(s, GRAD_ACC_MIN, GRAD_ACC_MAX)


def _grad_acc_s0(val: float) -> float:
    return _logit((val - GRAD_ACC_MIN) / (GRAD_ACC_MAX - GRAD_ACC_MIN))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    s0: float
    c0: float
    alpha: float
    to_real: Any


BASE_PARAM_SPECS: Sequence[ParamSpec] = (
    ParamSpec("u_flops", _util_s0(INIT_UTIL_FLOPS), C_UTIL, ADAM_ALPHA, _util_from_s),
    ParamSpec("u_mem", _util_s0(INIT_UTIL_MEM), C_UTIL, ADAM_ALPHA, _util_from_s),
    ParamSpec("u_net_tp", _net_util_s0(INIT_UTIL_NET_TP), C_UTIL, ADAM_ALPHA, _net_util_from_s),
    ParamSpec("u_net_lp", _net_util_s0(INIT_UTIL_NET_LP), C_UTIL, ADAM_ALPHA, _net_util_from_s),
    ParamSpec("u_net_dp", _net_util_s0(INIT_UTIL_NET_DP), C_UTIL, ADAM_ALPHA, _net_util_from_s),
    ParamSpec("tp_sp_overlap", _overlap_s0(INIT_TP_SP_OVERLAP), C_UTIL, ADAM_ALPHA, _overlap_from_s),
    ParamSpec("net_latency", _net_lat_s0(INIT_NET_LAT), C_LAT, ADAM_ALPHA_LAT, _net_lat_from_s),
    ParamSpec(
        "kernel_launch_overhead",
        _kernel_launch_s0(INIT_KERNEL_LAUNCH),
        C_OVER,
        ADAM_ALPHA_KL,
        _kernel_launch_from_s,
    ),
    ParamSpec(
        "grad_acc_overhead",
        _grad_acc_s0(INIT_GRAD_ACC),
        C_OVER,
        ADAM_ALPHA_KL,
        _grad_acc_from_s,
    ),
)

PARAM_SPECS: Sequence[ParamSpec] = tuple(
    spec for spec in BASE_PARAM_SPECS if spec.name not in FIXED_PARAMS
)


# ==============================================================================
# DATA LOADING AND SAMPLING
# ==============================================================================


def _is_comm_light(row: hf.BenchmarkRow) -> bool:
    return row.dp <= 2 and row.tp <= 2 and row.pp <= 2


def _bin_dp(row: hf.BenchmarkRow) -> str:
    return "dp=1" if row.dp == 1 else "dp>1"


def _bin_tp(row: hf.BenchmarkRow) -> str:
    return "tp=1" if row.tp == 1 else "tp>1"


def _bin_pp(row: hf.BenchmarkRow) -> str:
    return "pp=1" if row.pp == 1 else "pp>1"


def _bin_mbs(row: hf.BenchmarkRow) -> str:
    return "mbs=1" if row.mbs == 1 else "mbs>1"


def _bin_acc(row: hf.BenchmarkRow) -> str:
    return "batch_accum<=8" if row.batch_accum <= 8 else "batch_accum>=8"


def _build_strata(rows: Sequence[hf.BenchmarkRow]) -> Dict[Tuple[str, ...], List[hf.BenchmarkRow]]:
    strata: Dict[Tuple[str, ...], List[hf.BenchmarkRow]] = {}
    for row in rows:
        if _is_comm_light(row):
            continue
        key = (
            _bin_dp(row),
            _bin_tp(row),
            _bin_pp(row),
            _bin_mbs(row),
            _bin_acc(row),
        )
        strata.setdefault(key, []).append(row)
    return {k: v for k, v in strata.items() if len(v) >= MIN_STRATUM_ROWS}


def _sample_batch(
    strata: Dict[Tuple[str, ...], List[hf.BenchmarkRow]],
    batch_size: int,
    rng: random.Random,
) -> List[hf.BenchmarkRow]:
    if not strata:
        return []

    strata_items = list(strata.items())
    rng.shuffle(strata_items)
    selected: List[hf.BenchmarkRow] = []
    selected_ids = set()

    def _pick(rows: Sequence[hf.BenchmarkRow]) -> None:
        if not rows:
            return
        for _ in range(10):
            row = rows[rng.randrange(len(rows))]
            if row.row_index not in selected_ids:
                selected_ids.add(row.row_index)
                selected.append(row)
                return

    if len(strata_items) >= batch_size:
        for _, rows in strata_items[:batch_size]:
            _pick(rows)
        return selected

    for _, rows in strata_items:
        _pick(rows)

    remaining = batch_size - len(selected)
    if remaining <= 0:
        return selected

    pool: List[hf.BenchmarkRow] = []
    for _, rows in strata_items:
        for row in rows:
            if row.row_index not in selected_ids:
                pool.append(row)

    if len(pool) >= remaining:
        selected.extend(rng.sample(pool, remaining))
    else:
        all_rows = [row for _, rows in strata_items for row in rows]
        selected.extend(rng.choices(all_rows, k=remaining))

    return selected


# ==============================================================================
# HARDWARE CONFIG OVERRIDES
# ==============================================================================


def _apply_params_to_hw_config(
    base_hw: Dict[str, Any],
    params: Dict[str, float],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw)

    tech = cfg.get("tech_param")
    if tech is None:
        tech = cfg.setdefault("tech_config", {})
    core = tech.setdefault("core", {})
    core["util"] = float(params["u_flops"])

    if "DRAM" in tech:
        dram = tech["DRAM"]
    else:
        dram = tech.setdefault("DRAM", {})
    dram["util"] = float(params["u_mem"])

    sw = cfg.setdefault("sw_param", {})
    sw["kernel_launch_overhead"] = float(params["kernel_launch_overhead"])
    sw["grad_acc_overhead"] = float(params["grad_acc_overhead"])

    net = cfg.setdefault("network", {})
    overlap = net.setdefault("overlap", {})
    overlap["tp_sp_overlap"] = float(params["tp_sp_overlap"])
    dims = net.get("dimensions", [])
    for idx, dim in enumerate(dims):
        topo = dim.setdefault("topology", {})
        topo["latency"] = float(params["net_latency"])
        if idx == 0:
            topo["util"] = float(params["u_net_tp"])
        elif idx == 1:
            topo["util"] = float(params["u_net_lp"])
        elif idx == 2:
            topo["util"] = float(params["u_net_dp"])

    return cfg


# ==============================================================================
# EVALUATION
# ==============================================================================


def _params_from_s(s: np.ndarray) -> Dict[str, float]:
    params = {}
    for idx, spec in enumerate(PARAM_SPECS):
        params[spec.name] = float(spec.to_real(float(s[idx])))
    for key, value in FIXED_PARAMS.items():
        params[key] = float(value)
    return params


def _error_stats(comparisons: Sequence[hf.ComparisonResult]) -> Dict[str, float]:
    errors = []
    signed = []
    fail_count = 0
    for comp in comparisons:
        actual = comp.row.tok_s_gpu
        pred = comp.rapid_result.tok_s_gpu
        if (
            actual is None
            or actual <= 0
            or not comp.rapid_result.success
            or pred is None
            or pred <= 0
        ):
            fail_count += 1
            continue
        diff = math.log(float(pred)) - math.log(float(actual))
        signed.append(diff)
        errors.append(abs(diff))

    success_count = len(errors)
    total_count = success_count + fail_count
    fail_rate = float(fail_count) / float(total_count) if total_count > 0 else float("nan")

    if success_count == 0:
        return {
            "count": 0,
            "fail": fail_count,
            "fail_rate": fail_rate,
            "mean_abs": float("nan"),
            "mean_abs_pct": float("nan"),
            "median_abs": float("nan"),
            "median_abs_pct": float("nan"),
            "p90_abs": float("nan"),
            "p90_abs_pct": float("nan"),
            "p95_abs": float("nan"),
            "p95_abs_pct": float("nan"),
            "mean_signed": float("nan"),
            "mean_signed_pct": float("nan"),
        }

    arr = np.array(errors, dtype=float)
    signed_arr = np.array(signed, dtype=float)
    mean_abs = float(np.mean(arr))
    median_abs = float(np.median(arr))
    p90_abs = float(np.percentile(arr, 90))
    p95_abs = float(np.percentile(arr, 95))
    mean_signed = float(np.mean(signed_arr))
    mean_signed_pct = math.copysign(math.expm1(abs(mean_signed)), mean_signed)
    return {
        "count": int(success_count),
        "fail": int(fail_count),
        "fail_rate": fail_rate,
        "mean_abs": mean_abs,
        "mean_abs_pct": math.expm1(mean_abs),
        "median_abs": median_abs,
        "median_abs_pct": math.expm1(median_abs),
        "p90_abs": p90_abs,
        "p90_abs_pct": math.expm1(p90_abs),
        "p95_abs": p95_abs,
        "p95_abs_pct": math.expm1(p95_abs),
        "mean_signed": mean_signed,
        "mean_signed_pct": mean_signed_pct,
    }


def _evaluate_loss(
    rows: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
    params: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    cfg = _apply_params_to_hw_config(base_hw, params)
    comparisons = hf.evaluate_rows(
        rows,
        cfg,
        mode=MODE,
        num_workers=NUM_WORKERS,
        flash_attention=hf.USE_FLASH_ATTENTION,
        attention_tile_size=hf.ATTENTION_TILE_SIZE,
        emit_progress=False,
        fast_mode=FAST_MODE,
        timeout_s=PHASE1_TIMEOUT_S,
    )
    stats = _error_stats(comparisons)
    loss = _loss_from_stats(stats)
    return float(loss), stats


def _loss_from_stats(stats: Dict[str, float]) -> float:
    mean_abs = stats.get("mean_abs", float("nan"))
    fail_rate = stats.get("fail_rate", float("nan"))
    if mean_abs is None or not math.isfinite(mean_abs):
        mean_abs = float(LOSS_FAIL_VALUE)
    if fail_rate is None or not math.isfinite(fail_rate):
        fail_rate = 1.0
    return float(mean_abs) + float(LOSS_FAIL_LAMBDA) * float(fail_rate)


def _phase0_eval(
    rows: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
    params: Dict[str, float],
    timeout_s: float,
) -> Tuple[List[hf.ComparisonResult], set]:
    cfg = _apply_params_to_hw_config(base_hw, params)
    comparisons = hf.evaluate_rows(
        rows,
        cfg,
        mode=MODE,
        num_workers=NUM_WORKERS,
        flash_attention=hf.USE_FLASH_ATTENTION,
        attention_tile_size=hf.ATTENTION_TILE_SIZE,
        emit_progress=True,
        fast_mode=True,
        timeout_s=timeout_s,
    )
    hard_fail_ids = set()
    for comp in comparisons:
        pred = comp.rapid_result.tok_s_gpu
        if not comp.rapid_result.success or pred is None or pred <= 0:
            hard_fail_ids.add(comp.row.row_index)
    return comparisons, hard_fail_ids


def _select_util_test_batch(
    rows: Sequence[hf.BenchmarkRow],
    rng: random.Random,
) -> List[hf.BenchmarkRow]:
    if not rows:
        return []
    batch_size = min(int(UTIL_TEST_BATCH), len(rows))
    strata = _build_strata(rows)
    if strata:
        return _sample_batch(strata, batch_size, rng)
    if batch_size >= len(rows):
        return list(rows)
    return rng.sample(list(rows), batch_size)


def _run_util_sensitivity(
    rows: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
    base_params: Dict[str, float],
) -> None:
    print("UTIL_TEST: begin")
    loss_base, stats_base = _evaluate_loss(rows, base_hw, base_params)
    base_err_pct = stats_base.get("mean_abs_pct", float("nan")) * 100.0
    base_fail_pct = stats_base.get("fail_rate", float("nan")) * 100.0

    tests = [
        ("u_flops", UTIL_TEST_UTIL_LOW, 1.0),
        ("u_mem", UTIL_TEST_UTIL_LOW, 1.0),
        ("u_net_tp", UTIL_TEST_UTIL_LOW, 1.0),
        ("u_net_lp", UTIL_TEST_UTIL_LOW, 1.0),
        ("u_net_dp", UTIL_TEST_UTIL_LOW, 1.0),
        ("tp_sp_overlap", 0.0, 1.0),
        ("net_latency", NET_LAT_MAX, NET_LAT_MAX),
        ("kernel_launch_overhead", KERNEL_LAUNCH_MAX, KERNEL_LAUNCH_MAX),
        ("grad_acc_overhead", GRAD_ACC_MAX, GRAD_ACC_MAX),
    ]
    tests = [item for item in tests if item[0] not in FIXED_PARAMS]

    base_pct_map = {
        "u_flops": base_params.get("u_flops", float("nan")) * 100.0,
        "u_mem": base_params.get("u_mem", float("nan")) * 100.0,
        "u_net_tp": base_params.get("u_net_tp", float("nan")) * 100.0,
        "u_net_lp": base_params.get("u_net_lp", float("nan")) * 100.0,
        "u_net_dp": base_params.get("u_net_dp", float("nan")) * 100.0,
        "net_latency": _pct_of(base_params.get("net_latency", float("nan")), NET_LAT_MAX),
        "kernel_launch_overhead": _pct_of(
            base_params.get("kernel_launch_overhead", float("nan")),
            KERNEL_LAUNCH_MAX,
        ),
        "grad_acc_overhead": _pct_of(
            base_params.get("grad_acc_overhead", float("nan")),
            GRAD_ACC_MAX,
        ),
    }

    for name, alt_value, denom in tests:
        params_alt = dict(base_params)
        params_alt[name] = float(alt_value)

        loss_alt, stats_alt = _evaluate_loss(rows, base_hw, params_alt)
        alt_err_pct = stats_alt.get("mean_abs_pct", float("nan")) * 100.0
        alt_fail_pct = stats_alt.get("fail_rate", float("nan")) * 100.0

        base_loss = float(loss_base) if math.isfinite(loss_base) else 0.0
        denom_loss = abs(base_loss) if abs(base_loss) > 1e-12 else 1.0
        diff_pct = abs(float(loss_alt) - base_loss) / denom_loss * 100.0
        status = "OK" if diff_pct > UTIL_TEST_DIFF_PCT else "WEAK"

        base_pct = base_pct_map.get(name, float("nan"))
        alt_pct = _pct_of(alt_value, denom) if denom else float("nan")

        print(
            "UTIL_TEST {name}: base={base}% alt={alt}% "
            "err={err0}%->{err1}% fail={fail0}%->{fail1}% diff={diff}% {status}".format(
                name=name,
                base=_fmt_pct_value(base_pct),
                alt=_fmt_pct_value(alt_pct),
                err0=_fmt_pct_value(base_err_pct),
                err1=_fmt_pct_value(alt_err_pct),
                fail0=_fmt_pct_value(base_fail_pct),
                fail1=_fmt_pct_value(alt_fail_pct),
                diff=_fmt_pct_value(diff_pct),
                status=status,
            )
        )

    print("UTIL_TEST: done")


def _run_full_eval(
    label: str,
    rows_all: Sequence[hf.BenchmarkRow],
    train_ids: Optional[set],
    val_ids: Optional[set],
    base_hw: Dict[str, Any],
    params: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    cfg = _apply_params_to_hw_config(base_hw, params)
    comparisons = hf.evaluate_rows(
        rows_all,
        cfg,
        mode=MODE,
        num_workers=NUM_WORKERS,
        flash_attention=hf.USE_FLASH_ATTENTION,
        attention_tile_size=hf.ATTENTION_TILE_SIZE,
        emit_progress=True,
        fast_mode=FAST_MODE,
        timeout_s=PHASE1_TIMEOUT_S,
    )
    by_id = {c.row.row_index: c for c in comparisons}

    def _subset(ids: Optional[set]) -> List[hf.ComparisonResult]:
        if not ids:
            return []
        return [by_id[row_id] for row_id in ids if row_id in by_id]

    stats_all = _error_stats(comparisons)
    stats_train = _error_stats(_subset(train_ids)) if train_ids else stats_all
    stats_val = _error_stats(_subset(val_ids)) if val_ids else {}

    print(f"\n=== Full Eval ({label}) ===")
    _print_eval_table(stats_train, stats_val, stats_all)
    return {"train": stats_train, "val": stats_val, "all": stats_all}


def _print_eval_table(
    stats_train: Dict[str, float],
    stats_val: Dict[str, float],
    stats_all: Dict[str, float],
) -> None:
    headers = [
        "split",
        "n",
        "fail%",
        "mean_abs%",
        "median_abs%",
        "p90_abs%",
        "p95_abs%",
        "mean_signed%",
    ]
    print(" | ".join(headers))
    print("-" * 100)

    def _fmt_row(name: str, stats: Dict[str, float]) -> str:
        if not stats:
            return f"{name} | 0 | 0 | n/a | n/a | n/a | n/a | n/a"
        return " | ".join(
            [
                name,
                str(stats.get("count", 0)),
                f"{stats.get('fail_rate', float('nan')) * 100:.2f}",
                f"{stats.get('mean_abs_pct', float('nan')) * 100:.2f}",
                f"{stats.get('median_abs_pct', float('nan')) * 100:.2f}",
                f"{stats.get('p90_abs_pct', float('nan')) * 100:.2f}",
                f"{stats.get('p95_abs_pct', float('nan')) * 100:.2f}",
                f"{stats.get('mean_signed_pct', float('nan')) * 100:.2f}",
            ]
        )

    print(_fmt_row("train", stats_train))
    if stats_val:
        print(_fmt_row("val", stats_val))
    print(_fmt_row("all", stats_all))


def _trunc(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor


def _fmt_loss(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    pct = math.expm1(float(value)) * 100.0
    trimmed = _trunc(pct, 1)
    return f"{trimmed:.1f}"


def _fmt_util(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{float(value):.3f}"


def _fmt_exp(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    val = float(value)
    if val == 0.0:
        return "0.0e0"
    sign = "-" if val < 0 else ""
    val = abs(val)
    exp = int(math.floor(math.log10(val)))
    mantissa = val / (10 ** exp)
    mantissa = _trunc(mantissa, 1)
    return f"{sign}{mantissa:.1f}e{exp:+d}"


def _fmt_rate(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    pct = float(value) * 100.0
    trimmed = _trunc(pct, 1)
    return f"{trimmed:.1f}"


def _fmt_pct_value(value: float, decimals: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    trimmed = _trunc(float(value), decimals)
    return f"{trimmed:.{decimals}f}"


def _pct_of(value: float, max_value: float) -> float:
    if max_value <= 0:
        return float("nan")
    return (float(value) / float(max_value)) * 100.0


def _fmt_iter_line(
    idx: int,
    total: int,
    loss_plus: float,
    loss_minus: float,
    fail_rate: float,
    params: Dict[str, float],
) -> str:
    return (
        f"{idx}/{total}"
        f"|L+:{_fmt_loss(loss_plus)}"
        f"|L-:{_fmt_loss(loss_minus)}"
        f"|FR:{_fmt_rate(fail_rate)}"
        f"|Uf:{_fmt_util(params['u_flops'])}"
        f"|Um:{_fmt_util(params['u_mem'])}"
        f"|Un1:{_fmt_util(params['u_net_tp'])}"
        f"|Un2:{_fmt_util(params['u_net_lp'])}"
        f"|Un3:{_fmt_util(params['u_net_dp'])}"
        f"|Ov:{_fmt_util(params['tp_sp_overlap'])}"
        f"|Lat:{_fmt_exp(params['net_latency'])}"
        f"|KL:{_fmt_exp(params['kernel_launch_overhead'])}"
        f"|GA:{_fmt_exp(params['grad_acc_overhead'])}"
    )


def _format_stats_block(label: str, stats: Dict[str, float]) -> List[str]:
    if not stats:
        return [f"{label}: n/a"]
    lines = [
        f"{label}:",
        (
            "  count={count} fail={fail} fail_rate_pct={rate:.3f}".format(
                count=stats.get("count", 0),
                fail=stats.get("fail", 0),
                rate=stats.get("fail_rate", float("nan")) * 100,
            )
        ),
        f"  mean_abs_pct={stats.get('mean_abs_pct', float('nan')) * 100:.4f}",
        f"  median_abs_pct={stats.get('median_abs_pct', float('nan')) * 100:.4f}",
        f"  p90_abs_pct={stats.get('p90_abs_pct', float('nan')) * 100:.4f}",
        f"  p95_abs_pct={stats.get('p95_abs_pct', float('nan')) * 100:.4f}",
        f"  mean_signed_pct={stats.get('mean_signed_pct', float('nan')) * 100:.4f}",
    ]
    return lines


def _write_status_report(
    path: Path,
    reason: str,
    best_params: Optional[Dict[str, float]],
    best_eval_stats: Optional[Dict[str, Dict[str, float]]],
    current_params: Optional[Dict[str, float]],
    current_train_stats: Optional[Dict[str, float]],
    extra_lines: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "SPSA status report",
        f"reason: {reason}",
        "",
    ]
    if best_params:
        lines.append("best_params:")
        for key in sorted(best_params.keys()):
            lines.append(f"  {key}: {best_params[key]}")
        lines.append("")
    if best_eval_stats:
        lines.append("best_eval_stats:")
        lines.extend(_format_stats_block("train", best_eval_stats.get("train", {})))
        lines.extend(_format_stats_block("val", best_eval_stats.get("val", {})))
        lines.extend(_format_stats_block("all", best_eval_stats.get("all", {})))
        lines.append("")
    if current_params:
        lines.append("current_params:")
        for key in sorted(current_params.keys()):
            lines.append(f"  {key}: {current_params[key]}")
        lines.append("")
    if current_train_stats is not None:
        lines.append("current_train_stats:")
        lines.extend(_format_stats_block("train", current_train_stats))
        lines.append("")
    if extra_lines:
        lines.append("notes:")
        lines.extend(extra_lines)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==============================================================================
# SPSA OPTIMIZATION
# ==============================================================================


def _init_state() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s0 = np.array([spec.s0 for spec in PARAM_SPECS], dtype=float)
    c0 = np.array([spec.c0 for spec in PARAM_SPECS], dtype=float)
    alpha = np.array([spec.alpha for spec in PARAM_SPECS], dtype=float)
    m = np.zeros_like(s0)
    v = np.zeros_like(s0)
    return s0, c0, alpha, m, v


def _spsa_step(
    s: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    c0: np.ndarray,
    k: int,
    rng: random.Random,
    rows: Sequence[hf.BenchmarkRow],
    base_hw: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    if SPSA_DECAY_C:
        c_k = c0 / ((k + 1) ** SPSA_C_DECAY_POWER)
    else:
        c_k = c0

    grad = np.zeros_like(s)
    loss_plus = float("nan")
    loss_minus = float("nan")
    fail_rate_sum = 0.0
    fail_rate_count = 0

    for _ in range(max(1, int(SPSA_DIRECTIONS))):
        delta = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(s))], dtype=float)
        s_plus = s + c_k * delta
        s_minus = s - c_k * delta
        params_plus = _params_from_s(s_plus)
        params_minus = _params_from_s(s_minus)

        loss_plus, stats_plus = _evaluate_loss(rows, base_hw, params_plus)
        loss_minus, stats_minus = _evaluate_loss(rows, base_hw, params_minus)

        for stats in (stats_plus, stats_minus):
            rate = stats.get("fail_rate", float("nan"))
            if rate is not None and math.isfinite(rate):
                fail_rate_sum += float(rate)
                fail_rate_count += 1

        grad += (loss_plus - loss_minus) / (2.0 * c_k) * delta

    grad /= max(1, int(SPSA_DIRECTIONS))
    grad = np.clip(grad, -GRAD_CLIP, GRAD_CLIP)

    m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad
    v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * (grad ** 2)
    t = k + 1
    m_hat = m / (1.0 - ADAM_BETA1 ** t)
    v_hat = v / (1.0 - ADAM_BETA2 ** t)
    s = s - alpha * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    fail_rate = fail_rate_sum / fail_rate_count if fail_rate_count > 0 else float("nan")
    return s, m, v, float(loss_plus), float(loss_minus), float(fail_rate)


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    global NUM_WORKERS, SPSA_DIRECTIONS, SPSA_DECAY_C

    parser = argparse.ArgumentParser(description="SPSA calibration for HF tok/s/gpu.")
    parser.add_argument("--iters", type=int, default=SPSA_ITERS, help="Number of SPSA iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Worker count.")
    parser.add_argument(
        "--no-val-split",
        action="store_true",
        help="Disable train/val split (use all rows for train+eval).",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=SPSA_DIRECTIONS,
        help="Number of SPSA directions per iteration.",
    )
    parser.add_argument(
        "--decay-c",
        action="store_true",
        help="Enable c_k decay.",
    )
    args = parser.parse_args()

    NUM_WORKERS = int(args.num_workers)
    SPSA_DIRECTIONS = int(args.directions)
    SPSA_DECAY_C = bool(args.decay_c)

    use_val = USE_VAL_SPLIT and not args.no_val_split

    rng = random.Random(RNG_SEED)
    split_rng = random.Random(RNG_SEED + 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_DIR / "spsa_status.txt"

    base_hw = hf.load_base_hw_config(BASE_HW_CONFIG_PATH)

    hf.ENABLE_RESULT_CACHE = False

    best_params: Optional[Dict[str, float]] = None
    best_eval_stats: Optional[Dict[str, Dict[str, float]]] = None
    current_params: Optional[Dict[str, float]] = None
    iter_progress = None
    hard_fail_note: Optional[str] = None

    rows_all = hf.load_success_rows(
        hf.CSV_PATH,
        assume_bf16=hf.ASSUME_BF16,
        filter_pp_fix=hf.FILTER_PP_FIX,
        max_rows=None,
        mode=MODE,
    )
    rows_all = [row for row in rows_all if row.tok_s_gpu and row.tok_s_gpu > 0]
    if RESTRICT_TO_PP1:
        rows_all = [row for row in rows_all if row.pp == 1]
        print(f"[spsa] Restricting to category pp=1: {len(rows_all)} rows")
    if RESTRICT_TO_TP_RANGE or RESTRICT_TO_DP_RANGE:
        tp_lo, tp_hi = RESTRICT_TO_TP_RANGE
        dp_lo, dp_hi = RESTRICT_TO_DP_RANGE
        rows_all = [
            row
            for row in rows_all
            if tp_lo <= row.tp <= tp_hi and dp_lo <= row.dp <= dp_hi
        ]
        print(
            "[spsa] Restricting to tp={}-{}, dp={}-{}: {} rows".format(
                tp_lo, tp_hi, dp_lo, dp_hi, len(rows_all)
            )
        )

    if not rows_all:
        raise RuntimeError("No Success rows with tok/s/gpu available.")

    indices = list(range(len(rows_all)))
    split_rng.shuffle(indices)
    if use_val:
        val_count = int(round(len(indices) * VAL_FRACTION))
        val_count = min(max(1, val_count), max(1, len(indices) - 1))
        val_idx = set(indices[:val_count])
        train_idx = set(indices[val_count:])
    else:
        val_idx = set()
        train_idx = set(indices)

    rows_train = [rows_all[i] for i in train_idx]
    rows_val = [rows_all[i] for i in val_idx]
    s, c0, alpha, m, v = _init_state()
    params = _params_from_s(s)
    current_params = params

    if UTIL_TEST:
        util_rng = random.Random(RNG_SEED + 4242)
        test_pool = rows_train if rows_train else rows_all
        batch_rows = _select_util_test_batch(test_pool, util_rng)
        if not batch_rows:
            raise RuntimeError("UTIL_TEST batch is empty after filtering.")
        _run_util_sensitivity(batch_rows, base_hw, params)
        return

    phase0_comparisons, hard_fail_ids = _phase0_eval(
        rows_all,
        base_hw,
        params,
        PHASE0_TIMEOUT_S,
    )
    if hard_fail_ids:
        print(
            "Phase0 hard-fail filter (>={:.0f}s): {}/{}".format(
                PHASE0_TIMEOUT_S,
                len(hard_fail_ids),
                len(rows_all),
            )
        )
        hard_fail_note = "phase0_hard_fail_count={}".format(len(hard_fail_ids))

    phase0_by_id = {comp.row.row_index: comp for comp in phase0_comparisons}
    phase0_train_ids = {
        row.row_index for row in rows_train if row.row_index not in hard_fail_ids
    }
    phase0_val_ids = {row.row_index for row in rows_val}

    phase0_train_stats = (
        _error_stats([phase0_by_id[i] for i in phase0_train_ids if i in phase0_by_id])
        if phase0_train_ids
        else {}
    )
    phase0_val_stats = (
        _error_stats([phase0_by_id[i] for i in phase0_val_ids if i in phase0_by_id])
        if phase0_val_ids
        else {}
    )
    phase0_all_stats = _error_stats(phase0_comparisons)

    print("\n=== Full Eval (start, phase0 timeout) ===")
    _print_eval_table(phase0_train_stats, phase0_val_stats, phase0_all_stats)

    rows_train = [row for row in rows_train if row.row_index not in hard_fail_ids]
    if not rows_train:
        raise RuntimeError("Train split is empty after hard-fail filtering.")
    train_ids = {row.row_index for row in rows_train}
    val_ids = {row.row_index for row in rows_val}

    strata = _build_strata(rows_train)
    if not strata:
        rows_train = [row for row in rows_train if not _is_comm_light(row)]
        strata = {("all",): rows_train}

    try:
        start_stats = {
            "train": phase0_train_stats,
            "val": phase0_val_stats,
            "all": phase0_all_stats,
        }
        best_loss = _loss_from_stats(start_stats["val"] if use_val else start_stats["all"])
        if math.isnan(best_loss):
            best_loss = float("inf")
        best_params = params
        best_eval_stats = start_stats

        total_iters = int(args.iters)
        half_point = max(1, total_iters // 2)

        try:
            from tqdm import tqdm
            iter_progress = tqdm(total=total_iters, desc="SPSA", unit="iter")
        except Exception:
            iter_progress = None

        for k in range(total_iters):
            batch_size = BATCH_SIZE_EARLY if k < half_point else BATCH_SIZE_LATE
            batch_rows = _sample_batch(strata, batch_size, rng)
            if not batch_rows:
                raise RuntimeError("Batch sampling returned no rows.")

            s, m, v, loss_plus, loss_minus, fail_rate = _spsa_step(
                s,
                m,
                v,
                alpha,
                c0,
                k,
                rng,
                batch_rows,
                base_hw,
            )
            params = _params_from_s(s)
            current_params = params

            line = _fmt_iter_line(k + 1, total_iters, loss_plus, loss_minus, fail_rate, params)
            if iter_progress is not None:
                iter_progress.write(line)
                iter_progress.update(1)
            else:
                print(line)

            if k + 1 == half_point:
                stats = _run_full_eval("mid", rows_all, train_ids, val_ids, base_hw, params)
                val_loss = _loss_from_stats(stats["val"] if use_val else stats["all"])
                if not math.isnan(val_loss) and val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    best_eval_stats = stats

        stats = _run_full_eval("end", rows_all, train_ids, val_ids, base_hw, params)
        val_loss = _loss_from_stats(stats["val"] if use_val else stats["all"])
        if not math.isnan(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            best_eval_stats = stats
    except Exception as exc:
        extra_lines: List[str] = []
        if hard_fail_note:
            extra_lines.append(hard_fail_note)
        current_train_stats: Optional[Dict[str, float]] = None
        if current_params is not None and rows_train:
            try:
                cfg = _apply_params_to_hw_config(base_hw, current_params)
                comps = hf.evaluate_rows(
                    rows_train,
                    cfg,
                    mode=MODE,
                    num_workers=NUM_WORKERS,
                    flash_attention=hf.USE_FLASH_ATTENTION,
                    attention_tile_size=hf.ATTENTION_TILE_SIZE,
                    emit_progress=False,
                    fast_mode=FAST_MODE,
                    timeout_s=PHASE1_TIMEOUT_S,
                )
                current_train_stats = _error_stats(comps)
            except Exception as eval_exc:
                extra_lines.append(f"train_eval_error: {eval_exc}")

        _write_status_report(
            status_path,
            str(exc),
            best_params,
            best_eval_stats,
            current_params,
            current_train_stats,
            extra_lines=extra_lines if extra_lines else None,
        )
        raise
    finally:
        if iter_progress is not None:
            iter_progress.close()

    tuned_cfg = _apply_params_to_hw_config(base_hw, best_params)
    TUNED_HW_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNED_HW_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tuned_cfg, handle, sort_keys=False)

    print(f"\nSaved tuned config: {TUNED_HW_CONFIG_PATH}")


if __name__ == "__main__":
    main()
