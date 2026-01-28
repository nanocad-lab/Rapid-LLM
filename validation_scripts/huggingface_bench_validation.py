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
HuggingFace Benchmark Validation Script

Validates RAPID LLM memory and performance predictions against
bench_final2_mfu2.csv benchmark data.

Usage:
    python huggingface_bench_validation.py

Configure via GLOBAL CONFIG section below.
"""

import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import multiprocessing
import queue
from contextlib import nullcontext
from multiprocessing import Queue
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================

def _ensure_project_root_on_path():
    """Guarantee project root is importable (works for fork/spawn workers)."""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ensure_project_root_on_path()
CSV_PATH = PROJECT_ROOT / "validation_scripts" / "huggingface_data" / "bench_final2_mod_add.csv"
BASE_HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB.yaml"
CALIBRATED_HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB_calibrated_tuned.yaml"
USE_CALIBRATED_HW = True
HW_CONFIG_PATH = CALIBRATED_HW_CONFIG_PATH if USE_CALIBRATED_HW else BASE_HW_CONFIG_PATH
OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "huggingface"

# Execution mode: "mem_only", "perf_only", "both"
MODE = "both"
# Memory error target: "res" (reserved) or "alloc" (allocated)
MEMORY_ERROR_TARGET = "res"
# OOM classification threshold for predicted peak memory (GB).
OOM_PRED_THRESHOLD_GB = 80.0

# Parallel execution
NUM_WORKERS = 100

# Filtering options
ASSUME_BF16 = True          # Treat empty dtype as bfloat16
FILTER_PP_FIX = False       # Only use rows where after_pp_fix=True
MAX_ROWS = None             # Limit rows for testing (None = all)

# Selective runs: only execute the listed config IDs (ignore cache).
# IDs can be row_index values (as strings) or config labels
# (see _format_config_label / BenchmarkRow.label).
SELECTIVE_RUN = False
# add these: 1171, 1371, 1377, 32, 39, 47, 50, 198
SELECTIVE_RUN_IDS: Sequence[str] = ()
SELECTIVE_RUN_OUTPUT_DIR = OUTPUT_DIR / "selective_runs"
SELECTIVE_DUMP_HW_CONFIGS = False

# Model options
USE_FLASH_ATTENTION = True
ATTENTION_TILE_SIZE = 128  # Set to int (e.g., 128) if using flash attention

# Plot and output
ENABLE_PLOTS = True
ENABLE_GLOBAL_PLOTS = True
ENABLE_CATEGORY_PLOTS = True
ERROR_PCT_CAP = 200.0
EMIT_LOGS = True
SUPPRESS_WORKER_OUTPUT = True
SHUFFLE_SEED: Optional[int] = 1337

# Debug: dump the first filtered row's hardware config and exit early.
DUMP_FIRST_ROW_HW_CONFIG = False
DUMP_FIRST_ROW_HW_CONFIG_NAME = "first_row_hw_config.yaml"

# Chunked execution / caching to limit peak memory
CHUNK_SIZE = 600               # Process rows in chunks (None or <=0 disables)
ENABLE_RESULT_CACHE = True
CACHE_PATH = OUTPUT_DIR / "validation_cache.jsonl"
CACHE_KEY_VERSION = 15
HARD_CLEANUP_BETWEEN_CHUNKS = True
# Cache-only rebuild mode (skip RAPID execution)
REBUILD_FROM_CACHE_ONLY = True
# Fast mode: enforce per-row worker timeouts (best-effort)
FAST_MODE = True
FAST_MODE_TIMEOUT_S = 45.0

# AstraSim isolation (avoid cache contention / temp spam in repo root)
ASTRA_CACHE_MODE: Optional[str] = "NO_CACHE"  # NO_CACHE | CACHE_READONLY | CACHE_READWRITE | None (keep env)
ASTRA_TMP_ROOT = PROJECT_ROOT / "tmp" / "huggingface_validation_runs"
CLEANUP_ASTRA_TMP = True

# Network selection: GPUs per node threshold for intra vs inter bandwidth.
GCN = 8

# tok/s/gpu conversion: denominator for wall time calculation
# Options: "num_gpus", "dp_tp", "dp", "dp_tp_pp" (alias of num_gpus)
TOKS_PER_GPU_DENOM = "num_gpus"

# Microbatch + pipeline memory heuristics
USE_DERIVED_MICROBATCH_COUNT = True
MB_GRAPH_CAP: Optional[int] = None  # cap microbatch count in graph to keep runs fast (None disables)
PIPELINE_MEM_SCHEDULE = "1f1b"   # "gpipe" to disable, "1f1b"/"auto" to scale activations
PIPELINE_ACTIVATION_WINDOW_MULT = 2.0  # effective in-flight window = ceil(pp * multiplier)

# Category plotting
CATEGORY_RUN: Optional[Sequence[str] | str] = ("all")  # "auto", "default", "all", or list of category names
CATEGORY_MIN_ROWS = 10
CATEGORY_OUTPUT_DIR = OUTPUT_DIR / "categories"
CATEGORY_AUTO_TOP_K = 5
CATEGORY_AUTO_MIN_ROWS = 5
CATEGORY_AUTO_METRICS: Sequence[str] = ("tok_s", "mem")
CATEGORY_PRESETS: Dict[str, Sequence[str]] = {
    "default": (
        "pp=1",
        "pp>1",
        "pp>=8",
        "tp=1",
        "tp>1",
        "tp>=8",
        "dp=1",
        "dp>1",
        "dp>=8",
        "zero=0",
        "zero=1",
        "batch_accum<=8",
        "batch_accum>=64",
        "mbs=1",
        "mbs>=8",
        "after_pp_fix=true",
        "after_pp_fix=false",
        "attention=gqa",
        "attention=mha",
        "model=small",
        "model=large",
        "status=Success",
        "status=OOM",   
    ),
    "pp1":{
        "pp=1",
    },
    "parallelism": (
        "pp=1",
        "pp>1",
        "pp>=8",
        "tp=1",
        "tp>1",
        "tp>=8",
        "dp=1",
        "dp>1",
        "dp>=8",
    ),
    "training": (
        "batch_accum<=8",
        "batch_accum>=64",
        "mbs=1",
        "mbs>=8",
        "zero=0",
        "zero=1",
    ),
    "model": (
        "attention=gqa",
        "attention=mha",
        "model=small",
        "model=mid",
        "model=large",
    ),
    "status": (
        "status=Success",
        "status=OOM",
    )
}


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class BenchmarkRow:
    """Parsed row from benchmark CSV."""
    # Parallelism
    dp: int
    pp: int
    tp: int
    nodes: int

    # Batch config
    mbs: int              # micro_batch_size
    batch_accum: int      # gradient accumulation steps
    gbs: int              # global batch size

    # Model config
    seq_len: int
    hidden_size: int
    interm_dim: Optional[int]
    num_layers: int
    num_heads: int
    num_kv_heads: int     # For GQA
    vocab_size: int

    # Training config
    zero_stage: int
    dtype: str            # 'torch.bfloat16' or empty

    # Network bandwidths (GB/s) from CSV
    ar_inter: float       # AllReduce (GB/s)
    ag_inter: float       # AllGather (GB/s)
    rs_inter: float       # ReduceScatter (GB/s)
    ar_intra: float       # AR Intra-node (GB/s)
    ag_intra: float       # AG Intra-node (GB/s)
    rs_intra: float       # RS Intra-node (GB/s)

    # Ground truth
    mem_alloc_gb: float
    mem_res_gb: float
    mtflops: float
    tok_s_gpu: float
    mfu: float

    # Status
    status: str           # 'Success', 'OOM', 'Other'
    oom_error: bool
    after_pp_fix: bool

    # Derived
    row_index: int
    num_gpus: int         # dp * pp * tp

    @property
    def label(self) -> str:
        return f"dp{self.dp}_tp{self.tp}_pp{self.pp}_gbs{self.gbs}_mbs{self.mbs}_h{self.hidden_size}_L{self.num_layers}"


@dataclass
class RAPIDResult:
    """Results from RAPID LLM estimation."""
    success: bool
    error: Optional[str] = None

    # Memory
    peak_gb: Optional[float] = None
    capacity_exceeded: bool = False

    # Performance
    training_time_s: Optional[float] = None
    tok_s_gpu: Optional[float] = None
    mfu: Optional[float] = None


@dataclass
class ComparisonResult:
    """Comparison between RAPID prediction and ground truth."""
    row: BenchmarkRow
    rapid_result: RAPIDResult

    # Memory errors (%)
    mem_alloc_error_pct: Optional[float] = None
    mem_res_error_pct: Optional[float] = None

    # Performance errors (%)
    time_error_pct: Optional[float] = None
    mfu_error_pct: Optional[float] = None
    tok_s_error_pct: Optional[float] = None


@dataclass(frozen=True)
class CategoryDef:
    """Named filter for grouping comparisons."""
    name: str
    predicate: Callable[["ComparisonResult"], bool]


def _attention_type(row: BenchmarkRow) -> str:
    return "gqa" if row.num_kv_heads < row.num_heads else "mha"


def _eq(attr: str, value: Any) -> Callable[[ComparisonResult], bool]:
    return lambda c: getattr(c.row, attr) == value


def _gt(attr: str, value: Any) -> Callable[[ComparisonResult], bool]:
    return lambda c: getattr(c.row, attr) > value


def _ge(attr: str, value: Any) -> Callable[[ComparisonResult], bool]:
    return lambda c: getattr(c.row, attr) >= value


def _le(attr: str, value: Any) -> Callable[[ComparisonResult], bool]:
    return lambda c: getattr(c.row, attr) <= value


def _and(*preds: Callable[[ComparisonResult], bool]) -> Callable[[ComparisonResult], bool]:
    return lambda c: all(pred(c) for pred in preds)


def _category_definitions() -> List[CategoryDef]:
    """Define category filters for plot breakdowns."""
    defs = [
        CategoryDef("pp=1", _eq("pp", 1)),
        CategoryDef("pp>1", _gt("pp", 1)),
        CategoryDef("pp>=8", _ge("pp", 8)),
        CategoryDef("tp=1", _eq("tp", 1)),
        CategoryDef("tp>1", _gt("tp", 1)),
        CategoryDef("tp>=8", _ge("tp", 8)),
        CategoryDef("dp=1", _eq("dp", 1)),
        CategoryDef("dp>1", _gt("dp", 1)),
        CategoryDef("dp>=8", _ge("dp", 8)),
        CategoryDef("zero=0", _eq("zero_stage", 0)),
        CategoryDef("zero=1", _eq("zero_stage", 1)),
        CategoryDef("batch_accum<=8", _le("batch_accum", 8)),
        CategoryDef("batch_accum>=64", _ge("batch_accum", 64)),
        CategoryDef("mbs=1", _eq("mbs", 1)),
        CategoryDef("mbs>=8", _ge("mbs", 8)),
        CategoryDef("pp=1,mbs>=8", lambda c: c.row.pp == 1 and c.row.mbs >= 8),
        CategoryDef("after_pp_fix=true", lambda c: bool(c.row.after_pp_fix)),
        CategoryDef("after_pp_fix=false", lambda c: not bool(c.row.after_pp_fix)),
        CategoryDef("attention=gqa", lambda c: _attention_type(c.row) == "gqa"),
        CategoryDef("attention=mha", lambda c: _attention_type(c.row) == "mha"),
        CategoryDef("model=small", lambda c: c.row.hidden_size <= 2048 or c.row.num_layers <= 16),
        CategoryDef("model=mid", lambda c: (3072 <= c.row.hidden_size <= 4096) or (28 <= c.row.num_layers <= 32)),
        CategoryDef("model=large", lambda c: c.row.hidden_size >= 8192 or c.row.num_layers >= 80),
        CategoryDef(
            "comm_heavy",
            lambda c: c.row.dp >= 8 or c.row.tp >= 8 or c.row.pp >= 8,
        ),
        CategoryDef(
            "comm_light",
            lambda c: c.row.dp <= 2 and c.row.tp <= 2 and c.row.pp <= 2,
        ),
        CategoryDef("status=Success", lambda c: c.row.status == "Success"),
        CategoryDef("status=OOM", lambda c: c.row.status == "OOM"),
    ]

    for value in (2, 4, 8, 16, 32):
        defs.append(CategoryDef(f"pp={value}", _eq("pp", value)))
        defs.append(CategoryDef(f"tp={value}", _eq("tp", value)))
        defs.append(CategoryDef(f"dp={value}", _eq("dp", value)))
    for value in (64, 128):
        defs.append(CategoryDef(f"dp={value}", _eq("dp", value)))
    for value in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        defs.append(CategoryDef(f"batch_accum={value}", _eq("batch_accum", value)))
    for value in (1, 2, 4, 8, 16, 32, 64):
        defs.append(CategoryDef(f"mbs={value}", _eq("mbs", value)))

    def _pp1_success(*preds: Callable[[ComparisonResult], bool]) -> Callable[[ComparisonResult], bool]:
        return _and(_eq("pp", 1), lambda c: c.row.status == "Success", *preds)

    defs.append(
        CategoryDef(
            "pp=1,tp=2-16,dp=8-64,status=Success",
            _pp1_success(
                lambda c: 2 <= c.row.tp <= 16,
                lambda c: 8 <= c.row.dp <= 64,
            ),
        )
    )

    defs.append(
        CategoryDef(
            "pp=1,tp=2-8,dp=8-64,status=Success",
            _pp1_success(
                lambda c: 2 <= c.row.tp <= 8,
                lambda c: 8 <= c.row.dp <= 64,
            ),
        )
    )

    # base_tp_dp = [
    #     (8, 2),
    #     (8, 16),
    #     (8, 32),
    #     (8, 64),
    #     (4, 4),
    #     (4, 32),
    #     (4, 64),
    #     (16, 4),
    #     (16, 32),
    #     (32, 1),
    #     (32, 2),
    #     (32, 4),
    #     (2, 64),
    #     (2, 128),
    # ]
    # for tp, dp in base_tp_dp:
    #     defs.append(
    #         CategoryDef(
    #             f"pp=1,tp={tp},dp={dp},status=Success",
    #             _pp1_success(_eq("tp", tp), _eq("dp", dp)),
    #         )
    #     )

    # mbs8_tp_dp = [
    #     (16, 8),
    #     (16, 16),
    #     (32, 16),
    #     (8, 16),
    #     (8, 2),
    #     (32, 1),
    # ]
    # for tp, dp in mbs8_tp_dp:
    #     defs.append(
    #         CategoryDef(
    #             f"pp=1,tp={tp},dp={dp},mbs>=8,status=Success",
    #             _pp1_success(_eq("tp", tp), _eq("dp", dp), _ge("mbs", 8)),
    #         )
    #     )

    # accum_ge_16_tp_dp = [
    #     (8, 2),
    #     (4, 4),
    #     (16, 8),
    # ]
    # for tp, dp in accum_ge_16_tp_dp:
    #     defs.append(
    #         CategoryDef(
    #             f"pp=1,tp={tp},dp={dp},batch_accum>=16,status=Success",
    #             _pp1_success(_eq("tp", tp), _eq("dp", dp), _ge("batch_accum", 16)),
    #         )
    #     )

    # accum_le_8_tp_dp = [
    #     (32, 16),
    #     (16, 16),
    #     (8, 32),
    #     (16, 8),
    #     (8, 16),
    #     (4, 32),
    #     (16, 32),
    #     (4, 64),
    #     (2, 64),
    #     (8, 64),
    # ]
    # for tp, dp in accum_le_8_tp_dp:
    #     defs.append(
    #         CategoryDef(
    #             f"pp=1,tp={tp},dp={dp},batch_accum<=8,status=Success",
    #             _pp1_success(_eq("tp", tp), _eq("dp", dp), _le("batch_accum", 8)),
    #         )
    #     )

    # hidden_sizes = (2048, 3072, 4096)
    # for tp, dp in base_tp_dp:
    #     for hs in hidden_sizes:
    #         defs.append(
    #             CategoryDef(
    #                 f"pp=1,tp={tp},dp={dp},h={hs},status=Success",
    #                 _pp1_success(_eq("tp", tp), _eq("dp", dp), _eq("hidden_size", hs)),
    #             )
    #         )

    # layer_counts = (16, 28, 32)
    # for tp, dp in base_tp_dp:
    #     for layers in layer_counts:
    #         defs.append(
    #             CategoryDef(
    #                 f"pp=1,tp={tp},dp={dp},L={layers},status=Success",
    #                 _pp1_success(_eq("tp", tp), _eq("dp", dp), _eq("num_layers", layers)),
    #             )
    #         )

    return defs


def _sanitize_category_name(name: str) -> str:
    normalized = name.strip()
    replacements = (
        (">=", "ge"),
        ("<=", "le"),
        ("!=", "ne"),
        ("==", "eq"),
        (">", "gr"),
        ("<", "lt"),
        ("=", "eq"),
    )
    for symbol, token in replacements:
        normalized = normalized.replace(symbol, f"_{token}_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "category"


def _select_categories(run_spec: Optional[Sequence[str] | str]) -> List[CategoryDef]:
    if run_spec is None:
        return []
    all_defs = _category_definitions()
    by_name = {c.name: c for c in all_defs}

    if isinstance(run_spec, str):
        spec = run_spec.strip().lower()
        if spec == "none":
            return []
        if spec == "all":
            names = list(by_name.keys())
        elif spec in CATEGORY_PRESETS:
            names = list(CATEGORY_PRESETS[spec])
        else:
            names = [run_spec]
    else:
        names = list(run_spec)

    missing = [n for n in names if n not in by_name]
    if missing and EMIT_LOGS:
        print(f"[category] WARNING: unknown categories skipped: {missing}")

    return [by_name[n] for n in names if n in by_name]


def _collect_category_stats(
    comparisons: List[ComparisonResult],
    mode: str,
    min_rows: int,
) -> Dict[str, Dict[str, Any]]:
    stats_map: Dict[str, Dict[str, Any]] = {}
    for category in _category_definitions():
        subset = [c for c in comparisons if category.predicate(c)]
        if len(subset) < max(0, min_rows):
            continue
        stats = compute_aggregate_stats(subset, mode)
        if not stats:
            continue
        stats_map[category.name] = {
            "stats": stats,
            "count": len(subset),
            "success_count": sum(1 for c in subset if c.row.status == "Success"),
            "oom_count": sum(1 for c in subset if c.row.status == "OOM"),
            "rapid_fail_count": sum(1 for c in subset if not c.rapid_result.success),
        }
    return stats_map


def _auto_select_categories(
    comparisons: List[ComparisonResult],
    mode: str,
    min_rows: int,
    top_k: int,
    metrics: Sequence[str],
) -> Tuple[List[CategoryDef], Dict[str, List[str]]]:
    stats_map = _collect_category_stats(comparisons, mode, min_rows)
    if not stats_map:
        return [], {}

    metric_key_map = {
        "mem": _memory_metric_key(),
        "mem_res": "mem_res",
        "mem_alloc": "mem_alloc",
        "tok_s": "tok_s",
    }
    selected: List[str] = []
    reasons: Dict[str, List[str]] = {}

    for metric in metrics:
        key = metric_key_map.get(metric)
        if not key:
            continue
        candidates: List[Tuple[str, float, int]] = []
        for name, info in stats_map.items():
            stat = info["stats"].get(key)
            if not stat:
                continue
            candidates.append((name, stat.mean_abs_error, int(info["count"])))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[1], -item[2]))
        best = candidates[: max(0, top_k)]
        worst = list(reversed(candidates))[: max(0, top_k)]

        for name, _, _ in best:
            reasons.setdefault(name, []).append(f"best_{metric}")
        for name, _, _ in worst:
            reasons.setdefault(name, []).append(f"worst_{metric}")

        for name, _, _ in best + worst:
            if name not in selected:
                selected.append(name)

    all_defs = {c.name: c for c in _category_definitions()}
    selected_defs = [all_defs[name] for name in selected if name in all_defs]
    return selected_defs, reasons


def _select_categories_with_reasons(
    comparisons: List[ComparisonResult],
    mode: str,
    run_spec: Optional[Sequence[str] | str],
    min_rows: int,
) -> Tuple[List[CategoryDef], Dict[str, List[str]]]:
    if run_spec is None:
        return [], {}
    if isinstance(run_spec, str) and run_spec.strip().lower() == "auto":
        auto_min = max(0, CATEGORY_AUTO_MIN_ROWS)
        return _auto_select_categories(
            comparisons,
            mode,
            auto_min,
            CATEGORY_AUTO_TOP_K,
            CATEGORY_AUTO_METRICS,
        )
    return _select_categories(run_spec), {}


# ==============================================================================
# CSV PARSING
# ==============================================================================

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the benchmark CSV."""
    df = pd.read_csv(csv_path)
    return df


def parse_row(row: pd.Series, idx: int) -> BenchmarkRow:
    """Convert a DataFrame row to BenchmarkRow dataclass."""
    row_name = str(row.get('name', '')).strip() if 'name' in row else ''

    def _fail(message: str) -> None:
        label = f"row {idx}"
        if row_name:
            label += f" ({row_name})"
        raise ValueError(f"{label}: {message}")

    if 'status' not in row or pd.isna(row['status']):
        _fail("missing value for 'status'")
    status = str(row['status'])
    status_norm = status.strip()
    if status_norm != "Success":
        def safe_int(val, default=0):
            try:
                if pd.isna(val):
                    return default
                return int(float(val))
            except (ValueError, TypeError):
                return default

        def safe_float(val, default=0.0):
            try:
                if pd.isna(val):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_optional_int(val):
            try:
                if pd.isna(val) or val == "":
                    return None
                return int(float(val))
            except (ValueError, TypeError):
                return None

        def safe_bool(val, default=False):
            if pd.isna(val):
                return default
            if isinstance(val, bool):
                return val
            return str(val).lower() in ('true', '1', 'yes')

        dp = safe_int(row.get('dp', 1), 1)
        pp = safe_int(row.get('pp', 1), 1)
        tp = safe_int(row.get('tp', 1), 1)
        num_heads = safe_int(row.get('num_heads', 32), 32)

        return BenchmarkRow(
            dp=dp,
            pp=pp,
            tp=tp,
            nodes=safe_int(row.get('nodes', 1), 1),
            mbs=safe_int(row.get('mbs', 1), 1),
            batch_accum=safe_int(row.get('batch_accum', 1), 1),
            gbs=safe_int(row.get('gbs', 1), 1),
            seq_len=safe_int(row.get('seq_len', 2048), 2048),
            hidden_size=safe_int(row.get('hidden_size', 2048), 2048),
            interm_dim=safe_optional_int(row.get('interm_dim')),
            num_layers=safe_int(row.get('num_layers', 16), 16),
            num_heads=num_heads,
            num_kv_heads=safe_int(row.get('num_kv_heads', num_heads), num_heads),
            vocab_size=safe_int(row.get('vocab_size', 32000), 32000),
            zero_stage=safe_int(row.get('zero_stage', 0), 0),
            dtype=str(row.get('dtype', '')) if not pd.isna(row.get('dtype')) else '',
            ar_inter=safe_float(row.get('AllReduce (GB/s)', 0)),
            ag_inter=safe_float(row.get('AllGather (GB/s)', 0)),
            rs_inter=safe_float(row.get('ReduceScatter (GB/s)', 0)),
            ar_intra=safe_float(row.get('AR Intra-node (GB/s)', 0)),
            ag_intra=safe_float(row.get('AG Intra-node (GB/s)', 0)),
            rs_intra=safe_float(row.get('RS Intra-node (GB/s)', 0)),
            mem_alloc_gb=safe_float(row.get('Mem Alloc (GB)', 0)),
            mem_res_gb=safe_float(row.get('Mem Res (GB)', 0)),
            mtflops=safe_float(row.get('mTFLOPs', 0)),
            tok_s_gpu=safe_float(row.get('tok/s/gpu', 0)),
            mfu=safe_float(row.get('mfu', 0)),
            status=status,
            oom_error=safe_bool(row.get('oom_error', False)),
            after_pp_fix=safe_bool(row.get('after_pp_fix', False)),
            row_index=idx,
            num_gpus=dp * pp * tp,
        )

    def _require_value(key: str):
        if key not in row:
            _fail(f"missing column '{key}'")
        val = row[key]
        if pd.isna(val):
            _fail(f"missing value for '{key}'")
        return val

    def _parse_int(val, key: str) -> int:
        try:
            fval = float(val)
        except (ValueError, TypeError):
            _fail(f"invalid int for '{key}': {val!r}")
        if abs(fval - round(fval)) > 1e-6:
            _fail(f"non-integer value for '{key}': {val!r}")
        return int(round(fval))

    def _require_int(key: str) -> int:
        return _parse_int(_require_value(key), key)

    def _require_float(key: str) -> float:
        val = _require_value(key)
        try:
            return float(val)
        except (ValueError, TypeError):
            _fail(f"invalid float for '{key}': {val!r}")

    def _require_bool(key: str) -> bool:
        val = _require_value(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and val in (0, 1):
            return bool(int(val))
        sval = str(val).strip().lower()
        if sval in ("true", "1", "yes"):
            return True
        if sval in ("false", "0", "no"):
            return False
        _fail(f"invalid bool for '{key}': {val!r}")

    def _optional_int(key: str) -> Optional[int]:
        if key not in row:
            _fail(f"missing column '{key}'")
        val = row[key]
        if pd.isna(val) or val == "":
            return None
        return _parse_int(val, key)

    dp = _require_int('dp')
    pp = _require_int('pp')
    tp = _require_int('tp')

    return BenchmarkRow(
        dp=dp,
        pp=pp,
        tp=tp,
        nodes=_require_int('nodes'),
        mbs=_require_int('mbs'),
        batch_accum=_require_int('batch_accum'),
        gbs=_require_int('gbs'),
        seq_len=_require_int('seq_len'),
        hidden_size=_require_int('hidden_size'),
        interm_dim=_optional_int('interm_dim'),
        num_layers=_require_int('num_layers'),
        num_heads=_require_int('num_heads'),
        num_kv_heads=_require_int('num_kv_heads'),
        vocab_size=_require_int('vocab_size'),
        zero_stage=_require_int('zero_stage'),
        dtype=str(_require_value('dtype')),
        ar_inter=_require_float('AllReduce (GB/s)'),
        ag_inter=_require_float('AllGather (GB/s)'),
        rs_inter=_require_float('ReduceScatter (GB/s)'),
        ar_intra=_require_float('AR Intra-node (GB/s)'),
        ag_intra=_require_float('AG Intra-node (GB/s)'),
        rs_intra=_require_float('RS Intra-node (GB/s)'),
        mem_alloc_gb=_require_float('Mem Alloc (GB)'),
        mem_res_gb=_require_float('Mem Res (GB)'),
        mtflops=_require_float('mTFLOPs'),
        tok_s_gpu=_require_float('tok/s/gpu'),
        mfu=_require_float('mfu'),
        status=status,
        oom_error=_require_bool('oom_error'),
        after_pp_fix=_require_bool('after_pp_fix'),
        row_index=idx,
        num_gpus=dp * pp * tp,
    )


# ==============================================================================
# FILTERING
# ==============================================================================

def filter_rows(
    rows: List[BenchmarkRow],
    mode: str,
    assume_bf16: bool,
    filter_pp_fix: bool,
    max_rows: Optional[int],
) -> List[BenchmarkRow]:
    """Filter rows based on mode and configuration."""
    filtered = []

    for row in rows:
        if not row.interm_dim or row.interm_dim <= 0:
            continue
        # Skip empty dtype if not assuming bf16
        if not assume_bf16 and not row.dtype:
            continue

        # Filter by pp_fix if enabled
        if filter_pp_fix and not row.after_pp_fix:
            continue

        # Always skip 'Other' status (timeout/error)
        if row.status not in ('Success', 'OOM'):
            continue

        # perf_only: skip OOM rows
        if mode == 'perf_only' and row.status != 'Success':
            continue

        filtered.append(row)

    # Apply max_rows limit
    if max_rows is not None:
        filtered = filtered[:max_rows]

    return filtered


def _should_restrict_to_categories(run_spec: Optional[Sequence[str] | str]) -> bool:
    if run_spec is None:
        return False
    if isinstance(run_spec, str):
        spec = run_spec.strip().lower()
        if spec in ("all", "none", "auto"):
            return False
    return True


def _restrict_rows_by_categories(
    rows: List[BenchmarkRow],
    run_spec: Optional[Sequence[str] | str],
    emit_logs: bool,
) -> List[BenchmarkRow]:
    if not rows or not _should_restrict_to_categories(run_spec):
        return rows
    categories = _select_categories(run_spec)
    if not categories:
        return rows

    stub_result = RAPIDResult(success=True)
    selected: List[BenchmarkRow] = []
    for row in rows:
        comp = ComparisonResult(row=row, rapid_result=stub_result)
        if any(category.predicate(comp) for category in categories):
            selected.append(row)

    if emit_logs:
        names = [c.name for c in categories]
        print(f"[category] Pre-filter rows: {len(selected)}/{len(rows)} using {names}")
    return selected


def load_success_rows(
    csv_path: Path,
    assume_bf16: bool = ASSUME_BF16,
    filter_pp_fix: bool = FILTER_PP_FIX,
    max_rows: Optional[int] = None,
    mode: str = "perf_only",
) -> List[BenchmarkRow]:
    """Load Success-only rows (skip OOM/Other before parsing)."""
    df = load_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "Success"].copy()
    rows = [parse_row(row, int(idx)) for idx, row in df.iterrows()]
    return filter_rows(rows, mode, assume_bf16, filter_pp_fix, max_rows)


# ==============================================================================
# CONFIG BUILDING (IN-MEMORY)
# ==============================================================================

def load_base_hw_config(hw_config_path: Path) -> Dict[str, Any]:
    """Load H100_SXM5_80GB.yaml as base template."""
    with open(hw_config_path, 'r') as f:
        return yaml.safe_load(f)


def rapid_mb_and_gas(row: BenchmarkRow) -> Tuple[int, int]:
    """Return (mb, gradient_accumulation_steps) for RAPID vs. Nanotron PP handling."""
    batch_accum = max(1, int(row.batch_accum) if row.batch_accum else 1)
    if row.pp <= 1:
        return 1, batch_accum
    return batch_accum, 1


def _avg_positive(values: Sequence[float]) -> Optional[float]:
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return sum(positive) / float(len(positive))

def _min_positive(values: Sequence[float]) -> Optional[float]:
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return min(positive)

def build_hw_config_dict(
    base_config: Dict[str, Any],
    row: BenchmarkRow,
) -> Dict[str, Any]:
    """Build hardware config dict with overrides from benchmark row."""
    config = copy.deepcopy(base_config)
    rapid_mb, _ = rapid_mb_and_gas(row)

    # Override parallelism
    config['parallelism'] = {
        'auto': False,
        'tp': row.tp,
        'pp': row.pp,
        'cp': 1,
        'mb': rapid_mb,
        'tp_sp': True,
        'train': {'dp': row.dp, 'ep': 1, 'tp_ep': True},
        'inference': {'replica_count': 1, 'moe_dp': 1},
    }

    if 'sw_param' not in config:
        config['sw_param'] = {}
    config['sw_param']['dp_zero_stage'] = row.zero_stage

    # Override network dimensions with bandwidths from CSV.
    # Always use 3 dims: inner=tp/cp, mid=pp, outer=dp.
    tp_use_inter = row.tp > GCN
    pp_use_inter = (row.tp * row.pp) > GCN
    dp_use_inter = (row.nodes > 1) and (row.dp > 1)

    # tp_bw = _avg_positive(
    #     [row.rs_inter, row.ag_inter] if tp_use_inter else [row.rs_intra, row.ag_intra]
    # )
    # pp_bw = _avg_positive(
    #     [row.rs_inter, row.ag_inter] if pp_use_inter else [row.rs_intra, row.ag_intra]
    # )
    # dp_bw = _avg_positive(
    #     [row.ar_inter] if dp_use_inter else [row.ar_intra]
    # )

    tp_bw = _min_positive(
        [row.rs_inter, row.ag_inter] if tp_use_inter else [row.rs_intra, row.ag_intra]
    )
    tp_bw_intra = _min_positive([row.rs_intra, row.ag_intra])
    tp_bw_inter = _min_positive([row.rs_inter, row.ag_inter])
    pp_bw = _min_positive(
        [row.rs_inter, row.ag_inter] if pp_use_inter else [row.rs_intra, row.ag_intra]
    )
    dp_bw = _min_positive(
        [row.ar_inter] if dp_use_inter else [row.ar_intra]
    )

    dims = config.get('network', {}).get('dimensions', [])
    if len(dims) >= 1:
        dims[0]['parallelisms'] = ['tp', 'cp']
        topo0 = dims[0].setdefault('topology', {})
        if tp_bw is not None and tp_bw > 0:
            topo0['bandwidth'] = f"{int(round(tp_bw))} GB"
        tp_total = 1
        for axis in dims[0]['parallelisms']:
            try:
                tp_total *= int(config['parallelism'].get(axis, 1))
            except (TypeError, ValueError):
                tp_total *= 1
        if tp_total > 8:
            topo0['type'] = 'FC-Ring2D'
            dims[0]['size'] = (8, 'auto')
            tp_intra = tp_bw_intra if tp_bw_intra and tp_bw_intra > 0 else tp_bw
            tp_inter = tp_bw_inter if tp_bw_inter and tp_bw_inter > 0 else tp_bw
            if tp_intra is not None and tp_inter is not None:
                topo0['bandwidth'] = (
                    f"{int(round(tp_intra))} GB",
                    f"{int(round(tp_inter))} GB",
                )
    if len(dims) >= 2:
        dims[1]['parallelisms'] = ['pp']
        if pp_bw is not None and pp_bw > 0:
            dims[1].setdefault('topology', {})['bandwidth'] = f"{int(round(pp_bw))} GB"
    if len(dims) >= 3:
        dims[2]['parallelisms'] = ['dp']
        if dp_bw is not None and dp_bw > 0:
            dims[2].setdefault('topology', {})['bandwidth'] = f"{int(round(dp_bw))} GB"

    return config


def build_model_config_dict(
    row: BenchmarkRow,
    flash_attention: bool = False,
    attention_tile_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Build model config dict from benchmark row."""
    # GQA vs MHA
    attention_type = 'gqa' if row.num_kv_heads < row.num_heads else 'mha'

    _, rapid_gas = rapid_mb_and_gas(row)

    intermediate_size = int(row.interm_dim) if row.interm_dim else None
    if not intermediate_size or intermediate_size <= 0:
        raise ValueError("Missing interm_dim for model config.")

    return {
        'model_param': {
            'mode': 'LLM',
            'run_type': 'training',
            'model_type': 'llama',
            'tied_embeddings': True,
            'global_batch_size': row.gbs,
            'gradient_accumulation_steps': rapid_gas,
            'seq_len': row.seq_len,
            'hidden_dim': row.hidden_size,
            'num_layers': row.num_layers,
            'intermediate_size': intermediate_size,
            'vocab_size': row.vocab_size,
            'moe': {
                'num_experts': 1,
                'top_k': 1,
                'moe_intermediate_size': intermediate_size,
                'n_shared_experts': 0,
                'moe_layer_freq': 1,
                'first_k_dense_replace': row.num_layers,
            },
            'attention': {
                'attention_type': attention_type,
                'num_heads': row.num_heads,
                'kv_heads': row.num_kv_heads if attention_type == 'gqa' else None,
                'use_flashattention': flash_attention,
                'attention_tile_size': attention_tile_size,
            },
        }
    }


# ==============================================================================
# RAPID RUNNER
# ==============================================================================

# Global for worker context
_BASE_HW_CONFIG: Optional[Dict[str, Any]] = None
_WORKER_STDOUT = None
_WORKER_STDERR = None


def _worker_init(base_hw_config: Dict[str, Any]) -> None:
    """Initialize worker with base config."""
    _ensure_project_root_on_path()
    global _BASE_HW_CONFIG, _WORKER_STDOUT, _WORKER_STDERR
    _BASE_HW_CONFIG = base_hw_config
    if SUPPRESS_WORKER_OUTPUT:
        # Silence stdout/stderr from worker processes to keep logs clean.
        _WORKER_STDOUT = open(os.devnull, 'w')
        _WORKER_STDERR = open(os.devnull, 'w')
        sys.stdout = _WORKER_STDOUT
        sys.stderr = _WORKER_STDERR


def _infer_actual_microbatch_count(row: BenchmarkRow) -> int:
    """Return the expected microbatch count per DP rank."""
    rapid_mb, _ = rapid_mb_and_gas(row)
    if row.pp <= 1:
        return max(1, int(rapid_mb))
    if not USE_DERIVED_MICROBATCH_COUNT:
        return max(1, int(rapid_mb))
    denom = int(row.dp) * int(row.mbs) if row.dp and row.mbs else 0
    if denom > 0:
        derived = row.gbs / denom
        if abs(derived - round(derived)) < 1e-6:
            return max(1, int(round(derived)))
    return max(1, int(rapid_mb))


def _infer_microbatch_count(row: BenchmarkRow) -> int:
    """Return microbatch count used to build the pipeline graph (bounded)."""
    mb_actual = _infer_actual_microbatch_count(row)
    mb_graph = mb_actual
    if MB_GRAPH_CAP is not None and int(MB_GRAPH_CAP) > 0:
        mb_graph = min(mb_graph, int(MB_GRAPH_CAP))
    return max(1, int(mb_graph))


def _pipeline_activation_window_factor(row: BenchmarkRow, tc: Any) -> float:
    """Approximate 1F1B activation window vs GPipe microbatch count."""
    schedule = str(PIPELINE_MEM_SCHEDULE or "").strip().lower()
    if schedule in ("gpipe", "off", "none"):
        return 1.0
    if row.pp <= 1:
        return 1.0
    mb_actual = _infer_actual_microbatch_count(row)
    mb_graph = max(1, int(getattr(tc, "mb", 1)))
    pp = max(1, int(getattr(tc, "pp", 1)))
    if mb_actual <= 1 or pp <= 1 or mb_graph <= 0:
        return 1.0
    # window = max(1, int(math.ceil(pp * PIPELINE_ACTIVATION_WINDOW_MULT)))
    # try pp * mult and also (pp*mult*2-2)
    window = max(1, int(math.ceil(pp * PIPELINE_ACTIVATION_WINDOW_MULT * 2 - 2)))
    window = min(window, mb_actual)
    return float(window) / float(mb_graph)


def _apply_activation_window_scaling(memory_data: Dict[str, Any], factor: float) -> None:
    """Scale activation bytes to approximate in-flight 1F1B window."""
    if factor <= 0 or abs(factor - 1.0) < 1e-6:
        return
    try:
        from memory_estimation import MemKind, TRANSFORMER_OP_KINDS
    except Exception:
        MemKind = None
        TRANSFORMER_OP_KINDS = None

    for key in ("persistent_bytes_by_kind", "transient_bytes_by_kind"):
        mapping = memory_data.get(key)
        if not mapping:
            continue
        for kind in list(mapping.keys()):
            if MemKind is not None and TRANSFORMER_OP_KINDS is not None:
                if kind not in TRANSFORMER_OP_KINDS:
                    continue
            mapping[kind] = float(mapping.get(kind, 0.0) or 0.0) * factor

    if "activation_mem_per_layer" in memory_data:
        memory_data["activation_mem_per_layer"] = float(
            memory_data.get("activation_mem_per_layer", 0.0) or 0.0
        ) * factor

    if "total_mem_per_layer" in memory_data:
        weight = float(memory_data.get("weight_mem_per_layer", 0.0) or 0.0)
        memory_data["total_mem_per_layer"] = weight + float(
            memory_data.get("activation_mem_per_layer", 0.0) or 0.0
        )


def _prepare_astra_tmp_dir() -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """Create a per-run temp directory and isolate AstraSim cache/output."""
    if ASTRA_TMP_ROOT is None:
        return None, {}
    tmp_root = Path(ASTRA_TMP_ROOT)
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="hf_validation_", dir=str(tmp_root))
    cache_dir = os.path.join(run_dir, "astra_cache")
    os.makedirs(cache_dir, exist_ok=True)
    prev_env = {
        "ASTRA_CACHE_DIR": os.environ.get("ASTRA_CACHE_DIR"),
        "RAPID_ASTRA_CACHE_MODE": os.environ.get("RAPID_ASTRA_CACHE_MODE"),
    }
    os.environ["ASTRA_CACHE_DIR"] = cache_dir
    if ASTRA_CACHE_MODE is not None:
        os.environ["RAPID_ASTRA_CACHE_MODE"] = str(ASTRA_CACHE_MODE)
    return run_dir, prev_env


def _restore_astra_env(prev_env: Dict[str, Optional[str]]) -> None:
    """Restore environment variables modified for AstraSim isolation."""
    for key, value in prev_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _compute_peak_tflops(hw_config) -> float:
    """Derive peak TFLOPs per GPU from hardware config."""
    tech = hw_config.tech_config
    freq = tech.core.operating_frequency
    sms = tech.core.num_bundles
    flops_per_mcu = tech.core.nominal_flop_rate_per_mcu
    mcus_per_bundle = tech.core.num_mcu_per_bundle
    peak = freq * sms * flops_per_mcu * mcus_per_bundle / 1e12
    return peak



def _tok_s_denominator(row: BenchmarkRow, mode: Optional[str] = None) -> int:
    mode = (mode or TOKS_PER_GPU_DENOM or "num_gpus").strip().lower()
    if mode in ("dp_tp", "dp*tp"):
        denom = row.dp * row.tp
    elif mode in ("dp",):
        denom = row.dp
    elif mode in ("num_gpus", "dp_tp_pp", "dp*tp*pp"):
        denom = row.num_gpus
    else:
        denom = row.num_gpus
    return max(int(denom), 1)


def _tok_s_to_time(row: BenchmarkRow, denom_mode: Optional[str] = None) -> float:
    """Convert tok/s/gpu to training time for one batch."""
    if row.tok_s_gpu <= 0:
        return 0.0
    total_tokens = row.gbs * row.seq_len
    total_tok_s = row.tok_s_gpu * _tok_s_denominator(row, denom_mode)
    return total_tokens / total_tok_s


def run_rapid_estimation(
    row: BenchmarkRow,
    mode: str,
    flash_attention: bool = False,
    attention_tile_size: Optional[int] = None,
) -> RAPIDResult:
    """Run RAPID LLM estimation and return results."""
    global _BASE_HW_CONFIG

    if _BASE_HW_CONFIG is None:
        return RAPIDResult(success=False, error="Worker not initialized")

    result = RAPIDResult(success=True)
    run_dir: Optional[str] = None
    prev_env: Dict[str, Optional[str]] = {}

    try:
        run_dir, prev_env = _prepare_astra_tmp_dir()

        # Build configs in memory
        hw_config_dict = build_hw_config_dict(_BASE_HW_CONFIG, row)
        model_config_dict = build_model_config_dict(row, flash_attention, attention_tile_size)

        # Import RAPID modules (done here to avoid import issues in workers)
        from config import HWConfig, LLMConfig, ModelConfig, convert, validate_configs
        from train_timing import TimeCalculationLLM

        # Apply convert() to handle string units
        convert(hw_config_dict)
        convert(model_config_dict)

        # Parse configs
        hw_config = HWConfig.from_dict(hw_config_dict)
        llm_config = LLMConfig.from_dict(model_config_dict['model_param'])
        model_config = ModelConfig(model_config=llm_config, inference_config=None)

        # Validate
        validate_configs(hw_config, model_config)

        if mode in ('mem_only', 'both'):
            # Memory estimation via TimeCalculationLLM
            tc = TimeCalculationLLM(hw_config, model_config, 'LLM', output_dir=run_dir)
            activation_factor = _pipeline_activation_window_factor(row, tc)
            if activation_factor >= 1.0:
                mem_result = tc.estimate_memory_only()
                result.peak_gb = mem_result.get('peak_gb')
                result.capacity_exceeded = mem_result.get('capacity_exceeded', False)
            else:
                from llm_execution import LLMExecutionDispatcher
                mem_estimator, memory_data = tc._build_training_graphs_and_memory_data()
                _apply_activation_window_scaling(memory_data, activation_factor)
                dispatcher = LLMExecutionDispatcher(
                    time_calc=tc,
                    pipeline_graph=tc.pipeline_graph,
                    pipeline_root=tc.pipeline_root,
                    interconnect_params=tc.pipeline_interconnect,
                    transformer_graph=tc.transformer_graph,
                    transformer_forward_root=tc.transformer_forward_root,
                    transformer_backward_root=tc.transformer_backward_root,
                    no_data_parallel=False,
                )
                memory_root = dispatcher.build_flattened_root_for_memory()
                _, training_peak_gb = mem_estimator.simulate_peak(
                    memory_root,
                    memory_data,
                    mode="training",
                    filename="memory_graph_training",
                )
                result.peak_gb = float(training_peak_gb)
                result.capacity_exceeded = bool(getattr(tc, "memory_capacity_exceeded", False))

        if mode in ('perf_only', 'both') and row.status == 'Success':
            # Performance estimation
            tc = TimeCalculationLLM(hw_config, model_config, 'LLM', output_dir=run_dir)
            tc.calc_time_llm()
            result.training_time_s = tc.get_time()

            # Compute derived metrics
            if result.training_time_s and result.training_time_s > 0:
                total_tokens = row.gbs * row.seq_len
                result.tok_s_gpu = total_tokens / (result.training_time_s * row.num_gpus)

                # MFU
                peak_tflops = _compute_peak_tflops(hw_config)
                # Skip TFLOPs calculation for now per user request
                result.mfu = None

    except Exception as e:
        result.success = False
        result.error = str(e)
    finally:
        if prev_env:
            _restore_astra_env(prev_env)
        if run_dir and CLEANUP_ASTRA_TMP:
            shutil.rmtree(run_dir, ignore_errors=True)

    return result


def _evaluate_row_worker(
    row: BenchmarkRow,
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
) -> ComparisonResult:
    """Compute a ComparisonResult for a single row (worker-safe)."""
    rapid_result = run_rapid_estimation(
        row,
        mode=mode,
        flash_attention=flash_attention,
        attention_tile_size=attention_tile_size,
    )
    return compute_comparison(row, rapid_result, mode)


def evaluate_rows(
    rows: Sequence[BenchmarkRow],
    base_hw_config: Dict[str, Any],
    mode: str = "perf_only",
    num_workers: int = NUM_WORKERS,
    flash_attention: bool = USE_FLASH_ATTENTION,
    attention_tile_size: Optional[int] = ATTENTION_TILE_SIZE,
    emit_progress: bool = False,
    fast_mode: bool = False,
    timeout_s: Optional[float] = None,
) -> List[ComparisonResult]:
    """Evaluate a list of rows without caching or file outputs."""
    if not rows:
        return []

    worker_count = min(int(num_workers), len(rows), os.cpu_count() or 1)
    if worker_count <= 1:
        _ensure_project_root_on_path()
        global _BASE_HW_CONFIG
        _BASE_HW_CONFIG = base_hw_config
        comparisons = [
            _evaluate_row_worker(row, mode, flash_attention, attention_tile_size)
            for row in rows
        ]
        comparisons.sort(key=lambda c: c.row.row_index)
        return comparisons

    progress = None
    if emit_progress:
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(rows), desc="Evaluating")
        except Exception:
            progress = None

    comparisons: List[ComparisonResult] = []

    if fast_mode:
        def _handle_result(result_data: Dict[str, Any], row: BenchmarkRow) -> None:
            comparisons.append(_result_data_to_comparison(result_data, row, mode))

        _run_rows_parallel_inmemory(
            list(rows),
            base_hw_config,
            mode,
            flash_attention,
            attention_tile_size,
            worker_count,
            progress,
            _handle_result,
            fast_mode=True,
            timeout_s=timeout_s,
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=worker_count,
            initializer=_worker_init,
            initargs=(base_hw_config,),
        ) as pool:
            args = [(row, mode, flash_attention, attention_tile_size) for row in rows]
            if progress is None:
                comparisons = pool.starmap(_evaluate_row_worker, args)
            else:
                for comp in pool.starmap(_evaluate_row_worker, args):
                    comparisons.append(comp)
                    progress.update(1)

    if progress is not None:
        progress.close()

    comparisons.sort(key=lambda c: c.row.row_index)
    return comparisons


def _process_single_row(row: BenchmarkRow) -> ComparisonResult:
    """Process a single benchmark row (called by worker)."""
    rapid_result = run_rapid_estimation(
        row,
        mode=MODE,
        flash_attention=USE_FLASH_ATTENTION,
        attention_tile_size=ATTENTION_TILE_SIZE,
    )
    return compute_comparison(row, rapid_result, MODE)


def _worker_process_row(
    row_dict: Dict[str, Any],
    hw_config_path: str,
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
    result_queue: Queue,
) -> None:
    """
    Standalone worker function for processing a single row.
    Loads everything from scratch to minimize fork memory.
    """
    row_index = row_dict.get('row_index', -1)
    try:
        # Ensure project root is on path
        _ensure_project_root_on_path()

        # Suppress output if configured
        if SUPPRESS_WORKER_OUTPUT:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        # Load hw config from disk (not passed via fork)
        with open(hw_config_path, 'r') as f:
            base_hw_config = yaml.safe_load(f)

        # Reconstruct BenchmarkRow from dict
        row = BenchmarkRow(**row_dict)

        # Build configs
        hw_config_dict = build_hw_config_dict(base_hw_config, row)
        model_config_dict = build_model_config_dict(row, flash_attention, attention_tile_size)

        # Prepare temp dir for AstraSim
        run_dir, prev_env = _prepare_astra_tmp_dir()

        try:
            # Import RAPID modules
            from config import HWConfig, LLMConfig, ModelConfig, convert, validate_configs
            from train_timing import TimeCalculationLLM

            convert(hw_config_dict)
            convert(model_config_dict)

            hw_config = HWConfig.from_dict(hw_config_dict)
            llm_config = LLMConfig.from_dict(model_config_dict['model_param'])
            model_config = ModelConfig(model_config=llm_config, inference_config=None)
            validate_configs(hw_config, model_config)

            # Results to return (minimal data)
            result_data = {
                'row_index': row_index,
                'success': True,
                'error': None,
                'peak_gb': None,
                'capacity_exceeded': False,
                'training_time_s': None,
                'tok_s_gpu': None,
                'mfu': None,
            }

            if mode in ('mem_only', 'both'):
                tc = TimeCalculationLLM(hw_config, model_config, 'LLM', output_dir=run_dir)
                activation_factor = _pipeline_activation_window_factor(row, tc)
                if activation_factor >= 1.0:
                    mem_result = tc.estimate_memory_only()
                    result_data['peak_gb'] = mem_result.get('peak_gb')
                    result_data['capacity_exceeded'] = mem_result.get('capacity_exceeded', False)
                else:
                    from llm_execution import LLMExecutionDispatcher
                    mem_estimator, memory_data = tc._build_training_graphs_and_memory_data()
                    _apply_activation_window_scaling(memory_data, activation_factor)
                    dispatcher = LLMExecutionDispatcher(
                        time_calc=tc,
                        pipeline_graph=tc.pipeline_graph,
                        pipeline_root=tc.pipeline_root,
                        interconnect_params=tc.pipeline_interconnect,
                        transformer_graph=tc.transformer_graph,
                        transformer_forward_root=tc.transformer_forward_root,
                        transformer_backward_root=tc.transformer_backward_root,
                        no_data_parallel=False,
                    )
                    memory_root = dispatcher.build_flattened_root_for_memory()
                    _, training_peak_gb = mem_estimator.simulate_peak(
                        memory_root,
                        memory_data,
                        mode="training",
                        filename="memory_graph_training",
                    )
                    result_data['peak_gb'] = float(training_peak_gb)
                    result_data['capacity_exceeded'] = bool(
                        getattr(tc, "memory_capacity_exceeded", False)
                    )

            if mode in ('perf_only', 'both') and row.status == 'Success':
                tc = TimeCalculationLLM(hw_config, model_config, 'LLM', output_dir=run_dir)
                tc.calc_time_llm()
                training_time_s = tc.get_time()
                result_data['training_time_s'] = training_time_s

                if training_time_s and training_time_s > 0:
                    total_tokens = row.gbs * row.seq_len
                    result_data['tok_s_gpu'] = total_tokens / (training_time_s * row.num_gpus)

            result_queue.put(result_data)

        finally:
            if prev_env:
                _restore_astra_env(prev_env)
            if run_dir and CLEANUP_ASTRA_TMP:
                shutil.rmtree(run_dir, ignore_errors=True)

    except Exception as e:
        result_queue.put({
            'row_index': row_index,
            'success': False,
            'error': str(e),
            'peak_gb': None,
            'capacity_exceeded': False,
            'training_time_s': None,
            'tok_s_gpu': None,
            'mfu': None,
        })


def _worker_process_row_inmemory(
    row_dict: Dict[str, Any],
    base_hw_config: Dict[str, Any],
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
    result_queue: Queue,
) -> None:
    """Worker that uses an in-memory hardware config (no file IO)."""
    row_index = row_dict.get('row_index', -1)
    try:
        _ensure_project_root_on_path()
        if SUPPRESS_WORKER_OUTPUT:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        row = BenchmarkRow(**row_dict)
        global _BASE_HW_CONFIG
        _BASE_HW_CONFIG = base_hw_config

        rapid_result = run_rapid_estimation(
            row,
            mode=mode,
            flash_attention=flash_attention,
            attention_tile_size=attention_tile_size,
        )

        result_queue.put({
            'row_index': row_index,
            'success': rapid_result.success,
            'error': rapid_result.error,
            'peak_gb': rapid_result.peak_gb,
            'capacity_exceeded': rapid_result.capacity_exceeded,
            'training_time_s': rapid_result.training_time_s,
            'tok_s_gpu': rapid_result.tok_s_gpu,
            'mfu': rapid_result.mfu,
        })
    except Exception as e:
        result_queue.put({
            'row_index': row_index,
            'success': False,
            'error': str(e),
            'peak_gb': None,
            'capacity_exceeded': False,
            'training_time_s': None,
            'tok_s_gpu': None,
            'mfu': None,
        })


# ==============================================================================
# COMPARISON
# ==============================================================================

def _pct_error(predicted: float, actual: float) -> Optional[float]:
    """Compute signed percentage error."""
    if actual == 0 or math.isnan(actual) or predicted is None or math.isnan(predicted):
        return None
    return ((predicted - actual) / actual) * 100.0


def _memory_metric_key() -> str:
    target = str(MEMORY_ERROR_TARGET or "res").strip().lower()
    if target in ("alloc", "allocated", "allocation", "mem_alloc"):
        return "mem_alloc"
    return "mem_res"


def _memory_metric_label() -> str:
    return "Alloc" if _memory_metric_key() == "mem_alloc" else "Res"


def _memory_error_value(result: ComparisonResult) -> Optional[float]:
    return result.mem_alloc_error_pct if _memory_metric_key() == "mem_alloc" else result.mem_res_error_pct


def _memory_actual_value(row: BenchmarkRow) -> float:
    return row.mem_alloc_gb if _memory_metric_key() == "mem_alloc" else row.mem_res_gb


def _matches_selective_id(row: BenchmarkRow, token: str) -> bool:
    token_norm = str(token or "").strip()
    if not token_norm:
        return False
    if token_norm.isdigit() and int(token_norm) == row.row_index:
        return True
    config_id = _format_config_label(row)
    if token_norm == config_id or token_norm == row.label:
        return True
    return False


def compute_comparison(
    row: BenchmarkRow,
    rapid_result: RAPIDResult,
    mode: str,
) -> ComparisonResult:
    """Compute error metrics between RAPID and ground truth."""
    result = ComparisonResult(row=row, rapid_result=rapid_result)

    if mode in ('mem_only', 'both'):
        if rapid_result.peak_gb is not None:
            if row.mem_alloc_gb > 0:
                result.mem_alloc_error_pct = _pct_error(rapid_result.peak_gb, row.mem_alloc_gb)
            if row.mem_res_gb > 0:
                result.mem_res_error_pct = _pct_error(rapid_result.peak_gb, row.mem_res_gb)

    if mode in ('perf_only', 'both') and row.status == 'Success':
        if rapid_result.tok_s_gpu is not None and row.tok_s_gpu > 0:
            result.tok_s_error_pct = _pct_error(rapid_result.tok_s_gpu, row.tok_s_gpu)

        # MFU comparison (skipped for now)
        if rapid_result.mfu is not None and row.mfu > 0:
            result.mfu_error_pct = _pct_error(rapid_result.mfu, row.mfu)

    return result


# ==============================================================================
# STATISTICS AND REPORTING
# ==============================================================================

@dataclass
class AggregateStats:
    """Aggregate error statistics."""
    metric_name: str
    count: int
    mean_error: float
    median_error: float
    std_error: float
    mean_abs_error: float
    p90_error: float
    p95_error: float


def compute_aggregate_stats(
    comparisons: List[ComparisonResult],
    mode: str,
) -> Dict[str, AggregateStats]:
    """Compute aggregate statistics for all metrics."""
    import numpy as np

    def _compute_stats(name: str, errors: List[float]) -> AggregateStats:
        arr = np.array(errors)
        return AggregateStats(
            metric_name=name,
            count=len(arr),
            mean_error=float(np.mean(arr)),
            median_error=float(np.median(arr)),
            std_error=float(np.std(arr)),
            mean_abs_error=float(np.mean(np.abs(arr))),
            p90_error=float(np.percentile(np.abs(arr), 90)),
            p95_error=float(np.percentile(np.abs(arr), 95)),
        )

    stats = {}

    if mode in ('mem_only', 'both'):
        mem_errors = [
            _memory_error_value(c) for c in comparisons
            if _memory_error_value(c) is not None and not math.isnan(_memory_error_value(c))
        ]
        if mem_errors:
            memory_key = _memory_metric_key()
            stats[memory_key] = _compute_stats(
                f"Memory vs {_memory_metric_label()} Error",
                mem_errors,
            )

    if mode in ('perf_only', 'both'):
        tok_errors = [c.tok_s_error_pct for c in comparisons
                      if c.tok_s_error_pct is not None and not math.isnan(c.tok_s_error_pct)]
        if tok_errors:
            stats['tok_s'] = _compute_stats('tok/s/gpu Error', tok_errors)

    return stats


def _cap_error_overflow(errors: Sequence[float], cap: float) -> Tuple[List[float], int]:
    capped: List[float] = []
    overflow = 0
    for err in errors:
        if err > cap:
            capped.append(cap)
            overflow += 1
        else:
            capped.append(err)
    return capped, overflow


def _label_overflow_xtick(ax, cap: float, label: str) -> None:
    ticks = list(ax.get_xticks())
    if not any(math.isclose(tick, cap, abs_tol=1e-6) for tick in ticks):
        ticks.append(cap)
        ticks.sort()
    labels = []
    for tick in ticks:
        if math.isclose(tick, cap, abs_tol=1e-6):
            labels.append(label)
        else:
            labels.append(f"{tick:g}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _should_log_scale(values: Sequence[float]) -> bool:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return False
    min_val = min(vals)
    max_val = max(vals)
    if min_val <= 0:
        return False
    return True


def _oom_confusion_counts(
    comparisons: List[ComparisonResult],
    threshold_gb: float,
) -> Dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unknown": 0}
    for comp in comparisons:
        peak_gb = comp.rapid_result.peak_gb
        if peak_gb is None or math.isnan(peak_gb):
            counts["unknown"] += 1
            continue
        pred_oom = float(peak_gb) > float(threshold_gb)
        actual_oom = comp.row.status == "OOM"
        if pred_oom and actual_oom:
            counts["tp"] += 1
        elif pred_oom and not actual_oom:
            counts["fp"] += 1
        elif (not pred_oom) and actual_oom:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def _generate_oom_confusion_plot(
    comparisons: List[ComparisonResult],
    output_dir: Path,
    threshold_gb: float,
    title_suffix: Optional[str] = None,
) -> Optional[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    counts = _oom_confusion_counts(comparisons, threshold_gb)
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    if total == 0:
        return None

    labels = [
        "TP (OOM)",
        "FP (pred OOM)",
        "FN (missed OOM)",
        "TN (correct non-OOM)",
    ]
    values = [counts["tp"], counts["fp"], counts["fn"], counts["tn"]]
    colors = ["#2ca02c", "#d62728", "#d62728", "#1f77b4"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    title = f"OOM classification (pred > {threshold_gb:g} GB, n={total}"
    if counts["unknown"]:
        title += f", unknown={counts['unknown']}"
    title += ")"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(1, total * 0.01),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plot_path = output_dir / "oom_classification_counts.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def generate_plots(
    comparisons: List[ComparisonResult],
    mode: str,
    output_dir: Path,
    title_suffix: Optional[str] = None,
) -> List[Path]:
    """Generate error distribution plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    def _scatter_pred_vs_actual(
        actuals: List[float],
        preds: List[float],
        errors: List[float],
        title: str,
        filename: str,
        x_label: str,
        y_label: str,
    ) -> Optional[Path]:
        if not actuals or not preds:
            return None
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(actuals, preds, s=14, alpha=0.6, edgecolors="none")
        min_val = min(min(actuals), min(preds))
        max_val = max(max(actuals), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.0, label="x=y")
        if _should_log_scale(actuals + preds):
            ax.set_xscale("log")
            ax.set_yscale("log")
        mean_abs = float(np.mean(np.abs(errors))) if errors else float("nan")
        title_line = f"{title} (n={len(actuals)}, mean_abs={mean_abs:.2f}%)"
        if title_suffix:
            title_line += f" - {title_suffix}"
        ax.set_title(title_line)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        plt.tight_layout()
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        return plot_path

    if mode in ('mem_only', 'both'):
        mem_errors = [
            _memory_error_value(c) for c in comparisons
            if _memory_error_value(c) is not None and not math.isnan(_memory_error_value(c))
        ]
        if mem_errors:
            mem_plot_errors, mem_overflow = _cap_error_overflow(mem_errors, ERROR_PCT_CAP)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist(mem_plot_errors, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Error (%)')
            ax.set_ylabel('Count')
            title = f"Memory vs Mem {_memory_metric_label()} (n={len(mem_errors)})"
            if title_suffix:
                title += f" - {title_suffix}"
            ax.set_title(title)
            ax.axvline(x=0, color='r', linestyle='--')
            if mem_overflow:
                _label_overflow_xtick(ax, ERROR_PCT_CAP, f">{int(ERROR_PCT_CAP)}%")

            plt.tight_layout()
            mem_plot = output_dir / 'memory_error_distribution.png'
            fig.savefig(mem_plot, dpi=200)
            plt.close(fig)
            plot_paths.append(mem_plot)

            mem_actuals = []
            mem_preds = []
            mem_scatter_errors = []
            for comp in comparisons:
                pred = comp.rapid_result.peak_gb
                actual = _memory_actual_value(comp.row)
                if pred is None or actual is None:
                    continue
                if not math.isfinite(float(pred)) or not math.isfinite(float(actual)):
                    continue
                if float(actual) <= 0:
                    continue
                mem_actuals.append(float(actual))
                mem_preds.append(float(pred))
                mem_scatter_errors.append(_pct_error(float(pred), float(actual)))
            scatter_path = _scatter_pred_vs_actual(
                mem_actuals,
                mem_preds,
                mem_scatter_errors,
                f"Memory vs Mem {_memory_metric_label()}",
                "memory_scatter.png",
                f"Actual Mem {_memory_metric_label()} (GB)",
                "Predicted Peak (GB)",
            )
            if scatter_path is not None:
                plot_paths.append(scatter_path)

        oom_plot = _generate_oom_confusion_plot(
            comparisons,
            output_dir,
            OOM_PRED_THRESHOLD_GB,
            title_suffix=title_suffix,
        )
        if oom_plot is not None:
            plot_paths.append(oom_plot)

    if mode in ('perf_only', 'both'):
        tok_errors = [c.tok_s_error_pct for c in comparisons
                      if c.tok_s_error_pct is not None and not math.isnan(c.tok_s_error_pct)]
        if tok_errors:
            tok_plot_errors, tok_overflow = _cap_error_overflow(tok_errors, ERROR_PCT_CAP)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist(tok_plot_errors, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Error (%)')
            ax.set_ylabel('Count')
            title = f"tok/s/gpu Error (n={len(tok_errors)})"
            if title_suffix:
                title += f" - {title_suffix}"
            ax.set_title(title)
            ax.axvline(x=0, color='r', linestyle='--')
            if tok_overflow:
                _label_overflow_xtick(ax, ERROR_PCT_CAP, f">{int(ERROR_PCT_CAP)}%")

            plt.tight_layout()
            perf_plot = output_dir / 'performance_error_distribution.png'
            fig.savefig(perf_plot, dpi=200)
            plt.close(fig)
            plot_paths.append(perf_plot)

            tok_actuals = []
            tok_preds = []
            tok_scatter_errors = []
            for comp in comparisons:
                if comp.tok_s_error_pct is None:
                    continue
                pred = comp.rapid_result.tok_s_gpu
                actual = comp.row.tok_s_gpu
                if pred is None or actual is None:
                    continue
                if not math.isfinite(float(pred)) or not math.isfinite(float(actual)):
                    continue
                if float(actual) <= 0:
                    continue
                tok_actuals.append(float(actual))
                tok_preds.append(float(pred))
                tok_scatter_errors.append(_pct_error(float(pred), float(actual)))
            scatter_path = _scatter_pred_vs_actual(
                tok_actuals,
                tok_preds,
                tok_scatter_errors,
                "tok/s/gpu",
                "performance_scatter.png",
                "Actual tok/s/gpu",
                "Predicted tok/s/gpu",
            )
            if scatter_path is not None:
                plot_paths.append(scatter_path)

    return plot_paths


def _get_device_capacity_gb(hw_config_path: Path) -> float:
    with open(hw_config_path, "r") as f:
        hw_config_dict = yaml.safe_load(f)
    from config import convert
    convert(hw_config_dict)
    tech_cfg = hw_config_dict.get("tech_param") or hw_config_dict.get("tech_config") or {}
    dram_cfg = tech_cfg.get("DRAM") or tech_cfg.get("dram") or {}
    size_bytes = dram_cfg.get("size")
    if size_bytes is None:
        raise ValueError(f"Missing DRAM size in hardware config: {hw_config_path}")
    return float(size_bytes) / float(1024 ** 3)


def generate_oom_plots(
    comparisons: List[ComparisonResult],
    output_dir: Path,
    hw_config_path: Path,
) -> List[Path]:
    """Generate OOM-only memory plots (delta histogram + waterfall)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    max_plot_gb = 200.0
    oom_peaks = [
        c.rapid_result.peak_gb
        for c in comparisons
        if c.row.status == "OOM"
        and c.rapid_result.peak_gb is not None
    ]
    if not oom_peaks:
        return []

    capacity_gb = _get_device_capacity_gb(hw_config_path)
    peaks_raw = np.array(oom_peaks, dtype=float)
    peaks = np.minimum(peaks_raw, max_plot_gb)
    delta = peaks - capacity_gb
    total = len(delta)
    misclassified = int(np.sum(delta < 0))
    clipped = int(np.sum(peaks_raw > max_plot_gb))

    plot_paths: List[Path] = []

    # Delta-to-capacity histogram.
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(delta, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.0, color='black', linestyle='--', label='Capacity boundary')
    ax.axvspan(delta.min(), 0.0, color='red', alpha=0.12, label='False negatives')
    ax.set_xlabel('Predicted peak - device capacity (GB)')
    ax.set_ylabel('Count')
    ax.set_title(
        f"OOM delta-to-capacity (cap {int(max_plot_gb)} GB, "
        f"n={total}, clipped={clipped}, below cap={misclassified})"
    )
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    delta_plot = output_dir / "oom_mem_plot.png"
    fig.savefig(delta_plot, dpi=200)
    plt.close(fig)
    plot_paths.append(delta_plot)

    # Waterfall of predicted peaks.
    sorted_peaks = np.sort(peaks)[::-1]
    x = np.arange(len(sorted_peaks))
    colors = np.where(sorted_peaks >= capacity_gb, "#7f7f7f", "#d62728")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x, sorted_peaks, color=colors, width=1.0)
    ax.axhline(y=capacity_gb, color='black', linestyle='--', label='Device capacity')
    ax.set_ylabel('Predicted peak (GB)')
    ax.set_xlabel('OOM runs (sorted)')
    ax.set_title(
        f"OOM predicted peak waterfall (cap {int(max_plot_gb)} GB, n={total}, clipped={clipped})"
    )
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-1, max(1, len(sorted_peaks)))
    plt.tight_layout()
    waterfall_plot = output_dir / "oom_mem_waterfall.png"
    fig.savefig(waterfall_plot, dpi=200)
    plt.close(fig)
    plot_paths.append(waterfall_plot)

    return plot_paths


def generate_category_outputs(
    comparisons: List[ComparisonResult],
    mode: str,
    output_dir: Path,
    run_spec: Optional[Sequence[str] | str],
    min_rows: int,
    enable_plots: bool = True,
) -> Tuple[Dict[str, List[Path]], Optional[Path]]:
    """Generate per-category plots and a summary CSV."""
    categories, selection_reasons = _select_categories_with_reasons(
        comparisons,
        mode,
        run_spec,
        min_rows,
    )
    if not categories:
        return {}, None

    if selection_reasons and EMIT_LOGS:
        print("[category] Auto-selected categories:")
        for category in categories:
            reasons = ", ".join(sorted(set(selection_reasons.get(category.name, []))))
            if reasons:
                print(f"  - {category.name}: {reasons}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_map: Dict[str, List[Path]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for category in categories:
        subset = [c for c in comparisons if category.predicate(c)]
        if len(subset) < max(0, min_rows):
            if EMIT_LOGS:
                print(f"[category] Skip '{category.name}' (n={len(subset)} < {min_rows})")
            continue

        cat_dir = output_dir / _sanitize_category_name(category.name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        stats = compute_aggregate_stats(subset, mode)
        success_count = sum(1 for c in subset if c.row.status == "Success")
        oom_count = sum(1 for c in subset if c.row.status == "OOM")
        rapid_fail_count = sum(1 for c in subset if not c.rapid_result.success)
        selected_for = ",".join(sorted(set(selection_reasons.get(category.name, []))))

        # Emit per-category detailed CSV for offline analysis.
        export_results_csv(subset, cat_dir / "validation_results.csv")
        export_results_short_csv(subset, cat_dir / "val_results_short.csv")

        # Emit human-friendly summary for quick inspection.
        summary_lines = [
            f"Category: {category.name}",
            f"Rows: {len(subset)} (Success: {success_count}, OOM: {oom_count}, Rapid failures: {rapid_fail_count})",
            f"tok/s time denom: {TOKS_PER_GPU_DENOM}",
            "",
            "Metric summary (error %):",
        ]
        metric_order = (_memory_metric_key(), "tok_s")
        for metric in metric_order:
            stat = stats.get(metric)
            if not stat:
                summary_lines.append(f"- {metric}: n/a")
                continue
            summary_lines.append(
                f"- {metric}: n={stat.count}, mean={stat.mean_error:.2f}, "
                f"median={stat.median_error:.2f}, mean_abs={stat.mean_abs_error:.2f}, "
                f"p90_abs={stat.p90_error:.2f}, p95_abs={stat.p95_error:.2f}"
            )
        summary_path = cat_dir / "summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        for key, stat in stats.items():
            summary_rows.append(
                {
                    "category": category.name,
                    "selected_for": selected_for,
                    "metric": key,
                    "category_rows": len(subset),
                    "success_rows": success_count,
                    "oom_rows": oom_count,
                    "rapid_fail_rows": rapid_fail_count,
                    "count": stat.count,
                    "mean_error": stat.mean_error,
                    "median_error": stat.median_error,
                    "std_error": stat.std_error,
                    "mean_abs_error": stat.mean_abs_error,
                    "p90_abs_error": stat.p90_error,
                    "p95_abs_error": stat.p95_error,
                }
            )

        if enable_plots:
            plot_map[category.name] = generate_plots(
                subset,
                mode,
                cat_dir,
                title_suffix=category.name,
            )

    summary_path: Optional[Path] = None
    if summary_rows:
        summary_path = output_dir / "category_stats.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if EMIT_LOGS:
            print(f"[category] Saved summary: {summary_path}")

    return plot_map, summary_path


def export_results_csv(
    comparisons: List[ComparisonResult],
    output_path: Path,
):
    """Export detailed results to CSV for further analysis."""
    rows = []
    for c in comparisons:
        attention_type = 'gqa' if c.row.num_kv_heads < c.row.num_heads else 'mha'
        intra_bw_vals = [c.row.ar_intra, c.row.ag_intra, c.row.rs_intra]
        inter_bw_vals = [c.row.ar_inter, c.row.ag_inter, c.row.rs_inter]
        intra_bw_avg = sum(v for v in intra_bw_vals if v > 0) / max(1, sum(1 for v in intra_bw_vals if v > 0)) if any(v > 0 for v in intra_bw_vals) else None
        inter_bw_avg = sum(v for v in inter_bw_vals if v > 0) / max(1, sum(1 for v in inter_bw_vals if v > 0)) if any(v > 0 for v in inter_bw_vals) else None
        row_dict = {
            'row_index': c.row.row_index,
            'label': c.row.label,
            'dp': c.row.dp,
            'pp': c.row.pp,
            'tp': c.row.tp,
            'zero_stage': c.row.zero_stage,
            'gbs': c.row.gbs,
            'mbs': c.row.mbs,
            'batch_accum': c.row.batch_accum,
            'seq_len': c.row.seq_len,
            'hidden_size': c.row.hidden_size,
            'num_layers': c.row.num_layers,
            'num_heads': c.row.num_heads,
            'num_kv_heads': c.row.num_kv_heads,
            'vocab_size': c.row.vocab_size,
            'attention_type': attention_type,
            'use_flashattention': USE_FLASH_ATTENTION,
            'attention_tile_size': ATTENTION_TILE_SIZE,
            'num_gpus': c.row.num_gpus,
            'dtype': c.row.dtype,
            'status': c.row.status,
            'oom_error': c.row.oom_error,
            'after_pp_fix': c.row.after_pp_fix,
            'rapid_success': c.rapid_result.success,
            'rapid_error': c.rapid_result.error,
            'predicted_peak_gb': c.rapid_result.peak_gb,
            'actual_mem_alloc_gb': c.row.mem_alloc_gb,
            'mem_alloc_error_pct': c.mem_alloc_error_pct,
            'actual_mem_res_gb': c.row.mem_res_gb,
            'mem_res_error_pct': c.mem_res_error_pct,
            'actual_mem_target_gb': _memory_actual_value(c.row),
            'mem_error_pct': _memory_error_value(c),
            'predicted_tok_s_gpu': c.rapid_result.tok_s_gpu,
            'actual_tok_s_gpu': c.row.tok_s_gpu,
            'tok_s_error_pct': c.tok_s_error_pct,
            'ar_inter_gbps': c.row.ar_inter,
            'ag_inter_gbps': c.row.ag_inter,
            'rs_inter_gbps': c.row.rs_inter,
            'ar_intra_gbps': c.row.ar_intra,
            'ag_intra_gbps': c.row.ag_intra,
            'rs_intra_gbps': c.row.rs_intra,
            'inter_bw_avg_gbps': inter_bw_avg,
            'intra_bw_avg_gbps': intra_bw_avg,
        }
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def _format_config_label(row: BenchmarkRow) -> str:
    return (
        f"dp{row.dp}_tp{row.tp}_pp{row.pp}_mbs{row.mbs}_acc{row.batch_accum}"
        f"_gbs{row.gbs}_seq{row.seq_len}_h{row.hidden_size}_L{row.num_layers}_zero{row.zero_stage}"
    )


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_yaml_with_header(path: Path, header_lines: Sequence[str], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        if header_lines:
            handle.write("\n")
        yaml.safe_dump(payload, handle, sort_keys=False)


def export_results_short_csv(
    comparisons: List[ComparisonResult],
    output_path: Path,
) -> None:
    """Export a compact results-only CSV."""
    def _fmt_pct(value: Optional[float]) -> Optional[float]:
        if value is None or math.isnan(value):
            return None
        return round(float(value), 1)

    rows = []
    for c in comparisons:
        rows.append(
            {
                "label": c.row.label,
                "actual_mem_target_gb": _memory_actual_value(c.row),
                "predicted_mem_gb": c.rapid_result.peak_gb,
                "mem_error_pct": _fmt_pct(_memory_error_value(c)),
                "actual_tok_s_gpu": c.row.tok_s_gpu,
                "predicted_tok_s_gpu": c.rapid_result.tok_s_gpu,
                "tok_s_error_pct": _fmt_pct(c.tok_s_error_pct),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# ==============================================================================
# CACHE AND CHUNKING HELPERS
# ==============================================================================

def _get_file_signature(path: Path) -> Dict[str, Any]:
    """Build a lightweight signature for cache invalidation."""
    try:
        stat = path.stat()
        return {
            'path': str(path.resolve()),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
        }
    except FileNotFoundError:
        return {
            'path': str(path),
            'mtime': None,
            'size': None,
        }


def _build_cache_key(
    csv_path: Path,
    hw_config_path: Path,
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
) -> str:
    """Create a stable cache key from config and input signatures."""
    payload = {
        'version': CACHE_KEY_VERSION,
        'csv': _get_file_signature(csv_path),
        'hw': _get_file_signature(hw_config_path),
        'mode': mode,
        'flash_attention': flash_attention,
        'attention_tile_size': attention_tile_size,
        'astra_cache_mode': ASTRA_CACHE_MODE,
        'microbatch_count_derived': USE_DERIVED_MICROBATCH_COUNT,
        'mb_graph_cap': MB_GRAPH_CAP,
        'pipeline_mem_schedule': PIPELINE_MEM_SCHEDULE,
        'pipeline_activation_window_mult': PIPELINE_ACTIVATION_WINDOW_MULT,
        'fast_mode': FAST_MODE,
        'fast_mode_timeout_s': FAST_MODE_TIMEOUT_S,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _load_cached_results(
    cache_path: Path,
    cache_key: str,
    emit_logs: bool,
) -> Dict[int, Dict[str, Any]]:
    """Load cached per-row results keyed by row_index."""
    cached: Dict[int, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cached
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('cache_key') != cache_key:
                    continue
                row_index = record.get('row_index')
                if row_index is None:
                    continue
                cached[int(row_index)] = record
    except OSError as e:
        if emit_logs:
            print(f"Warning: failed to read cache {cache_path}: {e}")
    return cached


class _CacheWriter:
    """Append-only JSONL cache writer."""

    def __init__(self, cache_path: Path, cache_key: str):
        self._cache_path = cache_path
        self._cache_key = cache_key
        self._fh = None

    def __enter__(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._cache_path, 'a', encoding='utf-8')
        return self

    def write(self, result_data: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        record = dict(result_data)
        record['cache_key'] = self._cache_key
        self._fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    def flush(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def __exit__(self, exc_type, exc, tb):
        try:
            self.flush()
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def _chunk_rows(rows: List[BenchmarkRow], chunk_size: int) -> List[List[BenchmarkRow]]:
    """Split rows into fixed-size chunks."""
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def _hard_cleanup_children() -> None:
    """Aggressively terminate any lingering child processes and run GC."""
    children = multiprocessing.active_children()
    for proc in children:
        proc.terminate()
    if children:
        import time
        time.sleep(0.1)
    for proc in multiprocessing.active_children():
        if proc.is_alive():
            try:
                proc.kill()
            except AttributeError:
                proc.terminate()
    for proc in multiprocessing.active_children():
        proc.join(timeout=1)
    gc.collect()


def _result_data_to_comparison(
    result_data: Dict[str, Any],
    row: BenchmarkRow,
    mode: str,
) -> ComparisonResult:
    """Convert worker result dict to ComparisonResult."""
    rapid_result = RAPIDResult(
        success=result_data.get('success', False),
        error=result_data.get('error'),
        peak_gb=result_data.get('peak_gb'),
        capacity_exceeded=result_data.get('capacity_exceeded', False),
        training_time_s=result_data.get('training_time_s'),
        tok_s_gpu=result_data.get('tok_s_gpu'),
        mfu=result_data.get('mfu'),
    )
    return compute_comparison(row, rapid_result, mode)


def _cleanup_processes(
    pending_procs: Dict[int, Tuple[multiprocessing.Process, BenchmarkRow]],
    result_queue: Optional[multiprocessing.Queue],
) -> None:
    """Terminate and join all pending processes, then close the queue."""
    for row_id, (proc, _) in list(pending_procs.items()):
        if proc.is_alive():
            proc.terminate()
    if pending_procs:
        import time
        time.sleep(0.1)
    for row_id, (proc, _) in list(pending_procs.items()):
        if proc.is_alive():
            try:
                proc.kill()
            except AttributeError:
                proc.terminate()
        proc.join(timeout=1)
    pending_procs.clear()
    if result_queue is not None:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass


def _run_rows_parallel(
    rows: List[BenchmarkRow],
    hw_config_path: Path,
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
    num_workers: int,
    progress: Optional[Any],
    result_handler,
) -> bool:
    """Run rows in parallel processes and stream results via callback."""
    if not rows:
        return False

    import time

    ctx = multiprocessing.get_context('spawn')
    result_queue: multiprocessing.Queue = ctx.Queue()
    pending: Dict[int, Tuple[multiprocessing.Process, BenchmarkRow]] = {}
    completed: set = set()
    start_times: Dict[int, float] = {}

    row_lookup: Dict[int, BenchmarkRow] = {r.row_index: r for r in rows}
    row_dicts = [asdict(r) for r in rows]
    row_dict_iter = iter(row_dicts)

    worker_count = min(num_workers, len(rows), os.cpu_count() or 1)

    def _start_next() -> bool:
        try:
            row_dict = next(row_dict_iter)
        except StopIteration:
            return False
        row = row_lookup[row_dict['row_index']]
        proc = ctx.Process(
            target=_worker_process_row,
            args=(row_dict, str(hw_config_path), mode, flash_attention, attention_tile_size, result_queue),
        )
        proc.start()
        pending[row.row_index] = (proc, row)
        start_times[row.row_index] = time.monotonic()
        return True

    def _handle_result(result_data: Dict[str, Any], row: BenchmarkRow) -> None:
        if row.row_index in completed:
            return
        if result_handler is not None:
            result_handler(result_data, row)
        completed.add(row.row_index)
        if progress is not None:
            progress.update(1)

    interrupted = False

    try:
        for _ in range(worker_count):
            if not _start_next():
                break

        while pending or len(completed) < len(rows):
            completed_ids = []
            for row_id, (proc, row) in list(pending.items()):
                if FAST_MODE and proc.is_alive():
                    elapsed = time.monotonic() - start_times.get(row_id, time.monotonic())
                    if elapsed > FAST_MODE_TIMEOUT_S:
                        proc.terminate()
                        time.sleep(0.05)
                        if proc.is_alive():
                            try:
                                proc.kill()
                            except AttributeError:
                                proc.terminate()
                        proc.join(timeout=0.2)
                        completed_ids.append(row_id)
                        fail_result = {
                            'row_index': row_id,
                            'success': False,
                            'error': f"Timed out after {FAST_MODE_TIMEOUT_S:.0f}s",
                            'peak_gb': None,
                            'capacity_exceeded': False,
                            'training_time_s': None,
                            'tok_s_gpu': None,
                            'mfu': None,
                        }
                        _handle_result(fail_result, row)
                        continue
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    completed_ids.append(row_id)
                    if proc.exitcode != 0:
                        fail_result = {
                            'row_index': row_id,
                            'success': False,
                            'error': f"Process crashed with exit code {proc.exitcode}",
                            'peak_gb': None,
                            'capacity_exceeded': False,
                            'training_time_s': None,
                            'tok_s_gpu': None,
                            'mfu': None,
                        }
                        _handle_result(fail_result, row)

            for row_id in completed_ids:
                pending.pop(row_id, None)
                start_times.pop(row_id, None)

            while True:
                try:
                    result_data = result_queue.get_nowait()
                except queue.Empty:
                    break
                row_id = result_data.get('row_index')
                row = row_lookup.get(row_id)
                if row is None:
                    continue
                _handle_result(result_data, row)

            while len(pending) < worker_count:
                if not _start_next():
                    break

            if pending:
                time.sleep(0.05)

    except KeyboardInterrupt:
        interrupted = True
        _cleanup_processes(pending, result_queue)
    finally:
        if not interrupted:
            _cleanup_processes(pending, result_queue)
    return interrupted


def _run_rows_parallel_inmemory(
    rows: List[BenchmarkRow],
    base_hw_config: Dict[str, Any],
    mode: str,
    flash_attention: bool,
    attention_tile_size: Optional[int],
    num_workers: int,
    progress: Optional[Any],
    result_handler,
    fast_mode: bool,
    timeout_s: Optional[float],
) -> bool:
    """Run rows in parallel with per-row timeout using in-memory config."""
    if not rows:
        return False

    import time

    ctx = multiprocessing.get_context('spawn')
    result_queue: multiprocessing.Queue = ctx.Queue()
    pending: Dict[int, Tuple[multiprocessing.Process, BenchmarkRow]] = {}
    completed: set = set()
    start_times: Dict[int, float] = {}

    row_lookup: Dict[int, BenchmarkRow] = {r.row_index: r for r in rows}
    row_dicts = [asdict(r) for r in rows]
    row_dict_iter = iter(row_dicts)

    worker_count = min(num_workers, len(rows), os.cpu_count() or 1)
    timeout_val = float(timeout_s) if timeout_s is not None else FAST_MODE_TIMEOUT_S

    def _start_next() -> bool:
        try:
            row_dict = next(row_dict_iter)
        except StopIteration:
            return False
        row = row_lookup[row_dict['row_index']]
        proc = ctx.Process(
            target=_worker_process_row_inmemory,
            args=(
                row_dict,
                base_hw_config,
                mode,
                flash_attention,
                attention_tile_size,
                result_queue,
            ),
        )
        proc.start()
        pending[row.row_index] = (proc, row)
        start_times[row.row_index] = time.monotonic()
        return True

    def _handle_result(result_data: Dict[str, Any], row: BenchmarkRow) -> None:
        if row.row_index in completed:
            return
        if result_handler is not None:
            result_handler(result_data, row)
        completed.add(row.row_index)
        if progress is not None:
            progress.update(1)

    interrupted = False

    try:
        for _ in range(worker_count):
            if not _start_next():
                break

        while pending or len(completed) < len(rows):
            completed_ids = []
            for row_id, (proc, row) in list(pending.items()):
                if fast_mode and proc.is_alive():
                    elapsed = time.monotonic() - start_times.get(row_id, time.monotonic())
                    if timeout_val > 0 and elapsed > timeout_val:
                        proc.terminate()
                        time.sleep(0.05)
                        if proc.is_alive():
                            try:
                                proc.kill()
                            except AttributeError:
                                proc.terminate()
                        proc.join(timeout=0.2)
                        completed_ids.append(row_id)
                        fail_result = {
                            'row_index': row_id,
                            'success': False,
                            'error': f"Timed out after {timeout_val:.0f}s",
                            'peak_gb': None,
                            'capacity_exceeded': False,
                            'training_time_s': None,
                            'tok_s_gpu': None,
                            'mfu': None,
                        }
                        _handle_result(fail_result, row)
                        continue
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    completed_ids.append(row_id)
                    if proc.exitcode != 0:
                        fail_result = {
                            'row_index': row_id,
                            'success': False,
                            'error': f"Process crashed with exit code {proc.exitcode}",
                            'peak_gb': None,
                            'capacity_exceeded': False,
                            'training_time_s': None,
                            'tok_s_gpu': None,
                            'mfu': None,
                        }
                        _handle_result(fail_result, row)

            for row_id in completed_ids:
                pending.pop(row_id, None)
                start_times.pop(row_id, None)

            while True:
                try:
                    result_data = result_queue.get_nowait()
                except queue.Empty:
                    break
                row_id = result_data.get('row_index')
                row = row_lookup.get(row_id)
                if row is None:
                    continue
                _handle_result(result_data, row)

            while len(pending) < worker_count:
                if not _start_next():
                    break

            if pending:
                time.sleep(0.05)

    except KeyboardInterrupt:
        interrupted = True
        _cleanup_processes(pending, result_queue)
    finally:
        if not interrupted:
            _cleanup_processes(pending, result_queue)
    return interrupted

# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def run(
    csv_path: Path = CSV_PATH,
    hw_config_path: Path = HW_CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    mode: str = MODE,
    num_workers: int = NUM_WORKERS,
    assume_bf16: bool = ASSUME_BF16,
    filter_pp_fix: bool = FILTER_PP_FIX,
    max_rows: Optional[int] = MAX_ROWS,
    enable_plots: bool = ENABLE_PLOTS,
    enable_global_plots: bool = ENABLE_GLOBAL_PLOTS,
    enable_category_plots: bool = ENABLE_CATEGORY_PLOTS,
    category_run: Optional[Sequence[str] | str] = CATEGORY_RUN,
    category_min_rows: int = CATEGORY_MIN_ROWS,
    category_output_dir: Path = CATEGORY_OUTPUT_DIR,
    emit_logs: bool = EMIT_LOGS,
    chunk_size: Optional[int] = CHUNK_SIZE,
    enable_cache: bool = ENABLE_RESULT_CACHE,
    cache_path: Optional[Path] = None,
    rebuild_from_cache_only: bool = REBUILD_FROM_CACHE_ONLY,
) -> Dict[str, Any]:
    """
    Run HuggingFace benchmark validation.

    Returns dict with:
        - stats: aggregate statistics
        - comparisons: list of ComparisonResult
        - plots: list of plot paths
        - results_csv: path to results CSV
    """
    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = CACHE_PATH if output_dir == OUTPUT_DIR else (output_dir / "validation_cache.jsonl")
    if rebuild_from_cache_only and not enable_cache:
        enable_cache = True

    if SELECTIVE_RUN:
        output_dir = SELECTIVE_RUN_OUTPUT_DIR
        enable_cache = False
        rebuild_from_cache_only = False
        if category_output_dir == CATEGORY_OUTPUT_DIR:
            category_output_dir = output_dir / "categories"

    # Load CSV data
    if emit_logs:
        print(f"Loading CSV: {csv_path}")
    df = load_csv(csv_path)
    rows = [parse_row(row, idx) for idx, (_, row) in enumerate(df.iterrows())]
    if emit_logs:
        print(f"Loaded {len(rows)} total rows")

    # Randomize order before filtering (e.g., when MAX_ROWS is set).
    if not SELECTIVE_RUN:
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(rows)

    if SELECTIVE_RUN:
        ids = [str(entry).strip() for entry in SELECTIVE_RUN_IDS if str(entry).strip()]
        if not ids:
            raise ValueError("SELECTIVE_RUN is enabled but SELECTIVE_RUN_IDS is empty.")
        selected = [row for row in rows if any(_matches_selective_id(row, token) for token in ids)]
        missing = [token for token in ids if not any(_matches_selective_id(row, token) for row in rows)]
        if emit_logs:
            print(f"[selective] IDs requested: {len(ids)}, matched rows: {len(selected)}")
            if missing:
                print(f"[selective] WARNING: no match for IDs: {missing}")
        rows = selected

    # Filter rows
    filtered_rows = filter_rows(rows, mode, assume_bf16, filter_pp_fix, max_rows)
    filtered_rows = _restrict_rows_by_categories(filtered_rows, category_run, emit_logs)
    if emit_logs:
        print(f"After filtering: {len(filtered_rows)} rows")
        success_count = sum(1 for r in filtered_rows if r.status == 'Success')
        oom_count = sum(1 for r in filtered_rows if r.status == 'OOM')
        print(f"  Success: {success_count}, OOM: {oom_count}")
    if DUMP_FIRST_ROW_HW_CONFIG:
        if not filtered_rows:
            print("No rows after filtering; skipping hardware config dump.")
            return {}
        base_hw_config = load_base_hw_config(hw_config_path)
        hw_config_dict = build_hw_config_dict(base_hw_config, filtered_rows[0])
        dump_path = output_dir / DUMP_FIRST_ROW_HW_CONFIG_NAME
        with open(dump_path, "w") as f:
            yaml.safe_dump(hw_config_dict, f, sort_keys=False)
        print(f"Saved first-row hardware config to: {dump_path}")
        return {}

    # Workers load hw config from disk to minimize fork memory
    comparisons: List[ComparisonResult] = []
    completed_row_ids: set = set()

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(filtered_rows), desc="Validating")
    except ImportError:
        progress = None

    # Set global mode for workers (still used by module-level code paths)
    global MODE, USE_FLASH_ATTENTION, ATTENTION_TILE_SIZE
    MODE = mode
    row_lookup: Dict[int, BenchmarkRow] = {r.row_index: r for r in filtered_rows}

    cache_key: Optional[str] = None
    cached_results: Dict[int, Dict[str, Any]] = {}
    if enable_cache:
        cache_key = _build_cache_key(
            csv_path=csv_path,
            hw_config_path=hw_config_path,
            mode=mode,
            flash_attention=USE_FLASH_ATTENTION,
            attention_tile_size=ATTENTION_TILE_SIZE,
        )
        cached_results = _load_cached_results(cache_path, cache_key, emit_logs)
        cached_results = {row_id: data for row_id, data in cached_results.items() if row_id in row_lookup}
        if emit_logs and cached_results:
            print(f"Loaded {len(cached_results)} cached rows from {cache_path}")
        for row_id, result_data in cached_results.items():
            row = row_lookup[row_id]
            comparisons.append(_result_data_to_comparison(result_data, row, mode))
            completed_row_ids.add(row_id)

    if progress is not None and completed_row_ids:
        progress.update(len(completed_row_ids))

    rows_to_process = [r for r in filtered_rows if r.row_index not in completed_row_ids]
    if emit_logs:
        print(f"Configured workers: {num_workers}")
        if enable_cache:
            print(f"Cache file: {cache_path}")
        if chunk_size and chunk_size > 0:
            print(f"Chunk size: {chunk_size}")
        if completed_row_ids:
            print(f"Skipping {len(completed_row_ids)} cached rows; {len(rows_to_process)} remaining.")
        if rebuild_from_cache_only:
            print("Rebuild-from-cache-only enabled: skipping RAPID execution.")

    if rebuild_from_cache_only:
        if emit_logs and rows_to_process:
            print(f"Cache missing {len(rows_to_process)} rows; exporting cached subset only.")
        rows_to_process = []
        if progress is not None:
            progress.close()
            progress = None

    interrupted = False
    if rows_to_process:
        effective_chunk_size = chunk_size if chunk_size and chunk_size > 0 else len(rows_to_process)
        chunks = _chunk_rows(rows_to_process, effective_chunk_size)
        if emit_logs:
            print(f"Processing {len(rows_to_process)} rows in {len(chunks)} chunks.")

        for chunk_idx, chunk_rows in enumerate(chunks, start=1):
            if emit_logs:
                chunk_workers = min(num_workers, len(chunk_rows), os.cpu_count() or 1)
                print(f"Chunk {chunk_idx}/{len(chunks)}: {len(chunk_rows)} rows with {chunk_workers} workers")

            cache_ctx = _CacheWriter(cache_path, cache_key) if enable_cache else nullcontext()
            with cache_ctx as cache_writer:
                def _handle_result(result_data: Dict[str, Any], row: BenchmarkRow) -> None:
                    if row.row_index in completed_row_ids:
                        return
                    if cache_writer is not None:
                        cache_writer.write(result_data)
                    comparisons.append(_result_data_to_comparison(result_data, row, mode))
                    completed_row_ids.add(row.row_index)

                interrupted = _run_rows_parallel(
                    chunk_rows,
                    hw_config_path,
                    mode,
                    USE_FLASH_ATTENTION,
                    ATTENTION_TILE_SIZE,
                    num_workers,
                    progress,
                    _handle_result,
                )

            if HARD_CLEANUP_BETWEEN_CHUNKS:
                _hard_cleanup_children()
            if interrupted:
                if emit_logs:
                    print("\nInterrupted! Stopping remaining chunks...")
                break

    if progress is not None:
        progress.close()

    # Sort by original order
    comparisons.sort(key=lambda c: c.row.row_index)

    # Compute aggregate statistics
    stats = compute_aggregate_stats(comparisons, mode)

    if emit_logs:
        print("\n=== Aggregate Statistics ===")
        for key, stat in stats.items():
            print(f"\n{stat.metric_name} (n={stat.count}):")
            print(f"  Mean Error: {stat.mean_error:.2f}%")
            print(f"  Median Error: {stat.median_error:.2f}%")
            print(f"  Std Dev: {stat.std_error:.2f}%")
            print(f"  Mean Abs Error: {stat.mean_abs_error:.2f}%")
            print(f"  P90 Abs Error: {stat.p90_error:.2f}%")
            print(f"  P95 Abs Error: {stat.p95_error:.2f}%")

    # Generate plots
    plot_paths: List[Path] = []
    oom_plot_paths: List[Path] = []
    if enable_plots and enable_global_plots:
        plot_paths = generate_plots(comparisons, mode, output_dir)
        oom_plot_paths = generate_oom_plots(comparisons, output_dir, hw_config_path)
        if emit_logs:
            for path in plot_paths:
                print(f"Saved plot: {path}")
            for path in oom_plot_paths:
                print(f"Saved OOM plot: {path}")

    category_plot_paths: Dict[str, List[Path]] = {}
    category_stats_path: Optional[Path] = None
    if enable_category_plots:
        category_plot_paths, category_stats_path = generate_category_outputs(
            comparisons,
            mode,
            category_output_dir,
            category_run,
            category_min_rows,
            enable_plots=enable_plots,
        )
        if emit_logs and category_plot_paths:
            print(f"[category] Generated {len(category_plot_paths)} category plot set(s)")

    # Export detailed results
    results_csv = output_dir / 'validation_results.csv'
    export_results_csv(comparisons, results_csv)
    if emit_logs:
        print(f"Saved results CSV: {results_csv}")

    short_results_csv = output_dir / 'val_results_short.csv'
    export_results_short_csv(comparisons, short_results_csv)
    if emit_logs:
        print(f"Saved short results CSV: {short_results_csv}")

    if SELECTIVE_RUN and SELECTIVE_DUMP_HW_CONFIGS:
        if comparisons:
            base_hw_config = load_base_hw_config(hw_config_path)
            dump_dir = output_dir / "hw_configs"
            dump_dir.mkdir(parents=True, exist_ok=True)
            model_dump_dir = output_dir / "model_configs"
            model_dump_dir.mkdir(parents=True, exist_ok=True)
            run_perf_path = PROJECT_ROOT / "run_perf.py"
            for comp in comparisons:
                hw_config_dict = build_hw_config_dict(base_hw_config, comp.row)
                model_config_dict = build_model_config_dict(
                    comp.row,
                    flash_attention=USE_FLASH_ATTENTION,
                    attention_tile_size=ATTENTION_TILE_SIZE,
                )
                label = _format_config_label(comp.row)
                safe_label = _sanitize_filename(label)
                dump_path = dump_dir / f"{comp.row.row_index}_{safe_label}.yaml"
                model_dump_path = model_dump_dir / f"{comp.row.row_index}_{safe_label}.yaml"
                cmd_lines = [
                    "Debug run (from any dir):",
                    (
                        "RAPID_PERSIST_ASTRASIM_ARTIFACTS=1 "
                        "RAPID_VISUALIZE_GRAPHS=1 "
                        "RAPID_PERSIST_ARTIFACT_VIZ=1 "
                        f"uv run \"{run_perf_path}\" --hardware_config \"{dump_path}\" --model_config \"{model_dump_path}\""
                    ),
                ]
                _write_yaml_with_header(dump_path, cmd_lines, hw_config_dict)
                with open(model_dump_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(model_config_dict, handle, sort_keys=False)
            if emit_logs:
                print(f"Saved {len(comparisons)} hardware configs to: {dump_dir}")
                print(f"Saved {len(comparisons)} model configs to: {model_dump_dir}")
        elif emit_logs:
            print("Selective HW config dump requested, but no comparisons were produced.")

    return {
        'stats': stats,
        'comparisons': comparisons,
        'plots': plot_paths,
        'oom_plots': oom_plot_paths,
        'category_plots': category_plot_paths,
        'category_stats_path': category_stats_path,
        'results_csv': results_csv,
    }


if __name__ == '__main__':
    # Use spawn context to avoid fork() memory issues on Linux
    multiprocessing.set_start_method('spawn', force=True)
    run()
