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
Fault sensitivity sweep for RAPID-LLM parallelism configurations.

Generates a representative set of parallelism tuples that map to a fixed GPU
budget, evaluates baseline runtimes, and then injects random soft faults across
network dimensions to gauge runtime variability.
"""

import argparse
import atexit
import datetime
import copy
import json
import math
import os
import random
import signal
import shlex
import sys
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

try:  # Optional styling dependency
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover
    sns = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

from parallelism_sweep import (  # noqa: E402
    MODEL_CONFIG_PATH,
    ASTRA_CACHE_MODE,
    build_parallelism_settings,
    determine_model_mode,
    evaluate_parallelism,
    make_temp_hw_config,
    read_yaml,
    set_astrasim_cache_mode,
    tp_cp_product_is_power_of_two_square,
)
import memory_estimation  # noqa: E402


class FlowSeq(list):
    """List subclass that forces YAML flow-style serialization."""


def _represent_flow_seq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.SafeDumper.add_representer(FlowSeq, _represent_flow_seq)


def _quantize_weight(value: float, decimals: int = 2) -> float:
    return round(float(value), decimals)

def _safe_int(value: object, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _model_run_type(model_config_obj: object) -> str:
    model_cfg = getattr(model_config_obj, "model_config", None)
    run_type = getattr(model_cfg, "run_type", "training") if model_cfg is not None else "training"
    return str(run_type).lower()


def _base_parallelism_defaults(base_hw_dict: Optional[Dict[str, object]]) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "mb": 1,
        "tp_sp": True,
        "dp": 1,
        "ep": 1,
        "tp_ep": True,
        "replica_count": 1,
        "moe_dp": 1,
    }
    if not isinstance(base_hw_dict, dict):
        return defaults
    parallelism = base_hw_dict.get("parallelism")
    if not isinstance(parallelism, dict):
        return defaults
    defaults["tp"] = _safe_int(parallelism.get("tp", defaults["tp"]), defaults["tp"])
    defaults["cp"] = _safe_int(parallelism.get("cp", defaults["cp"]), defaults["cp"])
    defaults["pp"] = _safe_int(parallelism.get("pp", defaults["pp"]), defaults["pp"])
    defaults["mb"] = _safe_int(parallelism.get("mb", defaults["mb"]), defaults["mb"])
    defaults["tp_sp"] = bool(parallelism.get("tp_sp", defaults["tp_sp"]))
    train_block = parallelism.get("train")
    if isinstance(train_block, dict):
        defaults["dp"] = _safe_int(train_block.get("dp", defaults["dp"]), defaults["dp"])
        defaults["ep"] = _safe_int(train_block.get("ep", defaults["ep"]), defaults["ep"])
        defaults["tp_ep"] = bool(train_block.get("tp_ep", defaults["tp_ep"]))
    inference_block = parallelism.get("inference")
    if isinstance(inference_block, dict):
        defaults["replica_count"] = _safe_int(
            inference_block.get("replica_count", defaults["replica_count"]),
            defaults["replica_count"],
        )
        defaults["moe_dp"] = _safe_int(inference_block.get("moe_dp", defaults["moe_dp"]), defaults["moe_dp"])
    return defaults


def _resolve_flat_parallelism(
    settings: Dict[str, object],
    base_hw_dict: Optional[Dict[str, object]],
    run_type: str,
) -> Dict[str, object]:
    flat = _base_parallelism_defaults(base_hw_dict)
    for key in ("tp", "cp", "pp", "mb"):
        if key in settings:
            flat[key] = _safe_int(settings[key], flat.get(key, 1))
    if "tp_sp" in settings:
        flat["tp_sp"] = bool(settings["tp_sp"])
    if "tp_ep" in settings:
        flat["tp_ep"] = bool(settings["tp_ep"])
    if "replica_count" in settings:
        flat["replica_count"] = _safe_int(settings["replica_count"], flat.get("replica_count", 1))

    if "mb" not in settings and "pp" in settings:
        flat["mb"] = 1 if int(flat.get("pp", 1) or 1) == 1 else int(flat.get("pp", 1) or 1)

    if run_type == "inference" and SWEEP_MOE_DP:
        moe_dp = settings.get("dp", settings.get("moe_dp", flat.get("moe_dp", 1)))
        flat["moe_dp"] = _safe_int(moe_dp, flat.get("moe_dp", 1))
        flat["ep"] = flat["moe_dp"]
        flat["dp"] = 1
    else:
        if "dp" in settings:
            flat["dp"] = _safe_int(settings["dp"], flat.get("dp", 1))
        if "ep" in settings:
            flat["ep"] = _safe_int(settings["ep"], flat.get("ep", 1))
        if "moe_dp" in settings:
            flat["moe_dp"] = _safe_int(settings["moe_dp"], flat.get("moe_dp", 1))
    return flat


def _normalize_parallelism_settings(
    base_hw_dict: Dict[str, object],
    settings: Dict[str, object],
    run_type: str,
) -> Dict[str, object]:
    if isinstance(settings.get("train"), dict) and isinstance(settings.get("inference"), dict):
        return settings
    flat = _resolve_flat_parallelism(settings, base_hw_dict, run_type)
    return build_parallelism_settings(flat)


def _axis_sizes_from_settings(
    settings: Dict[str, object],
    base_hw_dict: Optional[Dict[str, object]],
    run_type: str,
) -> Dict[str, int]:
    flat = _resolve_flat_parallelism(settings, base_hw_dict, run_type)
    return {
        "tp": _safe_int(flat.get("tp", 1), 1),
        "cp": _safe_int(flat.get("cp", 1), 1),
        "dp": _safe_int(flat.get("dp", 1), 1),
        "pp": _safe_int(flat.get("pp", 1), 1),
        "ep": _safe_int(flat.get("ep", 1), 1),
    }


def _dp_label() -> str:
    return "moe_dp" if RUN_TYPE == "inference" and SWEEP_MOE_DP else "dp"


def _format_parallelism_label(settings: Dict[str, object]) -> str:
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    pp = settings.get("pp")
    ep = settings.get("ep")
    dp_label = _dp_label()
    def _fmt(val: object) -> str:
        return str(int(val)) if isinstance(val, (int, float)) else str(val)
    parts = [f"tp{_fmt(tp)}"]
    if not MOE_MODE:
        parts.append(f"cp{_fmt(cp)}")
    if MOE_MODE:
        parts.append(f"ep{_fmt(ep if ep is not None else 1)}")
    elif RUN_TYPE == "training" and ep not in (None, 1, 1.0):
        parts.append(f"ep{_fmt(ep)}")
    if not MOE_MODE:
        parts.append(f"{dp_label}{_fmt(dp)}")
    parts.append(f"pp{_fmt(pp)}")
    return "-".join(parts)


def _format_parallelism_tag(settings: Dict[str, object]) -> str:
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    pp = settings.get("pp")
    ep = settings.get("ep")
    dp_label = "moe" if RUN_TYPE == "inference" and SWEEP_MOE_DP else "dp"
    def _fmt(val: object) -> str:
        return str(int(val)) if isinstance(val, (int, float)) else str(val)
    parts = [f"tp{_fmt(tp)}"]
    if not MOE_MODE:
        parts.append(f"cp{_fmt(cp)}")
    if MOE_MODE:
        parts.append(f"ep{_fmt(ep if ep is not None else 1)}")
    elif RUN_TYPE == "training" and ep not in (None, 1, 1.0):
        parts.append(f"ep{_fmt(ep)}")
    if not MOE_MODE:
        parts.append(f"{dp_label}{_fmt(dp)}")
    parts.append(f"pp{_fmt(pp)}")
    return "_".join(parts)


def _parallelism_identity(settings: Dict[str, object]) -> Tuple[int, int, int, int, int]:
    return (
        _safe_int(settings.get("tp", 1), 1),
        _safe_int(settings.get("cp", 1), 1),
        _safe_int(settings.get("dp", 1), 1),
        _safe_int(settings.get("pp", 1), 1),
        _safe_int(settings.get("ep", 1), 1),
    )


def _record_key(record: Dict[str, object]) -> Optional[Tuple[int, int, int, int, int]]:
    settings = record.get("parallelism")
    if isinstance(settings, dict):
        return _parallelism_identity(settings)
    return None


def _record_display_label(record: Dict[str, object]) -> str:
    settings = record.get("parallelism")
    if isinstance(settings, dict):
        return _format_parallelism_label(settings)
    return str(record.get("label", "") or "")


def _ensure_ep_in_network(network: Dict[str, object]) -> None:
    dimensions = network.get("dimensions")
    if not isinstance(dimensions, list):
        return
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        parallelisms = dim.get("parallelisms")
        if isinstance(parallelisms, list) and "ep" in parallelisms:
            return
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        parallelisms = dim.get("parallelisms")
        if not isinstance(parallelisms, list):
            continue
        if any(axis in ("tp", "cp") for axis in parallelisms):
            parallelisms.append("ep")
            return


def _apply_network_override(base_hw_dict: Dict[str, object], network_config_path: Optional[str]) -> None:
    if not network_config_path:
        return
    override_doc = read_yaml(network_config_path)
    if not isinstance(override_doc, dict):
        raise ValueError(f"Network override {network_config_path} did not parse to a mapping.")
    network_override = override_doc.get("network")
    if not isinstance(network_override, dict):
        raise ValueError(f"Network override {network_config_path} missing a 'network' section.")
    base_hw_dict["network"] = copy.deepcopy(network_override)


def _tag_output_path(base_path: str, tag: str) -> str:
    p = Path(base_path)
    safe_tag = str(tag).replace(os.sep, "_")
    return str(p.with_name(f"{p.stem}.{safe_tag}{p.suffix}"))


def _suffix_output_path(base_path: str, suffix: str) -> str:
    p = Path(base_path)
    safe_suffix = str(suffix).replace(os.sep, "_")
    stem = p.stem if p.suffix else p.name
    return str(p.with_name(f"{stem}_{safe_suffix}{p.suffix}"))

# -----------------------------------------------------------------------------
# Fault sweep configuration
# -----------------------------------------------------------------------------

GLM_MODE = True
GLM_TRAIN = True
MAX_ATTEMPTS = 1
PLOT_ONLY = False
PLOT_NUM = 0
PLOT_SKIP_TOP = 0
ONE_DIM = True  # 1_DIM: restrict faults to the first network dimension (tp/ep only).
globals()["1_DIM"] = ONE_DIM


def _one_dim_enabled() -> bool:
    return bool(globals().get("1_DIM", ONE_DIM))

if GLM_MODE:
    if GLM_TRAIN:
        HARDWARE_CONFIG_PATH = "validation_scripts/validation_configs/hardware-config/a100_80GB_fault.yaml"
        MODEL_CONFIG_PATH = "configs/model-config/GLM4.7_358B.yaml"
        NETWORK_CONFIG_PATH: Optional[str] = None
        # NETWORK_CONFIG_PATH: Optional[str] = "configs/hardware-config/a100_80GB.yaml"
        TARGET_NUM_GPUS = 60
        MIN_ALLOWED_TP = 1
        MAX_ALLOWED_TP = 29
        MIN_ALLOWED_CP = 1
        MAX_ALLOWED_CP = 1
        MIN_ALLOWED_DP = 1
        MAX_ALLOWED_DP = 1
        MIN_ALLOWED_PP = 1
        MAX_ALLOWED_PP = 16
        MIN_ALLOWED_EP = 1
        MAX_ALLOWED_EP = 29
    else:
        HARDWARE_CONFIG_PATH = "configs/hardware-config/H100_SXM5_80GB_moe.yaml"
        MODEL_CONFIG_PATH = "configs/model-config/GLM4.7_358B_inf.yaml"
        NETWORK_CONFIG_PATH: Optional[str] = "configs/hardware-config/a100_80GB.yaml"
        TARGET_NUM_GPUS = 64
        MIN_ALLOWED_TP = 1
        MAX_ALLOWED_TP = 32
        MIN_ALLOWED_CP = 1
        MAX_ALLOWED_CP = 1
        MIN_ALLOWED_DP = 1
        MAX_ALLOWED_DP = 32
        MIN_ALLOWED_PP = 1
        MAX_ALLOWED_PP = 4
        MIN_ALLOWED_EP = 1
        MAX_ALLOWED_EP = 1
else:
    HARDWARE_CONFIG_PATH = "validation_scripts/validation_configs/hardware-config/a100_80GB_fault.yaml"
    MODEL_CONFIG_PATH = "configs/model-config/Llama3.1-70B.yaml"
    NETWORK_CONFIG_PATH: Optional[str] = None
    TARGET_NUM_GPUS = 96
    MIN_ALLOWED_TP = 1
    MAX_ALLOWED_TP = 24
    MIN_ALLOWED_CP = 1
    MAX_ALLOWED_CP = 24
    MIN_ALLOWED_DP = 1
    MAX_ALLOWED_DP = 4
    MIN_ALLOWED_PP = 1
    MAX_ALLOWED_PP = 4
    MIN_ALLOWED_EP = 1
    MAX_ALLOWED_EP = 1

# When True, discard configurations whose tp*cp product is not a square power of two.
ENFORCE_SQUARE_TP_CP = False
MIN_TP_CP_PROD = 1
MAX_TP_CP_PROD = 99

RUN_TYPE = "training"
SWEEP_MOE_DP = False
MOE_MODE = False
BASE_HW_DICT: Optional[Dict[str, object]] = None

SAMPLE_COUNT = 25
FAULT_ITER = 300
FAULT_WORKERS = 110
FAULT_MAG = (0.5, 0.0)  # May also be a (mean, std) tuple
NUM_FAULTS = [1]
HARD_FAULT_ITER = 300
HARD_NUM_FAULTS = [1]

ALLOWED_FAULT_DIMS: Optional[List[int]] = None

OUTPUT_BASE_DIR = Path("outputs") / ("fault_glm" if GLM_MODE else "fault")
PLOT_OUTPUT_PATH = str(OUTPUT_BASE_DIR / "fault_sweep.png")
FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(PLOT_OUTPUT_PATH, "filtered")
REPORT_OUTPUT_PATH = str(OUTPUT_BASE_DIR / "fault_sweep.tsv")
HARD_PLOT_OUTPUT_PATH = str(OUTPUT_BASE_DIR / "fault_sweep_hard.png")
HARD_FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(HARD_PLOT_OUTPUT_PATH, "filtered")
HARD_REPORT_OUTPUT_PATH = str(OUTPUT_BASE_DIR / "fault_sweep_hard.tsv")
COMBINED_PLOT_OUTPUT_PATH = str(OUTPUT_BASE_DIR / "fault_sweep_combined.png")
COMBINED_FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(COMBINED_PLOT_OUTPUT_PATH, "filtered")
FAULT_OUTPUT_DIR = str(OUTPUT_BASE_DIR / "astra_cache_faults")
RESULTS_JSON_PATH = OUTPUT_BASE_DIR / "results.json"

RANDOM_SEED = 1337
RESULTS_WRITE_LOCK = threading.Lock()
_ACTIVE_EXECUTOR: Optional[ProcessPoolExecutor] = None
_OLD_SIGINT_HANDLER = None
_OLD_SIGTERM_HANDLER = None
_SHUTDOWN_REQUESTED = False

# Debug dumping controls
DEBUG_MODE = False
DEBUG_OUTPUT_BASE = Path("dbg_hw_conf")
DEBUG_RUN_DIR: Optional[Path] = None
DEBUG_SAVED_PATHS: List[Path] = []
_DEBUG_FILE_INDEX = 0

# Failure dump controls
FAILURE_OUTPUT_BASE = DEBUG_OUTPUT_BASE / "failures"
FAILURE_DUMP_DIR: Optional[Path] = None
FAILURE_SAVED_PATHS: List[Path] = []
_FAILURE_FILE_INDEX = 0


def _parse_int_list(value: Optional[str]) -> List[int]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(',')]
    result: List[int] = []
    for part in parts:
        if not part:
            continue
        result.append(int(part))
    return result


def _parse_float_tuple(value: Optional[str]) -> Tuple[float, float]:
    if not value:
        return (0.0, 0.0)
    parts = [item.strip() for item in value.split(',') if item.strip()]
    if len(parts) == 1:
        val = float(parts[0])
        return (val, 0.0)
    if len(parts) >= 2:
        return (float(parts[0]), float(parts[1]))
    return (0.0, 0.0)


def _is_2d_size_spec(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) == 2
    if isinstance(value, str):
        normalized = value.strip()
        return normalized.startswith("(") and normalized.endswith(")")
    return False


def _choose_2d_shape(total: int) -> Tuple[int, int]:
    if total <= 0:
        return (1, 1)
    root = int(math.isqrt(total))
    for factor in range(root, 0, -1):
        if total % factor == 0:
            return (factor, total // factor)
    return (1, total)

# Network topologies in AstraSim that understand per-link fault annotations.
# The neighbor builders below encode each topology's link structure so that
# fault injection only targets physically valid edges.
TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT = {
    "ring",
    "torus2d",
    "mesh",
    "mesh2d",
    "kingmesh2d",
}

_TOPOLOGIES_EXPAND_TO_2D = {"mesh2d", "torus2d"}


def _normalize_topology_name(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip().lower().replace("-", "").replace("_", "")


def _infer_non_recursive_from(dimensions: Sequence[object]) -> int:
    if not dimensions:
        return 0
    first = dimensions[0]
    topo = ""
    if isinstance(first, dict):
        topo_dict = first.get("topology")
        if isinstance(topo_dict, dict):
            topo = _normalize_topology_name(topo_dict.get("type", ""))
    return 2 if topo in _TOPOLOGIES_EXPAND_TO_2D else 0


def _expanded_start_indices(dimensions: Sequence[object]) -> List[int]:
    starts: List[int] = []
    cursor = 0
    for dim in dimensions:
        starts.append(cursor)
        topo = ""
        if isinstance(dim, dict):
            topo_dict = dim.get("topology")
            if isinstance(topo_dict, dict):
                topo = _normalize_topology_name(topo_dict.get("type", ""))
        cursor += 2 if topo in _TOPOLOGIES_EXPAND_TO_2D else 1
    return starts


def _parallelism_product(names: Iterable[object], parallelism: Dict[str, object]) -> int:
    product = 1
    for entry in names:
        axis = str(entry).strip().lower()
        if not axis:
            continue
        if axis in {"dp", "ep"}:
            train_block = parallelism.get("train", {}) if isinstance(parallelism, dict) else {}
            value = train_block.get(axis, 1)
        else:
            value = parallelism.get(axis, 1)
        try:
            factor = int(value)
        except (TypeError, ValueError):
            factor = 1
        product *= max(1, factor)
    return max(1, product)


def _resolve_dimension_size(dim: dict, parallelism: Dict[str, object]) -> int:
    size_raw = dim.get("size", "auto")
    if isinstance(size_raw, str):
        normalized = size_raw.strip().lower()
        if normalized == "auto":
            return _parallelism_product(dim.get("parallelisms", []) or [], parallelism)
        if normalized.startswith("(") and normalized.endswith(")"):
            inner = normalized[1:-1].strip()
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    return max(1, int(parts[0]) * int(parts[1]))
                except (TypeError, ValueError):
                    return 1
        try:
            return max(1, int(normalized))
        except (TypeError, ValueError):
            return 1
    if isinstance(size_raw, (list, tuple)):
        if len(size_raw) == 2:
            try:
                return max(1, int(size_raw[0]) * int(size_raw[1]))
            except (TypeError, ValueError):
                return 1
        if len(size_raw) == 1:
            try:
                return max(1, int(size_raw[0]))
            except (TypeError, ValueError):
                return 1
    try:
        return max(1, int(size_raw))
    except (TypeError, ValueError):
        return 1


def _dimension_sizes(dimensions: Sequence[object], parallelism: Dict[str, object]) -> List[int]:
    sizes: List[int] = []
    for dim in dimensions:
        if not isinstance(dim, dict):
            sizes.append(1)
            continue
        sizes.append(_resolve_dimension_size(dim, parallelism))
    return sizes


def _apply_torus2d_rect_shape(hw_dict: Dict[str, object]) -> None:
    network = hw_dict.get("network")
    if not isinstance(network, dict):
        return
    dimensions = network.get("dimensions")
    if not isinstance(dimensions, list):
        return
    parallelism = hw_dict.get("parallelism")
    if not isinstance(parallelism, dict):
        parallelism = {}
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        topology = dim.get("topology")
        topo_type = ""
        if isinstance(topology, dict):
            topo_type = _normalize_topology_name(topology.get("type", ""))
        if topo_type != "torus2d":
            continue
        size_raw = dim.get("size", "auto")
        if _is_2d_size_spec(size_raw):
            continue
        total = _resolve_dimension_size(dim, parallelism)
        if total < 1:
            continue
        if ENFORCE_SQUARE_TP_CP:
            root = int(math.isqrt(total))
            if root * root == total:
                dim["size"] = "auto"
                continue
        dim_x, dim_y = _choose_2d_shape(total)
        if dim_x <= 1 or dim_y <= 1:
            continue
        dim["size"] = [int(dim_x), int(dim_y)]


def _pair_set(pairs: Iterable[Tuple[int, int]]) -> set[Tuple[int, int]]:
    normalized = set()
    for a, b in pairs:
        lo, hi = (a, b) if a <= b else (b, a)
        normalized.add((int(lo), int(hi)))
    return normalized


def _collect_possible_fault_links(hw_dict: Dict[str, object]) -> set[Tuple[int, int]]:
    network = hw_dict.get("network")
    if not isinstance(network, dict):
        return set()
    dimensions = network.get("dimensions")
    if not isinstance(dimensions, list):
        return set()
    parallelism = hw_dict.get("parallelism")
    if not isinstance(parallelism, dict):
        parallelism = {}
    dim_sizes = _dimension_sizes(dimensions, parallelism)
    expanded_starts = _expanded_start_indices(dimensions)
    non_recursive_from = _infer_non_recursive_from(dimensions)
    possible: set[Tuple[int, int]] = set()
    for dim_index, dim in enumerate(dimensions):
        if not isinstance(dim, dict):
            continue
        topology = dim.get("topology")
        topo_type = ""
        if isinstance(topology, dict):
            topo_type = _normalize_topology_name(topology.get("type", ""))
        if topo_type not in TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT:
            continue
        dim_size = dim_sizes[dim_index] if dim_index < len(dim_sizes) else 1
        if dim_size < 2:
            continue
        neighbor_map = _neighbors_for_topology(topo_type, dim_size)
        stride = 1
        for idx in range(dim_index):
            if idx < len(dim_sizes):
                stride *= max(1, int(dim_sizes[idx]))
        stride = max(1, stride)
        expanded_start = expanded_starts[dim_index] if dim_index < len(expanded_starts) else dim_index
        non_recursive_dim = expanded_start >= non_recursive_from
        if non_recursive_dim or stride <= 1:
            offsets = [0]
        else:
            offsets = list(range(stride))
        for offset in offsets:
            for src, neighbors in enumerate(neighbor_map):
                for dst in neighbors:
                    src_id = offset + int(src) * stride
                    dst_id = offset + int(dst) * stride
                    possible.add((min(src_id, dst_id), max(src_id, dst_id)))
    return possible


def _run_fault_injection_self_test() -> None:
    mesh2d_ring_hw = {
        "parallelism": {
            "tp": 2,
            "cp": 2,
            "pp": 1,
            "train": {"dp": 2, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {"type": "Mesh2D"},
                    "size": "auto",
                    "parallelisms": ["tp", "cp"],
                },
                {
                    "id": "dim1",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["dp"],
                },
            ]
        },
    }
    mesh2d_ring_expected = _pair_set(
        [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4),
        ]
    )

    ring_ring_hw = {
        "parallelism": {
            "tp": 4,
            "cp": 1,
            "pp": 1,
            "train": {"dp": 4, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["tp"],
                },
                {
                    "id": "dim1",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["dp"],
                },
            ]
        },
    }
    ring_ring_expected = _pair_set(
        [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (8, 9), (9, 10), (10, 11), (11, 8),
            (12, 13), (13, 14), (14, 15), (15, 12),
            (0, 4), (4, 8), (8, 12), (12, 0),
        ]
    )

    cases = [
        ("mesh2d_ring_2x2_plus_ring2", mesh2d_ring_hw, mesh2d_ring_expected),
        ("ring_ring_4x4", ring_ring_hw, ring_ring_expected),
    ]

    for label, hw_dict, expected in cases:
        observed = _collect_possible_fault_links(hw_dict)
        if not observed:
            raise RuntimeError(f"[self-test] {label}: no candidate fault links generated.")
        extras = observed - expected
        if extras:
            extras_sorted = ", ".join(f"{a}-{b}" for a, b in sorted(extras))
            raise RuntimeError(
                f"[self-test] {label}: observed links outside expected set: {extras_sorted}"
            )

    banner = "=" * 72
    print(f"\n{banner}\nCHECKMARK: FAULT INJECTION SELF-TEST PASSED\n{banner}\n")

def _build_ring_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    if node_count == 1:
        return [[]]
    neighbors: List[List[int]] = []
    for idx in range(node_count):
        left = (idx - 1) % node_count
        right = (idx + 1) % node_count
        entries: List[int] = []
        for candidate in (left, right):
            if candidate != idx and candidate not in entries:
                entries.append(candidate)
        neighbors.append(entries)
    return neighbors


def _build_mesh_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    neighbors: List[List[int]] = []
    for idx in range(node_count):
        entries: List[int] = []
        if idx - 1 >= 0:
            entries.append(idx - 1)
        if idx + 1 < node_count:
            entries.append(idx + 1)
        neighbors.append(entries)
    return neighbors


def _build_mesh2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    dim = int(math.isqrt(node_count))
    if dim * dim != node_count:
        raise ValueError(f"Mesh2D topology requires perfect square node count, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    for idx in range(node_count):
        row, col = divmod(idx, dim)
        entries: List[int] = []
        if col - 1 >= 0:
            entries.append(row * dim + (col - 1))
        if col + 1 < dim:
            entries.append(row * dim + (col + 1))
        if row - 1 >= 0:
            entries.append((row - 1) * dim + col)
        if row + 1 < dim:
            entries.append((row + 1) * dim + col)
        neighbors[idx] = entries
    return neighbors


def _build_torus2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    rows, cols = _choose_2d_shape(node_count)
    if rows * cols != node_count:
        raise ValueError(f"Torus2D topology requires a valid 2D shape, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    if rows == 1 and cols == 1:
        return neighbors
    for idx in range(node_count):
        row, col = divmod(idx, cols)
        left = row * cols + ((col - 1) % cols)
        right = row * cols + ((col + 1) % cols)
        up = ((row - 1) % rows) * cols + col
        down = ((row + 1) % rows) * cols + col
        entries = []
        for candidate in (left, right, up, down):
            if candidate != idx and candidate not in entries:
                entries.append(candidate)
        neighbors[idx] = entries
    return neighbors


def _build_kingmesh2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    dim = int(math.isqrt(node_count))
    if dim * dim != node_count:
        raise ValueError(f"KingMesh2D topology requires perfect square node count, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    for idx in range(node_count):
        row, col = divmod(idx, dim)
        entries: List[int] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nr = row + dy
                nc = col + dx
                if 0 <= nr < dim and 0 <= nc < dim:
                    candidate = nr * dim + nc
                    if candidate != idx:
                        entries.append(candidate)
        neighbors[idx] = entries
    return neighbors


_NEIGHBOR_BUILDERS = {
    "ring": _build_ring_neighbors,
    "mesh": _build_mesh_neighbors,
    "mesh2d": _build_mesh2d_neighbors,
    "torus2d": _build_torus2d_neighbors,
    "kingmesh2d": _build_kingmesh2d_neighbors,
}


_NEIGHBOR_CACHE: Dict[Tuple[str, int], Tuple[Tuple[int, ...], ...]] = {}


def _neighbors_for_topology(topo_type: str, node_count: int) -> Tuple[Tuple[int, ...], ...]:
    key = (topo_type, int(node_count))
    cached = _NEIGHBOR_CACHE.get(key)
    if cached is not None:
        return cached
    builder = _NEIGHBOR_BUILDERS.get(topo_type)
    if builder is None:
        raise ValueError(f"No neighbor builder registered for topology '{topo_type}'.")
    neighbor_lists = builder(int(node_count))
    normalized = tuple(tuple(neigh) for neigh in neighbor_lists)
    _NEIGHBOR_CACHE[key] = normalized
    return normalized


def _candidate_fault_dimensions(settings: Dict[str, int]) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    axis_sizes = _axis_sizes_from_settings(settings, BASE_HW_DICT, RUN_TYPE)
    dim0_nodes = axis_sizes["tp"] * axis_sizes["cp"] * axis_sizes["ep"]
    if dim0_nodes >= 2 and (ALLOWED_FAULT_DIMS is None or 0 in ALLOWED_FAULT_DIMS):
        candidates.append((0, dim0_nodes))
    if _one_dim_enabled():
        return candidates
    dim1_nodes = axis_sizes["pp"] * axis_sizes["dp"]
    if dim1_nodes >= 2 and (ALLOWED_FAULT_DIMS is None or 1 in ALLOWED_FAULT_DIMS):
        candidates.append((1, dim1_nodes))
    return candidates


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPID-LLM fault sensitivity sweep")
    parser.add_argument("--hardware-config", type=str, default=HARDWARE_CONFIG_PATH)
    parser.add_argument("--model-config", type=str, default=MODEL_CONFIG_PATH)
    parser.add_argument(
        "--network-config",
        type=str,
        default=NETWORK_CONFIG_PATH,
        help="Optional hardware config path to copy the network section from.",
    )
    parser.add_argument("--target-num-gpus", type=int, default=TARGET_NUM_GPUS)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--fault-iter", type=int, default=FAULT_ITER)
    parser.add_argument("--fault-workers", type=int, default=FAULT_WORKERS)
    parser.add_argument("--fault-mag", type=str, default=f"{FAULT_MAG[0]},{FAULT_MAG[1] if isinstance(FAULT_MAG, (tuple, list)) else 0.0}", help="Mean[,std] for fault magnitude")
    parser.add_argument("--num-faults", type=str, default=",".join(str(n) for n in NUM_FAULTS), help="Comma-separated list of number of faults per evaluation")
    parser.add_argument("--min-tp", type=int, default=MIN_ALLOWED_TP)
    parser.add_argument("--max-tp", type=int, default=MAX_ALLOWED_TP)
    parser.add_argument("--min-cp", type=int, default=MIN_ALLOWED_CP)
    parser.add_argument("--max-cp", type=int, default=MAX_ALLOWED_CP)
    parser.add_argument("--min-dp", type=int, default=MIN_ALLOWED_DP)
    parser.add_argument("--max-dp", type=int, default=MAX_ALLOWED_DP)
    parser.add_argument("--min-ep", type=int, default=MIN_ALLOWED_EP)
    parser.add_argument("--max-ep", type=int, default=MAX_ALLOWED_EP)
    parser.add_argument("--min-pp", type=int, default=MIN_ALLOWED_PP)
    parser.add_argument("--max-pp", type=int, default=MAX_ALLOWED_PP)
    parser.add_argument("--allowed-fault-dims", type=str, default=None, help="Comma-separated list of network dimension indices eligible for faults")
    parser.add_argument("--plot-output", type=str, default=PLOT_OUTPUT_PATH)
    parser.add_argument(
        "--plot-num",
        type=int,
        default=PLOT_NUM,
        help="Generate a filtered plot with the top-N configs by baseline runtime",
    )
    parser.add_argument(
        "--plot-skip-top",
        type=int,
        default=PLOT_SKIP_TOP,
        help="Skip the fastest N configs (lowest baseline runtime) before applying --plot-num",
    )
    parser.add_argument("--report-output", type=str, default=REPORT_OUTPUT_PATH)
    parser.add_argument("--results-json", type=str, default=str(RESULTS_JSON_PATH))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--enforce-square-tp-cp",
        action="store_true",
        help="Require tp*cp to be a square power of two when sampling parallelism tuples.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug dumps of hardware configs and exit after generation.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation and regenerate the plot from the report TSV.",
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Parallelism sampling helpers
# -----------------------------------------------------------------------------

def _divisors(n: int) -> List[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def _moe_num_experts(model_config_obj: object) -> Optional[int]:
    model_cfg = getattr(model_config_obj, "model_config", None)
    if model_cfg is None:
        return None
    if not bool(getattr(model_cfg, "use_moe", False)):
        return None
    try:
        num_experts = int(getattr(model_cfg, "num_experts", 1))
    except (TypeError, ValueError):
        return None
    return num_experts if num_experts > 1 else None


def _moe_group_valid_for_inference(tp: int, moe_dp: int, moe_num_experts: int) -> bool:
    moe_group = int(tp) * int(moe_dp)
    if moe_group <= 0:
        return False
    if moe_group > moe_num_experts:
        return False
    return moe_num_experts % moe_group == 0


def _training_moe_parallelism_valid(
    model_config_obj: object,
    tp: int,
    ep: int,
    pp: int,
) -> bool:
    model_cfg = getattr(model_config_obj, "model_config", None)
    if model_cfg is None or not bool(getattr(model_cfg, "use_moe", False)):
        return True
    try:
        num_experts = int(getattr(model_cfg, "num_experts", 1))
        top_k = int(getattr(model_cfg, "top_k", 1))
        global_batch = int(getattr(model_cfg, "global_batch_size", 1))
        grad_accum = int(getattr(model_cfg, "gradient_accumulation_steps", 1))
        seq_len = int(getattr(model_cfg, "seq_len", 1))
    except (TypeError, ValueError):
        return False
    grad_accum = max(1, grad_accum)
    if global_batch % grad_accum != 0:
        return False
    batch_size = global_batch // grad_accum
    if ep <= 0 or num_experts <= 0:
        return False
    if batch_size % ep != 0:
        return False
    if num_experts % ep != 0:
        return False

    dp_dense = ep
    mini_batch = batch_size // dp_dense
    mb = 1 if pp <= 1 else pp
    if mini_batch % mb != 0:
        return False
    effective_batch = mini_batch
    if pp > 1:
        effective_batch = mini_batch // mb
    elif dp_dense <= 1:
        effective_batch = batch_size

    seq_per_rank = int(math.ceil(float(seq_len) / 1.0))
    tokens_owner = int(effective_batch) * int(seq_per_rank)
    tokens_dispatched = tokens_owner * int(top_k)
    if tokens_dispatched % ep != 0:
        return False
    experts_per_rank = num_experts // ep
    if experts_per_rank <= 0:
        return False
    tokens_local = tokens_dispatched // ep
    if tokens_local % experts_per_rank != 0:
        return False
    return True


def _enumerate_training_moe_vectors(
    num_gpus: int,
    model_config_obj: object,
) -> List[Tuple[int, int, int, int, int]]:
    model_cfg = getattr(model_config_obj, "model_config", None)
    if model_cfg is None or not bool(getattr(model_cfg, "use_moe", False)):
        return []
    try:
        num_experts = int(getattr(model_cfg, "num_experts", 1))
        global_batch = int(getattr(model_cfg, "global_batch_size", 1))
        grad_accum = int(getattr(model_cfg, "gradient_accumulation_steps", 1))
    except (TypeError, ValueError):
        return []
    grad_accum = max(1, grad_accum)
    if global_batch % grad_accum != 0:
        return []
    batch_size = global_batch // grad_accum
    ep_candidates = [ep for ep in _divisors(num_experts) if ep <= batch_size]

    vectors: List[Tuple[int, int, int, int, int]] = []
    for ep in ep_candidates:
        if not (MIN_ALLOWED_EP <= ep <= MAX_ALLOWED_EP):
            continue
        for tp in range(MIN_ALLOWED_TP, MAX_ALLOWED_TP + 1):
            denom = tp * ep
            if denom <= 0 or num_gpus % denom != 0:
                continue
            pp = num_gpus // denom
            if not (MIN_ALLOWED_PP <= pp <= MAX_ALLOWED_PP):
                continue
            if not _training_moe_parallelism_valid(model_config_obj, tp, ep, pp):
                continue
            vectors.append((tp, 1, 1, pp, ep))
    return vectors


def _enumerate_parallelism_vectors_full_range(num_gpus: int) -> List[Tuple[int, int, int, int]]:
    vectors: List[Tuple[int, int, int, int]] = []
    for tp in range(MIN_ALLOWED_TP, MAX_ALLOWED_TP + 1):
        for cp in range(MIN_ALLOWED_CP, MAX_ALLOWED_CP + 1):
            for dp in range(MIN_ALLOWED_DP, MAX_ALLOWED_DP + 1):
                denom = tp * cp * dp
                if denom <= 0:
                    continue
                if num_gpus % denom != 0:
                    continue
                pp = num_gpus // denom
                vectors.append((tp, cp, dp, pp))
    return vectors


def _enumerate_parallelism_vectors(num_gpus: int) -> List[Tuple[int, int, int, int]]:
    vectors: List[Tuple[int, int, int, int]] = []
    for tp in _divisors(num_gpus):
        rem_tp = num_gpus // tp
        for cp in _divisors(rem_tp):
            rem_cp = rem_tp // cp
            for dp in _divisors(rem_cp):
                pp = rem_cp // dp
                vectors.append((tp, cp, dp, pp))
    return vectors


def _log_distance(a: Sequence[int], b: Sequence[int]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += abs(math.log2(max(x, 1)) - math.log2(max(y, 1)))
    return total


def _select_representative_vectors(candidates: List[Tuple[int, ...]], count: int) -> List[Tuple[int, ...]]:
    if not candidates:
        return []
    selected: List[Tuple[int, ...]] = []

    # Seed with extreme configurations
    dim_count = len(candidates[0])
    extremes = [max(candidates, key=lambda v, idx=i: v[idx]) for i in range(dim_count)]
    for cand in extremes:
        if cand not in selected:
            selected.append(cand)
        if len(selected) >= count:
            return selected[:count]

    # Farthest-point sampling to promote coverage
    remaining = [v for v in candidates if v not in selected]
    while remaining and len(selected) < count:
        best_vec = None
        best_score = -1.0
        for vec in remaining:
            min_dist = min(_log_distance(vec, sel) for sel in selected)
            if min_dist > best_score:
                best_score = min_dist
                best_vec = vec
        if best_vec is None:
            break
        selected.append(best_vec)
        remaining.remove(best_vec)

    # If still short, append random leftovers
    while remaining and len(selected) < count:
        candidate = remaining.pop(random.randrange(len(remaining)))
        selected.append(candidate)

    return selected[:count]


def _filtered_parallelism_vectors(
    num_gpus: int,
    run_type: str,
    moe_num_experts: Optional[int],
    model_config_obj: Optional[object] = None,
) -> List[Tuple[int, ...]]:
    use_training_ep_sweep = run_type == "training" and GLM_MODE and GLM_TRAIN
    use_moe_filter = bool(moe_num_experts) and run_type == "inference" and SWEEP_MOE_DP
    if use_training_ep_sweep and model_config_obj is not None:
        tuples: List[Tuple[int, ...]] = _enumerate_training_moe_vectors(num_gpus, model_config_obj)
    elif use_moe_filter:
        tuples = _enumerate_parallelism_vectors_full_range(num_gpus)
    else:
        tuples = _enumerate_parallelism_vectors(num_gpus)
    filtered: List[Tuple[int, ...]] = []
    for vector in tuples:
        if len(vector) == 4:
            tp, cp, dp, pp = vector
            ep = 1
        elif len(vector) == 5:
            tp, cp, dp, pp, ep = vector
        else:
            continue
        if not (MIN_ALLOWED_TP <= tp <= MAX_ALLOWED_TP):
            continue
        if not (MIN_ALLOWED_CP <= cp <= MAX_ALLOWED_CP):
            continue
        if not (MIN_ALLOWED_DP <= dp <= MAX_ALLOWED_DP):
            continue
        if not (MIN_ALLOWED_PP <= pp <= MAX_ALLOWED_PP):
            continue
        if not (MIN_ALLOWED_EP <= ep <= MAX_ALLOWED_EP):
            continue
        if tp * cp < MIN_TP_CP_PROD:
            continue
        if tp * cp > MAX_TP_CP_PROD:
            continue
        if ENFORCE_SQUARE_TP_CP and not tp_cp_product_is_power_of_two_square(tp, cp):
            continue
        if use_moe_filter and moe_num_experts is not None:
            if not _moe_group_valid_for_inference(tp, dp, moe_num_experts):
                continue
        filtered.append(vector)
    return filtered


def _make_parallelism_settings(tp: int, cp: int, dp: int, pp: int, ep: int = 1) -> Dict[str, int]:
    return {
        "tp": tp,
        "cp": cp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "mb": 1 if pp == 1 else 2*pp,
        "tp_sp": True,
    }


def _vector_to_settings(vector: Sequence[int]) -> Dict[str, int]:
    if len(vector) == 4:
        tp, cp, dp, pp = vector
        ep = 1
    elif len(vector) == 5:
        tp, cp, dp, pp, ep = vector
    else:
        raise ValueError(f"Unsupported parallelism vector length: {len(vector)}")
    return _make_parallelism_settings(tp, cp, dp, pp, ep=ep)


def generate_parallelism_samples(
    num_gpus: int,
    sample_count: int,
    base_hw_dict: Dict[str, object],
    model_config_obj: object,
    mode: str,
    workers: Optional[int] = None,
) -> List[Dict[str, int]]:
    run_type = _model_run_type(model_config_obj)
    moe_num_experts = _moe_num_experts(model_config_obj)
    vectors = _filtered_parallelism_vectors(num_gpus, run_type, moe_num_experts, model_config_obj)
    if not vectors:
        return []
    # Keep ordering stable but cover space by using representative sampling.
    prioritized = deque(_select_representative_vectors(vectors, len(vectors)))
    selected: List[Dict[str, int]] = []

    worker_count = max(1, int(workers) if workers is not None else 1)
    worker_count = min(worker_count, max(1, os.cpu_count() or 1))

    print(f"Choosing out of {len(vectors)} candidate parallelism configurations...")

    if worker_count <= 1:
        while prioritized and len(selected) < sample_count:
            vector = prioritized.popleft()
            settings = _vector_to_settings(vector)
            try:
                # Use fast memory check instead of full evaluation
                mem_check = check_memory_capacity(base_hw_dict, model_config_obj, mode, settings)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"Warning: failed to check memory for parallelism {settings} during sampling: {exc}",
                    file=sys.stderr,
                )
                continue
            if bool(mem_check.get("memory_exceeded", False)):
                continue
            selected.append(settings)
    else:
        backlog_target = worker_count * 3
        in_flight: Dict[object, Dict[str, int]] = {}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallelism_worker_init,
            initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode, run_type, SWEEP_MOE_DP),
        ) as executor:
            while (prioritized or in_flight) and len(selected) < sample_count:
                while (
                    prioritized
                    and len(in_flight) < backlog_target
                    and len(selected) + len(in_flight) < sample_count + backlog_target
                ):
                    vector = prioritized.popleft()
                    settings = _vector_to_settings(vector)
                    fut = executor.submit(_sampling_worker, settings)
                    in_flight[fut] = settings
                if not in_flight:
                    break
                completed = next(as_completed(list(in_flight.keys())))
                settings = in_flight.pop(completed, None)
                try:
                    result = completed.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"Warning: failed to evaluate parallelism {settings} during sampling: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if result.get("error"):
                    print(
                        f"Warning: failed to evaluate parallelism {settings} during sampling: {result.get('error')}",
                        file=sys.stderr,
                    )
                    continue
                if bool(result.get("memory_exceeded", False)):
                    continue
                if settings is not None:
                    selected.append(settings)
                    print(f"Selected parallelism configuration: {settings}")

    if len(selected) < sample_count and not prioritized:
        print(
            f"Warning: only {len(selected)} parallelism configuration(s) fit in memory "
            f"out of requested {sample_count}.",
            file=sys.stderr,
        )
    return selected


# -----------------------------------------------------------------------------
# Fault injection helpers
# -----------------------------------------------------------------------------

def sample_fault_magnitude() -> float:
    if isinstance(FAULT_MAG, (tuple, list)) and len(FAULT_MAG) == 2:
        mean, std = FAULT_MAG
        value = random.gauss(mean, std)
    else:
        value = float(FAULT_MAG)
    value = max(0.0, min(0.95, value))
    value = _quantize_weight(value)
    return max(0.1, value) # min of 10% bw, max of 95% bw


def sample_hard_fault_magnitude() -> float:
    return 0.0


def _hard_faults_enabled() -> bool:
    if HARD_FAULT_ITER <= 0:
        return False
    if not HARD_NUM_FAULTS:
        return False
    try:
        return int(HARD_NUM_FAULTS[0]) > 0
    except (TypeError, ValueError):
        return False


def make_fault_mutator(fault_specs: Sequence[Tuple[int, float, int]]):
    fault_specs = [
        (int(dim_idx), float(weight), int(node_count))
        for dim_idx, weight, node_count in fault_specs
    ]

    def _mutator(hw_dict: Dict[str, object]) -> None:
        network = hw_dict.get("network")
        if not isinstance(network, dict):
            return
        dimensions = network.get("dimensions")
        if not isinstance(dimensions, list):
            return
        parallelism = hw_dict.get("parallelism")
        if not isinstance(parallelism, dict):
            parallelism = {}
        dim_sizes = _dimension_sizes(dimensions, parallelism)
        expanded_starts = _expanded_start_indices(dimensions)
        non_recursive_from = _infer_non_recursive_from(dimensions)
        fault_entries: List[FlowSeq] = []
        for dimension_index, weight, node_count in fault_specs:
            if not (0 <= dimension_index < len(dimensions)):
                continue
            dim = dimensions[dimension_index]
            if not isinstance(dim, dict):
                continue
            topology = dim.get("topology")
            topo_type = ""
            if isinstance(topology, dict):
                topo_raw = topology.get("type", "")
                if isinstance(topo_raw, str):
                    topo_type = _normalize_topology_name(topo_raw)
            supports_faulty_links = topo_type in TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT
            if not supports_faulty_links:
                label = dim.get("label", f"dim{dimension_index}")
                raise ValueError(
                    f"Topology '{topo_type or 'unknown'}' on network dimension '{label}' "
                    "does not support faulty_links injection."
                )
            dim_size = dim_sizes[dimension_index] if dimension_index < len(dim_sizes) else node_count
            if dim_size < 2:
                continue
            try:
                neighbor_map = _neighbors_for_topology(topo_type, dim_size)
            except ValueError as exc:
                raise ValueError(
                    f"Unable to determine neighbors for topology '{topo_type or 'unknown'}' "
                    f"with node_count={dim_size}: {exc}"
                ) from exc
            valid_sources = [idx for idx, neigh in enumerate(neighbor_map) if neigh]
            if not valid_sources:
                continue
            src = random.choice(valid_sources)
            dst_candidates = neighbor_map[src]
            dst = random.choice(dst_candidates)
            stride = 1
            for idx in range(dimension_index):
                if idx < len(dim_sizes):
                    stride *= max(1, int(dim_sizes[idx]))
            stride = max(1, stride)
            expanded_start = expanded_starts[dimension_index] if dimension_index < len(expanded_starts) else dimension_index
            non_recursive_dim = expanded_start >= non_recursive_from
            lower_offset = 0
            if not non_recursive_dim and stride > 1:
                lower_offset = random.randrange(stride)
            src = lower_offset + int(src) * stride
            dst = lower_offset + int(dst) * stride
            weight_clamped = float(max(0.0, min(1.0, weight)))
            weight_clamped = _quantize_weight(weight_clamped)
            dim.pop("faulty_links", None)
            fault_entries.append(FlowSeq([int(src), int(dst), weight_clamped]))

        if fault_entries:
            network["faulty_links"] = FlowSeq(fault_entries)
        else:
            network.pop("faulty_links", None)

    return _mutator


def _compose_hw_mutators(*mutators):
    def _mutator(hw_dict: Dict[str, object]) -> None:
        for mut in mutators:
            if mut is not None:
                mut(hw_dict)
    return _mutator


# -----------------------------------------------------------------------------
# Fast memory checking using new memory estimation API
# -----------------------------------------------------------------------------

def check_memory_capacity(base_hw_dict, model_config_obj, mode, parallel_settings, hw_mutator=None):
    """
    Fast memory capacity check using the new memory estimation API.

    Returns a dictionary with:
        - memory_exceeded: bool indicating if capacity is violated
        - memory_violation_gb: float indicating how much over capacity (if exceeded)
        - summary: the full memory estimation summary dict

    This is much faster than running full timing calculations.
    """
    run_type = _model_run_type(model_config_obj)
    normalized = _normalize_parallelism_settings(base_hw_dict, parallel_settings, run_type)
    mutator = _apply_torus2d_rect_shape
    if hw_mutator is not None:
        mutator = _compose_hw_mutators(mutator, hw_mutator)
    hw_config, _ = make_temp_hw_config(base_hw_dict, normalized, hw_mutator=mutator)

    try:
        # Use the dispatcher function which automatically determines inference vs training
        summary = memory_estimation.estimate_memory(
            hw_config,
            model_config_obj,
            mode=mode,
            output_dir=None,  # Don't write outputs for fast checks
        )

        # Check if this is training (has capacity_exceeded key) or inference (has max_peak_gb key)
        if 'capacity_exceeded' in summary:
            # Training workload
            capacity_exceeded = bool(summary.get('capacity_exceeded', False))
            violation = float(summary.get('capacity_violation_gb', 0.0) or 0.0)
            return {
                "memory_exceeded": capacity_exceeded,
                "memory_violation_gb": violation,
                "summary": summary,
            }
        else:
            # Inference workload - check if max_peak exceeds capacity
            max_peak = summary.get('max_peak_gb', 0.0)
            capacity = summary.get('capacity_gb')
            if capacity is not None and max_peak > capacity:
                violation = max_peak - capacity
                return {
                    "memory_exceeded": True,
                    "memory_violation_gb": violation,
                    "summary": summary,
                }
            return {
                "memory_exceeded": False,
                "memory_violation_gb": 0.0,
                "summary": summary,
            }
    except Exception as exc:
        # If memory check fails, return conservatively (assume no violation)
        # The full evaluate_parallelism will catch any real errors
        print(f"Warning: fast memory check failed: {exc}", file=sys.stderr)
        return {
            "memory_exceeded": False,
            "memory_violation_gb": 0.0,
            "summary": {},
        }


# -----------------------------------------------------------------------------
# Multiprocessing helpers
# -----------------------------------------------------------------------------

_WORKER_HW_DICT = None
_WORKER_MODEL_CONFIG = None
_WORKER_MODE = None


def _parallelism_worker_init(base_hw_dict, model_config_path, mode, run_type, sweep_moe_dp):
    global _WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE, RUN_TYPE, SWEEP_MOE_DP
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)
    _WORKER_HW_DICT = base_hw_dict
    _WORKER_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)
    _WORKER_MODE = mode
    RUN_TYPE = str(run_type or "training").lower()
    SWEEP_MOE_DP = bool(sweep_moe_dp)


def _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task: Dict[str, object]) -> Dict[str, object]:
    kind = task.get("kind")
    settings = dict(task.get("settings", {}))
    key = task.get("key")
    settings_key = tuple(task.get("settings_key", ()))
    task_id = task.get("task_id")
    fault_iter = task.get("fault_iter")
    fault_mode = task.get("fault_mode") if kind == "fault" else None
    faults: List[Tuple[int, float, int]] = []
    num_faults = 0
    if kind == "fault":
        raw_faults = task.get("faults", [])
        try:
            faults = [tuple(spec) for spec in raw_faults]
        except Exception:
            faults = []
        num_faults = int(task.get("num_faults", len(faults)))
    try:
        mutator = _apply_torus2d_rect_shape
        if kind == "fault":
            mutator = _compose_hw_mutators(mutator, make_fault_mutator(faults))
        run_type = _model_run_type(model_config_obj)
        normalized = _normalize_parallelism_settings(base_hw_dict, settings, run_type)
        metrics = evaluate_parallelism(
            base_hw_dict,
            model_config_obj,
            mode,
            normalized,
            hw_mutator=mutator,
        )
        mem_violation = metrics.get("memory_violation_gb", 0.0) or 0.0
        result: Dict[str, object] = {
            "status": "ok",
            "kind": kind,
            "key": key,
            "settings": settings,
            "settings_key": settings_key,
            "runtime": float(metrics["runtime"]),
            "hw_yaml": metrics.get("hw_yaml"),
            "task_id": task_id,
            "memory_exceeded": bool(metrics.get("memory_exceeded", False)),
            "memory_violation_gb": float(mem_violation),
        }
        if fault_iter is not None:
            result["fault_iter"] = int(fault_iter)
        if kind == "fault":
            result.update(
                {
                    "faults": faults,
                    "num_faults": num_faults,
                    "fault_mode": fault_mode or "soft",
                }
            )
        else:
            result.update({"faults": [], "num_faults": 0})
        return result
    except Exception as exc:
        error_result = {
            "status": "error",
            "kind": kind,
            "key": key,
            "settings": settings,
            "settings_key": settings_key,
            "task_id": task_id,
            "error": str(exc),
        }
        if fault_iter is not None:
            error_result["fault_iter"] = int(fault_iter)
        if kind == "fault":
            error_result.update({"faults": faults, "num_faults": num_faults, "fault_mode": fault_mode or "soft"})
        else:
            error_result.update({"faults": [], "num_faults": 0})
        return error_result


def _parallelism_worker_task(task: Dict[str, object]) -> Dict[str, object]:
    if _WORKER_HW_DICT is None or _WORKER_MODEL_CONFIG is None or _WORKER_MODE is None:
        raise RuntimeError("Worker initialisation missing before task execution.")
    return _execute_parallelism_task(_WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE, task)


def _sampling_worker(settings: Dict[str, int]) -> Dict[str, object]:
    if _WORKER_HW_DICT is None or _WORKER_MODEL_CONFIG is None or _WORKER_MODE is None:
        raise RuntimeError("Worker initialisation missing before task execution.")
    try:
        # Use fast memory check instead of full evaluation during sampling
        mem_check = check_memory_capacity(_WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE, settings)
        return {
            "settings": settings,
            "memory_exceeded": bool(mem_check.get("memory_exceeded", False)),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"settings": settings, "memory_exceeded": True, "error": str(exc)}


def _parallelism_key(settings: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(settings.items()))


def _serialize_settings_key(key: Tuple[Tuple[str, object], ...]) -> List[List[object]]:
    return [[k, v] for k, v in key]


def _deserialize_settings_key(raw: Iterable[Iterable[object]]) -> Tuple[Tuple[str, object], ...]:
    return tuple((str(k), v) for k, v in raw)


def _normalise_faults_raw(faults: Iterable[Iterable[object]]) -> List[List[object]]:
    normalised: List[List[object]] = []
    for dim, weight, nodes in faults:
        normalised.append([int(dim), _quantize_weight(float(weight)), int(nodes)])
    return normalised


def _task_identifier(
    kind: str,
    settings_key: Tuple[Tuple[str, object], ...],
    *,
    faults: Sequence[Tuple[int, float, int]] | None = None,
    num_faults: int | None = None,
    fault_iter: Optional[int] = None,
    fault_mode: Optional[str] = None,
) -> str:
    payload: Dict[str, object] = {
        "kind": kind,
        "settings_key": _serialize_settings_key(settings_key),
    }
    if fault_iter is not None:
        payload["fault_iter"] = int(fault_iter)
    if kind == "fault":
        if fault_mode and fault_mode != "soft":
            payload["fault_mode"] = str(fault_mode)
        effective_faults = faults or ()
        payload["num_faults"] = int(num_faults if num_faults is not None else len(effective_faults))
        payload["faults"] = _normalise_faults_raw(effective_faults)
    return json.dumps(payload, sort_keys=True)


def _load_existing_results(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        print(f"Warning: results file {path} did not contain a JSON object. Ignoring existing data.", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to load results file {path}: {exc}", file=sys.stderr)
    return {}


def _record_partial_result(results_store: Dict[str, Dict[str, object]], entry: Dict[str, object]) -> None:
    task_id = entry.get("task_id")
    if not task_id:
        return
    results_store[task_id] = {
        "kind": entry.get("kind"),
        "settings_key": _serialize_settings_key(tuple(entry.get("settings_key", ()))),
        "runtime": entry.get("runtime"),
        "memory_exceeded": bool(entry.get("memory_exceeded", False)),
        "memory_violation_gb": float(entry.get("memory_violation_gb", 0.0) or 0.0),
    }
    if entry.get("kind") == "fault":
        if "fault_iter" in entry:
            results_store[task_id]["fault_iter"] = int(entry.get("fault_iter"))
        if entry.get("fault_mode") is not None:
            results_store[task_id]["fault_mode"] = str(entry.get("fault_mode"))
        results_store[task_id]["num_faults"] = int(entry.get("num_faults", 0))
        results_store[task_id]["faults"] = _normalise_faults_raw(entry.get("faults", []))
    tmp_path = RESULTS_JSON_PATH.with_suffix(".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_WRITE_LOCK:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(results_store, handle, indent=2)
        tmp_path.replace(RESULTS_JSON_PATH)


def _dump_debug_hw_config(result: Dict[str, object]) -> Optional[Path]:
    if not DEBUG_MODE:
        return None
    yaml_text = result.get("hw_yaml")
    if not isinstance(yaml_text, str) or not yaml_text.strip():
        return None
    if DEBUG_RUN_DIR is None:
        return None
    global _DEBUG_FILE_INDEX
    _DEBUG_FILE_INDEX += 1
    settings = result.get("settings") or {}
    kind = str(result.get("kind") or "task")
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    pp = settings.get("pp")
    label = f"{kind}_{_DEBUG_FILE_INDEX:03d}"
    if all(isinstance(val, (int, float)) for val in (tp, cp, dp, pp)):
        label += f"_{_format_parallelism_tag(settings)}"
    num_faults = result.get("num_faults")
    fault_mode = result.get("fault_mode")
    if kind == "fault" and fault_mode:
        label += f"_{fault_mode}"
    if kind == "fault" and isinstance(num_faults, int):
        label += f"_{num_faults}faults"
    file_path = DEBUG_RUN_DIR / f"{label}.yaml"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(yaml_text)
    DEBUG_SAVED_PATHS.append(file_path)
    return file_path


def _generate_hw_yaml_from_task(base_hw_dict: Dict[str, object], task: Dict[str, object]) -> Optional[str]:
    if base_hw_dict is None:
        return None
    settings = dict(task.get("settings", {}) or {})
    try:
        normalized = _normalize_parallelism_settings(base_hw_dict, settings, RUN_TYPE)
    except Exception as exc:
        print(f"Warning: failed to normalize parallelism settings for debug dump: {exc}", file=sys.stderr)
        return None

    mutator = None
    faults: Sequence[Tuple[int, float, int]] = ()
    if task.get("kind") == "fault":
        raw_faults = task.get("faults") or []
        try:
            faults = [tuple(spec) for spec in raw_faults]  # type: ignore[assignment]
        except Exception:
            faults = ()
        mutator = make_fault_mutator(faults)

    try:
        _, debug_yaml = make_temp_hw_config(base_hw_dict, normalized, hw_mutator=mutator)
        return debug_yaml
    except Exception as exc:
        print(f"Warning: failed to serialize debug hardware config: {exc}", file=sys.stderr)
        return None


def _ensure_failure_dump_dir() -> Path:
    global FAILURE_DUMP_DIR
    if FAILURE_DUMP_DIR is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        FAILURE_DUMP_DIR = FAILURE_OUTPUT_BASE / timestamp
        FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving failing hardware configs to {FAILURE_DUMP_DIR}")
    return FAILURE_DUMP_DIR


def _failure_label_from_settings(
    kind: str,
    settings: Dict[str, object],
    num_faults: Optional[int],
    fault_mode: Optional[str] = None,
) -> str:
    label = kind or "task"
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    pp = settings.get("pp")
    if all(isinstance(val, (int, float)) for val in (tp, cp, dp, pp)):
        label += f"_{_format_parallelism_tag(settings)}"
    if kind == "fault" and fault_mode:
        label += f"_{fault_mode}"
    if kind == "fault" and isinstance(num_faults, int):
        label += f"_{num_faults}faults"
    return label


def _dump_failure_hw_config(
    base_hw_dict: Dict[str, object],
    result: Dict[str, object],
    error_message: Optional[str] = None,
) -> Optional[Path]:
    task_like = {
        "kind": result.get("kind"),
        "settings": result.get("settings"),
        "faults": result.get("faults"),
        "num_faults": result.get("num_faults"),
    }
    yaml_text = _generate_hw_yaml_from_task(base_hw_dict, task_like)
    failure_dir = _ensure_failure_dump_dir()
    global _FAILURE_FILE_INDEX
    _FAILURE_FILE_INDEX += 1
    settings = dict(result.get("settings", {}) or {})
    kind = str(result.get("kind") or "task")
    num_faults = result.get("num_faults")
    fault_mode = result.get("fault_mode")
    label = _failure_label_from_settings(
        kind,
        settings,
        num_faults if isinstance(num_faults, int) else None,
        fault_mode=str(fault_mode) if fault_mode else None,
    )
    base_filename = f"failure_{_FAILURE_FILE_INDEX:03d}_{label}"
    yaml_path = failure_dir / f"{base_filename}.yaml"
    command_line = None
    if isinstance(yaml_text, str) and yaml_text.strip():
        with yaml_path.open("w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        FAILURE_SAVED_PATHS.append(yaml_path)
        print(f"Stored failing hardware config at {yaml_path}")
        cli_hw_path = os.path.relpath(yaml_path, os.getcwd())
        command_line = (
            f"uv run run_perf.py --hardware_config {shlex.quote(cli_hw_path)} "
            f"--model_config {shlex.quote(MODEL_CONFIG_PATH)}"
        )
    else:
        yaml_path = None
    if error_message or command_line:
        log_path = failure_dir / f"{base_filename}_error.txt"
        with log_path.open("w", encoding="utf-8") as handle:
            if error_message:
                handle.write(str(error_message).strip() + "\n")
            if command_line:
                handle.write("\n# Reproduce failing configuration\n")
                handle.write(command_line + "\n")
        FAILURE_SAVED_PATHS.append(log_path)
    return yaml_path


def _terminate_active_executor() -> None:
    global _ACTIVE_EXECUTOR
    executor = _ACTIVE_EXECUTOR
    if executor is None:
        return
    _ACTIVE_EXECUTOR = None
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    processes = getattr(executor, "_processes", None)
    if isinstance(processes, dict):
        for proc in processes.values():
            try:
                if proc is not None and proc.is_alive():
                    proc.kill()
            except Exception:
                continue


def _signal_handler(signum, frame):  # type: ignore[override]
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        os._exit(128 + signum)
    _SHUTDOWN_REQUESTED = True
    _terminate_active_executor()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt()
    sys.exit(128 + signum)


def _install_signal_handlers() -> None:
    global _OLD_SIGINT_HANDLER, _OLD_SIGTERM_HANDLER
    _OLD_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
    _OLD_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_terminate_active_executor)


# -----------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------

def summarise_fault_runs(runtimes: Iterable[float]) -> Tuple[float, float, float]:
    values = [float(v) for v in runtimes if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return min(values), max(values), float(sum(values) / len(values))


def read_fault_report(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        print(f"Fault report not found at {path}", file=sys.stderr)
        return []
    with open(path, "r") as handle:
        header_line = handle.readline().strip()
        if not header_line:
            return []
        columns = [col.strip() for col in header_line.split("\t") if col.strip()]
        idx = {name: i for i, name in enumerate(columns)}

        def _get(parts: List[str], key: str, default: object = None) -> object:
            pos = idx.get(key)
            if pos is None or pos >= len(parts):
                return default
            return parts[pos]

        dp_col = None
        for candidate in ("moe_dp", "dp", "moe"):
            if candidate in idx:
                dp_col = candidate
                break
        if dp_col is None:
            print("Fault report missing dp/moe_dp column.", file=sys.stderr)
            return []

        records: List[Dict[str, object]] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            try:
                tp = int(_get(parts, "tp", 1))
                cp = int(_get(parts, "cp", 1))
                ep = int(_get(parts, "ep", 1))
                dp = int(_get(parts, dp_col, 1))
                pp = int(_get(parts, "pp", 1))
                baseline = float(_get(parts, "baseline_runtime", float("nan")))
                fault_min = float(_get(parts, "fault_min", float("nan")))
                fault_max = float(_get(parts, "fault_max", float("nan")))
                fault_mean = float(_get(parts, "fault_mean", float("nan")))
                label = str(_get(parts, "label", "") or "")
            except (TypeError, ValueError):
                continue
            settings = _make_parallelism_settings(tp, cp, dp, pp, ep=ep)
            if not label:
                label = _format_parallelism_label(settings)
            records.append(
                {
                    "parallelism": settings,
                    "baseline_runtime": baseline,
                    "fault_min": fault_min,
                    "fault_max": fault_max,
                    "fault_mean": fault_mean,
                    "label": label,
                }
            )
    return records


def write_fault_report(records: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dp_label = _dp_label()
    header = [
        "tp",
        "cp",
        "ep",
        dp_label,
        "pp",
        "baseline_runtime",
        "fault_min",
        "fault_max",
        "fault_mean",
        "label",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for record in records:
            row = [
                str(record["parallelism"]["tp"]),
                str(record["parallelism"]["cp"]),
                str(record["parallelism"].get("ep", 1)),
                str(record["parallelism"]["dp"]),
                str(record["parallelism"]["pp"]),
                f"{record['baseline_runtime']:.6f}",
                f"{record['fault_min']:.6f}" if math.isfinite(record["fault_min"]) else "nan",
                f"{record['fault_max']:.6f}" if math.isfinite(record["fault_max"]) else "nan",
                f"{record['fault_mean']:.6f}" if math.isfinite(record["fault_mean"]) else "nan",
                str(record.get("label", "")),
            ]
            handle.write("\t".join(row) + "\n")


def _baseline_sort_key(record: Dict[str, object]) -> float:
    try:
        baseline = float(record.get("baseline_runtime", float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return baseline if math.isfinite(baseline) else float("inf")


def _select_top_by_baseline(
    records: List[Dict[str, object]],
    limit: int,
    *,
    skip_top: int = 0,
) -> List[Dict[str, object]]:
    if not records:
        return []
    safe_skip = max(0, int(skip_top) if skip_top is not None else 0)
    ordered = sorted(records, key=_baseline_sort_key)
    start = min(safe_skip, len(ordered))
    end = len(ordered) if limit <= 0 else min(start + limit, len(ordered))
    return ordered[start:end]


def _format_runtime(value: object) -> str:
    try:
        runtime = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{runtime:.4f}" if math.isfinite(runtime) else "nan"


def _print_top_records(records: List[Dict[str, object]], label: str, *, skip_top: int = 0) -> None:
    if not records:
        return
    if skip_top > 0:
        print(
            f"Selected {len(records)} configs by baseline runtime after skipping {skip_top} fastest ({label}):"
        )
    else:
        print(f"Top {len(records)} configs by baseline runtime ({label}):")
    for record in records:
        baseline = _format_runtime(record.get("baseline_runtime"))
        fault_mean = _format_runtime(record.get("fault_mean"))
        print(f"  {_record_display_label(record)}: baseline={baseline}s, fault_mean={fault_mean}s")


def _filter_records_by_keys(
    records: List[Dict[str, object]],
    keys: Sequence[Tuple[int, int, int, int, int]],
) -> List[Dict[str, object]]:
    record_map = {}
    for record in records:
        key = _record_key(record)
        if key is not None:
            record_map[key] = record
    return [record_map[key] for key in keys if key in record_map]


def plot_fault_sensitivity(
    records: List[Dict[str, object]],
    output_path: str,
    num_gpus: int,
    *,
    fault_color: str = "#d7301f",
    fault_range_color: str = "#fb6a4a",
    fault_fill_color: str = "#fdd0a2",
    fault_label: str = "Fault mean ± range",
    normalize: bool = False,
) -> None:
    if not records:
        print("No records to plot.", file=sys.stderr)
        return

    if sns is not None:  # pragma: no branch
        sns.set_theme(style="whitegrid")

    xs = np.arange(len(records))
    baseline = np.array([rec["baseline_runtime"] for rec in records], dtype=float)
    fault_mean = np.array([rec["fault_mean"] for rec in records], dtype=float)
    fault_min = np.array([rec["fault_min"] for rec in records], dtype=float)
    fault_max = np.array([rec["fault_max"] for rec in records], dtype=float)

    if normalize:
        finite_baseline = baseline[np.isfinite(baseline)]
        norm_factor = float(np.min(finite_baseline)) if finite_baseline.size else 1.0
        if not math.isfinite(norm_factor) or norm_factor <= 0:
            norm_factor = 1.0
        baseline = baseline / norm_factor
        fault_mean = fault_mean / norm_factor
        fault_min = fault_min / norm_factor
        fault_max = fault_max / norm_factor

    config_labels = [_record_display_label(rec) for rec in records]

    plt.figure(figsize=(12, 6))
    plt.scatter(xs, baseline, color="#1f78b4", marker="o", s=80, label="Baseline")

    fault_lower = []
    fault_upper = []
    for mean_val, min_val, max_val in zip(fault_mean, fault_min, fault_max):
        if math.isfinite(mean_val) and math.isfinite(min_val):
            fault_lower.append(max(mean_val - min_val, 0.0))
        else:
            fault_lower.append(0.0)
        if math.isfinite(mean_val) and math.isfinite(max_val):
            fault_upper.append(max(max_val - mean_val, 0.0))
        else:
            fault_upper.append(0.0)

    plt.errorbar(
        xs,
        fault_mean,
        yerr=[fault_lower, fault_upper],
        fmt="s",
        color=fault_color,
        ecolor=fault_range_color,
        elinewidth=2,
        capsize=6,
        capthick=1.6,
        markersize=7,
        label=fault_label,
    )

    # Candlestick-style shading for min/max bounds
    for x, fmin, fmax in zip(xs, fault_min, fault_max):
        if math.isfinite(fmin) and math.isfinite(fmax):
            plt.fill_between(
                [x - 0.22, x + 0.22],
                [fmin, fmin],
                [fmax, fmax],
                color=fault_fill_color,
                alpha=0.4,
            )

    plt.xticks(xs, config_labels, rotation=45, ha="right", fontsize=16)
    ylabel = "Normalized runtime (fastest=1.0)" if normalize else "Runtime (s)"
    plt.ylabel(ylabel, fontsize=19)
    plt.title(f"Fault Sensitivity Across Parallelism Configurations (Num GPUs = {num_gpus})", fontsize=18)
    plt.grid(alpha=0.3, axis="y")
    plt.legend(fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved fault sensitivity plot to {output_path}")


def plot_fault_sensitivity_combined(
    soft_records: List[Dict[str, object]],
    hard_records: List[Dict[str, object]],
    output_path: str,
    num_gpus: int,
    *,
    normalize: bool = False,
) -> None:
    if not soft_records or not hard_records:
        print("No combined records to plot.", file=sys.stderr)
        return

    if sns is not None:  # pragma: no branch
        sns.set_theme(style="whitegrid")

    soft_map = {}
    for rec in soft_records:
        key = _record_key(rec)
        if key is not None:
            soft_map[key] = rec
    hard_map = {}
    for rec in hard_records:
        key = _record_key(rec)
        if key is not None:
            hard_map[key] = rec
    keys = []
    labels = []
    for rec in soft_records:
        key = _record_key(rec)
        if key is None or key not in hard_map:
            continue
        keys.append(key)
        labels.append(_record_display_label(rec))
    if not keys:
        print("No overlapping configs between soft and hard fault runs.", file=sys.stderr)
        return

    xs = np.arange(len(keys))
    baseline = np.array([soft_map[key]["baseline_runtime"] for key in keys], dtype=float)

    norm_factor = 1.0
    if normalize:
        finite_baseline = baseline[np.isfinite(baseline)]
        norm_factor = float(np.min(finite_baseline)) if finite_baseline.size else 1.0
        if not math.isfinite(norm_factor) or norm_factor <= 0:
            norm_factor = 1.0
        baseline = baseline / norm_factor

    def _stats(records_map):
        mean = np.array([records_map[key]["fault_mean"] for key in keys], dtype=float)
        min_vals = np.array([records_map[key]["fault_min"] for key in keys], dtype=float)
        max_vals = np.array([records_map[key]["fault_max"] for key in keys], dtype=float)
        lower = []
        upper = []
        for mean_val, min_val, max_val in zip(mean, min_vals, max_vals):
            if math.isfinite(mean_val) and math.isfinite(min_val):
                lower.append(max(mean_val - min_val, 0.0))
            else:
                lower.append(0.0)
            if math.isfinite(mean_val) and math.isfinite(max_val):
                upper.append(max(max_val - mean_val, 0.0))
            else:
                upper.append(0.0)
        return mean, min_vals, max_vals, lower, upper

    soft_mean, soft_min, soft_max, soft_lower, soft_upper = _stats(soft_map)
    hard_mean, hard_min, hard_max, hard_lower, hard_upper = _stats(hard_map)
    if normalize:
        soft_mean = soft_mean / norm_factor
        soft_min = soft_min / norm_factor
        soft_max = soft_max / norm_factor
        soft_lower = [val / norm_factor for val in soft_lower]
        soft_upper = [val / norm_factor for val in soft_upper]
        hard_mean = hard_mean / norm_factor
        hard_min = hard_min / norm_factor
        hard_max = hard_max / norm_factor
        hard_lower = [val / norm_factor for val in hard_lower]
        hard_upper = [val / norm_factor for val in hard_upper]

    plt.figure(figsize=(12, 6))
    plt.scatter(xs, baseline, color="#1f78b4", marker="o", s=80, label="Baseline")

    offset = 0.14
    soft_x = xs - offset
    hard_x = xs + offset

    plt.errorbar(
        soft_x,
        soft_mean,
        yerr=[soft_lower, soft_upper],
        fmt="s",
        color="#d7301f",
        ecolor="#fb6a4a",
        elinewidth=2,
        capsize=5,
        capthick=1.4,
        markersize=6,
        label="Soft fault mean ± range",
        alpha=0.9,
    )
    plt.errorbar(
        hard_x,
        hard_mean,
        yerr=[hard_lower, hard_upper],
        fmt="^",
        color="#f1c40f",
        ecolor="#f7dc6f",
        elinewidth=2,
        capsize=5,
        capthick=1.4,
        markersize=6,
        label="Hard fault mean ± range",
        alpha=0.9,
    )

    for x, fmin, fmax in zip(soft_x, soft_min, soft_max):
        if math.isfinite(fmin) and math.isfinite(fmax):
            plt.fill_between([x - 0.16, x + 0.16], [fmin, fmin], [fmax, fmax], color="#fdd0a2", alpha=0.35)
    for x, fmin, fmax in zip(hard_x, hard_min, hard_max):
        if math.isfinite(fmin) and math.isfinite(fmax):
            plt.fill_between([x - 0.16, x + 0.16], [fmin, fmin], [fmax, fmax], color="#fff2b2", alpha=0.35)

    plt.xticks(xs, labels, rotation=45, ha="right", fontsize=16)
    ylabel = "Normalized runtime (fastest=1.0)" if normalize else "Runtime (s)"
    plt.ylabel(ylabel, fontsize=19)
    plt.title(
        f"Soft vs Hard Fault Sensitivity (GLM4.7 [358B,32B active]), Num GPUs = {num_gpus})",
        fontsize=18,
    )
    plt.grid(alpha=0.3, axis="y")
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved combined fault sensitivity plot to {output_path}")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main(cli_args: Optional[Sequence[str]] = None) -> int:
    global TARGET_NUM_GPUS, SAMPLE_COUNT, FAULT_ITER, FAULT_WORKERS, FAULT_MAG
    global NUM_FAULTS, MIN_ALLOWED_TP, MAX_ALLOWED_TP, MIN_ALLOWED_CP, MAX_ALLOWED_CP
    global MIN_ALLOWED_DP, MAX_ALLOWED_DP, MIN_ALLOWED_EP, MAX_ALLOWED_EP, MIN_ALLOWED_PP, MAX_ALLOWED_PP
    global ALLOWED_FAULT_DIMS, PLOT_OUTPUT_PATH, FILTERED_PLOT_OUTPUT_PATH, REPORT_OUTPUT_PATH, RESULTS_JSON_PATH
    global RANDOM_SEED, ENFORCE_SQUARE_TP_CP, DEBUG_MODE, DEBUG_RUN_DIR, DEBUG_SAVED_PATHS
    global _DEBUG_FILE_INDEX, FAILURE_DUMP_DIR, FAILURE_SAVED_PATHS, _FAILURE_FILE_INDEX
    global HARDWARE_CONFIG_PATH, MODEL_CONFIG_PATH, NETWORK_CONFIG_PATH
    global RUN_TYPE, SWEEP_MOE_DP, MOE_MODE, BASE_HW_DICT, PLOT_ONLY, PLOT_NUM, PLOT_SKIP_TOP
    global HARD_PLOT_OUTPUT_PATH, HARD_REPORT_OUTPUT_PATH, COMBINED_PLOT_OUTPUT_PATH
    global HARD_FILTERED_PLOT_OUTPUT_PATH, COMBINED_FILTERED_PLOT_OUTPUT_PATH

    args = parse_args(cli_args)

    HARDWARE_CONFIG_PATH = args.hardware_config
    MODEL_CONFIG_PATH = args.model_config
    NETWORK_CONFIG_PATH = args.network_config
    PLOT_ONLY = bool(args.plot_only)

    TARGET_NUM_GPUS = args.target_num_gpus
    SAMPLE_COUNT = args.sample_count
    FAULT_ITER = args.fault_iter
    FAULT_WORKERS = args.fault_workers

    mag_mean, mag_std = _parse_float_tuple(args.fault_mag)
    FAULT_MAG = (mag_mean, mag_std)

    parsed_num_faults = _parse_int_list(args.num_faults)
    NUM_FAULTS = parsed_num_faults or [1]

    MIN_ALLOWED_TP = max(1, args.min_tp)
    MAX_ALLOWED_TP = max(MIN_ALLOWED_TP, args.max_tp)
    MIN_ALLOWED_CP = max(1, args.min_cp)
    MAX_ALLOWED_CP = max(MIN_ALLOWED_CP, args.max_cp)
    MIN_ALLOWED_DP = max(1, args.min_dp)
    MAX_ALLOWED_DP = max(MIN_ALLOWED_DP, args.max_dp)
    MIN_ALLOWED_EP = max(1, args.min_ep)
    MAX_ALLOWED_EP = max(MIN_ALLOWED_EP, args.max_ep)
    MIN_ALLOWED_PP = max(1, args.min_pp)
    MAX_ALLOWED_PP = max(MIN_ALLOWED_PP, args.max_pp)

    allowed_dims_list = _parse_int_list(args.allowed_fault_dims) if args.allowed_fault_dims else []
    ALLOWED_FAULT_DIMS = allowed_dims_list if allowed_dims_list else None

    PLOT_OUTPUT_PATH = args.plot_output
    FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(PLOT_OUTPUT_PATH, "filtered")
    REPORT_OUTPUT_PATH = args.report_output
    RESULTS_JSON_PATH = Path(args.results_json)
    HARD_PLOT_OUTPUT_PATH = _suffix_output_path(PLOT_OUTPUT_PATH, "hard")
    HARD_FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(HARD_PLOT_OUTPUT_PATH, "filtered")
    COMBINED_PLOT_OUTPUT_PATH = _suffix_output_path(PLOT_OUTPUT_PATH, "combined")
    COMBINED_FILTERED_PLOT_OUTPUT_PATH = _tag_output_path(COMBINED_PLOT_OUTPUT_PATH, "filtered")
    HARD_REPORT_OUTPUT_PATH = _suffix_output_path(REPORT_OUTPUT_PATH, "hard")
    PLOT_NUM = max(0, int(getattr(args, "plot_num", 0) or 0))
    PLOT_SKIP_TOP = max(0, int(getattr(args, "plot_skip_top", 0) or 0))

    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    ENFORCE_SQUARE_TP_CP = ENFORCE_SQUARE_TP_CP or bool(args.enforce_square_tp_cp)
    DEBUG_MODE = bool(args.debug_mode)
    _DEBUG_FILE_INDEX = 0
    _FAILURE_FILE_INDEX = 0
    FAILURE_SAVED_PATHS.clear()
    FAILURE_DUMP_DIR = None
    if DEBUG_MODE:
        FAULT_WORKERS = 1
        DEBUG_SAVED_PATHS.clear()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        DEBUG_RUN_DIR = DEBUG_OUTPUT_BASE / timestamp
        DEBUG_RUN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled. Hardware configs will be written to {DEBUG_RUN_DIR}")
    else:
        DEBUG_RUN_DIR = None
        DEBUG_SAVED_PATHS.clear()

    set_astrasim_cache_mode(ASTRA_CACHE_MODE)
    _install_signal_handlers()
    _run_fault_injection_self_test()

    mode = determine_model_mode(MODEL_CONFIG_PATH)
    model_config_obj = config.parse_config(MODEL_CONFIG_PATH, config_type=mode)
    RUN_TYPE = _model_run_type(model_config_obj)
    SWEEP_MOE_DP = RUN_TYPE == "inference"
    MOE_MODE = bool(_moe_num_experts(model_config_obj))

    if PLOT_ONLY:
        records = read_fault_report(REPORT_OUTPUT_PATH)
        filtered_records: List[Dict[str, object]] = []
        if records:
            plot_fault_sensitivity(records, PLOT_OUTPUT_PATH, TARGET_NUM_GPUS)
            if PLOT_NUM > 0:
                filtered_records = _select_top_by_baseline(records, PLOT_NUM, skip_top=PLOT_SKIP_TOP)
                _print_top_records(filtered_records, "soft faults", skip_top=PLOT_SKIP_TOP)
                plot_fault_sensitivity(
                    filtered_records,
                    FILTERED_PLOT_OUTPUT_PATH,
                    TARGET_NUM_GPUS,
                    normalize=True,
                )
        else:
            print("No records found for plot-only mode (soft faults).", file=sys.stderr)
        hard_records: List[Dict[str, object]] = []
        filtered_hard_records: List[Dict[str, object]] = []
        if _hard_faults_enabled() and os.path.exists(HARD_REPORT_OUTPUT_PATH):
            hard_records = read_fault_report(HARD_REPORT_OUTPUT_PATH)
            if hard_records:
                plot_fault_sensitivity(
                    hard_records,
                    HARD_PLOT_OUTPUT_PATH,
                    TARGET_NUM_GPUS,
                    fault_color="#f1c40f",
                    fault_range_color="#f7dc6f",
                    fault_fill_color="#fff2b2",
                    fault_label="Hard fault mean ± range",
                )
                if PLOT_NUM > 0 and filtered_records:
                    keys = []
                    for record in filtered_records:
                        key = _record_key(record)
                        if key is not None:
                            keys.append(key)
                    filtered_hard_records = _filter_records_by_keys(hard_records, keys)
                    if filtered_hard_records:
                        sorted_filtered_hard = sorted(filtered_hard_records, key=_baseline_sort_key)
                        plot_fault_sensitivity(
                            sorted_filtered_hard,
                            HARD_FILTERED_PLOT_OUTPUT_PATH,
                            TARGET_NUM_GPUS,
                            fault_color="#f1c40f",
                            fault_range_color="#f7dc6f",
                            fault_fill_color="#fff2b2",
                            fault_label="Hard fault mean ± range",
                            normalize=True,
                        )
            else:
                print("No records found for plot-only mode (hard faults).", file=sys.stderr)
        if records and hard_records:
            plot_fault_sensitivity_combined(records, hard_records, COMBINED_PLOT_OUTPUT_PATH, TARGET_NUM_GPUS)
        if filtered_records and filtered_hard_records:
            plot_fault_sensitivity_combined(
                filtered_records,
                filtered_hard_records,
                COMBINED_FILTERED_PLOT_OUTPUT_PATH,
                TARGET_NUM_GPUS,
                normalize=True,
            )
        return 0

    base_hw_dict = read_yaml(HARDWARE_CONFIG_PATH)
    _apply_network_override(base_hw_dict, NETWORK_CONFIG_PATH)
    if RUN_TYPE == "inference" and SWEEP_MOE_DP:
        network_block = base_hw_dict.get("network")
        if isinstance(network_block, dict):
            _ensure_ep_in_network(network_block)
    BASE_HW_DICT = base_hw_dict

    # existing_results = _load_existing_results(RESULTS_JSON_PATH)
    # if existing_results:
    #     print(f"Loaded {len(existing_results)} cached result entries from {RESULTS_JSON_PATH}")
    existing_results = {}
    results_store: Dict[str, Dict[str, object]] = dict(existing_results)

    parallelism_samples = generate_parallelism_samples(
        TARGET_NUM_GPUS, SAMPLE_COUNT, base_hw_dict, model_config_obj, mode, workers=FAULT_WORKERS
    )
    if not parallelism_samples:
        print("Unable to generate parallelism samples.", file=sys.stderr)
        return

    records: List[Dict[str, object]] = []
    records_by_key: Dict[Tuple[Tuple[str, object], ...], Dict[str, object]] = {}
    baseline_task_queue: List[Dict[str, object]] = []
    debug_task_queue: Optional[List[Dict[str, object]]] = [] if DEBUG_MODE else None

    def _enqueue_fault_debug_tasks(
        *,
        settings_copy: Dict[str, int],
        settings_key: Tuple[Tuple[str, object], ...],
        key: Tuple[Tuple[str, object], ...],
        candidates_for_config: List[Tuple[int, int]],
        fault_mode: str,
        fault_iter: int,
        num_faults_list: Sequence[int],
        fault_sampler,
    ) -> None:
        if debug_task_queue is None:
            return
        if not num_faults_list or fault_iter <= 0:
            return
        for num_faults in num_faults_list:
            repeats = max(1, int(num_faults))
            for iter_idx in range(fault_iter):
                fault_specs: List[Tuple[int, float, int]] = []
                for _ in range(repeats):
                    fault_value = float(fault_sampler())
                    dim_choice, node_count = random.choice(candidates_for_config)
                    fault_specs.append((dim_choice, fault_value, node_count))
                fault_task_id = _task_identifier(
                    "fault",
                    settings_key,
                    faults=fault_specs,
                    num_faults=repeats,
                    fault_iter=iter_idx,
                    fault_mode=fault_mode,
                )
                cached_fault = existing_results.get(fault_task_id)
                if cached_fault is not None:
                    continue
                debug_task_queue.append(
                    {
                        "kind": "fault",
                        "fault_mode": fault_mode,
                        "settings": settings_copy,
                        "key": key,
                        "settings_key": settings_key,
                        "task_id": fault_task_id,
                        "faults": fault_specs,
                        "num_faults": repeats,
                        "fault_iter": iter_idx,
                    }
                )

    for settings in parallelism_samples:
        settings_copy = dict(settings)
        key = _parallelism_key(settings_copy)
        settings_key = key

        candidates_for_config = _candidate_fault_dimensions(settings_copy)
        if not candidates_for_config:
            continue

        records_by_key[key] = {
            "parallelism": settings_copy,
            "baseline_runtime": None,
            "fault_runtimes": [],
            "fault_details": [],
            "hard_fault_runtimes": [],
            "hard_fault_details": [],
            "memory_exceeded": False,
            "memory_violation_gb": 0.0,
        }

        baseline_task_id = _task_identifier("baseline", settings_key)
        baseline_cached = existing_results.get(baseline_task_id)
        if baseline_cached is not None:
            runtime = baseline_cached.get("runtime")
            if runtime is not None:
                records_by_key[key]["baseline_runtime"] = float(runtime)
            mem_flag = bool(baseline_cached.get("memory_exceeded", False))
            records_by_key[key]["memory_exceeded"] = mem_flag
            records_by_key[key]["memory_violation_gb"] = float(
                baseline_cached.get("memory_violation_gb", 0.0) or 0.0
            )
        else:
            baseline_task_queue.append(
                {
                    "kind": "baseline",
                    "settings": settings_copy,
                    "key": key,
                    "settings_key": settings_key,
                    "task_id": baseline_task_id,
                }
            )
        if debug_task_queue is not None:
            _enqueue_fault_debug_tasks(
                settings_copy=settings_copy,
                settings_key=settings_key,
                key=key,
                candidates_for_config=candidates_for_config,
                fault_mode="soft",
                fault_iter=FAULT_ITER,
                num_faults_list=NUM_FAULTS,
                fault_sampler=sample_fault_magnitude,
            )
            if _hard_faults_enabled():
                _enqueue_fault_debug_tasks(
                    settings_copy=settings_copy,
                    settings_key=settings_key,
                    key=key,
                    candidates_for_config=candidates_for_config,
                    fault_mode="hard",
                    fault_iter=HARD_FAULT_ITER,
                    num_faults_list=HARD_NUM_FAULTS,
                    fault_sampler=sample_hard_fault_magnitude,
                )

    if DEBUG_MODE:
        debug_tasks: List[Dict[str, object]] = []
        if baseline_task_queue:
            debug_tasks.extend(baseline_task_queue)
        if debug_task_queue:
            debug_tasks.extend(debug_task_queue)
        if not debug_tasks:
            print("Debug mode enabled but no evaluation tasks were generated.")
        else:
            for task in debug_tasks:
                yaml_text = _generate_hw_yaml_from_task(base_hw_dict, task)
                debug_entry = {
                    "kind": task.get("kind"),
                    "settings": task.get("settings"),
                    "num_faults": task.get("num_faults"),
                    "faults": task.get("faults"),
                    "hw_yaml": yaml_text,
                }
                _dump_debug_hw_config(debug_entry)
        if DEBUG_RUN_DIR:
            if DEBUG_SAVED_PATHS:
                print(f"Debug mode: saved {len(DEBUG_SAVED_PATHS)} hardware config(s) to {DEBUG_RUN_DIR}")
            else:
                print(f"Debug mode enabled but no hardware configs were generated. Directory: {DEBUG_RUN_DIR}")
        else:
            print("Debug mode enabled but no output directory was initialised.")
        return 0

    def handle_task_result(result: Dict[str, object]) -> None:
        if not isinstance(result, dict):
            return
        key = result.get("key")
        rec = records_by_key.get(key)
        if rec is None:
            return
        status = result.get("status")
        if status != "ok":
            error_msg = result.get("error")
            print(
                f"Warning: task failed for settings {result.get('settings')}: {error_msg}",
                file=sys.stderr,
            )
            _dump_failure_hw_config(base_hw_dict, result, error_msg)
            return
        kind = result.get("kind")
        runtime = float(result.get("runtime", float("nan")))
        memory_exceeded = bool(result.get("memory_exceeded", False))
        memory_violation = float(result.get("memory_violation_gb", 0.0) or 0.0)
        if memory_exceeded:
            already_marked = rec.get("memory_exceeded", False)
            rec["memory_exceeded"] = True
            rec["memory_violation_gb"] = memory_violation
            if not already_marked:
                print(
                    f"Skipping configuration {result.get('settings')} due to memory capacity violation "
                    f"({memory_violation:.3f} GB).",
                    file=sys.stderr,
                )
            _record_partial_result(results_store, result)
            return
        if kind == "baseline":
            rec["baseline_runtime"] = runtime
        else:
            fault_mode = str(result.get("fault_mode") or "soft")
            runtimes_key = "hard_fault_runtimes" if fault_mode == "hard" else "fault_runtimes"
            details_key = "hard_fault_details" if fault_mode == "hard" else "fault_details"
            rec[runtimes_key].append(runtime)
            rec[details_key].append(
                {
                    "num_faults": result.get("num_faults"),
                    "faults": _normalise_faults_raw(result.get("faults", [])),
                }
            )
        _record_partial_result(results_store, result)
        _dump_debug_hw_config(result)

    def run_task_queue(
        task_queue: List[Dict[str, object]],
        desc: str,
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[Dict[str, object]]:
        if not task_queue:
            return []
        results: List[Dict[str, object]] = []
        if executor is not None:
            global _ACTIVE_EXECUTOR
            _ACTIVE_EXECUTOR = executor
            futures = [executor.submit(_parallelism_worker_task, task) for task in task_queue]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="task"):
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Warning: parallel task raised an exception: {exc}", file=sys.stderr)
                    continue
                results.append(result)
        else:
            if FAULT_WORKERS and FAULT_WORKERS > 1:
                available_cpus = max(1, os.cpu_count() or 1)
                worker_count = min(max(1, FAULT_WORKERS), len(task_queue), available_cpus)
                print(f"Launching ProcessPoolExecutor with {worker_count} worker(s) for {desc}.")
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_parallelism_worker_init,
                    initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode, RUN_TYPE, SWEEP_MOE_DP),
                )
                try:
                    _ACTIVE_EXECUTOR = executor
                    futures = [executor.submit(_parallelism_worker_task, task) for task in task_queue]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="task"):
                        try:
                            result = fut.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            print(f"Warning: parallel task raised an exception: {exc}", file=sys.stderr)
                            continue
                        results.append(result)
                finally:
                    _terminate_active_executor()
            else:
                print(f"Running {desc} sequentially (FAULT_WORKERS <= 1).")
                for task in tqdm(task_queue, desc=desc, unit="task"):
                    try:
                        result = _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"Warning: sequential task raised an exception: {exc}", file=sys.stderr)
                        continue
                    results.append(result)
        return results

    if baseline_task_queue:
        print(f"Prepared {len(baseline_task_queue)} baseline task(s).")
        baseline_results = run_task_queue(baseline_task_queue, "Baseline evaluations")
        for result in baseline_results:
            handle_task_result(result)
    else:
        print("No new baseline tasks; using cached results where available.")

    eligible_configs = []
    for settings in parallelism_samples:
        key = _parallelism_key(settings)
        state = records_by_key.get(key) or {}
        if state.get("memory_exceeded") or state.get("baseline_runtime") is None:
            continue
        candidates_for_config = _candidate_fault_dimensions(settings)
        if not candidates_for_config:
            continue
        eligible_configs.append((settings, key, candidates_for_config))

    def _run_fault_phase(
        *,
        fault_mode: str,
        fault_iter: int,
        num_faults_list: Sequence[int],
        fault_sampler,
    ) -> None:
        if not num_faults_list or fault_iter <= 0:
            print(f"No {fault_mode} fault iterations configured; skipping {fault_mode} fault evaluations.")
            return
        if not eligible_configs:
            print(f"No eligible configs for {fault_mode} fault evaluations.", file=sys.stderr)
            return
        print(
            f"Beginning {fault_mode} fault evaluations with fairness policy "
            f"(max retries per iteration: {MAX_ATTEMPTS})."
        )

        fault_executor: Optional[ProcessPoolExecutor] = None
        fault_worker_count = 0
        progress_bar = None
        try:
            if eligible_configs and FAULT_WORKERS and FAULT_WORKERS > 1:
                available_cpus = max(1, os.cpu_count() or 1)
                fault_worker_count = min(max(1, FAULT_WORKERS), fault_iter, available_cpus)
                if fault_worker_count > 1:
                    print(
                        f"Initializing shared ProcessPoolExecutor with {fault_worker_count} worker(s) "
                        f"for {fault_mode} fault iterations."
                    )
                    fault_executor = ProcessPoolExecutor(
                        max_workers=fault_worker_count,
                        initializer=_parallelism_worker_init,
                        initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode, RUN_TYPE, SWEEP_MOE_DP),
                    )
                    _ACTIVE_EXECUTOR = fault_executor

            def _build_fault_iteration(
                iter_idx: int,
                repeats: int,
            ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
                fault_task_queue: List[Dict[str, object]] = []
                pending_results: List[Dict[str, object]] = []
                for settings, key, candidates_for_config in eligible_configs:
                    fault_specs: List[Tuple[int, float, int]] = []
                    for _ in range(repeats):
                        fault_value = float(fault_sampler())
                        dim_choice, node_count = random.choice(candidates_for_config)
                        fault_specs.append((dim_choice, fault_value, node_count))

                    fault_task_id = _task_identifier(
                        "fault",
                        key,
                        faults=fault_specs,
                        num_faults=repeats,
                        fault_iter=iter_idx,
                        fault_mode=fault_mode,
                    )
                    cached_fault = existing_results.get(fault_task_id)
                    if cached_fault is not None:
                        if bool(cached_fault.get("memory_exceeded", False)):
                            pending_results.append(
                                {
                                    "status": "error",
                                    "kind": "fault",
                                    "fault_mode": fault_mode,
                                    "key": key,
                                    "settings": settings,
                                    "settings_key": key,
                                    "task_id": fault_task_id,
                                    "memory_exceeded": True,
                                    "memory_violation_gb": cached_fault.get("memory_violation_gb", 0.0),
                                }
                            )
                            continue
                        runtime = cached_fault.get("runtime")
                        if runtime is not None:
                            pending_results.append(
                                {
                                    "status": "ok",
                                    "kind": "fault",
                                    "fault_mode": fault_mode,
                                    "key": key,
                                    "settings": settings,
                                    "settings_key": key,
                                    "task_id": fault_task_id,
                                    "runtime": float(runtime),
                                    "faults": cached_fault.get("faults", fault_specs),
                                    "num_faults": cached_fault.get("num_faults", repeats),
                                    "fault_iter": iter_idx,
                                    "memory_exceeded": False,
                                    "memory_violation_gb": float(cached_fault.get("memory_violation_gb", 0.0) or 0.0),
                                }
                            )
                        continue

                    fault_task_queue.append(
                        {
                            "kind": "fault",
                            "fault_mode": fault_mode,
                            "settings": settings,
                            "key": key,
                            "settings_key": key,
                            "task_id": fault_task_id,
                            "faults": fault_specs,
                            "num_faults": repeats,
                            "fault_iter": iter_idx,
                        }
                    )
                return fault_task_queue, pending_results

            def _start_fault_iteration(iter_idx: int, repeats: int, attempt: int) -> None:
                task_queue, cached_results = _build_fault_iteration(iter_idx, repeats)
                if not task_queue and not cached_results:
                    if progress_bar is not None:
                        progress_bar.update(1)
                    return
                iteration_key = (iter_idx, repeats)
                iteration_state[iteration_key] = {
                    "attempt": attempt,
                    "remaining": len(task_queue),
                    "results": list(cached_results),
                }
                if not task_queue:
                    _finalise_iteration(iteration_key)
                    return
                for task in task_queue:
                    if fault_executor is not None:
                        fut = fault_executor.submit(_parallelism_worker_task, task)
                        in_flight[fut] = iteration_key
                    else:
                        try:
                            result = _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task)
                        except Exception as exc:  # pragma: no cover - defensive
                            result = {
                                "status": "error",
                                "kind": "fault",
                                "fault_mode": fault_mode,
                                "key": task.get("key"),
                                "settings": task.get("settings"),
                                "settings_key": task.get("settings_key"),
                                "task_id": task.get("task_id"),
                                "error": str(exc),
                            }
                        iteration_state[iteration_key]["results"].append(result)
                        iteration_state[iteration_key]["remaining"] -= 1
                if fault_executor is None:
                    _finalise_iteration(iteration_key)

            def _finalise_iteration(iteration_key: Tuple[int, int]) -> None:
                state = iteration_state.get(iteration_key)
                if state is None or state["remaining"] > 0:
                    return
                iter_idx, repeats = iteration_key
                all_results = state.get("results", [])
                failed = any(
                    (res.get("status") != "ok") or bool(res.get("memory_exceeded", False))
                    for res in all_results
                )
                if failed:
                    last_error = None
                    for res in reversed(all_results):
                        if (res.get("status") != "ok") or bool(res.get("memory_exceeded", False)):
                            if res.get("error"):
                                last_error = str(res.get("error"))
                                _dump_failure_hw_config(base_hw_dict, res, res.get("error"))
                            elif res.get("memory_exceeded"):
                                last_error = (
                                    f"memory exceeded ({float(res.get('memory_violation_gb', 0.0) or 0.0):.3f} GB over)"
                                )
                            else:
                                last_error = "unknown error"
                            break
                    attempt = state.get("attempt", 1)
                    if attempt >= MAX_ATTEMPTS:
                        msg = (
                            f"Omitting {fault_mode} fault iteration {iter_idx} for num_faults={repeats} "
                            f"across all configs after {attempt} failed attempt(s)."
                        )
                        if last_error:
                            msg += f" Last error: {last_error}"
                        print(msg, file=sys.stderr)
                        if progress_bar is not None:
                            progress_bar.update(1)
                    else:
                        pending_iterations.append((iter_idx, repeats, attempt + 1))
                    iteration_state.pop(iteration_key, None)
                    return
                for res in all_results:
                    handle_task_result(res)
                iteration_state.pop(iteration_key, None)
                if progress_bar is not None:
                    progress_bar.update(1)

            if eligible_configs and num_faults_list and fault_iter > 0:
                pending_iterations: deque[Tuple[int, int, int]] = deque()
                for num_faults in num_faults_list:
                    repeats = max(1, int(num_faults))
                    for iter_idx in range(fault_iter):
                        pending_iterations.append((iter_idx, repeats, 1))

                in_flight: Dict[object, Tuple[int, int]] = {}
                iteration_state: Dict[Tuple[int, int], Dict[str, object]] = {}
                backlog_target = max(1, (fault_worker_count or 1) * 4)
                total_iterations = len(pending_iterations)
                progress_bar = tqdm(
                    total=total_iterations,
                    desc=f"{fault_mode.capitalize()} fault iterations",
                    unit="iter",
                )

                while pending_iterations or in_flight:
                    while pending_iterations and (fault_executor is None or len(in_flight) < backlog_target):
                        iter_idx, repeats, attempt = pending_iterations.popleft()
                        _start_fault_iteration(iter_idx, repeats, attempt)

                    if fault_executor is None:
                        if not pending_iterations:
                            break
                        continue

                    if not in_flight:
                        continue
                    try:
                        next_future = next(as_completed(list(in_flight.keys()), timeout=None))
                    except StopIteration:
                        continue
                    iteration_key = in_flight.pop(next_future, None)
                    if iteration_key is None:
                        continue
                    try:
                        result = next_future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        result = {"status": "error", "error": str(exc), "fault_mode": fault_mode}
                    state = iteration_state.get(iteration_key)
                    if state is not None:
                        state["results"].append(result)
                        state["remaining"] = max(0, int(state["remaining"]) - 1)
                        _finalise_iteration(iteration_key)
        finally:
            try:
                if progress_bar is not None:
                    progress_bar.close()
            except Exception:
                pass
            if fault_executor is not None:
                _ACTIVE_EXECUTOR = fault_executor
                _terminate_active_executor()

    _run_fault_phase(
        fault_mode="soft",
        fault_iter=FAULT_ITER,
        num_faults_list=NUM_FAULTS,
        fault_sampler=sample_fault_magnitude,
    )
    if _hard_faults_enabled():
        _run_fault_phase(
            fault_mode="hard",
            fault_iter=HARD_FAULT_ITER,
            num_faults_list=HARD_NUM_FAULTS,
            fault_sampler=sample_hard_fault_magnitude,
        )
    else:
        print("Hard faults disabled; skipping hard fault evaluations.")

    def _build_fault_records(runtime_key: str) -> List[Dict[str, object]]:
        built: List[Dict[str, object]] = []
        for settings in parallelism_samples:
            key = _parallelism_key(settings)
            state = records_by_key.get(key)
            if not state:
                continue
            if state.get("memory_exceeded"):
                continue
            baseline_runtime = state.get("baseline_runtime")
            if baseline_runtime is None:
                print(f"Warning: missing baseline runtime for settings {settings}", file=sys.stderr)
                continue
            fault_min, fault_max, fault_mean = summarise_fault_runs(state.get(runtime_key, []))
            record = {
                "parallelism": state["parallelism"],
                "baseline_runtime": float(baseline_runtime),
                "fault_min": fault_min,
                "fault_max": fault_max,
                "fault_mean": fault_mean,
                "label": _format_parallelism_label(settings),
            }
            built.append(record)
        return built

    records = _build_fault_records("fault_runtimes")
    hard_records: List[Dict[str, object]] = []
    if _hard_faults_enabled():
        hard_records = _build_fault_records("hard_fault_runtimes")

    if not records:
        print("No successful sweep results collected for soft faults.", file=sys.stderr)
        return 0

    write_fault_report(records, REPORT_OUTPUT_PATH)
    print(f"Wrote fault sweep report to {REPORT_OUTPUT_PATH}")
    plot_fault_sensitivity(records, PLOT_OUTPUT_PATH, TARGET_NUM_GPUS)
    filtered_records: List[Dict[str, object]] = []
    if PLOT_NUM > 0:
        filtered_records = _select_top_by_baseline(records, PLOT_NUM, skip_top=PLOT_SKIP_TOP)
        _print_top_records(filtered_records, "soft faults", skip_top=PLOT_SKIP_TOP)
        plot_fault_sensitivity(
            filtered_records,
            FILTERED_PLOT_OUTPUT_PATH,
            TARGET_NUM_GPUS,
            normalize=True,
        )

    if hard_records:
        write_fault_report(hard_records, HARD_REPORT_OUTPUT_PATH)
        print(f"Wrote hard fault sweep report to {HARD_REPORT_OUTPUT_PATH}")
        plot_fault_sensitivity(
            hard_records,
            HARD_PLOT_OUTPUT_PATH,
            TARGET_NUM_GPUS,
            fault_color="#f1c40f",
            fault_range_color="#f7dc6f",
            fault_fill_color="#fff2b2",
            fault_label="Hard fault mean ± range",
        )
        filtered_hard_records: List[Dict[str, object]] = []
        if PLOT_NUM > 0 and filtered_records:
            keys = []
            for record in filtered_records:
                key = _record_key(record)
                if key is not None:
                    keys.append(key)
            filtered_hard_records = _filter_records_by_keys(hard_records, keys)
            if filtered_hard_records:
                sorted_filtered_hard = sorted(filtered_hard_records, key=_baseline_sort_key)
                plot_fault_sensitivity(
                    sorted_filtered_hard,
                    HARD_FILTERED_PLOT_OUTPUT_PATH,
                    TARGET_NUM_GPUS,
                    fault_color="#f1c40f",
                    fault_range_color="#f7dc6f",
                    fault_fill_color="#fff2b2",
                    fault_label="Hard fault mean ± range",
                    normalize=True,
                )
        plot_fault_sensitivity_combined(records, hard_records, COMBINED_PLOT_OUTPUT_PATH, TARGET_NUM_GPUS)
        if filtered_records and filtered_hard_records:
            plot_fault_sensitivity_combined(
                filtered_records,
                filtered_hard_records,
                COMBINED_FILTERED_PLOT_OUTPUT_PATH,
                TARGET_NUM_GPUS,
                normalize=True,
            )
    else:
        print("No hard fault records to report.", file=sys.stderr)
    if FAILURE_SAVED_PATHS and FAILURE_DUMP_DIR:
        print(
            f"Captured {len(FAILURE_SAVED_PATHS)} failing hardware config artefact(s) in {FAILURE_DUMP_DIR}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _terminate_active_executor()
        print("Interrupted; shutting down worker processes.", file=sys.stderr)
        sys.exit(130)
