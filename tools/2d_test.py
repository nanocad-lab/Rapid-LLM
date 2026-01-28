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
2D topology test harness for RAPID-LLM (training + inference).

Runs Mesh2D/Torus2D/KingMesh2D/HyperCube comparisons with and without GMap optimization,
prints tabular results, and emits bar plots.
"""

import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set()
sns.set_context("notebook", font_scale=1.5)

import config  # noqa: E402
from validation_scripts import huggingface_bench_validation as hfval  # noqa: E402
from astrasim_lib import ensure_chakra_available  # noqa: E402
from inference_timing import TimeCalculationLLMInference  # noqa: E402
from train_timing import TimeCalculationLLM  # noqa: E402

H100_GPU = True

if H100_GPU:
    TRAIN_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB_2d.yaml"
    INF_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB_2d.yaml"
else:
    TRAIN_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/a100_80GB_2d_gmap_train.yaml"
    INF_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/a100_80GB_2d_gmap_inf.yaml"

TRAIN_70B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/Llama3.1-70B_2d_train.yaml"
# TRAIN_GPT175B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/GPT_175_B_2d_train.yaml"
TRAIN_GLM45_106B_MODEL_CONFIG = "configs/model-config/GLM_4.5_AIR_106B.yaml"
INF_70B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/Llama3.1-70B_2d_inf.yaml"
# INF_GPT175B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/GPT_175_B_2d_inf.yaml"
INF_GLM45_106B_MODEL_CONFIG = "configs/model-config/GLM_4.5_AIR_106B_inf.yaml"

OUTPUT_ROOT = Path("output") / "2d_test"
PLOT_DIR = OUTPUT_ROOT
Y_LABEL = "Normalized batch runtime (s)"
CACHE_VERSION = 6
CACHE_PATH = OUTPUT_ROOT / "2d_test_cache.json"
VERBOSE = os.environ.get("RAPID_2D_TEST_VERBOSE", "1") != "0"
AUTO_PARALLELISM = True
PP_MAX = 8
os.environ.setdefault("RAPID_ALLOW_MOE_EXPERT_PADDING", "1")

NUM_WORKERS = int(os.environ.get("RAPID_2D_TEST_WORKERS", max(1, (os.cpu_count() or 1) - 1)))
hfval.ASTRA_CACHE_MODE = "NO_CACHE"
hfval.ASTRA_TMP_ROOT = Path("tmp") / "2d_test_runs"
hfval.CLEANUP_ASTRA_TMP = True

# COMPARE_TOPOLOGIES = ("Torus2D", "KingMesh2D", "HyperCube")
# Compare these topologies against Mesh2D in plots/tables.
COMPARE_TOPOLOGIES = ("Torus2D", "FullyConnected")
# If True, scale FullyConnected per-link BW to match Mesh2D total injection.
FC_FAIR = False
FC_FAIR_MESH_EDGES = 4

TOPOLOGIES = ("Mesh2D",) + COMPARE_TOPOLOGIES
COMP_TOPO = COMPARE_TOPOLOGIES


COMP_TOPO_SPECS = {
    "Torus2D": {
        "output": "2d_test_super_merged.png",
        "ylabel": "2D Torus over Mesh speedup",
        "title": "2D Torus vs Mesh runtime comparison",
        "dynamic_ylim": True,
    },
    "KingMesh2D": {
        "output": "2d_test_super_merged_kingmesh.png",
        "ylabel": "2D KingMesh over Mesh speedup",
        "title": "2D KingMesh vs Mesh runtime comparison",
        "dynamic_ylim": True,
    },
    "HyperCube": {
        "output": "2d_test_super_merged_hypercube.png",
        "ylabel": "HyperCube over Mesh speedup",
        "title": "HyperCube vs Mesh runtime comparison",
        "dynamic_ylim": True,
    },
    "FullyConnected": {
        "output": "2d_test_super_merged_fullyconnected.png",
        "ylabel": "Fully Connected over Mesh speedup",
        "title": "Fully Connected vs Mesh runtime comparison",
        "dynamic_ylim": True,
    },
}
# TOPOLOGIES = ("Mesh2D", "Torus2D")

BW_SWEEP_GBPS = (50,200,325,750,1500)
BW_SWEEP_LABELS = {
    10: "10 GB/s",
    25: "25 GB/s",
    50: "50 GB/s",
    100: "100 GB/s",
    200: "200 GB/s",
    325: "325 GB/s",
    750: "750 GB/s",
    1500: "1.5 TB/s",
}
# TRAIN_SHAPES = ((8, 8), (4, 16))
# TRAIN_MODELS = (
#     {
#         "label": "70B",
#         "config": TRAIN_70B_MODEL_CONFIG,
#         "parallelisms": (
#             {"tp": 32, "cp": 1, "pp": 2},
#             {"tp": 16, "cp": 1, "pp": 4},
#         ),
#         "axes": ("tp", "pp"),
#     },
#     {
#         "label": "GPT175B",
#         "config": TRAIN_GPT175B_MODEL_CONFIG,
#         "parallelisms": (
#             {"tp": 32, "cp": 1, "pp": 2},
#             {"tp": 16, "cp": 1, "pp": 4},
#         ),
#         "axes": ("tp", "pp"),
#     },
# )
TRAIN_SHAPES = ((4, 5), (4, 6), (4, 8), (6, 6))
TRAIN_MODELS = (
    {
        "label": "70B",
        "config": TRAIN_70B_MODEL_CONFIG,
        "parallelisms": (
            {"tp": 10, "cp": 1, "pp": 2},  #4,5 = 20 = 10 * 2
            {"tp": 10, "cp": 1, "pp": 3},  #5,6 = 30 = 3 * 10
            {"tp": 8, "cp": 1, "pp": 4},  #6,6 = 48 = 8 * 6
            {"tp": 6, "cp": 1, "pp": 6},  #4,9 = 72 = 8 * 9
        ),
        "axes": ("tp", "pp"),
    },
    {
        "label": "GLM4.5_106B",
        "config": TRAIN_GLM45_106B_MODEL_CONFIG,
        "parallelisms": (
            {"tp": 10, "cp": 1, "pp": 2, "ep": 1},  #4,5 = 20 = 10 * 2
            {"tp": 10, "cp": 1, "pp": 3, "ep": 1},  #5,6 = 30 = 3 * 10
            {"tp": 16, "cp": 1, "pp": 2, "ep": 1},  #4,8 = 32 = 16 * 2
            {"tp": 12, "cp": 1, "pp": 3, "ep": 1},  #6,6 = 36 = 12 * 3
        ),
        "axes": ("tp", "ep", "pp"),
    },
    # {
    #     "label": "GPT175B",
    #     "config": TRAIN_GPT175B_MODEL_CONFIG,
    #     "parallelisms": (
    #         {"tp": 10, "cp": 1, "pp": 2},  #4,5 = 20 = 10 * 2
    #         {"tp": 10, "cp": 1, "pp": 3},  #5,6 = 30 = 3 * 10
    #         {"tp": 8, "cp": 1, "pp": 4},  #6,6 = 48 = 8 * 6
    #         {"tp": 6, "cp": 1, "pp": 6},  #4,9 = 72 = 8 * 9
    #     ),
    #     "axes": ("tp", "pp"),
    # },
)
INF_SHAPES = ((4, 5), (4, 6), (4, 8), (6, 6))
INF_MODELS = (
    {"label": "70B", "config": INF_70B_MODEL_CONFIG},
    {"label": "GLM4.5_106B", "config": INF_GLM45_106B_MODEL_CONFIG},
    # {"label": "GPT175B", "config": INF_GPT175B_MODEL_CONFIG},
)

MODEL_DISPLAY_NAMES = {
    "70B": "Llama 3 70B",
    "GLM4.5_106B": "GLM 4.5 106B",
    # "GPT175B": "GPT 175B",
}

_WORKER_BASE_HW: Optional[Dict[str, Any]] = None


def _init_worker(base_hw_config: Dict[str, Any]) -> None:
    hfval._worker_init(base_hw_config)
    global _WORKER_BASE_HW
    _WORKER_BASE_HW = base_hw_config


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _file_signature(path: str) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha1(payload).hexdigest()


def determine_model_mode(model_path: str) -> str:
    model_dict = read_yaml(model_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError(f"model_param.mode must be defined in {model_path}")
    return str(mode)


_MODEL_MOE_CACHE: Dict[str, bool] = {}


def _is_moe_model(model_path: str) -> bool:
    cached = _MODEL_MOE_CACHE.get(model_path)
    if cached is not None:
        return cached
    model_dict = read_yaml(model_path)
    model_param = model_dict.get("model_param") or {}
    moe_cfg = model_param.get("moe") or {}
    try:
        num_experts = int(moe_cfg.get("num_experts", 1) or 1)
    except (TypeError, ValueError):
        num_experts = 1
    try:
        top_k = int(moe_cfg.get("top_k", 1) or 1)
    except (TypeError, ValueError):
        top_k = 1
    try:
        n_shared = int(moe_cfg.get("n_shared_experts", 0) or 0)
    except (TypeError, ValueError):
        n_shared = 0
    enabled = num_experts > 1 or top_k > 1 or n_shared > 0
    _MODEL_MOE_CACHE[model_path] = enabled
    return enabled


def _write_temp_yaml(data: Dict[str, Any]) -> str:
    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(data, temp, default_flow_style=False, sort_keys=False)
        temp.flush()
        return temp.name
    finally:
        try:
            temp.close()
        except Exception:
            pass


def _bw_tag(bw_gbps: int) -> str:
    return f"bw{int(bw_gbps)}"


def _bw_label(bw_gbps: int) -> str:
    return BW_SWEEP_LABELS.get(int(bw_gbps), f"{int(bw_gbps)} GB/s")


def _bw_config_string(bw_gbps: int) -> str:
    return f"{int(bw_gbps)} GB"


def _fc_fair_bandwidth_gbps(bw_gbps: int, node_count: int) -> int:
    if node_count <= 1:
        return int(bw_gbps)
    fc_edges = max(node_count - 1, 1)
    scaled = (FC_FAIR_MESH_EDGES * float(bw_gbps)) / fc_edges
    return max(1, int(round(scaled)))


def _with_bw_tag(filename: str, bw_tag: str) -> str:
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{bw_tag}{ext}"


def _update_hw_dict(
    base_hw: Dict[str, Any],
    *,
    topology: str,
    shape: Tuple[int, int],
    parallelism: Dict[str, int],
    mb_override: Optional[int],
    bandwidth_gbps: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw)
    par = cfg.setdefault("parallelism", {})
    par["tp"] = int(parallelism["tp"])
    par["cp"] = int(parallelism.get("cp", 1))
    par["pp"] = int(parallelism["pp"])
    if "tp_sp" in parallelism:
        par["tp_sp"] = bool(parallelism["tp_sp"])
    if mb_override is not None:
        par["mb"] = int(mb_override)
    train_block = par.setdefault("train", {})
    train_block["dp"] = 1
    ep = int(parallelism.get("ep", train_block.get("ep", 1) or 1))
    train_block["ep"] = ep
    train_block.setdefault("tp_ep", True)
    if ep > 1:
        par["tp_sp"] = True
    inference_block = par.setdefault("inference", {})
    inference_block.setdefault("replica_count", 1)
    inference_block.setdefault("moe_dp", 1)

    net = cfg.setdefault("network", {})
    dims = list(net.get("dimensions") or [])
    if not dims:
        raise ValueError("Hardware config missing network.dimensions")
    dim0 = dims[0]
    pp = int(parallelism.get("pp", 1) or 1)
    if pp > 1:
        # Hierarchical mode models a TP/CP/EP stage; scale the 2D mesh per stage.
        stage_shape = _stage_shape_for_pp(shape, pp)
    else:
        stage_shape = shape
    node_count = int(stage_shape[0]) * int(stage_shape[1])
    dim0["size"] = [int(stage_shape[0]), int(stage_shape[1])]
    topo_block = dim0.setdefault("topology", {})
    topo_block["type"] = topology
    base_bw_value = topo_block.get("bandwidth")
    if bandwidth_gbps is not None:
        base_bw_value = _bw_config_string(bandwidth_gbps)
    dim0_bw_value = base_bw_value
    if FC_FAIR and topology == "FullyConnected" and bandwidth_gbps is not None:
        dim0_bw_value = _bw_config_string(
            _fc_fair_bandwidth_gbps(bandwidth_gbps, node_count)
        )
    if bandwidth_gbps is not None:
        topo_block["bandwidth"] = dim0_bw_value
    topo_block["optimize_2dmap"] = False
    dim0["parallelisms"] = ["tp", "cp", "ep"]

    exec_backend = cfg.setdefault("execution_backend", {})
    if str(exec_backend.get("model", "analytical")).lower() == "astra":
        astra_block = exec_backend.setdefault("astra", {})
        astra_block["mode"] = "full_astrasim_hierarchical"

    bw_value = base_bw_value if base_bw_value is not None else topo_block.get("bandwidth")
    if isinstance(bw_value, (list, tuple)):
        bw_value = bw_value[0] if bw_value else bw_value
    dim1 = {
        "id": "dim1_pp_dp",
        "label": "pp_dp_dim",
        "size": int(pp) * int(train_block.get("dp", 1) or 1),
        "topology": {
            "type": "Ring",
            "bandwidth": bw_value,
            "latency": topo_block.get("latency"),
            "energy_per_bit": topo_block.get("energy_per_bit"),
            "util": topo_block.get("util", 1.0),
        },
        "parallelisms": ["pp", "dp"],
    }
    net["dimensions"] = [dim0, dim1]

    return cfg


def _format_runtime(value: Optional[float]) -> str:
    if value is None:
        return "error"
    return f"{value:.6f}"


def _format_mem_fields(
    exceeded: Optional[bool],
    violation_gb: Optional[float],
) -> Tuple[str, str]:
    if exceeded is None:
        return "error", ""
    if exceeded:
        if violation_gb is None:
            return "yes", ""
        return "yes", f"{violation_gb:.2f}"
    return "no", "0.00"


def _slug(parts: Sequence[str]) -> str:
    cooked = "_".join(part.replace("/", "_") for part in parts if part)
    return cooked.replace(" ", "").replace(":", "_")


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"version": CACHE_VERSION, "cases": {}}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {"version": CACHE_VERSION, "cases": {}}
    if data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "cases": {}}
    cases = data.get("cases")
    if not isinstance(cases, dict):
        data["cases"] = {}
    return data


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
    except Exception:
        pass


def _case_key(case: Dict[str, Any]) -> str:
    kind = case.get("kind")
    data: Dict[str, Any] = {
        "kind": kind,
        "model": case.get("model_label"),
        "model_sig": case.get("model_sig"),
        "hw_sig": case.get("hw_sig"),
        "topology": case.get("topology"),
        "shape": case.get("shape"),
    }
    bw_gbps = case.get("bandwidth_gbps")
    if bw_gbps is not None:
        data["bw_gbps"] = int(bw_gbps)
    if str(case.get("topology")) == "FullyConnected":
        data["fc_fair"] = bool(FC_FAIR)
    if kind == "train":
        parallelism = case.get("parallelism") or {}
        data.update(
            {
                "tp": parallelism.get("tp"),
                "cp": parallelism.get("cp"),
                "ep": parallelism.get("ep", 1),
                "pp": parallelism.get("pp"),
                "axes": list(case.get("parallelism_axes") or ()),
            }
        )
    elif kind == "inference":
        data["tp"] = int(case["shape"][0]) * int(case["shape"][1])
    return json.dumps(data, sort_keys=True)


def _is_cache_row(row: Dict[str, Any]) -> bool:
    return isinstance(row, dict) and "runtime_s" in row and "topology" in row


def _describe_case(case: Dict[str, Any]) -> str:
    kind = case.get("kind")
    bw_gbps = case.get("bandwidth_gbps")
    bw_text = f" bw={_bw_label(bw_gbps)}" if bw_gbps is not None else ""
    if kind == "train":
        parallelism = case.get("parallelism") or {}
        return (
            f"train {case.get('model_label')} {case.get('topology')} "
            f"{case.get('shape')[0]}x{case.get('shape')[1]} "
            f"tp{parallelism.get('tp')} cp{parallelism.get('cp')} "
            f"ep{parallelism.get('ep', 1)} pp{parallelism.get('pp')}{bw_text}"
        )
    if kind == "inference":
        shape = case.get("shape")
        tp = int(shape[0]) * int(shape[1])
        return (
            f"inference {case.get('model_label')} {case.get('topology')} "
            f"{shape[0]}x{shape[1]} tp{tp}{bw_text}"
        )
    return f"unknown {case}"


def _format_tokens(value: int) -> str:
    if value and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def _format_shape_label(shape_label: str) -> str:
    if shape_label == "4x5":
        return "4x5"
    return shape_label


def _shape_key(value: str) -> Tuple[int, int]:
    try:
        left, right = value.split("x", 1)
        right = right.split("[", 1)[0]
        return int(left), int(right)
    except Exception:
        return (0, 0)


def _stage_shape_for_pp(shape: Tuple[int, int], pp: int) -> Tuple[int, int]:
    if pp <= 1:
        return shape
    total = int(shape[0]) * int(shape[1])
    if total % pp != 0:
        raise ValueError(f"Shape {shape[0]}x{shape[1]} not divisible by pp={pp}")
    target_ratio = float(shape[0]) / float(shape[1]) if shape[1] else 1.0
    best = None
    best_score = None
    best_dist = None
    for p1 in range(1, pp + 1):
        if pp % p1 != 0:
            continue
        p2 = pp // p1
        if shape[0] % p1 != 0 or shape[1] % p2 != 0:
            continue
        cand = (int(shape[0]) // p1, int(shape[1]) // p2)
        ratio = float(cand[0]) / float(cand[1]) if cand[1] else 1.0
        score = abs(ratio - target_ratio)
        dist = abs(cand[0] - shape[0]) + abs(cand[1] - shape[1])
        if best is None or (score, dist) < (best_score, best_dist):
            best = cand
            best_score = score
            best_dist = dist
    if best is not None:
        return best

    stage_size = total // pp
    for a in range(1, stage_size + 1):
        if stage_size % a != 0:
            continue
        b = stage_size // a
        ratio = float(a) / float(b) if b else 1.0
        score = abs(ratio - target_ratio)
        dist = abs(a - shape[0]) + abs(b - shape[1])
        if best is None or (score, dist) < (best_score, best_dist):
            best = (int(a), int(b))
            best_score = score
            best_dist = dist
    if best is None:
        return (stage_size, 1)
    return best


def _candidate_parallelisms(
    total_devices: int,
    *,
    axes: Sequence[str],
) -> List[Dict[str, int]]:
    axes_set = {axis.strip().lower() for axis in axes}
    allow_ep = "ep" in axes_set
    allow_pp = "pp" in axes_set
    candidates = []
    ep_values = range(1, total_devices + 1) if allow_ep else (1,)
    for ep in ep_values:
        if total_devices % ep != 0:
            continue
        remaining = total_devices // ep
        for tp in range(1, remaining + 1):
            if remaining % tp != 0:
                continue
            pp = remaining // tp
            if not allow_pp and pp != 1:
                continue
            if pp > PP_MAX:
                continue
            candidates.append({"tp": tp, "cp": 1, "pp": pp, "ep": ep})
    return candidates


def _select_best_parallelisms(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    best_valid: Dict[Tuple[str, str], Tuple[float, Dict[str, int]]] = {}
    best_any: Dict[Tuple[str, str], Tuple[float, Dict[str, int]]] = {}
    for row in rows:
        try:
            runtime = float(row["runtime_s"])
        except Exception:
            continue
        model = str(row.get("model", ""))
        shape = str(row.get("shape", ""))
        key = (model, shape)
        par = {
            "tp": int(row.get("tp", 0) or 0),
            "cp": int(row.get("cp", 0) or 0),
            "ep": int(row.get("ep", 1) or 1),
            "pp": int(row.get("pp", 0) or 0),
        }
        if runtime > 0:
            current = best_any.get(key)
            if current is None or runtime < current[0]:
                best_any[key] = (runtime, par)
        if str(row.get("mem_exceeded", "")).lower() == "no" and runtime > 0:
            current = best_valid.get(key)
            if current is None or runtime < current[0]:
                best_valid[key] = (runtime, par)
    selected: Dict[Tuple[str, str], Dict[str, int]] = {}
    for key, value in best_any.items():
        selected[key] = best_valid.get(key, value)[1]
    return selected


def _format_train_label(shape_label: str, tp: int, pp: int) -> str:
    return f"{_format_shape_label(shape_label)}\nTP={tp}\nPP={pp}"


def _format_inf_label(shape_label: str, tp: int) -> str:
    return f"{_format_shape_label(shape_label)}\nTP={tp}"


def _run_training_case(
    base_hw: Dict[str, Any],
    *,
    model_config_path: str,
    model_label: str,
    topology: str,
    shape: Tuple[int, int],
    parallelism: Dict[str, int],
    bandwidth_gbps: Optional[int],
) -> Tuple[Optional[float], Optional[str], Optional[bool], Optional[float]]:
    mb = 2 * int(parallelism["pp"])
    run_dir = None
    prev_env: Dict[str, Optional[str]] = {}
    hw_dict = _update_hw_dict(
        base_hw,
        topology=topology,
        shape=shape,
        parallelism=parallelism,
        mb_override=mb,
        bandwidth_gbps=bandwidth_gbps,
    )
    temp_path = _write_temp_yaml(hw_dict)
    desc_parts = [
        "train",
        model_label,
        f"{topology}",
        f"{shape[0]}x{shape[1]}",
        f"tp{parallelism['tp']}",
        f"cp{parallelism['cp']}",
        f"ep{parallelism.get('ep', 1)}",
        f"pp{parallelism['pp']}",
    ]
    if bandwidth_gbps is not None:
        desc_parts.append(_bw_tag(bandwidth_gbps))
    desc = _slug(desc_parts)
    out_dir = OUTPUT_ROOT / "train" / desc
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir, prev_env = hfval._prepare_astra_tmp_dir()
        mode = determine_model_mode(model_config_path)
        hw_config = config.parse_config(temp_path, config_type="hardware")
        model_config = config.parse_config(model_config_path, config_type=mode)
        config.validate_configs(hw_config, model_config)
        ensure_chakra_available()
        calc = TimeCalculationLLM(hw_config, model_config, mode, output_dir=str(out_dir))
        runtime = float(calc.calc_time_llm())
        mem_exceeded = bool(getattr(calc, "memory_capacity_exceeded", False))
        mem_violation = float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0)
        return runtime, None, mem_exceeded, mem_violation
    except Exception as exc:
        return None, str(exc), None, None
    finally:
        if prev_env:
            hfval._restore_astra_env(prev_env)
        if run_dir and hfval.CLEANUP_ASTRA_TMP:
            shutil.rmtree(run_dir, ignore_errors=True)
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _run_inference_case(
    base_hw: Dict[str, Any],
    *,
    topology: str,
    shape: Tuple[int, int],
    model_config_path: str,
    bandwidth_gbps: Optional[int],
) -> Tuple[Optional[float], Optional[str], Optional[bool], Optional[float]]:
    run_dir = None
    prev_env: Dict[str, Optional[str]] = {}
    tp = int(shape[0]) * int(shape[1])
    hw_dict = _update_hw_dict(
        base_hw,
        topology=topology,
        shape=shape,
        parallelism={"tp": tp, "cp": 1, "pp": 1},
        mb_override=1,
        bandwidth_gbps=bandwidth_gbps,
    )
    temp_path = _write_temp_yaml(hw_dict)
    desc_parts = ["inference", f"{topology}", f"{shape[0]}x{shape[1]}", f"tp{tp}"]
    if bandwidth_gbps is not None:
        desc_parts.append(_bw_tag(bandwidth_gbps))
    desc = _slug(desc_parts)
    out_dir = OUTPUT_ROOT / "inference" / desc
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir, prev_env = hfval._prepare_astra_tmp_dir()
        mode = determine_model_mode(model_config_path)
        hw_config = config.parse_config(temp_path, config_type="hardware")
        model_config = config.parse_config(model_config_path, config_type=mode)
        config.validate_configs(hw_config, model_config)
        ensure_chakra_available()
        calc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        summary = calc.calc_total_inference_time()
        runtime = float(summary["total_inference_time"])
        mem_exceeded = bool(getattr(calc, "memory_capacity_exceeded", False))
        mem_violation = float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0)
        return runtime, None, mem_exceeded, mem_violation
    except Exception as exc:
        return None, str(exc), None, None
    finally:
        if prev_env:
            hfval._restore_astra_env(prev_env)
        if run_dir and hfval.CLEANUP_ASTRA_TMP:
            shutil.rmtree(run_dir, ignore_errors=True)
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _case_worker(case: Dict[str, Any]) -> Dict[str, Any]:
    base_hw = _WORKER_BASE_HW
    if base_hw is None:
        return {
            "error": "Worker base hardware config not initialized",
            "_case_key": case.get("case_key"),
        }
    if VERBOSE:
        print(f"[2d_test] start {_describe_case(case)}", flush=True)
    kind = case.get("kind")
    if kind == "train":
        runtime, error, mem_exceeded, mem_violation = _run_training_case(
            base_hw,
            model_config_path=case["model_config"],
            model_label=case["model_label"],
            topology=case["topology"],
            shape=case["shape"],
            parallelism=case["parallelism"],
            bandwidth_gbps=case.get("bandwidth_gbps"),
        )
        mem_flag, mem_delta = _format_mem_fields(mem_exceeded, mem_violation)
        row = {
            "model": case["model_label"],
            "topology": case["topology"],
            "shape": f"{case['shape'][0]}x{case['shape'][1]}",
            "tp": case["parallelism"]["tp"],
            "cp": case["parallelism"]["cp"],
            "ep": case["parallelism"].get("ep", 1),
            "pp": case["parallelism"]["pp"],
            "mb": 4 * case["parallelism"]["pp"],
            "runtime_s": _format_runtime(runtime),
            "mem_exceeded": mem_flag,
            "mem_violation_gb": mem_delta,
        }
        if error:
            row["error"] = error
        row["_case_key"] = case.get("case_key")
        if VERBOSE:
            status = row["runtime_s"]
            print(f"[2d_test] done {_describe_case(case)} -> {status}", flush=True)
        return row
    if kind == "inference":
        runtime, error, mem_exceeded, mem_violation = _run_inference_case(
            base_hw,
            topology=case["topology"],
            shape=case["shape"],
            model_config_path=case["model_config"],
            bandwidth_gbps=case.get("bandwidth_gbps"),
        )
        mem_flag, mem_delta = _format_mem_fields(mem_exceeded, mem_violation)
        tp = int(case["shape"][0]) * int(case["shape"][1])
        row = {
            "model": case["model_label"],
            "topology": case["topology"],
            "shape": f"{case['shape'][0]}x{case['shape'][1]}",
            "tp": tp,
            "cp": 1,
            "ep": 1,
            "pp": 1,
            "mb": 1,
            "runtime_s": _format_runtime(runtime),
            "mem_exceeded": mem_flag,
            "mem_violation_gb": mem_delta,
        }
        if error:
            row["error"] = error
        row["_case_key"] = case.get("case_key")
        if VERBOSE:
            status = row["runtime_s"]
            print(f"[2d_test] done {_describe_case(case)} -> {status}", flush=True)
        return row
    return {"error": f"Unknown case kind: {kind}", "_case_key": case.get("case_key")}


def _run_cases_parallel(
    cases: Sequence[Dict[str, Any]],
    base_hw: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not cases:
        return []
    if VERBOSE:
        print(f"[2d_test] running {len(cases)} cases (workers={NUM_WORKERS})", flush=True)
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(total=len(cases), desc="2d_test")
    except Exception:
        progress = None
    worker_count = min(NUM_WORKERS, len(cases), os.cpu_count() or 1)
    if worker_count <= 1:
        _init_worker(base_hw)
        results = []
        for case in cases:
            results.append(_case_worker(case))
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()
        return results
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(base_hw,),
    ) as pool:
        results = []
        for result in pool.imap_unordered(_case_worker, cases):
            results.append(result)
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()
        return results


def _print_table(rows: Sequence[Dict[str, Any]], header: Sequence[str]) -> None:
    rendered = [[str(row.get(col, "")) for col in header] for row in rows]
    widths = [len(col) for col in header]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = "  "
    header_line = sep.join(col.ljust(widths[idx]) for idx, col in enumerate(header))
    divider = sep.join("-" * widths[idx] for idx in range(len(header)))
    print(header_line)
    print(divider)
    for row in rendered:
        print(sep.join(row[idx].ljust(widths[idx]) for idx in range(len(header))))


def _plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_dual_topology_axis(
    ax: plt.Axes,
    labels: Sequence[str],
    mesh_values: Sequence[float],
    torus_values: Sequence[float],
    *,
    title: str,
    ylabel: Optional[str] = None,
) -> Tuple[List[Any], List[str]]:
    x = np.arange(len(labels))
    width = 0.38
    torus_handle = ax.bar(x - width / 2, torus_values, width, label="2D Torus")
    mesh_handle = ax.bar(x + width / 2, mesh_values, width, label="2D Mesh")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    return [torus_handle, mesh_handle], ["2D Torus", "2D Mesh"]


def _default_mode_colors() -> Tuple[str, str]:
    palette = plt.rcParams.get("axes.prop_cycle", None)
    colors = palette.by_key().get("color", []) if palette else []
    inf_color = colors[0] if len(colors) > 0 else "#1f77b4"
    train_color = colors[1] if len(colors) > 1 else "#ff7f0e"
    return inf_color, train_color


def _compute_speedup_ratio(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_topology: str,
    target_topology: str,
) -> Dict[Tuple[str, str], float]:
    groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        try:
            runtime = float(row["runtime_s"])
        except Exception:
            continue
        if runtime <= 0:
            continue
        model = str(row.get("model", ""))
        shape = str(row.get("shape", ""))
        topo = str(row.get("topology", ""))
        groups.setdefault(model, {}).setdefault(shape, {})[topo] = runtime
    ratios: Dict[Tuple[str, str], float] = {}
    for model, shape_map in groups.items():
        for shape_label, topo_map in shape_map.items():
            baseline_val = topo_map.get(baseline_topology)
            target_val = topo_map.get(target_topology)
            if baseline_val is None or target_val is None or target_val <= 0:
                continue
            ratios[(model, shape_label)] = baseline_val / target_val
    return ratios


def _speedup_ylim(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    default: Tuple[float, float] = (0.95, 1.4),
) -> Tuple[float, float]:
    values = [val for val in train_speedup.values()] + [val for val in inf_speedup.values()]
    if not values:
        return default
    lo = min(values)
    hi = max(values)
    lo = min(lo, 1.0)
    hi = max(hi, 1.0)
    span = max(hi - lo, 0.0)
    pad = max(0.05, span * 0.1)
    lower = max(0.0, lo - pad)
    upper = hi + pad
    if upper <= lower:
        upper = lower + 0.1
    return (lower, upper)


def _plot_speedup_by_model(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    model_order: Sequence[str],
    shapes: Sequence[str],
    output_path: Path,
) -> None:
    if not model_order or not shapes:
        return
    max_shapes = max(len(shapes), 1)
    axis_width = max(6.0, max_shapes * 0.7)
    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=(axis_width * len(model_order), 6),
        sharey=True,
    )
    if len(model_order) == 1:
        axes = [axes]
    inf_color, train_color = _default_mode_colors()
    x = np.arange(len(shapes))
    width = 0.36

    for idx, (ax, model_label) in enumerate(zip(axes, model_order)):
        title = MODEL_DISPLAY_NAMES.get(model_label, model_label)
        inf_vals = []
        train_vals = []
        for shape in shapes:
            inf_val = inf_speedup.get((model_label, shape))
            train_val = train_speedup.get((model_label, shape))
            inf_vals.append(inf_val if inf_val is not None else np.nan)
            train_vals.append(train_val if train_val is not None else np.nan)
        ax.bar(x - width / 2, inf_vals, width, label="inference", color=inf_color)
        ax.bar(x + width / 2, train_vals, width, label="train", color=train_color)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([_format_shape_label(s) for s in shapes], rotation=0, ha="center")
        ax.set_ylim(0.95, 1.4)
        if idx == 0:
            ax.set_ylabel("2D Torus over Mesh speedup")

    legend_axis = axes[1] if len(axes) > 1 else axes[0]
    legend_axis.legend(loc="center right")
    fig.suptitle("2D Torus vs Mesh inference/training runtime comparison", y=0.96)
    fig.supxlabel("2D topology shape")
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.98, wspace=0.08)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_super_merged_speedup(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    model_order: Sequence[str],
    shapes: Sequence[str],
    output_path: Path,
    ylabel: str = "2D Torus over Mesh speedup",
    title: str = "2D Torus vs Mesh inference/training runtime comparison",
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if not model_order or not shapes:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(shapes) * 0.9), 6))
    inf_color, train_color = _default_mode_colors()
    model_colors = {}
    for model_label in model_order:
        if model_label == "GLM4.5_106B":
            model_colors[model_label] = train_color
        else:
            model_colors[model_label] = inf_color
    mode_order = ["inference", "train"]
    mode_hatches = {"inference": "", "train": "//"}
    combos = [(model_label, mode) for model_label in model_order for mode in mode_order]
    width = 0.72 / max(1, len(combos))
    offsets = (np.arange(len(combos)) - (len(combos) - 1) / 2) * width
    x = np.arange(len(shapes))

    prev_hatch_lw = plt.rcParams.get("hatch.linewidth", None)
    combined_handles = []
    plt.rcParams["hatch.linewidth"] = 0.75
    try:
        for idx, (model_label, mode) in enumerate(combos):
            if mode == "train":
                values = [
                    train_speedup.get((model_label, shape))
                    if train_speedup.get((model_label, shape)) is not None
                    else np.nan
                    for shape in shapes
                ]
            else:
                values = [
                    inf_speedup.get((model_label, shape))
                    if inf_speedup.get((model_label, shape)) is not None
                    else np.nan
                    for shape in shapes
                ]
            ax.bar(
                x + offsets[idx],
                values,
                width,
                color=model_colors[model_label],
                hatch=mode_hatches[mode],
            )
            label_mode = "inf" if mode == "inference" else "train"
            display_name = MODEL_DISPLAY_NAMES.get(model_label, model_label)
            combined_handles.append(
                Patch(
                    facecolor=model_colors[model_label],
                    hatch=mode_hatches[mode],
                    label=f"{display_name} {label_mode}",
                )
            )
    finally:
        if prev_hatch_lw is not None:
            plt.rcParams["hatch.linewidth"] = prev_hatch_lw

    ax.set_xticks(x)
    ax.set_xticklabels([_format_shape_label(s) for s in shapes], rotation=0, ha="center")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(0.95, 1.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("2D topology and GPU count")
    ax.set_title(title)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)

    ax.legend(handles=combined_handles, loc="upper left")
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.1, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_for_bandwidth(
    *,
    bw_gbps: int,
    train_hw_base: Dict[str, Any],
    inf_hw_base: Dict[str, Any],
    train_hw_sig: str,
    inf_hw_sig: str,
    train_model_sigs: Dict[str, str],
    inf_model_sigs: Dict[str, str],
    cache: Dict[str, Any],
) -> None:
    bw_gbps = int(bw_gbps)
    bw_label = _bw_label(bw_gbps)
    bw_tag = _bw_tag(bw_gbps)
    if VERBOSE:
        print(f"[2d_test] bandwidth sweep: {bw_label}", flush=True)

    cache_cases = cache.setdefault("cases", {})

    train_cases: List[Dict[str, Any]] = []
    if AUTO_PARALLELISM:
        if "Mesh2D" not in TOPOLOGIES:
            raise ValueError("AUTO_PARALLELISM requires Mesh2D to be in TOPOLOGIES")
        selection_cases: List[Dict[str, Any]] = []
        for model in TRAIN_MODELS:
            model_axes = model.get("axes", ("tp", "cp", "pp"))
            require_ep = _is_moe_model(model["config"])
            for shape in TRAIN_SHAPES:
                total_devices = int(shape[0]) * int(shape[1])
                for parallelism in _candidate_parallelisms(total_devices, axes=model_axes):
                    ep_value = int(parallelism.get("ep", 1) or 1)
                    if require_ep and ep_value <= 1:
                        continue
                    if not require_ep and ep_value != 1:
                        continue
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model_axes,
                        "topology": "Mesh2D",
                        "shape": shape,
                        "parallelism": parallelism,
                        "bandwidth_gbps": bw_gbps,
                    }
                    case["case_key"] = _case_key(case)
                    selection_cases.append(case)
        selection_cached_rows = []
        selection_run_cases = []
        for case in selection_cases:
            cached = cache_cases.get(case["case_key"])
            if isinstance(cached, dict) and _is_cache_row(cached):
                selection_cached_rows.append(cached)
            else:
                selection_run_cases.append(case)
        if VERBOSE:
            print(
                f"[2d_test] auto-parallelism sweep: "
                f"{len(selection_cached_rows)}/{len(selection_cases)} cache hits",
                flush=True,
            )
        selection_rows = selection_cached_rows + (
            _run_cases_parallel(selection_run_cases, train_hw_base)
            if selection_run_cases
            else []
        )
        for row in selection_rows:
            case_key = row.get("_case_key")
            if case_key and _is_cache_row(row):
                cache_cases[case_key] = row
        best_parallelisms = _select_best_parallelisms(selection_rows)
        missing_best = []
        for model in TRAIN_MODELS:
            for shape in TRAIN_SHAPES:
                shape_label = f"{shape[0]}x{shape[1]}"
                parallelism = best_parallelisms.get((model["label"], shape_label))
                if not parallelism:
                    missing_best.append(f"{model['label']} {shape_label}")
                    continue
                for topology in TOPOLOGIES:
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model.get("axes", ("tp", "cp", "pp")),
                        "topology": topology,
                        "shape": shape,
                        "parallelism": parallelism,
                        "bandwidth_gbps": bw_gbps,
                    }
                    case["case_key"] = _case_key(case)
                    train_cases.append(case)
        if missing_best:
            print(
                "[2d_test] warning: no valid parallelism found for "
                + ", ".join(missing_best),
                flush=True,
            )
    else:
        for model in TRAIN_MODELS:
            parallelisms = list(model.get("parallelisms") or ())
            if len(parallelisms) != len(TRAIN_SHAPES):
                raise ValueError(
                    f"TRAIN_SHAPES ({len(TRAIN_SHAPES)}) must match parallelisms "
                    f"({len(parallelisms)}) for model {model.get('label')}"
                )
            for topology in TOPOLOGIES:
                for shape, parallelism in zip(TRAIN_SHAPES, parallelisms):
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model.get("axes", ("tp", "cp", "pp")),
                        "topology": topology,
                        "shape": shape,
                        "parallelism": parallelism,
                        "bandwidth_gbps": bw_gbps,
                    }
                    case["case_key"] = _case_key(case)
                    train_cases.append(case)

    inf_cases: List[Dict[str, Any]] = []
    for model in INF_MODELS:
        for shape in INF_SHAPES:
            for topology in TOPOLOGIES:
                case = {
                    "kind": "inference",
                    "model_label": model["label"],
                    "model_config": model["config"],
                    "model_sig": inf_model_sigs.get(model["config"], ""),
                    "hw_sig": inf_hw_sig,
                    "topology": topology,
                    "shape": shape,
                    "bandwidth_gbps": bw_gbps,
                }
                case["case_key"] = _case_key(case)
                inf_cases.append(case)

    train_cached_rows = []
    train_run_cases = []
    for case in train_cases:
        cached = cache_cases.get(case["case_key"])
        if isinstance(cached, dict) and _is_cache_row(cached):
            train_cached_rows.append(cached)
        else:
            train_run_cases.append(case)

    inf_cached_rows = []
    inf_run_cases = []
    for case in inf_cases:
        cached = cache_cases.get(case["case_key"])
        if isinstance(cached, dict) and _is_cache_row(cached):
            inf_cached_rows.append(cached)
        else:
            inf_run_cases.append(case)

    print(
        f"[2d_test] cache: train {len(train_cached_rows)}/{len(train_cases)} "
        f"hits, inf {len(inf_cached_rows)}/{len(inf_cases)} hits",
        flush=True,
    )
    if VERBOSE and train_run_cases:
        print(f"[2d_test] train cases pending: {len(train_run_cases)}", flush=True)
    if VERBOSE and inf_run_cases:
        print(f"[2d_test] inference cases pending: {len(inf_run_cases)}", flush=True)

    train_rows = train_cached_rows + (
        _run_cases_parallel(train_run_cases, train_hw_base) if train_run_cases else []
    )
    inf_rows = inf_cached_rows + (
        _run_cases_parallel(inf_run_cases, inf_hw_base) if inf_run_cases else []
    )
    for row in train_rows:
        row.setdefault("ep", 1)
    for row in inf_rows:
        row.setdefault("ep", 1)

    for row in train_rows + inf_rows:
        case_key = row.get("_case_key")
        if case_key and _is_cache_row(row):
            cache_cases[case_key] = row

    train_errors: List[str] = []
    for row in train_rows:
        error = row.get("error")
        if error:
            train_errors.append(
                f"{row.get('model')} {row.get('topology')} {row.get('shape')} tp{row.get('tp')} "
                f"cp{row.get('cp')} ep{row.get('ep', 1)} pp{row.get('pp')}: {error}"
            )

    inf_errors: List[str] = []
    for row in inf_rows:
        error = row.get("error")
        if error:
            inf_errors.append(
                f"{row.get('model')} {row.get('topology')} {row.get('shape')}: {error}"
            )

    train_rows.sort(
        key=lambda row: (
            row.get("model", ""),
            row.get("topology", ""),
            _shape_key(row.get("shape", "")),
            int(row.get("tp", 0) or 0),
            int(row.get("cp", 0) or 0),
            int(row.get("ep", 1) or 1),
            int(row.get("pp", 0) or 0),
        )
    )
    inf_rows.sort(
        key=lambda row: (
            row.get("model", ""),
            _shape_key(row.get("shape", "")),
            row.get("topology", ""),
        )
    )

    train_header = [
        "model",
        "topology",
        "shape",
        "tp",
        "cp",
        "ep",
        "pp",
        "mb",
        "runtime_s",
        "mem_exceeded",
        "mem_violation_gb",
        "error",
    ]
    inf_header = [
        "model",
        "topology",
        "shape",
        "tp",
        "cp",
        "ep",
        "pp",
        "mb",
        "runtime_s",
        "mem_exceeded",
        "mem_violation_gb",
        "error",
    ]

    print(f"\n=== Training Results (BW {bw_label}) ===")
    _print_table(train_rows, train_header)
    print(f"\n=== Inference Results (BW {bw_label}) ===")
    _print_table(inf_rows, inf_header)

    train_tsv = OUTPUT_ROOT / f"2d_test_train_{bw_tag}.tsv"
    inf_tsv = OUTPUT_ROOT / f"2d_test_inf_{bw_tag}.tsv"
    with train_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(train_header) + "\n")
        for row in train_rows:
            handle.write("\t".join(str(row.get(col, "")) for col in train_header) + "\n")
    with inf_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(inf_header) + "\n")
        for row in inf_rows:
            handle.write("\t".join(str(row.get(col, "")) for col in inf_header) + "\n")

    _save_cache(cache)

    train_titles: Dict[str, str] = {}
    for model in TRAIN_MODELS:
        params = read_yaml(model["config"]).get("model_param") or {}
        seq_len = int(params.get("seq_len", 0) or 0)
        display_name = MODEL_DISPLAY_NAMES.get(model["label"], model["label"])
        train_titles[model["label"]] = f"{display_name}\n[Seq={_format_tokens(seq_len)} tok]"

    inf_titles: Dict[str, str] = {}
    for model in INF_MODELS:
        params = read_yaml(model["config"]).get("model_param") or {}
        seq_len = int(params.get("seq_len", 0) or 0)
        decode_len = int(params.get("decode_len", 0) or 0)
        prefill_len = max(seq_len - decode_len, 0)
        display_name = MODEL_DISPLAY_NAMES.get(model["label"], model["label"])
        inf_titles[model["label"]] = (
            f"{display_name}\n"
            f"[Prefill={_format_tokens(prefill_len)} tok, "
            f"Decode={_format_tokens(decode_len)} tok]"
        )

    train_ok = [row for row in train_rows if row.get("runtime_s") != "error"]

    topo_groups: Dict[str, Dict[Tuple[str, int, int], Dict[str, float]]] = {}
    for row in train_ok:
        model = str(row.get("model", ""))
        combo = (
            str(row.get("shape", "")),
            int(row.get("tp", 0) or 0),
            int(row.get("pp", 0) or 0),
        )
        topo_groups.setdefault(model, {}).setdefault(combo, {})[row["topology"]] = float(
            row["runtime_s"]
        )

    train_experiments: Dict[str, List[Dict[str, Any]]] = {}
    for model_label, combo_map in topo_groups.items():
        experiments = []
        for combo, topo_map in combo_map.items():
            if "Mesh2D" not in topo_map or "Torus2D" not in topo_map:
                continue
            torus_val = topo_map["Torus2D"]
            if torus_val <= 0:
                continue
            shape_label, tp, pp = combo
            mesh_norm = topo_map["Mesh2D"] / torus_val
            experiments.append(
                {
                    "label": _format_train_label(shape_label, tp, pp),
                    "mesh": mesh_norm,
                    "torus": 1.0,
                }
            )
        experiments.sort(key=lambda item: item["label"])
        train_experiments[model_label] = experiments

    if len(train_experiments) > 1:
        label_sets = [
            set(item["label"] for item in items)
            for items in train_experiments.values()
            if items
        ]
        if label_sets:
            common_labels = set.intersection(*label_sets)
            if common_labels:
                for model_label, items in train_experiments.items():
                    train_experiments[model_label] = [
                        item for item in items if item["label"] in common_labels
                    ]
        for model_label, items in train_experiments.items():
            items.sort(key=lambda item: item["label"])

    train_order = [model["label"] for model in TRAIN_MODELS]
    if any(train_experiments.get(label) for label in train_order):
        max_items = max(
            len(train_experiments.get(label, [])) for label in train_order if train_order
        )
        axis_width = max(5.5, max_items * 0.7)
        fig, axes = plt.subplots(
            1,
            len(train_order),
            figsize=(axis_width * len(train_order), 6),
            sharey=True,
        )
        if len(train_order) == 1:
            axes = [axes]
        legend_handles: Optional[List[Any]] = None
        legend_labels: Optional[List[str]] = None
        for ax, model_label in zip(axes, train_order):
            entries = train_experiments.get(model_label, [])
            title = train_titles.get(model_label, model_label)
            if not entries:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            labels = [item["label"] for item in entries]
            mesh_vals = [item["mesh"] for item in entries]
            torus_vals = [item["torus"] for item in entries]
            handles, labels_text = _plot_dual_topology_axis(
                ax,
                labels,
                mesh_vals,
                torus_vals,
                title=title,
                ylabel=Y_LABEL if legend_handles is None else None,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels_text
        fig.suptitle("Training runtime vs 2D topology (64 GPUs)", y=0.99)
        if legend_handles and legend_labels:
            axes[-1].legend(legend_handles, legend_labels, loc="lower right")
        fig.subplots_adjust(top=0.81, bottom=0.11, left=0.1, right=0.98, wspace=0.08)
        fig.savefig(PLOT_DIR / f"2d_test_train_topology_{bw_tag}.png", dpi=200)
        plt.close(fig)

    inf_ok = [row for row in inf_rows if row.get("runtime_s") != "error"]
    inf_groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in inf_ok:
        model = str(row.get("model", ""))
        shape_label = str(row.get("shape", ""))
        inf_groups.setdefault(model, {}).setdefault(shape_label, {})[
            row["topology"]
        ] = float(row["runtime_s"])

    inf_experiments: Dict[str, List[Dict[str, Any]]] = {}
    for model_label, shape_map in inf_groups.items():
        entries = []
        for shape_label, topo_map in shape_map.items():
            if "Mesh2D" not in topo_map or "Torus2D" not in topo_map:
                continue
            torus_val = topo_map["Torus2D"]
            if torus_val <= 0:
                continue
            dims = _shape_key(shape_label)
            tp = int(dims[0]) * int(dims[1])
            mesh_norm = topo_map["Mesh2D"] / torus_val
            entries.append(
                {
                    "label": _format_inf_label(shape_label, tp),
                    "mesh": mesh_norm,
                    "torus": 1.0,
                }
            )
        entries.sort(key=lambda item: item["label"])
        inf_experiments[model_label] = entries

    if len(inf_experiments) > 1:
        label_sets = [
            set(item["label"] for item in items)
            for items in inf_experiments.values()
            if items
        ]
        if label_sets:
            common_labels = set.intersection(*label_sets)
            if common_labels:
                for model_label, items in inf_experiments.items():
                    inf_experiments[model_label] = [
                        item for item in items if item["label"] in common_labels
                    ]
        for model_label, items in inf_experiments.items():
            items.sort(key=lambda item: item["label"])

    inf_order = [model["label"] for model in INF_MODELS]
    if any(inf_experiments.get(label) for label in inf_order):
        max_items = max(
            len(inf_experiments.get(label, [])) for label in inf_order if inf_order
        )
        axis_width = max(5.5, max_items * 0.7)
        fig, axes = plt.subplots(
            1,
            len(inf_order),
            figsize=(axis_width * len(inf_order), 6),
            sharey=True,
        )
        if len(inf_order) == 1:
            axes = [axes]
        legend_handles = None
        legend_labels = None
        for ax, model_label in zip(axes, inf_order):
            entries = inf_experiments.get(model_label, [])
            title = inf_titles.get(model_label, model_label)
            if not entries:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            labels = [item["label"] for item in entries]
            mesh_vals = [item["mesh"] for item in entries]
            torus_vals = [item["torus"] for item in entries]
            handles, labels_text = _plot_dual_topology_axis(
                ax,
                labels,
                mesh_vals,
                torus_vals,
                title=title,
                ylabel=Y_LABEL if legend_handles is None else None,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels_text
        fig.suptitle("Inference runtime vs 2D topology (TP only)", y=0.99)
        if legend_handles and legend_labels:
            axes[-1].legend(legend_handles, legend_labels, loc="lower right")
        fig.subplots_adjust(top=0.81, bottom=0.11, left=0.08, right=0.98, wspace=0.08)
        fig.savefig(PLOT_DIR / f"2d_test_inf_topology_{bw_tag}.png", dpi=200)
        plt.close(fig)

    speedup_maps: Dict[str, Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]] = {}
    for topo in COMP_TOPO:
        speedup_maps[topo] = (
            _compute_speedup_ratio(
                train_ok,
                baseline_topology="Mesh2D",
                target_topology=topo,
            ),
            _compute_speedup_ratio(
                inf_ok,
                baseline_topology="Mesh2D",
                target_topology=topo,
            ),
        )
    train_speedup, inf_speedup = speedup_maps.get("Torus2D", ({}, {}))
    merged_shapes = sorted(
        {f"{shape[0]}x{shape[1]}" for shape in TRAIN_SHAPES}
        | {f"{shape[0]}x{shape[1]}" for shape in INF_SHAPES},
        key=_shape_key,
    )
    merged_order = [model["label"] for model in TRAIN_MODELS]
    for model in INF_MODELS:
        if model["label"] not in merged_order:
            merged_order.append(model["label"])
    if merged_shapes:
        _plot_speedup_by_model(
            train_speedup,
            inf_speedup,
            model_order=merged_order,
            shapes=merged_shapes,
            output_path=PLOT_DIR / f"2d_test_merged_by_model_{bw_tag}.png",
        )
        for topo in COMP_TOPO:
            if topo not in COMP_TOPO_SPECS:
                continue
            train_map, inf_map = speedup_maps.get(topo, ({}, {}))
            spec = COMP_TOPO_SPECS[topo]
            y_limits = None
            if spec.get("dynamic_ylim", False):
                y_limits = _speedup_ylim(train_map, inf_map)
            _plot_super_merged_speedup(
                train_map,
                inf_map,
                model_order=merged_order,
                shapes=merged_shapes,
                output_path=PLOT_DIR / _with_bw_tag(spec["output"], bw_tag),
                ylabel=spec["ylabel"],
                title=f"{spec['title']} (BW={bw_label})",
                y_limits=y_limits,
            )

    if train_errors or inf_errors:
        print("\nErrors:")
        for msg in train_errors + inf_errors:
            print(f"- {msg}")

    mem_failures = [
        row
        for row in (train_rows + inf_rows)
        if str(row.get("mem_exceeded", "")).lower() == "yes"
    ]
    if mem_failures:
        print("\nMemory capacity exceeded:")
        for row in mem_failures:
            label = (
                f"{row.get('model')} {row.get('topology')} {row.get('shape')} "
                f"tp{row.get('tp')} cp{row.get('cp')} ep{row.get('ep', 1)} pp{row.get('pp')}"
            )
            violation = row.get("mem_violation_gb", "")
            suffix = f" (over by {violation} GiB)" if violation else ""
            print(f"- {label}{suffix}")
    else:
        print("\nMemory capacity exceeded: none")


def main() -> None:
    hfval._ensure_project_root_on_path()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    train_hw_base = read_yaml(TRAIN_HW_CONFIG)
    inf_hw_base = read_yaml(INF_HW_CONFIG)
    train_hw_sig = _file_signature(TRAIN_HW_CONFIG)
    inf_hw_sig = _file_signature(INF_HW_CONFIG)
    train_model_sigs = {model["config"]: _file_signature(model["config"]) for model in TRAIN_MODELS}
    inf_model_sigs = {model["config"]: _file_signature(model["config"]) for model in INF_MODELS}

    cache = _load_cache()
    for bw_gbps in BW_SWEEP_GBPS:
        _run_for_bandwidth(
            bw_gbps=bw_gbps,
            train_hw_base=train_hw_base,
            inf_hw_base=inf_hw_base,
            train_hw_sig=train_hw_sig,
            inf_hw_sig=inf_hw_sig,
            train_model_sigs=train_model_sigs,
            inf_model_sigs=inf_model_sigs,
            cache=cache,
        )


if __name__ == "__main__":
    main()
