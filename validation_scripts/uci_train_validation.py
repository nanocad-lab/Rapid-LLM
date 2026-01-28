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
UCI training validation harness.
Standalone script: no imports from other project modules.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PERF = PROJECT_ROOT / "run_perf.py"
DEFAULT_HW_CONFIG = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB_train_validation.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
TRAIN_VALIDATION_DATA = PROJECT_ROOT / "validation_scripts" / "train_validation_data"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "validation_scripts" / "train_validation_data" / "uci_train.csv"
DEFAULT_INCLUDE_CSV = PROJECT_ROOT / "validation_scripts" / "train_validation_data" / "uci_train.csv"

# run_benchmark.py parameters
GLOBAL_BATCH_SIZE = 128
MICRO_BATCH_SIZE = 1

# Network modeling constants (PCIe ring for TP/CP, NVLink ring for PP/DP)
TP_ON_NVLINK = False
NVLINK_LINK_BW_GB = 12 * 25  # GB/s per GPU worth of NVLink throughput
PCIE_LINK_BW_GB = 25  # GB/s per GPU PCIe
NVLINK_LATENCY_S = 1e-6
PCIE_LATENCY_S = 5e-6
ENERGY_PER_BIT = 8e-12

DEFAULT_ASTRA_MODES = ("full_astrasim_hierarchical",)
ASTRA_MODE_ORDER = list(DEFAULT_ASTRA_MODES)
ASTRA_MODE_LABELS = {
    "full_astrasim_hierarchical": "hierarchical",
    "full_astrasim_flattened": "flattened",
}

COMPARE_COLOR_MAP: Dict[str, str] = {
    "actual": "#4c566a",
    "rapid_llm": "#1f77b4",
    "stg": "#ff7f0e",
    "mlsynth": "#2ca02c",
}
COMPARE_LABELS: Dict[str, str] = {
    "actual": "Actual",
    "rapid_llm": "Rapid-LLM",
    "stg": "STAGE",
    "mlsynth": "MLSynth",
}

_TRAIN_TIME_PATTERN = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")


@dataclass(frozen=True)
class Case:
    variant: str
    tp: int
    pp: int
    cp: int
    dp: int
    actual: float


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{timestamp}_{os.getpid()}"


def _normalize_astra_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"hierarchical", "full_astrasim_hierarchical"}:
        return "full_astrasim_hierarchical"
    if normalized in {"flattened", "full_astrasim_flattened"}:
        return "full_astrasim_flattened"
    return normalized


def _astra_label(mode: str) -> str:
    normalized = _normalize_astra_mode(mode)
    return ASTRA_MODE_LABELS.get(normalized, normalized)


def _astra_mode_string(mode: str) -> str:
    normalized = _normalize_astra_mode(mode)
    if normalized in {"full_astrasim_flattened", "full_astrasim_hierarchical"}:
        return normalized
    raise ValueError(f"Unsupported Astra mode: {mode!r}")


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _get_int(row: Mapping[str, str], *keys: str) -> Optional[int]:
    for key in keys:
        if key in row and row[key] not in ("", None):
            try:
                return int(float(row[key]))
            except ValueError:
                return None
    return None


def _case_key(variant: str, tp: int, pp: int, cp: int, dp: int) -> Tuple[str, int, int, int, int]:
    return (variant.lower(), tp, pp, cp, dp)


def _load_case_keys(csv_path: Path) -> set[Tuple[str, int, int, int, int]]:
    if not csv_path.exists():
        return set()
    keys: set[Tuple[str, int, int, int, int]] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        variant = str(row.get("variant") or row.get("Variant") or "ddp").strip().lower()
        tp = _get_int(row, "TP", "tp")
        pp = _get_int(row, "PP", "pp")
        cp = _get_int(row, "CP", "cp")
        dp = _get_int(row, "DP", "dp")
        if None in (tp, pp, cp, dp):
            continue
        keys.add(_case_key(variant, tp, pp, cp, dp))
    return keys


def _load_cases(
    csv_path: Path,
    variants: Sequence[str],
    *,
    include_keys: Optional[set[Tuple[str, int, int, int, int]]] = None,
    exclude_keys: Optional[set[Tuple[str, int, int, int, int]]] = None,
) -> List[Case]:
    cases: List[Case] = []
    variants_set = {v.lower() for v in variants}
    exclude_keys = exclude_keys or set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        status = str(row.get("Status", "")).strip().upper()
        if status and status != "SUCCESS":
            continue
        variant = str(row.get("variant") or row.get("Variant") or "ddp").strip()
        if variant.lower() not in variants_set:
            continue
        tp = _get_int(row, "TP", "tp")
        pp = _get_int(row, "PP", "pp")
        cp = _get_int(row, "CP", "cp")
        dp = _get_int(row, "DP", "dp")
        if None in (tp, pp, cp, dp):
            continue
        actual = row.get("Avg_Step_Time_s")
        if actual in (None, ""):
            actual = row.get("actual_time")
        if actual in (None, ""):
            continue
        actual_val = float(actual)

        # Apply same filters as uci_train_lime.
        if cp > 2:
            continue
        if tp == 2 and cp == 2:
            continue
        if tp >= 4:
            continue
        if pp == 2 and dp == 2:
            continue

        key = _case_key(variant, tp, pp, cp, dp)
        if include_keys is not None and key not in include_keys:
            continue
        if key in exclude_keys:
            continue
        cases.append(Case(variant=variant, tp=tp, pp=pp, cp=cp, dp=dp, actual=actual_val))

    return cases


def _bandwidth_label_gb(value_gb: float) -> str:
    return f"{value_gb:g} GB"


def _make_dimension(idx: int, label: str, axes: List[str], bandwidth_gb: float, latency_s: float, topology: str) -> Dict[str, object]:
    size: object = "auto" if axes else 1
    return {
        "id": f"dim{idx}",
        "label": label,
        "size": size,
        "topology": {
            "type": topology,
            "bandwidth": _bandwidth_label_gb(bandwidth_gb),
            "latency": latency_s,
            "energy_per_bit": ENERGY_PER_BIT,
            "util": 1.0,
            "optimize_2dmap": False,
        },
        "collective_override": {},
        "parallelisms": axes,
    }


def _build_network_override(tp: int, pp: int, cp: int, dp: int, astra_mode: str) -> Tuple[Dict[str, object], str]:
    tp_cp_bw = NVLINK_LINK_BW_GB if TP_ON_NVLINK else PCIE_LINK_BW_GB
    tp_cp_lat = NVLINK_LATENCY_S if TP_ON_NVLINK else PCIE_LATENCY_S
    pp_dp_bw = PCIE_LINK_BW_GB if TP_ON_NVLINK else NVLINK_LINK_BW_GB
    pp_dp_lat = PCIE_LATENCY_S if TP_ON_NVLINK else NVLINK_LATENCY_S
    tp_cp_label = "NVLink" if TP_ON_NVLINK else "PCIe"
    pp_dp_label = "PCIe" if TP_ON_NVLINK else "NVLink"

    dims = [
        _make_dimension(0, "tp_cp_ring", ["tp", "cp"], tp_cp_bw, tp_cp_lat, "Ring"),
        _make_dimension(1, "pp_dp_ring", ["pp", "dp"], pp_dp_bw, pp_dp_lat, "Ring"),
    ]
    mapping_desc = (
        f"2D Ring dim0=tp/cp ({tp_cp_label}, tp={tp}, cp={cp}), "
        f"dim1=pp/dp ({pp_dp_label}, pp={pp}, dp={dp})"
    )
    network_override: Dict[str, object] = {
        "network": {
            "dimensions": dims,
        },
        "execution_backend": {
            "astra": {
                "mode": _astra_mode_string(astra_mode),
            }
        },
    }
    return network_override, mapping_desc


def _deep_update(target: Dict[str, object], overrides: Mapping[str, object]) -> Dict[str, object]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            target[key] = _deep_update(dict(target[key]), value)  # type: ignore[index]
        else:
            target[key] = value
    return target


def _merge_dicts(base: Dict[str, object], overrides: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not overrides:
        return dict(base)
    merged = dict(base)
    _deep_update(merged, overrides)
    return merged


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {path}, got {type(data).__name__}")
    return data


def _write_yaml(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _materialize_configs(
    base_model_path: Path,
    base_hw_path: Path,
    model_overrides: Mapping[str, object],
    hardware_overrides: Mapping[str, object],
    dest_dir: Path,
) -> Tuple[Path, Path]:
    base_model = _load_yaml(base_model_path)
    base_hw = _load_yaml(base_hw_path)
    model_cfg = _merge_dicts(base_model, model_overrides)
    hw_cfg = _merge_dicts(base_hw, hardware_overrides)

    model_path = dest_dir / "model.yaml"
    hw_path = dest_dir / "hardware.yaml"
    _write_yaml(model_path, model_cfg)
    _write_yaml(hw_path, hw_cfg)
    return model_path, hw_path


def _parse_training_time(output: str) -> Optional[float]:
    match = _TRAIN_TIME_PATTERN.search(output or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _run_case(
    case: Case,
    idx: int,
    model_config_path: Path,
    hardware_config_path: Path,
    run_root: Path,
    astra_mode: str,
) -> Dict[str, object]:
    label = f"{case.variant.upper()} TP={case.tp} PP={case.pp} CP={case.cp} DP={case.dp}"
    slug = "_".join(label.replace("=", "_").split())
    spec_root = run_root / slug
    config_dir = spec_root / "configs"
    log_path = spec_root / "worker.log"
    spec_root.mkdir(parents=True, exist_ok=True)

    mb = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * case.dp)
    eff_mb = mb if int(case.pp) > 1 else 1
    network_override, mapping_desc = _build_network_override(case.tp, case.pp, case.cp, case.dp, astra_mode)

    model_overrides = {
        "model_param": {
            "seq_len": 4096,
            "run_type": "training",
        }
    }
    hw_overrides: Dict[str, object] = {
        "parallelism": {
            "tp": int(case.tp),
            "tp_sp": False,
            "cp": int(case.cp),
            "pp": int(case.pp),
            "mb": int(eff_mb),
            "train": {"dp": int(case.dp), "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            "dp_zero_stage": 0 if case.variant.upper() == "DDP" else 3,
        },
    }
    hw_overrides.update(network_override)

    model_path, hw_path = _materialize_configs(
        model_config_path,
        hardware_config_path,
        model_overrides,
        hw_overrides,
        config_dir,
    )

    env = os.environ.copy()
    env.setdefault("RAPID_ASTRA_CACHE_MODE", "NO_CACHE")
    env.setdefault("DEEPFLOW_ASTRA_CACHE_MODE", "no_cache")
    env["ASTRA_CACHE_DIR"] = str(spec_root / "astra_cache")

    cmd = [
        sys.executable,
        str(RUN_PERF),
        "--hardware_config",
        str(hw_path),
        "--model_config",
        str(model_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=spec_root,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_text = result.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output_text, encoding="utf-8")

    training_time = _parse_training_time(output_text) if result.returncode == 0 else None
    actual_time = case.actual
    if training_time is None or math.isnan(actual_time) or actual_time == 0:
        signed_pct_error = float("nan")
        pct_error = float("nan")
        success = False
    else:
        signed_pct_error = (training_time - actual_time) / actual_time * 100.0
        pct_error = signed_pct_error
        success = True

    error_text = "" if result.returncode == 0 else f"return code {result.returncode}"

    return {
        "variant": case.variant,
        "tp": case.tp,
        "pp": case.pp,
        "cp": case.cp,
        "dp": case.dp,
        "training_time_s": training_time if training_time is not None else float("nan"),
        "actual_training_time_s": actual_time,
        "signed_pct_error": signed_pct_error,
        "pct_error": pct_error,
        "network_mapping": mapping_desc,
        "success": success,
        "error": error_text,
        "raw_output": output_text,
        "astra": _normalize_astra_mode(astra_mode),
    }


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _norm_variant(value: object) -> str:
    return str(value).strip().lower()


def _compare_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        _norm_variant(row.get("variant", "")),
        str(row.get("tp", "")),
        str(row.get("pp", "")),
        str(row.get("cp", "")),
        str(row.get("dp", "")),
    )


def _case_order_key(case: Case) -> Tuple[str, str, str, str, str]:
    return (
        case.variant.strip().lower(),
        str(case.tp),
        str(case.pp),
        str(case.cp),
        str(case.dp),
    )


def _format_compare_label(row: Mapping[str, object]) -> str:
    return f"tp{row.get('tp', '')}-cp{row.get('cp', '')}-pp{row.get('pp', '')}-dp{row.get('dp', '')}"


def _build_compare_series(
    deepflow_rows: List[Dict[str, object]],
    stage_rows: List[Dict[str, str]],
    *,
    key_order: Optional[Sequence[Tuple[str, str, str, str, str]]] = None,
) -> Tuple[List[str], Dict[str, List[float]]]:
    labels: List[str] = []
    actual: List[float] = []
    rapid: List[float] = []
    stg: List[float] = []
    mlsynth: List[float] = []

    deepflow_map: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for row in deepflow_rows:
        key = _compare_key(row)
        if key not in deepflow_map:
            deepflow_map[key] = row

    stage_map: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    for row in stage_rows:
        key = _compare_key(row)
        if key not in stage_map:
            stage_map[key] = row

    keys = list(key_order) if key_order is not None else (list(deepflow_map.keys()) if deepflow_map else list(stage_map.keys()))
    for key in keys:
        df_row = deepflow_map.get(key, {})
        stg_row = stage_map.get(key, {})
        labels.append(_format_compare_label(df_row or stg_row))
        actual.append(_to_float(df_row.get("actual_training_time_s")))
        rapid.append(_to_float(df_row.get("training_time_s")))
        stg.append(_to_float(stg_row.get("stg_seconds")))
        mlsynth.append(_to_float(stg_row.get("mlsynth_seconds")))

    return labels, {
        "actual": actual,
        "rapid_llm": rapid,
        "stg": stg,
        "mlsynth": mlsynth,
    }


def _plot_compare_results(
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    output: Path,
    title: str,
) -> Path:
    tool_list = ["actual", "rapid_llm", "stg", "mlsynth"]
    width = 0.15
    x = list(range(len(labels)))

    fig_w = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    for idx, name in enumerate(tool_list):
        offsets = [pos + (idx - (len(tool_list) - 1) / 2) * width for pos in x]
        label = COMPARE_LABELS.get(name, name)
        values = series.get(name, [])
        ax.bar(
            offsets,
            values,
            width=width,
            label=label,
            color=COMPARE_COLOR_MAP.get(name, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Training time (s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def run(
    *,
    hardware_config: Optional[str] = None,
    model_config: Optional[str] = None,
    variants: Sequence[str] = ("ddp",),
    input_csv: Optional[Path] = None,
    include_csv: Optional[Path] = None,
    exclude_csv: Optional[Path] = None,
    enable_plot: bool = True,
    show_progress: bool = False,
    output_csv: Optional[Path] = None,
    stage_csv: Optional[Path] = None,
    compare_plot_output: Optional[Path] = None,
    astra_modes: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> List[Dict[str, object]]:
    hw_cfg = Path(hardware_config) if hardware_config else DEFAULT_HW_CONFIG
    model_cfg = Path(model_config) if model_config else DEFAULT_MODEL_CONFIG
    include_keys = _load_case_keys(include_csv) if include_csv else None
    exclude_keys = _load_case_keys(exclude_csv) if exclude_csv else set()
    cases = _load_cases(
        input_csv or DEFAULT_INPUT_CSV,
        variants,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
    )
    if not cases:
        raise ValueError("No validation cases found.")

    astra_modes_list = list(astra_modes) if astra_modes is not None else list(DEFAULT_ASTRA_MODES)
    normalized_modes: List[str] = []
    seen_modes = set()
    for mode in astra_modes_list:
        normalized = _normalize_astra_mode(mode)
        if normalized not in seen_modes:
            normalized_modes.append(normalized)
            seen_modes.add(normalized)
    if not normalized_modes:
        normalized_modes = list(DEFAULT_ASTRA_MODES)

    run_root = TRAIN_VALIDATION_DATA / "uci_train_validation_runs" / _new_run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    worker_cap = workers if workers is not None else (os.cpu_count() or 1)
    worker_count = max(1, min(len(cases), int(worker_cap)))

    for mode in normalized_modes:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_case, case, idx, model_cfg, hw_cfg, run_root, mode): idx
                for idx, case in enumerate(cases)
            }
            for fut in as_completed(futures):
                row = fut.result()
                all_rows.append(row)
                if show_progress:
                    label = f"{row.get('variant')} TP={row.get('tp')} PP={row.get('pp')} CP={row.get('cp')} DP={row.get('dp')}"
                    status = "ok" if row.get("success") else "error"
                    print(f"[uci_train_validation] finished {label} ({status})")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        seen = set()
        for row in all_rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(key)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[uci_train_validation] wrote results to {output_csv}")

    print(f"[uci_train_validation] run artifacts in {run_root}")

    if enable_plot and stage_csv and compare_plot_output:
        if stage_csv.exists():
            stage_rows = _read_csv(stage_csv)
            case_keys = [_case_order_key(case) for case in cases]
            labels, series = _build_compare_series(all_rows, stage_rows, key_order=case_keys)
            if labels:
                _plot_compare_results(
                    list(reversed(labels)),
                    {name: list(reversed(values)) for name, values in series.items()},
                    compare_plot_output,
                    "Runtime comparison on 4-GPU system",
                )
                print(f"[uci_train_validation] wrote plot to {compare_plot_output}")

    return all_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCI training validation (DDP + FSDP gridsearch).")
    parser.add_argument(
        "--hardware_config",
        default=str(DEFAULT_HW_CONFIG),
        help=f"Path to hardware config YAML (default: {DEFAULT_HW_CONFIG}).",
    )
    parser.add_argument(
        "--model_config",
        default=str(DEFAULT_MODEL_CONFIG),
        help=f"Path to model config YAML (default: {DEFAULT_MODEL_CONFIG}).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"CSV with UCI validation cases (default: {DEFAULT_INPUT_CSV}).",
    )
    parser.add_argument(
        "--include-csv",
        type=Path,
        default=DEFAULT_INCLUDE_CSV,
        help=f"CSV with cases to include (default: {DEFAULT_INCLUDE_CSV}).",
    )
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help="CSV with cases to exclude (optional).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ddp"],
        choices=["ddp", "fsdp"],
        help="Which benchmark variants to include (runs each separately).",
    )
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-run progress.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=TRAIN_VALIDATION_DATA / "uci_train_validation_result.csv",
        help="CSV output path for validation results.",
    )
    parser.add_argument(
        "--stage-mlsynth-csv",
        type=Path,
        default=TRAIN_VALIDATION_DATA / "uci_train_stg_mlsynth.csv",
        help="CSV with STAGE/MLSynth results for comparison plot.",
    )
    parser.add_argument(
        "--compare-plot-output",
        type=Path,
        default=TRAIN_VALIDATION_DATA / "uci_train_validation_compare.png",
        help="Output path for the comparison plot.",
    )
    parser.add_argument(
        "--astra-modes",
        nargs="+",
        default=None,
        help="Override Astra modes (e.g., full_astrasim_hierarchical).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: CPU count, capped by case count).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        hardware_config=args.hardware_config,
        model_config=args.model_config,
        input_csv=args.input_csv,
        include_csv=args.include_csv,
        exclude_csv=args.exclude_csv,
        variants=args.variants,
        enable_plot=args.enable_plot,
        show_progress=args.show_progress,
        output_csv=args.output_csv,
        stage_csv=args.stage_mlsynth_csv,
        compare_plot_output=args.compare_plot_output,
        astra_modes=args.astra_modes,
        workers=args.workers,
    )
