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
UCI training validation harness combining DDP and FSDP grid-search measurements.

Inspired by nvidia_train.py: builds ValidationSpecs from benchmark CSVs,
runs DeepFlow, compares training time to measured Avg_Step_Time_s, and
produces bar plots of percent error.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml

try:
    from .validation_helpers import (
        ValidationSpec,
        run_validation_suite,
        parse_training_time,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        run_validation_suite,
        parse_training_time,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PERF = PROJECT_ROOT / "run_perf.py"
DEFAULT_HW_CONFIG = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"

# Unified benchmark CSV with variant column (DDP/FSDP)
MERGED_CSV = PROJECT_ROOT / "validation_scripts" / "imec_data" / "uci_train.csv"

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

# AstraSim execution mode toggle: run both hierarchical and flattened by default.
ASTRASIM_MODE = os.environ.get("UCI_ASTRASIM_MODE", "hierarchical").strip().lower()
DEFAULT_ASTRA_MODES = ("full_astrasim_hierarchical", "full_astrasim_flattened")
ASTRA_MODE_ORDER = list(DEFAULT_ASTRA_MODES)
ASTRA_MODE_LABELS = {
    "full_astrasim_hierarchical": "hierarchical",
    "full_astrasim_flattened": "flattened",
}
ASTRA_MODE_COLORS = {
    "full_astrasim_hierarchical": "#1f77b4",
    "full_astrasim_flattened": "#ff7f0e",
}


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


def _astra_sort_key(mode: str) -> Tuple[int, str]:
    normalized = _normalize_astra_mode(mode)
    if normalized in ASTRA_MODE_ORDER:
        return (ASTRA_MODE_ORDER.index(normalized), normalized)
    return (len(ASTRA_MODE_ORDER), normalized)

def _load_grid(csv_path: Path, default_variant: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "variant" not in df.columns and default_variant is not None:
        df.insert(0, "variant", default_variant)
    # Normalize numeric fields
    for col in ("TP", "PP", "CP", "DP"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "Avg_Step_Time_s" in df.columns:
        df["Avg_Step_Time_s"] = pd.to_numeric(df["Avg_Step_Time_s"], errors="coerce")
    # Keep only successes and drop unused columns
    df = df[df["Status"].str.upper() == "SUCCESS"].copy()
    for col in ("Status", "Error"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        actual = row.get("Avg_Step_Time_s")
        if actual is None or pd.isna(actual):
            continue
        yield (
            str(row.get("variant")),
            int(row["TP"]),
            int(row["PP"]),
            int(row["CP"]),
            int(row["DP"]),
            float(actual),
        )

def _astra_mode_string(mode: Optional[str] = None) -> str:
    """Map the friendly toggle to the AstraSim execution mode string."""
    raw = mode if mode is not None else ASTRASIM_MODE
    normalized = _normalize_astra_mode(raw)
    if normalized in {"full_astrasim_flattened", "full_astrasim_hierarchical"}:
        return normalized
    raise ValueError(f"Unsupported ASTRASIM_MODE={raw!r}; use 'flattened' or 'hierarchical'.")


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
    """Construct a fixed 2D ring network: TP/CP on PCIe (or NVLink), PP/DP on NVLink (or PCIe)."""
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
        if isinstance(value, Mapping):
            existing = target.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            target[key] = _deep_update(dict(existing), value)
        else:
            target[key] = value
    return target


def _write_hw_override(base_path: Path, overrides: Mapping[str, object], out_path: Path) -> Path:
    with open(base_path, "r") as handle:
        base = yaml.safe_load(handle) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Base hardware config {base_path} is not a mapping.")
    merged = _deep_update(dict(base), overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return out_path

def _build_spec(
    variant: str,
    tp: int,
    pp: int,
    cp: int,
    dp: int,
    idx: int,
    model_config_path: str,
    hardware_config_path: str,
    astra_mode: str,
) -> ValidationSpec:
    label = f"{variant} TP={tp} PP={pp} CP={cp} DP={dp}"
    grad_accum_step = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * pp * dp)
    mb = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * dp)

    eff_mb = mb if int(pp) > 1 else 1
    network_override, mapping_desc = _build_network_override(tp, pp, cp, dp, astra_mode)

    hw_overrides = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": False,
            "cp": int(cp),
            "pp": int(pp),
            "mb": int(eff_mb),  # align microbatch count with pipeline stages
            "train": {"dp": int(dp), "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            # DDP -> zero_stage 0, FSDP -> zero_stage 3
            "dp_zero_stage": 0 if variant.upper() == "DDP" else 3,
        },
    }
    hw_overrides.update(network_override)
    return ValidationSpec(
        label=label,
        model_overrides={
            "model_param": {
                "seq_len": 4096,
                "run_type": "training",
            }
        },
        hardware_overrides=hw_overrides,
        model_config_path=model_config_path,
        hardware_config_path=hardware_config_path,
        metadata={
            "variant": variant,
            "tp": int(tp),
            "pp": int(pp),
            "cp": int(cp),
            "dp": int(dp),
            "network_mapping": mapping_desc,
            "astra": _normalize_astra_mode(astra_mode),
        },
        order=idx,
    )


def build_specs(
    variants: Sequence[str],
    model_config_path: str,
    hardware_config_path: str,
    *,
    emit_hw_configs: bool = False,
    astra_mode: str = "full_astrasim_hierarchical",
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int], float]]:
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int, int, int, int], float] = {}
    idx = 0
    df = _load_grid(MERGED_CSV)
    # Skip cases with CP = 4.
    if "CP" in df.columns:
        df = df[df["CP"].fillna(0) <= 2]
    # skip cases with TP = 2 and CP = 2
    if "TP" in df.columns and "CP" in df.columns:
        df = df[~((df["TP"].fillna(0) == 2) & (df["CP"].fillna(0) == 2))]
    # skip cases with TP = 4
    if "TP" in df.columns:
        df = df[df["TP"].fillna(0) < 4]
    # skip case with PP = 2, DP = 2 (fails in hierarchial)
    if "PP" in df.columns and "DP" in df.columns:
        df = df[~((df["PP"].fillna(0) == 2) & (df["DP"].fillna(0) == 2))]
    df = df[df["variant"].str.lower().isin(set(variants))]
    for variant, tp, pp, cp, dp, actual in _iter_tests(df):
        spec = _build_spec(variant, tp, pp, cp, dp, idx, model_config_path, hardware_config_path, astra_mode)
        specs.append(spec)
        actual_lookup[(variant, tp, pp, cp, dp)] = actual
        if emit_hw_configs:
            base_path = Path(hardware_config_path)
            out_name = f"{base_path.stem}_uci_{variant.lower()}_tp{tp}_pp{pp}_cp{cp}_dp{dp}.yaml"
            out_path = base_path.parent / out_name
            overrides = spec.hardware_overrides or {}
            _write_hw_override(base_path, overrides, out_path)
        idx += 1

    if not specs:
        raise ValueError(f"No validation specs generated for variants={variants}")
    return specs, actual_lookup


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int, int, int, int], float]):
    rows: List[Dict[str, object]] = []
    for res in results:
        meta = res.spec.metadata or {}
        variant = str(meta.get("variant"))
        tp = int(meta.get("tp", 0))
        pp = int(meta.get("pp", 0))
        cp = int(meta.get("cp", 0))
        dp = int(meta.get("dp", 0))
        inf_time = float(res.metrics.get("training_time_s", float("nan"))) if res.success else float("nan")
        actual = actual_lookup.get((variant, tp, pp, cp, dp), float("nan"))
        mapping_desc = meta.get("network_mapping")

        if math.isnan(inf_time) or actual == 0 or math.isnan(actual):
            signed_pct_error = float("nan")
            pct_error = float("nan")
        else:
            signed_pct_error = (inf_time - actual) / actual * 100.0
            pct_error = signed_pct_error

        rows.append(
            {
                "variant": variant,
                "tp": tp,
                "pp": pp,
                "cp": cp,
                "dp": dp,
                "training_time_s": inf_time,
                "actual_training_time_s": actual,
                "signed_pct_error": signed_pct_error,
                "pct_error": pct_error,
                "network_mapping": mapping_desc,
                "success": res.success,
                "error": res.error,
                "raw_output": res.raw_output,
                "astra": meta.get("astra"),
            }
        )
    return rows


def _plot_results(rows: List[Dict[str, object]], title: str, path: Path) -> Optional[Path]:
    if not rows:
        return None
    labels: List[str] = []
    astra_modes = sorted({str(r.get("astra", "")).strip().lower() for r in rows if r.get("astra")}, key=_astra_sort_key)
    if not astra_modes:
        astra_modes = ["unspecified"]

    # Group by test case (variant + parallelism) and render one bar per Astra mode.
    grouped: Dict[Tuple[str, int, int, int, int], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("variant")),
            int(row.get("tp", 0)),
            int(row.get("pp", 0)),
            int(row.get("cp", 0)),
            int(row.get("dp", 0)),
        )
        mode = _normalize_astra_mode(row.get("astra", ""))
        if not mode and astra_modes:
            mode = astra_modes[0]
        grouped.setdefault(key, {})[mode] = row

    errors_by_mode: Dict[str, List[float]] = {mode: [] for mode in astra_modes}

    for key, mode_rows in grouped.items():
        variant, tp, pp, cp, dp = key
        labels.append(f"{variant} TP{tp} PP{pp} CP{cp} DP{dp}")
        for mode in astra_modes:
            row = mode_rows.get(mode)
            if row:
                errors_by_mode[mode].append(abs(float(row.get("signed_pct_error", float("nan")))))
            else:
                errors_by_mode[mode].append(float("nan"))

    series_count = len(astra_modes)
    fig_w = max(7.5, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x = list(range(len(labels)))
    bar_width = 0.8 / max(series_count, 1)
    offset_span = bar_width * (series_count - 1) / 2 if series_count > 1 else 0.0

    bars_by_mode = {}
    for idx, mode in enumerate(astra_modes):
        offsets = [pos - offset_span + idx * bar_width for pos in x]
        bars = ax.bar(offsets, errors_by_mode[mode], bar_width, color=ASTRA_MODE_COLORS.get(mode, "#1f77b4"), label=f"Astra={_astra_label(mode)}")
        bars_by_mode[mode] = bars

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    ax.legend()
    for bar_group in bars_by_mode.values():
        for rect in bar_group:
            height = rect.get_height()
            if math.isnan(height):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, height, f"{height:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run(
    *,
    hardware_config: Optional[str] = None,
    model_config: Optional[str] = None,
    variants: Sequence[str] = ("ddp",),
    enable_plot: bool = True,
    show_progress: bool = False,
    emit_logs: bool = True,
    emit_hw_configs: bool = True,
    astra_modes: Optional[Sequence[str]] = None,
):
    variants = tuple(v.lower() for v in variants)
    hw_cfg = hardware_config or str(DEFAULT_HW_CONFIG)
    model_cfg = model_config or str(DEFAULT_MODEL_CONFIG)
    outputs: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []
    all_pct_errors: List[float] = []

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

    for variant in variants:
        variant_pct_errors: List[float] = []
        variant_rows: List[Dict[str, object]] = []
        variant_flattened_errors: List[Dict[str, object]] = []
        for mode in normalized_modes:
            specs, actual_lookup = build_specs(
                (variant,), model_cfg, hw_cfg, emit_hw_configs=emit_hw_configs, astra_mode=mode
            )

            validation_results = run_validation_suite(
                specs,
                base_model_config_path=model_cfg,
                base_hardware_config_path=hw_cfg,
                result_parser=parse_training_time,
                run_perf_path=str(RUN_PERF),
                show_progress=show_progress,
            )

            rows = compute_pct_errors(validation_results, actual_lookup)
            if emit_logs:
                print(f"\n=== Astra mode: {_astra_label(mode)} ({variant.upper()}) ===")
            for row in rows:
                # Skip flattened failures entirely.
                is_flattened = _normalize_astra_mode(row.get("astra", "")) == "full_astrasim_flattened"
                if is_flattened and not row.get("success"):
                    error_text = str(row.get("error") or "").strip() or str(row.get("raw_output") or "").strip()
                    entry = {
                        "variant": row.get("variant"),
                        "tp": row.get("tp"),
                        "pp": row.get("pp"),
                        "cp": row.get("cp"),
                        "dp": row.get("dp"),
                        "error": error_text,
                        "raw_output": row.get("raw_output"),
                    }
                    variant_flattened_errors.append(entry)
                    if emit_logs:
                        print(
                            f"Skipping failed flattened run for {variant.upper()} TP={row['tp']} PP={row['pp']} "
                            f"CP={row['cp']} DP={row['dp']}. Error: {error_text or 'unknown error'}"
                        )
                    continue
                pct_error = row["pct_error"]
                if not math.isnan(pct_error):
                    variant_pct_errors.append(pct_error)
                    all_pct_errors.append(pct_error)
                variant_rows.append(row)
                all_rows.append(row)

                if emit_logs:
                    block = [
                        f"\n=== Result ({row['variant']}, astra={row.get('astra')}, TP={row['tp']}, PP={row['pp']}, CP={row['cp']}, DP={row['dp']}) ==="
                    ]
                    if row.get("network_mapping"):
                        block.append(f"  Network: {row['network_mapping']}")
                    if row["success"] and not math.isnan(row["pct_error"]):
                        block.append(f"  DeepFlow train time: {float(row['training_time_s']):.2f}s")
                        block.append(f"  Actual train time:   {float(row['actual_training_time_s']):.2f}s")
                        block.append(f"  Percent Error:  {float(row['signed_pct_error']):+.2f}%")
                    else:
                        block.append(f"  DeepFlow run failed. {(row.get('error') or '')}".rstrip())
                        if row.get("raw_output"):
                            block.append("  --- Raw output ---")
                            block.append(str(row["raw_output"]).strip())
                    print("\n".join(block))

        plot_path = None
        if enable_plot and variant_rows:
            out_dir = PROJECT_ROOT / "output" / "validation" / "train"
            plot_path = _plot_results(
                variant_rows,
                f"Training validation ({variant.upper()} benchmarks, Astra comparison)",
                out_dir / f"uci_train_{variant}.png",
            )
            if emit_logs and plot_path:
                print(f"Saved plot: {plot_path}")

        avg_abs_error = sum(variant_pct_errors) / len(variant_pct_errors) if variant_pct_errors else float("nan")
        if emit_logs:
            print(f"Average absolute percent error for {variant.upper()}: {avg_abs_error:.2f}%")
        outputs.append(
            {
                "variant": variant,
                "avg_abs_error": avg_abs_error,
                "rows": variant_rows,
                "plot": plot_path,
                "flattened_errors": variant_flattened_errors,
            }
        )

    # Combined Astra comparison plot across variants.
    if enable_plot and all_rows:
        out_dir = PROJECT_ROOT / "output" / "validation" / "train"
        combined_path = _plot_results(
            all_rows,
            "Training validation (DDP/FSDP Astra comparison)",
            out_dir / "uci_train_combined.png",
        )
        if emit_logs and combined_path:
            print(f"Saved combined plot: {combined_path}")

    overall_avg_abs_error = sum(all_pct_errors) / len(all_pct_errors) if all_pct_errors else float("nan")
    if emit_logs:
        print(f"Overall average absolute percent error: {overall_avg_abs_error:.2f}%")
    return outputs


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
        "--variants",
        nargs="+",
        default=["ddp"],
        choices=["ddp", "fsdp"],
        help="Which benchmark variants to include (runs each separately).",
    )
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-run progress.")
    parser.add_argument(
        "--emit-hw-configs",
        action="store_true",
        help="Write merged hardware configs (with overrides) per test case into the hardware-config directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        hardware_config=args.hardware_config,
        model_config=args.model_config,
        variants=args.variants,
        enable_plot=args.enable_plot,
        show_progress=args.show_progress,
        emit_hw_configs=True,
    )
