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
runs RAPID-LLM, compares training time to measured Avg_Step_Time_s, and
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


def _build_spec(
    variant: str,
    tp: int,
    pp: int,
    cp: int,
    dp: int,
    idx: int,
    model_config_path: str,
    hardware_config_path: str,
) -> ValidationSpec:
    label = f"{variant} TP={tp} PP={pp} CP={cp} DP={dp}"
    hw_overrides = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": False,
            "cp": int(cp),
            "pp": int(pp),
            "mb": int(pp),  # align microbatch count with pipeline stages
            "train": {"dp": int(dp), "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            # DDP -> zero_stage 0, FSDP -> zero_stage 3
            "dp_zero_stage": 0 if variant.upper() == "DDP" else 3,
        },
    }
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
        },
        order=idx,
    )


def build_specs(
    variants: Sequence[str],
    model_config_path: str,
    hardware_config_path: str,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int], float]]:
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int, int, int, int], float] = {}
    idx = 0
    df = _load_grid(MERGED_CSV)
    df = df[df["variant"].str.lower().isin(set(variants))]
    for variant, tp, pp, cp, dp, actual in _iter_tests(df):
        spec = _build_spec(variant, tp, pp, cp, dp, idx, model_config_path, hardware_config_path)
        specs.append(spec)
        actual_lookup[(variant, tp, pp, cp, dp)] = actual
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

        if math.isnan(inf_time) or actual == 0 or math.isnan(actual):
            signed_pct_error = float("nan")
            pct_error = float("nan")
        else:
            signed_pct_error = (inf_time - actual) / actual * 100.0
            pct_error = abs(signed_pct_error)

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
                "success": res.success,
                "error": res.error,
                "raw_output": res.raw_output,
            }
        )
    return rows


def _plot_results(rows: List[Dict[str, object]], title: str, path: Path) -> Optional[Path]:
    if not rows:
        return None
    labels: List[str] = []
    errors: List[float] = []
    colors: List[str] = []
    color_map = {"DDP": "#1f77b4", "FSDP": "#ff8c00"}

    for row in rows:
        variant = str(row.get("variant"))
        tp = int(row.get("tp", 0))
        pp = int(row.get("pp", 0))
        cp = int(row.get("cp", 0))
        dp = int(row.get("dp", 0))
        labels.append(f"{variant} TP{tp} PP{pp} CP{cp} DP{dp}")
        errors.append(float(row.get("signed_pct_error", float("nan"))))
        colors.append(color_map.get(variant.upper(), "#888888"))

    fig_w = max(6.0, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bars = ax.bar(range(len(errors)), errors, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#ff8c00"),
    ]
    ax.legend(handles, ["DDP", "FSDP"])
    for rect, err in zip(bars, errors):
        if math.isnan(err):
            continue
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
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
    variants: Sequence[str] = ("ddp", "fsdp"),
    enable_plot: bool = True,
    show_progress: bool = False,
    emit_logs: bool = True,
):
    variants = tuple(v.lower() for v in variants)
    hw_cfg = hardware_config or str(DEFAULT_HW_CONFIG)
    model_cfg = model_config or str(DEFAULT_MODEL_CONFIG)
    outputs: List[Dict[str, object]] = []
    for variant in variants:
        specs, actual_lookup = build_specs((variant,), model_cfg, hw_cfg)

        validation_results = run_validation_suite(
            specs,
            base_model_config_path=model_cfg,
            base_hardware_config_path=hw_cfg,
            result_parser=parse_training_time,
            run_perf_path=str(RUN_PERF),
            show_progress=show_progress,
        )

        rows = compute_pct_errors(validation_results, actual_lookup)
        pct_errors = [r["pct_error"] for r in rows if not math.isnan(r["pct_error"])]

        if emit_logs:
            for row in rows:
                block = [
                    f"\n=== Result ({row['variant']}, TP={row['tp']}, PP={row['pp']}, CP={row['cp']}, DP={row['dp']}) ==="
                ]
                if row["success"] and not math.isnan(row["pct_error"]):
                    block.append(f"  RAPID-LLM train time: {float(row['training_time_s']):.2f}s")
                    block.append(f"  Actual train time:   {float(row['actual_training_time_s']):.2f}s")
                    block.append(f"  Percent Error:  {float(row['signed_pct_error']):+.2f}%")
                else:
                    block.append(f"  RAPID-LLM run failed. {(row.get('error') or '')}".rstrip())
                    if row.get("raw_output"):
                        block.append("  --- Raw output ---")
                        block.append(str(row["raw_output"]).strip())
                print("\n".join(block))

        plot_path = None
        if enable_plot and rows:
            out_dir = PROJECT_ROOT / "output" / "validation" / "train"
            plot_path = _plot_results(rows, f"Training validation ({variant.upper()} benchmarks)", out_dir / f"uci_train_{variant}.png")
            if emit_logs and plot_path:
                print(f"Saved plot: {plot_path}")

        avg_abs_error = sum(pct_errors) / len(pct_errors) if pct_errors else float("nan")
        if emit_logs:
            print(f"Average absolute percent error for {variant.upper()}: {avg_abs_error:.2f}%")
        outputs.append({"variant": variant, "avg_abs_error": avg_abs_error, "rows": rows, "plot": plot_path})
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
        default=["ddp", "fsdp"],
        choices=["ddp", "fsdp"],
        help="Which benchmark variants to include (runs each separately).",
    )
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-run progress.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        hardware_config=args.hardware_config,
        model_config=args.model_config,
        variants=args.variants,
        enable_plot=args.enable_plot,
        show_progress=args.show_progress,
    )
