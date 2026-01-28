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
UCI inference validation harness for OPT models (13B/30B/66B).

Loads uci_inf.csv, runs RAPID-LLM inference sweeps, and compares
TTFT/TPOT/throughput metrics against measured values.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Sequence
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .validation_helpers import (
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
        parse_ttft,
        parse_tpot,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
        parse_ttft,
        parse_tpot,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PERF = PROJECT_ROOT / "run_perf.py"
DEFAULT_HW_CONFIG = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "a100_80GB_uci.fitted.yaml"
)
MODEL_CONFIG_PATH = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config"
IMEC_CSV = PROJECT_ROOT / "validation_scripts" / "imec_data" / "uci_inf.csv"

MODEL_CONFIGS = {
    "opt-13b": "OPT-13B_inf.yaml",
    "opt-30b": "OPT-30B_inf.yaml",
    "opt-66b": "OPT-66B_inf.yaml",
}

# Network modeling constants (4-GPU ring: 2 NVLink, 2 PCIe).
NVLINK_LINK_BW_GB = 5 * 25  # GB/s per GPU worth of NVLink throughput
PCIE_LINK_BW_GB = 25  # GB/s per GPU PCIe
PCIE_EFFECTIVE_BW_GB = PCIE_LINK_BW_GB
AVG_NVLINK_PCIE_BW_GB = (NVLINK_LINK_BW_GB + 2 * PCIE_EFFECTIVE_BW_GB) / 3.0
NVLINK_LATENCY_S =  1e-6
PCIE_LATENCY_S = 5e-6


def _normalize_cuda_devices(value: object) -> str:
    return str(value).strip()


def _bandwidth_label_gb(value_gb: float) -> str:
    return f"{value_gb:g} GB"


def _make_dimension(
    idx: int,
    label: str,
    axes: List[str],
    bandwidth_gb: float,
    latency_s: float,
    topology: str,
) -> Dict[str, object]:
    size: object = "auto" if axes else 1
    return {
        "id": f"dim{idx}",
        "label": label,
        "size": size,
        "topology": {
            "type": topology,
            "bandwidth": _bandwidth_label_gb(bandwidth_gb),
            "latency": latency_s,
            "util": 1.0,
            "optimize_2dmap": False,
        },
        "collective_override": {},
        "parallelisms": axes,
    }


def _build_network_override(tp: int, cuda_devices: str) -> Tuple[Dict[str, object], str]:
    cuda_devices_norm = _normalize_cuda_devices(cuda_devices).lower()
    if int(tp) == 4:
        tp_cp_label = "NVLink"
        pp_dp_label = "PCIe"

        dims = [
            _make_dimension(0, "tp_cp_ring", ["tp", "cp"], NVLINK_LINK_BW_GB, NVLINK_LATENCY_S, "Ring"),
            _make_dimension(1, "pp_dp_ring", ["pp", "dp"], PCIE_LINK_BW_GB, PCIE_LATENCY_S, "Ring"),
        ]
        mapping_desc = (
            f"2D Ring dim0=tp/cp ({tp_cp_label}, tp={tp}), "
            f"dim1=pp/dp ({pp_dp_label})"
        )
        network_override: Dict[str, object] = {
            "network": {
                "dimensions": dims,
            },
        }
        return network_override, mapping_desc
        # dims: List[Dict[str, object]] = [
        #     {
        #         "id": "dim0",
        #         "label": "fcring2d_tp",
        #         "size": [2, 'auto'],
        #         "topology": {
        #             "type": "FC-Ring2D",
        #             "bandwidth": [
        #                 _bandwidth_label_gb(NVLINK_LINK_BW_GB),
        #                 _bandwidth_label_gb(PCIE_EFFECTIVE_BW_GB),
        #             ],
        #             "latency": PCIE_LATENCY_S,
        #             # "energy_per_bit": ENERGY_PER_BIT,
        #             "util": 1.0,
        #             "optimize_2dmap": False,
        #         },
        #         "collective_override": {},
        #         "parallelisms": ["tp"],
        #     },
        #     _make_dimension(1, "unused_dim1", [], PCIE_EFFECTIVE_BW_GB, PCIE_LATENCY_S, "FullyConnected"),
        #     _make_dimension(2, "unused_dim2", [], PCIE_EFFECTIVE_BW_GB, PCIE_LATENCY_S, "FullyConnected"),
        # ]
        # return {
        #     "network": {
        #         "dimensions": dims,
        #         "overlap": {"tp_sp_overlap": 0.0},
        #     }
        # }, "TP FC-Ring2D (NVLink pairs + PCIe cross links)"

    if int(tp) == 2 and cuda_devices_norm == "default":
        bandwidth_gb = NVLINK_LINK_BW_GB
        latency_s = NVLINK_LATENCY_S
        mapping_desc = "TP ring (NVLink)"
    elif int(tp) == 2:
        bandwidth_gb = PCIE_EFFECTIVE_BW_GB
        latency_s = PCIE_LATENCY_S
        mapping_desc = "TP ring (PCIe)"
    else:
        bandwidth_gb = AVG_NVLINK_PCIE_BW_GB
        latency_s = PCIE_LATENCY_S
        mapping_desc = "TP ring (avg NVLink/PCIe)"

    dims: List[Dict[str, object]] = [
        _make_dimension(0, "ring_tp", ["tp"], bandwidth_gb, latency_s, "Ring"),
        _make_dimension(1, "unused_dim1", [], PCIE_EFFECTIVE_BW_GB, PCIE_LATENCY_S, "FullyConnected"),
        _make_dimension(2, "unused_dim2", [], PCIE_EFFECTIVE_BW_GB, PCIE_LATENCY_S, "FullyConnected"),
    ]
    return {
        "network": {
            "dimensions": dims,
            "overlap": {"tp_sp_overlap": 0.0},
        }
    }, mapping_desc


def _load_grid(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "tensor_parallel_size",
        "input_len",
        "output_len",
        "num_prompts",
        "requests_per_sec",
        "total_tokens_per_sec",
        "output_tokens_per_sec",
        "ttft_ms",
        "tpot_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=[
            "model_name",
            "tensor_parallel_size",
            "cuda_devices",
            "input_len",
            "output_len",
            "requests_per_sec",
            "total_tokens_per_sec",
            "output_tokens_per_sec",
            "ttft_ms",
            "tpot_ms",
        ]
    )
    return df


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (
            str(row["model_name"]),
            int(row["tensor_parallel_size"]),
            _normalize_cuda_devices(row["cuda_devices"]),
            int(row["input_len"]),
            int(row["output_len"]),
            {
                "requests_per_sec": float(row["requests_per_sec"]),
                "total_tokens_per_sec": float(row["total_tokens_per_sec"]),
                "output_tokens_per_sec": float(row["output_tokens_per_sec"]),
                "ttft_ms": float(row["ttft_ms"]),
                "tpot_ms": float(row["tpot_ms"]),
            },
        )


def _build_spec(
    model_name: str,
    tp: int,
    cuda_devices: str,
    input_len: int,
    output_len: int,
    idx: int,
) -> ValidationSpec:
    model_key = model_name.lower()
    model_cfg = MODEL_CONFIGS.get(model_key)
    if model_cfg is None:
        raise ValueError(f"No model config mapping for {model_name!r}")

    label = f"{model_name} TP={tp} cuda={cuda_devices} in={input_len} out={output_len}"
    network_override, mapping_desc = _build_network_override(tp, cuda_devices)
    hw_overrides: Dict[str, object] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": True,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "train": {"dp": 1, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
    }
    hw_overrides.update(network_override)

    model_overrides = {
        "model_param": {
            "global_batch_size": 1,
            "seq_len": int(input_len) + int(output_len),
            "decode_len": int(output_len),
            "run_type": "inference",
        }
    }

    return ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=str(MODEL_CONFIG_PATH / model_cfg),
        hardware_config_path=str(DEFAULT_HW_CONFIG),
        metadata={
            "model": model_name,
            "tp": int(tp),
            "cuda_devices": cuda_devices,
            "input_len": int(input_len),
            "output_len": int(output_len),
            "network_mapping": mapping_desc,
        },
        order=idx,
    )


def build_specs(
    csv_path: Path,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, str, int, int], Dict[str, float]], str]:
    df = _load_grid(csv_path)
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int, str, int, int], Dict[str, float]] = {}
    idx = 0
    base_model_path: Optional[str] = None
    for model_name, tp, cuda_devices, input_len, output_len, actual in _iter_tests(df):
        spec = _build_spec(model_name, tp, cuda_devices, input_len, output_len, idx)
        specs.append(spec)
        actual_lookup[(model_name, tp, cuda_devices, input_len, output_len)] = actual
        base_model_path = base_model_path or spec.model_config_path
        idx += 1

    if not specs or base_model_path is None:
        raise ValueError("No validation specs generated from the IMEC CSV.")
    return specs, actual_lookup, base_model_path


def _parse_metrics(output: str, spec: ValidationSpec) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics.update(parse_ttft(output, spec))
    metrics.update(parse_tpot(output, spec))
    metrics.update(parse_inference_time(output, spec))
    return metrics


def _pct_error(predicted: float, actual: float) -> float:
    if math.isnan(predicted) or math.isnan(actual) or actual == 0:
        return float("nan")
    return abs(predicted - actual) / actual * 100.0


def _build_label(row: Mapping[str, object]) -> str:
    model = row.get("model")
    tp = row.get("tp")
    cuda_devices = row.get("cuda_devices")
    return f"{model} TP{tp} cuda={cuda_devices}"


def compute_rows(
    results,
    actual_lookup: Dict[Tuple[str, int, str, int, int], Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for res in results:
        meta = res.spec.metadata or {}
        model = str(meta.get("model"))
        tp = int(meta.get("tp", 0))
        cuda_devices = str(meta.get("cuda_devices"))
        input_len = int(meta.get("input_len", 0))
        output_len = int(meta.get("output_len", 0))
        key = (model, tp, cuda_devices, input_len, output_len)
        actual = actual_lookup.get(key, {})

        inf_time_s = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        ttft_s = float(res.metrics.get("ttft_s", float("nan"))) if res.success else float("nan")
        tpot_s = float(res.metrics.get("tpot_s", float("nan"))) if res.success else float("nan")

        pred_requests_per_sec = 1.0 / inf_time_s if inf_time_s > 0 else float("nan")
        pred_total_tokens_per_sec = (
            (input_len + output_len) / inf_time_s if inf_time_s > 0 else float("nan")
        )
        pred_output_tokens_per_sec = output_len / inf_time_s if inf_time_s > 0 else float("nan")
        pred_ttft_ms = ttft_s * 1000.0 if not math.isnan(ttft_s) else float("nan")
        pred_tpot_ms = tpot_s * 1000.0 if not math.isnan(tpot_s) else float("nan")

        row = {
            "model": model,
            "tp": tp,
            "cuda_devices": cuda_devices,
            "input_len": input_len,
            "output_len": output_len,
            "network_mapping": meta.get("network_mapping"),
            "training_time_s": inf_time_s,
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "pred_requests_per_sec": pred_requests_per_sec,
            "pred_total_tokens_per_sec": pred_total_tokens_per_sec,
            "pred_output_tokens_per_sec": pred_output_tokens_per_sec,
            "pred_ttft_ms": pred_ttft_ms,
            "pred_tpot_ms": pred_tpot_ms,
            "actual_requests_per_sec": float(actual.get("requests_per_sec", float("nan"))),
            "actual_total_tokens_per_sec": float(actual.get("total_tokens_per_sec", float("nan"))),
            "actual_output_tokens_per_sec": float(actual.get("output_tokens_per_sec", float("nan"))),
            "actual_ttft_ms": float(actual.get("ttft_ms", float("nan"))),
            "actual_tpot_ms": float(actual.get("tpot_ms", float("nan"))),
            "success": res.success,
            "error": res.error,
            "raw_output": res.raw_output,
        }

        row["pct_error_requests_per_sec"] = _pct_error(
            row["pred_requests_per_sec"], row["actual_requests_per_sec"]
        )
        row["pct_error_total_tokens_per_sec"] = _pct_error(
            row["pred_total_tokens_per_sec"], row["actual_total_tokens_per_sec"]
        )
        row["pct_error_output_tokens_per_sec"] = _pct_error(
            row["pred_output_tokens_per_sec"], row["actual_output_tokens_per_sec"]
        )
        row["pct_error_ttft_ms"] = _pct_error(row["pred_ttft_ms"], row["actual_ttft_ms"])
        row["pct_error_tpot_ms"] = _pct_error(row["pred_tpot_ms"], row["actual_tpot_ms"])
        row["label"] = _build_label(row)
        rows.append(row)
    return rows


def _plot_percent_error(rows: List[Dict[str, object]], metric: str, title: str, path: Path) -> Optional[Path]:
    if not rows:
        return None
    labels = [row["label"] for row in rows]
    values = [float(row.get(f"pct_error_{metric}", float("nan"))) for row in rows]
    fig_w = max(7.5, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(range(len(values)), values, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    for rect, value in zip(bars, values):
        if math.isnan(value):
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_predicted_vs_actual(
    rows: List[Dict[str, object]],
    metric: str,
    title: str,
    path: Path,
) -> Optional[Path]:
    if not rows:
        return None
    labels = [row["label"] for row in rows]
    pred_values = [float(row.get(f"pred_{metric}", float("nan"))) for row in rows]
    actual_values = [float(row.get(f"actual_{metric}", float("nan"))) for row in rows]
    fig_w = max(7.5, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x = list(range(len(labels)))
    bar_width = 0.35
    pred_bars = ax.bar(
        [pos - bar_width / 2 for pos in x],
        pred_values,
        bar_width,
        label="RAPID-LLM",
        color="#1f77b4",
    )
    actual_bars = ax.bar(
        [pos + bar_width / 2 for pos in x],
        actual_values,
        bar_width,
        label="Actual",
        color="#2ca02c",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    for rect, value in zip(pred_bars, pred_values):
        if math.isnan(value):
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for rect, value in zip(actual_bars, actual_values):
        if math.isnan(value):
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_combined_predicted_vs_actual(
    rows: List[Dict[str, object]],
    metrics: Sequence[Tuple[str, str]],
    title: str,
    path: Path,
) -> Optional[Path]:
    if not rows or not metrics:
        return None
    labels = [row["label"] for row in rows]
    fig_w = max(9.0, 0.65 * len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 7))
    axes = axes.flatten()
    x = list(range(len(labels)))
    bar_width = 0.35

    for ax, (metric, label) in zip(axes, metrics):
        pred_values = [float(row.get(f"pred_{metric}", float("nan"))) for row in rows]
        actual_values = [float(row.get(f"actual_{metric}", float("nan"))) for row in rows]
        ax.bar(
            [pos - bar_width / 2 for pos in x],
            pred_values,
            bar_width,
            label="RAPID-LLM",
            color="#1f77b4",
        )
        ax.bar(
            [pos + bar_width / 2 for pos in x],
            actual_values,
            bar_width,
            label="Actual",
            color="#2ca02c",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    if len(axes) > len(metrics):
        for ax in axes[len(metrics):]:
            ax.axis("off")

    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_combined_percent_error(
    rows: List[Dict[str, object]],
    metrics: Sequence[Tuple[str, str]],
    title: str,
    path: Path,
) -> Optional[Path]:
    if not rows or not metrics:
        return None
    labels = [row["label"] for row in rows]
    fig_w = max(9.0, 0.65 * len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 7))
    axes = axes.flatten()
    x = list(range(len(labels)))

    for ax, (metric, label) in zip(axes, metrics):
        values = [float(row.get(f"pct_error_{metric}", float("nan"))) for row in rows]
        ax.bar(x, values, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    if len(axes) > len(metrics):
        for ax in axes[len(metrics):]:
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run(
    *,
    hardware_config: Optional[str] = None,
    enable_plot: bool = True,
    show_progress: bool = False,
    emit_logs: bool = True,
):
    hw_cfg = hardware_config or str(DEFAULT_HW_CONFIG)
    specs, actual_lookup, base_model_path = build_specs(IMEC_CSV)

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_cfg,
        result_parser=_parse_metrics,
        run_perf_path=str(RUN_PERF),
        show_progress=show_progress,
    )

    rows = compute_rows(validation_results, actual_lookup)
    if emit_logs:
        for row in rows:
            print(f"\n=== Result ({row['label']}) ===")
            if row.get("network_mapping"):
                print(f"  Network: {row['network_mapping']}")
            if row["success"]:
                print(f"  RAPID-LLM total time: {float(row['training_time_s']):.2f}s")
                print(f"  TTFT (ms): pred={float(row['pred_ttft_ms']):.2f} actual={float(row['actual_ttft_ms']):.2f}")
                print(f"  TPOT (ms): pred={float(row['pred_tpot_ms']):.2f} actual={float(row['actual_tpot_ms']):.2f} (error={float(row['pct_error_tpot_ms']):.2f}%)")
                print(
                    "  Requests/s: pred={:.3f} actual={:.3f}".format(
                        float(row["pred_requests_per_sec"]),
                        float(row["actual_requests_per_sec"]),
                    )
                )
                print(
                    "  Total tokens/s: pred={:.2f} actual={:.2f}".format(
                        float(row["pred_total_tokens_per_sec"]),
                        float(row["actual_total_tokens_per_sec"]),
                    )
                )
                print(
                    "  Output tokens/s: pred={:.2f} actual={:.2f}".format(
                        float(row["pred_output_tokens_per_sec"]),
                        float(row["actual_output_tokens_per_sec"]),
                    )
                )
            else:
                print(f"  RAPID-LLM run failed. {(row.get('error') or '')}".rstrip())
                if row.get("raw_output"):
                    print("  --- Raw output ---")
                    print(str(row["raw_output"]).strip())

    if enable_plot:
        out_dir = PROJECT_ROOT / "output" / "validation" / "inf"
        metric_defs = [
            # ("requests_per_sec", "Requests/s"),
            # ("total_tokens_per_sec", "Total tokens/s"),
            # ("output_tokens_per_sec", "Output tokens/s"),
            # ("ttft_ms", "TTFT (ms)"),
            ("tpot_ms", "TPOT (ms)"),
        ]
        for metric, label in metric_defs:
            _plot_percent_error(
                rows,
                metric,
                f"UCI inference percent error ({label})",
                out_dir / f"uci_inf_pct_error_{metric}.png",
            )
            _plot_predicted_vs_actual(
                rows,
                metric,
                f"UCI inference predicted vs actual ({label})",
                out_dir / f"uci_inf_pred_{metric}.png",
            )
        combined_metrics = [item for item in metric_defs if item[0] != "requests_per_sec"]
        _plot_combined_percent_error(
            rows,
            combined_metrics,
            "UCI inference percent error (combined)",
            out_dir / "uci_inf_pct_error_combined.png",
        )
        _plot_combined_predicted_vs_actual(
            rows,
            combined_metrics,
            "UCI inference predicted vs actual (combined)",
            out_dir / "uci_inf_pred_combined.png",
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCI inference validation (OPT IMEC benchmarks).")
    parser.add_argument(
        "--hardware_config",
        default=str(DEFAULT_HW_CONFIG),
        help=f"Path to hardware config YAML (default: {DEFAULT_HW_CONFIG}).",
    )
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-run progress.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        hardware_config=args.hardware_config,
        enable_plot=args.enable_plot,
        show_progress=args.show_progress,
        emit_logs=True,
    )
