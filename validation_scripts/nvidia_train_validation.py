#!/usr/bin/env python3
"""
Run Rapid-LLM (DeepFlow) for NVIDIA training validation cases and export CSV.
Standalone script: no imports from other project modules.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PERF = REPO_ROOT / "run_perf.py"
MODEL_CONFIG_PATH = REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config"

MODEL_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "GPT 22B": "GPT_22_B.yaml",
    "GPT 175B": "GPT_175_B.yaml",
    "GPT 310B": "GPT_310_B.yaml",
    "GPT 530B": "GPT_530_B.yaml",
    "GPT 1T": "GPT_1T.yaml",
}

LEAF_SIZE_BY_MODEL = {
    "GPT 1T": 32,
    "GPT 310B": 30,
    "GPT 530B": 35,
}

SWITCH_HARDWARE_CONFIG = Path("validation_scripts/validation_configs/hardware-config/a100_80GB_train_validation_switch.yaml")
RAPID_TIME_COLUMN = "rapid_llm_time_s"

_TRAIN_TIME_PATTERN = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")

COLOR_MAP: Dict[str, str] = {
    "actual": "#4c566a",
    "rapid_llm": "#1f77b4",
    "stg": "#ff7f0e",
}
DISPLAY_LABELS: Dict[str, str] = {
    "actual": "Actual",
    "rapid_llm": "Rapid-LLM",
    "stg": "STAGE",
}


@dataclass
class ToolResult:
    name: str
    seconds: Optional[float]
    error: Optional[str] = None


def _slugify(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label).strip("_")
    return cleaned or "spec"


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{timestamp}_{os.getpid()}"


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return bool(value)


def _norm_tp_sp(value: object) -> str:
    return "True" if _parse_bool(value) else "False"


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_list_of_dicts(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, Mapping) for item in value)


def _merge_list_of_dicts(orig: list, overrides: list) -> list:
    orig_index = {item.get("id"): item for item in orig if isinstance(item, Mapping) and "id" in item}
    orig_order = [item.get("id") for item in orig if isinstance(item, Mapping) and "id" in item]

    merged = []
    for item in orig:
        if not isinstance(item, Mapping) or "id" not in item:
            merged.append(copy.deepcopy(item))
    for item in overrides:
        if not isinstance(item, Mapping) or "id" not in item:
            continue
        item_id = item["id"]
        base = orig_index.get(item_id, {})
        merged_item = _deep_update(copy.deepcopy(dict(base)), item)
        orig_index[item_id] = merged_item

    for item_id in orig_order:
        merged.append(orig_index[item_id])
    for item in overrides:
        if not isinstance(item, Mapping) or "id" not in item:
            continue
        item_id = item["id"]
        if item_id not in orig_order:
            merged.append(orig_index[item_id])
    return merged


def _deep_update(target: Dict[str, object], overrides: Mapping[str, object]) -> Dict[str, object]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            target[key] = _deep_update(dict(target[key]), value)  # type: ignore[index]
        elif _is_list_of_dicts(value) and _is_list_of_dicts(target.get(key)):
            target[key] = _merge_list_of_dicts(list(target[key]), list(value))  # type: ignore[list-item]
        else:
            target[key] = value
    return target


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {path}, got {type(data).__name__}")
    return data


def _merge_dicts(base: Dict[str, object], overrides: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not overrides:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    _deep_update(merged, overrides)
    return merged


def _write_yaml(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _materialize_configs(
    base_model_path: Path,
    base_hw_path: Path,
    model_overrides: Optional[Mapping[str, object]],
    hardware_overrides: Optional[Mapping[str, object]],
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


def _run_deepflow(model_path: Path, hw_path: Path, output_root: Path, cwd: Path) -> ToolResult:
    env = os.environ.copy()
    env["DEEPFLOW_OUTPUT_DIR"] = str(output_root)
    env["DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS"] = "1"
    env["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = "1"
    env["DEEPFLOW_ZERO_INTERNAL_SOFTMAX"] = "1"
    env["DEEPFLOW_ASTRA_CACHE_MODE"] = "no_cache"
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"

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
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_text = result.stdout or ""
    if output_text:
        print(output_text, end="")
    if result.returncode != 0:
        return ToolResult(name="rapid_llm", seconds=None, error=f"return code {result.returncode}")

    timing = _parse_training_time(output_text)
    return ToolResult(name="rapid_llm", seconds=timing)


def _short_label(row: Mapping[str, str]) -> str:
    model = row.get("model", "model")
    batch = row.get("batch", "?")
    mb = row.get("mb", "?")
    dp = row.get("dp", "?")
    tp = row.get("tp", "?")
    pp = row.get("pp", "?")
    cp = row.get("cp", "?")
    tp_sp = row.get("tp_sp", "?")
    recomputation = row.get("recomputation", "?")
    return f"{model} bs={batch}/mb={mb} dp={dp} tp={tp} pp={pp} cp={cp} tp_sp={tp_sp} recompute={recomputation}"


def _device_from_case(case: str) -> str:
    if case == "korthi":
        return "A100_korthi"
    if case == "selene":
        return "A100_selene"
    return ""


def _key_from_row(row: Mapping[str, str]) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        row.get("device", ""),
        row.get("model", ""),
        row.get("batch", ""),
        row.get("mb", ""),
        row.get("dp", ""),
        row.get("tp", ""),
        row.get("pp", ""),
        row.get("cp", ""),
        _norm_tp_sp(row.get("tp_sp", "")),
        row.get("recomputation", ""),
    )


def _key_from_stage(row: Mapping[str, str]) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        _device_from_case(row.get("case", "")),
        row.get("model", ""),
        row.get("batch", ""),
        row.get("mb", ""),
        row.get("dp", ""),
        row.get("tp", ""),
        row.get("pp", ""),
        row.get("cp", ""),
        _norm_tp_sp(row.get("tp_sp", "")),
        row.get("recomputation", ""),
    )


def _resolve_model_config(model: str) -> Path:
    model_cfg_name = MODEL_CONFIGS.get(model)
    if not model_cfg_name:
        raise ValueError(f"Unknown model '{model}' in input CSV.")
    return MODEL_CONFIG_PATH / model_cfg_name


def _resolve_hw_config(row: Mapping[str, str], default_hw: Path) -> Path:
    model = str(row.get("model", "")).strip().lower()
    if model == "gpt 175b":
        return SWITCH_HARDWARE_CONFIG
    dim1_topology = str(row.get("dim1_topology", "")).strip().lower()
    if dim1_topology == "switch":
        return SWITCH_HARDWARE_CONFIG
    return default_hw


def _build_overrides(row: Mapping[str, str]) -> Tuple[Dict[str, object], Dict[str, object]]:
    model = row.get("model", "")
    batch = int(row.get("batch", "0"))
    mb = int(row.get("mb", "0"))
    dp = int(row.get("dp", "0"))
    tp = int(row.get("tp", "0"))
    pp = int(row.get("pp", "0"))
    cp = int(row.get("cp", "0"))
    tp_sp = _parse_bool(row.get("tp_sp"))
    recomputation = str(row.get("recomputation", "")).strip().lower()

    full_recompute = recomputation in {"full", "true", "yes", "on", "1"}

    model_overrides = {
        "model_param": {
            "global_batch_size": int(batch),
        }
    }

    hw_overrides: Dict[str, object] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": bool(tp_sp),
            "cp": int(cp),
            "pp": int(pp),
            "mb": max(1, int(mb)),
            "train": {"dp": int(dp), "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            "full_recomputation": full_recompute,
        },
    }

    dim1_topology = str(row.get("dim1_topology", "")).strip().lower()
    dim1_leaf_size = LEAF_SIZE_BY_MODEL.get(model)
    if dim1_topology != "switch" and dim1_leaf_size is not None:
        hw_overrides.setdefault("network", {})
        hw_overrides["network"].setdefault("dimensions", [])
        hw_overrides["network"]["dimensions"].append(
            {
                "id": "dim1",
                "topology": {
                    "leaf_size": int(dim1_leaf_size),
                },
            }
        )

    return model_overrides, hw_overrides


@contextmanager
def _silence_worker(log_path: Path):
    """Redirect stdout/stderr for a worker into a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file, ExitStack() as stack:
        stack.enter_context(redirect_stdout(log_file))
        stack.enter_context(redirect_stderr(log_file))
        yield


def _compute_errors(actual: Optional[float], predicted: Optional[float]) -> Tuple[str, str, str]:
    if actual is None or predicted is None:
        return "", "", ""
    if actual == 0:
        return "", "", ""
    if math.isnan(actual) or math.isnan(predicted):
        return "", "", ""
    signed_pct = (predicted - actual) / actual * 100.0
    abs_pct = abs(signed_pct)
    abs_error = abs(predicted - actual)
    return str(signed_pct), str(abs_pct), str(abs_error)


def _run_case(
    row: Mapping[str, str],
    idx: int,
    default_hw: Path,
    run_root: Path,
) -> Dict[str, str]:
    label = _short_label(row)
    slug = _slugify(label)
    spec_root = run_root / slug
    config_dir = spec_root / "configs"
    log_path = spec_root / "worker.log"

    model_config = _resolve_model_config(row.get("model", ""))
    hardware_config = _resolve_hw_config(row, default_hw)
    model_overrides, hw_overrides = _build_overrides(row)
    model_path, hw_path = _materialize_configs(
        model_config,
        hardware_config,
        model_overrides,
        hw_overrides,
        config_dir,
    )
    output_root = spec_root / "tmp" / "deepflow" / slug

    with _silence_worker(log_path):
        tool_result = _run_deepflow(model_path, hw_path, output_root, spec_root)

    actual = _to_float(row.get("actual_time_s"))
    predicted = tool_result.seconds
    signed_pct, abs_pct, abs_error = _compute_errors(actual, predicted)

    out_row = dict(row)
    out_row["hardware_config"] = hardware_config.name
    out_row[RAPID_TIME_COLUMN] = "" if predicted is None else str(predicted)
    out_row["signed_pct_error"] = signed_pct
    out_row["abs_pct_error"] = abs_pct
    out_row["abs_error_s"] = abs_error
    out_row["output_root"] = str(spec_root)
    out_row["error"] = tool_result.error or ""
    return out_row


def _format_label(row: Mapping[str, str]) -> str:
    model = row.get("model", "")
    recomputation = row.get("recomputation", "")
    return (
        f"{model} "
        f"tp{row.get('tp', '')}-cp{row.get('cp', '')}-pp{row.get('pp', '')}-dp{row.get('dp', '')}"
        f"-recompute-{recomputation}"
    )


def _build_series(
    rows: List[Dict[str, str]],
    stage_rows: List[Dict[str, str]],
) -> Tuple[List[str], Dict[str, List[float]]]:
    labels: List[str] = []
    actual: List[float] = []
    rapid: List[float] = []
    stg: List[float] = []

    stage_map: Dict[Tuple[str, str, str, str, str, str, str, str, str, str], Dict[str, str]] = {}
    for row in stage_rows:
        tool = row.get("tool", "").lower()
        if tool not in {"stg", "stage"}:
            continue
        key = _key_from_stage(row)
        if key not in stage_map:
            stage_map[key] = row

    for row in rows:
        actual_val = _to_float(row.get("actual_time_s"))
        rapid_val = _to_float(row.get(RAPID_TIME_COLUMN))
        stg_row = stage_map.get(_key_from_row(row))
        stg_val = _to_float(stg_row.get("tool_seconds")) if stg_row else None
        if actual_val is None and rapid_val is None and stg_val is None:
            continue
        labels.append(_format_label(row))
        actual.append(actual_val if actual_val is not None else math.nan)
        rapid.append(rapid_val if rapid_val is not None else math.nan)
        stg.append(stg_val if stg_val is not None else math.nan)

    return labels, {"actual": actual, "rapid_llm": rapid, "stg": stg}


def _plot_results(
    labels: List[str],
    series: Dict[str, List[float]],
    output: Path,
    title: str,
) -> Path:
    tool_list = ["actual", "rapid_llm", "stg"]
    width = 0.2
    x = list(range(len(labels)))

    fig_w = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    for idx, name in enumerate(tool_list):
        offsets = [pos + (idx - (len(tool_list) - 1) / 2) * width for pos in x]
        label = DISPLAY_LABELS.get(name, name)
        values = series.get(name, [])
        ax.bar(
            offsets,
            values,
            width=width,
            label=label,
            color=COLOR_MAP.get(name, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Training time (s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> int:
    script_dir = Path(__file__).resolve().parent / "train_validation_data"
    parser = argparse.ArgumentParser(
        description="Run Rapid-LLM for NVIDIA training validation cases and export CSV.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=script_dir / "nvidia_train_validation_cases.csv",
        help="CSV containing the NVIDIA validation cases to run.",
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=Path("tools/comp/nvidia_graph/a100_80GB_train_validation.yaml"),
        help="Hardware config YAML to use for all cases.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "nvidia_train_validation_result.csv",
        help="Output CSV path for Rapid-LLM results.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=script_dir / "nvidia_train_validation_runs" / _new_run_id(),
        help="Output root for run artifacts and logs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: CPU count, capped by case count).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=script_dir / "nvidia_train_validation_compare.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--stage-csv",
        type=Path,
        default=script_dir / "STAGE_data.csv",
        help="CSV containing STAGE results to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Runtime comparison on large-scale system",
        help="Plot title.",
    )
    parser.add_argument(
        "--no-plot",
        dest="enable_plot",
        action="store_false",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.hardware_config.exists():
        raise FileNotFoundError(f"Hardware config not found: {args.hardware_config}")

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in CSV: {args.input_csv}")

    args.run_root.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    for field in (
        "hardware_config",
        RAPID_TIME_COLUMN,
        "signed_pct_error",
        "abs_pct_error",
        "abs_error_s",
        "output_root",
        "error",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        args.output_csv.unlink()
    write_header = True

    worker_cap = args.workers if args.workers is not None else (os.cpu_count() or 1)
    worker_count = max(1, min(len(rows), int(worker_cap)))
    print(f"Using {worker_count} workers")

    plotted_rows: List[Tuple[int, Dict[str, str]]] = []
    with args.output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            handle.flush()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            futures = {
                executor.submit(_run_case, row, idx, args.hardware_config, args.run_root): idx
                for idx, row in enumerate(rows)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                row = rows[idx]
                try:
                    out_row = fut.result()
                except Exception as exc:
                    label = _short_label(row)
                    out_row = dict(row)
                    out_row["hardware_config"] = args.hardware_config.name
                    out_row[RAPID_TIME_COLUMN] = ""
                    out_row["signed_pct_error"] = ""
                    out_row["abs_pct_error"] = ""
                    out_row["abs_error_s"] = ""
                    out_row["output_root"] = str(args.run_root / _slugify(label))
                    out_row["error"] = f"{type(exc).__name__}: {exc}"
                writer.writerow(out_row)
                handle.flush()
                plotted_rows.append((idx, out_row))
                label = _short_label(out_row)
                status = "ok" if out_row.get("error") == "" else "error"
                print(f"[nvidia_train_validation] finished {label} ({status})")

    print(f"Wrote Rapid-LLM CSV: {args.output_csv}")
    print(f"Artifacts stored under: {args.run_root}")
    if args.enable_plot:
        ordered_rows = [row for _, row in sorted(plotted_rows, key=lambda item: item[0])]
        stage_rows: List[Dict[str, str]] = []
        if args.stage_csv.exists():
            with args.stage_csv.open(newline="", encoding="utf-8") as handle:
                stage_rows = list(csv.DictReader(handle))
        labels, series = _build_series(ordered_rows, stage_rows)
        if labels:
            output = _plot_results(labels, series, args.plot_output, args.title)
            print(f"Wrote plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
