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

import argparse
import copy
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

COMPARE_LLMCOMPASS = True
COMPARE_VIDUR = True
COMPARE_GENZ = True
COMPARE_FLATTENED = False

RAPID_HIER_LABEL = "RAPID-LLM (hierarchical)"
RAPID_FLAT_LABEL = "RAPID-LLM (flattened)"

TOOL_COLORS = {
    RAPID_HIER_LABEL: "#1f77b4",
    RAPID_FLAT_LABEL: "#17becf",
    "Actual": "#0bbd37",
    "LLMCompass": "#ff7f0e",
    "Vidur": "#9467bd",
    "GenZ": "#8c564b",
}

FLATTENED_HW_OVERRIDE = {
    "execution_backend": {
        "astra": {"mode": "full_astrasim_flattened"},
    }
}

try:
    from .validation_helpers import (
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )

# Data used to be in  https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/performance.html in 2022.
# no longer available now, found via paper. "Performance Modeling and Workload Analysis of Distributed Large Language Model Training and Inference" by J. Kundu et al.

HW_CONFIGS = {
    "A100": "a100_80GB_inf.yaml",
    "H100": "H100_SXM5_80GB.yaml",
}

MODEL_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "Llama 2-13B": "Llama2-13B_inf.yaml",
    "Llama 2-70B": "Llama2-70B_inf.yaml",
}

MODEL_DISPLAY = {
    "Llama 2-7B": "Llama 2-7B",
    "Llama 2-13B": "Llama 2-13B",
    "Llama 2-70B": "Llama 2-70B",
}

NVIDIA_MODEL_CONFIGS = {
    "Llama 3.3-70B": "Llama3.1-70B_inf.yaml",
}

NVIDIA_MODEL_DISPLAY = {
    "Llama 3.3-70B": "Llama 3.3-70B",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config")
MODEL_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "model-config")
NVIDIA_MODEL_CONFIG_PATH = MODEL_CONFIG_PATH

FITTED_HW_SUFFIX = ".fitted.yaml"
IMEC_DATA_DIR = Path(__file__).parent / "imec_data"
NVIDIA_DATA_DIR = Path(__file__).parent / "nvidia_data"

# LLMCOMPASS_ROOT = os.path.join(PROJECT_ROOT, "..", "LLMCompass")
LLMCOMPASS_IMEC = IMEC_DATA_DIR / "A100_inf_llmcompass.csv"
LLMCOMPASS_NVIDIA = NVIDIA_DATA_DIR / "8xA100_bf16_Llama3_3-70B_llmcompass.csv"

# VIDUR_OUTPUT_DIR = Path("/u1/ee/nanoproj/users/yaoe888/vidur/simulator_output")
VIDUR_IMEC = IMEC_DATA_DIR / "A100_inf_vidur.csv"
VIDUR_NVIDIA = NVIDIA_DATA_DIR / "8xA100_bf16_Llama3_3-70B_vidur.csv"

# GENZ_ROOT = Path("/u1/ee/nanoproj/users/yaoe888/GenZ-LLM-Analyzer")
GENZ_IMEC = IMEC_DATA_DIR / "A100_inf_genz.csv"
GENZ_NVIDIA = NVIDIA_DATA_DIR / "8xA100_bf16_Llama3_3-70B_genz.csv"

NVIDIA_DATASETS = {
    "A100": {
        "csv": "8xA100_bf16_Llama3_3-70B.csv",
        "tp": 8,
        "model": "Llama 3.3-70B",
    },
    "H100": {
        "csv": "4xH100_fp16_Llama3_3-70B.csv",
        "tp": 4,
        "model": "Llama 3.3-70B",
    },
}

VIDUR_MODEL_MAP = {
    "meta-llama/llama-2-7b-hf": "Llama 2-7B",
    "meta-llama/llama-2-13b-hf": "Llama 2-13B",
    "meta-llama/llama-2-70b-hf": "Llama 2-70B",
}


def _resolve_hw_config_path(device: str, fit_model: bool = True) -> str:
    base_name = HW_CONFIGS.get(device)
    if base_name is None:
        raise ValueError(f"No hardware config mapping for device {device}")
    base_path = os.path.join(HARDWARE_CONFIG_PATH, base_name)
    if fit_model:
        fitted_name = base_name.replace(".yaml", FITTED_HW_SUFFIX)
        fitted_path = os.path.join(HARDWARE_CONFIG_PATH, fitted_name)
        return fitted_path if os.path.exists(fitted_path) else base_path
    else:
        return base_path


def _apply_compare_args(args: argparse.Namespace) -> None:
    global COMPARE_LLMCOMPASS, COMPARE_VIDUR, COMPARE_GENZ
    if args.llmcompass or args.vidur or args.genz:
        COMPARE_LLMCOMPASS = bool(args.llmcompass)
        COMPARE_VIDUR = bool(args.vidur)
        COMPARE_GENZ = bool(args.genz)


def _load_data(csv_path: Path) -> pd.DataFrame:
    # Skip the leading "// filepath: ..." line by treating '/' as a comment char.
    df = pd.read_csv(csv_path, comment="/")
    # Normalize dtypes
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df.dropna(subset=["device", "model", "TP", "actual"])


@lru_cache(maxsize=None)
def _load_device_data(device: str) -> pd.DataFrame:
    csv_path = IMEC_DATA_DIR / f"{device}_inf.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found for device {device}: {csv_path}")
    return _load_data(csv_path)


def _load_nvidia_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {
        "Input Tokens": "input_tokens",
        "Output Tokens": "output_tokens",
        "Concurrency": "concurrency",
        "TTFT (ms)": "ttft_ms",
        "ITL (ms)": "itl_ms",
        "Throughput (Tokens/s)": "throughput_tps",
    }
    missing = [key for key in rename_map if key not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in NVIDIA data {csv_path}: {missing}")
    df = df.rename(columns=rename_map)
    for col in ("input_tokens", "output_tokens", "concurrency", "ttft_ms", "itl_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["input_tokens", "output_tokens", "concurrency", "ttft_ms", "itl_ms"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["concurrency"] = df["concurrency"].astype(int)
    df["ttft"] = df["ttft_ms"] / 1000.0
    df["itl"] = df["itl_ms"] / 1000.0
    df["actual"] = (df["ttft_ms"] + df["itl_ms"] * (df["output_tokens"] - 1).clip(lower=0)) / 1000.0
    return df


def _load_llmcompass_imec_errors(csv_path: str, device: str = "A100") -> List[Dict[str, object]]:
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    required_cols = {"model", "tp", "total_latency"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"LLMCompass IMEC CSV missing columns: {sorted(missing)}")
        return []

    for col in ("tp", "total_latency", "batch_size", "input_tokens", "output_tokens"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "batch_size" in df.columns:
        df = df[df["batch_size"] == 1]
    if "input_tokens" in df.columns:
        df = df[df["input_tokens"] == 200]
    if "output_tokens" in df.columns:
        df = df[df["output_tokens"] == 200]

    df = df.dropna(subset=["model", "tp", "total_latency"])

    actual_df = _load_device_data(device)
    actual_lookup = {(row["model"], int(row["TP"])): float(row["actual"]) for _, row in actual_df.iterrows()}

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        model = row["model"]
        tp = int(row["tp"])
        total_latency = float(row["total_latency"])
        actual = actual_lookup.get((model, tp), float("nan"))
        if math.isnan(actual) or actual == 0:
            pct_error = float("nan")
        else:
            pct_error = abs(total_latency - actual) / actual * 100.0
        rows.append(
            {
                "device": device,
                "model": model,
                "tp": tp,
                "total_latency": total_latency,
                "pct_error": pct_error,
            }
        )
    return rows


def _load_llmcompass_nvidia_errors(csv_path: str, device: str = "A100") -> List[Dict[str, object]]:
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    required_cols = {"tp", "batch_size", "input_tokens", "output_tokens", "ttft", "itl", "total_latency"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"LLMCompass NVIDIA CSV missing columns: {sorted(missing)}")
        return []

    for col in ("tp", "batch_size", "input_tokens", "output_tokens", "ttft", "itl", "total_latency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["tp", "batch_size", "input_tokens", "output_tokens", "total_latency"])

    dataset = NVIDIA_DATASETS.get(device)
    model = dataset.get("model") if dataset else "Llama 3-70B"
    try:
        actual_df = _load_nvidia_device_data(device)
    except FileNotFoundError:
        return []
    actual_lookup = {
        (row["model"], int(row["TP"]), int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"])): float(
            row["actual"]
        )
        for _, row in actual_df.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        tp = int(row["tp"])
        batch_size = int(row["batch_size"])
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        ttft = float(row["ttft"])
        itl = float(row["itl"])
        total_latency = float(row["total_latency"])
        actual = actual_lookup.get((model, tp, input_tokens, output_tokens, batch_size), float("nan"))
        if math.isnan(actual) or actual == 0:
            pct_error = float("nan")
        else:
            pct_error = abs(total_latency - actual) / actual * 100.0
        rows.append(
            {
                "device": device,
                "model": model,
                "tp": tp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": batch_size,
                "ttft": ttft,
                "itl": itl,
                "total_latency": total_latency,
                "pct_error": pct_error,
            }
        )
    return rows


def _normalize_vidur_model(model_name: str) -> Optional[str]:
    normalized = model_name.strip().lower()
    mapped = VIDUR_MODEL_MAP.get(normalized)
    if mapped:
        return mapped
    if "llama-2-7b" in normalized:
        return "Llama 2-7B"
    if "llama-2-13b" in normalized:
        return "Llama 2-13B"
    if "llama-2-70b" in normalized:
        return "Llama 2-70B"
    return None


def _normalize_vidur_device(network_device: str) -> Optional[str]:
    normalized = network_device.strip().lower()
    if "a100" in normalized:
        return "A100"
    if "h100" in normalized:
        return "H100"
    return None


def _normalize_genz_device(device: str) -> Optional[str]:
    normalized = device.strip().lower()
    if "a100" in normalized:
        return "A100"
    if "h100" in normalized:
        return "H100"
    return None


def _load_vidur_imec_errors(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    required_cols = {
        "model_name",
        "tp",
        "network_device",
        "input_tokens",
        "output_tokens",
        "batch_size",
        "ttft_ms",
        "tpot_ms",
        "e2e_s",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Vidur IMEC CSV missing columns: {sorted(missing)}")
        return []

    for col in ("tp", "input_tokens", "output_tokens", "batch_size", "ttft_ms", "tpot_ms", "e2e_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model"] = df["model_name"].apply(_normalize_vidur_model)
    df["device"] = df["network_device"].apply(_normalize_vidur_device)

    df = df.dropna(subset=["model", "device", "tp", "input_tokens", "output_tokens", "batch_size", "e2e_s"])
    df = df[
        (df["batch_size"] == 1)
        & (df["input_tokens"] == 200)
        & (df["output_tokens"] == 200)
    ]

    rows: List[Dict[str, object]] = []
    for device in sorted(df["device"].unique()):
        try:
            actual_df = _load_device_data(device)
        except FileNotFoundError:
            continue
        actual_lookup = {
            (row["model"], int(row["TP"])): float(row["actual"])
            for _, row in actual_df.iterrows()
        }
        for _, row in df[df["device"] == device].iterrows():
            model = row["model"]
            tp = int(row["tp"])
            total_latency = float(row["e2e_s"])
            actual = actual_lookup.get((model, tp), float("nan"))
            if math.isnan(actual) or actual == 0:
                pct_error = float("nan")
            else:
                pct_error = abs(total_latency - actual) / actual * 100.0
            rows.append(
                {
                    "device": device,
                    "model": model,
                    "tp": tp,
                    "total_latency": total_latency,
                    "pct_error": pct_error,
                }
            )
    return rows


def _load_vidur_nvidia_errors(csv_path: Path, device: str = "A100") -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    required_cols = {"input_tokens", "output_tokens", "concurrency", "ttft_ms", "tpot_ms", "e2e_s"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Vidur NVIDIA CSV missing columns: {sorted(missing)}")
        return []

    for col in ("input_tokens", "output_tokens", "concurrency", "ttft_ms", "tpot_ms", "e2e_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["input_tokens", "output_tokens", "concurrency", "ttft_ms", "tpot_ms", "e2e_s"])

    dataset = NVIDIA_DATASETS.get(device)
    if not dataset:
        return []
    model = dataset.get("model")
    tp = int(dataset.get("tp"))

    try:
        actual_df = _load_nvidia_device_data(device)
    except FileNotFoundError:
        return []
    actual_lookup = {
        (row["model"], int(row["TP"]), int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"])): float(
            row["actual"]
        )
        for _, row in actual_df.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        concurrency = int(row["concurrency"])
        total_latency = float(row["e2e_s"])
        actual = actual_lookup.get((model, tp, input_tokens, output_tokens, concurrency), float("nan"))
        if math.isnan(actual) or actual == 0:
            pct_error = float("nan")
        else:
            pct_error = abs(total_latency - actual) / actual * 100.0
        rows.append(
            {
                "device": device,
                "model": model,
                "tp": tp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "ttft": float(row["ttft_ms"]) / 1000.0,
                "itl": float(row["tpot_ms"]) / 1000.0,
                "total_latency": total_latency,
                "pct_error": pct_error,
            }
        )
    return rows


def _load_genz_nvidia_errors(csv_path: Path, device: str = "A100") -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    required_cols = {
        "device",
        "model",
        "tp",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "ttft_s",
        "tpot_s",
        "e2e_s",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"GenZ NVIDIA CSV missing columns: {sorted(missing)}")
        return []
    for col in ("tp", "batch_size", "input_tokens", "output_tokens", "ttft_s", "tpot_s", "e2e_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["device"] = df["device"].apply(_normalize_genz_device)
    df = df.dropna(subset=["device", "model", "tp", "batch_size", "input_tokens", "output_tokens", "e2e_s"])
    df = df[df["device"] == device]

    dataset = NVIDIA_DATASETS.get(device)
    if not dataset:
        return []
    model = dataset.get("model")
    tp = int(dataset.get("tp"))
    df = df[(df["model"] == model) & (df["tp"] == tp)]

    try:
        actual_df = _load_nvidia_device_data(device)
    except FileNotFoundError:
        return []
    actual_lookup = {
        (row["model"], int(row["TP"]), int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"])): float(
            row["actual"]
        )
        for _, row in actual_df.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        concurrency = int(row["batch_size"])
        total_latency = float(row["e2e_s"])
        actual = actual_lookup.get((model, tp, input_tokens, output_tokens, concurrency), float("nan"))
        if math.isnan(actual) or actual == 0:
            pct_error = float("nan")
        else:
            pct_error = abs(total_latency - actual) / actual * 100.0
        rows.append(
            {
                "device": device,
                "model": model,
                "tp": tp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "total_latency": total_latency,
                "ttft": float(row["ttft_s"]),
                "itl": float(row["tpot_s"]),
                "pct_error": pct_error,
            }
        )
    return rows


def _load_genz_imec_errors(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    required_cols = {
        "device",
        "model",
        "tp",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "e2e_s",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"GenZ IMEC CSV missing columns: {sorted(missing)}")
        return []
    for col in ("tp", "batch_size", "input_tokens", "output_tokens", "e2e_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["device"] = df["device"].apply(_normalize_genz_device)
    df = df.dropna(subset=["device", "model", "tp", "batch_size", "input_tokens", "output_tokens", "e2e_s"])
    df = df[
        (df["batch_size"] == 1)
        & (df["input_tokens"] == 200)
        & (df["output_tokens"] == 200)
    ]

    rows: List[Dict[str, object]] = []
    for device in sorted(df["device"].unique()):
        try:
            actual_df = _load_device_data(device)
        except FileNotFoundError:
            continue
        actual_lookup = {
            (row["model"], int(row["TP"])): float(row["actual"])
            for _, row in actual_df.iterrows()
        }
        for _, row in df[df["device"] == device].iterrows():
            model = row["model"]
            tp = int(row["tp"])
            total_latency = float(row["e2e_s"])
            actual = actual_lookup.get((model, tp), float("nan"))
            if math.isnan(actual) or actual == 0:
                pct_error = float("nan")
            else:
                pct_error = abs(total_latency - actual) / actual * 100.0
            rows.append(
                {
                    "device": device,
                    "model": model,
                    "tp": tp,
                    "total_latency": total_latency,
                    "pct_error": pct_error,
                }
            )
    return rows


@lru_cache(maxsize=None)
def _load_nvidia_device_data(device: str) -> pd.DataFrame:
    dataset = NVIDIA_DATASETS.get(device)
    if not dataset:
        raise ValueError(f"No NVIDIA dataset defined for device {device}")
    csv_path = NVIDIA_DATA_DIR / dataset["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"NVIDIA CSV not found for device {device}: {csv_path}")
    df = _load_nvidia_data(csv_path)
    df["device"] = device
    df["model"] = dataset["model"]
    df["TP"] = int(dataset["tp"])
    return df


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (row["device"], row["model"], row["TP"])


def _iter_nvidia_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (
            row["device"],
            row["model"],
            row["TP"],
            row["input_tokens"],
            row["output_tokens"],
            row["concurrency"],
        )


def _build_spec(device: str, model: str, tp: int, idx: int, network_ignored: bool, fit_model: bool=True) -> Tuple[ValidationSpec, str, str]:
    label = f"{device} {model} TP={tp}"

    model_overrides = {
        "model_param": {
            "global_batch_size": 1,
            "seq_len": 400,
            "decode_len": 200
        }
    }

    hw_overrides: Dict[str, Dict[str, object]] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": True,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "train": {"dp": 1, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "1e-9" },
                    }
                ]
            },
        }
        hw_overrides.update(override_bw)

    spec = ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model)),
        hardware_config_path=_resolve_hw_config_path(device, fit_model),
        metadata={"device": device, "model": model, "tp": int(tp)},
        order=idx,
    )
    hw_config_path = spec.hardware_config_path  # type: ignore[arg-type]
    model_config_path = spec.model_config_path  # type: ignore[arg-type]
    return spec, hw_config_path, model_config_path


def _build_nvidia_spec(
    device: str,
    model: str,
    tp: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    idx: int,
    network_ignored: bool,
) -> Tuple[ValidationSpec, str, str]:
    label = (
        f"NVIDIA {device} {model} TP={tp} "
        f"in={input_tokens} out={output_tokens} bs={concurrency}"
    )

    model_overrides = {
        "model_param": {
            "global_batch_size": int(concurrency),
            "seq_len": int(input_tokens) + int(output_tokens),
            "decode_len": int(output_tokens),
        }
    }

    hw_overrides: Dict[str, Dict[str, object]] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": True,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "train": {"dp": 1, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "1e-9"},
                    }
                ]
            },
        }
        hw_overrides.update(override_bw)

    model_cfg = NVIDIA_MODEL_CONFIGS.get(model)
    if model_cfg is None:
        raise ValueError(f"No model config mapping for NVIDIA model {model}")

    spec = ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=os.path.join(NVIDIA_MODEL_CONFIG_PATH, model_cfg),
        hardware_config_path=_resolve_hw_config_path(device),
        metadata={
            "device": device,
            "model": model,
            "tp": int(tp),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "concurrency": int(concurrency),
            "suite": "NVIDIA",
        },
        order=idx,
    )
    hw_config_path = spec.hardware_config_path  # type: ignore[arg-type]
    model_config_path = spec.model_config_path  # type: ignore[arg-type]
    return spec, hw_config_path, model_config_path


def _lookup_actual(device: str, model: str, tp: int) -> float:
    data = _load_device_data(device)
    matches = data.loc[
        (data["device"] == device) & (data["model"] == model) & (data["TP"] == int(tp)),
        "actual",
    ]
    if matches.empty:
        raise ValueError(f"No reference inference time for {device} {model} TP={tp}")
    return float(matches.values[0])


def _lookup_nvidia_actual(
    device: str,
    model: str,
    tp: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> float:
    data = _load_nvidia_device_data(device)
    matches = data.loc[
        (data["device"] == device)
        & (data["model"] == model)
        & (data["TP"] == int(tp))
        & (data["input_tokens"] == int(input_tokens))
        & (data["output_tokens"] == int(output_tokens))
        & (data["concurrency"] == int(concurrency)),
        "actual",
    ]
    if matches.empty:
        raise ValueError(
            "No NVIDIA reference inference time for "
            f"{device} {model} TP={tp} in={input_tokens} out={output_tokens} bs={concurrency}"
        )
    return float(matches.values[0])


# def run_single(
#     device: str,
#     model: str,
#     tp: int,
#     *,
#     network_ignored: bool = True,
#     actual_inference_time_s: Optional[float] = None,
#     emit_logs: bool = False,
# ) -> Dict[str, object]:
#     spec, hw_config_path, model_config_path = _build_spec(device, model, tp, idx=0, network_ignored=network_ignored)
#     validation_results = run_validation_suite(
#         [spec],
#         base_model_config_path=model_config_path,
#         base_hardware_config_path=hw_config_path,
#         result_parser=parse_inference_time,
#         run_perf_path=RUN_PERF,
#     )
#
#     result = validation_results[0]
#     inference_time_s = float(result.metrics.get("inference_time_s", float("nan"))) if result.success else float("nan")
#     err_detail = None if result.success else (result.error or f"{RAPID_HIER_LABEL} run failed")
#
#     if actual_inference_time_s is None:
#         try:
#             actual_inference_time_s = _lookup_actual(device, model, tp)
#         except Exception:
#             actual_inference_time_s = float("nan")
#
#     if math.isnan(inference_time_s) or actual_inference_time_s == 0 or math.isnan(actual_inference_time_s):
#         pct_error = float("nan")
#     else:
#         pct_error = abs(inference_time_s - actual_inference_time_s) / actual_inference_time_s * 100.0
#
#     if emit_logs:
#         block_lines = [
#             f"\n=== Result (device={device}, model={model}, TP={tp}) ===",
#         ]
#         if not math.isnan(inference_time_s):
#             block_lines.append(f"  {RAPID_HIER_LABEL} Inference Time: {inference_time_s:.2f}s")
#             block_lines.append(f"  Actual Inference Time:   {actual_inference_time_s:.2f}s")
#             block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
#         else:
#             block_lines.append(f"  {RAPID_HIER_LABEL} run failed. {err_detail or ''}".rstrip())
#             if result.raw_output:
#                 block_lines.append(result.raw_output.strip())
#         print("\n".join(block_lines))
#
#     return {
#         "success": result.success,
#         "inference_time_s": inference_time_s,
#         "actual_inference_time_s": actual_inference_time_s,
#         "pct_error": pct_error,
#         "error": err_detail,
#         "raw_output": result.raw_output,
#     }
#
#
def build_specs_for_device(
    device: str,
    *,
    network_ignored: bool = True,
    models: Optional[Iterable[str]] = None,
    fit_model: bool = True,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int], float], str, str]:
    data = _load_device_data(device)
    model_filter = set(models) if models is not None else None
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int], float] = {}
    base_model_path: Optional[str] = None
    hw_config_path: Optional[str] = None
    idx = 0
    for device_val, model, tp in _iter_tests(data):
        if model_filter and model not in model_filter:
            continue
        spec, hw_path, model_path = _build_spec(device_val, model, tp, idx, network_ignored, fit_model)
        specs.append(spec)
        actual_lookup[(model, int(tp))] = _lookup_actual(device_val, model, int(tp))
        base_model_path = base_model_path or model_path
        hw_config_path = hw_config_path or hw_path
        idx += 1

    if not specs:
        raise ValueError(f"No validation specs generated for device={device} (models={models}).")
    return specs, actual_lookup, base_model_path, hw_config_path


def build_nvidia_specs_for_device(
    device: str,
    *,
    network_ignored: bool = True,
    models: Optional[Iterable[str]] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int], float], str, str]:
    data = _load_nvidia_device_data(device)
    model_filter = set(models) if models is not None else None
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int, int, int, int], float] = {}
    base_model_path: Optional[str] = None
    hw_config_path: Optional[str] = None
    idx = 0
    for device_val, model, tp, input_tokens, output_tokens, concurrency in _iter_nvidia_tests(data):
        if model_filter and model not in model_filter:
            continue
        spec, hw_path, model_path = _build_nvidia_spec(
            device_val,
            model,
            int(tp),
            int(input_tokens),
            int(output_tokens),
            int(concurrency),
            idx,
            network_ignored,
        )
        specs.append(spec)
        actual_lookup[(model, int(tp), int(input_tokens), int(output_tokens), int(concurrency))] = (
            _lookup_nvidia_actual(device_val, model, int(tp), int(input_tokens), int(output_tokens), int(concurrency))
        )
        base_model_path = base_model_path or model_path
        hw_config_path = hw_config_path or hw_path
        idx += 1

    if not specs:
        raise ValueError(f"No NVIDIA validation specs generated for device={device} (models={models}).")
    return specs, actual_lookup, base_model_path, hw_config_path


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int], float]):
    rows: List[Dict[str, object]] = []
    for res in results:
        metadata = res.spec.metadata or {}
        model = metadata.get("model")
        tp = int(metadata.get("tp")) if "tp" in metadata else None
        inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        actual = actual_lookup.get((model, tp)) if model is not None and tp is not None else float("nan")
        if math.isnan(inf_time) or actual is None or actual == 0 or math.isnan(actual):
            pct_error = float("nan")
        else:
            pct_error = abs(inf_time - actual) / actual * 100.0
        rows.append(
            {
                "device": metadata.get("device"),
                "model": model,
                "tp": tp,
                "inference_time_s": inf_time,
                "actual_inference_time_s": actual,
                "pct_error": pct_error,
                "display_model": MODEL_DISPLAY.get(str(model), str(model)),
                "success": res.success,
                "error": res.error,
            }
        )
    return rows


def compute_nvidia_pct_errors(
    results,
    actual_lookup: Dict[Tuple[str, int, int, int, int], float],
):
    rows: List[Dict[str, object]] = []
    for res in results:
        metadata = res.spec.metadata or {}
        model = metadata.get("model")
        tp = int(metadata.get("tp")) if "tp" in metadata else None
        input_tokens = int(metadata.get("input_tokens")) if "input_tokens" in metadata else None
        output_tokens = int(metadata.get("output_tokens")) if "output_tokens" in metadata else None
        concurrency = int(metadata.get("concurrency")) if "concurrency" in metadata else None
        inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        actual = (
            actual_lookup.get((model, tp, input_tokens, output_tokens, concurrency))
            if model is not None and tp is not None and input_tokens is not None
            and output_tokens is not None and concurrency is not None
            else float("nan")
        )
        if math.isnan(inf_time) or actual == 0 or math.isnan(actual):
            pct_error = float("nan")
        else:
            pct_error = abs(inf_time - actual) / actual * 100.0
        rows.append(
            {
                "device": metadata.get("device"),
                "model": model,
                "tp": tp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "inference_time_s": inf_time,
                "actual_inference_time_s": actual,
                "pct_error": pct_error,
                "display_model": NVIDIA_MODEL_DISPLAY.get(str(model), str(model)),
                "success": res.success,
                "error": res.error,
                "suite": metadata.get("suite", "NVIDIA"),
            }
        )
    return rows


def _safe_ratio(pred: Optional[float], actual: Optional[float]) -> float:
    if pred is None or actual is None:
        return float("nan")
    try:
        pred_val = float(pred)
        actual_val = float(actual)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(pred_val) or math.isnan(actual_val) or actual_val == 0:
        return float("nan")
    return pred_val / actual_val


def _deep_merge_dict(
    base: Optional[Dict[str, object]],
    override: Dict[str, object],
) -> Dict[str, object]:
    merged: Dict[str, object] = copy.deepcopy(base) if base is not None else {}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged.get(key), value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_flattened_specs(specs: Sequence[ValidationSpec]) -> List[ValidationSpec]:
    flattened_specs: List[ValidationSpec] = []
    for spec in specs:
        merged_hw = _deep_merge_dict(spec.hardware_overrides, FLATTENED_HW_OVERRIDE)
        metadata = dict(spec.metadata or {})
        metadata["tool"] = RAPID_FLAT_LABEL
        flattened_specs.append(
            ValidationSpec(
                label=f"{spec.label} (flattened)",
                model_overrides=spec.model_overrides,
                hardware_overrides=merged_hw,
                metadata=metadata,
                model_config_path=spec.model_config_path,
                hardware_config_path=spec.hardware_config_path,
                order=spec.order,
            )
        )
    return flattened_specs


def plot_device(
    df: pd.DataFrame,
    device: str,
    outdir: Path,
    llmcompass_rows: Optional[List[Dict[str, object]]] = None,
    vidur_rows: Optional[List[Dict[str, object]]] = None,
    genz_rows: Optional[List[Dict[str, object]]] = None,
    flattened_rows: Optional[List[Dict[str, object]]] = None,
) -> Path | None:
    sub = df[df["device"] == device].copy()
    if sub.empty:
        return None
    sub.sort_values(["model", "TP"], inplace=True)

    # Hierarchical grouping: model cluster (primary) -> TP subgroup (secondary with two bars: seconds & actual)
    # Enforce explicit primary model order
    desired_order = ["Llama 2-7B", "Llama 2-13B", "Llama 2-70B"]
    unique_models = list(pd.unique(sub["model"]))
    models = [m for m in desired_order if m in unique_models] + [
        m for m in unique_models if m not in desired_order
    ]
    display_models = [MODEL_DISPLAY.get(m, m) for m in models]

    llmcompass_lookup = None
    if llmcompass_rows:
        llmcompass_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in llmcompass_rows
            if row.get("device") == device
        }
        if not llmcompass_lookup:
            llmcompass_lookup = None

    vidur_lookup = None
    if vidur_rows:
        vidur_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in vidur_rows
            if row.get("device") == device
        }
        if not vidur_lookup:
            vidur_lookup = None

    genz_lookup = None
    if genz_rows:
        genz_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in genz_rows
            if row.get("device") == device
        }
        if not genz_lookup:
            genz_lookup = None

    flattened_lookup = None
    if flattened_rows:
        flattened_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("inference_time_s", float("nan")))
            for row in flattened_rows
            if row.get("device") == device
        }
        if not flattened_lookup:
            flattened_lookup = None

    num_series = (
        2
        + (1 if flattened_lookup else 0)
        + (1 if llmcompass_lookup else 0)
        + (1 if vidur_lookup else 0)
        + (1 if genz_lookup else 0)
    )
    if num_series == 2:
        bar_width = 0.28
    elif num_series == 3:
        bar_width = 0.22
    else:
        bar_width = 0.18
    tp_gap = 0.08  # gap between TP subgroups inside a model cluster
    model_gap = 0.7  # gap between model clusters

    x_seconds = []
    x_actual = []
    x_flattened = []
    x_llmcompass = []
    x_vidur = []
    x_genz = []
    seconds_vals = []
    actual_vals = []
    flattened_vals = []
    llmcompass_vals = []
    vidur_vals = []
    genz_vals = []
    tp_midpoints = []
    tp_labels = []
    model_centers = []
    model_bounds = []

    current_x = 0.0
    flat_offset = None
    llm_offset = None
    vidur_offset = None
    genz_offset = None
    offset_idx = 2
    if flattened_lookup:
        flat_offset = offset_idx
        offset_idx += 1
    if llmcompass_lookup:
        llm_offset = offset_idx
        offset_idx += 1
    if vidur_lookup:
        vidur_offset = offset_idx
        offset_idx += 1
    if genz_lookup:
        genz_offset = offset_idx
        offset_idx += 1

    for m in models:
        m_rows = sub[sub["model"] == m]
        tps = sorted(m_rows["TP"].unique())
        cluster_start = current_x
        for tp in tps:
            row = m_rows[m_rows["TP"] == tp].iloc[0]
            x_sec = current_x
            x_act = current_x + bar_width
            x_flat = current_x + (flat_offset * bar_width) if flat_offset is not None else None
            x_llm = current_x + (llm_offset * bar_width) if llm_offset is not None else None
            x_vid = current_x + (vidur_offset * bar_width) if vidur_offset is not None else None
            x_gen = current_x + (genz_offset * bar_width) if genz_offset is not None else None
            x_seconds.append(x_sec)
            x_actual.append(x_act)
            seconds_vals.append(row["seconds"])
            actual_vals.append(row["actual"])
            if flattened_lookup:
                x_flattened.append(x_flat)
                flattened_vals.append(flattened_lookup.get((m, int(tp)), float("nan")))
            if llmcompass_lookup:
                x_llmcompass.append(x_llm)
                llmcompass_vals.append(llmcompass_lookup.get((m, int(tp)), float("nan")))
            if vidur_lookup:
                x_vidur.append(x_vid)
                vidur_vals.append(vidur_lookup.get((m, int(tp)), float("nan")))
            if genz_lookup:
                x_genz.append(x_gen)
                genz_vals.append(genz_lookup.get((m, int(tp)), float("nan")))
            # Midpoint of the TP subgroup (between the two bars)
            span = num_series * bar_width
            tp_mid = x_sec + span / 2
            tp_midpoints.append(tp_mid)
            tp_labels.append(f"TP{tp}")
            current_x += span + tp_gap
        cluster_end = current_x - tp_gap  # last subgroup end (exclude trailing tp_gap)
        model_centers.append((cluster_start + cluster_end) / 2)
        model_bounds.append((cluster_start, cluster_end))
        current_x += model_gap  # gap after cluster

    # Dynamic figure width
    fig_w = max(8.0, 0.12 * len(tp_midpoints) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bars_seconds = ax.bar(
        x_seconds,
        seconds_vals,
        bar_width,
        label=RAPID_HIER_LABEL,
        color=TOOL_COLORS[RAPID_HIER_LABEL],
    )
    bars_actual = ax.bar(
        x_actual,
        actual_vals,
        bar_width,
        label="Actual",
        color=TOOL_COLORS["Actual"],
    )
    if flattened_vals:
        ax.bar(
            x_flattened,
            flattened_vals,
            bar_width,
            label=RAPID_FLAT_LABEL,
            color=TOOL_COLORS[RAPID_FLAT_LABEL],
        )
    if llmcompass_vals:
        ax.bar(
            x_llmcompass,
            llmcompass_vals,
            bar_width,
            label="LLMCompass",
            color=TOOL_COLORS["LLMCompass"],
        )
    if vidur_vals:
        ax.bar(x_vidur, vidur_vals, bar_width, label="Vidur", color=TOOL_COLORS["Vidur"])
    if genz_vals:
        ax.bar(x_genz, genz_vals, bar_width, label="GenZ", color=TOOL_COLORS["GenZ"])

    # Primary ticks: model names at cluster centers
    ax.set_xticks(model_centers)
    ax.set_xticklabels(display_models, fontsize=11)
    ax.set_ylabel("Inference Latency (s)")
    ax.set_title(
        f"Validation of Inference Latency on Systems of {device} (network ignored)"
    )

    # Secondary TP labels ABOVE bars (centered over subgroup)
    ymin, ymax = ax.get_ylim()
    pad = 0.02 * (ymax - ymin)
    subgroup_max = [max(s, a) for s, a in zip(seconds_vals, actual_vals)]
    if llmcompass_vals:
        subgroup_max = [max(v, l) for v, l in zip(subgroup_max, llmcompass_vals)]
    if flattened_vals:
        subgroup_max = [max(v, f) for v, f in zip(subgroup_max, flattened_vals)]
    if vidur_vals:
        subgroup_max = [max(v, l) for v, l in zip(subgroup_max, vidur_vals)]
    if genz_vals:
        subgroup_max = [max(v, l) for v, l in zip(subgroup_max, genz_vals)]
    new_ymax = max(ymax, max(subgroup_max) + 3 * pad)
    ax.set_ylim(ymin, new_ymax)
    for mid, lbl, v in zip(tp_midpoints, tp_labels, subgroup_max):
        ax.text(mid, v + pad, lbl, ha="center", va="bottom", fontsize=8)

    # Draw light separators between model clusters
    for start, end in model_bounds:
        ax.axvspan(
            start - bar_width * 0.5,
            end - bar_width * 0.5,
            facecolor="#000000",
            alpha=0.02,
        )

    # Custom legend
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.02, y=0.1)

    outpath = outdir / f"inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_nvidia_device(
    df: pd.DataFrame,
    device: str,
    outdir: Path,
    llmcompass_rows: Optional[List[Dict[str, object]]] = None,
    vidur_rows: Optional[List[Dict[str, object]]] = None,
    genz_rows: Optional[List[Dict[str, object]]] = None,
    flattened_rows: Optional[List[Dict[str, object]]] = None,
) -> Path | None:
    sub = df[df["device"] == device].copy()
    if sub.empty:
        return None
    sub.sort_values(["input_tokens", "output_tokens", "concurrency"], inplace=True)

    labels = [
        f"in{int(row['input_tokens'])}-out{int(row['output_tokens'])}-bs{int(row['concurrency'])}"
        for _, row in sub.iterrows()
    ]
    seconds_vals = sub["seconds"].astype(float).tolist()
    actual_vals = sub["actual"].astype(float).tolist()
    llmcompass_vals = None
    if llmcompass_rows:
        llmcompass_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
                row.get("tp"),
            ): float(row.get("total_latency", float("nan")))
            for row in llmcompass_rows
        }
        llmcompass_vals = [
            llmcompass_lookup.get(
                (int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"]), int(row["TP"])),
                float("nan"),
            )
            for _, row in sub.iterrows()
        ]

    vidur_vals = None
    if vidur_rows:
        vidur_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
                row.get("tp"),
            ): float(row.get("total_latency", float("nan")))
            for row in vidur_rows
            if row.get("device") == device
        }
        vidur_vals = [
            vidur_lookup.get(
                (int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"]), int(row["TP"])),
                float("nan"),
            )
            for _, row in sub.iterrows()
        ]

    genz_vals = None
    if genz_rows:
        genz_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
                row.get("tp"),
            ): float(row.get("total_latency", float("nan")))
            for row in genz_rows
            if row.get("device") == device
        }
        genz_vals = [
            genz_lookup.get(
                (int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"]), int(row["TP"])),
                float("nan"),
            )
            for _, row in sub.iterrows()
        ]

    flattened_vals = None
    if flattened_rows:
        flattened_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
                row.get("tp"),
            ): float(row.get("inference_time_s", float("nan")))
            for row in flattened_rows
            if row.get("device") == device
        }
        flattened_vals = [
            flattened_lookup.get(
                (int(row["input_tokens"]), int(row["output_tokens"]), int(row["concurrency"]), int(row["TP"])),
                float("nan"),
            )
            for _, row in sub.iterrows()
        ]

    series = [
        (RAPID_HIER_LABEL, seconds_vals, TOOL_COLORS[RAPID_HIER_LABEL]),
        ("Actual", actual_vals, TOOL_COLORS["Actual"]),
    ]
    if flattened_vals:
        series.append((RAPID_FLAT_LABEL, flattened_vals, TOOL_COLORS[RAPID_FLAT_LABEL]))
    if llmcompass_vals:
        series.append(("LLMCompass", llmcompass_vals, TOOL_COLORS["LLMCompass"]))
    if vidur_vals:
        series.append(("Vidur", vidur_vals, TOOL_COLORS["Vidur"]))
    if genz_vals:
        series.append(("GenZ", genz_vals, TOOL_COLORS["GenZ"]))

    x = list(range(len(labels)))
    num_series = len(series)
    if num_series == 2:
        bar_width = 0.35
    elif num_series == 3:
        bar_width = 0.28
    else:
        bar_width = 0.22
    fig_w = max(8.0, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    offsets = [(idx - (num_series - 1) / 2) * bar_width for idx in range(num_series)]
    for idx, (label, vals, color) in enumerate(series):
        ax.bar([i + offsets[idx] for i in x], vals, bar_width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Inference Latency (s)")
    ax.set_title(f"NVIDIA inference validation on {device}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    outpath = outdir / f"nvidia_inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# def plot_nvidia_ttft_tpot(
#     df: pd.DataFrame,
#     device: str,
#     outdir: Path,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path | None:
#     sub = df[df["device"] == device].copy()
#     if sub.empty:
#         return None
#     sub.sort_values(["input_tokens", "output_tokens", "concurrency"], inplace=True)
#
#     labels = [
#         f"in{int(row['input_tokens'])}-out{int(row['output_tokens'])}-bs{int(row['concurrency'])}"
#         for _, row in sub.iterrows()
#     ]
#     actual_ttft = (sub["ttft_ms"].astype(float) / 1000.0).tolist()
#     actual_tpot = (sub["itl_ms"].astype(float) / 1000.0).tolist()
#
#     llm_ttft = None
#     llm_tpot = None
#     if llmcompass_rows:
#         llm_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#                 row.get("tp"),
#             ): (float(row.get("ttft", float("nan"))), float(row.get("itl", float("nan"))))
#             for row in llmcompass_rows
#         }
#         llm_ttft = []
#         llm_tpot = []
#         for _, row in sub.iterrows():
#             key = (
#                 int(row["input_tokens"]),
#                 int(row["output_tokens"]),
#                 int(row["concurrency"]),
#                 int(row["TP"]),
#             )
#             ttft_val, tpot_val = llm_lookup.get(key, (float("nan"), float("nan")))
#             llm_ttft.append(ttft_val)
#             llm_tpot.append(tpot_val)
#
#     vidur_ttft = None
#     vidur_tpot = None
#     if vidur_rows:
#         vidur_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#                 row.get("tp"),
#             ): (float(row.get("ttft", float("nan"))), float(row.get("itl", float("nan"))))
#             for row in vidur_rows
#             if row.get("device") == device
#         }
#         vidur_ttft = []
#         vidur_tpot = []
#         for _, row in sub.iterrows():
#             key = (
#                 int(row["input_tokens"]),
#                 int(row["output_tokens"]),
#                 int(row["concurrency"]),
#                 int(row["TP"]),
#             )
#             ttft_val, tpot_val = vidur_lookup.get(key, (float("nan"), float("nan")))
#             vidur_ttft.append(ttft_val)
#             vidur_tpot.append(tpot_val)
#
#     genz_ttft = None
#     genz_tpot = None
#     if genz_rows:
#         genz_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#                 row.get("tp"),
#             ): (float(row.get("ttft", float("nan"))), float(row.get("itl", float("nan"))))
#             for row in genz_rows
#             if row.get("device") == device
#         }
#         genz_ttft = []
#         genz_tpot = []
#         for _, row in sub.iterrows():
#             key = (
#                 int(row["input_tokens"]),
#                 int(row["output_tokens"]),
#                 int(row["concurrency"]),
#                 int(row["TP"]),
#             )
#             ttft_val, tpot_val = genz_lookup.get(key, (float("nan"), float("nan")))
#             genz_ttft.append(ttft_val)
#             genz_tpot.append(tpot_val)
#
#     flattened_ttft = None
#     flattened_tpot = None
#     if flattened_rows:
#         flattened_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#                 row.get("tp"),
#             ): (float(row.get("ttft", float("nan"))), float(row.get("itl", float("nan"))))
#             for row in flattened_rows
#             if row.get("device") == device
#         }
#         flattened_ttft = []
#         flattened_tpot = []
#         for _, row in sub.iterrows():
#             key = (
#                 int(row["input_tokens"]),
#                 int(row["output_tokens"]),
#                 int(row["concurrency"]),
#                 int(row["TP"]),
#             )
#             ttft_val, tpot_val = flattened_lookup.get(key, (float("nan"), float("nan")))
#             flattened_ttft.append(ttft_val)
#             flattened_tpot.append(tpot_val)
#
#     x = list(range(len(labels)))
#     series_count = (
#         1
#         + (1 if flattened_ttft else 0)
#         + (1 if llm_ttft else 0)
#         + (1 if vidur_ttft else 0)
#         + (1 if genz_ttft else 0)
#     )
#     if series_count == 1:
#         bar_width = 0.45
#     elif series_count == 2:
#         bar_width = 0.35
#     else:
#         bar_width = 0.28
#     fig_w = max(9.0, 0.5 * len(labels))
#     fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.5), sharey=False)
#
#     for ax, actual_vals, llm_vals, title in (
#         (axes[0], actual_ttft, llm_ttft, "TTFT"),
#         (axes[1], actual_tpot, llm_tpot, "TPOT"),
#     ):
#         series = [("Actual", actual_vals, TOOL_COLORS["Actual"])]
#         if title == "TTFT" and flattened_ttft:
#             series.append((RAPID_FLAT_LABEL, flattened_ttft, TOOL_COLORS[RAPID_FLAT_LABEL]))
#         if title == "TPOT" and flattened_tpot:
#             series.append((RAPID_FLAT_LABEL, flattened_tpot, TOOL_COLORS[RAPID_FLAT_LABEL]))
#         if llm_vals:
#             series.append(("LLMCompass", llm_vals, TOOL_COLORS["LLMCompass"]))
#         if title == "TTFT" and vidur_ttft:
#             series.append(("Vidur", vidur_ttft, TOOL_COLORS["Vidur"]))
#         if title == "TPOT" and vidur_tpot:
#             series.append(("Vidur", vidur_tpot, TOOL_COLORS["Vidur"]))
#         if title == "TTFT" and genz_ttft:
#             series.append(("GenZ", genz_ttft, TOOL_COLORS["GenZ"]))
#         if title == "TPOT" and genz_tpot:
#             series.append(("GenZ", genz_tpot, TOOL_COLORS["GenZ"]))
#
#         offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#         for idx, (label, vals, color) in enumerate(series):
#             ax.bar([i + offsets[idx] for i in x], vals, bar_width, label=label, color=color)
#         ax.set_title(title)
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
#         ax.set_ylabel("Seconds")
#         ax.grid(axis="y", linestyle="--", alpha=0.3)
#         ax.legend()
#
#     fig.suptitle(f"NVIDIA TTFT/TPOT validation on {device}")
#     fig.tight_layout()
#     outpath = outdir / f"nvidia_ttft_tpot_{device}.png"
#     fig.savefig(outpath, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return outpath
#
#
# def _plot_nvidia_error_bars(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     labels: List[str] = []
#     keys: List[Tuple[object, object, object, object, object]] = []
#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         input_tokens = row.get("input_tokens")
#         output_tokens = row.get("output_tokens")
#         concurrency = row.get("concurrency")
#         tp = row.get("tp")
#         labels.append(f"{model} in{input_tokens}-out{output_tokens}-bs{concurrency}")
#         keys.append((row.get("model"), tp, input_tokens, output_tokens, concurrency))

#     rapid_errors = [float(row.get("pct_error", float("nan"))) for row in rows]
#     llmcompass_rows = llmcompass_rows or []
#     llmcompass_lookup = {
#         (row.get("model"), row.get("tp"), row.get("input_tokens"), row.get("output_tokens"), row.get("concurrency")): float(
#             row.get("pct_error", float("nan"))
#         )
#         for row in llmcompass_rows
#     }
#     llmcompass_errors = [llmcompass_lookup.get(key, float("nan")) for key in keys]

#     vidur_rows = vidur_rows or []
#     vidur_lookup = {
#         (row.get("model"), row.get("tp"), row.get("input_tokens"), row.get("output_tokens"), row.get("concurrency")): float(
#             row.get("pct_error", float("nan"))
#         )
#         for row in vidur_rows
#     }
#     vidur_errors = [vidur_lookup.get(key, float("nan")) for key in keys]

#     genz_rows = genz_rows or []
#     genz_lookup = {
#         (row.get("model"), row.get("tp"), row.get("input_tokens"), row.get("output_tokens"), row.get("concurrency")): float(
#             row.get("pct_error", float("nan"))
#         )
#         for row in genz_rows
#     }
#     genz_errors = [genz_lookup.get(key, float("nan")) for key in keys]

#     series = [(RAPID_HIER_LABEL, rapid_errors, TOOL_COLORS[RAPID_HIER_LABEL])]
#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (row.get("model"), row.get("tp"), row.get("input_tokens"), row.get("output_tokens"), row.get("concurrency")): float(
#                 row.get("pct_error", float("nan"))
#             )
#             for row in flattened_rows
#         }
#         flattened_errors = [flattened_lookup.get(key, float("nan")) for key in keys]
#         series.append((RAPID_FLAT_LABEL, flattened_errors, TOOL_COLORS[RAPID_FLAT_LABEL]))
#     if llmcompass_rows:
#         series.append(("LLMCompass", llmcompass_errors, TOOL_COLORS["LLMCompass"]))
#     if vidur_rows:
#         series.append(("Vidur", vidur_errors, TOOL_COLORS["Vidur"]))
#     if genz_rows:
#         series.append(("GenZ", genz_errors, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.5 * len(labels))
#     fig, ax = plt.subplots(figsize=(fig_w, 4))
#     x = list(range(len(labels)))
#     if len(series) == 1:
#         bar_width = 0.45
#     elif len(series) == 2:
#         bar_width = 0.35
#     else:
#         bar_width = 0.25
#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     for idx, (label, values, color) in enumerate(series):
#         bars = ax.bar([i + offsets[idx] for i in x], values, bar_width, color=color, label=label)
#         for rect, err in zip(bars, values):
#             if math.isnan(err):
#                 continue
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
#     if len(series) > 1:
#         ax.legend()

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
#     ax.set_ylabel("Percent Error")
#     ax.set_title(title)
#     ax.grid(axis="y", linestyle="--", alpha=0.3)
#     fig.tight_layout()
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return path


# def _plot_nvidia_normalized_bars(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     labels: List[str] = []
#     keys: List[Tuple[object, object, object, object, object]] = []
#     actual_lookup: Dict[Tuple[object, object, object, object, object], float] = {}
#     rapid_vals: List[float] = []

#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         input_tokens = row.get("input_tokens")
#         output_tokens = row.get("output_tokens")
#         concurrency = row.get("concurrency")
#         tp = row.get("tp")
#         labels.append(f"{model} in{input_tokens}-out{output_tokens}-bs{concurrency}")
#         key = (row.get("model"), tp, input_tokens, output_tokens, concurrency)
#         keys.append(key)
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_vals.append(_safe_ratio(row.get("inference_time_s"), actual))

#     series = [(RAPID_HIER_LABEL, rapid_vals, TOOL_COLORS[RAPID_HIER_LABEL])]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#         }
#         flattened_vals = [
#             _safe_ratio(flattened_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append((RAPID_FLAT_LABEL, flattened_vals, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#         }
#         llmcompass_vals = [
#             _safe_ratio(llmcompass_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("LLMCompass", llmcompass_vals, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#         }
#         vidur_vals = [
#             _safe_ratio(vidur_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("Vidur", vidur_vals, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#         }
#         genz_vals = [
#             _safe_ratio(genz_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("GenZ", genz_vals, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.55 * len(labels))
#     fig, ax = plt.subplots(figsize=(fig_w, 4.5))

#     x = list(range(len(labels)))
#     all_vals: List[float] = []
#     for _, values, _ in series:
#         all_vals.extend([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)])
#     all_vals.append(1.0)
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         if math.isfinite(min_val) and math.isfinite(max_val):
#             span = max(max_val - min_val, 0.1)
#             pad = 0.05 * span
#             y_limits = (min_val - pad, max_val + pad)
#         else:
#             y_limits = None
#     else:
#         y_limits = None

#     if len(series) == 1:
#         bar_width = 0.6
#     elif len(series) == 2:
#         bar_width = 0.35
#     else:
#         bar_width = 0.25
#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     for idx, (label, values, color) in enumerate(series):
#         ax.bar([i + offsets[idx] for i in x], values, bar_width, color=color, label=label)

#     ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#     if y_limits is not None:
#         ax.set_ylim(*y_limits)
#     ax.set_ylabel("Pred / Actual")
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
#     ax.set_title(title)
#     ax.grid(axis="y", linestyle="--", alpha=0.3)
#     if len(series) > 1:
#         ax.legend()

#     fig.tight_layout()
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


# def _plot_nvidia_normalized_bars_by_tool(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     labels: List[str] = []
#     keys: List[Tuple[object, object, object, object, object]] = []
#     actual_lookup: Dict[Tuple[object, object, object, object, object], float] = {}
#     rapid_vals: List[float] = []

#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         input_tokens = row.get("input_tokens")
#         output_tokens = row.get("output_tokens")
#         concurrency = row.get("concurrency")
#         tp = row.get("tp")
#         labels.append(f"{model} in{input_tokens}-out{output_tokens}-bs{concurrency}")
#         key = (row.get("model"), tp, input_tokens, output_tokens, concurrency)
#         keys.append(key)
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_vals.append(_safe_ratio(row.get("inference_time_s"), actual))

#     series = [(RAPID_HIER_LABEL, rapid_vals, TOOL_COLORS[RAPID_HIER_LABEL])]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#         }
#         flattened_vals = [
#             _safe_ratio(flattened_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append((RAPID_FLAT_LABEL, flattened_vals, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#         }
#         llmcompass_vals = [
#             _safe_ratio(llmcompass_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("LLMCompass", llmcompass_vals, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#         }
#         vidur_vals = [
#             _safe_ratio(vidur_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("Vidur", vidur_vals, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (
#                 row.get("model"),
#                 row.get("tp"),
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#         }
#         genz_vals = [
#             _safe_ratio(genz_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("GenZ", genz_vals, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.55 * len(labels))
#     fig_h = max(2.6, 2.6 * len(series))
#     fig, axes = plt.subplots(len(series), 1, figsize=(fig_w, fig_h), sharex=True)
#     if len(series) == 1:
#         axes = [axes]

#     x = list(range(len(labels)))
#     all_vals: List[float] = []
#     for _, values, _ in series:
#         all_vals.extend([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)])
#     all_vals.append(1.0)
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         if math.isfinite(min_val) and math.isfinite(max_val):
#             span = max(max_val - min_val, 0.1)
#             pad = 0.05 * span
#             y_limits = (min_val - pad, max_val + pad)
#         else:
#             y_limits = None
#     else:
#         y_limits = None

#     legend_handles = [
#         plt.Rectangle((0, 0), 1, 1, color=color)
#         for _, _, color in series
#     ]
#     legend_labels = [label for label, _, _ in series]

#     for ax, (label, values, color) in zip(axes, series):
#         ax.bar(x, values, 0.6, color=color)
#         ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#         if y_limits is not None:
#             ax.set_ylim(*y_limits)
#         ax.set_ylabel("Pred / Actual")
#         ax.set_title(label)
#         ax.grid(axis="y", linestyle="--", alpha=0.3)

#     for ax in axes[:-1]:
#         ax.set_xticks(x)
#         ax.set_xticklabels([])
#     axes[-1].set_xticks(x)
#     axes[-1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

#     if legend_labels:
#         fig.legend(
#             legend_handles,
#             legend_labels,
#             loc="upper center",
#             bbox_to_anchor=(0.5, 0.95),
#             ncol=len(legend_labels),
#         )

#     fig.suptitle(title)
#     fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


# def _plot_nvidia_normalized_grid_by_tokens(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     device: Optional[str] = None,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     filtered_rows = [
#         row for row in rows
#         if device is None or row.get("device") in (None, device)
#     ]
#     if not filtered_rows:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         return path

#     token_pairs = sorted(
#         {(
#             int(row.get("input_tokens")),
#             int(row.get("output_tokens")),
#         ) for row in filtered_rows if row.get("input_tokens") is not None and row.get("output_tokens") is not None}
#     )
#     concurrencies = sorted(
#         {int(row.get("concurrency")) for row in filtered_rows if row.get("concurrency") is not None}
#     )

#     actual_lookup: Dict[Tuple[object, object, object], float] = {}
#     rapid_lookup: Dict[Tuple[object, object, object], float] = {}
#     for row in filtered_rows:
#         key = (
#             row.get("input_tokens"),
#             row.get("output_tokens"),
#             row.get("concurrency"),
#         )
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_lookup[key] = _safe_ratio(row.get("inference_time_s"), actual)

#     series: List[Tuple[str, Dict[Tuple[object, object, object], float], str]] = [
#         (RAPID_HIER_LABEL, rapid_lookup, TOOL_COLORS[RAPID_HIER_LABEL]),
#     ]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         flattened_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in flattened_lookup.items()
#         }
#         if any(math.isfinite(v) for v in flattened_ratios.values()):
#             series.append((RAPID_FLAT_LABEL, flattened_ratios, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         llmcompass_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in llmcompass_lookup.items()
#         }
#         if any(math.isfinite(v) for v in llmcompass_ratios.values()):
#             series.append(("LLMCompass", llmcompass_ratios, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         vidur_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in vidur_lookup.items()
#         }
#         if any(math.isfinite(v) for v in vidur_ratios.values()):
#             series.append(("Vidur", vidur_ratios, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (
#                 row.get("input_tokens"),
#                 row.get("output_tokens"),
#                 row.get("concurrency"),
#             ): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         genz_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in genz_lookup.items()
#         }
#         if any(math.isfinite(v) for v in genz_ratios.values()):
#             series.append(("GenZ", genz_ratios, TOOL_COLORS["GenZ"]))

#     all_vals: List[float] = [1.0]
#     for _, values, _ in series:
#         all_vals.extend([v for v in values.values() if isinstance(v, (int, float)) and math.isfinite(v)])
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         span = max(max_val - min_val, 0.1)
#         pad = 0.05 * span
#         y_limits = (min_val - pad, max_val + pad)
#     else:
#         y_limits = None

#     ncols = min(3, max(1, len(concurrencies)))
#     nrows = math.ceil(len(concurrencies) / ncols) if concurrencies else 1
#     subplot_w = max(3.6, 0.55 * len(token_pairs))
#     fig_w = max(6.0, subplot_w * ncols)
#     fig_h = 2.5 * nrows
#     fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
#     axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

#     if len(series) == 1:
#         bar_width = 0.6
#     elif len(series) == 2:
#         bar_width = 0.35
#     else:
#         bar_width = 0.25
#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     x = list(range(len(token_pairs)))
#     x_labels = [f"in{inp}-out{out}" for inp, out in token_pairs]

#     for idx, concurrency in enumerate(concurrencies):
#         ax = axes_list[idx]
#         for tool_idx, (label, values, color) in enumerate(series):
#             heights = []
#             for inp, out in token_pairs:
#                 key = (inp, out, concurrency)
#                 heights.append(values.get(key, float("nan")))
#             ax.bar([i + offsets[tool_idx] for i in x], heights, bar_width, color=color, label=label)
#         ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#         if y_limits is not None:
#             ax.set_ylim(*y_limits)
#         ax.set_title(f"bs={concurrency}")
#         ax.set_xticks(x)
#         ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
#         ax.grid(axis="y", linestyle="--", alpha=0.3)
#         if idx % ncols == 0:
#             ax.set_ylabel("Pred / Actual")

#     for j in range(len(concurrencies), len(axes_list)):
#         axes_list[j].axis("off")

#     fig.suptitle(title, y=0.98)
#     if series:
#         legend_handles = [
#             plt.Rectangle((0, 0), 1, 1, color=color)
#             for _, _, color in series
#         ]
#         legend_labels = [label for label, _, _ in series]
#         fig.legend(
#             legend_handles,
#             legend_labels,
#             loc="upper center",
#             bbox_to_anchor=(0.5, 0.92),
#             ncol=len(legend_labels),
#         )
#     fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


def run(
    enable_plot: bool = False,
    network_ignored: bool = True,
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    emit_logs: bool = True,
    test_model: bool = False,
):
    pct_errors = []
    data = _load_device_data(device)

    specs, actual_lookup, base_model_path, hw_config_path = build_specs_for_device(
        device, network_ignored=network_ignored, models=models, fit_model=not test_model
    )

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    rows = compute_pct_errors(validation_results, actual_lookup)
    flattened_rows = None
    if COMPARE_FLATTENED:
        flattened_specs = _build_flattened_specs(specs)
        flattened_results = run_validation_suite(
            flattened_specs,
            base_model_config_path=base_model_path,
            base_hardware_config_path=hw_config_path,
            result_parser=parse_inference_time,
            run_perf_path=RUN_PERF,
        )
        flattened_rows = compute_pct_errors(flattened_results, actual_lookup)

    for row in rows:
        pct_error = float(row["pct_error"])
        pct_errors.append(pct_error)
        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "seconds",
        ] = row["inference_time_s"]

        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "pct_error",
        ] = pct_error

        if emit_logs:
            block_lines = [
                f"\n=== Result (device={row['device']}, model={row['model']}, TP={row['tp']}) ===",
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  {RAPID_HIER_LABEL} Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  {RAPID_HIER_LABEL} run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    if emit_logs and flattened_rows:
        for row in flattened_rows:
            pct_error = float(row["pct_error"])
            block_lines = [
                f"\n=== Result (device={row['device']}, model={row['model']}, TP={row['tp']}) ===",
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  {RAPID_FLAT_LABEL} Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  {RAPID_FLAT_LABEL} run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))



    # if enable_plot:
    #     out_dir = os.path.join(PROJECT_ROOT, "output", "validation", "inf")
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_dir = Path(out_dir)

    #     outputs = []
    #     llmcompass_rows = None
    #     if COMPARE_LLMCOMPASS:
    #         llmcompass_rows = _load_llmcompass_imec_errors(LLMCOMPASS_IMEC, device=device)
    #     vidur_rows = None
    #     if COMPARE_VIDUR:
    #         vidur_rows = _load_vidur_imec_errors(VIDUR_IMEC)
    #     genz_rows = None
    #     if COMPARE_GENZ:
    #         genz_rows = _load_genz_imec_errors(GENZ_IMEC)
    #     for plot_device_name in ["A100", "H100"]:
    #         out = plot_device(
    #             data,
    #             plot_device_name,
    #             out_dir,
    #             llmcompass_rows=llmcompass_rows if plot_device_name == "A100" else None,
    #             vidur_rows=vidur_rows,
    #             genz_rows=genz_rows,
    #             flattened_rows=flattened_rows,
    #         )
    #         if out is not None:
    #             outputs.append(out)
    #     # Secondary bar plot matching the training style (combined devices).
    #     bar_path = _plot_error_bars(
    #         rows,
    #         path=out_dir / "inf_errors_bar.png",
    #         title="Inference validation (combined)",
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(bar_path)
    #     ratio_path = _plot_normalized_bars(
    #         rows,
    #         path=out_dir / "inf_ratio_bar.png",
    #         title="Inference validation normalized to actual (combined)",
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_path)
    #     ratio_tool_path = _plot_normalized_bars_by_tool(
    #         rows,
    #         path=out_dir / "inf_ratio_bar_by_tool.png",
    #         title="Inference validation normalized to actual (by tool)",
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_tool_path)
    #     ratio_grid_path = _plot_normalized_grid_by_pair(
    #         rows,
    #         path=out_dir / "inf_ratio_grid_by_pair.png",
    #         title="Inference validation normalized to actual (per model/TP)",
    #         device=run_device,
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_grid_path)
    #     if not outputs:
    #         print("No plots generated (no matching device rows).")
    #     else:
    #         for p in outputs:
    #             print(f"Saved: {p}")


    valid_errors = [e for e in pct_errors if not math.isnan(e)]
    avg_abs_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    if emit_logs:
        print("Average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error, "rows": rows, "flattened_rows": flattened_rows}


def run_nvidia(
    enable_plot: bool = False,
    network_ignored: bool = True,
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    emit_logs: bool = True,
):
    pct_errors = []
    data = _load_nvidia_device_data(device)

    specs, actual_lookup, base_model_path, hw_config_path = build_nvidia_specs_for_device(
        device, network_ignored=network_ignored, models=models
    )

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    rows = compute_nvidia_pct_errors(validation_results, actual_lookup)
    flattened_rows = None
    if COMPARE_FLATTENED:
        flattened_specs = _build_flattened_specs(specs)
        flattened_results = run_validation_suite(
            flattened_specs,
            base_model_config_path=base_model_path,
            base_hardware_config_path=hw_config_path,
            result_parser=parse_inference_time,
            run_perf_path=RUN_PERF,
        )
        flattened_rows = compute_nvidia_pct_errors(flattened_results, actual_lookup)

    for row in rows:
        pct_error = float(row["pct_error"])
        pct_errors.append(pct_error)
        data.loc[
            (data["device"] == row["device"])
            & (data["model"] == row["model"])
            & (data["TP"] == row["tp"])
            & (data["input_tokens"] == row["input_tokens"])
            & (data["output_tokens"] == row["output_tokens"])
            & (data["concurrency"] == row["concurrency"]),
            "seconds",
        ] = row["inference_time_s"]

        data.loc[
            (data["device"] == row["device"])
            & (data["model"] == row["model"])
            & (data["TP"] == row["tp"])
            & (data["input_tokens"] == row["input_tokens"])
            & (data["output_tokens"] == row["output_tokens"])
            & (data["concurrency"] == row["concurrency"]),
            "pct_error",
        ] = pct_error

        if emit_logs:
            block_lines = [
                (
                    f"\n=== NVIDIA Result (device={row['device']}, model={row['model']}, "
                    f"TP={row['tp']}, in={row['input_tokens']}, out={row['output_tokens']}, "
                    f"bs={row['concurrency']}) ==="
                ),
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  {RAPID_HIER_LABEL} Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  {RAPID_HIER_LABEL} run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    if emit_logs and flattened_rows:
        for row in flattened_rows:
            pct_error = float(row["pct_error"])
            block_lines = [
                (
                    f"\n=== NVIDIA Result (device={row['device']}, model={row['model']}, "
                    f"TP={row['tp']}, in={row['input_tokens']}, out={row['output_tokens']}, "
                    f"bs={row['concurrency']}) ==="
                ),
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  {RAPID_FLAT_LABEL} Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  {RAPID_FLAT_LABEL} run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    # if enable_plot:
    #     out_dir = os.path.join(PROJECT_ROOT, "output", "validation", "inf")
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_dir = Path(out_dir)
    #     outputs = []
    #     llmcompass_rows = None
    #     if COMPARE_LLMCOMPASS:
    #         llmcompass_rows = _load_llmcompass_nvidia_errors(LLMCOMPASS_NVIDIA, device=device)
    #     vidur_rows = None
    #     if COMPARE_VIDUR:
    #         vidur_rows = _load_vidur_nvidia_errors(VIDUR_NVIDIA, device=device)
    #     genz_rows = None
    #     if COMPARE_GENZ:
    #         genz_rows = _load_genz_nvidia_errors(GENZ_NVIDIA, device=device)
    #     out = plot_nvidia_device(
    #         data,
    #         device,
    #         out_dir,
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     if out is not None:
    #         outputs.append(out)

    #     out = plot_nvidia_ttft_tpot(
    #         data,
    #         device,
    #         out_dir,
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
        
    #     if out is not None:
    #         outputs.append(out)
    #     error_bar_path = _plot_nvidia_error_bars(
    #         rows,
    #         path=out_dir / f"nvidia_inf_errors_bar_{device}.png",
    #         title=f"NVIDIA inference validation ({device})",
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(error_bar_path)
    #     ratio_bar_path = _plot_nvidia_normalized_bars(
    #         rows,
    #         path=out_dir / f"nvidia_inf_ratio_bar_{device}.png",
    #         title=f"NVIDIA inference normalized to actual ({device})",
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_bar_path)
    #     ratio_tool_path = _plot_nvidia_normalized_bars_by_tool(
    #         rows,
    #         path=out_dir / f"nvidia_inf_ratio_bar_by_tool_{device}.png",
    #         title=f"NVIDIA inference normalized to actual by tool ({device})",
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_tool_path)
    #     ratio_grid_path = _plot_nvidia_normalized_grid_by_tokens(
    #         rows,
    #         path=out_dir / f"nvidia_inf_ratio_grid_by_tokens_{device}.png",
    #         title=f"NVIDIA inference normalized to actual by tokens ({device})",
    #         device=device,
    #         llmcompass_rows=llmcompass_rows,
    #         vidur_rows=vidur_rows,
    #         genz_rows=genz_rows,
    #         flattened_rows=flattened_rows,
    #     )
    #     outputs.append(ratio_grid_path)
    #     if not outputs:
    #         print("No NVIDIA plots generated (no matching device rows).")
    #     else:
    #         for p in outputs:
    #             print(f"Saved: {p}")

    valid_errors = [e for e in pct_errors if not math.isnan(e)]
    avg_abs_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    if emit_logs:
        print("NVIDIA average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error, "rows": rows, "flattened_rows": flattened_rows}


# def _plot_error_bars(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     # Build labels like "<model> xTP<tp>", drop device prefixes.
#     labels: List[str] = []
#     errors: List[float] = []
#     keys: List[Tuple[object, object]] = []
#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         tp = row.get("tp")
#         labels.append(f"{model} xTP{tp}")
#         errors.append(float(row.get("pct_error", float("nan"))))
#         keys.append((row.get("model"), tp))

#     series = [(RAPID_HIER_LABEL, errors, TOOL_COLORS[RAPID_HIER_LABEL])]
#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("pct_error", float("nan")))
#             for row in flattened_rows
#         }
#         flattened_errors = [flattened_lookup.get(key, float("nan")) for key in keys]
#         series.append((RAPID_FLAT_LABEL, flattened_errors, TOOL_COLORS[RAPID_FLAT_LABEL]))
#     if vidur_rows:
#         vidur_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("pct_error", float("nan")))
#             for row in vidur_rows
#         }
#         vidur_errors = [vidur_lookup.get(key, float("nan")) for key in keys]
#         series.append(("Vidur", vidur_errors, TOOL_COLORS["Vidur"]))
#     if genz_rows:
#         genz_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("pct_error", float("nan")))
#             for row in genz_rows
#         }
#         genz_errors = [genz_lookup.get(key, float("nan")) for key in keys]
#         series.append(("GenZ", genz_errors, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.6 * len(labels))
#     fig, ax = plt.subplots(figsize=(fig_w, 4))
#     x = list(range(len(labels)))
#     if len(series) == 1:
#         bar_width = 0.45
#     else:
#         bar_width = 0.35
#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     for idx, (label, vals, color) in enumerate(series):
#         bars = ax.bar([i + offsets[idx] for i in x], vals, bar_width, color=color, label=label)
#         for rect, err in zip(bars, vals):
#             if math.isnan(err):
#                 continue
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
#     ax.set_xticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
#     ax.set_ylabel("Percent Error")
#     ax.set_title(title)
#     if len(series) > 1:
#         ax.legend()
#     ax.grid(axis="y", linestyle="--", alpha=0.3)
#     fig.tight_layout()
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return path


# def _plot_normalized_bars(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     labels: List[str] = []
#     keys: List[Tuple[object, object]] = []
#     actual_lookup: Dict[Tuple[object, object], float] = {}
#     rapid_vals: List[float] = []

#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         tp = row.get("tp")
#         labels.append(f"{model} xTP{tp}")
#         key = (row.get("model"), tp)
#         keys.append(key)
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_vals.append(_safe_ratio(row.get("inference_time_s"), actual))

#     series = [(RAPID_HIER_LABEL, rapid_vals, TOOL_COLORS[RAPID_HIER_LABEL])]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#         }
#         flattened_vals = [
#             _safe_ratio(flattened_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append((RAPID_FLAT_LABEL, flattened_vals, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#         }
#         llmcompass_vals = [
#             _safe_ratio(llmcompass_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("LLMCompass", llmcompass_vals, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#         }
#         vidur_vals = [
#             _safe_ratio(vidur_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("Vidur", vidur_vals, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#         }
#         genz_vals = [
#             _safe_ratio(genz_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("GenZ", genz_vals, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.6 * len(labels))
#     fig, ax = plt.subplots(figsize=(fig_w, 4.5))

#     x = list(range(len(labels)))
#     all_vals: List[float] = []
#     for _, values, _ in series:
#         all_vals.extend([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)])
#     all_vals.append(1.0)
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         if math.isfinite(min_val) and math.isfinite(max_val):
#             span = max(max_val - min_val, 0.1)
#             pad = 0.05 * span
#             y_limits = (min_val - pad, max_val + pad)
#         else:
#             y_limits = None
#     else:
#         y_limits = None

#     if len(series) == 1:
#         bar_width = 0.6
#     elif len(series) == 2:
#         bar_width = 0.35
#     else:
#         bar_width = 0.25
#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     for idx, (label, values, color) in enumerate(series):
#         ax.bar([i + offsets[idx] for i in x], values, bar_width, color=color, label=label)

#     ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#     if y_limits is not None:
#         ax.set_ylim(*y_limits)
#     ax.set_ylabel("Pred / Actual")
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
#     ax.set_title(title)
#     ax.grid(axis="y", linestyle="--", alpha=0.3)
#     if len(series) > 1:
#         ax.legend()

#     fig.tight_layout()
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


# def _plot_normalized_bars_by_tool(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     labels: List[str] = []
#     keys: List[Tuple[object, object]] = []
#     actual_lookup: Dict[Tuple[object, object], float] = {}
#     rapid_vals: List[float] = []

#     for row in rows:
#         model = row.get("display_model") or row.get("model")
#         tp = row.get("tp")
#         labels.append(f"{model} xTP{tp}")
#         key = (row.get("model"), tp)
#         keys.append(key)
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_vals.append(_safe_ratio(row.get("inference_time_s"), actual))

#     series = [(RAPID_HIER_LABEL, rapid_vals, TOOL_COLORS[RAPID_HIER_LABEL])]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#         }
#         flattened_vals = [
#             _safe_ratio(flattened_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append((RAPID_FLAT_LABEL, flattened_vals, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#         }
#         llmcompass_vals = [
#             _safe_ratio(llmcompass_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("LLMCompass", llmcompass_vals, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#         }
#         vidur_vals = [
#             _safe_ratio(vidur_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("Vidur", vidur_vals, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#         }
#         genz_vals = [
#             _safe_ratio(genz_lookup.get(key), actual_lookup.get(key))
#             for key in keys
#         ]
#         series.append(("GenZ", genz_vals, TOOL_COLORS["GenZ"]))

#     fig_w = max(6.0, 0.6 * len(labels))
#     fig_h = max(2.6, 2.6 * len(series))
#     fig, axes = plt.subplots(len(series), 1, figsize=(fig_w, fig_h), sharex=True)
#     if len(series) == 1:
#         axes = [axes]

#     x = list(range(len(labels)))
#     all_vals: List[float] = []
#     for _, values, _ in series:
#         all_vals.extend([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)])
#     all_vals.append(1.0)
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         if math.isfinite(min_val) and math.isfinite(max_val):
#             span = max(max_val - min_val, 0.1)
#             pad = 0.05 * span
#             y_limits = (min_val - pad, max_val + pad)
#         else:
#             y_limits = None
#     else:
#         y_limits = None

#     for ax, (label, values, color) in zip(axes, series):
#         ax.bar(x, values, 0.6, color=color)
#         ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#         if y_limits is not None:
#             ax.set_ylim(*y_limits)
#         ax.set_ylabel("Pred / Actual")
#         ax.set_title(label)
#         ax.grid(axis="y", linestyle="--", alpha=0.3)

#     for ax in axes[:-1]:
#         ax.set_xticks(x)
#         ax.set_xticklabels([])
#     axes[-1].set_xticks(x)
#     axes[-1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

#     fig.suptitle(title)
#     fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


# def _plot_normalized_grid_by_pair(
#     rows: List[Dict[str, object]],
#     path: Path,
#     title: str,
#     device: Optional[str] = None,
#     llmcompass_rows: Optional[List[Dict[str, object]]] = None,
#     vidur_rows: Optional[List[Dict[str, object]]] = None,
#     genz_rows: Optional[List[Dict[str, object]]] = None,
#     flattened_rows: Optional[List[Dict[str, object]]] = None,
# ) -> Path:
#     filtered_rows = [
#         row for row in rows
#         if device is None or row.get("device") in (None, device)
#     ]
#     if not filtered_rows:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         return path

#     model_order = {"Llama 2-7B": 0, "Llama 2-13B": 1, "Llama 2-70B": 2}
#     models = sorted(
#         {row.get("model") for row in filtered_rows if row.get("model")},
#         key=lambda name: (model_order.get(name, 99), str(name)),
#     )
#     tps = sorted({int(row.get("tp")) for row in filtered_rows if row.get("tp") is not None})
#     if not models or not tps:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         return path

#     actual_lookup: Dict[Tuple[object, object], float] = {}
#     rapid_lookup: Dict[Tuple[object, object], float] = {}
#     for row in filtered_rows:
#         key = (row.get("model"), row.get("tp"))
#         actual = row.get("actual_inference_time_s")
#         actual_lookup[key] = float(actual) if actual is not None else float("nan")
#         rapid_lookup[key] = _safe_ratio(row.get("inference_time_s"), actual)

#     series: List[Tuple[str, Dict[Tuple[object, object], float], str]] = [
#         (RAPID_HIER_LABEL, rapid_lookup, TOOL_COLORS[RAPID_HIER_LABEL]),
#     ]

#     flattened_rows = flattened_rows or []
#     if flattened_rows:
#         flattened_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("inference_time_s", float("nan")))
#             for row in flattened_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         flattened_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in flattened_lookup.items()
#         }
#         if any(math.isfinite(v) for v in flattened_ratios.values()):
#             series.append((RAPID_FLAT_LABEL, flattened_ratios, TOOL_COLORS[RAPID_FLAT_LABEL]))

#     llmcompass_rows = llmcompass_rows or []
#     if llmcompass_rows:
#         llmcompass_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in llmcompass_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         llmcompass_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in llmcompass_lookup.items()
#         }
#         if any(math.isfinite(v) for v in llmcompass_ratios.values()):
#             series.append(("LLMCompass", llmcompass_ratios, TOOL_COLORS["LLMCompass"]))

#     vidur_rows = vidur_rows or []
#     if vidur_rows:
#         vidur_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in vidur_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         vidur_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in vidur_lookup.items()
#         }
#         if any(math.isfinite(v) for v in vidur_ratios.values()):
#             series.append(("Vidur", vidur_ratios, TOOL_COLORS["Vidur"]))

#     genz_rows = genz_rows or []
#     if genz_rows:
#         genz_lookup = {
#             (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
#             for row in genz_rows
#             if device is None or row.get("device") in (None, device)
#         }
#         genz_ratios = {
#             key: _safe_ratio(pred, actual_lookup.get(key))
#             for key, pred in genz_lookup.items()
#         }
#         if any(math.isfinite(v) for v in genz_ratios.values()):
#             series.append(("GenZ", genz_ratios, TOOL_COLORS["GenZ"]))

#     preferred_order = [RAPID_HIER_LABEL, RAPID_FLAT_LABEL, "GenZ", "Vidur"]
#     series = sorted(
#         series,
#         key=lambda item: (
#             preferred_order.index(item[0]) if item[0] in preferred_order else len(preferred_order),
#             item[0],
#         ),
#     )

#     all_vals: List[float] = [1.0]
#     for _, values, _ in series:
#         all_vals.extend([v for v in values.values() if isinstance(v, (int, float)) and math.isfinite(v)])
#     if all_vals:
#         min_val = min(all_vals)
#         max_val = max(all_vals)
#         span = max(max_val - min_val, 0.1)
#         pad = 0.05 * span
#         y_limits = (min_val - pad, max_val + pad)
#     else:
#         y_limits = None

#     nrows = 1
#     ncols = len(models)
#     fig_w = max(6.0, 3.6 * ncols)
#     fig_h = 2.5 * nrows
#     fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
#     axes_row = list(axes) if ncols > 1 else [axes]

#     if len(series) == 1:
#         bar_width = 0.6
#     elif len(series) == 2:
#         bar_width = 0.4
#     else:
#         bar_width = 0.3
#     x = list(range(len(tps)))
#     x_labels = [f"TP{tp}" for tp in tps]
#     legend_handles = [
#         plt.Rectangle((0, 0), 1, 1, color=color)
#         for _, _, color in series
#     ]
#     legend_labels = [label for label, _, _ in series]

#     offsets = [(idx - (len(series) - 1) / 2) * bar_width for idx in range(len(series))]
#     for col_idx, model in enumerate(models):
#         display_model = MODEL_DISPLAY.get(str(model), str(model))
#         ax = axes_row[col_idx]
#         has_any = False
#         model_tps = [tp for tp in tps if not (str(model) == "Llama 2-70B" and tp == 1)]
#         x = list(range(len(model_tps)))
#         x_labels = [f"TP{tp}" for tp in model_tps]
#         for tool_idx, (_, values, color) in enumerate(series):
#             heights = [values.get((model, tp), float("nan")) for tp in model_tps]
#             if any(isinstance(v, (int, float)) and math.isfinite(v) for v in heights):
#                 has_any = True
#             ax.bar([i + offsets[tool_idx] for i in x], heights, bar_width, color=color)
#         if not has_any:
#             ax.axis("off")
#             continue
#         ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
#         if y_limits is not None:
#             ax.set_ylim(*y_limits)
#         ax.set_title(display_model)
#         ax.grid(axis="y", linestyle="--", alpha=0.3)
#         ax.set_xticks(x)
#         ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
#         if col_idx == 0:
#             ax.set_ylabel("Pred / Actual")

#     fig.suptitle(title)
#     if legend_labels:
#         fig.legend(
#             legend_handles,
#             legend_labels,
#             loc="upper center",
#             bbox_to_anchor=(0.5, 0.92),
#             ncol=len(legend_labels),
#         )
#     fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     return path


def _plot_combined_a100_error_bars(
    imec_rows: List[Dict[str, object]],
    nvidia_rows: List[Dict[str, object]],
    outdir: Path,
    llmcompass_rows: Optional[List[Dict[str, object]]] = None,
    llmcompass_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    vidur_rows: Optional[List[Dict[str, object]]] = None,
    vidur_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    genz_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    genz_rows: Optional[List[Dict[str, object]]] = None,
    flattened_rows: Optional[List[Dict[str, object]]] = None,
    flattened_nvidia_rows: Optional[List[Dict[str, object]]] = None,
) -> Optional[Path]:
    imec_rows = [row for row in imec_rows if row.get("device") == "A100"]
    nvidia_rows = [row for row in nvidia_rows if row.get("device") == "A100"]
    if vidur_rows:
        vidur_rows = [row for row in vidur_rows if row.get("device") == "A100"]
    if vidur_nvidia_rows:
        vidur_nvidia_rows = [row for row in vidur_nvidia_rows if row.get("device") == "A100"]
    if genz_nvidia_rows:
        genz_nvidia_rows = [row for row in genz_nvidia_rows if row.get("device") == "A100"]
    if genz_rows:
        genz_rows = [row for row in genz_rows if row.get("device") == "A100"]
    if flattened_rows:
        flattened_rows = [row for row in flattened_rows if row.get("device") == "A100"]
    if flattened_nvidia_rows:
        flattened_nvidia_rows = [row for row in flattened_nvidia_rows if row.get("device") == "A100"]
    if not imec_rows or not nvidia_rows:
        return None

    imec_order = {"Llama 2-7B": 0, "Llama 2-13B": 1, "Llama 2-70B": 2}
    llmcompass_rows = llmcompass_rows or []
    llmcompass_models = {row.get("model") for row in llmcompass_rows if row.get("model")}
    vidur_models = {row.get("model") for row in (vidur_rows or []) if row.get("model")}
    genz_models = {row.get("model") for row in (genz_rows or []) if row.get("model")}
    imec_models = sorted(
        {row.get("model") for row in imec_rows if row.get("model")}
        | llmcompass_models
        | vidur_models
        | genz_models,
        key=lambda m: (imec_order.get(m, 99), str(m)),
    )
    model_hatch_patterns = ["", "//", "xx", "..", "oo", "**", "++"]
    model_hatches = {
        model: model_hatch_patterns[idx % len(model_hatch_patterns)]
        for idx, model in enumerate(imec_models)
        if model
    }

    font_params = {
        "font.size": plt.rcParams.get("font.size", 10.0) * 1.3,
        "axes.titlesize": plt.rcParams.get("axes.titlesize", 12.0) * 1.3,
        "axes.labelsize": plt.rcParams.get("axes.labelsize", 11.0) * 1.3,
        "xtick.labelsize": plt.rcParams.get("xtick.labelsize", 10.0) * 1.3,
        "ytick.labelsize": plt.rcParams.get("ytick.labelsize", 10.0) * 1.3,
        "legend.fontsize": plt.rcParams.get("legend.fontsize", 10.0) * 1.3,
    }

    # Top: IMEC (prefill/decode 200/200).
    imec_tps = sorted({int(row.get("tp")) for row in imec_rows if row.get("tp") is not None})
    imec_vals: Dict[Tuple[str, int], float] = {}
    for row in imec_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        imec_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    llmcompass_vals: Dict[Tuple[str, int], float] = {}
    for row in llmcompass_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        llmcompass_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    vidur_rows = vidur_rows or []
    vidur_vals: Dict[Tuple[str, int], float] = {}
    for row in vidur_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        vidur_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    genz_rows = genz_rows or []
    genz_vals: Dict[Tuple[str, int], float] = {}
    for row in genz_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        genz_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    flattened_rows = flattened_rows or []
    flattened_imec_vals: Dict[Tuple[str, int], float] = {}
    for row in flattened_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        flattened_imec_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    methods = [RAPID_HIER_LABEL]
    if flattened_imec_vals:
        methods.append(RAPID_FLAT_LABEL)
    if llmcompass_vals:
        methods.append("LLMCompass")
    if vidur_vals:
        methods.append("Vidur")
    if genz_vals:
        methods.append("GenZ")
    method_colors = {
        method: TOOL_COLORS.get(method, TOOL_COLORS[RAPID_HIER_LABEL])
        for method in methods
    }

    top_bar_width = 0.18 if len(methods) > 1 else 0.2
    group_gap = 0.3
    top_positions = []
    top_labels = []
    top_x = []
    top_heights = []
    top_colors = []
    top_hatches = []
    current_x = 0.0
    for tp in imec_tps:
        group_start = current_x
        model_span = top_bar_width * len(methods)
        for idx, model in enumerate(imec_models):
            model_start = group_start + idx * model_span
            for method_idx, method in enumerate(methods):
                if method == RAPID_HIER_LABEL:
                    value = imec_vals.get((model, tp))
                elif method == RAPID_FLAT_LABEL:
                    value = flattened_imec_vals.get((model, tp))
                elif method == "LLMCompass":
                    value = llmcompass_vals.get((model, tp))
                elif method == "Vidur":
                    value = vidur_vals.get((model, tp))
                else:
                    value = genz_vals.get((model, tp))
                if value is None or math.isnan(value):
                    continue
                top_x.append(model_start + method_idx * top_bar_width)
                top_heights.append(value)
                top_colors.append(method_colors.get(method, TOOL_COLORS[RAPID_HIER_LABEL]))
                top_hatches.append(model_hatches.get(model, ""))
        group_width = len(imec_models) * model_span
        group_center = group_start + (group_width - top_bar_width) / 2
        top_positions.append(group_center)
        top_labels.append(f"(200/200) TP{tp}")
        current_x += group_width + group_gap

    # Bottom: NVIDIA (include batch size; grouped by input/output + TP).
    nvidia_entries = []
    bs_levels = sorted(
        {int(row.get("concurrency")) for row in nvidia_rows if row.get("concurrency") is not None}
    )
    hatch_map = {1: "", 5: "xx", 25: "oo"}
    for row in nvidia_rows:
        model = row.get("model")
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (model, input_tokens, output_tokens, tp, concurrency):
            continue
        value = float(row.get("pct_error", float("nan")))
        if math.isnan(value):
            continue
        nvidia_entries.append(
            {
                "model": str(model),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "tp": int(tp),
                "concurrency": int(concurrency),
                "pct_error": value,
            }
        )
    nvidia_entries.sort(
        key=lambda item: (item["input_tokens"], item["output_tokens"], item["tp"], item["concurrency"])
    )

    group_keys = []
    nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for entry in nvidia_entries:
        key = (entry["input_tokens"], entry["output_tokens"], entry["tp"])
        if key not in group_keys:
            group_keys.append(key)
        nvidia_vals[(entry["input_tokens"], entry["output_tokens"], entry["tp"], entry["concurrency"])] = (
            entry["pct_error"]
        )

    llmcompass_nvidia_rows = llmcompass_nvidia_rows or []
    llmcompass_nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for row in llmcompass_nvidia_rows:
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (input_tokens, output_tokens, tp, concurrency):
            continue
        llmcompass_nvidia_vals[(int(input_tokens), int(output_tokens), int(tp), int(concurrency))] = float(
            row.get("pct_error", float("nan"))
        )

    vidur_nvidia_rows = vidur_nvidia_rows or []
    vidur_nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for row in vidur_nvidia_rows:
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (input_tokens, output_tokens, tp, concurrency):
            continue
        vidur_nvidia_vals[(int(input_tokens), int(output_tokens), int(tp), int(concurrency))] = float(
            row.get("pct_error", float("nan"))
        )

    genz_nvidia_rows = genz_nvidia_rows or []
    genz_nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for row in genz_nvidia_rows:
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (input_tokens, output_tokens, tp, concurrency):
            continue
        genz_nvidia_vals[(int(input_tokens), int(output_tokens), int(tp), int(concurrency))] = float(
            row.get("pct_error", float("nan"))
        )

    flattened_nvidia_rows = flattened_nvidia_rows or []
    flattened_nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for row in flattened_nvidia_rows:
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (input_tokens, output_tokens, tp, concurrency):
            continue
        flattened_nvidia_vals[(int(input_tokens), int(output_tokens), int(tp), int(concurrency))] = float(
            row.get("pct_error", float("nan"))
        )

    bottom_methods = [RAPID_HIER_LABEL]
    if flattened_nvidia_vals:
        bottom_methods.append(RAPID_FLAT_LABEL)
    if llmcompass_nvidia_vals:
        bottom_methods.append("LLMCompass")
    if vidur_nvidia_vals:
        bottom_methods.append("Vidur")
    if genz_nvidia_vals:
        bottom_methods.append("GenZ")
    bottom_method_colors = {
        method: TOOL_COLORS.get(method, TOOL_COLORS[RAPID_HIER_LABEL])
        for method in bottom_methods
    }

    bottom_bar_width = 0.18 if len(bottom_methods) > 1 else 0.2
    bottom_positions = []
    bottom_labels = []
    bottom_x = []
    bottom_heights = []
    bottom_hatches = []
    bottom_colors = []
    current_x = 0.0
    bs_count = max(1, len(bs_levels))
    for input_tokens, output_tokens, tp in group_keys:
        group_center = current_x
        for idx, bs in enumerate(bs_levels):
            bs_center = group_center + (idx - (bs_count - 1) / 2) * (bottom_bar_width * len(bottom_methods))
            bs_start = bs_center - (len(bottom_methods) - 1) * bottom_bar_width / 2
            for method_idx, method in enumerate(bottom_methods):
                if method == RAPID_HIER_LABEL:
                    value = nvidia_vals.get((input_tokens, output_tokens, tp, bs))
                elif method == RAPID_FLAT_LABEL:
                    value = flattened_nvidia_vals.get((input_tokens, output_tokens, tp, bs))
                elif method == "LLMCompass":
                    value = llmcompass_nvidia_vals.get((input_tokens, output_tokens, tp, bs))
                elif method == "Vidur":
                    value = vidur_nvidia_vals.get((input_tokens, output_tokens, tp, bs))
                else:
                    value = genz_nvidia_vals.get((input_tokens, output_tokens, tp, bs))
                if value is None or math.isnan(value):
                    continue
                bottom_x.append(bs_start + method_idx * bottom_bar_width)
                bottom_heights.append(value)
                bottom_hatches.append(hatch_map.get(bs, "///"))
                bottom_colors.append(bottom_method_colors.get(method, TOOL_COLORS[RAPID_HIER_LABEL]))
        bottom_positions.append(group_center)
        bottom_labels.append(f"({input_tokens}/{output_tokens}) TP{tp}")
        current_x += bs_count * bottom_bar_width * len(bottom_methods) + group_gap

    with plt.rc_context(font_params):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=False,
            gridspec_kw={"hspace": 0.3},
        )
        bars = axes[0].bar(
            top_x,
            top_heights,
            top_bar_width,
            color=top_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, hatch in zip(bars, top_hatches):
            rect.set_hatch(hatch)
        axes[0].set_title("NVIDIA Llama2 (bs=1)")
        axes[0].set_ylabel("")
        axes[0].set_xticks(top_positions)
        axes[0].set_xticklabels(top_labels, fontstyle="italic")
        axes[0].grid(axis="y", linestyle="--", alpha=0.8)
        if top_heights:
            top_max = max(top_heights)
            axes[0].set_ylim(0, top_max * 1.3)

        model_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=model_hatches.get(model, ""))
            for model in imec_models
        ]
        model_labels = [MODEL_DISPLAY.get(model, model) for model in imec_models]
        model_legend = axes[0].legend(model_handles, model_labels, loc="upper left")
        if len(methods) > 1:
            axes[0].add_artist(model_legend)
            method_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[method], edgecolor="black")
                for method in methods
            ]
            axes[0].legend(method_handles, methods, loc="upper right")

        bars = axes[1].bar(
            bottom_x,
            bottom_heights,
            bottom_bar_width,
            color=bottom_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, hatch in zip(bars, bottom_hatches):
            rect.set_hatch(hatch)
        axes[1].set_title("NVIDIA NIM Llama3-70B")
        axes[1].set_ylabel("")
        axes[1].set_xticks(bottom_positions)
        axes[1].set_xticklabels(bottom_labels, fontstyle="italic")
        axes[1].grid(axis="y", linestyle="--", alpha=0.8)
        if bottom_heights:
            bottom_max = max(bottom_heights)
            axes[1].set_ylim(0, bottom_max * 1.375)

        bs_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch_map.get(bs, "///"))
            for bs in bs_levels
        ]
        bs_labels = [f"bs={bs}" for bs in bs_levels]
        bs_legend = axes[1].legend(bs_handles, bs_labels, loc="upper right")
        if len(bottom_methods) > 1:
            axes[1].add_artist(bs_legend)
            method_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=bottom_method_colors[method], edgecolor="black")
                for method in bottom_methods
            ]
            axes[1].legend(method_handles, bottom_methods, loc="upper left")

        pad = max(top_bar_width, bottom_bar_width) * 0.7
        if top_x:
            axes[0].set_xlim(min(top_x) - pad, max(top_x) + pad)
            axes[0].margins(x=0)
        if bottom_x:
            axes[1].set_xlim(min(bottom_x) - pad, max(bottom_x) + pad)
            axes[1].margins(x=0)

        fig.suptitle("Inference Validation (A100 80 GB)", y=0.975)
        fig.supylabel("Total runtime estimation error (%)")
        fig.supxlabel("(Input tokens/Output tokens) and TP degree")
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.09, right=0.98)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "inf_errors_bar_combined_a100.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_combined_ratio_grids(
    imec_rows: List[Dict[str, object]],
    nvidia_rows: List[Dict[str, object]],
    outdir: Path,
    llmcompass_rows: Optional[List[Dict[str, object]]] = None,
    llmcompass_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    vidur_rows: Optional[List[Dict[str, object]]] = None,
    vidur_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    genz_nvidia_rows: Optional[List[Dict[str, object]]] = None,
    genz_rows: Optional[List[Dict[str, object]]] = None,
    flattened_rows: Optional[List[Dict[str, object]]] = None,
    flattened_nvidia_rows: Optional[List[Dict[str, object]]] = None,
) -> Optional[Path]:
    imec_rows = [row for row in imec_rows if row.get("device") == "A100"]
    nvidia_rows = [row for row in nvidia_rows if row.get("device") == "A100"]
    if vidur_rows:
        vidur_rows = [row for row in vidur_rows if row.get("device") == "A100"]
    if vidur_nvidia_rows:
        vidur_nvidia_rows = [row for row in vidur_nvidia_rows if row.get("device") == "A100"]
    if genz_nvidia_rows:
        genz_nvidia_rows = [row for row in genz_nvidia_rows if row.get("device") == "A100"]
    if genz_rows:
        genz_rows = [row for row in genz_rows if row.get("device") == "A100"]
    if flattened_rows:
        flattened_rows = [row for row in flattened_rows if row.get("device") == "A100"]
    if flattened_nvidia_rows:
        flattened_nvidia_rows = [
            row for row in flattened_nvidia_rows if row.get("device") == "A100"
        ]
    if not imec_rows or not nvidia_rows:
        return None

    model_order = {"Llama 2-7B": 0, "Llama 2-13B": 1, "Llama 2-70B": 2}
    models = sorted(
        {row.get("model") for row in imec_rows if row.get("model")},
        key=lambda name: (model_order.get(name, 99), str(name)),
    )
    tps = sorted({int(row.get("tp")) for row in imec_rows if row.get("tp") is not None})
    token_pairs = sorted(
        {(
            int(row.get("input_tokens")),
            int(row.get("output_tokens")),
        ) for row in nvidia_rows if row.get("input_tokens") is not None and row.get("output_tokens") is not None}
    )
    concurrencies = sorted(
        {int(row.get("concurrency")) for row in nvidia_rows if row.get("concurrency") is not None}
    )
    if not models or not tps or not token_pairs or not concurrencies:
        return None

    imec_actual: Dict[Tuple[object, object], float] = {}
    imec_rapid: Dict[Tuple[object, object], float] = {}
    for row in imec_rows:
        key = (row.get("model"), row.get("tp"))
        actual = row.get("actual_inference_time_s")
        imec_actual[key] = float(actual) if actual is not None else float("nan")
        imec_rapid[key] = _safe_ratio(row.get("inference_time_s"), actual)

    nvidia_actual: Dict[Tuple[object, object, object], float] = {}
    nvidia_rapid: Dict[Tuple[object, object, object], float] = {}
    for row in nvidia_rows:
        key = (
            row.get("input_tokens"),
            row.get("output_tokens"),
            row.get("concurrency"),
        )
        actual = row.get("actual_inference_time_s")
        nvidia_actual[key] = float(actual) if actual is not None else float("nan")
        nvidia_rapid[key] = _safe_ratio(row.get("inference_time_s"), actual)

    tool_order = [RAPID_HIER_LABEL, RAPID_FLAT_LABEL, "GenZ", "Vidur", "LLMCompass"]
    imec_series: Dict[str, Dict[Tuple[object, object], float]] = {RAPID_HIER_LABEL: imec_rapid}
    nvidia_series: Dict[str, Dict[Tuple[object, object, object], float]] = {RAPID_HIER_LABEL: nvidia_rapid}

    flattened_rows = flattened_rows or []
    if flattened_rows:
        flattened_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("inference_time_s", float("nan")))
            for row in flattened_rows
        }
        imec_series[RAPID_FLAT_LABEL] = {
            key: _safe_ratio(pred, imec_actual.get(key))
            for key, pred in flattened_lookup.items()
        }

    flattened_nvidia_rows = flattened_nvidia_rows or []
    if flattened_nvidia_rows:
        flattened_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
            ): float(row.get("inference_time_s", float("nan")))
            for row in flattened_nvidia_rows
        }
        nvidia_series[RAPID_FLAT_LABEL] = {
            key: _safe_ratio(pred, nvidia_actual.get(key))
            for key, pred in flattened_lookup.items()
        }

    llmcompass_rows = llmcompass_rows or []
    if llmcompass_rows:
        llmcompass_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in llmcompass_rows
        }
        imec_series["LLMCompass"] = {
            key: _safe_ratio(pred, imec_actual.get(key))
            for key, pred in llmcompass_lookup.items()
        }

    llmcompass_nvidia_rows = llmcompass_nvidia_rows or []
    if llmcompass_nvidia_rows:
        llmcompass_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
            ): float(row.get("total_latency", float("nan")))
            for row in llmcompass_nvidia_rows
        }
        nvidia_series["LLMCompass"] = {
            key: _safe_ratio(pred, nvidia_actual.get(key))
            for key, pred in llmcompass_lookup.items()
        }

    vidur_rows = vidur_rows or []
    if vidur_rows:
        vidur_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in vidur_rows
        }
        imec_series["Vidur"] = {
            key: _safe_ratio(pred, imec_actual.get(key))
            for key, pred in vidur_lookup.items()
        }

    vidur_nvidia_rows = vidur_nvidia_rows or []
    if vidur_nvidia_rows:
        vidur_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
            ): float(row.get("total_latency", float("nan")))
            for row in vidur_nvidia_rows
        }
        nvidia_series["Vidur"] = {
            key: _safe_ratio(pred, nvidia_actual.get(key))
            for key, pred in vidur_lookup.items()
        }

    genz_rows = genz_rows or []
    if genz_rows:
        genz_lookup = {
            (row.get("model"), row.get("tp")): float(row.get("total_latency", float("nan")))
            for row in genz_rows
        }
        imec_series["GenZ"] = {
            key: _safe_ratio(pred, imec_actual.get(key))
            for key, pred in genz_lookup.items()
        }

    genz_nvidia_rows = genz_nvidia_rows or []
    if genz_nvidia_rows:
        genz_lookup = {
            (
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("concurrency"),
            ): float(row.get("total_latency", float("nan")))
            for row in genz_nvidia_rows
        }
        nvidia_series["GenZ"] = {
            key: _safe_ratio(pred, nvidia_actual.get(key))
            for key, pred in genz_lookup.items()
        }

    tool_names = []
    for name in tool_order:
        imec_vals = imec_series.get(name, {})
        nvidia_vals = nvidia_series.get(name, {})
        has_vals = any(
            math.isfinite(v) for v in imec_vals.values()
        ) or any(math.isfinite(v) for v in nvidia_vals.values())
        if has_vals:
            tool_names.append(name)
    if not tool_names:
        return None

    tool_colors = {name: TOOL_COLORS.get(name, TOOL_COLORS[RAPID_HIER_LABEL]) for name in tool_names}
    imec_values = {name: imec_series.get(name, {}) for name in tool_names}
    nvidia_values = {name: nvidia_series.get(name, {}) for name in tool_names}

    top_h = 3.4
    bottom_rows = math.ceil(len(concurrencies) / min(3, len(concurrencies)))
    bottom_h = 3.0 * bottom_rows
    tool_scale = 0.8 + 0.2 * len(tool_names)
    subplot_w = 3.2 * tool_scale
    fig_w = max(8.0, subplot_w * max(len(models), min(3, len(concurrencies))))
    fig_h = 5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[top_h, bottom_h], hspace=0.7)

    top_gs = gs[0].subgridspec(1, len(models), wspace=0.25)
    bottom_ncols = min(3, max(1, len(concurrencies)))
    bottom_nrows = math.ceil(len(concurrencies) / bottom_ncols) if concurrencies else 1
    bottom_gs = gs[1].subgridspec(bottom_nrows, bottom_ncols, wspace=0.25, hspace=0.35)

    bar_width = 0.6 if len(tool_names) == 1 else (0.35 if len(tool_names) == 2 else 0.25)
    offsets = [(idx - (len(tool_names) - 1) / 2) * bar_width for idx in range(len(tool_names))]

    top_all_vals = [1.0]
    for name in tool_names:
        top_all_vals.extend(
            [v for v in imec_values.get(name, {}).values() if isinstance(v, (int, float)) and math.isfinite(v)]
        )
    top_span = max(max(top_all_vals) - min(top_all_vals), 0.1) if top_all_vals else 0.1
    top_pad = 0.05 * top_span
    top_limits = (min(top_all_vals) - top_pad, max(top_all_vals) + top_pad) if top_all_vals else None

    for col_idx, model in enumerate(models):
        ax = fig.add_subplot(top_gs[0, col_idx])
        display_model = MODEL_DISPLAY.get(str(model), str(model))
        has_any = False
        model_tps = [tp for tp in tps if not (str(model) == "Llama 2-70B" and tp == 1)]
        group_gap = 1.3
        x_positions = [idx * group_gap for idx in range(len(model_tps))]
        for tool_idx, name in enumerate(tool_names):
            heights = [imec_values.get(name, {}).get((model, tp), float("nan")) for tp in model_tps]
            if any(isinstance(v, (int, float)) and math.isfinite(v) for v in heights):
                has_any = True
            positions = [i + offsets[tool_idx] for i in x_positions]
            ax.bar(positions, heights, bar_width, color=tool_colors[name])
            if name == "Vidur":
                for pos, height in zip(positions, heights):
                    if not (isinstance(height, (int, float)) and math.isfinite(height)):
                        ax.bar(
                            pos,
                            1.0,
                            bar_width,
                            color="#c9c9c9",
                            alpha=0.4,
                            edgecolor="#8a8a8a",
                            hatch="//",
                        )
                        ax.text(
                            pos,
                            0.5,
                            "x",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="#555555",
                        )
        if not has_any:
            ax.axis("off")
            continue
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        if top_limits is not None:
            ax.set_ylim(*top_limits)
        ax.set_title(display_model)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"TP{tp}" for tp in model_tps], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel("Pred / Actual")

    bottom_all_vals = [1.0]
    for name in tool_names:
        bottom_all_vals.extend(
            [v for v in nvidia_values.get(name, {}).values() if isinstance(v, (int, float)) and math.isfinite(v)]
        )
    bottom_span = max(max(bottom_all_vals) - min(bottom_all_vals), 0.1) if bottom_all_vals else 0.1
    bottom_pad = 0.05 * bottom_span
    bottom_limits = (min(bottom_all_vals) - bottom_pad, max(bottom_all_vals) + bottom_pad) if bottom_all_vals else None

    group_gap = 1.3
    x = [idx * group_gap for idx in range(len(token_pairs))]
    x_labels = [f"{inp}/{out}" for inp, out in token_pairs]
    axes_list = []
    for idx, concurrency in enumerate(concurrencies):
        row_idx = idx // bottom_ncols
        col_idx = idx % bottom_ncols
        ax = fig.add_subplot(bottom_gs[row_idx, col_idx])
        axes_list.append(ax)
        for tool_idx, name in enumerate(tool_names):
            heights = [
                nvidia_values.get(name, {}).get((inp, out, concurrency), float("nan"))
                for inp, out in token_pairs
            ]
            ax.bar([i + offsets[tool_idx] for i in x], heights, bar_width, color=tool_colors[name])
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        if bottom_limits is not None:
            ax.set_ylim(*bottom_limits)
        ax.set_title(f"batch size = {concurrency}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel("Pred / Actual")

    for j in range(len(concurrencies), bottom_nrows * bottom_ncols):
        row_idx = j // bottom_ncols
        col_idx = j % bottom_ncols
        fig.add_subplot(bottom_gs[row_idx, col_idx]).axis("off")

    fig.suptitle("Normalized Inference Validation (A100 Systems)", y=0.995)
    handles = [plt.Rectangle((0, 0), 1, 1, color=tool_colors[name]) for name in tool_names]
    fig.legend(
        handles,
        tool_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(tool_names),
    )
    fig.text(0.5, 0.48, "Llama 3-70B", ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(bottom=0.2, top=0.85)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "inf_ratio_grid_combined_a100.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NVIDIA inference validation and comparison plots."
    )
    parser.add_argument(
        "--llmcompass",
        action="store_true",
        help="Include LLMCompass comparisons (default: on when no tool flags are provided).",
    )
    parser.add_argument(
        "--vidur",
        action="store_true",
        help="Include Vidur comparisons (default: on when no tool flags are provided).",
    )
    parser.add_argument(
        "--genz",
        action="store_true",
        help="Include GenZ comparisons (default: on when no tool flags are provided).",
    )
    parser.add_argument(
        "--plot",
        choices=("errors", "ratio", "both"),
        default="ratio",
        help="Choose which combined plot to generate (default: both).",
    )
    args = parser.parse_args()
    _apply_compare_args(args)

    imec_rows = None
    imec_flattened_rows = None
    nvidia_rows = None
    nvidia_flattened_rows = None
    llmcompass_rows = None
    llmcompass_nvidia_rows = None
    vidur_rows = None
    vidur_nvidia_rows = None
    genz_nvidia_rows = None
    genz_rows = None
    # for device_name in ("A100", "H100"):
    for device_name in ("A100",):
        try:
            # Only build combined plots below; skip per-device plots here.
            result = run(network_ignored=False, device=device_name)
            if device_name == "A100":
                imec_rows = result.get("rows")
                imec_flattened_rows = result.get("flattened_rows")
        except FileNotFoundError:
            pass
    # for device_name in ("A100", "H100"):
    for device_name in ("A100",):
        try:
            # Only build combined plots below; skip per-device plots here.
            result = run_nvidia(network_ignored=False, device=device_name)
            if device_name == "A100":
                nvidia_rows = result.get("rows")
                nvidia_flattened_rows = result.get("flattened_rows")
        except FileNotFoundError:
            pass
    if imec_rows and nvidia_rows:
        out_dir = Path(PROJECT_ROOT) / "output" / "validation" / "inf"
        if COMPARE_LLMCOMPASS:
            llmcompass_rows = _load_llmcompass_imec_errors(LLMCOMPASS_IMEC, device="A100")
            llmcompass_nvidia_rows = _load_llmcompass_nvidia_errors(LLMCOMPASS_NVIDIA, device="A100")
        if COMPARE_VIDUR:
            vidur_rows = _load_vidur_imec_errors(VIDUR_IMEC)
            vidur_nvidia_rows = _load_vidur_nvidia_errors(VIDUR_NVIDIA, device="A100")
        if COMPARE_GENZ:
            genz_nvidia_rows = _load_genz_nvidia_errors(GENZ_NVIDIA, device="A100")
            genz_rows = _load_genz_imec_errors(GENZ_IMEC)
        if args.plot in ("errors", "both"):
            combined = _plot_combined_a100_error_bars(
                imec_rows,
                nvidia_rows,
                out_dir,
                llmcompass_rows=llmcompass_rows,
                llmcompass_nvidia_rows=llmcompass_nvidia_rows,
                vidur_rows=vidur_rows,
                vidur_nvidia_rows=vidur_nvidia_rows,
                genz_nvidia_rows=genz_nvidia_rows,
                genz_rows=genz_rows,
                flattened_rows=imec_flattened_rows,
                flattened_nvidia_rows=nvidia_flattened_rows,
            )
            if combined is not None:
                print(f"Saved: {combined}")
        if args.plot in ("ratio", "both"):
            combined_ratio = _plot_combined_ratio_grids(
                imec_rows,
                nvidia_rows,
                out_dir,
                llmcompass_rows=llmcompass_rows,
                llmcompass_nvidia_rows=llmcompass_nvidia_rows,
                vidur_rows=vidur_rows,
                vidur_nvidia_rows=vidur_nvidia_rows,
                genz_nvidia_rows=genz_nvidia_rows,
                genz_rows=genz_rows,
                flattened_rows=imec_flattened_rows,
                flattened_nvidia_rows=nvidia_flattened_rows,
            )
            if combined_ratio is not None:
                print(f"Saved: {combined_ratio}")
