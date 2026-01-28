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

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import matplotlib.pyplot as plt
import pandas as pd


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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config")
MODEL_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "model-config")
DATA_DIR = Path(__file__).parent / "imec_data"

HW_CONFIGS = {
  "A100": "a100_80GB.yaml",
  "A100_korthi": "a100_80GB_korthikanti.yaml",
  "A100_selene": "a100_80GB_selene_sc.yaml",
}

MODEL_CONFIGS = {
  "Llama 2-7B": "Llama2-7B_inf.yaml",
  "GPT 22B": "GPT_22_B.yaml",
  "GPT 175B": "GPT_175_B.yaml",
  "GPT 310B": "GPT_310_B.yaml",
  "GPT 530B": "GPT_530_B.yaml",
  "GPT 1T": "GPT_1T.yaml",
}


def _load_data(csv_path: Path):
  # Load training validation CSVs; treat leading '//' as comments for path hints.
  df = pd.read_csv(csv_path, comment="/")
  if "mb" not in df.columns:
    raise ValueError(f"Training validation CSV missing required 'mb' column: {csv_path}")
  for col in ("batch", "mb", "dp", "tp", "pp", "cp"):
    if col in df.columns:
      df[col] = df[col].astype(int)
  if "tp_sp" in df.columns:
    df["tp_sp"] = df["tp_sp"].astype(bool)
  if "recomputation" in df.columns:
    df["recomputation"] = df["recomputation"].astype(str)
  return df


@lru_cache(maxsize=None)
def _load_device_data(device: str):
  candidates = [DATA_DIR / f"{device}_train.csv"]
  base = device.split("_")[0]
  if base and base != device:
    candidates.append(DATA_DIR / f"{base}_train.csv")
  candidates.append(DATA_DIR / "A100_train.csv")

  for path in candidates:
    if path.exists():
      df = _load_data(path)
      if "device" in df.columns:
        sub = df[df["device"] == device]
        if not sub.empty:
          return sub
        else:
          continue
      return df
  raise FileNotFoundError(f"Validation CSV not found for device {device}: tried {', '.join(str(p) for p in candidates)}")


def _iter_tests(df):
  for _, row in df.iterrows():
    yield (
      row["device"],
      row["model"],
      row["batch"],
      row["mb"],
      row["dp"],
      row["tp"],
      row["pp"],
      row["cp"],
      row.get("tp_sp", False),
      row.get("recomputation", "partial"),
    )


def _build_spec(
  device: str,
  model: str,
  batch: int,
  mb: int,
  dp: int,
  tp: int,
  pp: int,
  cp: int,
  tp_sp: bool,
  recomputation: str,
  idx: int,
) -> Tuple[ValidationSpec, str, str]:
  label = f"{device} {model} bs={batch} mb={mb} dp={dp} tp={tp} pp={pp} cp={cp} tp_sp={tp_sp}"

  recomputation_mode = str(recomputation).strip().lower()
  full_recompute = recomputation_mode in {"full", "true", "yes", "on", "1"}

  model_overrides = {
    "model_param": {
      "global_batch_size": int(batch),
    }
  }

  hw_overrides = {
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
      # sw_param.full_recomputation toggles full activation recompute during backward.
      "full_recomputation": full_recompute,
    },
  }

  spec = ValidationSpec(
    label=label,
    model_overrides=model_overrides,
    hardware_overrides=hw_overrides,
    model_config_path=os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model)),
    hardware_config_path=os.path.join(HARDWARE_CONFIG_PATH, HW_CONFIGS.get(device)),
    metadata={
      "device": device,
      "model": model,
      "batch": int(batch),
      "mb": int(mb),
      "dp": int(dp),
      "tp": int(tp),
      "pp": int(pp),
      "cp": int(cp),
      "tp_sp": bool(tp_sp),
      "recomputation": str(recomputation),
    },
    order=idx,
  )
  return spec, spec.model_config_path, spec.hardware_config_path  # type: ignore[return-value]


def _lookup_actual(device: str, model: str, batch: int, mb: int, dp: int, tp: int, pp: int, cp: int, tp_sp: bool, recomputation: str) -> float:
  data = _load_device_data(device)
  matches = data.loc[
    (data["device"] == device)
    & (data["model"] == model)
    & (data["batch"] == int(batch))
    & (data["mb"] == int(mb))
    & (data["dp"] == int(dp))
    & (data["tp"] == int(tp))
    & (data["pp"] == int(pp))
    & (data["cp"] == int(cp))
    & (data.get("tp_sp", False) == bool(tp_sp))
    & (data.get("recomputation", "").astype(str) == str(recomputation)),
    "actual",
  ]
  if matches.empty:
    raise ValueError(
      f"No reference training time for {device} {model} bs={batch} mb={mb} dp={dp} tp={tp} pp={pp} cp={cp} tp_sp={tp_sp} recomputation={recomputation}"
    )
  return float(matches.values[0])


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int, int, int, int, int, int, bool, str], float]):
  rows: List[Dict[str, object]] = []
  for res in results:
    meta = res.spec.metadata or {}
    model = meta.get("model")
    batch = int(meta.get("batch")) if "batch" in meta else None
    mb = int(meta.get("mb")) if "mb" in meta else None
    dp = int(meta.get("dp")) if "dp" in meta else None
    tp = int(meta.get("tp")) if "tp" in meta else None
    pp = int(meta.get("pp")) if "pp" in meta else None
    cp = int(meta.get("cp")) if "cp" in meta else None
    tp_sp = bool(meta.get("tp_sp")) if "tp_sp" in meta else False
    recomputation = str(meta.get("recomputation")) if "recomputation" in meta else "partial"
    key = (model, batch, mb, dp, tp, pp, cp, tp_sp, recomputation)
    train_time = float(res.metrics.get("training_time_s", float("nan"))) if res.success else float("nan")
    actual = actual_lookup.get(key, float("nan"))
    if math.isnan(train_time) or actual == 0 or math.isnan(actual):
      signed_pct_error = float("nan")
      pct_error = float("nan")
    else:
      signed_pct_error = (train_time - actual) / actual * 100.0
      pct_error = abs(signed_pct_error)
    rows.append(
      {
        "device": meta.get("device"),
        "model": model,
        "batch": batch,
        "mb": mb,
        "dp": dp,
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "tp_sp": tp_sp,
        "recomputation": recomputation,
        "training_time_s": train_time,
        "actual_inference_time_s": actual,
        "signed_pct_error": signed_pct_error,
        "pct_error": pct_error,
        "success": res.success,
        "error": res.error,
        "raw_output": res.raw_output,
      }
    )
  return rows


def build_specs_for_device(
  device: str,
  *,
  models: Optional[Iterable[str]] = None,
  tp_sp_only: Optional[bool] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int, int, int, bool, str], float], str, str]:
  data = _load_device_data(device)
  model_filter = set(models) if models is not None else None
  specs: List[ValidationSpec] = []
  actual_lookup: Dict[Tuple[str, int, int, int, int, int, int, bool, str], float] = {}
  base_model_path: Optional[str] = None
  hw_config_path: Optional[str] = None
  idx = 0
  for device_val, model, batch, mb, dp, tp, pp, cp, tp_sp, recomputation in _iter_tests(data):
    if model_filter and model not in model_filter:
      continue
    if tp_sp_only is not None and bool(tp_sp) != bool(tp_sp_only):
      continue
    spec, model_path, hw_path = _build_spec(
      device_val, model, batch, mb, dp, tp, pp, cp, tp_sp, recomputation, idx
    )
    specs.append(spec)
    actual_lookup[(model, int(batch), int(mb), int(dp), int(tp), int(pp), int(cp), bool(tp_sp), str(recomputation))] = _lookup_actual(
      device_val, model, batch, mb, dp, tp, pp, cp, tp_sp, recomputation
    )
    base_model_path = base_model_path or model_path
    hw_config_path = hw_config_path or hw_path
    idx += 1
  if not specs:
    raise ValueError(f"No validation specs generated for device={device} (models={models}).")
  return specs, actual_lookup, base_model_path, hw_config_path


def run(
  device: str = "A100",
  models: Optional[Sequence[str]] = None,
  tp_sp: Optional[bool] = None,
  enable_plot: bool = True,
  show_progress: bool = False,
  emit_logs: bool = True,
):
  specs, actual_lookup, base_model_path, hw_config_path = build_specs_for_device(
    device, models=models, tp_sp_only=tp_sp
  )

  validation_results = run_validation_suite(
    specs,
    base_model_config_path=base_model_path,
    base_hardware_config_path=hw_config_path,
    result_parser=parse_training_time,
    run_perf_path=RUN_PERF,
    show_progress=show_progress,
  )

  rows = compute_pct_errors(validation_results, actual_lookup)
  pct_errors = [r["pct_error"] for r in rows if not math.isnan(r["pct_error"])]

  if emit_logs:
    for row in rows:
      block = [
        f"\n=== Result (device={row['device']}, model={row['model']}, bs={row['batch']}, mb={row['mb']}, dp={row['dp']}, tp={row['tp']}, pp={row['pp']}, cp={row['cp']}, tp_sp={row['tp_sp']}, recomputation={row['recomputation']}) ==="
      ]
      if row["success"] and not math.isnan(row["pct_error"]):
        block.append(f"  RAPID-LLM train time: {float(row['training_time_s']):.2f}s")
        block.append(f"  Actual train time:   {float(row['actual_inference_time_s']):.2f}s")
        block.append(f"  Percent Error:  {float(row['signed_pct_error']):+.2f}%")
      else:
        block.append(f"  RAPID-LLM run failed. {(row.get('error') or '')}".rstrip())
        if row.get("raw_output"):
          block.append("  --- Raw output ---")
          block.append(str(row["raw_output"]).strip())
      print("\n".join(block))

  plot_path = None
  if enable_plot and rows:
    out_dir = Path(PROJECT_ROOT) / "output" / "validation" / "train"
    plot_path = _plot_training_rows(
      rows,
      title=f"Training validation ({device})",
      path=out_dir / f"train_{device}.png",
      include_device=False,
    )

  avg_abs_error = sum(pct_errors) / len(pct_errors) if pct_errors else float("nan")
  if emit_logs:
    print("Average absolute percent error across all training tests: {:.2f}%".format(avg_abs_error))
    if plot_path:
      print(f"Saved plot: {plot_path}")
  return {"avg_abs_error": avg_abs_error, "rows": rows, "plot": plot_path}


def _plot_training_rows(rows, title: str, path: Path, include_device: bool = True) -> Path:
  out_dir = path.parent
  out_dir.mkdir(parents=True, exist_ok=True)
  labels: List[str] = []
  errors: List[float] = []
  colors: List[str] = []
  color_map = {"full": "#ff8c00", "partial": "#1f77b4", "selective": "#1f77b4"}
  for row in rows:
    if str(row.get("model")) == "GPT 22B":
      continue
    gpus = int(row["dp"]) * int(row["tp"]) * int(row["pp"]) * int(row["cp"])
    prefix = ""
    labels.append(f"{prefix}{row['model']} x{gpus}")
    errors.append(row["signed_pct_error"])
    colors.append(color_map.get(str(row.get("recomputation")).lower(), "#1f77b4"))

  fig_w = max(6.0, 0.6 * len(labels))
  fig, ax = plt.subplots(figsize=(fig_w, 4))
  bars = ax.bar(range(len(errors)), errors, color=colors)
  ax.set_xticks(range(len(labels)))
  ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
  ax.set_ylabel("Percent Error")
  ax.set_title(title)
  handles = [
    plt.Rectangle((0, 0), 1, 1, color="#ff8c00"),
    plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
  ]
  ax.legend(handles, ["recompute=full", "recompute=partial/selective"])
  for rect, err in zip(bars, errors):
    if math.isnan(err):
      continue
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
  ax.grid(axis="y", linestyle="--", alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=200)
  plt.close(fig)
  return path


if __name__ == "__main__":
  devices = [
    "A100_korthi",  # Uncomment to include korthi runs.
    "A100_selene",
  ]
  combined_rows: List[Dict[str, object]] = []
  for device in devices:
    print(f"=== Running {device} training validation ===")
    result = run(device=device, emit_logs=True, show_progress=True)
    combined_rows.extend(result.get("rows", []))

  if combined_rows:
    combined_path = _plot_training_rows(
      combined_rows,
      title="Training validation (combined)",
      path=Path(PROJECT_ROOT) / "output" / "validation" / "train" / "train_combined.png",
      include_device=False,
    )
    print(f"Saved combined plot: {combined_path}")
