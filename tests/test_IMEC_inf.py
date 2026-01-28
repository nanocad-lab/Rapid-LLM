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
from pathlib import Path

import pandas as pd
import pytest

from validation_scripts import nvidia_inf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "validation_scripts/imec_data"


DEFAULT_THRESHOLD = 10.0
# Override per (device, model) average pct error.
# Error is actual + 0.5% margin of error, rounded up to nearest 0.5%.
MODEL_THRESHOLDS = {
  ("A100", "Llama 2-7B"): 5.5,   # 4.62 + 0.5 -> ceil to 5.5
  ("A100", "Llama 2-13B"): 3.0,  # 2.44 + 0.5 -> ceil to 3.0
  ("A100", "Llama 2-70B"): 13.0, # 12.09 + 0.5 -> ceil to 13.0
}
# Override per (device, TP) average pct error across all models at that TP.
TP_THRESHOLDS = {
  ("A100", 1): 4.5,  # 3.98 + 0.5 -> ceil to 4.5
  ("A100", 2): 8.5,  # 7.60 + 0.5 -> ceil to 8.5
  ("A100", 4): 5.5,  # 4.91 + 0.5 -> ceil to 5.5
  ("A100", 8): 7.0,  # 6.34 + 0.5 -> ceil to 7.0
}


def _load_params(by: str):
  params = []
  for device in ("A100", "H100"):
    csv_path = DATA_DIR / f"{device}_inf.csv"
    if not csv_path.exists():
      continue
    df = pd.read_csv(csv_path, comment="/")
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["device", "model", "TP", "actual"])
    if by == "model":
      grouped = df.groupby("model")
    else:
      grouped = df.groupby("TP")
    for key, group in grouped:
      marks = ()
      if device == "H100":
        marks = (pytest.mark.xfail(reason="Pending model refinements", strict=False),)
      params.append(
        pytest.param(
          device,
          key,
          marks=marks,
          id=f"{device}-{by}-{key}",
        )
      )
  return params


MODEL_PARAMS = _load_params("model")
TP_PARAMS = _load_params("TP")


@pytest.fixture(scope="module")
def cached_rows():
  cache = {}
  for device in ("A100", "H100"):
    try:
      result = nvidia_inf.run(
        enable_plot=False,
        network_ignored=False,
        device=device,
        emit_logs=False,
        test_model=True,
      )
    except FileNotFoundError:
      continue
    cache[device] = result["rows"]
  return cache


@pytest.mark.parametrize("device,model", MODEL_PARAMS)
def test_avg_abs_error_by_model(device, model, cached_rows, record_validation):
  label = f"{device} | {model}"
  threshold = MODEL_THRESHOLDS.get((device, model), DEFAULT_THRESHOLD)
  rows = [r for r in cached_rows.get(device, []) if r["model"] == model]
  assert rows, f"{label} produced no validation rows"

  pct_errors = []
  for row in rows:
    pct_error = row["pct_error"]
    assert not math.isnan(pct_error), f"{label}-TP{row['tp']} produced no valid measurements"
    pct_errors.append(pct_error)

  avg_error = sum(pct_errors) / len(pct_errors)
  val_label = f"{label} avg_abs_error"
  record_validation(val_label, avg_error, expected_pct=threshold)
  assert avg_error <= threshold, f"{label} avg error {avg_error:.2f}% exceeds {threshold}%"


@pytest.mark.parametrize("device,tp", TP_PARAMS)
def test_avg_abs_error_by_tp(device, tp, cached_rows, record_validation):
  label = f"{device} | TP{tp}"
  threshold = TP_THRESHOLDS.get((device, int(tp)), DEFAULT_THRESHOLD)
  rows = [r for r in cached_rows.get(device, []) if int(r["tp"]) == int(tp)]
  assert rows, f"{label} produced no validation rows"

  pct_errors = []
  for row in rows:
    pct_error = row["pct_error"]
    assert not math.isnan(pct_error), f"{label}-{row['model']} produced no valid measurements"
    pct_errors.append(pct_error)

  avg_error = sum(pct_errors) / len(pct_errors)
  val_label = f"{label} avg_abs_error"
  record_validation(val_label, avg_error, expected_pct=threshold)
  assert avg_error <= threshold, f"{label} avg error {avg_error:.2f}% exceeds {threshold}%"
