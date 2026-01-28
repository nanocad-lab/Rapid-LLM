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

import pytest

from validation_scripts import nvidia_train

DEFAULT_THRESHOLD = 20.0
# Optional overrides per (device, tp_sp_enabled)
GROUP_THRESHOLDS = {
  # ("A100_korthi", True): 10.0,
}


def _run_and_check(device: str, tp_sp: bool, record_validation):
  result = nvidia_train.run(
    device=device,
    tp_sp=tp_sp,
    enable_plot=False,
    emit_logs=False,
  )
  rows = result["rows"]
  assert rows, f"{device} tp_sp={tp_sp} produced no validation rows"
  pct_errors = []
  for row in rows:
    pct = row["pct_error"]
    assert not math.isnan(pct), f"{device} tp_sp={tp_sp} produced NaN pct_error for {row['model']}"
    pct_errors.append(pct)
  avg_error = sum(pct_errors) / len(pct_errors)
  threshold = GROUP_THRESHOLDS.get((device, tp_sp), DEFAULT_THRESHOLD)
  record_validation(f"{device} tp_sp={tp_sp}", avg_error, expected_pct=threshold)
  assert avg_error <= threshold, f"{device} tp_sp={tp_sp} avg error {avg_error:.2f}% exceeds {threshold}%"


# def test_train_a100_korthi_tp_sp_disabled(record_validation):
#   _run_and_check("A100_korthi", False, record_validation)


# def test_train_a100_korthi_tp_sp_enabled(record_validation):
#   _run_and_check("A100_korthi", True, record_validation)


# def test_train_a100_selene_tp_sp_disabled(record_validation):
#   _run_and_check("A100_selene", False, record_validation)
