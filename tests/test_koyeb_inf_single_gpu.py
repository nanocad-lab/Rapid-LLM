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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import koyeb


def test_koyeb_average_absolute_error_below_threshold(record_validation):
  results = koyeb.run_all_and_plot(enable_plot=False, verbose=False)
  avg_abs_error = results["avg_abs_error"]
  record_validation("avg_abs_error", avg_abs_error, expected_pct=6.5)
  assert not math.isnan(avg_abs_error), "Koyeb validation produced no valid measurements"
  # Historically the average absolute error bottoms out at roughly 5.97%, so we allow
  # a little slack (6.5%) to absorb minor RAPID-LLM changes while still flagging regressions.
  assert avg_abs_error <= 6.5, f"Average absolute error {avg_abs_error:.2f}% exceeds 6.5%"
