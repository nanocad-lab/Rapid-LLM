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

import os
import math
from typing import Dict, Tuple, List, Any
import seaborn as sns

import matplotlib.pyplot as plt

from .validation_helpers import (
  ValidationSpec,
  parse_decode_time,
  run_validation_suite,
)


# Llama-3.1-8B Instruct - A100 SXM (from Koyeb matrix, using actual Input/Output tokens)
# https://www.koyeb.com/docs/hardware/gpu-benchmarks
#
# Keys below are NORMALIZED to (prefill_len, decode_len, batch_size):
#   prefill_len = (input_tokens_agg / batch_size) rounded to nearest int
#   decode_len  = (decode_tokens_agg / batch_size) rounded to nearest int
# Values are tuples: (total_decode_time_seconds, expected_tokens_per_second_aggregate)
#
# Groups:
# - small  = "token shape 512x512"
# - medium = "token shape 1024x1024"
# - large  = "token shape 4096x1024"
# 
# We compare Deepflow's average token/s (calculated from decode time) with Koyeb's expected tokens/s.


# We need to convert the Koyeb data to the same format as the Deepflow data.
# Koyeb data is input_tokens_agg, decode_tokens_agg so we divided by batch size.
KOYEB_DATA: Dict[str, Dict[Tuple[int, int, int], Tuple[float, float]]] = {
  "small": {
    # From: (1484, 512, 1) -> (1484, 512, 1)
    (1484, 512, 1): (6.50, 78.77),
    # From: (11896, 4096, 8) -> (1487, 512, 8)
    (1487, 512, 8): (6.79, 603.29),
    # From: (48416, 16384, 32) -> (1513, 512, 32)
    (1513, 512, 32): (8.48, 1932.73),
  },
  "medium": {
    # From: (2948, 467, 1) -> (2948, 467, 1)
    (2948, 467, 1): (5.84, 79.99),
    # From: (23504, 3934, 8) -> (2938, 492, 8)
    (2938, 492, 8): (8.53, 461.46),
    # From: (94176, 23798, 32) -> (2943, 744, 32)
    (2943, 744, 32): (17.22, 1382.08),
  },
  "large": {
    # From: (11525, 449, 1) -> (11525, 449, 1)
    (11525, 449, 1): (5.97, 75.26),
    # From: (92608, 5923, 8) -> (11576, 740, 8)
    (11576, 740, 8): (13.67, 433.16),

    # The following data point is extremely noisy on the koyeb website.
    # Koyeb reports that A100 PCIE is 2x faster than A100 SXM?? Also, output tokens are very low.
    # So we're using the PCIE data instead for this one only. 
    # If you want to use the SXM data, comment/uncomment the following lines:

    ## SXM DATA FOR BATCH SIZE 32, "LARGE" (NOISY) ##
    # # From: (370592, 640, 32) -> (11581, 20, 32)
    # (11581, 20, 32): (1.34, 477.82),
    ## PCIE DATA FOR BATCH SIZE 32, "LARGE" (CLEANER)""
    # From: (369184, 21075, 32) --> (11537, 659, 32)
    (11537, 659, 32): (25.06, 840.88),
  },
}
sns.set()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config/a100_80GB_no_parallelism.yaml")
BASE_MODEL_CONFIG = os.path.join(VALIDATION_CONFIG_ROOT, "model-config/Llama3.1-8B_inf.yaml")


def _collect_points_in_order(data: Dict[str, Dict[Tuple[int, int, int], Tuple[float, float]]]) -> List[Tuple[str, Tuple[int, int, int], Tuple[float, float]]]:
  ordered = []
  for size in ["small", "medium", "large"]:
    if size not in data:
      continue
    # sort by batch size ascending, then by prefill_len
    keys = sorted(list(data[size].keys()), key=lambda k: (k[2], k[0], k[1]))
    for k in keys:
      ordered.append((size, k, data[size][k]))
  return ordered


def run_all_and_plot(*, enable_plot: bool = True, verbose: bool = True) -> Dict[str, Any]:
  experiments = _collect_points_in_order(KOYEB_DATA)
  if not experiments:
    if verbose:
      print("No experiments to run.")
    return {
      "labels": [],
      "expected_tokps": [],
      "predicted_tokps": [],
      "percent_errors": [],
      "avg_error": float("nan"),
      "avg_abs_error": float("nan"),
      "plot_path": None,
    }

  log = print if verbose else (lambda *args, **kwargs: None)
  labels: List[str] = []
  expected_tokps: List[float] = []
  predicted_tokps: List[float] = []
  percent_errors: List[float] = []
  specs: List[ValidationSpec] = []

  for idx, (size, (prefill_len, decode_len, batch_size), (expected_decode_time_s, expected_tokps_agg)) in enumerate(experiments):
    label = f"{size} bs={batch_size}"
    metadata = {
      "size": size,
      "prefill_len": int(prefill_len),
      "decode_len": int(decode_len),
      "batch_size": int(batch_size),
      "expected_tokps": float(expected_tokps_agg),
      "expected_decode_time_s": float(expected_decode_time_s),
    }
    model_overrides = {
      "model_param": {
        "seq_len": int(prefill_len + decode_len),
        "decode_len": int(decode_len),
        "global_batch_size": int(batch_size),
      },
    }
    specs.append(
      ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=None,
        metadata=metadata,
        order=idx,
      )
    )

  validation_results = run_validation_suite(
    specs,
    base_model_config_path=BASE_MODEL_CONFIG,
    base_hardware_config_path=HARDWARE_CONFIG,
    result_parser=parse_decode_time,
    run_perf_path=RUN_PERF,
  )

  for result in validation_results:
    meta = result.spec.metadata
    label = result.spec.label
    prefill_len = int(meta["prefill_len"])
    decode_len = int(meta["decode_len"])
    batch_size = int(meta["batch_size"])
    expected_tokps_agg = float(meta["expected_tokps"])
    labels.append(label)
    expected_tokps.append(expected_tokps_agg)

    if result.success:
      decode_time_s = float(result.metrics.get("decode_time_s", float("nan")))
    else:
      decode_time_s = float("nan")

    total_tokens_agg = int(decode_len) * int(batch_size)
    if decode_time_s and decode_time_s > 0:
      predicted_tokps_agg = float(total_tokens_agg) / float(decode_time_s)
    else:
      predicted_tokps_agg = float("nan")

    if expected_tokps_agg > 0 and not math.isnan(predicted_tokps_agg):
      pct_error = (predicted_tokps_agg - expected_tokps_agg) / expected_tokps_agg * 100.0
    else:
      pct_error = float("nan")

    predicted_tokps.append(predicted_tokps_agg)
    percent_errors.append(pct_error)

    block_lines = [
      f"\n=== Result {label} (prefill={prefill_len}, decode={decode_len}, batch={batch_size}) ===",
    ]
    if result.success and not math.isnan(predicted_tokps_agg):
      block_lines.append(f"Predicted agg tok/s = {predicted_tokps_agg:.2f} (decode_time={decode_time_s:.2f}s)")
      block_lines.append(f"Expected agg tok/s  = {expected_tokps_agg:.2f}")
      block_lines.append(f"Error = {pct_error:+.2f}%")
    else:
      err_detail = result.error or "RAPID-LLM run failed"
      block_lines.append(f"ERROR: {err_detail}")
      if result.raw_output:
        block_lines.append(result.raw_output.strip())
    log("\n".join(block_lines))

  # Print summary statistics
  log("\n" + "="*70)
  log("SUMMARY")
  log("="*70)
  valid_errors = [e for e in percent_errors if not math.isnan(e)]
  if valid_errors:
    avg_error = sum(valid_errors) / len(valid_errors)
    avg_abs_error = sum(abs(e) for e in valid_errors) / len(valid_errors)
    log(f"Average error:          {avg_error:+.2f}%")
    log(f"Average absolute error: {avg_abs_error:.2f}%")
  else:
    avg_error = float("nan")
    avg_abs_error = float("nan")
    log("No valid error measurements.")
  log("="*70)

  plot_path = None
  if enable_plot:
    # Plot
    fig = plt.figure(figsize=(12, 6))
    x = list(range(len(labels)))
    plt.plot(x, predicted_tokps, marker="o", linestyle="-", color="#1f77b4", label="RAPID-LLM predicted tok/s (aggregate)")
    plt.plot(x, expected_tokps, marker="o", linestyle="--", color="#ff7f0e", alpha=0.7, label="Expected tok/s (Koyeb)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Tokens per second (aggregate)")
    plt.title("Koyeb vs RAPID-LLM (single A100 SXM, Llama3.1-8B)")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()

    # Add note about large bs=32 datapoint
    note_text = ( "X-axis is prefill length + batch size following Koyebs format (small = 512x512, medium = 1024x1024, large = 4096x1024)\n"
                  "For large bs=32 datapoint, we use PCIE data point, as SXM data is very noisy \n"
                 "on the official website (it's 2x slower than PCIE). You can change this in the code.")
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic', color='#555555', wrap=True)

    out_dir = os.path.join(PROJECT_ROOT, "output", "validation")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "koyeb_a100_sxm_no_parallelism.png")
    plt.tight_layout(rect=[0, 0.06, 1, 1])  # Leave space at bottom for note
    plt.savefig(plot_path, dpi=160)
    log(f"Saved plot to: {plot_path}")

  return {
    "labels": labels,
    "expected_tokps": expected_tokps,
    "predicted_tokps": predicted_tokps,
    "percent_errors": percent_errors,
    "avg_error": avg_error,
    "avg_abs_error": avg_abs_error,
    "plot_path": plot_path,
  }


def _parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="Validate RAPID-LLM against Koyeb benchmarks.")
  parser.add_argument("--no-plot", action="store_true", help="Skip generating the matplotlib plot/output file.")
  parser.add_argument("--quiet", action="store_true", help="Suppress detailed per-experiment logs.")
  return parser.parse_args()


if __name__ == "__main__":
  args = _parse_args()
  results = run_all_and_plot(enable_plot=not args.no_plot, verbose=not args.quiet)
  if args.quiet:
    avg_abs = results["avg_abs_error"]
    if math.isnan(avg_abs):
      print("Average absolute error: NaN (no valid measurements)")
    else:
      print(f"Average absolute error: {avg_abs:.2f}%")
