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
Meta sweep orchestrator for RAPID-LLM fault sensitivity experiments.

Runs a sequence of configurations by invoking ``fault_sweep.py`` with tailored
arguments, collects artefacts under ``tools/fault_sweep_results/``, and skips
any configuration whose plot already exists.
"""

import subprocess
import sys
from pathlib import Path
from typing import List


RESULTS_DIR = Path("tools", "fault_sweep_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SKIP_EXISTING_PLOTS = False


def run_config(name: str, extra_args: List[str]) -> None:
    plot_path = RESULTS_DIR / f"{name}.png"
    if SKIP_EXISTING_PLOTS and plot_path.exists():
        print(f"[meta] Skipping '{name}' (plot already exists).")
        return

    report_path = RESULTS_DIR / f"{name}.tsv"
    results_json = RESULTS_DIR / f"{name}_results.json"

    cmd = [
        sys.executable,
        "tools/fault_sweep.py",
        "--plot-output",
        str(plot_path),
        "--report-output",
        str(report_path),
        "--results-json",
        str(results_json),
    ]
    cmd.extend(extra_args)

    print(f"[meta] Running configuration '{name}'")
    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Configuration '{name}' failed with exit code {completed.returncode}")


def main() -> None:
    common_args = [
        "--target-num-gpus",
        "256",
        "--fault-iter",
        "200",
        "--fault-workers",
        "86",
        "--fault-mag",
        "0.5,0.1",
        "--num-faults",
        "1",
        "--min-tp",
        "1",
        "--max-tp",
        "16",
        "--min-cp",
        "1",
        "--max-cp",
        "1",
        "--min-dp",
        "1",
        "--max-dp",
        "256",
        "--min-pp",
        "1",
        "--max-pp",
        "128",
        "--seed",
        "1337",
    ]

    configs = [
        (
            "dense_1fault_alldim_16",
            [
                "--target-num-gpus",
                "16",
                "--fault-iter",
                "50",
                "--fault-workers",
                "70",
                "--fault-mag",
                "0.5,0.1",
                "--num-faults",
                "1",
                "--sample-count",
                "8",
                "--min-tp",
                "1",
                "--max-tp",
                "256",
                "--min-cp",
                "1",
                "--max-cp",
                "256",
                "--min-dp",
                "1",
                "--max-dp",
                "64",
                "--min-pp",
                "1",
                "--max-pp",
                "4",
                "--seed",
                "1337",
            ],
        ),
        # (
        #     "dense_1fault_alldim_256",
        #     common_args
        #     + [
        #         "--sample-count",
        #         "25",
        #         "--fault-iter",
        #         "50",
        #         "--fault-workers",
        #         "96",
        #     ],
        # ),
        # (
        #     "dense_1fault_dim0_256",
        #     common_args
        #     + [
        #         "--sample-count",
        #         "48",
        #         "--allowed-fault-dims",
        #         "0",
        #     ],
        # ),
        # (
        #     "sparse_1fault_alldim_tpcponly_256",
        #     [
        #         "--target-num-gpus",
        #         "256",
        #         "--fault-iter",
        #         "100",
        #         "--fault-workers",
        #         "86",
        #         "--fault-mag",
        #         "0.5,0.1",
        #         "--num-faults",
        #         "1",
        #         "--sample-count",
        #         "10",
        #         "--min-tp",
        #         "1",
        #         "--max-tp",
        #         "64",
        #         "--min-cp",
        #         "1",
        #         "--max-cp",
        #         "128",
        #         "--min-dp",
        #         "1",
        #         "--max-dp",
        #         "1",
        #         "--min-pp",
        #         "1",
        #         "--max-pp",
        #         "1",
        #         "--seed",
        #         "1337",
        #     ],
        # ),
        # (
        #     "sparse_1fault_alldim_dplponly_256",
        #     [
        #         "--target-num-gpus",
        #         "256",
        #         "--fault-iter",
        #         "200",
        #         "--fault-workers",
        #         "86",
        #         "--fault-mag",
        #         "0.5,0.1",
        #         "--num-faults",
        #         "1",
        #         "--sample-count",
        #         "20",
        #         "--min-tp",
        #         "1",
        #         "--max-tp",
        #         "1",
        #         "--min-cp",
        #         "1",
        #         "--max-cp",
        #         "1",
        #         "--min-dp",
        #         "1",
        #         "--max-dp",
        #         "256",
        #         "--min-pp",
        #         "1",
        #         "--max-pp",
        #         "128",
        #         "--seed",
        #         "1337",
        #     ],
        # ),
        # (
        #     "dense_1fault_alldim_256_harsh",
        #     common_args
        #     + [
        #         "--sample-count",
        #         "48",
        #         "--fault-mag",
        #         "0.3,0.2",
        #     ],
        # ),
    ]

    for name, args in configs:
        run_config(name, args)


if __name__ == "__main__":
    main()
