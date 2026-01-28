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
Execute the LLM AstraSim debug harness under multiple faulty-link scenarios.

For each predefined configuration this script patches the
``configs/hardware-config/a100_80GB_faraz.yaml`` file with the desired
``network.faulty_links`` list, invokes ``./examples/llm_astra_inference_debug_graphviz.sh``,
parses the reported training time, and records the results in a summary table.

The original YAML contents are restored once the sweep finishes (even if a run fails).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware-config" / "a100_80GB_faraz.yaml"
RUN_SCRIPT = "./examples/llm_astra_inference_debug_graphviz.sh"

# Faulty-link sweep configurations.
# Each entry provides a human-readable label and the list of faulty links to inject.
# Faulty links follow the AstraSim convention: [src, dst, reliability_weight]
FAULT_SWEEP_CASES: Sequence[dict[str, Any]] = [
    {"label": "baseline", "faulty_links": []},
    {"label": "tp_top_left_harsh", "faulty_links": [[0, 1, 0.3]]},
    {"label": "tp_top_left", "faulty_links": [[0, 1, 0.5]]},
    {"label": "tp_sequential", "faulty_links": [[0, 1, 0.5], [1, 2, 0.5]]},
    {"label": "tp_top_left_vert", "faulty_links": [[0, 1, 0.5], [0, 4, 0.5]]},
    {"label": "tp_sequential_harsh", "faulty_links": [[0, 1, 0.3], [1, 2, 0.3]]},
    {"label": "tp_top_left_vert_harsh", "faulty_links": [[0, 1, 0.3], [0, 4, 0.3]]},
]


TRAINING_TIME_PATTERN = re.compile(r"Training time for batch:\s*([0-9.]+)s")


@dataclass
class SweepResult:
    label: str
    links: Sequence[Sequence[float]]
    returncode: int
    training_time: float | None
    stdout: str
    stderr: str


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _set_faulty_links(doc: Any, links: Sequence[Sequence[float]]) -> None:
    network_block = doc.get("network")
    if isinstance(network_block, dict):
        network_block["faulty_links"] = copy.deepcopy(list(links))
        return
    raise ValueError("Hardware config network section must be a mapping with 'faulty_links'.")


def _run_script(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [RUN_SCRIPT],
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
        shell=False,
    )


def _parse_training_time(output: str) -> float | None:
    match = TRAINING_TIME_PATTERN.search(output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_summary(results: Iterable[SweepResult]) -> str:
    lines: List[str] = []
    header = f"{'Label':<24} {'Training Time (s)':>18} {'Return Code':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        time_str = f"{result.training_time:.2f}" if result.training_time is not None else "n/a"
        lines.append(f"{result.label:<24} {time_str:>18} {result.returncode:>12}")
    return "\n".join(lines)


def run_sweep(cases: Sequence[dict[str, Any]], *, dry_run: bool = False) -> List[SweepResult]:
    original_yaml = _load_yaml(CONFIG_PATH)
    original_links = copy.deepcopy(original_yaml.get("network", {}).get("faulty_links"))
    results: List[SweepResult] = []

    try:
        for case in cases:
            label = case.get("label", "unnamed")
            links = case.get("faulty_links", [])
            print(f"[exec_with_fault] Running case '{label}' with links={links}")

            next_yaml = copy.deepcopy(original_yaml)
            _set_faulty_links(next_yaml, links)
            if not dry_run:
                _dump_yaml(CONFIG_PATH, next_yaml)

            completed: subprocess.CompletedProcess[str]
            if dry_run:
                completed = subprocess.CompletedProcess(
                    args=[RUN_SCRIPT], returncode=0, stdout="", stderr=""
                )
            else:
                completed = _run_script(PROJECT_ROOT)

            combined_output = f"{completed.stdout}\n{completed.stderr}"
            training_time = _parse_training_time(combined_output)
            if training_time is None:
                print(
                    f"[exec_with_fault] Warning: training time not found in output for case '{label}'.",
                    file=sys.stderr,
                )

            results.append(
                SweepResult(
                    label=label,
                    links=links,
                    returncode=completed.returncode,
                    training_time=training_time,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
            if completed.returncode != 0:
                print(
                    f"[exec_with_fault] Case '{label}' exited with return code {completed.returncode}.",
                    file=sys.stderr,
                )
    finally:
        # Restore original faulty links (or remove the key if it was missing)
        restored_yaml = _load_yaml(CONFIG_PATH)
        network_block = restored_yaml.get("network")
        if isinstance(network_block, dict):
            if original_links is None:
                network_block.pop("faulty_links", None)
            else:
                network_block["faulty_links"] = original_links
            if not dry_run:
                _dump_yaml(CONFIG_PATH, restored_yaml)

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute AstraSim runs with predefined faulty links.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not modify files or run the benchmark; useful for quick validation.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional subset of case labels to run. Defaults to the entire FAULT_SWEEP_CASES list.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print("[exec_with_fault] Dry run enabled: configurations will not be written or executed.")

    selected_cases = []
    if args.cases:
        case_set = set(args.cases)
        for case in FAULT_SWEEP_CASES:
            if case.get("label") in case_set:
                selected_cases.append(case)
        missing = case_set - {case.get("label") for case in selected_cases}
        if missing:
            parser.error(f"Unknown case label(s): {', '.join(sorted(missing))}")
    else:
        selected_cases = list(FAULT_SWEEP_CASES)

    if not selected_cases:
        parser.error("No cases selected for execution.")

    results = run_sweep(selected_cases, dry_run=args.dry_run)
    print("\nSweep summary:\n")
    print(_format_summary(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
