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

import atexit
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Tuple

_LOG_MESSAGES: List[Tuple[Optional[str], str]] = []
_LOG_LOCK = threading.Lock()

_SECTION_ORDER = [
    ("network", "TOPOLOGY/NETWORK"),
    ("faults", "FAULTY LINKS"),
    ("results", "RESULTS"),
]

_REPO_ROOT = os.path.abspath(os.environ.get("RAPID_REPO_ROOT", os.getcwd()))


def log_message(message: str, category: Optional[str] = None) -> None:
    """Append ``message`` to the shared log queue."""
    if message is None:
        return
    text = str(message)
    if not text:
        return
    cat_normalized = str(category).strip().lower() if category else None
    with _LOG_LOCK:
        _LOG_MESSAGES.append((cat_normalized, text))


def extend_log(lines: Iterable[str], category: Optional[str] = None) -> None:
    """Append multiple lines to the log queue."""
    for line in lines:
        log_message(line, category=category)


def drain_log_messages() -> List[Tuple[Optional[str], str]]:
    """Return and clear all queued log messages (category, message)."""
    with _LOG_LOCK:
        if not _LOG_MESSAGES:
            return []
        drained = list(_LOG_MESSAGES)
        _LOG_MESSAGES.clear()
        return drained


def flush_log_queue() -> None:
    """Print and clear the queued log messages grouped by category."""
    entries = drain_log_messages()
    print("\n")
    if not entries:
        return
    categorized = {cat: [] for cat, _ in _SECTION_ORDER}
    uncategorized: List[str] = []
    for category, message in entries:
        if category in categorized:
            categorized[category].append(message)
        else:
            uncategorized.append(message)

    sections_printed = False
    section_border = "=" * 60
    for cat, title in _SECTION_ORDER:
        lines = categorized.get(cat) or []
        if not lines:
            continue
        sections_printed = True
        print(f"{section_border}\n{title}\n{section_border}")
        for line in lines:
            print(line)
    if sections_printed:
        print(section_border)

    for line in uncategorized:
        print(line)


def relpath_display(path: str) -> str:
    """Return ``path`` relative to the repo root when possible."""
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, start=_REPO_ROOT)
    except Exception:
        return abs_path
    if rel.startswith(".."):
        return abs_path
    return rel


def _collect_parallelism_values(hw_config, *, run_type: str) -> dict:
    sch_config = getattr(hw_config, "sch_config", None)
    if sch_config is None:
        return {}
    normalized = str(run_type or "training").lower()
    if normalized == "inference":
        return {
            "tp": sch_config.tp,
            "cp": sch_config.cp,
            "ep": sch_config.inference.moe_dp,
            "pp": sch_config.pp,
            "dp": 1,
        }
    return {
        "tp": sch_config.tp,
        "cp": sch_config.cp,
        "ep": sch_config.train.ep,
        "pp": sch_config.pp,
        "dp": sch_config.train.dp,
    }


def _format_parallelism_terms(dim, parallelism_values):
    terms = []
    for axis in getattr(dim, "parallelisms", ()):
        axis_name = str(axis).lower()
        if axis_name not in parallelism_values:
            continue
        factor = parallelism_values[axis_name]
        try:
            factor_value = int(factor)
        except (TypeError, ValueError):
            factor_value = factor
        terms.append(f"{axis_name} {factor_value}")
    return terms


def network_topology_summary_training(hw_config):
    parallelism_values = _collect_parallelism_values(hw_config, run_type="training")
    dimensions = list(getattr(hw_config.network_layout, "dimensions", ()))
    ordered_axes = ["tp", "cp", "ep", "pp", "dp"]
    formatted_terms = []
    for axis in ordered_axes:
        value = parallelism_values.get(axis)
        if value is None:
            continue
        formatted_terms.append(f"{axis}:{value}")
    formatted_parallelisms = ", ".join(formatted_terms) if formatted_terms else "none"
    lines = [
        f"Parallelisms: {formatted_parallelisms}",
        f"Network Topology [dims={len(dimensions)}]",
    ]
    aggregate = 1

    for dim in dimensions:
        terms = _format_parallelism_terms(dim, parallelism_values)
        axis_repr = " × ".join(terms) if terms else "(none)"
        size_value = getattr(dim, "size", 1)
        try:
            size_int = int(size_value)
        except (TypeError, ValueError):
            size_int = None
            size_display = size_value
        else:
            aggregate *= size_int
            size_display = size_int
        lines.append(
            f"  • {dim.id} {dim.label} : {axis_repr} ⇒ size {size_display}"
        )

    lines.append(f"  ⇒  total {aggregate} devices")
    return lines


def network_topology_summary_inference(hw_config):
    parallelism_values = _collect_parallelism_values(hw_config, run_type="inference")
    sch_config = getattr(hw_config, "sch_config", None)
    replica_count = 1
    if sch_config is not None:
        replica_count = max(1, int(sch_config.inference.replica_count))
    ordered_axes = ["tp", "cp", "ep", "pp", "dp"]
    formatted_terms = []
    for axis in ordered_axes:
        value = parallelism_values.get(axis)
        if value is None:
            continue
        formatted_terms.append(f"{axis}:{value}")
    formatted_parallelisms = ", ".join(formatted_terms) if formatted_terms else "none"

    all_dimensions = list(getattr(hw_config.network_layout, "dimensions", ()))
    filtered_dimensions = [
        dim
        for dim in all_dimensions
        if any(axis != "dp" for axis in getattr(dim, "parallelisms", ())) or not dim.parallelisms
    ]
    lines = [
        f"Parallelisms: {formatted_parallelisms}",
        f"Network Topology [dims={len(filtered_dimensions)}]",
    ]

    aggregate_per_replica = 1
    for dim in filtered_dimensions:
        terms = _format_parallelism_terms(dim, parallelism_values)
        axis_repr = " × ".join(terms) if terms else "(none)"
        size_value = getattr(dim, "size", 1)
        try:
            size_int = int(size_value)
        except (TypeError, ValueError):
            size_int = None
            size_display = size_value
        else:
            aggregate_per_replica *= size_int
            size_display = size_int
        lines.append(
            f"  • {dim.id} {dim.label} : {axis_repr} ⇒ size {size_display}"
        )

    total_aggregate = 1
    total_numeric = True
    for dim in all_dimensions:
        size_value = getattr(dim, "size", 1)
        try:
            size_int = int(size_value)
        except (TypeError, ValueError):
            total_numeric = False
            break
        total_aggregate *= size_int
    if not total_numeric:
        total_aggregate = None

    if isinstance(total_aggregate, float) and total_aggregate.is_integer():
        total_aggregate = int(total_aggregate)
    lines.append(f"  replica_count: {replica_count} (throughput-only)")
    lines.append(f"  => aggregate = {total_aggregate} GPUs per modeled replica")
    if isinstance(total_aggregate, int):
        lines.append(f"  => effective total GPUs = {total_aggregate * replica_count}")
    return lines

def print_error(message):
  sys.exit(message)

# Async Graphviz helpers (moved from graphviz_async.py)
_ENV_FLAGS = ("RAPID_VISUALIZE_GRAPHS", "RAPID_PERSIST_ARTIFACT_VIZ")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


class _GraphVizAsyncManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: List[Tuple[str, Future[Any], Optional[str]]] = []
        self._completed_messages: List[str] = []
        self._enabled: Optional[bool] = None
        atexit.register(self.wait_for_all)

    def _compute_enabled(self) -> bool:
        return any(_env_truthy(flag) for flag in _ENV_FLAGS)

    def is_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = self._compute_enabled()
        return self._enabled

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="graphviz")

    def submit(
        self,
        description: str,
        fn: Callable[..., Any],
        *args: Any,
        print_message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Future[Any]]:
        if not self.is_enabled():
            fn(*args, **kwargs)
            if print_message:
                with self._lock:
                    self._completed_messages.append(print_message)
            return None

        self._ensure_executor()

        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._tasks.append((description, future, print_message))
        return future

    def wait_for_all(self) -> None:
        with self._lock:
            tasks = list(self._tasks)
            self._tasks = []

        completed: List[str] = []

        for description, future, message in tasks:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - best effort logging only
                print(f"[WARN] Graph visualization task '{description}' failed: {exc}")
            else:
                if message:
                    completed.append(message)

        with self._lock:
            if self._completed_messages:
                completed.extend(self._completed_messages)
                self._completed_messages = []

        if completed:
            print("[Debug] Graph visualizations generated:")
            for entry in completed:
                print(entry)

    def clear_cached_state(self) -> None:
        with self._lock:
            self._tasks = []
        self._enabled = None


_GRAPHVIZ_MANAGER = _GraphVizAsyncManager()


def graphviz_is_enabled() -> bool:
    return _GRAPHVIZ_MANAGER.is_enabled()


def graphviz_submit(
    description: str,
    fn: Callable[..., Any],
    *args: Any,
    print_message: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Future[Any]]:
    return _GRAPHVIZ_MANAGER.submit(description, fn, *args, print_message=print_message, **kwargs)


def graphviz_wait_for_all() -> None:
    _GRAPHVIZ_MANAGER.wait_for_all()


def graphviz_reset_for_tests() -> None:
    _GRAPHVIZ_MANAGER.clear_cached_state()
