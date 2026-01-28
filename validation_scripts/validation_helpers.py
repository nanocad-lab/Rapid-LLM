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

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from tools.parallelism_sweep import set_astrasim_cache_mode  # type: ignore

VALIDATION_WORKERS_ENV = "RAPID_VALIDATION_WORKERS"
VALIDATION_QUIET_ENV = "RAPID_VALIDATION_QUIET"
VALIDATION_KEEP_TMP_ENV = "RAPID_VALIDATION_KEEP_TMP"
DEFAULT_WORKER_COUNT = 8

ResultParser = Callable[[str, "ValidationSpec"], Dict[str, Any]]


@dataclass(frozen=True)
class ValidationSpec:
  label: str
  model_overrides: Optional[Mapping[str, Any]] = None
  hardware_overrides: Optional[Mapping[str, Any]] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  model_config_path: Optional[str] = None
  hardware_config_path: Optional[str] = None
  order: int = 0


@dataclass
class ValidationResult:
  spec: ValidationSpec
  success: bool
  metrics: Dict[str, Any]
  raw_output: str
  error: Optional[str] = None
  returncode: Optional[int] = None
  duration_s: float = 0.0
  model_config_used: Optional[str] = None
  hardware_config_used: Optional[str] = None


@dataclass
class _WorkerContext:
  run_perf_path: str
  python_executable: str
  tmp_root: str
  env_overrides: Dict[str, str]
  model_config_cache: Dict[str, Dict[str, Any]]
  hardware_config_cache: Dict[str, Dict[str, Any]]
  default_model_config_path: str
  default_hardware_config_path: str
  result_parser: ResultParser
  cache_mode: str
  extra_run_perf_args: Tuple[str, ...]


_WORKER_CONTEXT: Optional[_WorkerContext] = None


def _worker_init(context: _WorkerContext) -> None:
  global _WORKER_CONTEXT
  _WORKER_CONTEXT = context
  set_astrasim_cache_mode(context.cache_mode)

def _is_list_of_dicts(value: Any) -> bool:
  if not isinstance(value, list):
    return False  
  return all(isinstance(item, dict) for item in value)

def _merge_list_of_dicts(orig: List[Dict[str, Any]], 
                         overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  orig_index = {d["id"]: d for d in orig}
  orig_order = [d["id"] for d in orig]
  
  # Apply deep updates (or add new items)
  for d in overrides:
    id = d["id"]
    if id in orig_index:
      orig_index[id] = _deep_update(copy.deepcopy(orig_index[id]), d)
    else:
      orig_index[id] = copy.deepcopy(d)

  # Rebuild list preserving original order, appending new items at the end
  result: List[Dict[str, Any]] = []
  for orig_id in orig_order:
    result.append(orig_index[orig_id])
  for d in overrides:
    id = d["id"]
    if id not in orig_order:
      result.append(orig_index[id])
  return result


def _summarize_network_config(hw_config_path: str) -> str:
  try:
    with open(hw_config_path, "r") as handle:
      data = yaml.safe_load(handle)
  except Exception as exc:
    return f"[validation] Failed to read hardware config {hw_config_path}: {exc}"
  if not isinstance(data, dict):
    return f"[validation] Hardware config {hw_config_path} is not a mapping."
  network = data.get("network", {})
  dims = network.get("dimensions")
  if not isinstance(dims, list):
    return f"[validation] Hardware config {hw_config_path} has no network dimensions."
  lines = [f"[validation] Network summary ({hw_config_path}):"]
  for dim in dims:
    if not isinstance(dim, dict):
      continue
    label = dim.get("label") or dim.get("id", "<unnamed>")
    size = dim.get("size")
    topo = dim.get("topology", {})
    topo_type = topo.get("type")
    bandwidth = topo.get("bandwidth")
    latency = topo.get("latency")
    parallelisms = dim.get("parallelisms", [])
    lines.append(
      f"  {label}: size={size}, topo={topo_type}, bw={bandwidth}, lat={latency}, axes={parallelisms}"
    )
  return "\n".join(lines)


def _find_network_config(search_roots: Sequence[Path]) -> Optional[Path]:
  candidates: List[Path] = []
  for root in search_roots:
    if not root.exists():
      continue
    candidates.extend(root.rglob("network_analytical_*.yml"))
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _log_astrasim_artifacts(tmp_dir: str, env: Mapping[str, str]) -> None:
  roots = [Path(tmp_dir)]
  astra_cache = env.get("ASTRA_CACHE_DIR")
  if astra_cache:
    roots.append(Path(astra_cache))
  roots.append(Path(tmp_dir) / "output")
  roots.append(Path(PROJECT_ROOT) / "output")
  path = _find_network_config(roots)
  if path is None:
    return
  work_dir = path.parent
  cache_desc = f" (ASTRA_CACHE_DIR={astra_cache})" if astra_cache else ""
  print(f"[validation] AstraSim artifacts: work_dir={work_dir}{cache_desc}")
  print(f"[validation] AstraSim network config: {path}")

def _deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
  for key, value in overrides.items():
    if isinstance(value, Mapping):
      existing = target.get(key)
      if not isinstance(existing, dict):
        existing = {}
      target[key] = _deep_update(existing, value)
    elif isinstance(value, list):
      existing_list = target.get(key)
      if isinstance(existing_list, list) and _is_list_of_dicts(existing_list) and _is_list_of_dicts(value):
        target[key] = _merge_list_of_dicts(existing_list, value)
      else:
        target[key] = copy.deepcopy(value)
    else:
      target[key] = value
  return target


def _load_yaml(path: str) -> Dict[str, Any]:
  with open(path, "r") as handle:
    data = yaml.safe_load(handle)
  if not isinstance(data, dict):
    raise ValueError(f"{path} did not contain a YAML dictionary.")
  return data


def _write_yaml(path: str, data: Dict[str, Any]) -> None:
  with open(path, "w") as handle:
    yaml.safe_dump(data, handle, sort_keys=False)


def _prepare_config(
  tmp_dir: str,
  base_path: str,
  base_dict: Dict[str, Any],
  overrides: Optional[Mapping[str, Any]],
  filename: str,
) -> str:
  if not overrides:
    return base_path
  updated = copy.deepcopy(base_dict)
  _deep_update(updated, overrides)
  config_path = os.path.join(tmp_dir, filename)
  _write_yaml(config_path, updated)
  return config_path


def _read_env_worker_setting() -> Tuple[int, str]:
  quiet = str(os.environ.get(VALIDATION_QUIET_ENV, "")).lower() in {"1", "true", "yes"}
  env_raw = os.environ.get(VALIDATION_WORKERS_ENV)
  if env_raw is None:
    return DEFAULT_WORKER_COUNT, f"{VALIDATION_WORKERS_ENV} unset (default {DEFAULT_WORKER_COUNT})"
  try:
    value = int(env_raw)
  except (TypeError, ValueError):
    if not quiet:
      print(f"[validation] WARNING: {VALIDATION_WORKERS_ENV} must be a positive integer (got {env_raw!r}); using default {DEFAULT_WORKER_COUNT}.")
    return DEFAULT_WORKER_COUNT, f"{VALIDATION_WORKERS_ENV} invalid ({env_raw!r}) -> default {DEFAULT_WORKER_COUNT}"
  if value <= 0:
    if not quiet:
      print(f"[validation] WARNING: {VALIDATION_WORKERS_ENV} must be > 0 (got {env_raw!r}); using default {DEFAULT_WORKER_COUNT}.")
    return DEFAULT_WORKER_COUNT, f"{VALIDATION_WORKERS_ENV} <= 0 ({env_raw!r}) -> default {DEFAULT_WORKER_COUNT}"
  return value, f"{VALIDATION_WORKERS_ENV}={value}"


def _determine_worker_count(spec_count: int, explicit_max: Optional[int]) -> int:
  quiet = str(os.environ.get(VALIDATION_QUIET_ENV, "")).lower() in {"1", "true", "yes"}
  env_workers, env_desc = _read_env_worker_setting()
  requested = explicit_max if explicit_max is not None else env_workers
  request_desc = f"max_workers={explicit_max}" if explicit_max is not None else env_desc
  if requested <= 0:
    if not quiet:
      print(f"[validation] WARNING: requested worker budget {requested} is not positive; bumping to 1.")
    requested = 1
  cpu_limit = max(1, os.cpu_count() or 1)
  if explicit_max is None and requested > cpu_limit:
    if not quiet:
      print(f"[validation] WARNING: {VALIDATION_WORKERS_ENV} requested {requested} worker(s) but only {cpu_limit} CPU core(s) detected. Capping to {cpu_limit}.")
    requested = cpu_limit
  elif explicit_max is not None and requested > cpu_limit:
    if not quiet:
      print(f"[validation] WARNING: max_workers requested {requested} worker(s) but only {cpu_limit} CPU core(s) detected. Capping to {cpu_limit}.")
    requested = cpu_limit
  worker_count = max(1, min(spec_count, requested))
  if not quiet:
    print(f"[validation] Using {worker_count} worker(s) for {spec_count} experiment(s) "
          f"(requested {request_desc}, cpu_limit={cpu_limit}).")
  return worker_count


def _execute_spec(spec: ValidationSpec) -> ValidationResult:
  if _WORKER_CONTEXT is None:
    raise RuntimeError("Worker context is not initialised.")
  ctx = _WORKER_CONTEXT
  tmp_dir = tempfile.mkdtemp(prefix="validation_run_", dir=ctx.tmp_root)
  env = os.environ.copy()
  env.update(ctx.env_overrides)
  env.setdefault("ASTRA_CACHE_DIR", os.path.join(tmp_dir, "astra_cache"))
  start_time = time.time()
  base_model_path = os.path.abspath(spec.model_config_path or ctx.default_model_config_path)
  base_hw_path = os.path.abspath(spec.hardware_config_path or ctx.default_hardware_config_path)
  if base_model_path not in ctx.model_config_cache:
    raise ValueError(f"Model config {base_model_path} was not preloaded for validation.")
  if base_hw_path not in ctx.hardware_config_cache:
    raise ValueError(f"Hardware config {base_hw_path} was not preloaded for validation.")
  model_config_path = base_model_path
  hardware_config_path = base_hw_path
  try:
    model_config_path = _prepare_config(
      tmp_dir,
      base_model_path,
      ctx.model_config_cache[base_model_path],
      spec.model_overrides,
      "model.yaml",
    )
    hardware_config_path = _prepare_config(
      tmp_dir,
      base_hw_path,
      ctx.hardware_config_cache[base_hw_path],
      spec.hardware_overrides,
      "hardware.yaml",
    )
    cmd = [
      ctx.python_executable,
      ctx.run_perf_path,
      "--hardware_config",
      hardware_config_path,
      "--model_config",
      model_config_path,
    ]
    if ctx.extra_run_perf_args:
      cmd.extend(ctx.extra_run_perf_args)
    proc = subprocess.run(
      cmd,
      cwd=tmp_dir,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      check=False,
      env=env,
    )
    duration = time.time() - start_time
    output = proc.stdout or ""
    _log_astrasim_artifacts(tmp_dir, env)
    if proc.returncode != 0:
      if "Assertion `0 <= dest && dest < devices_count' failed" in output:
        summary = _summarize_network_config(hardware_config_path)
        output = f"{output}\n\n{summary}"
        print(summary)
      return ValidationResult(
        spec=spec,
        success=False,
        metrics={},
        raw_output=output,
        error=f"RAPID-LLM exited with status {proc.returncode}",
        returncode=proc.returncode,
        duration_s=duration,
        model_config_used=model_config_path,
        hardware_config_used=hardware_config_path,
      )
    try:
      metrics = ctx.result_parser(output, spec)
    except Exception as exc:
      return ValidationResult(
        spec=spec,
        success=False,
        metrics={},
        raw_output=output,
        error=f"Failed to parse RAPID-LLM output: {exc}",
        returncode=proc.returncode,
        duration_s=duration,
        model_config_used=model_config_path,
        hardware_config_used=hardware_config_path,
      )
    return ValidationResult(
      spec=spec,
      success=True,
      metrics=metrics,
      raw_output=output,
      error=None,
      returncode=proc.returncode,
      duration_s=duration,
      model_config_used=model_config_path,
      hardware_config_used=hardware_config_path,
    )
  finally:
    keep_tmp = str(os.environ.get(VALIDATION_KEEP_TMP_ENV, "")).lower() in {"1", "true", "yes"}
    if keep_tmp:
      print(f"[validation] Keeping temp dir: {tmp_dir}")
    else:
      shutil.rmtree(tmp_dir, ignore_errors=True)


def run_validation_suite(
  specs: Sequence[ValidationSpec],
  *,
  base_model_config_path: str,
  base_hardware_config_path: str,
  result_parser: ResultParser,
  run_perf_path: Optional[str] = None,
  python_executable: Optional[str] = None,
  tmp_root: Optional[str] = None,
  max_workers: Optional[int] = None,
  env_overrides: Optional[Dict[str, str]] = None,
  cache_mode: str = "NO_CACHE",
  extra_run_perf_args: Optional[Sequence[str]] = None,
  show_progress: bool = False,
) -> List[ValidationResult]:
  if not specs:
    return []
  run_perf_path = os.path.abspath(run_perf_path or os.path.join(PROJECT_ROOT, "run_perf.py"))
  python_executable = python_executable or sys.executable
  tmp_root = os.path.abspath(tmp_root or os.path.join(PROJECT_ROOT, "tmp", "validation_runs"))
  os.makedirs(tmp_root, exist_ok=True)
  set_astrasim_cache_mode(cache_mode)
  env_map = dict(env_overrides or {})
  env_map.setdefault("RAPID_ASTRA_CACHE_MODE", os.environ.get("RAPID_ASTRA_CACHE_MODE", cache_mode))
  model_paths = {os.path.abspath(base_model_config_path)}
  hardware_paths = {os.path.abspath(base_hardware_config_path)}
  for spec in specs:
    if spec.model_config_path:
      model_paths.add(os.path.abspath(spec.model_config_path))
    if spec.hardware_config_path:
      hardware_paths.add(os.path.abspath(spec.hardware_config_path))
  model_cache = {path: _load_yaml(path) for path in model_paths}
  hardware_cache = {path: _load_yaml(path) for path in hardware_paths}
  extra_args: Tuple[str, ...] = tuple(extra_run_perf_args or ())
  assigned_specs = [replace(spec, order=idx) for idx, spec in enumerate(specs)]
  worker_count = _determine_worker_count(len(assigned_specs), max_workers)
  context = _WorkerContext(
    run_perf_path=run_perf_path,
    python_executable=python_executable,
    tmp_root=tmp_root,
    env_overrides=env_map,
    model_config_cache=model_cache,
    hardware_config_cache=hardware_cache,
    default_model_config_path=os.path.abspath(base_model_config_path),
    default_hardware_config_path=os.path.abspath(base_hardware_config_path),
    result_parser=result_parser,
    cache_mode=os.environ.get("RAPID_ASTRA_CACHE_MODE", cache_mode),
    extra_run_perf_args=extra_args,
  )
  results: List[ValidationResult] = []
  progress = None
  if show_progress:
    try:
      from tqdm import tqdm  # type: ignore
      progress = tqdm(total=len(assigned_specs), desc="validation", leave=True)
    except Exception:
      progress = None

  with ProcessPoolExecutor(max_workers=worker_count, initializer=_worker_init, initargs=(context,)) as executor:
    future_map = {executor.submit(_execute_spec, spec): spec for spec in assigned_specs}
    for future in as_completed(future_map):
      spec = future_map[future]
      try:
        result = future.result()
      except Exception as exc:
        result = ValidationResult(
          spec=spec,
          success=False,
          metrics={},
          raw_output="",
          error=str(exc),
          returncode=None,
          duration_s=0.0,
          model_config_used=None,
          hardware_config_used=None,
        )
      results.append(result)
      if progress is not None:
        progress.update(1)
  if progress is not None:
    progress.close()
  results.sort(key=lambda item: item.spec.order)
  return results


_DECODE_TIME_REGEX = re.compile(
  r"\[prefill\]\s*time:\s*[0-9.]+s,\s*\[decode\]\s*time:\s*([0-9.]+)s,\s*\[total\]\s*time:\s*[0-9.]+s"
)

_INF_TIME_REGEX = re.compile(
  r"LLM inference time:\s*([0-9]+(?:\.[0-9]+)?)s"
)
_TTFT_REGEX = re.compile(
  r"LLM time to first token:\s*([0-9]+(?:\.[0-9]+)?)s"
)
_TPOT_REGEX = re.compile(
  r"midpoint=([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)s"
)
_TRAIN_TIME_REGEX = re.compile(
  r"(?:Training time for batch|LLM inference time):\s*([0-9]+(?:\.[0-9]+)?)s"
)

def parse_decode_time(output: str, spec: ValidationSpec) -> Dict[str, Any]:
  match = _DECODE_TIME_REGEX.search(output)
  if not match:
    raise ValueError(f"Failed to parse decode time for experiment '{spec.label}'.")
  return {"decode_time_s": float(match.group(1))}

def parse_inference_time(output: str, spec: ValidationSpec) -> Dict[str, Any]:
  match = _INF_TIME_REGEX.search(output)
  if not match:
    raise ValueError(f"Failed to parse inference time for experiment '{spec.label}'.")
  return {"inference_time_s": float(match.group(1))}

def parse_ttft(output: str, spec: ValidationSpec) -> Dict[str, Any]:
  match = _TTFT_REGEX.search(output)
  if not match:
    raise ValueError(f"Failed to parse time-to-first-token for experiment '{spec.label}'.")
  return {"ttft_s": float(match.group(1))}

def parse_tpot(output: str, spec: ValidationSpec) -> Dict[str, Any]:
  match = _TPOT_REGEX.search(output)
  if not match:
    raise ValueError(f"Failed to parse per-token output time for experiment '{spec.label}'.")
  return {"tpot_s": float(match.group(1))}

def parse_training_time(output: str, spec: ValidationSpec) -> Dict[str, Any]:
  match = _TRAIN_TIME_REGEX.search(output)
  if not match:
    raise ValueError(f"Failed to parse training time for experiment '{spec.label}'.")
  return {"training_time_s": float(match.group(1))}

# ---- Pytest integration for quiet validation reporting ----
_VALIDATION_METRICS: List[Tuple[str, str, float, Optional[float]]] = []

# magic function, needed for pytest to work
def pytest_configure(config):
  _VALIDATION_METRICS.clear()


@pytest.fixture
def record_validation(request):
  def _record(label: str, achieved_pct: float, expected_pct: Optional[float] = None) -> None:
    #node id is tests/test_IMEC_A100_no_network.py::test_avg_abs_error_below_threshold_network_ignored
    # I want just the filename under tests/, not the function name
    filename = request.node.nodeid.split("::")[0]
    filename = filename.split("/")[-1]
    _VALIDATION_METRICS.append(
      (filename, label, achieved_pct, expected_pct)
    )
  return _record


def _collect_func_test_stats(terminalreporter):
  prefix = "tests/test_func_test.py::test_func_test"
  stats = terminalreporter.stats
  passed = [rep for rep in stats.get("passed", []) if rep.nodeid.startswith(prefix)]
  failed = [rep for rep in stats.get("failed", []) if rep.nodeid.startswith(prefix)]
  skipped = [rep for rep in stats.get("skipped", []) if rep.nodeid.startswith(prefix)]
  xfailed = [rep for rep in stats.get("xfailed", []) if rep.nodeid.startswith(prefix)]
  xpassed = [rep for rep in stats.get("xpassed", []) if rep.nodeid.startswith(prefix)]
  total = len(passed) + len(failed) + len(skipped) + len(xfailed) + len(xpassed)
  if total == 0:
    return None
  return {
    "total": total,
    "passed": len(passed) + len(xpassed),
    "xfailed": len(xfailed),
  }


def pytest_terminal_summary(terminalreporter, exitstatus):
  func_stats = _collect_func_test_stats(terminalreporter)
  if not _VALIDATION_METRICS and not func_stats:
    return
  tr = terminalreporter
  tr.write_sep("=", "Test results")
  for nodeid, label, achieved, expected in _VALIDATION_METRICS:
    line = f"{nodeid}: {label} achieved {achieved:.2f}%"
    if expected is not None:
      line += f" (expected <= {expected:.2f}%)"
    tr.write_line(line)
  if func_stats:
    tr.write_line(
      f"{func_stats['passed']}/{func_stats['total']} functionality tests passed ({func_stats['xfailed']} xfailed)"
    )

__all__ = [
  "ValidationSpec",
  "ValidationResult",
  "run_validation_suite",
  "parse_decode_time",
  "parse_inference_time",
  "parse_training_time",
  "record_validation",
]
