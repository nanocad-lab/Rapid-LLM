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

"""AstraSim cache management and execution utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bootstrap import ensure_chakra_available
from .config_generation import (
    ASTRA_DEBUG,
    choose_collective,
    compute_intra_inter_ib_ll_from_hw,
    derive_topology_from_hw,
    generate_astrasim_configs_from_hw,
)
from .et_utils import (
    chakra_encode,
    ensure_dir,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
    size_label,
    write_comm_microbenchmark,
    write_et_node,
    write_point_to_point_microbenchmark,
)

ensure_chakra_available()

_DEFAULT_ASTRA_CACHE_DIR = os.path.abspath("./astra_cache")

_CACHE_MODES = {
    "NO_CACHE",
    "CACHE_READONLY",
    "CACHE_READWRITE",
}

_COLLECTIVE_ENUMS = {
    "all_gather": pb.ALL_GATHER,
    "allreduce": pb.ALL_REDUCE,
    "all_reduce": pb.ALL_REDUCE,
    "reduce_scatter": pb.REDUCE_SCATTER,
    "all_to_all": pb.ALL_TO_ALL,
    "alltoall": pb.ALL_TO_ALL,
}


def _cache_mode() -> str:
    """Return the normalized cache mode set via ``RAPID_ASTRA_CACHE_MODE``."""

    mode = os.environ.get("RAPID_ASTRA_CACHE_MODE", "CACHE_READWRITE")
    if mode is None:
        return "CACHE_READWRITE"
    normalized = mode.strip().upper()
    if normalized not in _CACHE_MODES:
        return "CACHE_READWRITE"
    return normalized


def _collectives_from_hw(hw_obj, topo: str) -> Dict[str, str]:
    """Extract collective overrides from ``hw_obj`` for a topology ``topo``."""

    exec_backend = getattr(hw_obj, "execution_backend", None)
    if exec_backend and exec_backend.astra:
        coll = exec_backend.astra.collectives
        return {
            "all_gather": choose_collective(coll.all_gather, topo, "all-gather"),
            "all_reduce": choose_collective(coll.all_reduce, topo, "all-reduce"),
            "reduce_scatter": choose_collective(coll.reduce_scatter, topo, "reduce-scatter"),
            "all_to_all": choose_collective(coll.all_to_all, topo, "all-to-all"),
        }
    return {
        "all_gather": choose_collective("auto", topo, "all-gather"),
        "all_reduce": choose_collective("auto", topo, "all-reduce"),
        "reduce_scatter": choose_collective("auto", topo, "reduce-scatter"),
        "all_to_all": choose_collective("auto", topo, "all-to-all"),
    }


def _sys_options_from_hw(hw_obj) -> Optional[Dict[str, Any]]:
    """Return AstraSim system overrides encoded on ``hw_obj`` if present."""

    exec_backend = getattr(hw_obj, "execution_backend", None)
    if not exec_backend or not exec_backend.astra:
        return None
    sys_opts = getattr(exec_backend.astra, "sys_options", None)
    if sys_opts is None:
        return None
    if hasattr(sys_opts, "_asdict"):
        opts_dict = sys_opts._asdict()
    else:
        opts_dict = {
            "endpoint_delay": getattr(sys_opts, "endpoint_delay", None),
            "active_chunks_per_dimension": getattr(sys_opts, "active_chunks_per_dimension", None),
            "preferred_dataset_splits": getattr(sys_opts, "preferred_dataset_splits", None),
            "collective_arbitration": getattr(sys_opts, "collective_arbitration", None),
        }
    filtered = {k: v for k, v in opts_dict.items() if v is not None}
    return filtered or None


def _canonical_sig(sig: Dict[str, Any]) -> str:
    """Serialize a cache signature dictionary deterministically."""

    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _hash_sig(canonical: str) -> str:
    """Hash a canonical signature string using SHA-256."""

    h = hashlib.sha256()
    h.update(canonical.encode("utf-8"))
    return h.hexdigest()


def _hash_file_bundle(paths: Iterable[str]) -> str:
    """Hash the contents of all existing files in ``paths`` for cache lookups."""

    h = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path or not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def _path_within_dir(path: Optional[str], directory: str) -> bool:
    """Return True when ``path`` is within ``directory``."""

    if not path:
        return False
    try:
        path_abs = os.path.abspath(path)
        dir_abs = os.path.abspath(directory)
        return os.path.commonpath([path_abs, dir_abs]) == dir_abs
    except (OSError, ValueError):
        return False


def _load_cache(cache_path: str) -> Dict[str, Any]:
    """Load a JSON cache manifest if available, otherwise return an empty dict."""

    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


def _save_cache(cache_path: str, cache: Dict[str, Any]) -> None:
    """Persist ``cache`` to ``cache_path`` safely via atomic rename."""

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp_path, cache_path)


def ensure_cache_file_exists(cache_path: str = "./astra_cache/cache.json") -> None:
    """Create an empty cache file at ``cache_path`` if it does not exist."""

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({}, fh)
    except OSError:
        pass


def get_remote_memory_path() -> str:
    """Return the bundled remote-memory configuration shipped with AstraSim."""

    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "astra-sim",
        "examples",
        "remote_memory",
        "analytical",
        "no_memory_expansion.json",
    )


def generate_workload_et(
    comm: str,
    npus_count: int,
    size_bytes: int,
    astra_config_dir: str = "./astra_cache",
) -> str:
    """Emit microbenchmark ET files for ``comm`` of ``size_bytes`` across ``npus_count`` ranks."""

    comm_lower = comm.lower()
    label = size_label(size_bytes)
    base_dir = os.path.join(astra_config_dir, "workload", comm_lower, f"{npus_count}npus_{label}")
    prefix = os.path.join(base_dir, f"{comm_lower}_{label}")

    if comm_lower == "pipeline":
        return write_point_to_point_microbenchmark(prefix, size_bytes)

    if comm_lower not in _COLLECTIVE_ENUMS:
        raise ValueError(f"Unsupported comm type: {comm}")

    return write_comm_microbenchmark(prefix, npus_count, _COLLECTIVE_ENUMS[comm_lower], size_bytes)


def generate_concurrent_collectives_et(
    npus_count: int,
    collectives: List[Tuple[str, int, int]],
    prefix_path: str,
) -> str:
    """Write ET traces that sequence ``collectives`` on ``npus_count`` ranks."""

    ensure_dir(os.path.dirname(prefix_path))

    for rank in range(npus_count):
        et_path = f"{prefix_path}.{rank}.et"
        with open(et_path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            node_id = 0
            for i, (coll_name, size_bytes, delay_ns) in enumerate(collectives):
                if coll_name not in _COLLECTIVE_ENUMS:
                    raise ValueError(f"Unsupported collective: {coll_name}")
                coll_type = _COLLECTIVE_ENUMS[coll_name]
                if delay_ns > 0:
                    delay_micros = delay_ns // 1000
                    delay_node = new_comp_node(node_id, f"delay_{i}_{delay_micros}us", delay_micros)
                    write_et_node(fh, delay_node)
                    delay_node_id = node_id
                    node_id += 1
                    comm_node = new_comm_node(node_id, f"{coll_name}_{i}", coll_type, size_bytes)
                    comm_node.ctrl_deps.append(delay_node_id)
                    write_et_node(fh, comm_node)
                    node_id += 1
                else:
                    comm_node = new_comm_node(node_id, f"{coll_name}_{i}", coll_type, size_bytes)
                    write_et_node(fh, comm_node)
                    node_id += 1
    return prefix_path


def _astrasim_binary_path() -> str:
    """Return the default AstraSim analytical binary path relative to the repo root."""

    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "astra-sim",
        "build",
        "astra_analytical",
        "build",
        "bin",
        "AstraSim_Analytical_Congestion_Aware",
    )


def run_astrasim_analytical(
    workload_prefix: Optional[str],
    system_json: str,
    network_yaml: str,
    remote_memory_json: Optional[str] = None,
    comm_group_json: Optional[str] = None,
    binary_path: Optional[str] = None,
) -> Tuple[List[float], float]:
    """Execute AstraSim analytically and return per-rank and max wall times."""

    bin_path = binary_path or _astrasim_binary_path()
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"AstraSim binary not found at {bin_path}")

    if not workload_prefix:
        raise ValueError("workload_prefix is required for analytical AstraSim runs")

    cmd = [
        bin_path,
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_json}",
        f"--network-configuration={network_yaml}",
    ]

    if remote_memory_json:
        cmd.append(f"--remote-memory-configuration={remote_memory_json}")
    if comm_group_json and os.path.exists(comm_group_json):
        cmd.append(f"--comm-group-configuration={comm_group_json}")

    if ASTRA_DEBUG:
        print("Command:")
        print(" ".join(cmd))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"AstraSim execution failed with code {proc.returncode}:\n{output}")

    times_cycles: List[int] = []
    for line in output.splitlines():
        wall_match = re.search(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)", line)
        if wall_match:
            times_cycles.append(int(wall_match.group(2)))
            continue
        # finish_match = re.search(r"sys\[(\d+)\] finished,\s*(\d+) cycles", line)
        # if finish_match:
        #     times_cycles.append(int(finish_match.group(2)))

    per_node_sec = [t * 1e-9 for t in times_cycles]
    max_sec = max(per_node_sec) if per_node_sec else 0.0
    return per_node_sec, max_sec


def run_cache_astrasim(
    hw_obj,
    comm: str,
    npus_count: int,
    size_bytes: int,
    astra_config_dir: str = "./astra_cache",
    cache_path: str = "./astra_cache/cache.json",
    manifest_json_path: Optional[str] = None,
    workload_prefix: Optional[str] = None,
    comm_group_json: Optional[str] = None,
    axes_filter: Optional[Sequence[str]] = None,
    transform_2d_to_1d: bool = False,
    files = None,
) -> Tuple[List[float], float]:
    """Run AstraSim with caching to avoid recomputation when inputs match."""

    if isinstance(hw_obj, str):
        raise TypeError("run_cache_astrasim expects a parsed HWConfig object, not a path")

    comm_lower = str(comm).lower()

    cache_mode = _cache_mode()
    allow_read = cache_mode in {"CACHE_READONLY", "CACHE_READWRITE"}
    allow_write = cache_mode == "CACHE_READWRITE"

    temp_cache_dir: Optional[str] = None
    if cache_mode == "NO_CACHE":
        temp_cache_dir = tempfile.mkdtemp(prefix="astrasim_cache_")
        astra_config_dir = temp_cache_dir
        cache_path = os.path.join(temp_cache_dir, "cache.json")
    else:
        override_dir = os.environ.get("ASTRA_CACHE_DIR")
        if override_dir:
            astra_config_dir = override_dir
            cache_path = os.path.join(override_dir, "cache.json")
        if allow_read or allow_write:
            ensure_cache_file_exists(cache_path)

    ephemeral_paths: List[str] = []

    try:
        (intra_ib_bps, intra_ll_s), _ = compute_intra_inter_ib_ll_from_hw(hw_obj)
        if not intra_ib_bps or intra_ib_bps <= 0 or not intra_ll_s or intra_ll_s <= 0:
            raise ValueError("Invalid intra-node IB/LL computed from HW config")
        ib_gbps = round((intra_ib_bps) / float(1 << 30), 6)
        ll_ns = round(intra_ll_s * 1e9, 3)
        topo = derive_topology_from_hw(hw_obj)
        colls = _collectives_from_hw(hw_obj, topo)
        sys_opts_sig = _sys_options_from_hw(hw_obj)
        axes_filter_tuple: Optional[Tuple[str, ...]] = tuple(axes_filter) if axes_filter else None

        cache_dir_abs = os.path.abspath(astra_config_dir)
        ephemeral_configs = (
            files is None
            and comm_lower != "graph"
            and cache_dir_abs == _DEFAULT_ASTRA_CACHE_DIR
        )

        sig: Dict[str, Any] = {
            "comm": comm_lower,
            "npus": int(npus_count),
            "size_bytes": int(size_bytes),
            "topology": topo,
            "physical_topology": topo,
            "ib_gbps": ib_gbps,
            "ll_ns": ll_ns,
            "collectives": colls,
            "backend": "analytical",
        }
        if sys_opts_sig is not None:
            sig["sys_options"] = sys_opts_sig
        if manifest_json_path:
            sig["multinode"] = True
        if axes_filter_tuple is not None:
            sig["axes_filter"] = axes_filter_tuple
        if not files:
            files = generate_astrasim_configs_from_hw(
                hw_obj,
                out_dir=astra_config_dir,
                npus_count=npus_count,
                axes_filter=axes_filter_tuple,
                transform_2d_to_1d=transform_2d_to_1d,
                ephemeral_outputs=ephemeral_configs,
            )
            if ephemeral_configs:
                ephemeral_paths.extend(
                    [
                        files.get("system_json"),
                        files.get("network_yaml"),
                    ]
                )
        topo_list = files.get("topology_list")
        npus_per_dim = files.get("npus_per_dim")
        if topo_list:
            sig["topology"] = list(topo_list)
            sig["physical_topology"] = list(topo_list)
        if npus_per_dim:
            sig["npus_per_dim"] = list(npus_per_dim)

        canonical = _canonical_sig(sig)

        if workload_prefix:
            pass
        elif not manifest_json_path:
            label = size_label(size_bytes)
            base_dir = os.path.join(astra_config_dir, "workload", comm_lower, f"{npus_count}npus_{label}")
            workload_prefix = os.path.join(base_dir, f"{comm_lower}_{label}")
            expected = [f"{workload_prefix}.{r}.et" for r in range(npus_count)]
            if not all(os.path.exists(p) for p in expected):
                generate_workload_et(comm, npus_count, size_bytes, astra_config_dir=astra_config_dir)
        else:
            workload_prefix = os.path.splitext(manifest_json_path)[0]

        remote_mem_path = get_remote_memory_path()

        if manifest_json_path:
            bundle_list: List[str] = [manifest_json_path]
            for path in (files["system_json"], files["network_yaml"]):
                if path and os.path.exists(path):
                    bundle_list.append(path)
            if remote_mem_path and os.path.exists(remote_mem_path):
                bundle_list.append(remote_mem_path)
            if comm_group_json and os.path.exists(comm_group_json):
                bundle_list.append(comm_group_json)
            cache_key = _hash_file_bundle(bundle_list)
        else:
            cache_key = _hash_sig(canonical)

        if allow_read:
            cache = _load_cache(cache_path)
            entry = cache.get(cache_key) if cache else None
            if entry:
                cached_per = entry.get("per_node_sec", [])
                cached_max = float(entry.get("max_sec", 0.0))
                if cached_per and cached_max > 0 and len(cached_per) == int(npus_count) and all((t > 0 for t in cached_per)):
                    if comm_lower == "graph":
                        try:
                            print(
                                f"[AstraSim] Cache HIT: comm={comm}, npus={npus_count}, size={size_bytes}, total={cached_max:.6f}s"
                            )
                        except Exception:
                            pass
                    return cached_per, cached_max
                if allow_write and cache_key in cache:
                    try:
                        cache.pop(cache_key)
                        _save_cache(cache_path, cache)
                    except Exception:
                        pass

        attempts = 0
        per_node_sec: List[float] = []
        max_sec: float = 0.0
        last_outcome_zero = False

        # Capture command details for debugging
        bin_path = _astrasim_binary_path()
        debug_cmd = [
            bin_path,
            f"--workload-configuration={workload_prefix}",
            f"--system-configuration={files['system_json']}",
            f"--network-configuration={files['network_yaml']}",
        ]
        if remote_mem_path:
            debug_cmd.append(f"--remote-memory-configuration={remote_mem_path}")
        if comm_group_json and os.path.exists(comm_group_json):
            debug_cmd.append(f"--comm-group-configuration={comm_group_json}")
        cmd_str = " ".join(debug_cmd)
        # print(f"Full AstraSim command:\n{cmd_str}")

        while attempts < 2:
            attempts += 1
            per_node_sec, max_sec = run_astrasim_analytical(
                workload_prefix=workload_prefix,
                system_json=files["system_json"],
                network_yaml=files["network_yaml"],
                remote_memory_json=remote_mem_path,
                comm_group_json=comm_group_json,
            )
            if per_node_sec and max_sec > 0 and len(per_node_sec) == int(npus_count) and all((t > 0 for t in per_node_sec)):
                last_outcome_zero = False
                break
            last_outcome_zero = True
            time.sleep(0.1)
        if last_outcome_zero:
            raise RuntimeError(
                f"AstraSim returned zero time for {comm} size={size_bytes} npus={npus_count} after 2 retries.\n"
                f"Full AstraSim command:\n{cmd_str}"
            )

        cache_entry = {
            "signature": sig,
            "canonical": canonical,
            "per_node_sec": per_node_sec,
            "max_sec": max_sec,
            "workload_prefix": workload_prefix,
            "system_json": files["system_json"],
            "network_yaml": files["network_yaml"],
        }
        if manifest_json_path:
            cache_entry["manifest_path"] = manifest_json_path

        if allow_write:
            cache = _load_cache(cache_path)
            cache[cache_key] = cache_entry
            _save_cache(cache_path, cache)

        return per_node_sec, max_sec
    finally:
        if ephemeral_paths:
            for cleanup_path in {p for p in ephemeral_paths if p}:
                if _path_within_dir(cleanup_path, astra_config_dir):
                    try:
                        os.remove(cleanup_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
        if temp_cache_dir is not None:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)


__all__ = [
    "ASTRA_DEBUG",
    "ensure_cache_file_exists",
    "generate_concurrent_collectives_et",
    "generate_workload_et",
    "get_remote_memory_path",
    "run_astrasim_analytical",
    "run_cache_astrasim",
]
