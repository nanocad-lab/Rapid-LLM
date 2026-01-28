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

"""AstraSim integration helpers for RAPID-LLM with optional dependency handling."""

from typing import Optional

from .bootstrap import add_chakra_to_sys_path, ensure_chakra_available

_ASTRASIM_AVAILABLE = True
_ASTRASIM_IMPORT_ERROR: Optional[BaseException] = None

try:
    from .config_generation import (
        compute_intra_inter_ib_ll_from_hw,
        derive_topology_from_hw,
        generate_astrasim_configs_from_hw,
        ASTRA_DEBUG,
    )
    from .integration import (
        ensure_cache_file_exists,
        generate_concurrent_collectives_et,
        generate_workload_et,
        get_remote_memory_path,
        run_astrasim_analytical,
        run_cache_astrasim,
    )
    from .et_utils import (
        new_comm_node,
        new_comp_node,
        new_recv_node,
        new_send_node,
        write_et_node,
    )
    from .graph_debug import build_debug_view, debug_view_to_json
    from .executor import (
        convert_rapid_llm_graph_to_chakra_et,
        run_astra_simulation_only_onepath,
    )
except RuntimeError as exc:
    msg = str(exc)
    if "Chakra protobuf" not in msg:
        raise
    _ASTRASIM_AVAILABLE = False
    _ASTRASIM_IMPORT_ERROR = exc
except ModuleNotFoundError as exc:
    # Missing Chakra modules will surface here when ensure_chakra_available
    # has not run yet; treat them the same as the RuntimeError path.
    _ASTRASIM_AVAILABLE = False
    _ASTRASIM_IMPORT_ERROR = exc

if not _ASTRASIM_AVAILABLE:
    ASTRA_DEBUG = False

    def _raise_unavailable() -> None:
        base = "AstraSim integration is unavailable. Build the astra-sim submodule or configure execution_backend.model='analytical'."
        if _ASTRASIM_IMPORT_ERROR is not None:
            detail = f"{base} ({_ASTRASIM_IMPORT_ERROR})"
            raise RuntimeError(detail) from _ASTRASIM_IMPORT_ERROR
        raise RuntimeError(base)

    def _stub(*_args, **_kwargs):
        _raise_unavailable()

    compute_intra_inter_ib_ll_from_hw = _stub  # type: ignore
    derive_topology_from_hw = _stub  # type: ignore
    generate_astrasim_configs_from_hw = _stub  # type: ignore
    ensure_cache_file_exists = _stub  # type: ignore
    generate_concurrent_collectives_et = _stub  # type: ignore
    generate_workload_et = _stub  # type: ignore
    get_remote_memory_path = _stub  # type: ignore
    run_astrasim_analytical = _stub  # type: ignore
    run_cache_astrasim = _stub  # type: ignore
    new_comm_node = _stub  # type: ignore
    new_comp_node = _stub  # type: ignore
    new_recv_node = _stub  # type: ignore
    new_send_node = _stub  # type: ignore
    write_et_node = _stub  # type: ignore
    build_debug_view = _stub  # type: ignore
    debug_view_to_json = _stub  # type: ignore
    convert_rapid_llm_graph_to_chakra_et = _stub  # type: ignore
    run_astra_simulation_only_onepath = _stub  # type: ignore


ASTRASIM_AVAILABLE = _ASTRASIM_AVAILABLE


def is_astrasim_available() -> bool:
    """Return ``True`` when AstraSim/Chakra dependencies are importable."""

    return ASTRASIM_AVAILABLE


__all__ = [
    "ASTRA_DEBUG",
    "ASTRASIM_AVAILABLE",
    "add_chakra_to_sys_path",
    "ensure_chakra_available",
    "is_astrasim_available",
    "ensure_cache_file_exists",
    "compute_intra_inter_ib_ll_from_hw",
    "derive_topology_from_hw",
    "generate_astrasim_configs_from_hw",
    "generate_concurrent_collectives_et",
    "generate_workload_et",
    "get_remote_memory_path",
    "run_astrasim_analytical",
    "run_cache_astrasim",
    "new_comm_node",
    "new_comp_node",
    "new_recv_node",
    "new_send_node",
    "write_et_node",
    "build_debug_view",
    "debug_view_to_json",
    "convert_rapid_llm_graph_to_chakra_et",
    "run_astra_simulation_only_onepath",
]
