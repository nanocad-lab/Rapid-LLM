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

"""Bootstrap helpers for Chakra/AstraSim dependencies."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHK_PB_DIR = _REPO_ROOT / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"
_CHK_UTILS_DIR = _REPO_ROOT / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src" / "third_party" / "utils"


def _chakra_paths() -> tuple[str, str]:
    return (str(_CHK_PB_DIR), str(_CHK_UTILS_DIR))


def add_chakra_to_sys_path() -> None:
    """Ensure Chakra directories are available on ``sys.path``."""
    for path in reversed(_chakra_paths()):
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_chakra_available() -> None:
    """Verify that Chakra protobuf helpers can be imported.

    Raises:
        RuntimeError: if the Chakra protobuf modules are not importable.
    """
    add_chakra_to_sys_path()
    try:
        importlib.import_module("et_def_pb2")
        importlib.import_module("protolib")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "AstraSim Chakra protobuf dependencies are missing. "
            "Ensure the astra-sim submodule is built and Python can import its bindings."
        ) from exc


__all__ = ["ensure_chakra_available", "add_chakra_to_sys_path"]
