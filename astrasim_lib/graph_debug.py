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

"""Debug helpers for RAPID-LLM->AstraSim graph conversion."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class NodeDebugView:
    """Snapshot of a node's metadata and adjacency for inspection."""

    __slots__ = (
        "name",
        "type",
        "stage",
        "deps",
        "children",
    )

    def __init__(
        self,
        *,
        name: str,
        node_type: str,
        stage: Optional[int],
        deps: Iterable[str],
        children: Iterable[str],
    ) -> None:
        self.name = name
        self.type = node_type
        self.stage = stage
        self.deps = list(deps)
        self.children = list(children)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "stage": self.stage,
            "deps": list(self.deps),
            "children": list(self.children),
        }


def build_debug_view(
    nodes: Iterable[Any],
    *,
    stage_lookup: Optional[Dict[Any, int]] = None,
) -> List[NodeDebugView]:
    """Create a sortable debug view for a set of graph objects."""

    views: List[NodeDebugView] = []
    stage_lookup = stage_lookup or {}

    for node in nodes:
        name = getattr(node, "name", f"object_{id(node)}")
        node_type = node.__class__.__name__
        stage = stage_lookup.get(node)
        deps = [getattr(parent, "name", f"object_{id(parent)}") for parent in getattr(node, "parents", [])]
        children = [
            getattr(child, "name", f"object_{id(child)}")
            for child in getattr(node, "children", [])
        ]
        views.append(
            NodeDebugView(
                name=name,
                node_type=node_type,
                stage=stage,
                deps=deps,
                children=children,
            )
        )

    views.sort(key=lambda v: v.name)
    return views


def debug_view_to_json(views: Iterable[NodeDebugView]) -> List[Dict[str, Any]]:
    """Convert a list of ``NodeDebugView`` to JSON-serializable form."""

    return [view.to_dict() for view in views]
