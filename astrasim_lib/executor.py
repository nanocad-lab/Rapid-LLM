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

"""AstraSim execution helpers used by RAPID-LLM's comparison workflow.

This module converts RAPID-LLM graphs (with communication sizes) to AstraSim Chakra ET
format and executes AstraSim simulation for comparison with RAPID-LLM analytical timing.

Non-mainlined test functionality - designed to be easily removable.

Note: AstraSim's scheduler prioritizes lower node IDs when multiple GPU comm nodes are
ready. To keep 1-byte pipeline control sends from being starved behind large collectives,
we renumber control send IDs just before writing each ET so they occupy the lowest
indices. See `_RankTrace._renumber_control_priority` for details.
"""

import itertools
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict, deque
from itertools import count as _it_count
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

sys.setrecursionlimit(100000)

from graphviz import Digraph
import util
from timing_model import CollectiveType

from .config_generation import ASTRA_DEBUG, generate_astrasim_configs_from_hw
from . import gmap
from .et_utils import (
    chakra_decode,
    chakra_encode,
    chakra_open,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
    write_et_node,
)
from .integration import get_remote_memory_path, run_astrasim_analytical, run_cache_astrasim
from .layout_utils import derive_axes_filter
from simulate_train_graph import visualize_graph
from util import relpath_display


def _env_truthy(name: str) -> bool:
    """Return ``True`` when environment variable ``name`` is set to a truthy value."""

    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


def _clean_astrasim_artifacts(directory: str) -> None:
    """Remove generated AstraSim artifacts under ``directory`` if they exist."""

    try:
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if entry.startswith("llm_graph.") and entry.endswith(".et"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry == "manifest.json":
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry == "comm_groups.json" or entry.startswith("comm_groups_"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry.startswith("system_native_collectives"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry.startswith("network_analytical_") and entry.endswith(".yml"):
                try:
                    os.remove(path)
                except OSError:
                    pass
        workload_dir = os.path.join(directory, "workload")
        if os.path.isdir(workload_dir):
            shutil.rmtree(workload_dir)
    except FileNotFoundError:
        pass

_TP_GROUP_BASE_ID = 1000
_LAST_TP_GROUPS: Optional[Dict[str, List[int]]] = None
_LAST_STAGE_GROUPS: Optional[List[List[int]]] = None
_TP_LABEL_DP_TO_ID: Dict[Tuple[str, int], str] = {}
_TP_MEMBERS_TO_GID: Dict[Tuple[int, ...], str] = {}


def _extract_axis_layout(rank_layout: Optional[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    if not isinstance(rank_layout, dict):
        return [], {}, {}
    axis_order = list(rank_layout.get("axis_order", []))
    raw_sizes = rank_layout.get("axis_sizes", {})
    if isinstance(axis_order, tuple):
        axis_order = list(axis_order)
    axis_sizes: Dict[str, int] = {}
    if isinstance(raw_sizes, dict):
        for key, value in raw_sizes.items():
            try:
                axis_sizes[str(key)] = max(1, int(value))
            except Exception:
                axis_sizes[str(key)] = 1
    raw_strides = rank_layout.get("axis_strides", {})
    axis_strides: Dict[str, int] = {}
    if isinstance(raw_strides, dict):
        for key, value in raw_strides.items():
            try:
                axis_strides[str(key)] = int(value)
            except Exception:
                axis_strides[str(key)] = 0
    else:
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes.get(axis, 1)
    return axis_order, axis_sizes, axis_strides


def _remap_stages_for_mapping(
    *,
    permutation: Sequence[int],
    collector: Optional[gmap.GMapCollector],
    stage_ids: List[int],
    stage_to_ranks: Dict[int, List[int]],
    stage_index: Dict[int, int],
    rank_traces: Dict[int, "_RankTrace"],
    rank_meta: Dict[int, Dict[str, int]],
    dp_count: int,
    num_stages: int,
    output_dir: str,
) -> Tuple[List[int], Dict[int, int]]:
    """Reorder stages so AstraSim ranks match the SCOTCH permutation.

    We apply the permutation *before* ET emission so the rest of the converter
    remains unaware of the remap. Only stage ordering / rank bookkeeping is
    touched here, which keeps the change localized and avoids renaming files
    after the fact.
    """

    if collector is None or not permutation:
        return stage_ids, stage_index

    if len(permutation) != collector.vertex_count:
        raise ValueError("Permutation length does not match first-dimension vertex count.")

    def _sort_key(stage: int) -> Tuple[int, ...]:
        coords = collector.stage_axis_coords.get(stage, {})
        higher = tuple(int(coords.get(axis, 0)) for axis in collector.replication_axes)
        local_idx = collector.local_index_for_stage(stage)
        target_idx = permutation[local_idx]
        return higher + (target_idx,)

    ordered_stages = sorted(stage_ids, key=_sort_key)
    new_stage_index = {stage: idx for idx, stage in enumerate(ordered_stages)}

    new_stage_to_ranks: Dict[int, List[int]] = {stage: [0] * dp_count for stage in ordered_stages}
    new_rank_traces: Dict[int, _RankTrace] = {}
    new_rank_meta: Dict[int, Dict[str, int]] = {}

    for stage in stage_ids:
        for dp_idx in range(dp_count):
            old_rank = stage_to_ranks[stage][dp_idx]
            new_idx = new_stage_index[stage]
            new_rank = dp_idx * num_stages + new_idx
            trace = rank_traces[old_rank]
            trace.rank = new_rank
            trace.path = os.path.join(output_dir, f"llm_graph.{new_rank}.et")
            new_rank_traces[new_rank] = trace
            new_stage_to_ranks[stage][dp_idx] = new_rank
            new_rank_meta[new_rank] = {"stage": stage, "dp": dp_idx}

    stage_to_ranks.clear()
    stage_to_ranks.update(new_stage_to_ranks)
    rank_traces.clear()
    rank_traces.update(new_rank_traces)
    rank_meta.clear()
    rank_meta.update(new_rank_meta)

    stage_ids[:] = ordered_stages
    stage_index = new_stage_index
    return ordered_stages, stage_index


def _compute_stage_axis_coords(
    stage_ids: Sequence[int],
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
) -> Dict[int, Dict[str, int]]:
    coords: Dict[int, Dict[str, int]] = {}
    if not axis_order:
        return coords
    for stage in stage_ids:
        remaining = int(stage)
        stage_coords: Dict[str, int] = {}
        for axis in axis_order:
            size = max(1, int(axis_sizes.get(axis, 1)))
            if size > 1:
                stage_coords[axis] = remaining % size
                remaining //= size
            else:
                stage_coords[axis] = 0
        coords[stage] = stage_coords
    return coords


def _build_axis_groups(
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
    stage_axis_coords: Mapping[int, Dict[str, int]],
    stage_to_ranks: Mapping[int, List[int]],
) -> Dict[str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]]:
    axis_groups: Dict[
        str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]
    ] = {}
    if not axis_order:
        return axis_groups
    for axis in axis_order:
        size = max(1, int(axis_sizes.get(axis, 1)))
        if size <= 1:
            continue
        groups: Dict[
            Tuple[int, Tuple[Tuple[str, int], ...]], List[int]
        ] = defaultdict(list)
        for stage, coords in stage_axis_coords.items():
            key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax != axis)
            ranks = stage_to_ranks.get(stage, [])
            for dp_idx, rank in enumerate(ranks):
                groups[(dp_idx, key_base)].append(rank)
        axis_groups[axis] = {
            key: sorted(set(values)) for key, values in groups.items() if values
        }
    return axis_groups


def _assign_collective_labels(
    tp_collective_groups: Mapping[str, List[Any]],
    collective_info: Mapping[Any, Dict[str, Any]],
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
    stage_axis_coords: Mapping[int, Dict[str, int]],
    axis_groups: Mapping[str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]],
    stage_to_ranks: Mapping[int, List[int]],
    dp_count: int,
) -> Tuple[Dict[Any, str], Dict[str, List[Any]], Dict[str, Set[Tuple[str, int, Tuple[int, ...]]]]]:
    tp_collective_labels: Dict[Any, str] = {}
    label_to_edges: Dict[str, List[Any]] = {}
    label_group_tokens: Dict[str, Set[Tuple[str, int, Tuple[int, ...]]]] = defaultdict(set)
    group_key_to_label: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    label_suffix_counter: Dict[str, int] = defaultdict(int)

    def _composite_members(stage: int, dp_idx: int, axes: Sequence[str]) -> List[int]:
        coords = stage_axis_coords.get(stage, {})
        key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax not in axes)
        members: List[int] = []
        for stage_id, stage_coords in stage_axis_coords.items():
            stage_key = tuple((ax, stage_coords.get(ax, 0)) for ax in axis_order if ax not in axes)
            if stage_key != key_base:
                continue
            ranks = stage_to_ranks.get(stage_id, [])
            if dp_idx < len(ranks):
                members.append(ranks[dp_idx])
        return sorted(set(members))

    for base_name, edges in tp_collective_groups.items():
        if not edges:
            continue
        for edge in sorted(edges, key=lambda e: getattr(e, "op_id", 0)):
            info = collective_info[edge]
            axis = str(info.get("interconnect_type", "tp")).lower()
            stage = info["stage"]
            coords = stage_axis_coords.get(stage, {})
            key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax != axis)
            participants = int(info.get("participants", 0) or 0)

            composite_axes: Optional[Tuple[str, ...]] = None
            if axis == "ep":
                tp_size = max(1, int(axis_sizes.get("tp", 1)))
                ep_size = max(1, int(axis_sizes.get("ep", 1)))
                if tp_size > 1 and participants == tp_size * ep_size:
                    composite_axes = ("tp", "ep")

            members_per_dp: List[Tuple[int, Tuple[int, ...]]] = []
            for dp_idx in range(dp_count):
                members = None
                if composite_axes is not None:
                    members = _composite_members(stage, dp_idx, composite_axes)
                else:
                    axis_map = axis_groups.get(axis)
                    if axis_map is not None:
                        members = axis_map.get((dp_idx, key_base))
                if not members:
                    ranks = stage_to_ranks.get(stage, [])
                    if dp_idx < len(ranks):
                        members = [ranks[dp_idx]]
                if members:
                    members_tuple = tuple(sorted(members))
                    members_per_dp.append((dp_idx, members_tuple))

            if not members_per_dp:
                continue

            primary_members = members_per_dp[0][1]
            label_key = (base_name, primary_members)
            label = group_key_to_label.get(label_key)
            if label is None:
                suffix = label_suffix_counter[base_name]
                label = base_name if suffix == 0 else f"{base_name}_{suffix}"
                label_suffix_counter[base_name] += 1
                group_key_to_label[label_key] = label

            tp_collective_labels[edge] = label
            label_to_edges.setdefault(label, []).append(edge)
            for dp_idx, members_tuple in members_per_dp:
                label_group_tokens[label].add((axis, dp_idx, members_tuple))

    return tp_collective_labels, label_to_edges, label_group_tokens


class _RankTrace:
    """Helper to build AstraSim ET for a single hardware rank."""

    def __init__(self, hw_id: int, rank: int, path: str) -> None:
        self.hw_id = hw_id
        self.rank = rank
        self.path = path
        self.nodes: List[pb.Node] = []

    @property
    def next_id(self) -> int:
        return len(self.nodes)

    def append_node(self, node: pb.Node) -> None:
        """Append ``node`` to this trace without mutating its id."""

        self.nodes.append(node)

    def _renumber_control_priority(self) -> None:
        """Renumber nodes so pipeline control sends receive the lowest IDs.

        AstraSim schedules multiple ready GPU comm operations strictly by
        ascending node ID. Our pipeline control sends are single-byte messages
        that unblock downstream collectives; if they keep their "natural"
        IDs (created late in the build), they lose the race to the heavy
        collectives and we deadlock.  This routine rewrites IDs and
        dependencies so that every ``COMM_SEND_NODE`` whose name ends with
        "_send_control" occupies the lowest indices, while all other nodes
        retain their relative order above that block.
        """

        if not self.nodes:
            return

        def _is_control_send(node: pb.Node) -> bool:
            return (
                node.type == pb.COMM_SEND_NODE
                and (node.name or "").endswith("_send_control")
            )
        def _is_control_recv(node: pb.Node) -> bool:
            return (
                node.type == pb.COMM_RECV_NODE
                and (node.name or "").endswith("_recv_control")
            )

        control_nodes: List[pb.Node] = []
        regular_nodes: List[pb.Node] = []
        for node in self.nodes:
            if _is_control_send(node):
                control_nodes.append(node)
            elif _is_control_recv(node):
                control_nodes.append(node)
            else:
                regular_nodes.append(node)

        if not control_nodes:
            return

        id_map: Dict[int, int] = {}
        new_order: List[pb.Node] = []

        for node in control_nodes:
            new_id = len(new_order)
            id_map[int(node.id)] = new_id
            node.id = new_id
            new_order.append(node)

        for node in regular_nodes:
            new_id = len(new_order)
            id_map[int(node.id)] = new_id
            node.id = new_id
            new_order.append(node)

        for node in new_order:
            remapped: List[int] = []
            for dep in node.ctrl_deps:
                remapped.append(id_map.get(int(dep), int(dep)))
            node.ctrl_deps[:] = remapped

        self.nodes = new_order

    def close(self) -> None:
        self._renumber_control_priority()
        with open(self.path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            for node in self.nodes:
                write_et_node(fh, node)


def get_collective_type(comm_type: CollectiveType) -> int:
    """Map RAPID-LLM comm types to AstraSim protobuf enums."""
    if comm_type is None:
        raise ValueError("Collective comm_type is required")
    if not isinstance(comm_type, CollectiveType):
        raise TypeError(f"comm_type must be CollectiveType (got {type(comm_type).__name__})")
    if comm_type == CollectiveType.PIPELINE:
        raise ValueError("Pipeline comm_type should not be mapped to a collective enum")
    mapping = {
        CollectiveType.ALL_REDUCE: pb.ALL_REDUCE,
        CollectiveType.ALL_GATHER: pb.ALL_GATHER,
        CollectiveType.REDUCE_SCATTER: pb.REDUCE_SCATTER,
        CollectiveType.ALL_TO_ALL: pb.ALL_TO_ALL,
    }
    return mapping[comm_type]


def _attr_to_dict(node: pb.Node) -> Dict[str, Any]:
    attr_map: Dict[str, Any] = {}
    for attr in node.attr:
        name = attr.name or f"attr_{len(attr_map)}"
        which = attr.WhichOneof("value")
        if not which:
            continue
        raw = getattr(attr, which)
        if hasattr(raw, "values"):
            attr_map[name] = list(raw.values)
        else:
            attr_map[name] = raw
    return attr_map


def _visualize_et_files(et_paths: List[str]) -> None:
    """Render dot graphs for Chakra ET files to aid debugging when enabled."""

    if not et_paths:
        return

    # prune et_paths to only include the first 10
    et_paths = et_paths[:20]

    def _render_et_file(et_path: str) -> None:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for visualization: {exc}")
            return

        meta = pb.GlobalMetadata()
        if not chakra_decode(fh, meta):
            print(f"[WARN] {et_path} does not contain GlobalMetadata; skipping graph output")
            fh.close()
            return

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        dot = Digraph(comment=os.path.basename(et_path), format="svg",graph_attr={"rankdir": "TB", "fontsize": "10"})

        id_to_node = {int(node.id): node for node in nodes}
        type_color = {
            pb.COMP_NODE: "lightblue",
            pb.COMM_COLL_NODE: "palegreen",
            pb.COMM_SEND_NODE: "khaki",
            pb.COMM_RECV_NODE: "lightsalmon",
        }

        for node in nodes:
            attr_map = _attr_to_dict(node)
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            label_lines = [node.name or f"node_{node.id}", f"id={node.id}", node_type]
            if node.duration_micros:
                label_lines.append(f"dur={node.duration_micros}us")
            if "comm_type" in attr_map:
                try:
                    comm_name = pb.CollectiveCommType.Name(int(attr_map["comm_type"]))
                except ValueError:
                    comm_name = str(attr_map["comm_type"])
                label_lines.append(f"comm={comm_name}")
            if "comm_dst" in attr_map:
                label_lines.append(f"dst={attr_map['comm_dst']}")
            if "comm_src" in attr_map:
                label_lines.append(f"src={attr_map['comm_src']}")
            if "comm_tag" in attr_map:
                label_lines.append(f"tag={attr_map['comm_tag']}")
            if "comm_size" in attr_map:
                label_lines.append(f"bytes={attr_map['comm_size']}")

            color = type_color.get(node.type, "white")
            node_id = str(node.id)
            dot.node(node_id, label="\n".join(label_lines), style="filled", fillcolor=color, shape="box")

        for node in nodes:
            for dep in node.ctrl_deps:
                if dep in id_to_node:
                    dot.edge(str(dep), str(node.id))

        dir_name, base_name = os.path.split(et_path)
        viz_base = base_name + ".viz"
        try:
            output_path = dot.render(viz_base, directory=dir_name or None, format="svg", cleanup=True)
            final_svg = et_path + ".svg"
            if output_path and output_path != final_svg:
                try:
                    os.replace(output_path, final_svg)
                except FileNotFoundError:
                    # Some graphviz versions return path without creating file when empty graph
                    pass
        except Exception as exc:
            print(f"[WARN] Failed to render Graphviz graph for {et_path}: {exc}")

    if not et_paths:
        return

    if len(et_paths) == 1:
        et_path = et_paths[0]
        display_path = relpath_display(f"{et_path}.svg")
        message = f" | ET Graph saved to {display_path}"
        util.graphviz_submit(
            f"et:{os.path.basename(et_path)}",
            _render_et_file,
            et_path,
            print_message=message,
        )
        return

    first_path = et_paths[0]
    display_first = relpath_display(f"{first_path}.svg")
    summary_message = f" | {len(et_paths)} ET graphs saved to {display_first} (..)"
    util.graphviz_submit(
        f"et:{os.path.basename(first_path)}",
        _render_et_file,
        first_path,
        print_message=summary_message,
    )
    for et_path in et_paths[1:]:
        util.graphviz_submit(
            f"et:{os.path.basename(et_path)}",
            _render_et_file,
            et_path,
            print_message=None,
        )



def _dump_et_text(et_paths: List[str]) -> None:
    if not et_paths:
        return

    for et_path in et_paths:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for text dump: {exc}")
            continue

        meta = pb.GlobalMetadata()
        has_meta = chakra_decode(fh, meta)

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        id_to_node = {int(node.id): node for node in nodes}

        lines: List[str] = []
        lines.append(f"ET file: {os.path.basename(et_path)}")
        if has_meta:
            lines.append(f"GlobalMetadata: version={meta.version}")
        lines.append(f"Nodes: {len(nodes)}")
        lines.append("")

        def fmt_attr_map(node: pb.Node) -> str:
            items: List[str] = []
            for attr in node.attr:
                name = attr.name or "unnamed"
                which = attr.WhichOneof("value")
                if not which:
                    continue
                raw = getattr(attr, which)
                if hasattr(raw, "values"):
                    val = list(raw.values)
                else:
                    val = raw
                items.append(f"{name}={val}")
            return ", ".join(items)

        lines.append("Nodes detail:")
        for node in nodes:
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            attr_str = fmt_attr_map(node)
            if node.type == pb.COMP_NODE:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} dur_us={node.duration_micros} attrs=[{attr_str}]"
                )
            else:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} attrs=[{attr_str}]"
                )
        lines.append("")

        lines.append("Edges (ctrl_deps):")
        for node in nodes:
            if not node.ctrl_deps:
                continue
            deps_str = ", ".join(str(int(d)) for d in node.ctrl_deps if int(d) in id_to_node)
            lines.append(f"- {node.id} <- [{deps_str}]")

        out_path = et_path + ".txt"
        try:
            with open(out_path, "w") as outf:
                outf.write("\n".join(lines) + "\n")
            # print(f"[AstraSim] Saved ET text dump to {out_path}")
        except OSError as exc:
            print(f"[WARN] Failed to write ET text dump for {et_path}: {exc}")


def _write_comm_groups_json(base_output_dir: str, dp_count: int, rank_ids: List[int]) -> Optional[str]:
    try:
        dp = int(dp_count)
    except Exception:
        dp = 1
    if not rank_ids:
        return None
    rank_ids = sorted(rank_ids)
    total_ranks = len(rank_ids)

    groups: Dict[str, List[int]] = {}

    # Emit DP stage groups only when dp > 1
    if dp > 1:
        if total_ranks % dp != 0:
            raise ValueError(f"Cannot partition {total_ranks} ranks into {dp} stages evenly")
        # dp-major mapping: ranks are ordered by dp first, then stage
        num_stages = total_ranks // dp
        for stage_idx in range(num_stages):
            group = []
            for dp_idx in range(dp):
                rank = dp_idx * num_stages + stage_idx
                group.append(rank)
            # AstraSim requires communicator group IDs > 0
            groups[str(stage_idx + 1)] = group

    # Merge in any TP groups computed during conversion (present even if dp == 1)
    global _LAST_TP_GROUPS
    if _LAST_TP_GROUPS:
        for gid, members in _LAST_TP_GROUPS.items():
            groups[str(gid)] = list(sorted(members))

    if not groups:
        return None

    os.makedirs(base_output_dir, exist_ok=True)
    path = os.path.join(base_output_dir, "comm_groups.json")
    try:
        with open(path, "w") as f:
            json.dump(groups, f, indent=2)
        return path
    except OSError as exc:
        print(f"[WARN] Failed to write comm_groups.json: {exc}")
        return None

def convert_rapid_llm_graph_to_chakra_et(
    graph_root,
    dp_size: int,
    output_dir: str,
) -> Tuple[str, List[int], str]:
    """Convert RAPID-LLM graph to AstraSim ET format by scheduling per stage and DP rank.
    This means converting the single RAPID-LLM DAG into a set of different AstraSim ET files.

    INPUTS: ``graph_root`` is the RAPID-LLM DAG (after flattening). ``dp_size`` is the
    data-parallel replication degree. ``output_dir`` is where ET traces + metadata
    should be written.

    OUTPUTS: returns ``(et_prefix, rank_ids, manifest_path)``. ``et_prefix`` is the
    file prefix for ``llm_graph.<rank>.et``. ``rank_ids`` is the list of produced rank
    IDs, and ``manifest_path`` points to the per-rank summary JSON.

    CONCEPTS: a *stage* is a unique ``hw_id`` (flattened hardware coordinate) of the
    compute graph. Each (stage, DP replica) pair becomes an AstraSim rank. The function
    recovers tensor/context/pipeline axes, schedules tasks per stage and per DP rank, and
    emits Chakra ET traces with matching communicator metadata so AstraSim can replay the
    workload deterministically.
    
    This function is very complicated. Don't touch it unless you know what you are doing.
    Prefer changes *anywhere* else in the codebase if possible. I have tried to document it
    as much as possible, but it definitely needs further cleaning and breaking up. This is TODO."""

    os.makedirs(output_dir, exist_ok=True)
    debug_enabled = _env_truthy("RAPID_DUMP_CONVERTER_DEBUG")

    # Step 1: Traverse the graph once to snapshot every object reachable from the
    # provided root(s). Collecting upfront avoids mutating the graph while we
    # analyse dependencies.
    def collect_objects(root) -> List[Any]:
        visited: Set[int] = set()
        ordered: List[Any] = []

        def dfs(obj: Any) -> None:
            if id(obj) in visited:
                return
            visited.add(id(obj))
            ordered.append(obj)
            for child in getattr(obj, "children", []):
                dfs(child)

        if isinstance(root, (list, tuple)):
            for item in root:
                dfs(item)
        else:
            dfs(root)
        return ordered

    all_objects = collect_objects(graph_root)
    compute_nodes = [
        obj
        for obj in all_objects
        if getattr(obj, "hw_id", None) is not None
        and obj.hw_id >= 0
        and not getattr(obj, "flatten_placeholder", False)
    ]
    if not compute_nodes:
        raise ValueError("RAPID-LLM graph did not expose any executable compute nodes (hw_id >= 0).")

    # Step 2: Each unique ``hw_id`` defines a logical stage (one flattened hardware
    # coordinate). We build a DP-major rank ordering so AstraSim sees one rank per
    # (stage, data-parallel replica) pair.
    stage_ids = sorted({node.hw_id for node in compute_nodes})
    dp_count = max(int(dp_size) if dp_size else 1, 1)

    stage_index = {stage: idx for idx, stage in enumerate(stage_ids)}
    stage_to_ranks: Dict[int, List[int]] = {stage: [] for stage in stage_ids}
    rank_meta: Dict[int, Dict[str, int]] = {}
    rank_traces: Dict[int, _RankTrace] = {}
    num_stages = len(stage_ids)

    # dp-major mapping: rank = dp_idx * num_stages + stage_index
    for dp_idx in range(dp_count):
        for stage in stage_ids:
            rank = dp_idx * num_stages + stage_index[stage]
            rank_meta[rank] = {"stage": stage, "dp": dp_idx}
            path = f"{output_dir}/llm_graph.{rank}.et"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rank_traces[rank] = _RankTrace(stage, rank, path)
            stage_to_ranks[stage].append(rank)

    def rank_for(stage: int, dp_idx: int) -> int:
        return stage_to_ranks[stage][dp_idx]

    def _compute_duration_seconds(task: Any, dp_idx: int) -> float:
        """Return the per-DP duration (in seconds) for ``task``."""
        profile = getattr(task, "duration_profile", None)
        if profile:
            if len(profile) != dp_count:
                raise ValueError(
                    f"Duration profile for node '{getattr(task, 'name', '<unnamed>')}' "
                    f"has length {len(profile)} but dp_count={dp_count}."
                )
            return float(profile[dp_idx])
        duration_sec = getattr(task, "duration", 0.0) or 0.0
        return float(duration_sec)

    # Step 3: Recover the multi-dimensional axis layout (tp/cp/pp, etc.) so that
    # communicator membership can be reconstructed deterministically later on.
    rank_layout = getattr(graph_root, "_astrasim_rank_layout", None)
    axis_order, axis_sizes, axis_strides = _extract_axis_layout(rank_layout)
    stage_axis_coords = _compute_stage_axis_coords(stage_ids, axis_order, axis_sizes)
    axis_groups = _build_axis_groups(axis_order, axis_sizes, stage_axis_coords, stage_to_ranks)
    if hasattr(graph_root, "_optimize_2dmap"):
        optimize_cfg = getattr(graph_root, "_optimize_2dmap")
    else:
        optimize_cfg = None
    collector = gmap.begin_collection(optimize_cfg, axis_sizes, stage_axis_coords, output_dir)

    # Step 4: Attribute each comm edge to the stage that owns it. This lets the
    # dependency walkers treat collectives the same way they treat compute nodes.
    edge_stage: Dict[Any, int] = {}
    for obj in all_objects:
        comm = getattr(obj, "comm_type", None)
        if comm and comm != CollectiveType.PIPELINE:
            stage = None
            parent = None
            if stage is None:
                local_hw = getattr(obj, "local_hw_id", None)
                if local_hw is not None:
                    stage = local_hw
                else:
                    for candidate in getattr(obj, "parents", []):
                        if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                            stage = candidate.hw_id
                            parent = candidate
                            break
                    if stage is None:
                        for candidate in getattr(obj, "children", []):
                            if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                                stage = candidate.hw_id
                                break


            children = getattr(obj, "children", [])
            parents = getattr(obj, "parents", [])
            if len(children) == 0 and len(parents) == 0:
                raise ValueError(f"Object {obj.name} has no children or parents")

            if stage is None:
                # print children and parents
                children = getattr(obj, "children", [])
                parents = getattr(obj, "parents", [])
                for child in children:
                    print(f"Child: {child.name}")
                for parent in parents:
                    print(f"Parent: {parent.name}")

                raise ValueError(f"Stage not found for object {obj.name}")
            edge_stage[obj] = stage

    pipeline_edge_map: Dict[Tuple[Any, Any], Any] = {}

    def _record_pipeline_edges_for_first_dim(collector_obj: gmap.GMapCollector) -> None:
        if not collector_obj.include_pipeline:
            return
        for obj in all_objects:
            if getattr(obj, "comm_type", None) != CollectiveType.PIPELINE:
                continue
            parents = getattr(obj, "parents", [])
            src_stage = None
            for parent in parents:
                if hasattr(parent, "hw_id") and parent.hw_id is not None and int(parent.hw_id) >= 0:
                    src_stage = int(parent.hw_id)
                    break
            if src_stage is None:
                raise ValueError("Unable to resolve source stage for pipeline edge during gmap collection.")
            children = getattr(obj, "children", [])
            dst_stage = None
            for child in children:
                if hasattr(child, "hw_id") and child.hw_id is not None and int(child.hw_id) >= 0:
                    dst_stage = int(child.hw_id)
                    break
            if dst_stage is None:
                raise ValueError("Unable to resolve destination stage for pipeline edge during gmap collection.")
            if src_stage == dst_stage:
                continue
            if not hasattr(obj, "comm_size_bytes"):
                raise ValueError("Pipeline edge is missing comm_size_bytes for gmap collection.")
            size_bytes = int(getattr(obj, "comm_size_bytes"))
            collector_obj.record_pipeline(src_stage=src_stage, dst_stage=dst_stage, size_bytes=size_bytes)

    # Step 5a: Walk backwards from a compute node to classify dependencies into
    # stage-local, cross-stage (pipeline), or collective inputs.
    def analyze_for_compute(node: Any) -> Tuple[Set[Any], Set[Any], Set[Any]]:
        stage = node.hw_id
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        collective_deps: Set[Any] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(node, "parents", [])]
        debug_en = False
        
        if debug_en:
            print(f"\n=== DEBUG: analyze_for_compute for node '{node.name}' ===")
            print(f"Node stage: {stage}")
            print(f"Initial parents: {[getattr(p, 'name', str(id(p))) for p in getattr(node, 'parents', [])]}")
            print(f"Initial stack size: {len(stack)}")
            print()
        
        iteration = 0
        while stack:
            iteration += 1
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            
            if debug_en:
                print(f"--- Iteration {iteration} ---")
                print(f"Processing: {getattr(cur, 'name', str(id(cur)))}")
                print(f"via_collective: {via_collective}")
                print(f"key: {key}")
                print(f"Stack size before pop: {len(stack) + 1}")
            
            if key in visited:
                if debug_en:
                    print(f"Already visited, skipping")
                    print()
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if debug_en:
                print(f"hw_id: {hw}")
            
            if hw is not None and hw >= 0:
                if debug_en:
                    print(f"Node has valid hw_id: {hw}")
                    print(f"Comparing with stage: {stage}")
                
                if hw == stage:
                    if not via_collective:
                        if debug_en:
                            print(f"Adding to stage_deps: {getattr(cur, 'name', str(id(cur)))}")
                        stage_deps.add(cur)
                    else:
                        if debug_en:
                            print(f"Same stage but via_collective=True, not adding to stage_deps")
                else:
                    if debug_en:
                        print(f"Different stage, adding to pipeline_deps: {getattr(cur, 'name', str(id(cur)))}")
                    pipeline_deps.add(cur)
                    pipeline_edge_map.setdefault((cur, node), None)
                
                if debug_en:
                    print(f"Continuing to next iteration")
                    print()
                continue

            comm = getattr(cur, "comm_type", None)
            if debug_en:
                print(f"comm_type: {comm}")
            
            if comm:
                if comm == CollectiveType.PIPELINE:
                    if debug_en:
                        print(f"Processing pipeline comm")

                    # compute sources vvv
                    srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                    # also collect dp-collective sources (they have local_hw_id)
                    coll_srcs = [p for p in cur.parents if getattr(p, "local_hw_id", None) is not None and p.local_hw_id >= 0]
                    if debug_en:
                        print(f"Pipeline sources: {[getattr(s, 'name', str(id(s))) for s in srcs]}")
                        print(f"DP-collective sources: {[getattr(s, 'name', str(id(s))) for s in coll_srcs]}")
                    
                    if srcs:
                        for src in srcs:
                            if debug_en:
                                print(f"  Processing src: {getattr(src, 'name', str(id(src)))} (hw_id: {src.hw_id})")
                            
                            if src.hw_id == stage:
                                if not via_collective:
                                    if debug_en:
                                        print(f"    Adding pipeline src to stage_deps: {getattr(src, 'name', str(id(src)))}")
                                    stage_deps.add(src)
                                else:
                                    if debug_en:
                                        print(f"    Same stage pipeline src but via_collective=True, not adding")
                            else:
                                if debug_en:
                                    print(f"    Adding pipeline src to pipeline_deps: {getattr(src, 'name', str(id(src)))}")
                                pipeline_deps.add(src)
                                pipeline_edge_map[(src, node)] = cur

                    if coll_srcs:
                        for src in coll_srcs:
                            if debug_en:
                                print(f"  Processing coll_src: {getattr(src, 'name', str(id(src)))} (local_hw_id: {src.local_hw_id})")
                            if src.local_hw_id == stage:
                                if not via_collective:
                                    if debug_en:
                                        print(f"    Adding dp-collective src to stage_deps: {getattr(src, 'name', str(id(src)))}")
                                    collective_deps.add(src)
                            else:
                                # for now we do not support cross-pipeline dp collective deps so ignore.
                                if debug_en:
                                    print(f"    Ignoring cross-pipeline dp collective src: {getattr(src, 'name', str(id(src)))}")
                                
                    if debug_en:
                        print(f"Pipeline processing complete, continuing")
                        print()
                    continue
                else:
                    if debug_en:
                        print(f"Processing non-pipeline collective: {comm}")
                        print(f"Adding to collective_deps: {getattr(cur, 'name', str(id(cur)))}")
                    collective_deps.add(cur)
                    parents_to_add = getattr(cur, "parents", [])
                    if debug_en:
                        print(f"Adding collective parents to stack: {[getattr(p, 'name', str(id(p))) for p in parents_to_add]}")
                    for parent in parents_to_add:
                        stack.append((parent, True))
                    
                    if debug_en:
                        print(f"Collective processing complete, continuing")
                        print()
                    continue

            parents_to_add = getattr(cur, "parents", [])
            if debug_en:
                print(f"No comm_type or hw_id, adding regular parents to stack: {[getattr(p, 'name', str(id(p))) for p in parents_to_add]}")
            for parent in parents_to_add:
                stack.append((parent, via_collective))
            
            if debug_en:
                print(f"Stack size after adding parents: {len(stack)}")
                print()

        if debug_en:
            print(f"=== FINAL RESULTS ===")
            print(f"stage_deps before discard: {[getattr(d, 'name', str(id(d))) for d in stage_deps]}")
            
        stage_deps.discard(node)
        
        if debug_en:
            print(f"stage_deps after discard: {[getattr(d, 'name', str(id(d))) for d in stage_deps]}")
            print(f"pipeline_deps: {[getattr(d, 'name', str(id(d))) for d in pipeline_deps]}")
            print(f"collective_deps: {[getattr(d, 'name', str(id(d))) for d in collective_deps]}")
            print(f"pipeline_edge_map entries added: {len(pipeline_edge_map)}")
            for (src, dst), edge in pipeline_edge_map.items():
                if dst == node:
                    print(f"  {getattr(src, 'name', str(id(src)))} -> {getattr(dst, 'name', str(id(dst)))} via {getattr(edge, 'name', str(id(edge))) if edge else 'None'}")
            print(f"=== END DEBUG ===")
        
        return stage_deps, pipeline_deps, collective_deps

    # Step 5b: Same classification for collective edges (they often use local_hw_id
    # instead of hw_id, so we check both).
    def analyze_for_collective(edge: Any) -> Tuple[Set[Any], Set[Any], Set[Any]]:
        stage = edge_stage[edge]
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        collective_deps: Set[Any] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(edge, "parents", [])]

        while stack:
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            if key in visited:
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if hw is None:
                hw = getattr(cur, "local_hw_id", None)
            if hw is not None and hw >= 0:
                if hw == stage:
                    if not via_collective:
                        stage_deps.add(cur)
                else:
                    pipeline_deps.add(cur)
                    pipeline_edge_map[(cur, edge)] = cur
                continue

            comm = getattr(cur, "comm_type", None)
            if comm == CollectiveType.PIPELINE:
                srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                coll_srcs = [p for p in cur.parents if getattr(p, "local_hw_id", None) is not None and getattr(p, "local_hw_id") >= 0]
                if srcs:
                    for src in srcs:
                        if src.hw_id == stage:
                            if not via_collective:
                                stage_deps.add(src)
                        else:
                            pipeline_deps.add(src)
                            pipeline_edge_map[(src, edge)] = cur
                if coll_srcs:
                    for src in coll_srcs:
                        if src.local_hw_id == stage:
                            if not via_collective:
                                collective_deps.add(src)
                        # else:
                        #     pipeline_deps.add(src)
                        #     pipeline_edge_map[(src, edge)] = cur
                continue
            elif comm:
                collective_deps.add(cur)
                for parent in getattr(cur, "parents", []):
                    stack.append((parent, True))
                continue

            for parent in getattr(cur, "parents", []):
                stack.append((parent, via_collective))

        return stage_deps, pipeline_deps, collective_deps

    # Aggregate the analysis results so later stages can fetch dependencies by node.
    compute_info: Dict[Any, Dict[str, Any]] = {}
    for node in compute_nodes:
        stage_deps, pipeline_deps, collective_deps = analyze_for_compute(node)
        compute_info[node] = {
            "stage": node.hw_id,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "collective_deps": collective_deps,
            "name": node.name,
        }

    collective_info: Dict[Any, Dict[str, Any]] = {}
    for edge, stage in edge_stage.items():
        stage_deps, pipeline_deps, collective_deps = analyze_for_collective(edge)
        entry = {
            "stage": stage,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "collective_deps": collective_deps,
            "size": int(getattr(edge, "comm_size_bytes", 0)),
            "comm_type": get_collective_type(edge.comm_type),
            "name": edge.name,
            "interconnect_type": getattr(edge, "comm_interconnect_type", None),
            "participants": int(getattr(edge, "participants", 0) or 0),
        }
        collective_info[edge] = entry
        if collector:
            axis_name = entry["interconnect_type"]
            if axis_name:
                axis_norm = str(axis_name).strip().lower()
                if axis_norm in collector.subset_axes:
                    should_double = entry["comm_type"] in (pb.ALL_REDUCE, pb.ALL_TO_ALL)
                    collector.record_collective(
                        axis=axis_norm,
                        stage_id=stage,
                        size_bytes=entry["size"],
                        participant_count=entry["participants"],
                        double_weight=should_double,
                    )

    if debug_enabled:
        print("[ConverterDebug] === COMPUTE INFO (stage deps / pipeline deps / collectives) ===")
        for node, info in compute_info.items():
            name = getattr(node, "name", str(id(node)))
            stage_names = [getattr(dep, "name", str(id(dep))) for dep in info["stage_deps"]]
            pipe_names = [getattr(dep, "name", str(id(dep))) for dep in info["pipeline_deps"]]
            coll_names = [getattr(dep, "name", str(id(dep))) for dep in info["collective_deps"]]
            print(
                f"[ConverterDebug] compute {name}: stage={info['stage']}"
                f" stage_deps={stage_names} pipeline_deps={pipe_names} collectives={coll_names}"
            )

        print("[ConverterDebug] === COLLECTIVE INFO (stage deps / pipeline deps / collectives) ===")
        for edge, info in collective_info.items():
            name = getattr(edge, "name", str(id(edge)))
            stage_names = [getattr(dep, "name", str(id(dep))) for dep in info["stage_deps"]]
            pipe_names = [getattr(dep, "name", str(id(dep))) for dep in info["pipeline_deps"]]
            coll_names = [getattr(dep, "name", str(id(dep))) for dep in info["collective_deps"]]
            print(
                f"[ConverterDebug] collective {name}: stage={info['stage']}"
                f" stage_deps={stage_names} pipeline_deps={pipe_names} collectives={coll_names}"
            )
    if collector:
        _record_pipeline_edges_for_first_dim(collector)
    # Step 6: Group collectives by label/axis so we can reconstruct communicator
    # memberships (tp/cp/pp) without relying on the original ordering.
    tp_collective_groups: Dict[str, List[Any]] = defaultdict(list)
    for edge, info in collective_info.items():
        interconnect_type = info.get("interconnect_type")
        if interconnect_type and interconnect_type not in {"dp", "pp", "pipeline"}:
            tp_collective_groups[info["name"]].append(edge)

    (
        tp_collective_labels,
        label_to_edges,
        label_group_tokens,
    ) = _assign_collective_labels(
        tp_collective_groups,
        collective_info,
        axis_order,
        axis_sizes,
        stage_axis_coords,
        axis_groups,
        stage_to_ranks,
        dp_count,
    )

    # Step 7: Build a per-stage task list (compute + collectives). If a collective
    # introduces a new stage we extend ``stage_to_ranks`` so every stage has the
    # full (dp-major) rank list.
    stage_tasks: Dict[int, Set[Any]] = {stage: set() for stage in stage_ids}
    for node in compute_nodes:
        stage_tasks[node.hw_id].add(node)
    for edge, info in collective_info.items():
        stage = info["stage"]
        if stage not in stage_tasks:
            stage_tasks[stage] = set()
            # Ensure mapping present even for stages discovered only via collectives
            stage_idx = stage_index.setdefault(stage, len(stage_index))
            stage_to_ranks[stage] = [dp_idx * num_stages + stage_idx for dp_idx in range(dp_count)]
            stage_ids.append(stage)
        stage_tasks[stage].add(edge)

    stage_adj: Dict[int, Dict[Any, Set[Any]]] = {stage: defaultdict(set) for stage in stage_tasks}
    stage_indegree: Dict[int, Dict[Any, int]] = {stage: defaultdict(int) for stage in stage_tasks}

    for stage, tasks in stage_tasks.items():
        for task in tasks:
            stage_indegree[stage].setdefault(task, 0)

    for node, info in compute_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(node)
                stage_indegree[stage][node] += 1
        for edge in info["collective_deps"]:
            stage_edge = collective_info.get(edge, {}).get("stage")
            if stage_edge == stage:
                stage_adj[stage][edge].add(node)
                stage_indegree[stage][node] += 1

    for edge, info in collective_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(edge)
                stage_indegree[stage][edge] += 1
        for dep in info.get("collective_deps", set()):
            dep_stage = collective_info.get(dep, {}).get("stage")
            if dep_stage == stage:
                stage_adj[stage][dep].add(edge)
                stage_indegree[stage][edge] += 1


    stage_order: Dict[int, List[Any]] = {}
    for stage, tasks in stage_tasks.items():
        indeg = stage_indegree[stage]
        queue = deque(task for task in tasks if indeg.get(task, 0) == 0)
        order: List[Any] = []
        while queue:
            task = queue.popleft()
            order.append(task)
            for neighbor in stage_adj[stage].get(task, set()):
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(tasks):
            if debug_enabled:
                print(f"[ConverterDebug] cycle stage {stage}")
                for task in tasks:
                    name = getattr(task, "name", str(id(task)))
                    deps = [
                        getattr(src, "name", str(id(src)))
                        for src in stage_adj[stage]
                        if task in stage_adj[stage][src]
                    ]
                    print(
                        f"[ConverterDebug]   task {name}: indegree={indeg.get(task)} deps={deps}"
                    )
            raise RuntimeError(f"Cycle detected in stage {stage} while scheduling")
        stage_order[stage] = order

    if debug_enabled:
        print("[ConverterDebug] === STAGE ORDER ===")
        for stage, order in stage_order.items():
            names = [getattr(task, "name", str(id(task))) for task in order]
            print(f"[ConverterDebug] stage {stage}: {names}")

        compute_debug = []
        for node, info in compute_info.items():
            compute_debug.append(
                {
                    "name": getattr(node, "name", str(id(node))),
                    "stage": info["stage"],
                    "type": node.__class__.__name__,
                    "stage_deps": [getattr(dep, "name", str(id(dep))) for dep in info["stage_deps"]],
                    "pipeline_deps": [getattr(dep, "name", str(id(dep))) for dep in info["pipeline_deps"]],
                    "collective_deps": [getattr(dep, "name", str(id(dep))) for dep in info["collective_deps"]],
                }
            )

        collective_debug = []
        for edge, info in collective_info.items():
            collective_debug.append(
                {
                    "name": getattr(edge, "name", str(id(edge))),
                    "stage": info["stage"],
                    "type": edge.__class__.__name__,
                    "stage_deps": [getattr(dep, "name", str(id(dep))) for dep in info["stage_deps"]],
                    "pipeline_deps": [getattr(dep, "name", str(id(dep))) for dep in info["pipeline_deps"]],
                }
            )

        pipeline_debug = []
        for obj in all_objects:
            if getattr(obj, "comm_type", None) == CollectiveType.PIPELINE:
                sources = [getattr(src, "name", str(id(src))) for src in getattr(obj, "parents", [])]
                targets = [getattr(dst, "name", str(id(dst))) for dst in getattr(obj, "children", [])]
                pipeline_debug.append(
                    {
                        "name": getattr(obj, "name", str(id(obj))),
                        "sources": sources,
                        "targets": targets,
                    }
                )

        debug_payload = {
            "compute": compute_debug,
            "collectives": collective_debug,
            "pipeline_edges": pipeline_debug,
            "stage_order": {stage: [getattr(task, "name", str(id(task))) for task in order] for stage, order in stage_order.items()},
        }

        debug_path = os.path.join(output_dir, "converter_debug.json")
        try:
            with open(debug_path, "w", encoding="utf-8") as fh:
                json.dump(debug_payload, fh, indent=2)
        except Exception:
            pass

    compute_et_ids: Dict[Tuple[Any, int], int] = {}
    collective_et_ids: Dict[Tuple[Any, int], int] = {}
    pipeline_recv_cache: Dict[Tuple[Any, Any, int], int] = {}
    tag_counter = itertools.count(start=1)

    # Step 8 helper: materialise SEND/RECV control edges across stages whenever a
    # pipeline dependency exists between ``parent`` and ``target``.
    def ensure_pipeline(parent: Any, target: Any, dp_idx: int) -> int:
        parent_stage = getattr(parent, "hw_id", None)
        if parent_stage == None:
            parent_stage = getattr(parent, "local_hw_id", None)
        if parent_stage == None:
            raise ValueError(f"Stage not found for parent {parent}")
        target_stage = collective_info[target]["stage"] if target in collective_info else compute_info[target]["stage"]
        if parent_stage == target_stage:
            if parent in collective_info:
                return collective_et_ids[(parent, rank_for(parent_stage, dp_idx))]
            return compute_et_ids[(parent, rank_for(parent_stage, dp_idx))]

        edge_obj = pipeline_edge_map.get((parent, target))

        key = (parent, target_stage, dp_idx, edge_obj)
        # if the key has been seen before, then this is a duplicate pipeline dependency
        # just return old recv_id
        cached = pipeline_recv_cache.get(key)
        if cached is not None:
            return cached

        # if not edge_obj:
        #     print("Pipeline edge map:")
        #     import pprint
        #     pprint.pprint(pipeline_edge_map)
        #     raise ValueError(f"No edge object found for parent {parent} and target {target}")
        # If there is no edge, assume it is a control dependency (size set to 0 automatically)
        size = int(getattr(edge_obj, "comm_size_bytes", 0))
        tag = getattr(edge_obj, "op_id", next(tag_counter))

        src_rank = rank_for(parent_stage, dp_idx)
        dst_rank = rank_for(target_stage, dp_idx)
        send_trace = rank_traces[src_rank]
        recv_trace = rank_traces[dst_rank]
        is_control = False
        if size == 0 or edge_obj.comm_type != CollectiveType.PIPELINE:
            # control dependancy
            size = 1 # has to be at least 1 byte for astrasim.
            is_control = True

        send_id = send_trace.next_id
        if is_control:
            send_name = f"{getattr(edge_obj, 'name', 'pipeline')}_send_control"
        else:
            send_name = f"{getattr(edge_obj, 'name', 'pipeline')}_send_dp{dp_idx}"
        send_node = new_send_node(send_id, send_name, size, dst_rank, tag)
        if parent in collective_info:
            try:
                send_node.ctrl_deps.append(collective_et_ids[(parent, src_rank)])
            except KeyError as e:
                print(f"Collective dict state:")
                print(f"Keys")
                for key, value in collective_et_ids.items():
                    print(f"Key: {key}")
                raise e
        elif parent in compute_info:
            try:
                send_node.ctrl_deps.append(compute_et_ids[(parent, src_rank)])
            except KeyError as e:
                print(f"Compute dict state:")
                print(f"Keys")
                for key, value in compute_et_ids.items():
                    print(f"Key: {key}")
                raise e
        else:
            raise ValueError(f"Parent {parent} not found in collective_info or compute_info")

        send_trace.append_node(send_node)

        recv_id = recv_trace.next_id
        if is_control:
            recv_name = f"{getattr(edge_obj, 'name', 'pipeline')}_recv_control"
        else:
            recv_name = f"{getattr(edge_obj, 'name', 'pipeline')}_recv_dp{dp_idx}"
        recv_node = new_recv_node(recv_id, recv_name, size, src_rank, tag)
        # recv_node.ctrl_deps.append(send_id)
        recv_trace.append_node(recv_node)

        pipeline_recv_cache[key] = recv_id
        return recv_id

    mapping_result = gmap.finalize_collection(collector)
    if mapping_result:
        stage_ids, stage_index = _remap_stages_for_mapping(
            permutation=mapping_result.permutation,
            collector=collector,
            stage_ids=stage_ids,
            stage_to_ranks=stage_to_ranks,
            stage_index=stage_index,
            rank_traces=rank_traces,
            rank_meta=rank_meta,
            dp_count=dp_count,
            num_stages=num_stages,
            output_dir=output_dir,
        )
        # Rebuild communicator layout after remapping so pg_name and comm_groups.json
        # reflect the SCOTCH permutation.
        axis_groups = _build_axis_groups(axis_order, axis_sizes, stage_axis_coords, stage_to_ranks)
        (
            tp_collective_labels,
            label_to_edges,
            label_group_tokens,
        ) = _assign_collective_labels(
            tp_collective_groups,
            collective_info,
            axis_order,
            axis_sizes,
            stage_axis_coords,
            axis_groups,
            stage_to_ranks,
            dp_count,
        )

    # Build TP communicator groups (ids start at 1000) per label and dp using the
    # (potentially remapped) label tokens.
    global _LAST_TP_GROUPS, _TP_LABEL_DP_TO_ID, _TP_MEMBERS_TO_GID
    _LAST_TP_GROUPS = {}
    _TP_LABEL_DP_TO_ID = {}
    _TP_MEMBERS_TO_GID = {}
    tp_gid_counter = _it_count(start=_TP_GROUP_BASE_ID)

    for label in sorted(label_group_tokens.keys()):
        tokens = sorted(label_group_tokens[label])
        for axis, dp_idx, members_tuple in tokens:
            members_list = list(members_tuple)
            members_key = tuple(members_list)
            gid = _TP_MEMBERS_TO_GID.get(members_key)
            if gid is None:
                gid = str(next(tp_gid_counter))
                _TP_MEMBERS_TO_GID[members_key] = gid
                if gid not in _LAST_TP_GROUPS:
                    _LAST_TP_GROUPS[gid] = members_list
            _TP_LABEL_DP_TO_ID[(label, dp_idx)] = gid

    # Step 10: Emit ET nodes per stage (collectives first, then compute nodes),
    # iterating in the topological order computed above.
    for stage in stage_ids:
        order = stage_order.get(stage)
        if not order:
            continue
        for task in order:
            if task in collective_info:
                info = collective_info[task]
                # Skip stage collectives entirely when dp_count <= 1
                if dp_count <= 1 and task not in tp_collective_labels:
                    continue
                for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                    trace = rank_traces[rank]
                    deps: List[int] = []
                    for dep in info["stage_deps"]:
                        if dep in collective_info:
                            deps.append(collective_et_ids[(dep, rank_for(collective_info[dep]["stage"], dp_idx))])
                        else:
                            # if dep.hw_id exists, do this. Else look at local_hw_id.
                            if hasattr(dep, "hw_id"):
                                deps.append(compute_et_ids[(dep, rank_for(dep.hw_id, dp_idx))])
                            else:
                                try:
                                    deps.append(compute_et_ids[(dep, rank_for(dep.local_hw_id, dp_idx))])
                                except KeyError as e:
                                    print(f"Compute et id not found for {info}")
                                    print(f"Missing dep: {dep}")
                                    # print(f"Compute et ids: {compute_et_ids}")
                                    raise e
                    for dep in info.get("collective_deps", set()):
                        if dep in collective_info:
                            try:
                                deps.append(collective_et_ids[(dep, rank_for(collective_info[dep]["stage"], dp_idx))])
                            except KeyError as e:
                                print(f"Collective et id not found for {info}")
                                print(f"Missing dep: {dep}")
                                # print(f"Collective et ids: {collective_et_ids}")
                                raise e
                        else:
                            if hasattr(dep, "hw_id"):
                                deps.append(compute_et_ids[(dep, rank_for(dep.hw_id, dp_idx))])
                            else:
                                deps.append(compute_et_ids[(dep, rank_for(dep.local_hw_id, dp_idx))])
                    # for parent in info["pipeline_deps"]:
                    #     deps.append(ensure_pipeline(parent, task, dp_idx))
                    unique_deps = []
                    for dep in deps:
                        if dep not in unique_deps:
                            unique_deps.append(dep)

                    node_id = trace.next_id
                    if task in tp_collective_labels:
                        comm_name = tp_collective_labels[task]
                    else:
                        comm_name = f"{task.name}_{task.op_id}_dp{dp_idx}"
                    comm_node = new_comm_node(
                        node_id,
                        comm_name,
                        info["comm_type"],
                        info["size"],
                    )
                    # Attach communicator group
                    # DP groups for non-TP collectives; TP groups for TP-labeled edges
                    if task in tp_collective_labels:
                        label = tp_collective_labels[task]
                        gid = _TP_LABEL_DP_TO_ID.get((label, dp_idx))
                        if gid:
                            comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=str(gid)))
                    elif dp_count > 1:
                        stage_idx = stage_index[stage]
                        group_id = str(stage_idx + 1)
                        comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=group_id))
                    comm_node.ctrl_deps.extend(unique_deps)
                    trace.append_node(comm_node)
                    collective_et_ids[(task, rank)] = node_id
            else:
                info = compute_info[task]
                stage = info["stage"]
                for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                    trace = rank_traces[rank]
                    deps: List[int] = []
                    for parent in info["stage_deps"]:
                        deps.append(compute_et_ids[(parent, rank_for(parent.hw_id, dp_idx))])
                    for edge in info["collective_deps"]:
                        stage_edge = collective_info.get(edge, {}).get("stage")
                        if stage_edge is not None and stage_edge == stage:
                            deps.append(collective_et_ids[(edge, rank)] )
                    unique_deps = []
                    for dep in deps:
                        if dep not in unique_deps:
                            unique_deps.append(dep)

                    duration_sec = _compute_duration_seconds(task, dp_idx)
                    duration_micros = int(round(duration_sec * 1e6)) if duration_sec else 0
                    node_id = trace.next_id
                    comp_node = new_comp_node(
                        node_id,
                        f"{task.name}_{task.op_id}",
                        max(duration_micros, 0)
                    )
                    comp_node.ctrl_deps.extend(unique_deps)
                    trace.append_node(comp_node)
                    compute_et_ids[(task, rank)] = node_id

                    # NOTE: Do not generate collective nodes here. Collectives
                    # are created in their own 'task in collective_info' branch
                    # above to avoid duplication. Here we only reference them
                    # via dependencies when needed.

    # Step 11: Attach cross-stage (pipeline) dependencies by wiring the RECV
    # nodes created via ``ensure_pipeline`` into the ET nodes’ control deps.
    for stage, order in stage_order.items():
        for task in order:
            if task in collective_info:
                info = collective_info[task]
            else:
                info = compute_info[task]

            if not info["pipeline_deps"]:
                continue

            for dp_idx, rank in enumerate(stage_to_ranks[stage]):
                # Ensure SEND/RECV nodes exist and collect RECV ids local to this rank
                recv_ids: List[int] = []
                for parent in info["pipeline_deps"]:
                    if parent not in collective_info and parent not in compute_info:
                        # orphaned parent, ignore
                        # TODO: debug and make sure this never happens?!??!?!
                        continue
                    try:
                        recv_ids.append(ensure_pipeline(parent, task, dp_idx))
                    except Exception as e:
                        print(f"Attempted to map parent {parent} to task {task} for dp_idx {dp_idx}")
                        print(f"Info: {info}")
                        raise e

                # Deduplicate
                unique_recv_ids: List[int] = []
                for rid in recv_ids:
                    if rid not in unique_recv_ids:
                        unique_recv_ids.append(rid)

                # Append RECV deps to the already-created node for this task
                if task in collective_info:
                    node_id = collective_et_ids[(task, rank)]
                else:
                    node_id = compute_et_ids[(task, rank)]
                node = rank_traces[rank].nodes[node_id]
                for rid in unique_recv_ids:
                    if rid not in node.ctrl_deps:
                        node.ctrl_deps.append(rid)


    # Step 12: Build manifest ops directly from the in-memory traces. This lets the
    # caller reuse the same metadata that will be written into the ET files.
    manifest_ranks: Dict[str, List[List]] = {}
    def _manifest_op_key(op: List) -> tuple:
        # Sort primarily by size/duration, with stable type ordering
        try:
            kind = op[0]
        except Exception:
            kind = None
        if kind == "COMP":
            # ["COMP", duration_us]
            return (0, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "COMM":
            # ["COMM", comm_type, size_bytes, participants]
            size_val = int(op[2]) if len(op) > 2 else 0
            ctype_val = int(op[1]) if len(op) > 1 else -1
            return (1, size_val, ctype_val)
        if kind == "SEND":
            # ["SEND", size_bytes]
            return (2, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "RECV":
            # ["RECV", size_bytes]
            return (3, int(op[1]) if len(op) > 1 else 0, 0)
        # Fallback
        return (9, 0, 0)

    for rank, trace in sorted(rank_traces.items()):
        rank_ops: List[List] = []
        for node in trace.nodes:
            t = int(node.type)
            if t == pb.COMP_NODE:
                dur = int(node.duration_micros or 0)
                rank_ops.append(["COMP", dur])
            elif t == pb.COMM_COLL_NODE:
                ctype = None
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_type":
                        which = attr.WhichOneof("value")
                        ctype = int(getattr(attr, which)) if which else None
                    elif attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["COMM", int(ctype or -1), int(csize or 0), None])
            elif t == pb.COMM_SEND_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["SEND", int(csize or 0)])
            elif t == pb.COMM_RECV_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["RECV", int(csize or 0)])
        # Sort ops to stabilize manifest across runs
        rank_ops_sorted = sorted(rank_ops, key=_manifest_op_key)
        manifest_ranks[str(int(rank))] = rank_ops_sorted

    # Step 13: Write the Chakra ET traces (with control-send renumbering) and
    # persist the manifest/metadata alongside them.
    for trace in rank_traces.values():
        trace.close()

    et_prefix = f"{output_dir}/llm_graph"
    rank_ids = sorted(rank_traces.keys())
    print(f"[AstraSim] Generated ET files for ranks: {et_prefix}.{{0..{len(rank_ids)-1}}}.et")

    # Emit manifest.json in the same output directory (graph-based signature)
    manifest_path = os.path.join(output_dir, "manifest.json")
    try:
        with open(manifest_path, "w") as mf:
            json.dump({
                "version": "df-astra-manifest/1",
                "npus": len(rank_ids),
                "ranks": manifest_ranks,
            }, mf, sort_keys=True, separators=(",", ":"))
        print(f"[AstraSim] Wrote graph manifest to {manifest_path}")
    except Exception as exc:
        print(f"[WARN] Failed to write manifest: {exc}")
    return et_prefix, rank_ids, manifest_path

def run_astra_simulation_only_onepath(
    fwdbwd_root,
    time_calc_obj,
    output_dir: str = "astra_comparison_output",
    dp_override: Optional[int] = None,
    persist_artifacts: Optional[bool] = None,
    faulty_links_override: Optional[Sequence[Tuple[int, int, float]]] = None,
):
    """
    Run AstraSim simulation on RAPID-LLM graph and print results.

    Args:
        fwdbwd_root: Forward and backward graph root node
        time_calc_obj: TimeCalculationLLM object with hw_config and dp attributes
        output_dir: Directory for temporary files and results
        faulty_links_override: Optional remapped faulty link list for this run
    """
    print("\n" + "="*60)
    print("ASTRASIM SIMULATION RESULTS")
    print("="*60)

    persist = persist_artifacts if persist_artifacts is not None else _env_truthy(
        "RAPID_PERSIST_ASTRASIM_ARTIFACTS"
    )
    os.makedirs(output_dir, exist_ok=True)
    work_dir: str
    if persist:
        work_dir = output_dir
        _clean_astrasim_artifacts(work_dir)
    else:
        work_dir = tempfile.mkdtemp(prefix="astrasim_", dir=output_dir)

    try:
        # Convert both forward and backward graphs to Chakra ET format
        astrasim_start = time.time()

        # For now, just convert forward graph (can extend to include backward later)
        print(f"[AstraSim] Converting graph...")
        user_dp = max(1, getattr(time_calc_obj, "dp", 1))
        run_type = getattr(getattr(time_calc_obj, "model", None), "run_type", "")
        dp_count = dp_override if dp_override is not None else user_dp
        fwd_et_prefix, rank_ids, fwd_manifest = convert_rapid_llm_graph_to_chakra_et(
            fwdbwd_root,
            dp_count,
            work_dir,
        )
        rank_count = len(rank_ids)
        # Astrasim doesn't play well with only 1 rank.
        # When that happens, let's duplicate to 2 ranks. No collectives exist between the two so this should not have an effect.
        synthetic_pair = False
        if rank_count == 1:
            # duplicate the .et file
            src = os.path.join(work_dir, "llm_graph.0.et")
            dst = os.path.join(work_dir, "llm_graph.1.et")
            shutil.copy(src, dst)
            rank_ids = [rank_ids[0], rank_ids[0]+1]
            synthetic_pair = True
        rank_count = len(rank_ids)

        # Handle artifact visualization if enabled
        if persist and _env_truthy("RAPID_PERSIST_ARTIFACT_VIZ"):
            et_paths = []
            for rank in rank_ids:
                et_path = os.path.join(work_dir, f"llm_graph.{rank}.et")
                if os.path.exists(et_path):
                    et_paths.append(et_path)

            if et_paths:
                print(f"[AstraSim] Visualizing {len(et_paths)} persisted ET files...")
                _visualize_et_files(et_paths)
                _dump_et_text(et_paths)

        # Generate AstraSim configuration files using actual hardware config
        print(f"[AstraSim] Generating configuration files...")
        comm_groups_dp = dp_count if dp_override is not None else user_dp
        comm_groups_path = _write_comm_groups_json(work_dir, comm_groups_dp, rank_ids)
        if os.environ.get("RAPID_ASTRA_SKIP_EXEC"):
            print("[AstraSim] RAPID_ASTRA_SKIP_EXEC set. Exiting after ET artifact generation.")
            exit()

        rank_layout = getattr(fwdbwd_root, "_astrasim_rank_layout", None)
        axis_order, axis_sizes, _ = _extract_axis_layout(rank_layout)
        preferred_axes_for_synthetic = tuple(axis_order) if axis_order else tuple()
        axes_filter = derive_axes_filter(axis_order, axis_sizes, dp_count)
        if not axes_filter:
            axes_filter = None
        if synthetic_pair:
            axes_filter = ["synthetic2"]
        astra_configs = generate_astrasim_configs_from_hw(
            time_calc_obj.hw_config,
            work_dir,
            rank_count,
            axes_filter=axes_filter,
            faulty_links_override=faulty_links_override,
            preferred_axes_for_synthetic=preferred_axes_for_synthetic,
        )

        # Run AstraSim simulation on forward graph (cached via manifest)
        print(f"[AstraSim] Executing forward simulation with {rank_count} ranks...")
        cache_override_env = os.environ.pop("ASTRA_CACHE_DIR", None)
        local_cache_dir = work_dir
        local_cache_path = os.path.join(work_dir, "cache.json")
        try:
            fwd_times, fwd_total = run_cache_astrasim(
                time_calc_obj.hw_config,
                comm="graph",
                npus_count=rank_count,
                size_bytes=0,
                astra_config_dir=local_cache_dir,
                cache_path=local_cache_path,
                manifest_json_path=fwd_manifest,
                workload_prefix=fwd_et_prefix,
                comm_group_json=comm_groups_path,
                axes_filter=axes_filter,
                files=astra_configs,
            )
        finally:
            if cache_override_env is not None:
                os.environ["ASTRA_CACHE_DIR"] = cache_override_env


        conversion_and_sim_time = time.time() - astrasim_start

        # Print results
        # include times per node
        if len(fwd_times) > 5:
            printstr = ""
            for i in range(5):
                #reverse the list
                rev_times = list(reversed(fwd_times))
                printstr += f" {round(rev_times[i],2)},"
            printstr += "...."
            print(f"[AstraSim] Times per node:{printstr}")
        else:
            print(f"[AstraSim] Times per node: {fwd_times}")
        print(f"[AstraSim] Total execution time: {fwd_total:.6f} seconds")
        print(f"[AstraSim] Simulation duration: {conversion_and_sim_time:.3f} seconds")

        print("="*60)

        return fwd_times, fwd_total

    except Exception as e:
        print(f"[AstraSim] ERROR: Failed to run simulation: {e}")
        print("="*60)
        raise
    finally:
        if not persist:
            shutil.rmtree(work_dir, ignore_errors=True)



if __name__ == "__main__":

    def my_save_graph(roots, output_folder = "output_graph/", filename="graph"):
        dot_fw = visualize_graph(roots, filename=output_folder + filename)
        dot_fw.render(output_folder + filename , format="svg", cleanup=True)
        print("graph saved to %s%s.svg" % (output_folder , filename ))

    import config
    import pickle
    # exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_hw_path = os.path.expandvars(os.path.expanduser("configs/hardware-config/a100_80GB_tp.yaml"))
    exp_model_path = os.path.expandvars(os.path.expanduser("configs/model-config/LLM.yaml"))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type="LLM")
    with open("fw_bw_graph.pkl", "rb") as f:
        fw_bw_root = pickle.load(f)
    # make a fake object 
    class FakeTimeCalculationLLM:
        def __init__(self, hw_config, model_config, mode):
            self.hw_config = hw_config
            self.model_config = model_config    
            self.mode = mode
            self.dp = 2
    time_calc_obj = FakeTimeCalculationLLM(exp_hw_config, exp_model_config, "LLM")
    my_save_graph(fw_bw_root, "./astra_comparison_output", "fw_bw_graph_astra")
    paths = []
    paths.append("/app/nanocad/projects/rapid_llm_dev/RAPID-LLM/RAPID-LLM_george/astra_cache/workload/all_reduce/2npus_1.50GB/all_reduce_1.50GB.0.et")
    _dump_et_text(paths)
    run_astra_simulation_only_onepath(fw_bw_root, time_calc_obj, "./astra_comparison_output")   
