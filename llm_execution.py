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

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import simulate_train_graph as llm_simulation
from memory_estimation import mem_kind_from_op_name
from astrasim_lib import run_astra_simulation_only_onepath
from astrasim_lib.fault_projection import FaultProjectionResult, FaultSpace
from astrasim_lib.layout_utils import axis_layout_from_descriptor
from simulate_train_graph import Graph
from timing_model import CollectiveType
from util import log_message

if TYPE_CHECKING:
    from train_timing import TimeCalculationLLM


def _mode_label(mode: Any) -> str:
    for attr in ("value", "name"):
        if hasattr(mode, attr):
            return str(getattr(mode, attr)).lower()
    return str(mode).lower()


def _copy_node_metadata(source: llm_simulation.Node, target: llm_simulation.Node) -> None:
    for attr in (
        "micro_batch_index",
        "layer_index",
        "direction",
        "stage_id",
        "tp_rank",
        "cp_rank",
        "mem_kind",
        "recompute",
        "param_gather",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def _copy_edge_metadata(source: llm_simulation.Edge, target: llm_simulation.Edge) -> None:
    for attr in (
        "local_hw_id",
        "stage_id",
        "micro_batch_index",
        "layer_index",
        "direction",
        "tp_rank",
        "cp_rank",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def _detach_edge(parent: Any, child: Any) -> None:
    children = getattr(parent, "children", None)
    if isinstance(children, list):
        try:
            children.remove(child)
        except ValueError:
            pass
    parents = getattr(child, "parents", None)
    if isinstance(parents, list):
        try:
            parents.remove(parent)
        except ValueError:
            pass


def _connect_edge(parent: Any, child: Any) -> None:
    if child in getattr(parent, "children", []):
        return
    parent.add_child(child)


def _split_tp_node(
    node: llm_simulation.Node,
    tp_children: List[llm_simulation.Edge],
    overlap: float,
) -> None:
    if overlap <= 0.0 or not tp_children:
        return

    duration = float(getattr(node, "duration", 0.0) or 0.0)
    if duration <= 0.0:
        return

    if overlap >= 1.0:
        parents = list(getattr(node, "parents", []))
        tp_succs: List[Any] = []
        for tp_edge in tp_children:
            _detach_edge(node, tp_edge)
            for parent in parents:
                _connect_edge(parent, tp_edge)
            for succ in list(getattr(tp_edge, "children", []) or []):
                tp_succs.append(succ)
        for succ in tp_succs:
            _connect_edge(node, succ)
        return

    head_duration = duration * (1.0 - overlap)
    tail_duration = duration * overlap
    if head_duration <= 0.0 or tail_duration <= 0.0:
        return

    tail = node
    tail.duration = tail_duration
    head = llm_simulation.Node(
        name=f"{tail.name}_head",
        op_id=getattr(tail, "op_id", 0),
        hw_id=tail.hw_id,
        duration=head_duration,
        fwd=tail.fwd,
    )
    _copy_node_metadata(tail, head)

    parents = list(getattr(tail, "parents", []))
    for parent in parents:
        _detach_edge(parent, tail)
        _connect_edge(parent, head)

    _connect_edge(head, tail)

    for tp_edge in tp_children:
        _detach_edge(tail, tp_edge)
        _connect_edge(head, tp_edge)
        for succ in list(getattr(tp_edge, "children", []) or []):
            _connect_edge(tail, succ)


def _apply_tp_overlap_transforms(root: Any, parallelism_mode: Any, tp_overlap: float, tp_sp_overlap: float) -> Any:
    mode = _mode_label(parallelism_mode)
    if mode == "tensor_sequence":
        overlap = tp_sp_overlap
    elif mode in {"tensor", "tensor_context_hybrid"}:
        overlap = tp_overlap
    else:
        return root
    if overlap <= 0.0:
        return root

    nodes_to_process: List[llm_simulation.Node] = []
    visited: Set[int] = set()
    stack: List[Any] = list(root) if isinstance(root, (list, tuple)) else [root]
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if isinstance(obj, llm_simulation.Node):
            nodes_to_process.append(obj)
        for child in getattr(obj, "children", []):
            stack.append(child)

    for node in nodes_to_process:
        tp_children = [
            child for child in getattr(node, "children", [])
            if isinstance(child, llm_simulation.Edge) and getattr(child, "comm_interconnect_type", None) == "tp"
        ]
        if not tp_children:
            continue
        _split_tp_node(node, tp_children, overlap)
    return root


def _apply_cp_overlap_transforms(root: Any, parallelism_mode: Any, cp_overlap: float) -> Any:
    mode = _mode_label(parallelism_mode)
    if mode not in {"context", "tensor_context_hybrid"}:
        return root
    overlap = cp_overlap
    if overlap <= 0.0:
        return root

    cp_edges: List[llm_simulation.Edge] = []
    visited: Set[int] = set()
    stack: List[Any] = list(root) if isinstance(root, (list, tuple)) else [root]
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if isinstance(obj, llm_simulation.Edge) and getattr(obj, "comm_interconnect_type", None) == "cp":
            cp_edges.append(obj)
        for child in getattr(obj, "children", []):
            stack.append(child)

    for edge in cp_edges:
        _split_cp_edge(edge, overlap)
    return root


def _split_cp_edge(edge: llm_simulation.Edge, overlap: float) -> None:
    attention_children = [
        child for child in getattr(edge, "children", [])
        if isinstance(child, llm_simulation.Node) and "attention" in str(getattr(child, "name", "")).lower()
    ]
    if not attention_children:
        return

    total_bytes = int(getattr(edge, "comm_size_bytes", 0) or 0)
    preds = list(getattr(edge, "parents", []))
    succs = list(getattr(edge, "children", []))

    if overlap >= 1.0 or total_bytes <= 0:
        for attention in attention_children:
            _detach_edge(edge, attention)
            for pred in preds:
                _connect_edge(pred, attention)
        for succ in succs:
            if succ in attention_children:
                continue
            for attention in attention_children:
                _connect_edge(attention, succ)
            _connect_edge(edge, succ)
        return

    block_bytes = int(math.ceil(total_bytes * (1.0 - overlap)))
    ovlp_bytes = max(0, total_bytes - block_bytes)
    if block_bytes <= 0:
        _split_cp_edge(edge, 1.0)
        return

    block_edge = llm_simulation.Edge(
        name=f"{edge.name}_block",
        op_id=getattr(edge, "op_id", 0),
        duration=0,
        is_dp=edge.is_dp,
        comm_size_bytes=block_bytes,
        comm_type=edge.comm_type,
        participants=edge.participants,
        comm_interconnect_type=edge.comm_interconnect_type,
    )
    ovlp_edge = None
    if ovlp_bytes > 0:
        ovlp_edge = llm_simulation.Edge(
            name=f"{edge.name}_ovlp",
            op_id=getattr(edge, "op_id", 0),
            duration=0,
            is_dp=edge.is_dp,
            comm_size_bytes=ovlp_bytes,
            comm_type=edge.comm_type,
            participants=edge.participants,
            comm_interconnect_type=edge.comm_interconnect_type,
        )
        _copy_edge_metadata(edge, ovlp_edge)
    _copy_edge_metadata(edge, block_edge)

    for pred in preds:
        _detach_edge(pred, edge)
        _connect_edge(pred, block_edge)

    for attention in attention_children:
        _detach_edge(edge, attention)
        _connect_edge(block_edge, attention)

    if ovlp_edge is not None:
        _connect_edge(block_edge, ovlp_edge)

    for succ in succs:
        if succ in attention_children:
            continue
        _detach_edge(edge, succ)
        for attention in attention_children:
            _connect_edge(attention, succ)
        if ovlp_edge is not None:
            _connect_edge(ovlp_edge, succ)
        else:
            _connect_edge(block_edge, succ)


def apply_overlap_transforms(
    root: Any,
    parallelism_mode: Any,
    tp_overlap: float,
    tp_sp_overlap: float,
    cp_overlap: float,
) -> Any:
    """Apply TP/TP_SP/CP overlap rewrites to a constructed graph."""
    if root is None:
        return None
    root = _apply_tp_overlap_transforms(root, parallelism_mode, tp_overlap, tp_sp_overlap)
    root = _apply_cp_overlap_transforms(root, parallelism_mode, cp_overlap)
    return root


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}





class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"
    
    
@dataclass
class ExecutionResult:
    total_time: float
    graph_root: Any
    mode: ExecutionMode


@dataclass
class TransformerTimings:
    forward: float
    backward: float
    
class PipelineGraphFlattener:
    """Expand pipeline transformer nodes into explicit tensor-parallel subgraphs."""

    def __init__(
        self,
        pipeline_graph: Graph,
        transformer_graph: Graph,
        *,
        moe_transformer_graph: Optional[Graph] = None,
        rank_layout: Optional[Dict[str, Any]] = None,
    ) -> None:
        if transformer_graph is None:
            raise ValueError("Transformer graph is required for flattening")

        transformer_cfg = getattr(transformer_graph, "transformer_cfg", None) or {}
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM template is missing")

        self.pipeline_graph = pipeline_graph
        self.transformer_graph = transformer_graph
        self.moe_transformer_graph = moe_transformer_graph
        self._gemm_entries = list(gemm_entries)
        self._gemm_entries_moe: Optional[List[Dict[str, Any]]] = None
        if moe_transformer_graph is not None:
            moe_cfg = getattr(moe_transformer_graph, "transformer_cfg", None) or {}
            moe_entries = moe_cfg.get("gemms")
            if not moe_entries:
                raise ValueError("MoE transformer GEMM template is missing")
            self._gemm_entries_moe = list(moe_entries)
        ep_size = int(getattr(transformer_graph, "ep", 1))
        par_degree = transformer_graph.tp * transformer_graph.cp * ep_size
        self._par_degree = max(1, int(par_degree))
        self._zero_stage = int(getattr(transformer_graph, "misc_metadata", {}).get("dp_zero_stage", 0))
        self._layout_axis_order: Optional[List[str]] = None
        self._layout_axis_sizes: Dict[str, int] = {}
        self._layout_axis_strides: Dict[str, int] = {}
        self._tp_size = int(getattr(transformer_graph, "tp", 1))
        self._cp_size = int(getattr(transformer_graph, "cp", 1))
        self._ep_size = int(ep_size)
        self._pp_size = int(getattr(pipeline_graph, "pp", 1))
        if rank_layout is not None:
            self._configure_rank_layout(rank_layout)

        # Track original ZeRO-3 transformer gather edges -> per-rank clones
        self._clone_cache: Dict[int, Any] = {}
        self._op_id_counter: int = 0
        self._stage_span = 1

    def _configure_rank_layout(self, descriptor: Dict[str, Any]) -> None:
        axis_order = list(descriptor.get("axis_order", []))
        axis_sizes = dict(descriptor.get("axis_sizes", {}))
        axis_strides = dict(descriptor.get("axis_strides", {}))
        if not axis_order:
            self._layout_axis_order = None
            return
        self._layout_axis_order = axis_order
        self._layout_axis_sizes = axis_sizes
        if not axis_strides:
            span = 1
            for axis in axis_order:
                axis_strides[axis] = span
                span *= axis_sizes.get(axis, 1)
        self._layout_axis_strides = axis_strides
        self._tp_size = max(1, axis_sizes.get("tp", self._tp_size))
        self._cp_size = max(1, axis_sizes.get("cp", self._cp_size))
        self._ep_size = max(1, axis_sizes.get("ep", self._ep_size))
        self._pp_size = max(1, axis_sizes.get("pp", self._pp_size))
        if self._par_degree != self._tp_size * self._cp_size * self._ep_size:
            raise ValueError(
                f"Inconsistent tensor/context/expert parallel factors: tp={self._tp_size}, cp={self._cp_size}, "
                f"ep={self._ep_size}, "
                f"product does not equal par_degree={self._par_degree}"
            )
        stage_span = 1
        for axis in axis_order:
            if axis == "dp":
                continue
            stage_span *= axis_sizes.get(axis, 1)
        self._stage_span = stage_span

    def _should_shard_zero3_transformer(self, edge: Any) -> bool:
        if self._par_degree <= 1 or self._zero_stage < 3:
            return False
        if isinstance(edge, llm_simulation.Edge):
            if getattr(edge, "tp_shard", False):
                return True
        return False

    def _ensure_zero3_per_rank_edges(
        self,
        edge: llm_simulation.Edge,
        cross_device = False,
        rank_heads = None,
        rank_tails = None,
        hw_ids = None
    ) -> List[llm_simulation.Edge]:

        transformer_mode = ""
        per_rank_edges: List[llm_simulation.Edge] = []
        is_backward = False
        if rank_tails and rank_heads:
            # this is used in expand_transformer_node
            # use them as anchors
            wire_anchors = rank_tails
            direction = getattr(edge, "direction", None)
            if direction and str(direction).lower() == "backward":
                is_backward = True
                wire_anchors = rank_heads
            iterable = wire_anchors
            transformer_mode = True
        elif hw_ids:
            # this is used for softmax embedding edges for raw cloning.
            iterable = hw_ids
            transformer_mode = False

        if transformer_mode == "":
            raise Exception("Invalid _ensure_zero3_per_rank_edges call. At least one of (rank_tails,rank_heads) or (hw_ids) must be provided.")
         
        for r, item in enumerate(iterable):
            base_bytes = getattr(edge, "comm_size_bytes", 0)
            per_rank_bytes = int(base_bytes)
            gather_edge = llm_simulation.Edge(
                name=f"{edge.name}_rank{r}",
                op_id=self._next_op_id(),
                duration=0,
                is_dp=True,
                comm_size_bytes=per_rank_bytes,
                comm_type=getattr(edge, "comm_type", None),
                participants=getattr(edge, "participants", 0),
                comm_interconnect_type=getattr(edge, "comm_interconnect_type", None),
            )
            gather_edge.tp_rank = r
            gather_edge.stage_id = getattr(edge, "stage_id", None)
            gather_edge.micro_batch_index = getattr(edge, "micro_batch_index", None)
            gather_edge.layer_index = getattr(edge, "layer_index", None)
            gather_edge.direction = getattr(edge, "direction", None)
            gather_edge.tp_shard = True
            if transformer_mode:
                if not "bwd" in edge.name: # HACK!!!!
                    offset = self._par_degree
                else:
                    offset = -self._par_degree
                # print(f"TM Edge name: {edge.name}, item: {item}")
                if getattr(item, "hw_id", None) is not None:
                    # if cross device, dont map to item hw id but next hw ids!
                    if cross_device:
                        gather_edge.local_hw_id = (item.hw_id) + offset
                    else:
                        gather_edge.local_hw_id = item.hw_id
                else:
                    if cross_device:
                        gather_edge.local_hw_id = (getattr(item, "local_hw_id", None)) + offset
                    else:
                        gather_edge.local_hw_id = getattr(item, "local_hw_id", None)
                # has the same children as item
                for child in getattr(item, "children", []):
                    gather_edge.add_child(child)
            else:
                # print(f"EM Edge name: {edge.name}, item: {item}")
                gather_edge.local_hw_id = item
                # we cannot attach children here as we don't have them yet.


            per_rank_edges.append(gather_edge)

        return per_rank_edges


    def build(self, root: Any) -> Any:
        """Return a flattened clone of the provided pipeline root."""

        if root is None:
            raise ValueError("Pipeline root is required for flattening")
        return self._clone(root)

    def _clone(self, obj: Any) -> Any:
        if obj is None:
            return None

        obj_id = id(obj)
        if obj_id in self._clone_cache:
            return self._clone_cache[obj_id]

        if isinstance(obj, llm_simulation.Node):
            base_name = str(getattr(obj, "name", "") or "")
            if base_name.startswith("transformer_layer"):
                expanded = self._expand_transformer_node(obj)
                self._clone_cache[obj_id] = expanded
                return expanded

            if "linear_softmax" in obj.name:
                # for linear softmax we need to carefully look at hw id to choose.
                cloned = llm_simulation.Node(
                    obj.name,
                    self._next_op_id(),
                    self._hw_id_for_rank(obj.hw_id, 0),
                    obj.duration,
                    fwd=obj.fwd,
                )
            elif "optimizer" in obj.name:
                # Optimizer nodes need to be expanded per TP rank
                cloned_nodes = []
                for tp_rank in range(self._par_degree):
                    hw_id = self._hw_id_for_rank(obj.hw_id, tp_rank)
                    cloned_node = llm_simulation.Node(
                        name=f"{obj.name}_rank{tp_rank}",
                        op_id=self._next_op_id(),
                        hw_id=hw_id,
                        duration=obj.duration,
                        fwd=obj.fwd,
                    )
                    cloned_nodes.append(cloned_node)
                
                cloned_tuple = tuple(cloned_nodes)
                self._clone_cache[obj_id] = cloned_tuple
                
                # We also need to copy metadata if any
                for cloned_node in cloned_nodes:
                    self._copy_metadata(obj, cloned_node)
                    
                # Attach children (if any) to all clones? 
                # Let's follow the pattern:
                for child in getattr(obj, "children", []):
                    child_clone = self._clone(child)
                    if child_clone is not None:
                        self._attach(cloned_tuple, child_clone)
                        
                return cloned_tuple
            else:
                cloned = llm_simulation.Node(
                    obj.name,
                    self._next_op_id(),
                    obj.hw_id,
                    obj.duration,
                    fwd=obj.fwd,
                )


            # following _expand_transformer_node logic, we need to find siblings that are zero3 tp_shard=True.
            zero3_attachments: List[llm_simulation.Edge] = []
            for parent in getattr(obj, "parents", []):
                for sibling in getattr(parent, "children", []):
                    if sibling is obj:
                        continue
                    if self._should_shard_zero3_transformer(sibling):
                        zero3_attachments.append(sibling)


            hw_ids = []
            for tp_rank in range(self._par_degree):
                hw_ids.append(self._hw_id_for_rank(obj.hw_id, tp_rank))

            per_rank_edges: List[llm_simulation.Edge] = []
            for zero3_attachment in zero3_attachments:
                per_rank_edges = self._ensure_zero3_per_rank_edges(zero3_attachment, rank_heads=None, rank_tails=None, hw_ids=hw_ids)
                self._clone_cache[id(zero3_attachment)] = per_rank_edges

            self._clone_cache[obj_id] = cloned
            self._copy_metadata(obj, cloned)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    if per_rank_edges:
                        for per_rank_edge in per_rank_edges:
                            self._attach(per_rank_edge, child_clone)
                    self._attach(cloned, child_clone)
            return cloned

        if isinstance(obj, llm_simulation.Edge):
            cloned_edge = llm_simulation.Edge(
                obj.name,
                self._next_op_id(),
                obj.duration,
                is_dp=getattr(obj, "is_dp", False),
                comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                comm_type=getattr(obj, "comm_type", None),
                participants=getattr(obj, "participants", 1),
                comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
            )
            cloned_edge.local_hw_id = getattr(obj, "local_hw_id", None)
            self._clone_cache[obj_id] = cloned_edge
            self._copy_metadata(obj, cloned_edge)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_edge, child_clone)
            return cloned_edge

        if isinstance(obj, llm_simulation.Data_batch):
            cloned_batch = llm_simulation.Data_batch(obj.name, obj.batch_id, obj.duration)
            self._clone_cache[obj_id] = cloned_batch
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_batch, child_clone)
            return cloned_batch

        raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

    def _propagate_local_hw_ids(self, roots: Any) -> None:
        stack: List[Any] = []
        if isinstance(roots, (list, tuple)):
            stack.extend(list(roots))
        elif roots is not None:
            stack.append(roots)
        visited: Set[int] = set()
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            children = getattr(obj, "children", [])
            if isinstance(children, (list, tuple)):
                stack.extend(children)
            elif children is not None:
                stack.append(children)

            parents = getattr(obj, "parents", None)
            if parents:
                if isinstance(parents, (list, tuple)):
                    stack.extend(parents)
                else:
                    stack.append(parents)

            if isinstance(obj, llm_simulation.Edge):
                if getattr(obj, "comm_type", None) == CollectiveType.PIPELINE:
                    continue
                if getattr(obj, "tp_shard", False) and getattr(obj, "tp_rank", None) is not None:
                    # Preserve per-rank ZeRO-3 gather placement; parents may be shared across TP ranks.
                    continue
                new_hw = None
                for parent in getattr(obj, "parents", []) or []:
                    parent_hw = getattr(parent, "hw_id", None)
                    if parent_hw is None:
                        parent_hw = getattr(parent, "local_hw_id", None)
                    if parent_hw is not None and parent_hw >= 0:
                        new_hw = parent_hw
                        break
                if new_hw is None:
                    stage_id = getattr(obj, "stage_id", None)
                    if stage_id is not None:
                        try:
                            new_hw = self._hw_id_for_rank(stage_id, 0)
                        except Exception:
                            new_hw = None
                if new_hw is not None and new_hw >= 0:
                    obj.local_hw_id = new_hw


    def _expand_transformer_node(self, node: llm_simulation.Node) -> Tuple[Any, ...]:
        node_id = id(node)
        if node_id in self._clone_cache:
            cached_entry = self._clone_cache[node_id]
            if isinstance(cached_entry, (list, tuple)):
                return tuple(cached_entry)

        stage_id = getattr(node, "stage_id", node.hw_id)
        micro_batch = getattr(node, "micro_batch_index", None)
        layer_index = getattr(node, "layer_index", None)
        direction = getattr(node, "direction", "forward" if node.fwd else "backward")
        is_moe_layer = bool(getattr(node, "is_moe_layer", False))
        graph_for_layer = (
            self.moe_transformer_graph if is_moe_layer and self.moe_transformer_graph else self.transformer_graph
        )
        gemm_entries = (
            self._gemm_entries_moe if is_moe_layer and self._gemm_entries_moe else self._gemm_entries
        )

        rank_heads: List[Any] = []
        rank_tails: List[Any] = []

        for tp_rank in range(self._par_degree):
            previous: Optional[Any] = None
            head: Optional[Any] = None
            hw_id = self._hw_id_for_rank(stage_id, tp_rank)

            gemm_iterable = gemm_entries
            if direction == "backward":
                gemm_iterable = list(reversed(gemm_entries))

            for gemm_idx, entry in enumerate(gemm_iterable):
                entry_name = entry.get("name", f"g{gemm_idx}")
                cfg = entry.get(direction, {})
                duration = cfg.get("duration")
                if duration is None:
                    raise ValueError(
                        f"Missing duration for transformer entry '{entry_name}' in direction '{direction}'"
                    )

                gemm_node = llm_simulation.Node(
                    name=self._format_gemm_name(entry_name, direction, micro_batch, layer_index, tp_rank),
                    op_id=self._next_op_id(),
                    hw_id=hw_id,
                    duration=duration,
                    fwd=(direction == "forward"),
                    mem_kind=mem_kind_from_op_name(entry_name),
                )
                gemm_node.stage_id = stage_id
                gemm_node.tp_rank = tp_rank
                gemm_node.micro_batch_index = micro_batch
                gemm_node.layer_index = layer_index
                gemm_node.direction = direction
                gemm_node.recompute = bool(getattr(node, "recompute", False))
                gemm_node.param_gather = (gemm_idx == 0)
                gemm_node.is_moe_layer = is_moe_layer

                if previous is not None:
                    previous.add_child(gemm_node)
                previous = gemm_node
                if head is None:
                    head = gemm_node

                for comm_key in cfg.get("comm_keys", []):
                    comm_edge = self._create_transformer_comm_edge(
                        comm_key,
                        hw_id,
                        stage_id,
                        micro_batch,
                        layer_index,
                        direction,
                        tp_rank,
                        graph_override=graph_for_layer,
                    )
                    previous.add_child(comm_edge)
                    previous = comm_edge

            if head is None:
                raise ValueError("Transformer expansion produced no GEMM nodes")

            rank_heads.append(head)
            rank_tails.append(previous or head)

        dp_children: List[Any] = []
        other_children: List[Any] = []
        zero3_attachments: List[llm_simulation.Edge] = []

        for child in getattr(node, "children", []):
            comm_type = getattr(child, "comm_interconnect_type", None)
            if comm_type == "dp":
                dp_children.append(child)
            else:
                other_children.append(child)

        for parent in getattr(node, "parents", []):
            for sibling in getattr(parent, "children", []):
                if sibling is node:
                    continue
                if self._should_shard_zero3_transformer(sibling):
                    zero3_attachments.append(sibling)

        # Keep the main trunk pointing to the per-rank compute tails.
        downstream_parents: List[Any] = list(rank_tails)

        # Attach DP collectives as side branches from the compute tails, without
        # reparenting the trunk. This preserves the true cross-layer pipeline
        # edge between compute nodes for ET conversion.
        for child in dp_children:
            if self._should_shard_zero3_transformer(child):
                per_rank_edges = self._ensure_zero3_per_rank_edges(child, rank_heads=rank_heads, rank_tails=rank_tails, hw_ids=None)

            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(rank_tails[0], child_clone) # only attach to the first tail for DP collectives

        # Non-DP edges (e.g., cross_layer) stay on the trunk so (parent, target)
        # compute → compute pipeline edges remain visible.
        for child in other_children:
            # Special-case marked cross_layer edges (set in original graph):
            # create one per TP rank and wire tail[r] -> cross_layer_r -> next_head[r].
            is_pipeline_edge = False
            if isinstance(child, llm_simulation.Edge):
                comm_type = getattr(child, "comm_type", None)
                if comm_type == CollectiveType.PIPELINE:
                    is_pipeline_edge = True
            if is_pipeline_edge:
                # Determine per-rank byte size (ceil split)
                try:
                    total_bytes = int(getattr(child, "comm_size_bytes", 0))
                except Exception:
                    total_bytes = 0
                per_rank_bytes = int(math.ceil(float(total_bytes) / float(max(1, self._par_degree))))

                # Clone the original targets of this pipeline edge
                target_clones: List[Any] = []
                for tgt in getattr(child, "children", []):
                    tgt_clone = self._clone(tgt)
                    if tgt_clone is None:
                        continue
                    target_clones.append(tgt_clone)
                if not target_clones:
                    # No downstream target; skip safely
                    continue

                # For each TP rank, create its own pipeline edge and connect
                for r, tail in enumerate(rank_tails):
                    # Create rank-specific pipeline edge
                    edge_obj = llm_simulation.Edge(
                        name=f"{getattr(child, 'name', '')}_rank{r}",
                        op_id=self._next_op_id(),
                        duration=0,
                        is_dp=False,
                        comm_size_bytes=per_rank_bytes,
                        comm_type=CollectiveType.PIPELINE,
                        participants=2,
                        comm_interconnect_type="pp",
                    )
                    edge_obj.is_cross_layer = True
                    tail.add_child(edge_obj)
                    # Also anchor to the compute node (two parents) for mapping clarity
                    last_compute = rank_heads[r]
                    # Find the nearest compute ancestor for this rank: walk back from tail if needed
                    compute_anchor = None
                    cur = tail
                    visited_ids = set()
                    while cur is not None and id(cur) not in visited_ids:
                        visited_ids.add(id(cur))
                        if isinstance(cur, llm_simulation.Node):
                            compute_anchor = cur
                            break
                        parents = getattr(cur, "parents", [])
                        cur = parents[-1] if parents else None
                    if compute_anchor is not None and compute_anchor is not edge_obj:
                        if compute_anchor != tail:
                            compute_anchor.add_child(edge_obj)

                    # Connect to each cloned target, aligning ranks where possible
                    for tgt_clone in target_clones:
                        if isinstance(tgt_clone, (list, tuple)):
                            # Map by identity index when available
                            idx = r % len(tgt_clone)
                            edge_obj.add_child(tgt_clone[idx])
                        else:
                            edge_obj.add_child(tgt_clone)

                continue

            # Default path for non-pipeline children
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            
            # Special handling for optimizer nodes: 1-to-1 connection
            if isinstance(child, llm_simulation.Node) and "optimizer" in child.name and isinstance(child_clone, (list, tuple)):
                if len(child_clone) == len(rank_tails):
                    for r in range(len(rank_tails)):
                        rank_tails[r].add_child(child_clone[r])
                    continue
            
            self._attach(downstream_parents, child_clone)

        for zero3_edge in zero3_attachments:
            is_cross_device = zero3_edge.local_hw_id != stage_id
            per_rank_edges = self._ensure_zero3_per_rank_edges(zero3_edge, cross_device=is_cross_device, rank_heads=rank_heads, rank_tails=rank_tails, hw_ids=None)
            self._clone_cache[id(zero3_edge)] = per_rank_edges

        heads_tuple = tuple(rank_heads)
        self._clone_cache[node_id] = heads_tuple
        return heads_tuple

    def _create_transformer_comm_edge(
        self,
        comm_key: str,
        hw_id: int,
        stage_id: int,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        direction: str,
        tp_rank: int,
        *,
        graph_override: Optional[Graph] = None,
    ) -> llm_simulation.Edge:
        graph = graph_override or self.transformer_graph
        comm_info = graph.comm_metadata.get(comm_key, {})
        is_dp_edge = comm_info.get("interconnect_type") == "dp"

        comm_edge = graph.create_comm_edge(
            name=comm_key,
            op_id=self._next_op_id(),
            comm_key=comm_key,
            is_dp=is_dp_edge,
            local_hw_id=hw_id,
        )
        comm_edge.stage_id = stage_id
        comm_edge.micro_batch_index = micro_batch
        comm_edge.layer_index = layer_index
        comm_edge.direction = direction
        comm_edge.tp_rank = tp_rank
        return comm_edge

    def _copy_metadata(self, source: Any, target: Any) -> None:
        for attr in (
            "micro_batch_index",
            "layer_index",
            "direction",
            "stage_id",
            "tp_rank",
            "mem_kind",
            "recompute",
            "is_moe_layer",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _attach(self, parent: Any, child: Any) -> None:
        if parent is None or child is None:
            return

        if isinstance(parent, (list, tuple)):
            for item in parent:
                self._attach(item, child)
            return

        if isinstance(child, (list, tuple)):
            for item in child:
                self._attach(parent, item)
            return

        parent.add_child(child)

    def _format_gemm_name(
        self,
        base_name: str,
        direction: str,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        tp_rank: int,
    ) -> str:
        return f"{base_name}_{direction}_mb{micro_batch}_l{layer_index}_rank{tp_rank}"

    def _next_op_id(self) -> int:
        self._op_id_counter += 1
        return self._op_id_counter

    def _hw_id_for_rank(self, stage_id: int, tp_rank: int) -> int:
        stage_int = int(stage_id) if stage_id is not None else 0
        tp_rank_int = int(tp_rank)
        if tp_rank_int < 0 or tp_rank_int >= self._par_degree:
            raise ValueError(f"tp_rank {tp_rank_int} is out of range for par_degree {self._par_degree}")
        if not self._layout_axis_order:
            return stage_int * self._par_degree + tp_rank_int

        coords: Dict[str, int] = {}
        if "tp" in self._layout_axis_order:
            coords["tp"] = tp_rank_int % self._tp_size
        if "cp" in self._layout_axis_order:
            coords["cp"] = (tp_rank_int // self._tp_size) % self._cp_size
        if "ep" in self._layout_axis_order:
            coords["ep"] = (tp_rank_int // max(1, self._tp_size * self._cp_size)) % self._ep_size
        if "pp" in self._layout_axis_order:
            if stage_int < 0 or stage_int >= self._pp_size:
                raise ValueError(f"stage_id {stage_int} is out of range for pp={self._pp_size}")
            coords["pp"] = stage_int % self._pp_size

        linear_rank = 0
        for axis in self._layout_axis_order:
            coord = coords.get(axis, 0)
            size = self._layout_axis_sizes.get(axis, 1)
            if coord < 0 or coord >= size:
                raise ValueError(f"Coordinate {coord} for axis '{axis}' is out of range <{size}")
            stride = self._layout_axis_strides.get(axis)
            if stride is None:
                raise KeyError(f"Rank layout stride missing for axis '{axis}'")
            linear_rank += coord * stride
        return linear_rank
    
    
class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,  # somehow this works? TODO: fix this at some point so its not annotated by IDE.
        pipeline_graph: Graph,
        pipeline_root: Any,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_graph: Optional[Graph] = None,
        transformer_forward_root: Optional[Any] = None,
        transformer_backward_root: Optional[Any] = None,
        moe_transformer_graph: Optional[Graph] = None,
        moe_transformer_forward_root: Optional[Any] = None,
        moe_transformer_backward_root: Optional[Any] = None,
        no_data_parallel: bool = False,
    ) -> None:
        self.time_calc = time_calc
        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.interconnect_params = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.moe_transformer_graph = moe_transformer_graph
        self.moe_transformer_forward_root = moe_transformer_forward_root
        self.moe_transformer_backward_root = moe_transformer_backward_root
        self.flattened_root: Optional[Any] = None
        self._transformer_rank_layout: Dict[str, Any] = {}
        self._pipeline_rank_layout: Dict[str, Any] = {}
        self._network_dimensions: Tuple[Any, ...] = tuple()
        self._axis_dimension_map: Dict[str, int] = {}
        self._transformer_stage_dp_faults: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {}
        self._transformer_stage_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_stage_moe_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_baseline_timings: Optional[TransformerTimings] = None
        self._transformer_moe_baseline_timings: Optional[TransformerTimings] = None
        self.no_data_parallel = bool(no_data_parallel)
        self._rank_layout = self._build_rank_layout_descriptor()
        self._first_dim_optimize_cfg: Optional[Dict[str, Any]] = getattr(self, "_first_dim_optimize_cfg", None)
        self._fault_space: Optional[FaultSpace] = None
        self._fault_projections: Dict[str, FaultProjectionResult] = {}
        self._initialize_fault_mappings()

    def _build_rank_layout_descriptor(self) -> Dict[str, Any]:
        hw_config = getattr(self.time_calc, "hw_config", None)
        layout = getattr(hw_config, "network_layout", None)
        dimensions = getattr(layout, "dimensions", None) if layout is not None else None
        if not dimensions:
            return {}
        self._network_dimensions = tuple(dimensions)
        optimize_cfg: Optional[Dict[str, Any]] = None
        for idx, dim in enumerate(dimensions):
            if getattr(dim, "optimize_2dmap", False):
                if idx != 0:
                    raise ValueError("optimize_2dmap is only supported on the first network dimension.")
                if optimize_cfg is not None:
                    raise ValueError("Multiple network dimensions requested optimize_2dmap; only one is supported.")
                topo_type = getattr(dim, "topology_type", None)
                if not topo_type:
                    raise ValueError("optimize_2dmap requires a topology type on the target dimension.")
                size_value = getattr(dim, "size", None)
                if size_value is None:
                    raise ValueError("optimize_2dmap requires an explicit dimension size.")
                dims_value = getattr(dim, "size_2d", None)
                if dims_value is not None:
                    dims_value = (int(dims_value[0]), int(dims_value[1]))
                optimize_cfg = {
                    "dimension_index": idx,
                    "topology": str(topo_type),
                    "size": int(size_value),
                    "parallelisms": tuple(getattr(dim, "parallelisms", ()) or ()),
                }
                if dims_value:
                    optimize_cfg["dims"] = dims_value
        self._first_dim_optimize_cfg = optimize_cfg

        def _safe_int(value: Any, default: int = 1) -> int:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                candidate = default
            return max(1, candidate)

        tp_size = _safe_int(getattr(self.pipeline_graph, "tp", getattr(self.time_calc, "tp", 1)))
        cp_size = _safe_int(getattr(self.pipeline_graph, "cp", getattr(self.time_calc, "cp", 1)))
        ep_size = _safe_int(getattr(self.time_calc, "ep", 1))
        pp_size = _safe_int(getattr(self.pipeline_graph, "pp", getattr(self.time_calc, "pp", 1)))
        dp_size = _safe_int(getattr(self.time_calc, "dp", 1))

        axis_sizes: Dict[str, int] = {"tp": tp_size, "cp": cp_size, "ep": ep_size, "pp": pp_size, "dp": dp_size}
        axis_order: List[str] = []

        # Enforce axis ordering for hierarchical/hybrid modes: the first active
        # dimension must contain exactly {'tp','cp','ep'} (or the active subset).
        # Subsequent active
        # dimensions may contain 'pp' (optionally combined with 'dp'). This
        # matches the assumptions in the hierarchical graphs where a stage is a
        # TP/CP/EP cluster replicated across PP (and potentially DP) axes.
        enforce_layout = self.time_calc.execution_mode in {
            ExecutionMode.HYBRID,
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
        }
        first_active_checked = False

        for dim in dimensions:
            dim_axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            declared = int(getattr(dim, "size", 1))

            axes_without_dp = [axis for axis in dim_axes if axis != "dp"]
            if enforce_layout and declared > 1 and not first_active_checked and axes_without_dp:
                active_axes = [axis for axis in ("tp", "cp", "ep") if axis_sizes[axis] > 1]
                canon_active = sorted([axis for axis in axes_without_dp if axis_sizes[axis] > 1])
                expected = sorted(active_axes)
                if expected and canon_active != expected:
                    raise ValueError(
                        "For hierarchical/hybrid AstraSim modes, the first active network "
                        "dimension must contain the active TP/CP/EP axes to represent the transformer cluster."
                    )
                first_active_checked = True

            for name in dim_axes:
                if name not in axis_sizes:
                    raise ValueError(
                        f"Unsupported parallelism axis '{name}' in network layout. "
                        "Supported axes for AstraSim integration are: tp, cp, ep, pp, dp."
                    )
                if name not in axis_order:
                    axis_order.append(name)

            expected = 1
            for axis_name in dim_axes:
                expected *= axis_sizes.get(axis_name, 1)
            if expected != declared:
                raise ValueError(
                    f"Network dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' "
                    f"size mismatch: declared {declared}, but parallelism factors imply {expected}."
                )

        if axis_order:
            ordered_axes = ["tp", "cp", "ep", "pp", "dp"]
            axis_order = [axis for axis in ordered_axes if axis in axis_order]

        # Ensure the layout covers active parallel axes
        if tp_size > 1 and "tp" not in axis_order:
            raise ValueError("Network layout must include 'tp' when tensor parallelism > 1.")
        if cp_size > 1 and "cp" not in axis_order:
            raise ValueError("Network layout must include 'cp' when context parallelism > 1.")
        if ep_size > 1 and "ep" not in axis_order:
            raise ValueError("Network layout must include 'ep' when expert parallelism > 1.")
        if pp_size > 1 and "pp" not in axis_order:
            raise ValueError("Network layout must include 'pp' when pipeline parallelism > 1.")

        axis_strides: Dict[str, int] = {}
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes[axis]

        descriptor = {
            "axis_order": axis_order,
            "axis_sizes": axis_sizes,
            "axis_strides": axis_strides,
            "stage_span": span,
        }

        def _subset_layout(allowed: Sequence[str]) -> Optional[Dict[str, Any]]:
            subset = [axis for axis in axis_order if axis in allowed and axis_sizes.get(axis, 1) >= 1]
            if not subset:
                return None
            strides: Dict[str, int] = {}
            span = 1
            for axis in subset:
                strides[axis] = span
                span *= axis_sizes[axis]
            return {
                "axis_order": subset,
                "axis_sizes": {axis: axis_sizes[axis] for axis in subset},
                "axis_strides": strides,
                "stage_span": span,
            }

        # Transformer graphs only encode TP/CP axes; pipeline graphs encode PP
        # (DP replicas are handled externally). Store these subsets so callers
        # can attach the appropriate layout before invoking AstraSim.
        self._transformer_rank_layout = _subset_layout(["tp", "cp", "ep"])
        self._pipeline_rank_layout = _subset_layout(["pp", "dp"])

        return descriptor

    def _attach_optimize_hint(self, root: Any) -> None:
        if root is None:
            return
        if self._first_dim_optimize_cfg:
            setattr(root, "_optimize_2dmap", dict(self._first_dim_optimize_cfg))
        elif hasattr(root, "_optimize_2dmap"):
            delattr(root, "_optimize_2dmap")

    def _log_fault_summary(self, axis_order: Sequence[str], axis_sizes: Mapping[str, int]) -> None:
        if not axis_order:
            return
        space = self._fault_space
        if space is None or not space.entries:
            return
        network_dims = getattr(self, "_network_dimensions", tuple())
        if not network_dims:
            log_message("[RAPID-LLM][faults] Hardware dimensions unavailable; skipping fault summary.")
            return

        def coords_to_dict(coords: Tuple[Tuple[str, int], ...]) -> Dict[str, int]:
            return {name: value for name, value in coords}

        entries = space.entries
        for dim_index, dim in enumerate(network_dims):
            axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            if not axes:
                continue
            replication_axes: List[str] = []
            for future_dim in network_dims[dim_index + 1 :]:
                replication_axes.extend(
                    str(axis).strip().lower() for axis in getattr(future_dim, "parallelisms", ()) if axis
                )
            total_clusters = 1
            for axis in replication_axes:
                total_clusters *= max(1, axis_sizes.get(axis, 1))
            axes_label = ", ".join(axes) or "<none>"
            table_rows: List[Tuple[str, str, float]] = []
            for entry in entries:
                if not any(axis in axes for axis in entry.affected_axes):
                    continue
                src_dict = coords_to_dict(entry.src_coords)
                dst_dict = coords_to_dict(entry.dst_coords)
                src_label = f"{entry.original[0]} (" + ", ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in axes) + ")"
                dst_label = f"{entry.original[1]} (" + ", ".join(f"{axis}={dst_dict.get(axis, 0)}" for axis in axes) + ")"
                higher_coords = " ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in replication_axes)
                table_rows.append((higher_coords, f"{src_label} <-> {dst_label}", float(entry.original[2])))

            if not table_rows:
                log_message(f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: 0 / {total_clusters}", category="faults")
                continue

            log_message(
                f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: {len(table_rows)} / {total_clusters or 1}",
                category="faults",
            )
            header_axes = " ".join(replication_axes) if replication_axes else ""
            log_message(f"  {header_axes:<12} | {'source ↔ dest':<40} | derate", category="faults")
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")

            for higher_str, pair, derate in table_rows:
                log_message(
                    f"  {higher_str:<12} | {pair:<40} | {derate:>6.2f}",
                    category="faults",
                )
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")




    def _initialize_fault_mappings(self) -> None:
        hw_config = getattr(self.time_calc, "hw_config", None)
        network_layout = getattr(hw_config, "network_layout", None)
        faulty_links: Tuple[Tuple[int, int, float], ...] = tuple(
            getattr(network_layout, "faulty_links", ()) or ()
        )
        if faulty_links and self.time_calc.execution_mode in {ExecutionMode.ANALYTICAL, ExecutionMode.HYBRID}:
            raise ValueError("Faulty links require full AstraSim execution; analytical/hybrid modes are not supported.")
        axis_layout = axis_layout_from_descriptor(self._rank_layout)
        axis_dim_map = self._axis_to_dimension_map(network_layout)
        self._axis_dimension_map = dict(axis_dim_map)
        self._fault_space = FaultSpace(
            axis_layout,
            faulty_links,
            axis_to_dimension=axis_dim_map,
        )
        self._fault_projections = self._build_fault_projections_from_space(self._fault_space)
        self._validate_fault_coverage()
        self._transformer_stage_dp_faults = self._build_transformer_stage_dp_fault_map()
        axis_order = self._rank_layout.get("axis_order", []) if isinstance(self._rank_layout, dict) else []
        axis_sizes = self._rank_layout.get("axis_sizes", {}) if isinstance(self._rank_layout, dict) else {}
        self._log_fault_summary(axis_order, axis_sizes)

    def _axis_to_dimension_map(self, network_layout) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        if network_layout is None:
            return mapping
        dimensions = getattr(network_layout, "dimensions", None)
        if not dimensions:
            return mapping
        for idx, dim in enumerate(dimensions):
            for axis in getattr(dim, "parallelisms", ()) or ():
                normalized = str(axis).strip().lower()
                if normalized:
                    mapping[normalized] = idx
        return mapping

    def _build_fault_projections_from_space(
        self,
        space: FaultSpace,
    ) -> Dict[str, FaultProjectionResult]:
        projections: Dict[str, FaultProjectionResult] = {}
        if not space.entries:
            return projections

        global_axes = tuple(space.layout.axis_order)
        if global_axes:
            projections["global"] = space.project(global_axes)

        transformer_axes: Tuple[str, ...] = tuple(
            self._transformer_rank_layout.get("axis_order", ())
        ) if self._transformer_rank_layout else tuple()
        if transformer_axes:
            projections["transformer"] = space.project(transformer_axes)

        pipeline_axes: Tuple[str, ...] = tuple(
            self._pipeline_rank_layout.get("axis_order", ())
        ) if self._pipeline_rank_layout else tuple()
        if pipeline_axes:
            projections["pipeline"] = space.project(pipeline_axes)

        return projections

    def _validate_fault_coverage(self) -> None:
        if self._fault_space is None:
            return
        covered: Set[Tuple[int, int, float]] = set()
        for label in ("transformer", "pipeline"):
            proj = self._fault_projections.get(label)
            if proj:
                covered.update(proj.covered_originals)
        uncovered = [
            entry.original
            for entry in self._fault_space.entries
            if entry.original not in covered
        ]
        if uncovered:
            formatted = ", ".join(str(item) for item in uncovered)
            raise ValueError(
                "Faulty links do not map to any transformer or pipeline axis subset: "
                f"{formatted}"
            )

    def _axis_value_from_coords(self, coords: Tuple[Tuple[str, int], ...], axis: str) -> int:
        for name, value in coords:
            if name == axis:
                return int(value)
        layout = self._rank_layout or {}
        if isinstance(layout, dict):
            layout_axes = layout.get("axis_order", [])
            if axis not in layout_axes:
                return 0
            axis_sizes = layout.get("axis_sizes", {})
        else:
            axis_sizes = {}
        if axis_sizes.get(axis, 1) <= 1:
            return 0
        raise ValueError(f"Axis '{axis}' not present in coordinate tuple for faulty link.")

    def _build_transformer_stage_dp_fault_map(self) -> Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]]:
        projection = self._fault_projection_for("transformer")
        if not projection:
            return {}
        stage_dp_faults: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
        for detail in projection.entries:
            src_stage = self._axis_value_from_coords(detail.src_coords, "pp")
            dst_stage = self._axis_value_from_coords(detail.dst_coords, "pp")
            if src_stage != dst_stage:
                raise ValueError(
                    "Transformer faulty link spans multiple pipeline stages; "
                    "hierarchical execution requires faults limited to a single stage."
                )
            src_dp = self._axis_value_from_coords(detail.src_coords, "dp")
            dst_dp = self._axis_value_from_coords(detail.dst_coords, "dp")
            affected_dp = {src_dp, dst_dp}
            for dp_idx in affected_dp:
                stage_dp_faults.setdefault((dp_idx, src_stage), []).append(detail.remapped)
        return {key: tuple(links) for key, links in stage_dp_faults.items()}

    def _fault_projection_for(self, label: str) -> Optional[FaultProjectionResult]:
        return self._fault_projections.get(label)

    def _fault_links_for(self, label: str) -> Tuple[Tuple[int, int, float], ...]:
        projection = self._fault_projection_for(label)
        if projection is None:
            return tuple()
        return projection.remapped_links

    def _fault_override(self, label: str) -> Optional[Tuple[Tuple[int, int, float], ...]]:
        if self._fault_space is None:
            return None
        return self._fault_links_for(label)

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            return self._run_full_astrasim_flattened()

    def _run_pipeline_with_analytical_comm(self, declared_mode: ExecutionMode) -> ExecutionResult:
        if declared_mode == ExecutionMode.HYBRID:
            if self.no_data_parallel:
                filename = "/hybrid_graph_no_dp"
            else:
                filename = "/hybrid_graph"
            timed_root = self.pipeline_root
        else: # must be "ANALYTICAL"
            if self.no_data_parallel:
                filename = "/analytical_graph_no_dp"
            else:
                filename = "/analytical_graph"
        timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
            self.pipeline_root,
            self.time_calc.network_model,
            self.interconnect_params,
        )
            
        generate_graphs = _env_flag("RAPID_VISUALIZE_GRAPHS")
        if generate_graphs:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                filename,
            )

        # Persist timed root for any downstream consumer
        self.pipeline_root = timed_root
        total_time = self.pipeline_graph.simulate(timed_root)
        return ExecutionResult(total_time=total_time, graph_root=timed_root, mode=declared_mode)

    def _run_hybrid(self) -> ExecutionResult:
        generate_graphs = _env_flag("RAPID_VISUALIZE_GRAPHS")

        transformer_time, moe_transformer_time = self._run_transformer_astrasim(ExecutionMode.HYBRID)

        
        if generate_graphs:
            transformer_timed_forward_root = self.transformer_graph.convert_comm_sizes_to_times(
                self.transformer_forward_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            transformer_timed_backward_root = self.transformer_graph.convert_comm_sizes_to_times(
                self.transformer_backward_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            self.transformer_graph.save_graph(
                transformer_timed_forward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_forward",
            )
            self.transformer_graph.save_graph(
                transformer_timed_backward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_backward",
            )
            if self.moe_transformer_graph and self.moe_transformer_forward_root:
                moe_forward_root = self.moe_transformer_graph.convert_comm_sizes_to_times(
                    self.moe_transformer_forward_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )
                self.moe_transformer_graph.save_graph(
                    moe_forward_root,
                    self.time_calc.output_dir,
                    "/hybrid_graph_transformer_forward_moe",
                )
            if self.moe_transformer_graph and self.moe_transformer_backward_root:
                moe_backward_root = self.moe_transformer_graph.convert_comm_sizes_to_times(
                    self.moe_transformer_backward_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )
                self.moe_transformer_graph.save_graph(
                    moe_backward_root,
                    self.time_calc.output_dir,
                    "/hybrid_graph_transformer_backward_moe",
                )

        if transformer_time is not None or moe_transformer_time is not None:
            self._apply_transformer_time(transformer_time, moe_transformer_time)
        return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID)

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        transformer_time, moe_transformer_time = self._run_transformer_astrasim(ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)
        if transformer_time is not None or moe_transformer_time is not None:
            self._apply_transformer_time(transformer_time, moe_transformer_time)

        if _env_flag("RAPID_VISUALIZE_GRAPHS") and self.transformer_graph:
            transformer_timed_forward_root = self.transformer_graph.convert_comm_sizes_to_times(
                self.transformer_forward_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            self.transformer_graph.save_graph(
                transformer_timed_forward_root,
                self.time_calc.output_dir,
                "/hierarchical_graph_transformer_forward",
            )
            if self.transformer_backward_root is not None:
                transformer_timed_backward_root = self.transformer_graph.convert_comm_sizes_to_times(
                    self.transformer_backward_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )
                self.transformer_graph.save_graph(
                    transformer_timed_backward_root,
                    self.time_calc.output_dir,
                    "/hierarchical_graph_transformer_backward",
                )
            if self.moe_transformer_graph and self.moe_transformer_forward_root:
                moe_forward_root = self.moe_transformer_graph.convert_comm_sizes_to_times(
                    self.moe_transformer_forward_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )
                self.moe_transformer_graph.save_graph(
                    moe_forward_root,
                    self.time_calc.output_dir,
                    "/hierarchical_graph_transformer_forward_moe",
                )
            if self.moe_transformer_graph and self.moe_transformer_backward_root:
                moe_backward_root = self.moe_transformer_graph.convert_comm_sizes_to_times(
                    self.moe_transformer_backward_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )
                self.moe_transformer_graph.save_graph(
                    moe_backward_root,
                    self.time_calc.output_dir,
                    "/hierarchical_graph_transformer_backward_moe",
                )

        dp_count = getattr(self.time_calc, "dp", 1) or 1
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for AstraSim execution")

        # Use hierarchical artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        pipeline_fault_override = self._fault_override("pipeline")
        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
            "faulty_links_override": pipeline_fault_override,
        }
        # Inference runs force AstraSim dp_override=1; use run_type here to keep
        # transformer durations scalar without threading override through call sites.
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        if _env_flag("RAPID_VISUALIZE_GRAPHS") and self.pipeline_root is not None:
            filename = "/pipeline_graph_hierarchical_no_dp" if self.no_data_parallel else "/pipeline_graph_hierarchical"
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                filename,
            )

        pipeline_layout = getattr(self, "_pipeline_rank_layout", None)
        if pipeline_layout:
            setattr(self.pipeline_root, "_astrasim_rank_layout", pipeline_layout)
        elif hasattr(self.pipeline_root, "_astrasim_rank_layout"):
            delattr(self.pipeline_root, "_astrasim_rank_layout")
        self._attach_optimize_hint(self.pipeline_root)
        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            self.pipeline_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )
        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        return ExecutionResult(total_time=max_sec, graph_root=self.pipeline_root, mode=ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        if self.moe_transformer_graph is not None:
            raise NotImplementedError("MoE is not supported with full AstraSim flattened execution.")
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for flattening")
        if not self.transformer_graph:
            raise RuntimeError("Transformer graph metadata is required for flattening")

        flattener = PipelineGraphFlattener(
            pipeline_graph=self.pipeline_graph,
            transformer_graph=self.transformer_graph,
            moe_transformer_graph=self.moe_transformer_graph,
            rank_layout=self._rank_layout,
        )

        if _env_flag("RAPID_VISUALIZE_GRAPHS") and self.pipeline_root is not None :
            filename = "/pipeline_graph_pre_flatten_no_dp" if self.no_data_parallel else "/pipeline_graph_pre_flatten"
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                filename,
            )
        flattened_root = flattener.build(self.pipeline_root)
        if flattened_root is None:
            raise RuntimeError("Pipeline flattening produced an empty graph")

        flattened_root = apply_overlap_transforms(
            flattened_root,
            self.time_calc.get_parallelism_mode(),
            getattr(self.time_calc, "tp_overlap", 0.0),
            getattr(self.time_calc, "tp_sp_overlap", 0.0),
            getattr(self.time_calc, "cp_overlap", 0.0),
        )
        flattener._propagate_local_hw_ids(flattened_root)
        setattr(flattened_root, "_astrasim_rank_layout", self._rank_layout)
        self._attach_optimize_hint(flattened_root)
        self.time_calc.flattened_pipeline_root = flattened_root
        if _env_flag("RAPID_VISUALIZE_GRAPHS") and self.pipeline_root is not None:
            filename = "/pipeline_graph_post_flatten_no_dp" if self.no_data_parallel else "/pipeline_graph_post_flatten"
            self.pipeline_graph.save_graph(
                flattened_root,
                self.time_calc.output_dir,
                filename,
            )
        self.pipeline_root = flattened_root
        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_flattened")
        # dot = visualize_graph(flattened_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="svg", cleanup=True)
        # except Exception as exc:  # pragma: no cover - visualization best-effort
        #     print(f"[WARN] Failed to render flattened pipeline graph: {exc}")

        unique_hw_ids = self._collect_hw_ids(flattened_root)
        if not unique_hw_ids:
            raise RuntimeError("Flattened pipeline graph exposes no compute nodes with hardware IDs")

        # Use flattened artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
        }
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            flattened_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings")

        expected_rank_count = effective_dp * len(unique_hw_ids)

        # Special case: If expected rank count is 1, then 2 is fine, but we prune the extra result
        # this is done, since astrasim backend only supports >1 ranks, so we generate extra fake result for that case.
        if expected_rank_count == 1:
            if len(per_rank_sec) > 2:
                raise RuntimeError(
                    "AstraSim rank count mismatch for flattened execution: "
                    f"expected {expected_rank_count}, got {len(per_rank_sec)}"
                )
            per_rank_sec = per_rank_sec[:1]
        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration")

        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        self.time_calc.flattened_astrasim_per_rank = per_rank_sec
        self.time_calc.flattened_astrasim_total = max_sec

        return ExecutionResult(
            total_time=max_sec,
            graph_root=flattened_root,
            mode=ExecutionMode.FULL_ASTRASIM_FLATTENED,
        )

    def build_flattened_root_for_memory(self) -> Any:
        if self.flattened_root is not None:
            return self.flattened_root

        existing_flattened = getattr(self.time_calc, "flattened_pipeline_root", None)
        if existing_flattened is not None:
            self.flattened_root = existing_flattened
            return existing_flattened

        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for memory flattening")
        if not self.transformer_graph:
            raise RuntimeError("Transformer graph metadata is required for memory flattening")

        flattener = PipelineGraphFlattener(
            pipeline_graph=self.pipeline_graph,
            transformer_graph=self.transformer_graph,
            moe_transformer_graph=self.moe_transformer_graph,
            rank_layout=self._rank_layout,
        )
        flattened_root = flattener.build(self.pipeline_root)
        if flattened_root is None:
            raise RuntimeError("Memory flattening produced an empty graph")

        flattened_root = apply_overlap_transforms(
            flattened_root,
            self.time_calc.get_parallelism_mode(),
            getattr(self.time_calc, "tp_overlap", 0.0),
            getattr(self.time_calc, "tp_sp_overlap", 0.0),
            getattr(self.time_calc, "cp_overlap", 0.0),
        )
        flattener._propagate_local_hw_ids(flattened_root)

        unique_hw_ids = self._collect_hw_ids(flattened_root)
        if not unique_hw_ids:
            raise RuntimeError("Flattened memory graph exposes no compute nodes with hardware IDs")

        self.flattened_root = flattened_root
        return flattened_root

    def _collect_hw_ids(self, root: Any) -> Set[int]:
        visited: Set[int] = set()
        hw_ids: Set[int] = set()

        def enqueue_children(obj: Any) -> None:
            for child in getattr(obj, "children", []):
                stack.append(child)

        stack: List[Any]
        if isinstance(root, (list, tuple)):
            stack = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if isinstance(obj, llm_simulation.Node):
                hw_id = getattr(obj, "hw_id", None)
                if hw_id is not None and hw_id >= 0:
                    hw_ids.add(int(hw_id))

            enqueue_children(obj)

        return hw_ids



    def _run_transformer_astrasim(
        self,
        mode: ExecutionMode,
    ) -> Tuple[Optional[TransformerTimings], Optional[TransformerTimings]]:
        del mode  # mode currently unused but kept for signature consistency

        has_dense = bool(self.transformer_forward_root or self.transformer_backward_root)
        has_moe = bool(self.moe_transformer_forward_root or self.moe_transformer_backward_root)
        if not has_dense and not has_moe:
            if getattr(self, "_transformer_stage_dp_faults", {}):
                raise ValueError("Transformer faults require transformer graph metadata, but none is available.")
            return None, None

        layout = getattr(self, "_transformer_rank_layout", {})
        if self.transformer_forward_root:
            setattr(self.transformer_forward_root, "_astrasim_rank_layout", layout)
        if self.transformer_backward_root:
            setattr(self.transformer_backward_root, "_astrasim_rank_layout", layout)
        if self.moe_transformer_forward_root:
            setattr(self.moe_transformer_forward_root, "_astrasim_rank_layout", layout)
        if self.moe_transformer_backward_root:
            setattr(self.moe_transformer_backward_root, "_astrasim_rank_layout", layout)

        persist = self.time_calc.persist_astrasim_artifacts
        os.makedirs(self.time_calc.output_dir, exist_ok=True)
        self._transformer_stage_timings = {}
        self._transformer_stage_moe_timings = {}
        self._transformer_baseline_timings = None
        self._transformer_moe_baseline_timings = None

        baseline_timings: Optional[TransformerTimings] = None
        if has_dense:
            # Baseline run (no transformer faults)
            baseline_fwd_dir, baseline_bwd_dir = self._transformer_artifact_dirs(label=None, persist=persist)
            baseline_timings, baseline_fwd_per_rank, baseline_bwd_per_rank = self._execute_transformer_run(
                baseline_fwd_dir,
                baseline_bwd_dir,
                forward_root=self.transformer_forward_root,
                backward_root=self.transformer_backward_root,
                faulty_links_override=(),
            )
            self._transformer_baseline_timings = baseline_timings
            self.time_calc.transformer_astrasim_per_rank_forward = baseline_fwd_per_rank
            self.time_calc.transformer_astrasim_per_rank_backward = baseline_bwd_per_rank
            self.time_calc.transformer_astrasim_time_forward = baseline_timings.forward
            self.time_calc.transformer_astrasim_time_backward = baseline_timings.backward

            # Per-stage fault runs (dense only)
            stage_dp_faults = getattr(self, "_transformer_stage_dp_faults", {})
            for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
                label = f"fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=label, persist=persist)
                stage_timings, _, _ = self._execute_transformer_run(
                    stage_fwd_dir,
                    stage_bwd_dir,
                    forward_root=self.transformer_forward_root,
                    backward_root=self.transformer_backward_root,
                    faulty_links_override=fault_links,
                )
                self._transformer_stage_timings[(dp_idx, stage_id)] = stage_timings

        moe_timings: Optional[TransformerTimings] = None
        if has_moe:
            stage_dp_faults = getattr(self, "_transformer_stage_dp_faults", {})
            moe_fwd_dir, moe_bwd_dir = self._transformer_artifact_dirs(label="moe", persist=persist)
            moe_timings, moe_fwd_per_rank, moe_bwd_per_rank = self._execute_transformer_run(
                moe_fwd_dir,
                moe_bwd_dir,
                forward_root=self.moe_transformer_forward_root,
                backward_root=self.moe_transformer_backward_root,
                faulty_links_override=(),
            )
            self._transformer_moe_baseline_timings = moe_timings
            self.time_calc.transformer_astrasim_per_rank_forward_moe = moe_fwd_per_rank
            self.time_calc.transformer_astrasim_per_rank_backward_moe = moe_bwd_per_rank
            self.time_calc.transformer_astrasim_time_forward_moe = moe_timings.forward
            self.time_calc.transformer_astrasim_time_backward_moe = moe_timings.backward
            for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
                label = f"moe_fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=label, persist=persist)
                stage_timings, _, _ = self._execute_transformer_run(
                    stage_fwd_dir,
                    stage_bwd_dir,
                    forward_root=self.moe_transformer_forward_root,
                    backward_root=self.moe_transformer_backward_root,
                    faulty_links_override=fault_links,
                )
                self._transformer_stage_moe_timings[(dp_idx, stage_id)] = stage_timings

        return baseline_timings, moe_timings

    def _transformer_artifact_dirs(self, label: Optional[str], persist: bool) -> Tuple[str, str]:
        if not persist:
            base_dir = self.time_calc.output_dir
            os.makedirs(base_dir, exist_ok=True)
            return base_dir, base_dir
        base_dir = os.path.join(self.time_calc.output_dir, "astra_hier")
        os.makedirs(base_dir, exist_ok=True)
        if label is None:
            fwd_dir = os.path.join(base_dir, "fwd")
            bwd_dir = os.path.join(base_dir, "bwd")
        else:
            fwd_dir = os.path.join(base_dir, f"{label}_fwd")
            bwd_dir = os.path.join(base_dir, f"{label}_bwd")
        os.makedirs(fwd_dir, exist_ok=True)
        os.makedirs(bwd_dir, exist_ok=True)
        return fwd_dir, bwd_dir

    def _execute_transformer_run(
        self,
        artifact_dir_fwd: str,
        artifact_dir_bwd: str,
        *,
        forward_root: Optional[Any],
        backward_root: Optional[Any],
        faulty_links_override: Optional[Tuple[Tuple[int, int, float], ...]],
    ) -> Tuple[TransformerTimings, Optional[List[float]], Optional[List[float]]]:
        fwd_per_rank = None
        bwd_per_rank = None
        fwd_max = 0
        bwd_max = 0

        if forward_root:
            fwd_per_rank, fwd_max = run_astra_simulation_only_onepath(
                forward_root,
                self.time_calc,
                artifact_dir_fwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
            )
            if fwd_max <= 0:
                raise RuntimeError("AstraSim transformer forward execution returned non-positive duration")

        if backward_root:
            bwd_per_rank, bwd_max = run_astra_simulation_only_onepath(
                backward_root,
                self.time_calc,
                artifact_dir_bwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
            )
            if bwd_max < 0:
                raise RuntimeError("AstraSim transformer backward execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max), fwd_per_rank, bwd_per_rank

    def _apply_transformer_time(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings] = None,
    ) -> None:
        if timings is None and moe_timings is None:
            return
        if timings is not None and (timings.forward < 0 or timings.backward < 0):
            raise ValueError("AstraSim transformer times must be positive")
        if moe_timings is not None and (moe_timings.forward < 0 or moe_timings.backward < 0):
            raise ValueError("AstraSim transformer times must be positive")

        baseline_timings = timings or self._transformer_baseline_timings
        moe_baseline_timings = moe_timings or self._transformer_moe_baseline_timings
        stage_timings = getattr(self, "_transformer_stage_timings", {})
        stage_moe_timings = getattr(self, "_transformer_stage_moe_timings", {})

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if baseline_timings:
                comp_times["transformer_f"] = baseline_timings.forward
                comp_times["transformer_b"] = baseline_timings.backward
                comp_times["transformer_f_dense"] = baseline_timings.forward
                comp_times["transformer_b_dense"] = baseline_timings.backward
            if moe_baseline_timings:
                comp_times["transformer_f_moe"] = moe_baseline_timings.forward
                comp_times["transformer_b_moe"] = moe_baseline_timings.backward

        visited: Set[int] = set()
        roots: List[Any]
        if isinstance(self.pipeline_root, (list, tuple)):
            roots = list(self.pipeline_root)
        else:
            roots = [self.pipeline_root]

        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            dp_count = 1
        else:
            dp_count = max(1, getattr(self.time_calc, "dp", 1))

        for root in roots:
            self._assign_transformer_durations(
                root,
                visited,
                stage_timings,
                stage_moe_timings,
                baseline_timings,
                moe_baseline_timings,
                dp_count,
            )

    def _assign_transformer_durations(
        self,
        node: Any,
        visited: Set[int],
        stage_timings: Dict[Tuple[int, int], TransformerTimings],
        stage_moe_timings: Dict[Tuple[int, int], TransformerTimings],
        dense_timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings],
        dp_count: int,
    ) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, llm_simulation.Node):
            base_name = str(getattr(node, "name", "") or "")
            if base_name.startswith("transformer_layer"):
                is_moe_layer = bool(getattr(node, "is_moe_layer", False))
                timing_source = moe_timings if is_moe_layer and moe_timings is not None else dense_timings
                if timing_source is None:
                    timing_source = dense_timings
                if timing_source is None:
                    return
                hw_stage = None
                try:
                    hw_stage = int(getattr(node, "hw_id", None))
                except (TypeError, ValueError):
                    hw_stage = None
                def _per_dp_durations(is_forward: bool) -> List[float]:
                    values: List[float] = []
                    for dp_idx in range(dp_count):
                        default = timing_source.forward if is_forward else timing_source.backward
                        timing_override = None
                        if hw_stage is not None:
                            if is_moe_layer:
                                timing_override = stage_moe_timings.get((dp_idx, hw_stage))
                            else:
                                timing_override = stage_timings.get((dp_idx, hw_stage))
                        if timing_override:
                            values.append(timing_override.forward if is_forward else timing_override.backward)
                        else:
                            values.append(default)
                    return values

                per_dp_values = _per_dp_durations(is_forward=bool(getattr(node, "fwd", True)))

                if dp_count > 1:
                    node.duration = tuple(per_dp_values)
                else:
                    node.duration = per_dp_values[0]

        for child in getattr(node, "children", []):
            self._assign_transformer_durations(
                child,
                visited,
                stage_timings,
                stage_moe_timings,
                dense_timings,
                moe_timings,
                dp_count,
            )
