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

import math
from heapq import heappush, heappop
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from graphviz import Digraph
import os

import util
from timing_model import CollectiveType
from memory_estimation import MemKind, mem_kind_from_op_name, NON_TRANSFORMER_KINDS, TRANSFORMER_OP_KINDS
debug = False
BYTES_PER_GIB = 1024 ** 3

class Node:
    def __init__(
        self,
        name,
        op_id,
        hw_id,
        duration,
        fwd=True,
        mem_kind=None,
        recompute=False,
        param_gather=False,
    ):
        self.name = name
        self.op_id = op_id
        self.hw_id = hw_id
        self._duration_data = self._normalize_duration(duration)
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.memory = 0  # memory usage
        self.fwd = fwd  # forward or backward
        self.scheduled = False
        self.mem_kind = mem_kind
        self.recompute = bool(recompute)
        self.param_gather = bool(param_gather)

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self):
        return f"Node({self.name},op={self.op_id},hw={self.hw_id},{round(self.duration, 2)})"

    @staticmethod
    def _normalize_duration(value):
        if isinstance(value, (list, tuple)):
            entries = tuple(float(v) for v in value)
            if not entries:
                raise ValueError("Duration tuple must contain at least one entry.")
            return entries
        return float(value)

    @property
    def duration(self):
        data = self._duration_data
        if isinstance(data, tuple):
            return data[0]
        return data

    @duration.setter
    def duration(self, value):
        self._duration_data = self._normalize_duration(value)

    @property
    def duration_profile(self) -> Optional[Tuple[float, ...]]:
        data = self._duration_data
        if isinstance(data, tuple):
            return data
        return None

    def duration_storage(self):
        data = self._duration_data
        if isinstance(data, tuple):
            return tuple(data)
        return data
    
class Data_batch:
    def __init__(self, name, batch_id, duration):
        self.name = name
        self.batch_id = batch_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.scheduled = False
        # self.memory = 0  # memory usage

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)

    def remove_self_from_children(self):
        for child in self.children:
            child.parents.remove(self)
        self.children = []

class Edge:
  def __init__(self, name, op_id, duration, is_dp=False, comm_size_bytes=0, comm_type=None, participants=1, comm_interconnect_type=None):
    self.name = name
    self.op_id = op_id
    self.duration = duration
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.is_dp = is_dp
    self.scheduled = False
    self.comm_size_bytes = comm_size_bytes
    if comm_type is not None and not isinstance(comm_type, CollectiveType):
        raise TypeError(f"Edge.comm_type must be a CollectiveType or None (got {type(comm_type).__name__})")
    self.comm_type = comm_type
    self.participants = participants
    self.comm_interconnect_type = comm_interconnect_type

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)

  def __repr__(self):
      return f"Edge({self.name},op={self.op_id},{round(self.duration, 2)})"
      
class Graph:
    def __init__(
        self,
        mode: str,
        dp: int,
        pp: int,
        tp: int,
        cp: int,
        comp_times: Dict[str, Any],
        comm_metadata: Dict[str, Any],
        misc_metadata: Dict[str, Any],
        ep: int = 1,
    ) -> None:
        self.mode = mode
        self.dp = int(dp)
        self.pp = int(pp)
        self.tp = int(tp)
        self.cp = int(cp)
        self.ep = int(ep)
        self.comp_times = comp_times or {}
        self.comm_metadata = comm_metadata or {}
        self.misc_metadata = misc_metadata or {}
        self.full_recomputation = bool(self.misc_metadata.get("full_recomputation", False))
        self.flattened_mode = bool(self.misc_metadata.get("flattened_mode", False))
        self.pipeline_style_recompute = bool(self.misc_metadata.get("pipeline_style_recompute", False))

        self.num_batch = self.misc_metadata.get("num_batch", 0)
        self.num_layer = self.misc_metadata.get("num_layer", 0)
        self.layer_per_device = self.num_layer / self.pp if self.pp else self.num_layer
        self.dp_microbatch_mode = str(self.misc_metadata.get("dp_microbatch_mode", "every_mb")).lower()
        self.dp_zero_stage = int(self.misc_metadata.get("dp_zero_stage", 0) or 0)
        self.moe_layer_mask = list(self.misc_metadata.get("moe_layer_mask", []) or [])

        self.transformer_cfg = self.comp_times.get("transformer", {})
        self.T_grad_transformer = self.comp_times.get("grad_transformer", 0.0)
        self.layers_per_stage = self._compute_layers_per_stage()
        self.layer_to_stage = self._build_layer_to_stage()

    def _time(self, key: str, default: float = 0.0) -> float:
        value = self.comp_times.get(key)
        return float(value) if value is not None else default

    def _compute_layers_per_stage(self) -> List[int]:
        stage_count = max(1, int(self.pp) if self.pp else 1)
        if self.num_layer <= 0:
            return [0 for _ in range(stage_count)]
        base = self.num_layer // stage_count
        remainder = self.num_layer % stage_count
        layers_per_stage: List[int] = []
        for stage_idx in range(stage_count):
            count = base + (1 if stage_idx < remainder else 0)
            layers_per_stage.append(count)
        return layers_per_stage

    def _build_layer_to_stage(self) -> List[int]:
        mapping: List[int] = []
        for stage_idx, count in enumerate(self.layers_per_stage):
            if count <= 0:
                continue
            mapping.extend([stage_idx] * count)
        return mapping

    def _stage_for_layer(self, layer_idx: int) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layer:
            raise IndexError(f"Layer index {layer_idx} is out of bounds for {self.num_layer} layers.")
        if self.layer_to_stage:
            return self.layer_to_stage[layer_idx]
        return 0

    def _is_moe_layer(self, layer_idx: int) -> bool:
        if not self.moe_layer_mask:
            return False
        if layer_idx < 0 or layer_idx >= len(self.moe_layer_mask):
            return False
        return bool(self.moe_layer_mask[layer_idx])

    def create_comm_edge(self, name, op_id, comm_key, is_dp=False, local_hw_id=None):
        """Create a communication edge with optional local computation node.

        Args:
            name: Edge name
            op_id: Operation ID
            comm_key: Key into self.comm_metadata dict

        Returns:
            The communication edge (connection point for graph building)
        """
        comm_data = self.comm_metadata[comm_key]

        # Create communication edge with metadata
        # print(f"Creating edge with name: {name}, size: {comm_data['size']}, type: {comm_data['type']}")

        comm_edge = Edge(
            name=name,
            op_id=op_id,
            duration=0,  # Will be filled in second pass
            comm_size_bytes=comm_data['size'],
            comm_type=comm_data['type'],
            participants=comm_data['participants'],
            comm_interconnect_type=comm_data['interconnect_type'],
            is_dp=is_dp,
        )
        if local_hw_id is not None:
            try:
                comm_edge.local_hw_id = int(local_hw_id)
            except (TypeError, ValueError):
                comm_edge.local_hw_id = local_hw_id

        # If there's local computation time, create local node after edge
        local_comp_time = comm_data.get('local_comp_time', 0)
        if local_comp_time > 0:
            # TODO TODO TODO
            # We want to ignore this in the future. Skipping it directly so that we match astrasim for now.
            # FIX THIS (FIGURE OUT IF WE WANT TO SKIP OR NOT)
            local_comp_time = 0

        if comm_data.get('tp_shard'):
            comm_edge.tp_shard = True

        return comm_edge

    def extract_forward_graph(self, root: Any) -> Tuple[Any, Any]:
        """Detach backward branches and return separate forward/backward clones."""

        if root is None:
            return None, None

        clone_cache: Dict[str, Dict[int, Any]] = {"forward": {}, "backward": {}}
        creation_order: Dict[str, List[Any]] = {"forward": [], "backward": []}
        visited: Set[int] = set()

        def is_backward(node: Any) -> bool:
            direction = getattr(node, "direction", None)
            if direction is not None and str(direction).lower() == "backward":
                return True
            if hasattr(node, "fwd"):
                return bool(getattr(node, "fwd") is False)
            return False

        def classify(node: Any) -> str:
            return "backward" if is_backward(node) else "forward"

        def copy_optional_attrs(source: Any, target: Any) -> None:
            for attr in (
                "stage_id",
                "micro_batch_index",
                "layer_index",
                "direction",
                "tp_rank",
                "mem_kind",
                "recompute",
                "param_gather",
                "flatten_placeholder",
                "comm_interconnect_type",
                "comm_type",
                "comm_size_bytes",
                "participants",
                "is_cross_layer",
                "local_hw_id",
                "is_moe_layer",
            ):
                if hasattr(source, attr):
                    setattr(target, attr, getattr(source, attr))

        def ensure_clone(obj: Any, cls: str) -> Any:
            obj_id = id(obj)
            cache = clone_cache[cls]
            cached = cache.get(obj_id)
            if cached is not None:
                return cached

            if isinstance(obj, Node):
                clone = Node(
                    obj.name,
                    obj.op_id,
                    obj.hw_id,
                    obj.duration_storage(),
                    fwd=obj.fwd,
                    mem_kind=getattr(obj, "mem_kind", None),
                    recompute=getattr(obj, "recompute", False),
                    param_gather=getattr(obj, "param_gather", False),
                )
                clone.memory = getattr(obj, "memory", 0)
            elif isinstance(obj, Edge):
                clone = Edge(
                    obj.name,
                    obj.op_id,
                    obj.duration,
                    is_dp=getattr(obj, "is_dp", False),
                    comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                    comm_type=getattr(obj, "comm_type", None),
                    participants=getattr(obj, "participants", 1),
                    comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
                )
            elif isinstance(obj, Data_batch):
                clone = Data_batch(obj.name, obj.batch_id, obj.duration)

            else:
                raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

            copy_optional_attrs(obj, clone)
            cache[obj_id] = clone
            creation_order[cls].append(clone)
            return clone

        def traverse(obj: Any) -> None:
            if obj is None:
                return

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    traverse(item)
                return

            obj_id = id(obj)
            cls = classify(obj)
            current_clone = ensure_clone(obj, cls)

            for child in getattr(obj, "children", []):
                child_cls = classify(child)
                child_clone = ensure_clone(child, child_cls)
                if cls == child_cls:
                    if child_clone not in current_clone.children:
                        current_clone.add_child(child_clone)

            if obj_id in visited:
                return
            visited.add(obj_id)

            for child in getattr(obj, "children", []):
                traverse(child)

        traverse(root)

        def clone_structure(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, (list, tuple)):
                cloned_items: List[Any] = []
                for item in obj:
                    cloned = clone_structure(item)
                    if cloned is not None:
                        cloned_items.append(cloned)
                if not cloned_items:
                    return None
                return tuple(cloned_items) if isinstance(obj, tuple) else cloned_items

            cls = classify(obj)
            if cls != "forward":
                return None
            return ensure_clone(obj, cls)

        forward_root = clone_structure(root)
        if forward_root is None:
            forward_root = ensure_clone(root, classify(root))

        candidates: List[Any] = []
        seen: Set[int] = set()
        for clone in creation_order["backward"]:
            clone_id = id(clone)
            if clone_id in seen:
                continue
            seen.add(clone_id)
            backward_parents = [p for p in getattr(clone, "parents", []) if classify(p) == "backward"]
            if backward_parents:
                continue
            candidates.append(clone)

        if not candidates:
            backward_root: Optional[Any] = None
        elif len(candidates) == 1:
            backward_root = candidates[0]
        else:
            backward_root = Data_batch("backward_root", -1, 0.0)
            for candidate in candidates:
                backward_root.add_child(candidate)

        return forward_root, backward_root

    def extract_backward_graph(self, root: Any) -> Any:
        """Return a backward-only clone of the provided graph root."""

        if root is None:
            return None

        def is_backward(node):
            direction = getattr(node, "direction", None)
            if direction is not None:
                return str(direction).lower() == "backward"
            if hasattr(node, "fwd"):
                return bool(getattr(node, "fwd") is False)
            for child in getattr(node, "children", []):
                if getattr(child, "direction", None) == "backward" or getattr(child, "fwd", True) is False:
                    return True
            return False


        visited: Set[int] = set()
        all_objects: List[Any] = []

        def collect(obj: Any) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
                return

            all_objects.append(obj)
            for child in getattr(obj, "children", []):
                collect(child)

        collect(root)

        backward_objects = [obj for obj in all_objects if is_backward(obj)]
        if not backward_objects:
            return None

        included_ids = {id(obj) for obj in backward_objects}

        def should_include(obj: Any) -> bool:
            if is_backward(obj):
                return True
            if getattr(obj, "comm_type", None):
                return True
            name = getattr(obj, "name", "")
            if isinstance(obj, Node) and "local_comp" in name:
                return True
            return False

        queue = deque(backward_objects)
        while queue:
            current = queue.popleft()
            for child in getattr(current, "children", []):
                child_id = id(child)
                if child_id in included_ids:
                    continue
                if should_include(child):
                    included_ids.add(child_id)
                    backward_objects.append(child)
                    queue.append(child)

        backward_ids = included_ids
        clone_cache: Dict[int, Any] = {}

        def ensure_clone(obj: Any) -> Any:
            obj_id = id(obj)
            cached = clone_cache.get(obj_id)
            if cached is not None:
                return cached

            if isinstance(obj, Node):
                clone = Node(
                    obj.name,
                    obj.op_id,
                    obj.hw_id,
                    obj.duration_storage(),
                    fwd=obj.fwd,
                    mem_kind=getattr(obj, "mem_kind", None),
                    recompute=getattr(obj, "recompute", False),
                    param_gather=getattr(obj, "param_gather", False),
                )
                clone.memory = getattr(obj, "memory", 0)
            elif isinstance(obj, Edge):
                clone = Edge(
                    obj.name,
                    obj.op_id,
                    obj.duration,
                    is_dp=getattr(obj, "is_dp", False),
                    comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                    comm_type=getattr(obj, "comm_type", None),
                    participants=getattr(obj, "participants", 1),
                    comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
                )
            elif isinstance(obj, Data_batch):
                clone = Data_batch(obj.name, obj.batch_id, obj.duration)

            else:
                raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

            for attr in (
                "stage_id",
                "micro_batch_index",
                "layer_index",
                "direction",
                "tp_rank",
                "mem_kind",
                "recompute",
                "param_gather",
                "flatten_placeholder",
                "comm_interconnect_type",
                "comm_type",
                "comm_size_bytes",
                "participants",
                "is_cross_layer",
                "local_hw_id",
            ):
                if hasattr(obj, attr):
                    setattr(clone, attr, getattr(obj, attr))

            clone_cache[obj_id] = clone
            return clone

        for obj in backward_objects:
            ensure_clone(obj)

        for obj in backward_objects:
            clone = clone_cache[id(obj)]
            for child in getattr(obj, "children", []):
                if id(child) not in backward_ids:
                    continue
                child_clone = ensure_clone(child)
                if child_clone not in clone.children:
                    clone.add_child(child_clone)

        root_candidates: List[Any] = []
        for obj in backward_objects:
            parents = [p for p in getattr(obj, "parents", []) if id(p) in backward_ids]
            if parents:
                continue
            root_candidates.append(clone_cache[id(obj)])

        if not root_candidates:
            return None
        # if len(root_candidates) == 1:
        #     return root_candidates[0]

        # backward_root = Data_batch("backward_root", -1, 0.0)
        # for candidate in root_candidates:
            # backward_root.add_child(candidate)
        return root_candidates[0]

    @staticmethod
    def _reset_execution_state(root: Any) -> None:
        """Clear scheduling metadata so the graph can be re-simulated."""
        if root is None:
            return

        visited: Set[int] = set()
        if isinstance(root, (list, tuple, set)):
            stack: List[Any] = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if hasattr(obj, "done"):
                setattr(obj, "done", False)
            if hasattr(obj, "scheduled"):
                setattr(obj, "scheduled", False)
            if hasattr(obj, "finish_time"):
                setattr(obj, "finish_time", -1)

            stack.extend(getattr(obj, "children", []))

    def convert_comm_sizes_to_times(self, roots, network_model, interconnect_params):
        """
        Args:
            network_model: NetworkModel instance for collective timing
            interconnect_params: Dict with bandwidth/latency for each type
                                {'dp': (ib, ll), 'pp': (ib, ll), 'tp': (ib, ll)}
        """
        def traverse_and_convert(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return
            visited.add(id(node))

            # Process children (edges and nodes)
            for child in node.children:
                # If it's an edge with communication size, convert to time
                if hasattr(child, 'comm_size_bytes') and child.comm_size_bytes > 0:
                    # Get the appropriate bandwidth/latency for this interconnect type
                    interconnect_type = child.comm_interconnect_type
                    if interconnect_type and interconnect_type in interconnect_params:
                        ib, ll = interconnect_params[interconnect_type]
                    else:
                        raise ValueError(f"Invalid interconnect type: {interconnect_type}") 
                    if not isinstance(child.comm_type, CollectiveType):
                        raise TypeError(
                            f"Comm edge {getattr(child, 'name', '<unnamed>')} missing CollectiveType comm_type"
                        )

                    child.duration = network_model.collective(
                        kind=child.comm_type,
                        size_bytes=child.comm_size_bytes,
                        participants=child.participants,
                        ib=ib,
                        ll=ll,
                        local_bytes=0.0,
                        local_ops=0.0,
                        debug_label=f"{child.name}_conversion"
                    )
                    # print(f"Converted {child.name} size {child.comm_size_bytes} bytes to duration {child.duration:.6f} sec using {interconnect_type} (ib={ib}, ll={ll})")

                # Recursively process this child
                traverse_and_convert(child, visited)

        traverse_and_convert(roots)
        return roots

    def construct_fwd_bwd_graph(self, include_backward: bool = True, include_optimizer: bool = False):
        embedding_node = []
        data_batch_node = []
        softmax_node = []
        transformer_nodes = [[] for _ in range(self.num_batch)]  # transformer compute nodes per layer
        layer_entry_nodes = [[] for _ in range(self.num_batch)]  # lists of entry nodes per layer
        layer_exit_nodes = [[] for _ in range(self.num_batch)]   # transformer nodes per layer

        linear_softmax_f_time = self._time("linear_softmax_f")
        linear_softmax_b_time = self._time("linear_softmax_b")
        transformer_f_time = self._time("transformer_f")
        transformer_b_time = self._time("transformer_b")
        transformer_f_dense = self._time("transformer_f_dense", transformer_f_time)
        transformer_f_moe = self._time("transformer_f_moe", transformer_f_time)
        transformer_b_dense = self._time("transformer_b_dense", transformer_b_time)
        transformer_b_moe = self._time("transformer_b_moe", transformer_b_time)
        embedding_f_time = self._time("embedding_f")
        embedding_b_time = self._time("embedding_b")
        cross_layer_time = self._time("cross_layer_f")
        recompute_enabled = include_backward and self.full_recomputation and (
            self.flattened_mode or self.pipeline_style_recompute
        )


        def attach_parallel_edge(target, gather_edge, skip_non_comm_children=None, skip_comm_children=None):
            parents = list(getattr(target, "parents", []))
            for parent in parents:
                if hasattr(parent, "children") and gather_edge not in parent.children:
                    parent.add_child(gather_edge)
            children = list(getattr(target, "children", []))
            for child in children:
                if skip_non_comm_children:
                    if getattr(child, "comm_type", None) != CollectiveType.PIPELINE:
                        continue
                if skip_comm_children:
                    if getattr(child, "comm_type", None) == CollectiveType.PIPELINE:
                        continue
                if gather_edge not in getattr(child, "parents", []):
                    gather_edge.add_child(child)

        if include_backward:
            embedding_node_b = [[] for _ in range(self.num_batch)]
            softmax_node_b = [[] for _ in range(self.num_batch)]
            R_edge = [[] for _ in range(self.num_batch)]
            G_edge = [[] for _ in range(self.num_batch)]

            transformer_nodes_b = [[] for _ in range(self.num_batch)]  #
            for b in range(self.num_batch):
                transformer_nodes_b[b] = [[] for _ in range(self.num_layer)]
            recompute_nodes_b: Optional[List[List[Optional[Node]]]] = None
            if recompute_enabled:
                recompute_nodes_b = [[None for _ in range(self.num_layer)] for _ in range(self.num_batch)]

        op_id = 0  # operation ID, used to distinguish nodes and edges
        batch_id = 0  # batch ID, used to distinguish data batches
        
        data0 = Data_batch("data0", batch_id, 0)
        data_batch_node.append(data0)
        
        for i in range(1, self.num_batch):#create data batch node
            
            data_batch_node.append(Data_batch(f"data{i}", i, 0))
            data_batch_node[i-1].add_child(data_batch_node[i])

        for b in range(self.num_batch): #connect each data batch node with corresponding nodes
            linear_softmax = Node(
                f"linear_softmax{b}",
                op_id,
                self.pp - 1,
                linear_softmax_f_time,
                mem_kind=MemKind.SOFTMAX,
            )
            op_id += 1
            softmax_node.append(linear_softmax)
            emb = Node(
                f"embedding{b}",
                op_id,
                0,
                embedding_f_time,
                mem_kind=MemKind.EMBEDDING,
            )  # hw_id = 0
            op_id += 1
            embedding_node.append(emb)
            data_batch_node[b].add_child(embedding_node[b])

            transformer_nodes[b] = []
            layer_entry_nodes[b] = []
            layer_exit_nodes[b] = []

            for l in range(self.num_layer):
                hw_id = self._stage_for_layer(l)
                is_moe_layer = self._is_moe_layer(l)
                transformer_duration = transformer_f_moe if is_moe_layer else transformer_f_dense
                transformer_node = Node(
                    f"transformer_layer{l}",
                    op_id,
                    hw_id,
                    transformer_duration,
                    mem_kind=MemKind.TRANSFORMER,
                )
                transformer_node.micro_batch_index = b
                transformer_node.layer_index = l
                transformer_node.direction = "forward"
                transformer_node.stage_id = hw_id
                transformer_node.is_moe_layer = is_moe_layer
                op_id += 1
                transformer_nodes[b].append(transformer_node)
                entry_nodes = [transformer_node]
                layer_entry_nodes[b].append(entry_nodes)
                layer_exit_nodes[b].append(transformer_node)

            for l in range(1, self.num_layer):
                prev_node = transformer_nodes[b][l-1]        # previous layer compute node
                curr_node  = transformer_nodes[b][l]            # current layer compute node
                prev_exit = layer_exit_nodes[b][l-1]
                curr_entries = layer_entry_nodes[b][l]

                if prev_node.hw_id == curr_node.hw_id:
                    edge = Edge("cross_layer", op_id, 0, comm_type=CollectiveType.PIPELINE)  # on same GPU
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')  # on different GPU
                op_id += 1

                prev_exit.add_child(edge)
                for entry in curr_entries:
                    edge.add_child(entry)


            first_entries = layer_entry_nodes[b][0]   # first layer entry nodes list
            primary_entry = first_entries[0]
            if primary_entry.hw_id == embedding_node[b].hw_id:
                edge = Edge("Emb_node0", op_id, 0, comm_type=CollectiveType.PIPELINE)
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            embedding_node[b].add_child(edge) #connect embedding node and first transformer layer
            for entry in first_entries:
                edge.add_child(entry)


            last_exit = layer_exit_nodes[b][-1]  # last layer transformer node
            if last_exit.hw_id == softmax_node[b].hw_id:
                node_Softmax = Edge("node_Softmax", op_id, 0, comm_type=CollectiveType.PIPELINE)  # same GPU
            else:
                node_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            last_exit.add_child(node_Softmax) #connect last layer and softmax layer
            node_Softmax.add_child(softmax_node[b])

        #add dependency edges
        for b in range(self.num_batch - 1):
            gpu_index = 0
            last_transformer_layer = []
            first_transformer_layer = []
            first_transformer_layer.append(0)
            for l in range(self.num_layer-1):
                if transformer_nodes[b][l].hw_id != transformer_nodes[b][l+1].hw_id: # check if on different GPUs
                    last_transformer_layer.append(l) # record last layer on each GPU
                    first_transformer_layer.append(l+1) # record first layer on each GPU
                    gpu_index += 1
                    if transformer_nodes[b][l].hw_id == 0: #if first pipeline stage
                        layer_exit_nodes[b][l].add_child(embedding_node[b+1]) # dependency: finish stage before next batch embedding starts
                    else:
                        next_entries = layer_entry_nodes[b+1][first_transformer_layer[gpu_index-1]]
                        for entry in next_entries:
                            layer_exit_nodes[b][l].add_child(entry)

            next_entries = layer_entry_nodes[b+1][first_transformer_layer[-1]]
            for entry in next_entries:
                softmax_node[b].add_child(entry)

        for db_node in data_batch_node:
            db_node.remove_self_from_children()

        if not include_backward:
            return embedding_node[0]

        def _bwd_entry_node(mb_idx: int, layer_idx: int) -> Any:
            if recompute_enabled and recompute_nodes_b is not None:
                candidate = recompute_nodes_b[mb_idx][layer_idx]
                if candidate is not None:
                    return candidate
            return transformer_nodes_b[mb_idx][layer_idx]

        def _bwd_exit_node(mb_idx: int, layer_idx: int) -> Any:
            return transformer_nodes_b[mb_idx][layer_idx]

        for b in reversed(range(self.num_batch)): #connect each data batch node with corresponding nodes
            emb_b = Node(
                "embedding_b",
                op_id,
                0,
                embedding_b_time,
                fwd=False,
                mem_kind=MemKind.EMBEDDING,
            )  # hw_id = 0
            op_id += 1
            embedding_node_b[b] = emb_b
            linear_softmax_b = Node(
                "linear_softmax_b",
                op_id,
                self.pp - 1,
                linear_softmax_b_time,
                fwd=False,
                mem_kind=MemKind.SOFTMAX,
            )
            op_id += 1
            softmax_node_b[b] = linear_softmax_b
            softmax_node[b].add_child(linear_softmax_b)


            for l in reversed(range(self.num_layer)):
                hw_id = self._stage_for_layer(l)
                recompute_node = None
                if recompute_enabled and recompute_nodes_b is not None:
                    recompute_node = Node(
                        f"transformer_layer{l}_recompute",
                        op_id,
                        hw_id,
                        transformer_f_moe if self._is_moe_layer(l) else transformer_f_dense,
                        fwd=True,
                        mem_kind=MemKind.TRANSFORMER,
                        recompute=True,
                    )
                    recompute_node.micro_batch_index = b
                    recompute_node.layer_index = l
                    recompute_node.direction = "forward"
                    recompute_node.stage_id = hw_id
                    recompute_node.is_moe_layer = self._is_moe_layer(l)
                    op_id += 1
                    recompute_nodes_b[b][l] = recompute_node

                is_moe_layer = self._is_moe_layer(l)
                transformer_node_b = Node(
                    f"transformer_layer{l}_b",
                    op_id,
                    hw_id,
                    transformer_b_moe if is_moe_layer else transformer_b_dense,
                    fwd=False,
                    mem_kind=MemKind.TRANSFORMER,
                )
                transformer_node_b.micro_batch_index = b
                transformer_node_b.layer_index = l
                transformer_node_b.direction = "backward"
                transformer_node_b.stage_id = hw_id
                transformer_node_b.is_moe_layer = is_moe_layer
                op_id += 1
                transformer_nodes_b[b][l] = transformer_node_b
                if recompute_node is not None:
                    recompute_node.add_child(transformer_node_b)

            for l in reversed(range(1, self.num_layer)):
                curr_node = _bwd_exit_node(b, l)         # current layer's qkv_proj.
                next_ffn2  = _bwd_entry_node(b, l-1)            # next layer's layernorm.
                if curr_node.hw_id == next_ffn2.hw_id:
                    edge = Edge("cross_layer", op_id, 0, comm_type=CollectiveType.PIPELINE)  
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
                op_id += 1

                curr_node.add_child(edge); edge.add_child(next_ffn2)

            qkv_0_b = _bwd_exit_node(b, 0)     # first layer's qkv_proj
            if qkv_0_b.hw_id == emb_b.hw_id:
                edge = Edge("Emb_node0", op_id, 0, comm_type=CollectiveType.PIPELINE)
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            qkv_0_b.add_child(edge)
            edge.add_child(emb_b)


            prev_layer_norm2 = _bwd_entry_node(b, self.num_layer-1) # last layer's layernorm2
            if prev_layer_norm2.hw_id == softmax_node_b[b].hw_id:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, 0, comm_type=CollectiveType.PIPELINE)  # same GPU
            else:
                layernorm_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            softmax_node_b[b].add_child(layernorm_Softmax)
            layernorm_Softmax.add_child(prev_layer_norm2)

        zero3_embedding_key = "zero3_embedding_gather"
        if zero3_embedding_key in self.comm_metadata:
            zero3_entry_embedding_edge = self.create_comm_edge(
                f"{zero3_embedding_key}_b0_fwd_entry",
                op_id,
                zero3_embedding_key,
                is_dp=True,
                local_hw_id=embedding_node[0].hw_id,
            )
            op_id += 1
            root_forward_entry = zero3_entry_embedding_edge
            zero3_entry_embedding_edge.add_child(embedding_node[0])
        else:
            root_forward_entry = embedding_node[0]

        for b in range(self.num_batch):
            # Data parallel collectives (all reduce for DDP/ZeRO-1, reduce-scatter + all-gather for ZeRO-2/3)
            if self.dp > 1:
                apply_dp_all_mbs = self.dp_microbatch_mode != "last_mb" or self.dp_zero_stage >= 3
                if not apply_dp_all_mbs and b != 0: # mb is backwards in backpass
                    continue
                zero2_embedding_key = "zero2_embedding_gather"
                zero2_transformer_key = "zero2_transformer_gather"
                zero2_softmax_key = "zero2_softmax_gather"
                embedding_edge = self.create_comm_edge(
                    "embedding",
                    op_id,
                    "embedding",
                    is_dp=True,
                )
                R_edge[b].append(embedding_edge)
                op_id += 1
                postfix = ""
                if zero2_embedding_key in self.comm_metadata: # ZeRO-2/3, use reduce-scatter.
                    postfix = "_reduce_scatter"
                    gather_edge = self.create_comm_edge(
                        zero2_embedding_key,
                        op_id,
                        zero2_embedding_key,
                        is_dp=True,
                        local_hw_id=embedding_node[b].hw_id,
                    )
                    op_id += 1
                    embedding_edge.add_child(gather_edge)
                elif zero3_embedding_key in self.comm_metadata: # No ZeRO-2/3, use all-reduce.
                    postfix = "_reduce_scatter"
                else:
                    postfix = "_all_reduce"

                zero3_embedding_key = "zero3_embedding_gather"
                zero3_transformer_key = "zero3_transformer_gather"
                zero3_softmax_key = "zero3_softmax_gather"
                for layer_idx in range(self.num_layer):
                    comm_key = "transformer_moe" if self._is_moe_layer(layer_idx) else "transformer_dense"
                    reducer = self.create_comm_edge(
                        f"transformer_b{b}_layer{layer_idx}"+postfix,
                        op_id,
                        comm_key,
                        is_dp=True,
                        local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                    )
                    R_edge[b].append(reducer)
                    op_id += 1

                    if zero2_transformer_key in self.comm_metadata:
                        gather_edge = self.create_comm_edge(
                            zero2_transformer_key,
                            op_id,
                            zero2_transformer_key,
                            is_dp=True,
                            local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                        )
                        op_id += 1
                        reducer.add_child(gather_edge)

                if zero3_transformer_key in self.comm_metadata:
                    # create the edge for the embedding gather of the next batch (except for the last batch)
                    zero3_embedding_gather_edge = None
                    if b < self.num_batch - 1:
                        zero3_embedding_gather_edge = self.create_comm_edge(
                            f"{zero3_embedding_key}_b{b+1}_fwd",
                            op_id,
                            zero3_embedding_key,
                            is_dp=True,
                            local_hw_id=embedding_node[b].hw_id,
                        )
                        op_id += 1

                    gather_edge = self.create_comm_edge(
                        f"{zero3_transformer_key}_b{b}_layer0_fwd",
                        op_id,
                        zero3_transformer_key,
                        is_dp=True,
                        local_hw_id=embedding_node[b].hw_id,
                    )
                    op_id += 1
                    attach_parallel_edge(embedding_node[b], gather_edge)
                    for layer_idx in range(1, self.num_layer):
                        host = transformer_nodes[b][layer_idx - 1]
                        target = transformer_nodes[b][layer_idx]
                        gather_edge = self.create_comm_edge(
                            f"{zero3_transformer_key}_b{b}_layer{layer_idx}_fwd",
                            op_id,
                            zero3_transformer_key,
                            is_dp=True,
                            local_hw_id=target.hw_id,
                        )
                        op_id += 1
                        if host.hw_id == target.hw_id:
                            attach_parallel_edge(host, gather_edge)
                        else:
                            # Cross-device. Have to do this very carefully.
                            # We will need to attach the edge but make sure the dependencies only apply to the "cross_layer" comm.
                            attach_parallel_edge(host, gather_edge, skip_non_comm_children=True)
                            # Then, we will need to attach the zero3 embedding gather edge to the target device.
                            if zero3_embedding_gather_edge:
                                attach_parallel_edge(host, zero3_embedding_gather_edge, skip_comm_children=True)
                             

                softmax_edge = self.create_comm_edge(
                    "softmax"+postfix,
                    op_id,
                    "softmax",
                    is_dp=True,
                    local_hw_id=softmax_node[b].hw_id,
                )
                R_edge[b].append(softmax_edge)
                op_id += 1
                if zero2_softmax_key in self.comm_metadata:
                    gather_edge = self.create_comm_edge(
                        zero2_softmax_key,
                        op_id,
                        zero2_softmax_key,
                        is_dp=True,
                        local_hw_id=softmax_node[b].hw_id,
                    )
                    op_id += 1
                    softmax_edge.add_child(gather_edge)

                if zero3_softmax_key in self.comm_metadata and self.num_layer > 0:
                    host = transformer_nodes[b][self.num_layer - 1]
                    gather_edge = self.create_comm_edge(
                        f"{zero3_softmax_key}_b{b}_fwd",
                        op_id,
                        zero3_softmax_key,
                        is_dp=True,
                        local_hw_id=host.hw_id,
                    )
                    op_id += 1
                    attach_parallel_edge(host, gather_edge)

                softmax_node_b[b].add_child(R_edge[b][-1])
                embedding_node_b[b].add_child(R_edge[b][0])
                for layer_idx in range(self.num_layer):
                    _bwd_exit_node(b, layer_idx).add_child(R_edge[b][layer_idx + 1])

            if self.ep > 1 and self.dp > 1:
                apply_ep_all_mbs = self.dp_microbatch_mode != "last_mb" or self.dp_zero_stage >= 3
                if apply_ep_all_mbs or b == 0:
                    for layer_idx in range(self.num_layer):
                        comm_key = (
                            "transformer_moe_ep_sync"
                            if self._is_moe_layer(layer_idx)
                            else "transformer_dense_ep_sync"
                        )
                        if comm_key not in self.comm_metadata:
                            continue
                        ep_edge = self.create_comm_edge(
                            f"{comm_key}_b{b}_layer{layer_idx}",
                            op_id,
                            comm_key,
                            is_dp=False,
                            local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                        )
                        op_id += 1
                        _bwd_exit_node(b, layer_idx).add_child(ep_edge)


        last_transformer_layer = [-1] * self.pp  # Initialize with -1 for all GPUs
        first_transformer_layer = [-1] * self.pp  # Initialize with -1 for all GPUs

        # first_transformer_layer.append(0)
        gpu_index = self.pp - 1
        for l in range(self.num_layer - 1, 0, -1):
            if _bwd_exit_node(0, l).hw_id != _bwd_exit_node(0, l-1).hw_id:  # Check if on different GPU
                # print("Layer ", l, " is on GPU ", _bwd_exit_node(0, l).hw_id)
                first_transformer_layer[gpu_index-1] = l-1  # Record first layer on each GPU
                last_transformer_layer[gpu_index] = l  # Record last layer on each GPU
                gpu_index -= 1
        # for id in range(self.pp):
            # print("GPU ", id, " first layer ", first_transformer_layer[id], " last layer ", last_transformer_layer[id])



        for b in range(self.num_batch-1, 0, -1):
            gpu_index = self.pp - 1
            for l in range(self.num_layer - 1, 0, -1):
                if _bwd_exit_node(b, l).hw_id != _bwd_exit_node(b, l-1).hw_id:  # Check if on different GPUs
                    # last_transformer_layer.append(l)  # Record last layer on each GPU
                    # first_transformer_layer.append(l-1)  # Record first layer on each GPU
                    
                    if _bwd_exit_node(b, l).hw_id == self.pp - 1:
                        _bwd_exit_node(b, l).add_child(softmax_node_b[b-1])  # Add dependency edge
                    else:
                        _bwd_exit_node(b, l).add_child(_bwd_entry_node(b-1, first_transformer_layer[gpu_index]))  # Add dependency edge
                    gpu_index -= 1
            # Ensure embedding_node_b[b] is connected to the correct transformer node
            # if first_transformer_layer:
            embedding_node_b[b].add_child(_bwd_entry_node(b-1, first_transformer_layer[0]))  # Add dependency edge

        zero3_embedding_key = "zero3_embedding_gather"
        zero3_transformer_key = "zero3_transformer_gather"
        zero3_softmax_key = "zero3_softmax_gather"

        zero3_softmax_bwd_entry = None
        if zero3_softmax_key in self.comm_metadata and self.num_layer > 0:
            zero3_softmax_bwd_entry = self.create_comm_edge(
                f"{zero3_softmax_key}_b0_bwd_entry",
                op_id,
                zero3_softmax_key,
                is_dp=True,
                local_hw_id=softmax_node_b[0].hw_id,
            )
            op_id += 1
            attach_parallel_edge(softmax_node[-1], zero3_softmax_bwd_entry)
        for b in range(self.num_batch):
            zero3_softmax_bwd_edge = None
            if b > 0:
                if zero3_softmax_key in self.comm_metadata:
                    host = softmax_node[b]
                    gather_edge = self.create_comm_edge(
                        f"{zero3_softmax_key}_b{b}_bwd",
                        op_id,
                        zero3_softmax_key,
                        is_dp=True,
                        local_hw_id=host.hw_id,
                    )
                    op_id += 1
                    zero3_softmax_bwd_edge = gather_edge

            if zero3_transformer_key in self.comm_metadata and self.num_layer > 0:
                for layer_idx in reversed(range(self.num_layer)):
                    host = softmax_node_b[b] if layer_idx == self.num_layer - 1 else _bwd_exit_node(b, layer_idx + 1)
                    target = _bwd_entry_node(b, layer_idx)
                    gather_edge = self.create_comm_edge(
                        f"{zero3_transformer_key}_b{b}_layer{layer_idx}_bwd",
                        op_id,
                        zero3_transformer_key,
                        is_dp=True,
                        local_hw_id=target.hw_id,
                    )
                    op_id += 1
                    if host.hw_id == target.hw_id:
                        attach_parallel_edge(host, gather_edge)
                    else:
                        # Cross-device. Have to do this very carefully.
                        # We will need to attach the edge but make sure the dependencies only apply to the "cross_layer" comm.
                        attach_parallel_edge(host, gather_edge, skip_non_comm_children=True)
                        if zero3_softmax_bwd_edge:
                            attach_parallel_edge(host, zero3_softmax_bwd_edge, skip_comm_children=True)


            if zero3_embedding_key in self.comm_metadata and self.num_layer > 0:
                host = _bwd_exit_node(b, 0)
                gather_edge = self.create_comm_edge(
                    f"{zero3_embedding_key}_b{b}_bwd",
                    op_id,
                    zero3_embedding_key,
                    is_dp=True,
                    local_hw_id=host.hw_id,
                )
                op_id += 1
                attach_parallel_edge(host, gather_edge)

        if include_backward and include_optimizer:
            optimizer_time = self._time("optimizer")
            if optimizer_time > 0:
                # Find the last node for each stage to attach the optimizer
                # For stage 0, it's the embedding backward of the last microbatch
                # For other stages, it's the first transformer layer (lowest index) of that stage for the last microbatch
                
                # Helper to find min layer for each stage
                stage_min_layer = {}
                for l in range(self.num_layer):
                    stage = self._stage_for_layer(l)
                    if stage not in stage_min_layer:
                        stage_min_layer[stage] = l
                    else:
                        stage_min_layer[stage] = min(stage_min_layer[stage], l)
                                
                # bwd flips mbs, so we attach to first one, not last.
                for stage in range(self.pp):
                    last_node = None
                    if stage == 0:
                        last_node = embedding_node_b[0]
                    else:
                        min_l = stage_min_layer.get(stage)
                        if min_l is not None:
                            last_node = _bwd_exit_node(0, min_l)
                    
                    if last_node:
                        opt_node = Node(
                            f"optimizer_stage{stage}",
                            op_id,
                            stage,
                            optimizer_time,
                            fwd=False,
                            mem_kind=MemKind.OPTIMIZER,
                        )
                        op_id += 1
                        last_node.add_child(opt_node)

        return root_forward_entry

    def construct_transformer_graph(self, direction: str = "both"):
        transformer_cfg = self.transformer_cfg
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM times not provided")

        tp_degree = max(1, int(self.tp))
        cp_degree = max(1, int(self.cp))
        ep_degree = max(1, int(self.ep))

        root = Data_batch("transformer_root", 0, 0)
        op_id = 0

        def _split_comm_keys(comm_keys: Optional[List[str]]) -> Tuple[List[str], List[str]]:
            pre_keys: List[str] = []
            post_keys: List[str] = []
            if not comm_keys:
                return pre_keys, post_keys
            for key in comm_keys:
                placement = self.comm_metadata.get(key, {}).get("placement", "post")
                if placement == "pre":
                    pre_keys.append(key)
                elif placement == "post":
                    post_keys.append(key)
                else:
                    raise ValueError(
                        f"Unsupported comm placement '{placement}' for key '{key}' "
                        "(expected 'pre' or 'post')"
                    )
            return pre_keys, post_keys

        for ep_idx in range(ep_degree):
            for cp_idx in range(cp_degree):
                for tp_idx in range(tp_degree):
                    rank = tp_idx + cp_idx * tp_degree + ep_idx * tp_degree * cp_degree
                    previous = root

                    if direction in {"forward", "both"}:
                        for idx, entry in enumerate(gemm_entries):
                            entry_name = entry.get("name", f"g{idx}")
                            forward_cfg = entry.get("forward", {})
                            fwd_duration = forward_cfg.get("duration")
                            if fwd_duration is None:
                                raise ValueError("Transformer GEMM entry missing forward duration")

                            comm_keys = list(forward_cfg.get("comm_keys", []) or [])
                            pre_keys, post_keys = _split_comm_keys(comm_keys)

                            for comm_key in pre_keys:
                                if comm_key not in self.comm_metadata:
                                    raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                                comm_edge = self.create_comm_edge(
                                    name=comm_key,
                                    op_id=op_id,
                                    comm_key=comm_key,
                                    is_dp=False,
                                    local_hw_id=rank,
                                )
                                comm_edge.tp_rank = tp_idx
                                comm_edge.cp_rank = cp_idx
                                comm_edge.ep_rank = ep_idx
                                op_id += 1
                                previous.add_child(comm_edge)
                                previous = comm_edge

                            node = Node(
                                name=f"{entry_name}_fwd_rank{rank}",
                                op_id=op_id,
                                hw_id=rank,
                                duration=fwd_duration,
                                fwd=True,
                                mem_kind=mem_kind_from_op_name(entry_name),
                                param_gather=(idx == 0),
                            )
                            node.tp_rank = tp_idx
                            node.cp_rank = cp_idx
                            node.ep_rank = ep_idx
                            op_id += 1
                            previous.add_child(node)
                            previous = node
                            for comm_key in post_keys:
                                if comm_key not in self.comm_metadata:
                                    raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                                comm_edge = self.create_comm_edge(
                                    name=comm_key,
                                    op_id=op_id,
                                    comm_key=comm_key,
                                    is_dp=False,
                                    local_hw_id=rank,
                                )
                                comm_edge.tp_rank = tp_idx
                                comm_edge.cp_rank = cp_idx
                                comm_edge.ep_rank = ep_idx
                                op_id += 1
                                previous.add_child(comm_edge)
                                previous = comm_edge

                    if direction in {"backward", "both"}:
                        for idx, entry in enumerate(reversed(gemm_entries)):
                            entry_name = entry.get("name", f"g{idx}")
                            backward_cfg = entry.get("backward", {})
                            bwd_duration = backward_cfg.get("duration")
                            if bwd_duration is None:
                                raise ValueError("Transformer GEMM entry missing backward duration")

                            comm_keys = list(backward_cfg.get("comm_keys", []) or [])
                            pre_keys, post_keys = _split_comm_keys(comm_keys)

                            for comm_key in pre_keys:
                                if comm_key not in self.comm_metadata:
                                    raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                                comm_edge = self.create_comm_edge(
                                    name=comm_key,
                                    op_id=op_id,
                                    comm_key=comm_key,
                                    is_dp=False,
                                    local_hw_id=rank,
                                )
                                comm_edge.tp_rank = tp_idx
                                comm_edge.cp_rank = cp_idx
                                comm_edge.ep_rank = ep_idx
                                op_id += 1
                                previous.add_child(comm_edge)
                                previous = comm_edge

                            node = Node(
                                name=f"{entry_name}_bwd_rank{rank}",
                                op_id=op_id,
                                hw_id=rank,
                                duration=bwd_duration,
                                fwd=False,
                                mem_kind=mem_kind_from_op_name(entry_name),
                                param_gather=(idx == 0),
                            )
                            node.tp_rank = tp_idx
                            node.cp_rank = cp_idx
                            node.ep_rank = ep_idx
                            op_id += 1
                            previous.add_child(node)
                            previous = node

                            for comm_idx, comm_key in enumerate(post_keys):
                                if comm_key not in self.comm_metadata:
                                    raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                                comm_edge = self.create_comm_edge(
                                    name=comm_key,
                                    op_id=op_id,
                                    comm_key=comm_key,
                                    is_dp=False,
                                    local_hw_id=rank,
                                )
                                comm_edge.tp_rank = tp_idx
                                comm_edge.cp_rank = cp_idx
                                comm_edge.ep_rank = ep_idx
                                op_id += 1
                                previous.add_child(comm_edge)
                                previous = comm_edge

        return root
        
    def simulate(self, root):
        time = 0
        counter = 0
        event_queue = []
        ready_list = []

        self._reset_execution_state(root)

        ready_list.append(root)
        root.scheduled = True
        ###find number of devices needed
        base_devices = max(1, int(self.pp) if self.pp else 1)
        max_hw_id = -1
        visited_nodes: Set[int] = set()
        stack = list(root if isinstance(root, (list, tuple)) else [root])
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            hw_id = getattr(node, "hw_id", None)
            if hw_id is not None:
                try:
                    hw_val = int(hw_id)
                except (TypeError, ValueError):
                    hw_val = None
                if hw_val is not None and hw_val >= 0:
                    max_hw_id = max(max_hw_id, hw_val)

            stack.extend(getattr(node, "children", []))

        if max_hw_id >= 0:
            base_devices = max(base_devices, max_hw_id + 1)

        GPU_list = [True for _ in range(base_devices)]
        data_list = [False for i in range(0, self.num_batch)]

        heappush(event_queue, (root.duration, counter, root))
        if debug:
            print("{} enqueued at time {} batch id {}".format(root.name, 0, root.batch_id))
        ready_list.remove(root)
        counter = counter + 1

        while len(event_queue) > 0:
            time, _, event = heappop(event_queue)
            event.done = True
            event.scheduled = False
            event.finish_time = time
            if debug:
                print("Event {} finished at time {}".format(event.name, time))

            for child in event.children:

                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                # if is_ready == True:
                if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                    ready_list.append(child)
                    if debug:
                        print("child {}  ready at time {} ".format(child.name, time))

            if isinstance(event, Node):
                GPU_list[int(event.hw_id)] = True

                



            for event in ready_list[:]:
                enqueued = False
                if isinstance(event, Data_batch):
                    # if GPU_list[event.hw_id] == True:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{} enqueued at time".format(event.name,  time))
                    counter = counter + 1
                    data_list[event.batch_id] = True #data batch sent to gpu
                    # GPU_list[event.hw_id] = False
                    ready_list.remove(event)

                elif isinstance(event, Node): 
                    if GPU_list[int(event.hw_id)] == True:
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                        counter = counter + 1
                        GPU_list[int(event.hw_id)] = False
                        ready_list.remove(event)
                elif isinstance(event, Edge): 
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                    enqueued = True
                    counter = counter + 1
                    ready_list.remove(event)


        return time
    
    def simulate_memory(self, root, memory_data, mode = "training", output_folder = "output/LLM/", filename="memory_graph"):
        time = 0
        counter = 0
        event_queue = []
        ready_list = []

        self._reset_execution_state(root)

        
        ready_list.append(root)
        root.scheduled = True
        ###find number of devices needed
        base_devices = max(1, int(self.pp) if self.pp else 1)
        max_hw_id = -1
        visited_nodes: Set[int] = set()
        stack = list(root if isinstance(root, (list, tuple)) else [root])
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            hw_id = getattr(node, "hw_id", None)
            if hw_id is not None:
                try:
                    hw_val = int(hw_id)
                except (TypeError, ValueError):
                    hw_val = None
                if hw_val is not None and hw_val >= 0:
                    max_hw_id = max(max_hw_id, hw_val)

            stack.extend(getattr(node, "children", []))

        if max_hw_id >= 0:
            base_devices = max(base_devices, max_hw_id + 1)

        class _MemorySnapshot:
            def __init__(self, num_devices: int, output_dir: Optional[str], file_basename: str) -> None:
                self.static: List[float] = [0.0 for _ in range(num_devices)]
                self.current: List[float] = [0.0 for _ in range(num_devices)]
                self.peak: List[float] = [0.0 for _ in range(num_devices)]
                # self.activation_allocations: memory_data["activation_allocations"]
                self._log_files: List[Optional[Any]] = [None for _ in range(num_devices)]
                self._log_paths: List[Optional[str]] = [None for _ in range(num_devices)]
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    for gpu_idx in range(num_devices):
                        log_path = os.path.join(output_dir, f"{file_basename}_gpu_{gpu_idx}_memory_log.txt")
                        self._log_paths[gpu_idx] = log_path
                        log_file = open(log_path, "w", encoding="utf-8")
                        log_file.write(
                            "timestamp_s | action               | delta_gib   | static_gib  | current_gib | peak_gib    | details\n"
                        )
                        self._log_files[gpu_idx] = log_file

            def add_static(self, gpu_id: int, size_bytes: float, timestamp: float = 0.0) -> None:
                if size_bytes <= 0:
                    return
                self.static[gpu_id] += size_bytes
                self.current[gpu_id] += size_bytes
                self._update_peak(gpu_id)
                self._record_change(
                    gpu_id,
                    "static_allocate",
                    size_bytes,
                    timestamp,
                    details="static_mem",
                )

            def allocate_activation(
                self,
                gpu_id: int,
                node: Any,
                timestamp: float,
                size_bytes: float,
                action: str = "allocate_activation",
            ) -> None:
                node_identifier = getattr(node, "op_id", None)
                if node_identifier is None:
                    node_identifier = id(node)

                if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
                    return
                self.current[gpu_id] += size_bytes
                details = "node={} op_id={} fwd={}".format(
                    getattr(node, "name", "unknown"),
                    getattr(node, "op_id", "N/A"),
                    getattr(node, "fwd", "N/A"),
                )
                self._update_peak(gpu_id)
                self._record_change(gpu_id, action, size_bytes, timestamp, details=details)

            def release_activation(
                self,
                gpu_id: int,
                node: Any,
                timestamp: float,
                size_bytes: float,
                action: str = "release_activation",
            ) -> None:
                node_identifier = getattr(node, "op_id", None)
                if node_identifier is None:
                    node_identifier = id(node)

                if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
                    return
                self.current[gpu_id] = max(0.0, self.current[gpu_id] - size_bytes)
                details = "node={} op_id={} fwd={}".format(
                    getattr(node, "name", "unknown"),
                    getattr(node, "op_id", "N/A"),
                    getattr(node, "fwd", "N/A"),
                )
                self._record_change(gpu_id, action, -size_bytes, timestamp, details=details)

            def allocate_ephemeral(self, gpu_id: int, node: Any, timestamp: float, size_bytes: float) -> None:
                self.allocate_activation(
                    gpu_id,
                    node,
                    timestamp,
                    size_bytes,
                    action="allocate_ephemeral",
                )

            def release_ephemeral(self, gpu_id: int, node: Any, timestamp: float, size_bytes: float) -> None:
                self.release_activation(
                    gpu_id,
                    node,
                    timestamp,
                    size_bytes,
                    action="release_ephemeral",
                )

            def summary(self) -> List[Dict[str, float]]:
                summary: List[Dict[str, float]] = []
                for idx in range(len(self.current)):
                    static_gib = self.static[idx] / BYTES_PER_GIB
                    current_gib = self.current[idx] / BYTES_PER_GIB
                    peak_gib = self.peak[idx] / BYTES_PER_GIB
                    summary.append(
                        {
                            "gpu_id": idx,
                            "static_gib": static_gib,
                            "current_gib": current_gib,
                            "peak_gib": peak_gib,
                        }
                    )
                return summary

            def _update_peak(self, gpu_id: int) -> None:
                if self.current[gpu_id] > self.peak[gpu_id]:
                    self.peak[gpu_id] = self.current[gpu_id]

            def _record_change(
                self,
                gpu_id: int,
                action: str,
                delta_bytes: float,
                timestamp: float,
                *,
                details: str = "",
            ) -> None:
                if not self._log_files or gpu_id < 0 or gpu_id >= len(self._log_files):
                    return
                log_file = self._log_files[gpu_id]
                if log_file is None:
                    return

                delta_gib = delta_bytes / BYTES_PER_GIB
                static_gib = self.static[gpu_id] / BYTES_PER_GIB
                current_gib = self.current[gpu_id] / BYTES_PER_GIB
                peak_gib = self.peak[gpu_id] / BYTES_PER_GIB
                line = "{:.6f} | {:<20} | {:>+.6f} | {:>+.6f} | {:>+.6f} | {:>+.6f}".format(
                    timestamp,
                    action,
                    delta_gib,
                    static_gib,
                    current_gib,
                    peak_gib,
                )
                if details:
                    line = f"{line} | {details}"
                log_file.write(line + "\n")
                log_file.flush()

            def close(self) -> None:
                if not self._log_files:
                    return
                for handle in self._log_files:
                    if handle:
                        handle.close()

        persistent_by_kind = memory_data.get("persistent_bytes_by_kind", {}) or {}
        transient_by_kind = memory_data.get("transient_bytes_by_kind", {}) or {}
        persistent_by_kind_dense = memory_data.get("persistent_bytes_by_kind_dense", persistent_by_kind) or {}
        persistent_by_kind_moe = memory_data.get("persistent_bytes_by_kind_moe", persistent_by_kind_dense) or {}
        transient_by_kind_dense = memory_data.get("transient_bytes_by_kind_dense", transient_by_kind) or {}
        transient_by_kind_moe = memory_data.get("transient_bytes_by_kind_moe", transient_by_kind_dense) or {}
        valid_kinds = set(persistent_by_kind) | set(transient_by_kind)
        param_gather_bytes = float(memory_data.get("param_gather_bytes", 0.0) or 0.0)

        def _is_transformer_block(node: Any) -> bool:
            """Return True when node represents a transformer compute block."""
            if not isinstance(node, Node):
                return False
            mem_kind = getattr(node, "mem_kind", None)
            if not isinstance(mem_kind, MemKind):
                return False
            if mem_kind in NON_TRANSFORMER_KINDS:
                return False
            return mem_kind in TRANSFORMER_OP_KINDS or mem_kind in valid_kinds

        def _node_mem_kind(node: Any) -> Optional[MemKind]:
            if not isinstance(node, Node):
                return None
            mem_kind = getattr(node, "mem_kind", None)
            return mem_kind if isinstance(mem_kind, MemKind) else None

        def _bytes_for_kind(kind: MemKind, mapping: Dict[MemKind, float], label: str) -> float:
            if kind not in mapping:
                return 0.0
            return float(mapping[kind] or 0.0)

        def _persistent_bytes_for_node(node: Any) -> float:
            mem_kind = _node_mem_kind(node)
            mapping = persistent_by_kind_moe if getattr(node, "is_moe_layer", False) and _is_transformer_block(node) else persistent_by_kind_dense
            return _bytes_for_kind(mem_kind, mapping, "persistent")

        def _transient_bytes_for_node(node: Any) -> float:
            mem_kind = _node_mem_kind(node)
            mapping = transient_by_kind_moe if getattr(node, "is_moe_layer", False) and _is_transformer_block(node) else transient_by_kind_dense
            return _bytes_for_kind(mem_kind, mapping, "transient")

        def _count_transformer_layers_per_device(root_obj: Any, num_devices: int) -> Tuple[List[int], List[int]]:
            dense_sets: List[Set[Any]] = [set() for _ in range(num_devices)]
            moe_sets: List[Set[Any]] = [set() for _ in range(num_devices)]
            stack_local = list(root_obj if isinstance(root_obj, (list, tuple)) else [root_obj])
            visited_local: Set[int] = set()
            while stack_local:
                current = stack_local.pop()
                current_id = id(current)
                if current_id in visited_local:
                    continue
                visited_local.add(current_id)

                if isinstance(current, Node):
                    hw_id = getattr(current, "hw_id", None)
                    layer_idx = getattr(current, "layer_index", None)
                    if (
                        layer_idx is not None
                        and isinstance(hw_id, int)
                        and 0 <= hw_id < num_devices
                        and _is_transformer_block(current)
                        and getattr(current, "fwd", True)
                    ):
                        if getattr(current, "is_moe_layer", False):
                            moe_sets[hw_id].add(layer_idx)
                        else:
                            dense_sets[hw_id].add(layer_idx)

                stack_local.extend(getattr(current, "children", []))
            return ([len(layer_set) for layer_set in dense_sets], [len(layer_set) for layer_set in moe_sets])

        memory_output_dir: Optional[str] = None
        if output_folder:
            memory_output_dir = os.path.join(output_folder, "memory-summary")

        transformer_dense_layers_per_device, transformer_moe_layers_per_device = _count_transformer_layers_per_device(root, base_devices)
        static_mem_per_layer = memory_data.get("static_mem_per_layer", 0)
        static_mem_per_layer_dense = memory_data.get("static_mem_per_layer_dense", static_mem_per_layer)
        static_mem_per_layer_moe = memory_data.get("static_mem_per_layer_moe", static_mem_per_layer_dense)
        weight_mem_per_layer = memory_data.get("weight_mem_per_layer", 0)
        weight_mem_per_layer_dense = memory_data.get("weight_mem_per_layer_dense", weight_mem_per_layer)
        weight_mem_per_layer_moe = memory_data.get("weight_mem_per_layer_moe", weight_mem_per_layer_dense)
        kv_cache_bytes_per_layer = memory_data.get("kv_cache_bytes_per_layer", 0.0)
        extra_static_bytes_per_device = memory_data.get("extra_static_bytes_per_device", {}) or {}
        total_tracked_layers = sum(transformer_dense_layers_per_device) + sum(transformer_moe_layers_per_device)
        if total_tracked_layers == 0:
            transformer_dense_layers_per_device = [1 for _ in range(base_devices)]
            transformer_moe_layers_per_device = [0 for _ in range(base_devices)]

        GPU_list = [True for _ in range(base_devices)]
        data_list = [False for _ in range(0, self.num_batch)]
        memory_snapshot = _MemorySnapshot(base_devices, memory_output_dir, filename)
        
        for idx in range(base_devices):
            
            if mode == "inference":
                dense_count = transformer_dense_layers_per_device[idx] if idx < len(transformer_dense_layers_per_device) else 0
                moe_count = transformer_moe_layers_per_device[idx] if idx < len(transformer_moe_layers_per_device) else 0
                layer_count = dense_count + moe_count
                if layer_count == 0:
                    dense_count = 1
                    layer_count = 1
                static_bytes = (weight_mem_per_layer_dense * dense_count) + (weight_mem_per_layer_moe * moe_count)
                if kv_cache_bytes_per_layer:
                    static_bytes += kv_cache_bytes_per_layer * layer_count
            elif mode == "training":
                dense_count = transformer_dense_layers_per_device[idx] if idx < len(transformer_dense_layers_per_device) else 0
                moe_count = transformer_moe_layers_per_device[idx] if idx < len(transformer_moe_layers_per_device) else 0
                layer_count = dense_count + moe_count
                if layer_count == 0:
                    dense_count = 1
                    layer_count = 1
                static_bytes = (static_mem_per_layer_dense * dense_count) + (static_mem_per_layer_moe * moe_count)
            else:
                raise ValueError(f"Invalid mode '{mode}' for memory simulation")
            static_bytes += extra_static_bytes_per_device.get(idx, 0.0)
            memory_snapshot.add_static(idx, static_bytes, timestamp=0.0)
            
        self.memory_monitor_snapshot = memory_snapshot

        heappush(event_queue, (root.duration, counter, root))
        if debug:
            print("{} enqueued at time {} batch id {}".format(root.name, 0, root.batch_id))
        ready_list.remove(root)
        counter = counter + 1

        while len(event_queue) > 0:
            time, _, event = heappop(event_queue)
            event.done = True
            event.scheduled = False
            event.finish_time = time
            if debug:
                print("Event {} finished at time {}".format(event.name, time))

            for child in event.children:

                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                # if is_ready == True:
                if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                    ready_list.append(child)
                    if debug:
                        print("child {}  ready at time {} ".format(child.name, time))

            if isinstance(event, Node):
                GPU_list[int(event.hw_id)] = True
                persistent_bytes = _persistent_bytes_for_node(event)
                transient_bytes = _transient_bytes_for_node(event)
                if mode == "training":
                    if event.fwd and persistent_bytes:
                        memory_snapshot.allocate_activation(int(event.hw_id), event, time, persistent_bytes)
                    if event.fwd and transient_bytes:
                        memory_snapshot.allocate_activation(int(event.hw_id), event, time, transient_bytes)
                        memory_snapshot.release_activation(int(event.hw_id), event, time, transient_bytes)
                    if not event.fwd and persistent_bytes:
                        memory_snapshot.release_activation(int(event.hw_id), event, time, persistent_bytes)
                elif mode == "inference":
                    if event.fwd and transient_bytes:
                        memory_snapshot.allocate_activation(int(event.hw_id), event, time, transient_bytes)
                        memory_snapshot.release_activation(int(event.hw_id), event, time, transient_bytes)
                if mode == "training" and param_gather_bytes is not None and getattr(event, "param_gather", False):
                    memory_snapshot.release_ephemeral(
                        int(event.hw_id),
                        event,
                        time,
                        param_gather_bytes,
                    )

                
            for event in ready_list[:]:
                enqueued = False
                if isinstance(event, Data_batch):
                    # if GPU_list[event.hw_id] == True:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{} enqueued at time".format(event.name,  time))
                    counter = counter + 1
                    data_list[event.batch_id] = True #data batch sent to gpu
                    # GPU_list[event.hw_id] = False
                    ready_list.remove(event)

                elif isinstance(event, Node): 
                    if GPU_list[int(event.hw_id)] == True:
                        new_time = time + event.duration
                        if mode == "training" and param_gather_bytes and getattr(event, "param_gather", False):
                            memory_snapshot.allocate_ephemeral(
                                int(event.hw_id),
                                event,
                                time,
                                param_gather_bytes,
                            )
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                        counter = counter + 1
                        GPU_list[int(event.hw_id)] = False
                        ready_list.remove(event)
                elif isinstance(event, Edge): 
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                    enqueued = True
                    counter = counter + 1
                    ready_list.remove(event)


        summary = memory_snapshot.summary()
        memory_snapshot.close()
        self.memory_monitor_summary = summary
        peak_mem = max(entry["peak_gib"] for entry in summary) if summary else 0.0
        return time, peak_mem

    def save_graph(self, roots, output_folder = "output/LLM/", filename="graph"):
        os.makedirs(output_folder, exist_ok=True)

        base_path = os.path.normpath(f"{output_folder}{filename}")
        svg_path = f"{base_path}.svg"
        display_path = util.relpath_display(svg_path)
        printstr = f" | Graph saved to    {display_path}"
        def _render_graph() -> None:
            dot_fw = visualize_graph(roots, filename=base_path)
            dot_fw.render(base_path, format="svg", cleanup=True)

        util.graphviz_submit(f"{filename}.svg", _render_graph, print_message=printstr)


def visualize_graph(roots, filename="graph"):
    _ = filename  # unused, kept for backwards compatibility with callers

    dot = Digraph(comment="Computation Graph", format="svg")
    visited = set()

    def _format_duration(value: float, profile: Optional[Tuple[float, ...]] = None) -> str:
        def _format_single(entry: float) -> str:
            ms = entry * 1e3
            if abs(ms) > 1000:
                return f"{entry:.2f}s"
            return f"{ms:.2f}ms"

        if not profile:
            return _format_single(value)

        groups: List[Dict[str, Any]] = []
        for idx, entry in enumerate(profile):
            matched = False
            for group in groups:
                if math.isclose(entry, group["value"], rel_tol=1e-9, abs_tol=1e-12):
                    group["indices"].append(idx)
                    matched = True
                    break
            if not matched:
                groups.append({"value": entry, "indices": [idx]})

        if len(groups) == 1:
            return _format_single(groups[0]["value"])

        parts = []
        for group in groups:
            indices = ",".join(str(i) for i in group["indices"])
            parts.append(f"{{{indices}}}: {_format_single(group['value'])}")
        return ", ".join(parts)

    def _node_color(node) -> str:
        if isinstance(node, Node):
            return "lightblue" if node.fwd else "lightcoral"
        if isinstance(node, Data_batch):
            return "gray"
        if isinstance(node, Edge):
            if getattr(node, "is_dp", False):
                return "green"
            else:
                if getattr(node, "comm_type", None) == CollectiveType.PIPELINE:
                    return "white"
                else:
                    return "yellow"
        return "mediumorchid"

    def _node_label(node) -> str:
        if isinstance(node, Data_batch):
            return f"{node.name}\n(batch_id={node.batch_id}, dur={_format_duration(node.duration)})"
        if isinstance(node, Node):
            duration_display = _format_duration(node.duration, node.duration_profile)
            return (
                f"{node.name}\n(op_id={node.op_id}, hw_id={node.hw_id}, "
                f"dur={duration_display})"
            )
        if isinstance(node, Edge):
            duration_display = _format_duration(node.duration)
            if getattr(node, "local_hw_id", None) is not None:
                return (
                    f"{node.name}\n(op_id={node.op_id}, local_hw_id={node.local_hw_id}, "
                    f"dur={duration_display})"
                )
            return f"{node.name}\n(op_id={node.op_id}, dur={duration_display})"
        return str(node)

    def _visit(node):
        if node in visited:
            return

        visited.add(node)
        node_id = str(id(node))
        dot.node(node_id, label=_node_label(node), style="filled", fillcolor=_node_color(node), shape="box")

        for child in getattr(node, "children", []):
            child_id = str(id(child))
            dot.edge(node_id, child_id)
            _visit(child)

    if isinstance(roots, (list, tuple, set)):
        iterable = roots
    else:
        iterable = [roots]

    for root in iterable:
        _visit(root)

    return dot


        
