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

"""First-dimension traffic aggregation and SCOTCH mapping helpers."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .layout_utils import build_subset_strides

import scotchpy
from scotchpy.common import proper_int as sct_i
from scotchpy.maporder import Mapping as ScotchMapping
from scotchpy import strat as scotch_strat

_EDGE_WEIGHT_SCALE_BYTES = 1024 * 1024 * 1024  # convert bytes -> gigabytes for SCOTCH stability


@dataclass(frozen=True)
class MappingResult:
    permutation: Tuple[int, ...]
    graph_path: str
    map_path: str
    metrics_path: str


class GMapCollector:
    """Accumulates per-pair byte weights for the first network dimension."""

    def __init__(
        self,
        *,
        topology: str,
        subset_axes: Tuple[str, ...],
        axis_sizes: Mapping[str, int],
        subset_strides: Mapping[str, int],
        stage_axis_coords: Mapping[int, Mapping[str, int]],
        output_dir: str,
        vertex_count: int,
        include_pipeline: bool,
        dims: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not subset_axes:
            raise ValueError("optimize_2dmap requires at least one axis in the first network dimension")

        self.topology = str(topology).strip().lower()
        self.subset_axes = tuple(subset_axes)
        self.axis_sizes = dict(axis_sizes)
        self.subset_strides = dict(subset_strides)
        self.stage_axis_coords = stage_axis_coords
        self.output_dir = output_dir
        self.vertex_count = int(vertex_count)
        self.include_pipeline = include_pipeline
        self.dims: Optional[Tuple[int, int]] = tuple(int(v) for v in dims) if dims else None
        self._edge_weights: Dict[Tuple[int, int], float] = {}

        missing = [axis for axis in self.subset_axes if axis not in self.axis_sizes]
        if missing:
            raise ValueError(f"Subset axes missing from axis_sizes: {missing}")

        product = 1
        for axis in self.subset_axes:
            product *= int(self.axis_sizes[axis])
        if product != self.vertex_count:
            raise ValueError(
                f"Vertex count {self.vertex_count} does not match product of subset axis sizes {product}."
            )

        self.replication_axes: Tuple[str, ...] = tuple(
            axis
            for axis in self.axis_sizes
            if axis not in self.subset_axes and int(self.axis_sizes[axis]) > 1
        )
        self._axis_groups = self._build_axis_groups()
        self._canonical_signature = self._select_canonical_signature()

    # ------------------------------------------------------------------
    # Canonical cluster helpers (only one PP/DP slice contributes weights)
    # ------------------------------------------------------------------

    def _select_canonical_signature(self) -> Optional[Tuple[Tuple[str, int], ...]]:
        if not self.replication_axes:
            return None
        zero_signature = tuple((axis, 0) for axis in self.replication_axes)
        for coords in self.stage_axis_coords.values():
            if all(int(coords.get(axis, 0)) == 0 for axis in self.replication_axes):
                return zero_signature
        signatures = sorted(
            {
                tuple((axis, int(coords.get(axis, 0))) for axis in self.replication_axes)
                for coords in self.stage_axis_coords.values()
            }
        )
        if not signatures:
            raise ValueError("Unable to determine canonical cluster for optimize_2dmap.")
        return signatures[0]

    def _is_canonical_stage(self, stage_id: int) -> bool:
        if not self.replication_axes:
            return True
        coords = self.stage_axis_coords.get(stage_id)
        if coords is None:
            raise ValueError(f"Unknown stage id {stage_id} when checking canonical membership.")
        signature = tuple((axis, int(coords.get(axis, 0))) for axis in self.replication_axes)
        return signature == self._canonical_signature

    # ------------------------------------------------------------------
    # Axis indexing helpers
    # ------------------------------------------------------------------

    def _coord(self, coords: Mapping[str, int], axis: str) -> int:
        if axis not in coords:
            raise ValueError(f"Axis '{axis}' missing from coordinate tuple {coords}.")
        return int(coords[axis])

    def _build_axis_groups(self) -> Dict[str, Dict[Tuple[Tuple[str, int], ...], Tuple[int, ...]]]:
        axis_groups: Dict[str, Dict[Tuple[Tuple[str, int], ...], List[int]]] = {
            axis: {} for axis in self.subset_axes
        }
        for stage_id, coords in self.stage_axis_coords.items():
            for axis in self.subset_axes:
                key = self._axis_key_entries(axis, coords)
                axis_groups[axis].setdefault(key, []).append(stage_id)
        normalized: Dict[str, Dict[Tuple[Tuple[str, int], ...], Tuple[int, ...]]] = {}
        for axis, entries in axis_groups.items():
            normalized[axis] = {
                key: tuple(sorted(stages))
                for key, stages in entries.items()
            }
        return normalized

    def local_index_for_stage(self, stage_id: int) -> int:
        coords = self.stage_axis_coords.get(stage_id)
        if coords is None:
            raise ValueError(f"Unknown stage id {stage_id} when computing local index.")
        local = 0
        for axis in self.subset_axes:
            coord = self._coord(coords, axis)
            stride = self.subset_strides[axis]
            local += coord * stride
        return local

    def coords_from_local_index(self, local_index: int) -> Dict[str, int]:
        coords: Dict[str, int] = {}
        for axis in reversed(self.subset_axes):
            stride = self.subset_strides[axis]
            size = int(self.axis_sizes[axis])
            if stride == 0:
                coords[axis] = 0
                continue
            coords[axis] = (local_index // stride) % size
        return coords

    # ------------------------------------------------------------------
    # Byte aggregation
    # ------------------------------------------------------------------

    def _per_pair_bytes(self, total_bytes: int, participants: int, double_weight: bool) -> float:
        if participants <= 1:
            raise ValueError("participants must be > 1 to compute per-pair bytes")
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        per_pair = float(total_bytes) / (participants * (participants - 1))
        if double_weight:
            per_pair *= 2.0
        return per_pair

    def _accumulate_pairs(self, local_indices: Sequence[int], per_pair_bytes: float) -> None:
        unique = sorted(set(int(idx) for idx in local_indices))
        if len(unique) <= 1 or per_pair_bytes <= 0:
            return
        for i, src in enumerate(unique):
            for dst in unique[i + 1 :]:
                key = (src, dst)
                self._edge_weights[key] = self._edge_weights.get(key, 0.0) + per_pair_bytes

    def _axis_key_entries(self, axis: str, coords: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
        key_entries: List[Tuple[str, int]] = []
        for other_axis in self.subset_axes:
            if other_axis == axis:
                continue
            key_entries.append((other_axis, self._coord(coords, other_axis)))
        for repl_axis in self.replication_axes:
            key_entries.append((repl_axis, self._coord(coords, repl_axis)))
        return tuple(key_entries)

    def _participants_for_axis(self, axis: str, coords: Mapping[str, int]) -> Optional[Tuple[int, ...]]:
        axis_map = self._axis_groups.get(axis)
        if not axis_map:
            return None
        key = self._axis_key_entries(axis, coords)
        return axis_map.get(key)

    def record_collective(
        self,
        *,
        axis: str,
        stage_id: int,
        size_bytes: int,
        participant_count: int,
        double_weight: bool,
    ) -> None:
        if not self._is_canonical_stage(stage_id):
            return
        axis_norm = str(axis).strip().lower()
        coords = self.stage_axis_coords.get(stage_id)
        if coords is None:
            raise ValueError(f"Unknown stage id {stage_id} when resolving participants.")
        axis_candidates = [axis_norm] + [ax for ax in self.subset_axes if ax != axis_norm]
        participants: Optional[Tuple[int, ...]] = None
        chosen_axis = axis_norm
        for candidate in axis_candidates:
            group = self._participants_for_axis(candidate, coords)
            if not group:
                continue
            if participant_count > 1 and len(group) == participant_count:
                participants = group
                chosen_axis = candidate
                break
            if participants is None:
                participants = group
                chosen_axis = candidate
        if not participants:
            raise ValueError(f"Failed to resolve participants for axis '{axis_norm}' at stage {stage_id}.")
        axis_norm = chosen_axis
        effective_participants = participant_count if participant_count > 1 else len(participants)
        effective_participants = max(effective_participants, len(participants))
        if effective_participants <= 1:
            return
        local_indices = [self.local_index_for_stage(stage) for stage in participants]
        per_pair = self._per_pair_bytes(size_bytes, effective_participants, double_weight)
        self._accumulate_pairs(local_indices, per_pair)

    def record_pipeline(self, src_stage: int, dst_stage: int, size_bytes: int) -> None:
        if not self.include_pipeline:
            return
        if not (self._is_canonical_stage(src_stage) and self._is_canonical_stage(dst_stage)):
            return
        if size_bytes < 0:
            raise ValueError("Pipeline edge size must be non-negative")
        local_indices = [self.local_index_for_stage(src_stage), self.local_index_for_stage(dst_stage)]
        per_pair = self._per_pair_bytes(size_bytes, 2, False)
        self._accumulate_pairs(local_indices, per_pair)

    # ------------------------------------------------------------------
    # Graph emission helpers
    # ------------------------------------------------------------------

    def emit_graph(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "first_dim_comm.grf")
        adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(self.vertex_count)]
        for (src, dst), weight in self._edge_weights.items():
            weight /= _EDGE_WEIGHT_SCALE_BYTES
            w_int = max(1, int(round(weight)))
            adjacency[src].append((dst, w_int))
            adjacency[dst].append((src, w_int))
        num_arcs = sum(len(neigh) for neigh in adjacency)
        with open(path, "w", encoding="ascii") as handle:
            handle.write("0\n")
            handle.write(f"{self.vertex_count} {num_arcs}\n")
            handle.write("0 010\n")
            for neighbors in adjacency:
                neighbors.sort(key=lambda item: item[0])
                line_parts: List[str] = [str(len(neighbors))]
                for neighbor, weight in neighbors:
                    line_parts.extend([str(weight), str(neighbor)])
                handle.write(" ".join(line_parts) + "\n")
        return path

    def build_csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nbrs = [[] for _ in range(self.vertex_count)]
        weights = [[] for _ in range(self.vertex_count)]
        for (src, dst), weight in self._edge_weights.items():
            if src == dst or weight <= 0:
                continue
            # print(f"src: {src}, dst: {dst}, weight: {weight}")
            w_int = _scotch_weight(weight)
            # new weight
            # print(f"w_int: {w_int}")
            nbrs[src].append(dst)
            weights[src].append(w_int)
            nbrs[dst].append(src)
            weights[dst].append(w_int)
        dtype = sct_i if sct_i is not None else np.int64
        xadj = np.zeros(self.vertex_count + 1, dtype=dtype)
        adjncy: List[int] = []
        adjwgt: List[int] = []
        for idx in range(self.vertex_count):
            xadj[idx + 1] = xadj[idx] + len(nbrs[idx])
            adjncy.extend(nbrs[idx])
            adjwgt.extend(weights[idx])
        return xadj, np.array(adjncy, dtype=dtype), np.array(adjwgt, dtype=dtype)


def begin_collection(
    optimize_cfg: Optional[Mapping[str, object]],
    axis_sizes: Mapping[str, int],
    stage_axis_coords: Mapping[int, Mapping[str, int]],
    output_dir: str,
) -> Optional[GMapCollector]:
    if not optimize_cfg:
        return None
    subset_axes_raw = optimize_cfg.get("parallelisms")
    if not subset_axes_raw:
        raise ValueError("optimize_2dmap requires at least one parallelism axis in the first dimension.")
    subset_axes: Tuple[str, ...] = tuple(
        str(axis).strip().lower()
        for axis in subset_axes_raw
        if str(axis).strip()
    )
    subset_strides = build_subset_strides(subset_axes, axis_sizes)
    dims_raw = optimize_cfg.get("dims")
    if dims_raw is None and isinstance(optimize_cfg.get("size"), (tuple, list)):
        dims_raw = optimize_cfg.get("size")
    dims: Optional[Tuple[int, int]] = None
    if dims_raw is not None:
        if not isinstance(dims_raw, (tuple, list)) or len(dims_raw) != 2:
            raise ValueError("optimize_2dmap dims must be a two-item tuple/list when provided.")
        try:
            dims = (int(dims_raw[0]), int(dims_raw[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("optimize_2dmap dims entries must be integers.") from exc
        if dims[0] <= 0 or dims[1] <= 0:
            raise ValueError("optimize_2dmap dims entries must be > 0.")

    subset_product = 1
    for axis in subset_axes:
        if axis not in axis_sizes:
            raise ValueError(f"Axis '{axis}' from optimize_2dmap is missing in axis_sizes.")
        subset_product *= int(axis_sizes[axis])

    if dims:
        vertex_count = int(dims[0]) * int(dims[1])
        if vertex_count != subset_product:
            raise ValueError(
                f"optimize_2dmap dims product {vertex_count} does not match product of subset axes {subset_product}."
            )
    else:
        vertex_count = subset_product
    declared_size_raw = optimize_cfg.get("size", vertex_count)
    if isinstance(declared_size_raw, (tuple, list)):
        try:
            declared_size = int(declared_size_raw[0]) * int(declared_size_raw[1])
        except Exception:
            declared_size = vertex_count
    else:
        declared_size = int(declared_size_raw)
    if declared_size != vertex_count:
        raise ValueError(
            f"optimize_2dmap size {declared_size} does not match product of subset axes {vertex_count}."
        )
    topology = str(optimize_cfg.get("topology", "")).strip()
    if not topology:
        raise ValueError("optimize_2dmap requires a topology string.")
    include_pipeline = "pp" in subset_axes
    return GMapCollector(
        topology=topology,
        subset_axes=subset_axes,
        axis_sizes=axis_sizes,
        subset_strides=subset_strides,
        stage_axis_coords=stage_axis_coords,
        output_dir=output_dir,
        vertex_count=vertex_count,
        include_pipeline=include_pipeline,
        dims=dims,
    )


def finalize_collection(collector: Optional[GMapCollector]) -> Optional[MappingResult]:
    if collector is None:
        return None
    graph_path = collector.emit_graph()
    xadj, adjncy, adjwgt = collector.build_csr()
    g = scotchpy.Graph()
    g.build(
        vertnbr=int(collector.vertex_count),
        verttab=xadj,
        vendtab=None,
        velotab=None,
        vlbltab=None,
        edgenbr=int(len(adjncy)),
        edgetab=adjncy,
        edlotab=adjwgt,
    )
    dims = _infer_target_dims(collector)
    arch = _build_architecture(collector.topology, dims)
    baseline_map = np.arange(collector.vertex_count, dtype=np.int64)
    before_metrics = _collect_map_metrics(g, arch, baseline_map.copy(), run_map=False)
    maptab = np.empty(collector.vertex_count, dtype=np.int64)
    after_metrics = _collect_map_metrics(g, arch, maptab, run_map=True)
    before_commexp, after_commexp = _commexpan_values(before_metrics, after_metrics)
    should_apply_mapping = True
    if before_commexp is not None and after_commexp is not None:
        should_apply_mapping = after_commexp < before_commexp
    map_path = os.path.join(collector.output_dir, "first_dim_comm.out")
    final_map = maptab if should_apply_mapping else baseline_map.copy()
    _write_mapping_file(map_path, final_map)
    metrics_path = os.path.join(collector.output_dir, "first_dim_comm.metrics")
    _write_metrics_report(metrics_path, before_metrics, after_metrics)
    _maybe_print_commexpan_delta(before_commexp, after_commexp, should_apply_mapping)
    if _env_truthy("RAPID_GMAP_ONLY"):
        status = "applied" if should_apply_mapping else "skipped"
        print(
            "[GMapDebug] RAPID_GMAP_ONLY=1; "
            f"{status} mapping. Artifacts: map={map_path}, metrics={metrics_path}"
        )
        raise SystemExit(0)
    permutation = tuple(int(val) for val in final_map)
    return MappingResult(
        permutation=permutation,
        graph_path=graph_path,
        map_path=map_path,
        metrics_path=metrics_path,
    )


def _infer_target_dims(collector: GMapCollector) -> Tuple[int, int]:
    if collector.dims:
        dim_x, dim_y = collector.dims
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError("optimize_2dmap requires positive dims entries.")
        return dim_x, dim_y
    total = int(collector.vertex_count)
    if total <= 0:
        raise ValueError("optimize_2dmap requires a positive vertex count.")
    side = int(math.isqrt(total))
    if side * side != total:
        raise ValueError(
            f"optimize_2dmap first-dimension size {total} is not a perfect square; Mesh2D/Torus2D mapping now requires square layouts."
        )
    return side, side


def _build_architecture(topology: str, dims: Tuple[int, int]):
    dim_x, dim_y = dims
    topo = topology.lower()
    arch = scotchpy.Arch()
    if topo == "mesh2d":
        arch.mesh2(int(dim_x), int(dim_y))
        return arch
    if topo == "torus2d":
        arch.torus2(int(dim_x), int(dim_y))
        return arch
    if topo == "kingmesh2d":
        # SCOTCH lacks a KingMesh2D architecture; approximate with Mesh2D.
        arch.mesh2(int(dim_x), int(dim_y))
        return arch
    raise ValueError(f"Unsupported topology '{topology}' for optimize_2dmap.")


def _scotch_weight(weight_bytes: float) -> int:
    if weight_bytes <= 0:
        return 0
    scaled = weight_bytes / float(_EDGE_WEIGHT_SCALE_BYTES)
    if scaled < 1.0:
        scaled = 1.0
    return max(1, int(round(scaled)))


def _write_mapping_file(path: str, maptab: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as handle:
        handle.write(f"{len(maptab)}\n")
        for idx, target in enumerate(maptab):
            handle.write(f"{idx} {int(target)}\n")


def _write_metrics_report(path: str, before: str, after: str) -> None:
    with open(path, "w", encoding="ascii") as handle:
        handle.write("SCOTCH Mapping Metrics (graphMapView output)\n\n")
        handle.write("Before:\n")
        handle.write(before.rstrip() + "\n\n")
        handle.write("After:\n")
        handle.write(after.rstrip() + "\n")


def _collect_map_metrics(
    graph: scotchpy.Graph, arch: scotchpy.Arch, parttab: np.ndarray, *, run_map: bool
) -> str:
    mapping = ScotchMapping()
    graph.map_init(mapping, arch, parttab=parttab)
    try:
        if run_map:
            strategy = scotch_strat.Strat()
            graph.map_compute(mapping, strategy)
        with tempfile.TemporaryFile(mode="w+") as tmp:
            graph.map_view(mapping, tmp)
            tmp.seek(0)
            return tmp.read()
    finally:
        graph.map_exit(mapping)


def _maybe_print_commexpan_delta(
    before_val: Optional[float], after_val: Optional[float], applied: bool
) -> None:
    if before_val is None or after_val is None or before_val == 0:
        return
    if not (_env_truthy("RAPID_VISUALIZE_GRAPHS") or _env_truthy("RAPID_GMAP_ONLY")):
        return
    delta_pct = (after_val - before_val) / before_val * 100.0
    status = "applied" if applied else "skipped"
    print(
        "[GMapDebug] CommExpan before "
        f"{before_val:.6f} after {after_val:.6f} change {delta_pct:+.2f}% ({status})"
    )


def _extract_commexpan(metrics: str) -> Optional[float]:
    for line in metrics.splitlines():
        if "CommExpan=" in line:
            try:
                prefix = line.split("CommExpan=")[1]
                value_str = prefix.split("\t", 1)[0]
                return float(value_str)
            except (ValueError, IndexError):
                return None
    return None


def _commexpan_values(before: str, after: str) -> Tuple[Optional[float], Optional[float]]:
    return _extract_commexpan(before), _extract_commexpan(after)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}
