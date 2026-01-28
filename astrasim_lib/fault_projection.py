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

"""Utilities for projecting faulty link endpoints onto axis subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, List

from .layout_utils import AxisLayout, decode_axis_coordinates, build_subset_strides


def _coords_dict(coords: Tuple[Tuple[str, int], ...]) -> Dict[str, int]:
    return {axis: value for axis, value in coords}


def _normalize_axes(axes: Sequence[str]) -> Tuple[str, ...]:
    normalized: List[str] = []
    for entry in axes:
        axis = str(entry).strip().lower()
        if not axis:
            continue
        if axis not in normalized:
            normalized.append(axis)
    return tuple(normalized)


@dataclass(frozen=True)
class FaultSpaceEntry:
    original: Tuple[int, int, float]
    src_coords: Tuple[Tuple[str, int], ...]
    dst_coords: Tuple[Tuple[str, int], ...]
    affected_axes: Tuple[str, ...]
    affected_dimensions: Tuple[int, ...]


@dataclass(frozen=True)
class FaultProjectionDetail:
    original: Tuple[int, int, float]
    remapped: Tuple[int, int, float]
    subset_axes: Tuple[str, ...]
    affected_axes: Tuple[str, ...]
    affected_dimensions: Tuple[int, ...]
    src_coords: Tuple[Tuple[str, int], ...]
    dst_coords: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class FaultProjectionResult:
    subset_axes: Tuple[str, ...]
    remapped_links: Tuple[Tuple[int, int, float], ...]
    entries: Tuple[FaultProjectionDetail, ...]
    covered_originals: Tuple[Tuple[int, int, float], ...]
    skipped_originals: Tuple[Tuple[int, int, float], ...]


class FaultSpace:
    """Decoded view of global faulty links for downstream projections."""

    def __init__(
        self,
        layout: AxisLayout,
        faulty_links: Iterable[Tuple[int, int, float]],
        axis_to_dimension: Optional[Mapping[str, int]] = None,
    ) -> None:
        entries: List[FaultSpaceEntry] = []
        layout_order = layout.axis_order
        for idx, link in enumerate(faulty_links):
            if len(link) != 3:
                raise ValueError(f"faulty_links[{idx}] must be a (src, dst, weight) tuple.")
            src, dst, weight = link
            src_coords = decode_axis_coordinates(int(src), layout)
            dst_coords = decode_axis_coordinates(int(dst), layout)
            src_dict = _coords_dict(src_coords)
            dst_dict = _coords_dict(dst_coords)
            affected_axes = tuple(
                axis for axis in layout_order if src_dict.get(axis) != dst_dict.get(axis)
            )
            if axis_to_dimension is not None:
                dims = {
                    axis_to_dimension[axis]
                    for axis in affected_axes
                    if axis in axis_to_dimension
                }
                affected_dimensions = tuple(sorted(dims))
            else:
                affected_dimensions = tuple()
            entries.append(
                FaultSpaceEntry(
                    original=(int(src), int(dst), float(weight)),
                    src_coords=src_coords,
                    dst_coords=dst_coords,
                    affected_axes=affected_axes,
                    affected_dimensions=affected_dimensions,
                )
            )
        self._entries: Tuple[FaultSpaceEntry, ...] = tuple(entries)
        self._layout = layout

    @property
    def layout(self) -> AxisLayout:
        return self._layout

    @property
    def entries(self) -> Tuple[FaultSpaceEntry, ...]:
        return self._entries

    def project(self, subset_axes: Sequence[str]) -> FaultProjectionResult:
        normalized = _normalize_axes(subset_axes)
        if not normalized:
            raise ValueError("subset_axes must not be empty when projecting faults.")
        subset_axes_set = set(normalized)
        subset_strides = build_subset_strides(normalized, self._layout.axis_sizes)
        details: List[FaultProjectionDetail] = []
        remapped_links: List[Tuple[int, int, float]] = []
        covered: List[Tuple[int, int, float]] = []
        unmapped: List[Tuple[int, int, float]] = []

        for entry in self._entries:
            if not any(axis in subset_axes_set for axis in entry.affected_axes):
                unmapped.append(entry.original)
                continue
            src_dict = _coords_dict(entry.src_coords)
            dst_dict = _coords_dict(entry.dst_coords)
            src_linear = _linearize_subset(src_dict, normalized, subset_strides)
            dst_linear = _linearize_subset(dst_dict, normalized, subset_strides)
            remapped = (src_linear, dst_linear, entry.original[2])
            details.append(
                FaultProjectionDetail(
                    original=entry.original,
                    remapped=remapped,
                    subset_axes=normalized,
                    affected_axes=entry.affected_axes,
                    affected_dimensions=entry.affected_dimensions,
                    src_coords=entry.src_coords,
                    dst_coords=entry.dst_coords,
                )
            )
            remapped_links.append(remapped)
            covered.append(entry.original)

        return FaultProjectionResult(
            subset_axes=normalized,
            remapped_links=tuple(remapped_links),
            entries=tuple(details),
            covered_originals=tuple(covered),
            skipped_originals=tuple(unmapped),
        )


def _linearize_subset(
    coords: Mapping[str, int],
    subset_axes: Sequence[str],
    subset_strides: Mapping[str, int],
) -> int:
    linear = 0
    for axis in subset_axes:
        coord = coords.get(axis, 0)
        stride = subset_strides[axis]
        linear += coord * stride
    return linear
