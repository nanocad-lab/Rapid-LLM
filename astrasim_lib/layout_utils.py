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

"""Shared helpers for reasoning about RAPID-LLM axis layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple, List


def _normalize_axis_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Axis name must be a string, got {type(name)!r}")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Axis name cannot be empty or whitespace.")
    return normalized


@dataclass(frozen=True)
class AxisLayout:
    """Immutable view of the flattened axis layout that defines hw_id numbering."""

    axis_order: Tuple[str, ...]
    axis_sizes: Dict[str, int]
    axis_strides: Dict[str, int]
    total_ranks: int

    def __post_init__(self) -> None:
        if any(size < 1 for size in self.axis_sizes.values()):
            raise ValueError("Axis sizes must all be >= 1.")
        for axis in self.axis_order:
            if axis not in self.axis_sizes:
                raise ValueError(f"Axis '{axis}' is missing a declared size.")
            if axis not in self.axis_strides:
                raise ValueError(f"Axis '{axis}' is missing a declared stride.")


def axis_layout_from_descriptor(descriptor: Mapping[str, Any]) -> AxisLayout:
    """Create an :class:`AxisLayout` from `_build_rank_layout_descriptor` output."""
    if descriptor is None:
        raise ValueError("Axis layout descriptor must not be None.")

    axis_order_raw = descriptor.get("axis_order", [])
    if not isinstance(axis_order_raw, Sequence):
        raise TypeError("Axis layout descriptor axis_order must be a sequence.")
    axis_order: Tuple[str, ...] = tuple(_normalize_axis_name(entry) for entry in axis_order_raw)
    if not axis_order:
        raise ValueError("Axis layout descriptor axis_order is empty.")

    sizes_raw = descriptor.get("axis_sizes", {})
    if not isinstance(sizes_raw, Mapping):
        raise TypeError("Axis layout descriptor axis_sizes must be a mapping.")
    axis_sizes: Dict[str, int] = {}
    for key, value in sizes_raw.items():
        axis = _normalize_axis_name(str(key))
        try:
            size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Axis '{axis}' size must be an integer.") from exc
        if size < 1:
            raise ValueError(f"Axis '{axis}' size must be >= 1.")
        axis_sizes[axis] = size

    strides_raw = descriptor.get("axis_strides", {})
    if strides_raw and not isinstance(strides_raw, Mapping):
        raise TypeError("Axis layout descriptor axis_strides must be a mapping when provided.")

    axis_strides: Dict[str, int] = {}
    if strides_raw:
        for key, value in strides_raw.items():
            axis = _normalize_axis_name(str(key))
            try:
                stride = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Axis '{axis}' stride must be an integer.") from exc
            if stride < 0:
                raise ValueError(f"Axis '{axis}' stride must be >= 0.")
            axis_strides[axis] = stride

    if not axis_strides:
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes.get(axis, 1)

    total = 1
    for axis in axis_order:
        total *= axis_sizes[axis]

    return AxisLayout(
        axis_order=axis_order,
        axis_sizes=axis_sizes,
        axis_strides=axis_strides,
        total_ranks=total,
    )


def decode_axis_coordinates(hw_id: int, layout: AxisLayout) -> Tuple[Tuple[str, int], ...]:
    """Return ordered axis coordinates for ``hw_id``."""
    if hw_id < 0:
        raise ValueError(f"Hardware id must be >= 0, got {hw_id}.")
    if hw_id >= layout.total_ranks:
        raise ValueError(
            f"Hardware id {hw_id} exceeds layout span ({layout.total_ranks} ranks)."
        )
    coords: List[Tuple[str, int]] = []
    for axis in layout.axis_order:
        stride = layout.axis_strides[axis]
        size = layout.axis_sizes[axis]
        if size == 1:
            coords.append((axis, 0))
            continue
        coord = (hw_id // stride) % size
        coords.append((axis, coord))
    return tuple(coords)


def build_subset_strides(
    subset_axes: Sequence[str],
    axis_sizes: Mapping[str, int],
) -> Dict[str, int]:
    """Compute strides for a subset axis ordering."""
    strides: Dict[str, int] = {}
    span = 1
    for raw_axis in subset_axes:
        axis = _normalize_axis_name(raw_axis)
        if axis not in axis_sizes:
            raise ValueError(f"Axis '{axis}' is not defined in the layout axis_sizes.")
        size = axis_sizes[axis]
        if size < 1:
            raise ValueError(f"Axis '{axis}' size must be >= 1.")
        strides[axis] = span
        span *= size
    return strides


def derive_axes_filter(
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
    dp_count: int,
) -> List[str]:
    """Return the axes filter used to select network dimensions for AstraSim."""
    axes: List[str] = []
    for entry in axis_order:
        axis = _normalize_axis_name(entry)
        try:
            size = max(1, int(axis_sizes.get(axis, 1)))
        except Exception:
            size = 1
        if size > 1 and axis not in axes:
            axes.append(axis)
    if dp_count > 1:
        axes = [axis for axis in axes if axis != "dp"]
        axes.append("dp")
    return axes
