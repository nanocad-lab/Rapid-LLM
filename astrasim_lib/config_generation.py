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

"""Generate AstraSim configuration artifacts from RAPID-LLM hardware configs."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from types import SimpleNamespace
import math
from hw_component import Network

from .bootstrap import ensure_chakra_available

# Ensure Chakra dependencies are importable for downstream modules that rely on
# protobuf definitions. This module itself does not import them but provides the
# same setup entry point for consistency.
ensure_chakra_available()

ASTRA_DEBUG = False


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        import json as _json

        _json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)


def _gbps_from_bps(bps: float) -> float:
    """Convert raw bits-per-second throughput to gigabytes-per-second."""

    return float(bps) / float(1 << 30)


def _ns_from_s(sec: float) -> float:
    """Convert seconds to nanoseconds."""

    return float(sec) * 1e9


def _effective_bw_values(bandwidth: object, util: float) -> object:
    if isinstance(bandwidth, (list, tuple)):
        return tuple(float(val) * util for val in bandwidth)
    return float(bandwidth) * util


def _as_gbps(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(round(_gbps_from_bps(float(v)), 6) for v in value)
    return round(_gbps_from_bps(float(value)), 6)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def choose_collective(alg: str, topo: str, op: str) -> str:
    """Resolve ``auto`` policies for collective algorithms (post-2D unpacking)."""
    if alg != "auto":
        return alg
    if topo == "FullyConnected":
        return "direct"
    if topo == "Switch":
        return "halvingDoubling"
    if topo == "Mesh":
        return "mesh"
    if topo == "HyperCube":
        return "hypercube"
    if topo in ("KingMesh2D", "KingMesh"):
        return "kingmesh"
    return "ring"


def compute_intra_inter_ib_ll_from_hw(hw_obj) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return intra/inter bandwidth+latency tuples from a parsed RAPID-LLM config."""
    net = Network(hw_obj)
    intra_throughput, inter_throughput = net.calc_throughput()
    intra_latency, inter_latency = net.calc_latency()
    return (intra_throughput, intra_latency), (inter_throughput, inter_latency)


def derive_topology_from_hw(hw_obj) -> str:
    """Map RAPID-LLM network topology enums to AstraSim names."""
    layout = getattr(hw_obj, "network_layout", None)
    primary = layout.primary_dimension() if layout else None
    topo = (primary.topology_type if primary else "ring") or "ring"
    return _normalize_topology_name(topo)


def _normalize_topology_name(topo: str) -> str:
    topo_str = str(topo).lower()
    topo_flat = topo_str.replace("-", "").replace("_", "")
    if topo_str in ("fc", "fullyconnected", "fully_connected", "fully-connected"):
        return "FullyConnected"
    if topo_str in ("ring",):
        return "Ring"
    if topo_str in ("switch",):
        return "Switch"
    if topo_flat == "superpod":
        return "SuperPOD"
    if topo_flat == "fcring2d":
        return "FCRing2D"
    if topo_str in ("torus2d"):
        return "Torus2D"
    if topo_str in ("mesh",):
        return "Mesh"
    if topo_str in ("hypercube",):
        return "HyperCube"
    if topo_str in ("mesh2d", "mesh-2d"):
        return "Mesh2D"
    if topo_str in ("kingmesh2d", "king-mesh2d"):
        return "KingMesh2D"
    return "FullyConnected"


def _is_2d_topology(topo: str) -> bool:
    topo_str = str(topo).strip().lower().replace("-", "").replace("_", "")
    return topo_str in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}


def _resolve_2d_dims(
    dims_value: Sequence[object],
    axis_sizes: Mapping[str, int],
    *,
    dim_label: str,
    product_size: int,
) -> Tuple[int, int]:
    if not isinstance(dims_value, (list, tuple)) or len(dims_value) != 2:
        raise ValueError(
            f"Network dimension '{dim_label}' 2D size must be a two-item sequence."
        )
    resolved: List[Optional[int]] = [None, None]
    auto_idx: Optional[int] = None
    for idx, entry in enumerate(dims_value):
        if isinstance(entry, str):
            normalized = entry.strip().lower()
            if normalized == "auto":
                if auto_idx is not None:
                    raise ValueError(
                        f"Network dimension '{dim_label}' 2D size may include at most one 'auto' entry."
                    )
                auto_idx = idx
                continue
            if normalized in axis_sizes:
                resolved[idx] = int(axis_sizes[normalized])
                continue
            try:
                resolved[idx] = int(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Network dimension '{dim_label}' 2D size entries must be integers, "
                    "parallelism names, or 'auto'."
                ) from exc
        else:
            try:
                resolved[idx] = int(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Network dimension '{dim_label}' 2D size entries must be integers, "
                    "parallelism names, or 'auto'."
                ) from exc
        if resolved[idx] is not None and resolved[idx] < 1:
            raise ValueError(
                f"Network dimension '{dim_label}' 2D size entries must be >= 1."
            )

    known_product = 1
    for value in resolved:
        if value is not None:
            known_product *= value

    if auto_idx is not None:
        if known_product <= 0:
            raise ValueError(
                f"Network dimension '{dim_label}' 2D size known entries must be > 0."
            )
        if product_size % known_product != 0:
            raise ValueError(
                f"Network dimension '{dim_label}' 2D size mismatch: "
                f"product {product_size} is not divisible by known entries {tuple(dims_value)}."
            )
        auto_value = product_size // known_product
        if auto_value < 1:
            raise ValueError(
                f"Network dimension '{dim_label}' 2D size auto-resolved entry must be >= 1."
            )
        resolved[auto_idx] = auto_value

    if resolved[0] is None or resolved[1] is None:
        raise ValueError(
            f"Network dimension '{dim_label}' 2D size could not be fully resolved."
        )

    return int(resolved[0]), int(resolved[1])


def _suggest_divisors(value: int) -> List[int]:
    if value < 1:
        return []
    divisors: List[int] = []
    root = int(math.isqrt(value))
    for candidate in range(1, root + 1):
        if value % candidate != 0:
            continue
        divisors.append(candidate)
        other = value // candidate
        if other != candidate:
            divisors.append(other)
    return sorted(divisors)


def _expand_network_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand multi-dimensional mesh/torus entries into per-dimension entries."""

    expanded: List[Dict[str, Any]] = []
    for entry in entries:
        topo = entry.get("topology")
        npus_value = entry.get("npus")
        dims_value = entry.get("dims")
        bw_value = entry.get("bandwidth")
        lat_value = entry.get("latency")

        if isinstance(topo, str) and topo.lower().startswith("kingmesh"):
            if dims_value is None:
                raise ValueError("KingMesh2D requires an explicit 2D size (e.g., size: [8, auto]).")
            if not isinstance(dims_value, (list, tuple)) or len(dims_value) != 2:
                raise ValueError("KingMesh2D size must be a two-item tuple/list.")
            dim_x = int(dims_value[0])
            dim_y = int(dims_value[1])
            if dim_x < 1 or dim_y < 1:
                raise ValueError("KingMesh2D size entries must be >= 1.")
            expanded.append(
                {
                    **entry,
                    "npus": (dim_x, dim_y),
                }
            )
            continue

        if isinstance(topo, str) and topo.lower() == "superpod":
            dim = entry.get("dim")
            dim_label = getattr(dim, "label", getattr(dim, "id", "<unnamed>"))
            variant_raw = getattr(dim, "superpod_variant", None)
            variant = str(variant_raw).strip().lower() if variant_raw is not None else ""
            if variant != "h100":
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' superpod_variant must be 'h100' (got {variant_raw!r})"
                )

            leaf_size = getattr(dim, "superpod_leaf_size", None)
            leaf_switches_per_su = getattr(dim, "superpod_leaf_switches_per_su", None)
            spine_switches_per_su = getattr(dim, "superpod_spine_switches_per_su", None)
            if leaf_size is None or leaf_switches_per_su is None or spine_switches_per_su is None:
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' is missing leaf/spine configuration values"
                )
            leaf_size = int(leaf_size)
            leaf_switches_per_su = int(leaf_switches_per_su)
            spine_switches_per_su = int(spine_switches_per_su)
            if leaf_size < 1 or leaf_switches_per_su < 1 or spine_switches_per_su < 1:
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' leaf/spine configuration must be > 0"
                )

            try:
                total_boxes = int(npus_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' requires an integer size (got {npus_value!r})"
                ) from exc
            if total_boxes < 1:
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' size must be >= 1 (got {total_boxes})"
                )
            if total_boxes % leaf_size != 0:
                divisors = _suggest_divisors(total_boxes)
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' size mismatch: N_boxes={total_boxes} "
                    f"is not divisible by leaf_size={leaf_size}. Suggested leaf_size divisors: {divisors}"
                )

            num_su = total_boxes // leaf_size
            if num_su > 16:
                print(
                    "H100 RA introduces a core tier at 16 SU. "
                    "This model uses only leaf+spine and may be optimistic."
                )

            if not isinstance(bw_value, (list, tuple)) or len(bw_value) != 2:
                raise ValueError(
                    f"SuperPOD dimension '{dim_label}' bandwidth must be a two-entry list/tuple"
                )
            bw_leaf, bw_spine = bw_value
            total_spines = num_su * spine_switches_per_su
            eff_bw_leaf = float(bw_leaf) * float(leaf_switches_per_su)
            eff_bw_spine = float(bw_spine) * float(total_spines)

            expanded.append(
                {
                    **entry,
                    "topology": "Switch",
                    "npus": leaf_size,
                    "bandwidth": eff_bw_leaf,
                    "latency": lat_value,
                }
            )
            expanded.append(
                {
                    **entry,
                    "topology": "Switch",
                    "npus": num_su,
                    "bandwidth": eff_bw_spine,
                    "latency": lat_value,
                }
            )
            continue

        if isinstance(topo, str) and topo.lower() == "fcring2d":
            if dims_value is None:
                raise ValueError("FC-Ring2D requires an explicit 2D size (e.g., size: [8, auto]).")
            if not isinstance(dims_value, (list, tuple)) or len(dims_value) != 2:
                raise ValueError("FC-Ring2D size must be a two-item tuple/list.")
            bw_inner = bw_value
            bw_outer = bw_value
            if isinstance(bw_value, (list, tuple)):
                if len(bw_value) != 2:
                    raise ValueError("FC-Ring2D bandwidth tuple must have exactly two entries.")
                bw_inner, bw_outer = bw_value
            inner = int(dims_value[0])
            outer = int(dims_value[1])
            if inner < 1 or outer < 1:
                raise ValueError("FC-Ring2D size entries must be >= 1.")
            base = dict(entry)
            base.pop("dims", None)
            if inner > 1:
                expanded.append(
                    {
                        **base,
                        "topology": "FullyConnected",
                        "npus": inner,
                        "bandwidth": bw_inner,
                        "latency": lat_value,
                    }
                )
            if outer > 1:
                ring_topo = "FullyConnected" if outer <= 2 else "Ring"
                expanded.append(
                    {
                        **base,
                        "topology": ring_topo,
                        "npus": outer,
                        "bandwidth": bw_outer,
                        "latency": lat_value,
                    }
                )
            if inner <= 1 and outer <= 1:
                raise ValueError("FC-Ring2D requires at least one dimension with size > 1.")
            continue

        if isinstance(topo, str) and (
            topo.startswith("Torus")
            or topo.startswith("Mesh")
            or topo.startswith("KingMesh")
        ):
            dim_count: int
            if dims_value is not None and isinstance(dims_value, (list, tuple)):
                dim_count = len(dims_value)
            elif isinstance(npus_value, (list, tuple)):
                dim_count = len(npus_value)
            elif len(topo) >= 2 and topo[-2].isdigit():
                dim_count = int(topo[-2])
            else:
                dim_count = 1
            base_name = "Ring" if topo.startswith("Torus") else "Mesh"
            if dims_value is None and isinstance(npus_value, (list, tuple)) and len(npus_value) == dim_count:
                dims_value = tuple(npus_value)
            if isinstance(bw_value, (list, tuple)) and len(bw_value) != dim_count:
                raise ValueError(f"bandwidth entries for topology {topo} must match dimension count {dim_count}.")
            for dim_idx in range(dim_count):
                if dims_value is not None:
                    if not isinstance(dims_value, (list, tuple)) or len(dims_value) != dim_count:
                        raise ValueError(f"dims for topology {topo} must match dimension count {dim_count}.")
                    curr_npu = dims_value[dim_idx]
                else:
                    curr_npu = (
                        npus_value[dim_idx]
                        if isinstance(npus_value, (list, tuple))
                        else npus_value
                    )
                    if dim_count == 2:
                        root = int(math.isqrt(int(curr_npu)))
                        if root * root != int(curr_npu):
                            raise ValueError(f"npus ({curr_npu}) must be a perfect square for 2D topology {topo}.")
                        curr_npu = root
                curr_bw = bw_value[dim_idx] if isinstance(bw_value, (list, tuple)) else bw_value
                topology_name = "Mesh" if dim_count == 2 and curr_npu <= 2 else base_name
                expanded.append(
                    {
                        **entry,
                        "topology": topology_name,
                        "npus": curr_npu,
                        "bandwidth": curr_bw,
                        "latency": lat_value,
                    }
                )
        else:
            expanded.append(dict(entry))
    return expanded


def generate_astrasim_configs_from_hw(
    hw_obj,
    out_dir: str = "./astra_cache",
    npus_count: Optional[int] = None,
    *,
    axes_filter: Optional[Sequence[str]] = None,
    transform_2d_to_1d: bool = False,
    faulty_links_override: Optional[Sequence[Tuple[int, int, float]]] = None,
    ephemeral_outputs: bool = False,
    preferred_axes_for_synthetic: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Write AstraSim network/system configs derived from ``hw_obj``."""
    if npus_count is None:
        raise ValueError("npus_count must be provided explicitly when generating AstraSim configs.")

    layout = getattr(hw_obj, "network_layout", None)
    dimensions = list(getattr(layout, "dimensions", [])) if layout else []
    layout_faults: Tuple[Tuple[int, int, float], ...]
    if faulty_links_override is not None:
        layout_faults = tuple(
            (int(src), int(dst), float(weight))
            for src, dst, weight in faulty_links_override
        )
    else:
        layout_faults = tuple(getattr(layout, "faulty_links", ()) or ()) if layout else tuple()
    if not dimensions:
        raise ValueError("Hardware config is missing network dimensions required for AstraSim integration.")

    exec_backend = getattr(hw_obj, "execution_backend", None)
    astra_cfg = getattr(exec_backend, "astra", None) if exec_backend else None
    astra_mode = str(getattr(astra_cfg, "mode", "") or "").strip().lower()

    sch_config = getattr(hw_obj, "sch_config", None)

    axes_filter_original = tuple(axes_filter) if axes_filter else None
    axes_filter_normalized: Optional[Tuple[str, ...]] = None
    if axes_filter_original:
        axes_filter_normalized = tuple(str(axis).strip().lower() for axis in axes_filter_original)

    preferred_axes_synth: Tuple[str, ...] = tuple(
        str(axis).strip().lower()
        for axis in preferred_axes_for_synthetic or ()
        if str(axis).strip()
    )

    active = getattr(hw_obj, "active_parallelism", None)
    if isinstance(active, dict) and active:
        axis_sizes_full = {
            "tp": int(active.get("tp", 1) or 1),
            "cp": int(active.get("cp", 1) or 1),
            "ep": int(active.get("ep", 1) or 1),
            "pp": int(active.get("pp", 1) or 1),
            "dp": int(active.get("dp", 1) or 1),
        }
    else:
        axis_sizes_full = {
            "tp": int(sch_config.tp),
            "cp": int(sch_config.cp),
            "ep": int(sch_config.train.ep),
            "pp": int(sch_config.pp),
            "dp": int(sch_config.train.dp),
        }
    synthetic_only = axes_filter_normalized is not None and set(axes_filter_normalized) == {"synthetic2"}
    if synthetic_only:
        preferred_set = set(preferred_axes_synth)
        base_dim = None
        if dimensions:
            for dim in dimensions:
                dim_axes = [
                    str(axis).strip().lower()
                    for axis in getattr(dim, "parallelisms", ()) or ()
                    if str(axis).strip()
                ]
                if preferred_set and any(axis in preferred_set for axis in dim_axes):
                    base_dim = dim
                    break
                if base_dim is None and dim_axes:
                    base_dim = dim
        if base_dim is None:
            base_dim = dimensions[0] if dimensions else None
        if base_dim is None:
            raise ValueError(f"Synthetic dimension requested but no dimensions found in hardware config. Info: Called with filter {axes_filter_original}, npus_count={npus_count}, axes_sizes={axis_sizes_full}.")
        base_bw_values = _effective_bw_values(
            getattr(base_dim, "bandwidth", None),
            float(getattr(base_dim, "util", 1.0)),
        )
        if isinstance(base_bw_values, (list, tuple)):
            base_bw = float(base_bw_values[0]) if base_bw_values else 0.0
        else:
            base_bw = float(base_bw_values)
        base_latency = float(getattr(base_dim, "latency", None))
        base_topology = getattr(base_dim, "topology_type", None)
        normalized_topology = str(base_topology or "").strip().lower().replace("-", "").replace("_", "")
        if normalized_topology in {"superpod", "mesh2d", "torus2d", "kingmesh2d", "fcring2d"}:
            synthetic_topology = "Switch"
        else:
            synthetic_topology = base_topology or "Switch"
        base_collectives = getattr(base_dim, "collective_override", {}) or {}
        synthetic_dim = SimpleNamespace(
            label="synthetic2",
            parallelisms=("synthetic2",),
            size=2,
            topology_type=synthetic_topology,
            effective_bandwidth=base_bw,
            bandwidth=base_bw,
            latency=base_latency,
            collective_override=base_collectives,
            superpod_variant=getattr(base_dim, "superpod_variant", None),
            superpod_leaf_size=getattr(base_dim, "superpod_leaf_size", None),
            superpod_leaf_switches_per_su=getattr(base_dim, "superpod_leaf_switches_per_su", None),
            superpod_spine_switches_per_su=getattr(base_dim, "superpod_spine_switches_per_su", None),
            faulty_links=(),
        )
        dimensions = [synthetic_dim]
        axis_sizes_full = {"synthetic2": 2}
        axis_sizes = {"synthetic2": 2}
        axis_order_preference = ["synthetic2"]
    else:
        axis_sizes: Dict[str, int] = dict(axis_sizes_full)
        if axes_filter_normalized:
            axis_sizes = {axis: axis_sizes_full.get(axis, 1) for axis in axes_filter_normalized}
            axis_sizes.setdefault("tp", 1)
            axis_sizes.setdefault("cp", 1)
            axis_sizes.setdefault("ep", 1)
            axis_sizes.setdefault("pp", 1)
            axis_sizes.setdefault("dp", 1)
        axis_order_preference = ["tp", "cp", "ep", "pp", "dp"]

    allowed_axes = set(axis_sizes.keys()) if axes_filter_normalized else None
    # print(f"Allowed axes: {allowed_axes}")
    # print(f"Axes sizes: {axis_sizes}")
    dim_infos: List[Tuple[Any, List[str], int, int]] = []
    # print(f"Filter: {axes_filter}")
    for dim_idx, dim in enumerate(dimensions):
        axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
        # print(f"All axes for dimension {dim.label}: {axes}")
        if allowed_axes is not None and axes_filter_normalized:
            filtered_axes = [axis for axis in axes if axis in allowed_axes]
            axes = filtered_axes
        # print(f"Filtered axes for dimension {dim.label}: {axes}")
        effective = 1
        for axis in axes:
            if axis not in axis_sizes:
                raise ValueError(
                    f"Unsupported parallelism axis '{axis}' referenced by network dimension '{dim.label}'. "
                    "Supported axes are tp, cp, ep, pp, dp."
                )
            effective *= axis_sizes[axis]
        dim_infos.append((dim, axes, effective, dim_idx))

    active_dim_indices = [
        dim_idx
        for dim, _, _, dim_idx in dim_infos
        if int(getattr(dim, "size", 0) or 0) > 1
    ]
    if active_dim_indices:
        last_active_idx = active_dim_indices[-1]
        dp_in_active = False
        if axis_sizes.get("dp", 1) > 1:
            for dim, axes, _, dim_idx in dim_infos:
                if "dp" in axes:
                    if dim_idx != last_active_idx:
                        raise ValueError(
                            f"Data-parallel axis must be assigned to the last active network dimension. "
                            f"Dimension '{dim.label}' (index {dim_idx}) includes 'dp', but last active dimension index is {last_active_idx}."
                        )
                    dp_in_active = True
        if axis_sizes.get("dp", 1) > 1 and not dp_in_active:
            raise ValueError(
                "Hardware config declares dp > 1, but no network dimension with size > 1 includes the 'dp' axis."
            )
    else:
        if layout_faults:
            raise ValueError(
                "Faulty links require at least one active (size > 1) network dimension."
            )
    target = int(npus_count)
    
    axes_needed: List[str] = []
    remaining = target

    for axis in axis_order_preference:
        size = axis_sizes.get(axis, 1)
        
        if size <= 1:
            continue
        
        if remaining % size == 0:
            axes_needed.append(axis)
            remaining //= size
    
    if remaining != 1:
        raise ValueError(
            f"Unable to map requested npus_count={target} to network axes {axis_sizes}. Info: Called with filter {axes_filter_original}, npus_count={npus_count}, axes_sizes={axis_sizes}."
        )

    axes_needed_set = set(axes_needed)
    selected_dims: List[Tuple[Any, List[str], int, int]] = []
    accumulated = 1
    for dim, axes, _effective, dim_idx in dim_infos:
        selected_axes: List[str] = []
        size_contrib = 1
        for axis in axes:
            if axis in axes_needed_set:
                selected_axes.append(axis)
                size_contrib *= axis_sizes[axis]
        if not selected_axes:
            continue
        axes_needed_set.difference_update(selected_axes)
        accumulated *= size_contrib
        selected_dims.append((dim, selected_axes, size_contrib, dim_idx))
    if axes_needed_set:
        raise ValueError(
            f"Requested npus_count={target} requires axes {axes_needed} but no network dimension covers {axes_needed_set}."
        )
    if accumulated != target:
        raise ValueError(
            f"Selected network dimensions {[d.label for d, _, _, _ in selected_dims]} "
            f"describe {accumulated} NPUs but simulation expects {target} ranks."
        )

    topo_list: List[str] = []
    npus_list: List[Any] = []
    bw_list: List[float] = []
    lat_list: List[float] = []
    network_entries: List[Dict[str, Any]] = []

    for dim, axes_selected, product_size, _ in selected_dims:
        topo = _normalize_topology_name(dim.topology_type)
        full_size = int(getattr(dim, "size", product_size) or product_size)
        dims_value = getattr(dim, "size_2d", None)
        dims_tuple: Optional[Tuple[int, int]] = None
        use_shape = dims_value is not None and (product_size == full_size or topo == "KingMesh2D")
        if topo == "KingMesh2D" and dims_value is None:
            raise ValueError(
                f"KingMesh2D requires an explicit 2D size for network dimension '{dim.label}'."
            )
        if use_shape:
            dims_tuple = _resolve_2d_dims(
                dims_value,
                axis_sizes,
                dim_label=getattr(dim, "label", getattr(dim, "id", "<unnamed>")),
                product_size=product_size,
            )
            shape_product = dims_tuple[0] * dims_tuple[1]
            if shape_product != product_size:
                raise ValueError(
                    f"Network dimension '{dim.label}' shape {dims_tuple} product {shape_product} "
                    f"does not match parallelism product {product_size}."
                )
            size = shape_product
        else:
            size = product_size
        if transform_2d_to_1d and topo in ["Torus2D", "Mesh2D", "KingMesh2D"]:
            if topo == "Torus2D":
                topo = "Ring"
            if topo == "Mesh2D":
                topo = "Mesh"
            if topo == "KingMesh2D":
                topo = "HyperCube"
            # When flattening 2D, prefer the explicit shape product if available.
            size = size
        if topo == "Ring" and size <= 2:
            topo = "FullyConnected"
        effective_bw = _effective_bw_values(
            getattr(dim, "bandwidth", None),
            float(getattr(dim, "util", 1.0)),
        )
        if topo == "KingMesh2D" and isinstance(effective_bw, (list, tuple)):
            raise ValueError(
                f"KingMesh2D requires a scalar bandwidth for network dimension '{dim.label}'."
            )
        if transform_2d_to_1d and isinstance(effective_bw, (list, tuple)):
            effective_bw = effective_bw[0] if effective_bw else 0.0
        latency_s = float(dim.latency)
        entry = {
            "dim": dim,
            "axes": axes_selected,
            "npus": size,
            "topology": topo,
            "bandwidth": _as_gbps(effective_bw),
            "latency": round(_ns_from_s(latency_s), 3),
        }
        if dims_tuple is not None and topo in ["Torus2D", "Mesh2D", "KingMesh2D", "FCRing2D"]:
            entry["dims"] = dims_tuple
        network_entries.append(entry)

    network_entries = _expand_network_entries(network_entries)

    topo_list = [entry["topology"] for entry in network_entries]
    npus_list = [entry["npus"] for entry in network_entries]
    bw_list = [entry["bandwidth"] for entry in network_entries]
    lat_list = [entry["latency"] for entry in network_entries]
    non_recursive_from: Optional[int] = None
    if topo_list:
        first_topo = None
        if selected_dims:
            first_dim = selected_dims[0][0]
            first_topo = _normalize_topology_name(first_dim.topology_type)
        if first_topo and _is_2d_topology(first_topo) and not transform_2d_to_1d:
            if first_topo not in {"KingMesh2D"}:
                non_recursive_from = 2 # TORUS2D, MESH2D - ASSUME RECURSIVE INTERNAL
            else:
                non_recursive_from = 1 # KINGMESH
        elif first_topo and first_topo == "SuperPOD":
            non_recursive_from = 2 # SUPERPOD
        else:
            non_recursive_from = 0 # NO RECURSION EVER FOR OTHER CASES.
        if non_recursive_from > len(topo_list):
            raise ValueError(
                f"non_recursive_from={non_recursive_from} exceeds network dimensions ({len(topo_list)})."
            )

    if any(isinstance(entry, (list, tuple)) for entry in npus_list):
        signature_parts = []
        for entry in npus_list:
            if isinstance(entry, (list, tuple)):
                signature_parts.append("x".join(str(int(v)) for v in entry))
            else:
                signature_parts.append(str(entry))
    else:
        signature_parts = [str(size) for _dim, _axes, size, _ in selected_dims]
    dim_signature = "_".join(signature_parts) if signature_parts else f"{target}"

    unique_suffix = f"_{uuid.uuid4().hex}" if ephemeral_outputs else ""
    net_yaml = os.path.join(out_dir, f"network_analytical_{dim_signature}{unique_suffix}.yml")
    sys_json = os.path.join(out_dir, f"system_native_collectives_{dim_signature}{unique_suffix}.json")

    topo_str = ", ".join(topo_list)
    def _format_npus_entry(value: object) -> str:
        if isinstance(value, (list, tuple)):
            values = ", ".join(str(int(v)) for v in value)
            return f"[ {values} ]"
        return str(value)
    npus_str = ", ".join(_format_npus_entry(v) for v in npus_list)
    bw_str = ", ".join(str(v) for v in bw_list)
    lat_str = ", ".join(str(v) for v in lat_list)

    net_content = (
        f"topology: [ {topo_str} ]\n"
        f"npus_count: [ {npus_str} ]\n"
        f"bandwidth: [ {bw_str} ]  # GB/s\n"
        f"latency: [ {lat_str} ]   # ns\n"
    )
    if non_recursive_from is not None:
        net_content += f"non_recursive_from: {non_recursive_from}\n"

    faulty_links_tuple: Tuple[Tuple[int, int, float], ...] = tuple(layout_faults)
    if faulty_links_tuple:
        fault_entries = [
            f"[{src}, {dst}, {float(weight):g}]"
            for src, dst, weight in faulty_links_tuple
        ]
        faulty_links_str = f"[{', '.join(fault_entries)}]"
        net_content += f"faulty_links: {faulty_links_str}\n"

    os.makedirs(os.path.dirname(net_yaml), exist_ok=True)
    tmp_path = net_yaml + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(net_content)
    os.replace(tmp_path, net_yaml)

    if astra_cfg:
        coll = astra_cfg.collectives
        sys_opts = getattr(astra_cfg, "sys_options", None)
        default_ag = coll.all_gather
        default_ar = coll.all_reduce
        default_rs = coll.reduce_scatter
        default_a2a = coll.all_to_all
    else:
        default_ag = "auto"
        default_ar = "auto"
        default_rs = "auto"
        default_a2a = "auto"
        sys_opts = None

    def _collective_for_dimension(
        dim,
        topo_name: str,
        op: str,
        default_alg: str,
        npus_value: object,
    ) -> str:
        override = (
            dim.collective_override.get(op)
            or dim.collective_override.get(op.replace("-", "_"))
            or dim.collective_override.get(op.replace("_", "-"))
        )
        if override:
            return override
        if default_alg == "auto":
            raw_topo = _normalize_topology_name(getattr(dim, "topology_type", topo_name))
            if raw_topo == "KingMesh2D" and topo_name != "HyperCube":
                return "kingmesh"
        alg = choose_collective(default_alg, topo_name, op)
        if default_alg == "auto" and topo_name == "Switch" and alg == "halvingDoubling":
            try:
                count = int(npus_value)
            except (TypeError, ValueError):
                count = 0
            if count and not _is_power_of_two(count):
                return "ring"
        return alg

    ag_impl: List[str] = []
    ar_impl: List[str] = []
    rs_impl: List[str] = []
    a2a_impl: List[str] = []

    for entry in network_entries:
        dim = entry["dim"]
        topo_name = entry["topology"]
        npus_value = entry.get("npus")
        ag_impl.append(_collective_for_dimension(dim, topo_name, "all-gather", default_ag, npus_value))
        ar_impl.append(_collective_for_dimension(dim, topo_name, "all-reduce", default_ar, npus_value))
        rs_impl.append(_collective_for_dimension(dim, topo_name, "reduce-scatter", default_rs, npus_value))
        a2a_impl.append(_collective_for_dimension(dim, topo_name, "all-to-all", default_a2a, npus_value))

    system = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 4,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ar_impl,
        "all-gather-implementation": ag_impl,
        "reduce-scatter-implementation": rs_impl,
        "all-to-all-implementation": a2a_impl,
        "collective-optimization": "localBWAware",
        "local-mem-bw": 1600,
        "boost-mode": 0,
        "roofline-enabled": 0,
        "peak-perf": 900,
    }
    if sys_opts is not None:
        if getattr(sys_opts, "endpoint_delay", None) is not None:
            system["endpoint-delay"] = sys_opts.endpoint_delay
        if getattr(sys_opts, "active_chunks_per_dimension", None) is not None:
            acpd = sys_opts.active_chunks_per_dimension
            if isinstance(acpd, (list, tuple)):
                acpd = acpd[0] if acpd else 1
            system["active-chunks-per-dimension"] = int(acpd)
        if getattr(sys_opts, "preferred_dataset_splits", None) is not None:
            pds = sys_opts.preferred_dataset_splits
            if isinstance(pds, (list, tuple)):
                pds = pds[0] if pds else 1
            system["preferred-dataset-splits"] = int(pds)
        if getattr(sys_opts, "collective_arbitration", None) is not None:
            system["collective-arbitration"] = sys_opts.collective_arbitration

    _save_json(sys_json, system)

    return {
        "network_yaml": net_yaml,
        "system_json": sys_json,
        "topology_list": topo_list,
        "npus_per_dim": npus_list,
    }


__all__ = [
    "ASTRA_DEBUG",
    "choose_collective",
    "compute_intra_inter_ib_ll_from_hw",
    "derive_topology_from_hw",
    "generate_astrasim_configs_from_hw",
]
