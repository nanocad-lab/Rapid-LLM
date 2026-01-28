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

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import yaml as _yaml
from yaml import YAMLError as _YAMLError


_PRECISION_DTYPE_BYTES = {
    "fp8": 1.0,
    "fp16": 2.0,
    "half": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
    "single": 4.0,
}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


_MOE_PADDING_WARNED = False


@dataclass(frozen=True)
class PrecisionConfig:
    tensor: float
    kv_cache: float
    parameters: float
    gradients: float
    grad_communication: float
    optimizer_states: float
    stats: float
    master_parameters: float

    @property
    def activations(self) -> float:
        return self.tensor

    @property
    def tensor_format(self) -> float:
        return self.tensor

    @property
    def requires_master_copy(self) -> bool:
        return self.master_parameters > 0.0

def _coerce_precision_value(value, *, tensor_bytes: Optional[float] = None, allow_as_tensor: bool = False) -> float:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("precision byte size must be positive")
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported precision specification type: {type(value)!r}")

    normalized = value.strip().lower()
    if allow_as_tensor and normalized == "as_tensor_format":
        return float(tensor_bytes)

    if normalized in _PRECISION_DTYPE_BYTES:
        return _PRECISION_DTYPE_BYTES[normalized]

    parsed = float(normalized)
    if parsed <= 0:
        raise ValueError("precision byte size must be positive")
    return parsed


def _parse_precision_block(spec: dict) -> PrecisionConfig:
    """Parse precision configuration by directly reading each precision type.

    Each field can be:
    - A numeric byte count (e.g., 2.0, 4.0)
    - A dtype string (e.g., "fp16", "bf16", "fp32")
    - "as_tensor_format" (only for kv_cache, parameters, gradients, grad_communication)
    """
    tensor_bytes = _coerce_precision_value(spec["tensor_format"])

    # Parse each precision field directly
    kv_cache_bytes = _coerce_precision_value(
        spec["kv_cache"],
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    parameter_bytes = _coerce_precision_value(
        spec.get("parameters", "as_tensor_format"),
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    gradient_bytes = _coerce_precision_value(
        spec.get("gradients", "fp32"),  # Default to FP32 for accumulated gradients
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    grad_comm_bytes = _coerce_precision_value(
        spec.get("grad_communication", "as_tensor_format"),
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    optimizer_bytes = _coerce_precision_value(
        spec.get("optimizer_states", "fp32"),  # Default to FP32 for optimizer states
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    stats_bytes = _coerce_precision_value(
        spec.get("stats", "fp32"),  # Default to FP32 for stats
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    # master_parameters can be 0 (no master copy) or a positive value
    master_raw = spec.get("master_parameters", 0.0)
    if isinstance(master_raw, (int, float)) and master_raw == 0.0:
        master_bytes = 0.0
    else:
        master_bytes = _coerce_precision_value(
            master_raw,
            tensor_bytes=tensor_bytes,
            allow_as_tensor=False,
        )

    return PrecisionConfig(
        tensor=tensor_bytes,
        kv_cache=kv_cache_bytes,
        parameters=parameter_bytes,
        gradients=gradient_bytes,
        grad_communication=grad_comm_bytes,
        optimizer_states=optimizer_bytes,
        stats=stats_bytes,
        master_parameters=master_bytes,
    )


@dataclass
class CoreConfig:
    nominal_power_per_mcu: float
    nominal_flop_rate_per_mcu: float
    nominal_energy_per_flop: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    operating_area_per_mcu: float
    num_mcu_per_bundle: int
    FMA_dims: tuple
    dataflow: str
    util: float
    num_bundles: int = None
    operating_frequency: float = None
    nominal_frequency: float = None
    nominal_area_per_mcu: float = None

    @classmethod
    def from_dict(cls, core_config_dict):
        return cls(
            nominal_power_per_mcu=core_config_dict.get("nominal_power_per_mcu", 0.1),
            nominal_flop_rate_per_mcu=core_config_dict["nominal_flop_rate_per_mcu"],
            nominal_energy_per_flop=core_config_dict["nominal_energy_per_flop"],
            nominal_voltage=core_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=core_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=core_config_dict.get("margin_voltage", 0.1),
            operating_area_per_mcu=core_config_dict.get("operating_area_per_mcu", 0.1),
            num_mcu_per_bundle=core_config_dict["num_mcu_per_bundle"],
            FMA_dims=(core_config_dict["FMA_d1"], core_config_dict["FMA_d2"]),
            dataflow=core_config_dict["dataflow"],
            util=core_config_dict["util"],
            num_bundles=core_config_dict.get("num_bundles", None),
            operating_frequency=core_config_dict.get("operating_frequency", None),
            nominal_frequency=core_config_dict.get("nominal_frequency", None),
            nominal_area_per_mcu=core_config_dict.get("nominal_area_per_mcu", None),
        )


@dataclass
class DRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    stack_capacity: float
    area_per_stack: float
    latency: float
    mem_ctrl_area: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    num_links_per_mm: int
    num_links_per_stack: int
    max_voltage: float
    util: float
    size: float = None
    bandwidth: float = None
    num_stacks: int = None
    operating_frequency: float = None
    nominal_frequency: float = None

    @classmethod
    def from_dict(cls, dram_config_dict):
        return cls(
            dynamic_energy_per_bit=dram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=dram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=dram_config_dict.get("area_per_bit", 0.1),
            stack_capacity=dram_config_dict.get("stack_capacity", 0.1),
            area_per_stack=dram_config_dict.get("area_per_stack", 0.1),
            latency=dram_config_dict["latency"],
            mem_ctrl_area=dram_config_dict.get("mem_ctrl_area", 0.1),
            nominal_voltage=dram_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=dram_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=dram_config_dict.get("margin_voltage", 0.1),
            num_links_per_mm=dram_config_dict.get("num_links_per_mm", 1),
            num_links_per_stack=dram_config_dict.get("num_links_per_stack", 1),
            max_voltage=dram_config_dict.get("max_voltage", 0.1),
            util=dram_config_dict["util"],
            size=dram_config_dict.get("size", None),
            bandwidth=dram_config_dict.get("bandwidth", None),
            num_stacks=dram_config_dict.get("num_stacks", None),
            operating_frequency=dram_config_dict.get("operating_frequency", None),
            nominal_frequency=dram_config_dict.get("nominal_frequency", None),
        )


@dataclass
class SRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    bank_capacity: float
    controller_area_per_link: float
    latency: float
    overhead: float
    util: float
    size: float = None
    bandwidth: float = None

    @classmethod
    def from_dict(cls, sram_config_dict):
        return cls(
            dynamic_energy_per_bit=sram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=sram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=sram_config_dict.get("area_per_bit", 0.1),
            bank_capacity=sram_config_dict.get("bank_capacity", 0.1),
            controller_area_per_link=sram_config_dict.get("controller_area_per_link", 0.1),
            latency=sram_config_dict["latency"],
            overhead=sram_config_dict.get("overhead", 0.1),
            util=sram_config_dict["util"],
            size=sram_config_dict.get("size", None),
            bandwidth=sram_config_dict.get("bandwidth", None),
        )


@dataclass
class TechConfig:
    core: CoreConfig
    DRAM: DRAMConfig
    SRAML2: SRAMConfig
    SRAML1: SRAMConfig
    SRAMR: SRAMConfig

    @classmethod
    def from_dict(cls, tech_config_dict):
        return cls(
            core=CoreConfig.from_dict(tech_config_dict["core"]),
            DRAM=DRAMConfig.from_dict(tech_config_dict["DRAM"]),
            SRAML2=SRAMConfig.from_dict(tech_config_dict["SRAM-L2"]),
            SRAML1=SRAMConfig.from_dict(tech_config_dict["SRAM-L1"]),
            SRAMR=SRAMConfig.from_dict(tech_config_dict["SRAM-R"]),
        )


@dataclass
class AreaBreakdownConfig:
    proc_chip_area_budget: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    node_area_budget: float
    network: "NetworkAreaConfig"

    @classmethod
    def from_dict(cls, area_config_dict):
        return cls(
            proc_chip_area_budget=area_config_dict["proc_chip_area_budget"],
            core=area_config_dict["core"],
            DRAM=area_config_dict["DRAM"],
            L2=area_config_dict["L2"],
            L1=area_config_dict["L1"],
            reg_mem=area_config_dict["reg_mem"],
            node_area_budget=area_config_dict["device_area_budget"],
            network=NetworkAreaConfig.from_dict(area_config_dict["network"]),
        )


@dataclass
class PerimeterBreakdownConfig:
    DRAM: float
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, perimeter_config_dict):
        return cls(
            DRAM=perimeter_config_dict["DRAM"],
            inter_node=perimeter_config_dict["inter_node"],
            intra_node=perimeter_config_dict["intra_node"],
        )


@dataclass
class NetworkAreaConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_config_dict):
        return cls(
            inter_node=network_config_dict["inter_node"],
            intra_node=network_config_dict["intra_node"],
        )


@dataclass
class PowerBreakdownConfig:
    TDP: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    network: "NetworkPowerConfig"

    @classmethod
    def from_dict(cls, power_config_dict):
        return cls(
            TDP=power_config_dict["TDP"],
            core=power_config_dict["core"],
            DRAM=power_config_dict["DRAM"],
            L2=power_config_dict["L2"],
            L1=power_config_dict["L1"],
            reg_mem=power_config_dict["reg_mem"],
            network=NetworkPowerConfig.from_dict(power_config_dict["network"]),
        )


@dataclass
class NetworkPowerConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_power_config_dict):
        return cls(
            inter_node=network_power_config_dict["inter_node"],
            intra_node=network_power_config_dict["intra_node"],
        )


@dataclass(frozen=True)
class NetworkDimensionLayout:
    id: str
    label: str
    size: int
    topology_type: str
    bandwidth: object
    util: float
    latency: float
    size_2d: Optional[Tuple[int, int]] = None
    collective_override: Dict[str, str] = field(default_factory=dict)
    parallelisms: Tuple[str, ...] = field(default_factory=tuple)
    energy_per_bit: float = 0.0
    optimize_2dmap: bool = False
    superpod_variant: Optional[str] = None
    superpod_leaf_size: Optional[int] = None
    superpod_leaf_switches_per_su: Optional[int] = None
    superpod_spine_switches_per_su: Optional[int] = None

    @classmethod
    def from_raw(
        cls,
        raw: dict,
        *,
        parallelism_params: Dict[str, object],
        index: int,
    ) -> "NetworkDimensionLayout":
        if not isinstance(raw, dict):
            raise TypeError("each network dimension must be a mapping")

        raw_id = raw.get("id")
        dim_id = str(raw_id) if raw_id is not None else f"dim{index}"
        label = str(raw.get("label", dim_id))

        if "size" not in raw:
            raise ValueError(f"network dimension '{label}' is missing required field 'size'")
        size_raw = raw["size"]
        size_mode: str
        size_value: Optional[int] = None
        size_2d: Optional[Tuple[int, int]] = None
        tuple_entries: Optional[Tuple[object, object]] = None
        if isinstance(size_raw, str):
            normalized_size = size_raw.strip()
            if normalized_size.startswith("(") and normalized_size.endswith(")"):
                tuple_entries = tuple(
                    part.strip() for part in normalized_size[1:-1].split(",")
                )  # type: ignore[assignment]
                size_mode = "tuple"
            elif normalized_size.lower() == "auto":
                size_mode = "auto_scalar"
            else:
                try:
                    size_value = int(size_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"network dimension '{label}' size must be an integer or tuple") from exc
                if size_value < 1:
                    raise ValueError(f"network dimension '{label}' size must be >= 1")
                size_mode = "scalar"
        elif isinstance(size_raw, (list, tuple)):
            if len(size_raw) != 2:
                raise ValueError(f"network dimension '{label}' 2D size must have exactly two entries")
            tuple_entries = tuple(size_raw)  # type: ignore[assignment]
            size_mode = "tuple"
        else:
            size_mode = "auto_scalar"

        topo_dict = raw.get("topology")
        if not isinstance(topo_dict, dict):
            raise ValueError(f"network dimension '{label}' requires a 'topology' mapping")
        if "type" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'type'")
        topo_type = str(topo_dict["type"])
        normalized_topo = topo_type.lower().replace("-", "").replace("_", "")
        is_2d_topo = normalized_topo in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}
        is_superpod = normalized_topo == "superpod"

        superpod_variant: Optional[str] = None
        superpod_leaf_size: Optional[int] = None
        superpod_leaf_switches_per_su: Optional[int] = None
        superpod_spine_switches_per_su: Optional[int] = None

        if is_superpod:
            raw_variant = topo_dict.get("superpod_variant")
            if raw_variant is None:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD topology requires 'superpod_variant' set to 'h100'"
                )
            superpod_variant = str(raw_variant).strip().lower()
            if superpod_variant != "h100":
                raise ValueError(
                    f"network dimension '{label}' SuperPOD superpod_variant must be 'h100' (got {raw_variant!r})"
                )

            def _parse_superpod_int(field: str, default: int) -> int:
                raw_value = topo_dict.get(field, default)
                if raw_value is None:
                    return default
                try:
                    parsed = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"network dimension '{label}' SuperPOD {field} must be an integer"
                    ) from exc
                if parsed < 1:
                    raise ValueError(
                        f"network dimension '{label}' SuperPOD {field} must be > 0"
                    )
                return parsed

            superpod_leaf_size = _parse_superpod_int("leaf_size", 32)
            superpod_leaf_switches_per_su = _parse_superpod_int("leaf_switches_per_su", 8)
            superpod_spine_switches_per_su = _parse_superpod_int("spine_switches_per_su", 4)

        if "bandwidth" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'bandwidth'")
        bandwidth_raw = topo_dict["bandwidth"]
        bandwidth = parse_bandwidth_string(bandwidth_raw)
        if isinstance(bandwidth, (list, tuple)):
            if not (is_2d_topo or is_superpod):
                raise ValueError(
                    f"network dimension '{label}' bandwidth tuple is only supported for 2D topologies or SuperPOD"
                )
            if len(bandwidth) != 2:
                raise ValueError(
                    f"network dimension '{label}' bandwidth tuple must have exactly two entries"
                )
            for entry in bandwidth:
                if entry is None:
                    raise ValueError(
                        f"network dimension '{label}' bandwidth tuple entries must be numeric"
                    )
                if float(entry) <= 0:
                    raise ValueError(
                        f"network dimension '{label}' bandwidth tuple entries must be > 0"
                    )
        else:
            if bandwidth is None:
                raise ValueError(
                    f"network dimension '{label}' bandwidth must be numeric"
                )
            if float(bandwidth) <= 0:
                raise ValueError(
                    f"network dimension '{label}' bandwidth must be > 0"
                )
            if is_superpod:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD bandwidth must be a two-entry list/tuple"
                )

        try:
            util = float(topo_dict.get("util", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' topology util must be numeric"
            ) from exc
        if util <= 0:
            raise ValueError(f"network dimension '{label}' topology util must be > 0")

        if "energy_per_bit" not in topo_dict:
            raise ValueError(
                f"network dimension '{label}' topology missing required 'energy_per_bit'"
            )
        try:
            energy_per_bit = float(topo_dict["energy_per_bit"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be numeric"
            ) from exc
        if energy_per_bit < 0:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be >= 0"
            )

        if "latency" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'latency'")
        try:
            latency = float(topo_dict["latency"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network dimension '{label}' latency must be numeric") from exc

        raw_optimize = topo_dict.get("optimize_2dmap", False)
        if raw_optimize not in (True, False):
            raise ValueError(
                f"network dimension '{label}' topology optimize_2dmap must be a boolean when provided"
            )
        optimize_2dmap = bool(raw_optimize)
        if optimize_2dmap:
            if normalized_topo not in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}:
                raise ValueError(
                    f"network dimension '{label}' sets optimize_2dmap but topology type '{topo_type}'"
                    " is not Mesh2D/Torus2D/KingMesh2D/FC-Ring2D"
                )

        collectives_raw = raw.get("collective_override")
        if collectives_raw is None and "collectives" in raw:
            collectives_raw = raw.get("collectives")
        if not collectives_raw:
            collectives_raw = {}
        if not isinstance(collectives_raw, dict):
            raise ValueError(f"network dimension '{label}' collective_override must be a mapping if provided")
        collective_override = {str(k): str(v) for k, v in collectives_raw.items()}

        parallelisms_raw = raw.get("parallelisms", [])
        if parallelisms_raw is None:
            parallelisms_raw = []
        if not isinstance(parallelisms_raw, Sequence) or isinstance(parallelisms_raw, (str, bytes)):
            raise ValueError(
                f"network dimension '{label}' parallelisms must be a sequence of names"
            )

        normalized_parallelisms: List[str] = []
        alias_map: Dict[str, str] = {}
        for entry in parallelisms_raw:
            name = str(entry).strip()
            if not name:
                raise ValueError(f"network dimension '{label}' has an empty parallelism name")
            normalized = name.lower()
            normalized_parallelisms.append(normalized)
            alias_map[normalized] = name

        computed_product: Optional[int] = None
        auto_product: Optional[int] = None

        if size_mode in {"auto_scalar", "tuple"}:
            auto_product = _compute_dimension_parallelism_product(
                dimension_label=label,
                normalized_names=tuple(normalized_parallelisms),
                alias_map=alias_map,
                parallelism_params=parallelism_params,
            )
            if auto_product < 1:
                raise ValueError(f"network dimension '{label}' inferred size must be >= 1")

        if size_mode == "auto_scalar":
            size_value = auto_product
            computed_product = auto_product
        elif size_mode == "tuple":
            assert tuple_entries is not None
            resolved: List[Optional[int]] = [None, None]
            auto_entries = 0
            for idx, entry in enumerate(tuple_entries):
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized == "auto":
                        auto_entries += 1
                        if auto_entries > 1:
                            raise ValueError(
                                f"network dimension '{label}' 2D size may include at most one 'auto' entry"
                            )
                        resolved[idx] = None
                        continue
                    if normalized in parallelism_params:
                        try:
                            factor = int(parallelism_params[normalized])
                        except (TypeError, ValueError) as exc:
                            alias = alias_map.get(normalized, normalized)
                            raise ValueError(
                                f"network dimension '{label}' 2D size parallelism '{alias}' must be an integer"
                            ) from exc
                        if factor < 1:
                            alias = alias_map.get(normalized, normalized)
                            raise ValueError(
                                f"network dimension '{label}' 2D size parallelism '{alias}' must be >= 1"
                            )
                        resolved[idx] = factor
                        continue
                    try:
                        factor = int(entry)
                    except (TypeError, ValueError) as exc:
                        alias = alias_map.get(normalized, normalized)
                        raise ValueError(
                            f"network dimension '{label}' 2D size entry '{alias}' must be an integer, "
                            "parallelism name, or 'auto'"
                        ) from exc
                    if factor < 1:
                        raise ValueError(
                            f"network dimension '{label}' 2D size entries must be >= 1"
                        )
                    resolved[idx] = factor
                else:
                    try:
                        factor = int(entry)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"network dimension '{label}' 2D size entries must be integers, parallelism names, or 'auto'"
                        ) from exc
                    if factor < 1:
                        raise ValueError(f"network dimension '{label}' 2D size entries must be >= 1")
                    resolved[idx] = factor

            known_product = 1
            for val in resolved:
                if val:
                    known_product *= val

            if auto_entries:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product == 0:
                    raise ValueError(f"network dimension '{label}' 2D size known entries must be > 0")
                if auto_product % known_product != 0:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: auto product {auto_product} "
                        f"is not divisible by provided factors {tuple_entries}"
                    )
                auto_value = auto_product // known_product
                if auto_value < 1:
                    raise ValueError(
                        f"network dimension '{label}' 2D size auto-resolved entry must be >= 1 (got {auto_value})"
                    )
                for idx in range(2):
                    if resolved[idx] is None:
                        resolved[idx] = auto_value
                        break
            else:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product != auto_product:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: provided shape {tuple_entries} "
                        f"product {known_product} does not match parallelism product {auto_product}"
                    )

            if resolved[0] is None or resolved[1] is None:
                raise ValueError(f"network dimension '{label}' 2D size could not be fully resolved")

            size_2d = (int(resolved[0]), int(resolved[1]))
            size_value = int(size_2d[0]) * int(size_2d[1])
            computed_product = auto_product
        else:
            computed_product = None

        if size_value is None:
            raise ValueError(f"network dimension '{label}' size could not be resolved")

        _validate_dimension_parallelisms(
            dimension_label=label,
            dimension_size=int(size_value) if size_value is not None else 0,
            normalized_names=tuple(normalized_parallelisms),
            alias_map=alias_map,
            parallelism_params=parallelism_params,
            expected_product=computed_product,
        )

        if is_superpod:
            total_boxes = int(size_value)
            leaf_size = int(superpod_leaf_size or 0)
            if leaf_size < 1:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD leaf_size must be > 0"
                )
            if total_boxes % leaf_size != 0:
                divisors: List[int] = []
                root = int(math.isqrt(total_boxes)) if total_boxes >= 1 else 0
                for candidate in range(1, root + 1):
                    if total_boxes % candidate != 0:
                        continue
                    divisors.append(candidate)
                    other = total_boxes // candidate
                    if other != candidate:
                        divisors.append(other)
                divisors = sorted(divisors)
                raise ValueError(
                    f"network dimension '{label}' SuperPOD size mismatch: N_boxes={total_boxes} "
                    f"is not divisible by leaf_size={leaf_size}. Suggested leaf_size divisors: {divisors}"
                )
            num_su = total_boxes // leaf_size
            if num_su <= 1:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD requires more than 1 SU (got {num_su})."
                )

        return cls(
            id=dim_id,
            label=label,
            size=int(size_value) if size_value is not None else 0,
            size_2d=size_2d,
            topology_type=topo_type,
            bandwidth=bandwidth,
            util=util,
            latency=latency,
            collective_override=collective_override,
            parallelisms=tuple(normalized_parallelisms),
            energy_per_bit=energy_per_bit,
            optimize_2dmap=optimize_2dmap,
            superpod_variant=superpod_variant,
            superpod_leaf_size=superpod_leaf_size,
            superpod_leaf_switches_per_su=superpod_leaf_switches_per_su,
            superpod_spine_switches_per_su=superpod_spine_switches_per_su,
        )

    @property
    def effective_bandwidth(self) -> float:
        bw = self.bandwidth
        if isinstance(bw, (list, tuple)):
            if not bw:
                return 0.0
            return float(bw[0]) * float(self.util)
        return float(bw) * float(self.util)


def _validate_dimension_parallelisms(
    *,
    dimension_label: str,
    dimension_size: int,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
    expected_product: Optional[int] = None,
) -> None:
    if not normalized_names:
        return

    product = expected_product
    if product is None:
        product = _compute_dimension_parallelism_product(
            dimension_label=dimension_label,
            normalized_names=normalized_names,
            alias_map=alias_map,
            parallelism_params=parallelism_params,
        )

    if product != dimension_size:
        readable = [alias_map.get(name, name) for name in normalized_names]
        raise ValueError(
            f"Network dimension '{dimension_label}' size mismatch: declared size {dimension_size} "
            f"but parallelism factors ({readable}) imply {product}"
        )


def _compute_dimension_parallelism_product(
    *,
    dimension_label: str,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
) -> int:
    product = 1
    for name in normalized_names:
        if name not in parallelism_params:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' references parallelism '{alias}' "
                "which is not defined in parallelism"
            )
        value = parallelism_params[name]
        if value in (None, False):
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' parallelism '{alias}' must have a "
                "positive parallelism factor"
            )
        try:
            factor = int(value)
        except (TypeError, ValueError) as exc:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be an integer to compute network dimension sizes"
            ) from exc
        if factor < 1:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be >= 1 to compute network dimension sizes"
            )
        product *= factor

    return product


def _parse_network_layout(
    network_spec,
    parallelism_params: Dict[str, object],
) -> Tuple[Tuple[NetworkDimensionLayout, ...], Tuple[Tuple[int, int, float], ...], "NetworkOverlapConfig"]:
    if network_spec is None:
        raise ValueError("network section must be specified and include overlap settings")

    if not isinstance(network_spec, dict):
        raise ValueError("network must be provided as a mapping to supply overlap settings")

    faulty_links: Tuple[Tuple[int, int, float], ...] = _parse_faulty_links("network", network_spec.get("faulty_links", []))
    overlap_config = _parse_network_overlap(network_spec.get("overlap"))
    dimensions_spec = network_spec.get("dimensions")
    if dimensions_spec is None:
        raise ValueError("network.dimensions must be specified when network is a mapping")

    if not isinstance(dimensions_spec, Sequence) or isinstance(dimensions_spec, (str, bytes)):
        raise ValueError("network.dimensions must be a sequence of dimension mappings")

    dimensions: List[NetworkDimensionLayout] = []
    for index, entry in enumerate(dimensions_spec):
        dimensions.append(
            NetworkDimensionLayout.from_raw(
                entry,
                parallelism_params=parallelism_params,
                index=index,
            )
        )
    for idx, dim in enumerate(dimensions):
        topo_name = str(getattr(dim, "topology_type", "")).strip().lower()
        topo_name = topo_name.replace("-", "").replace("_", "")
        if idx > 0 and topo_name in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}:
            raise ValueError(
                f"2D topology '{dim.topology_type}' is only supported on the first network dimension."
            )
    return tuple(dimensions), faulty_links, overlap_config


@dataclass(frozen=True)
class NetworkOverlapConfig:
    tp_overlap: float
    tp_sp_overlap: float
    cp_overlap: float


@dataclass(frozen=True)
class NetworkLayoutConfig:
    dimensions: Tuple[NetworkDimensionLayout, ...]
    faulty_links: Tuple[Tuple[int, int, float], ...] = field(default_factory=tuple)
    parallelism_map: Dict[str, NetworkDimensionLayout] = field(default_factory=dict)
    overlap_config: "NetworkOverlapConfig" = None

    def primary_dimension(self) -> Optional[NetworkDimensionLayout]:
        return self.dimensions[0] if self.dimensions else None

    def dimension_for_parallelism(self, name: str) -> Optional[NetworkDimensionLayout]:
        normalized = str(name).strip().lower()
        if normalized in self.parallelism_map:
            return self.parallelism_map[normalized]
        return self.primary_dimension()

    def link_for_parallelism(self, name: str) -> Tuple[float, float]:
        dim = self.dimension_for_parallelism(name)
        if dim is None:
            return 0.0, 0.0
        return dim.effective_bandwidth, dim.latency


def _parse_faulty_links(owner_label: str, faulty_links_raw) -> Tuple[Tuple[int, int, float], ...]:
    entries: List[Tuple[int, int, float]] = []
    if not faulty_links_raw:
        return tuple()
    if not isinstance(faulty_links_raw, Sequence) or isinstance(faulty_links_raw, (str, bytes)):
        raise ValueError(
            f"{owner_label} faulty_links must be a sequence of [src, dst, weight] entries"
        )
    for idx, entry in enumerate(faulty_links_raw):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 3:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] must be a three-item sequence [src, dst, weight]"
            )
        src_raw, dst_raw, weight_raw = entry
        try:
            src = int(src_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][0] must be an integer endpoint"
            ) from exc
        try:
            dst = int(dst_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][1] must be an integer endpoint"
            ) from exc
        if src < 0 or dst < 0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] endpoints must be >= 0"
            )
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][2] must be a numeric reliability weight"
            ) from exc
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] weight must be between 0.0 and 1.0"
            )
        entries.append((src, dst, weight))
    return tuple(entries)

def _parse_network_overlap(overlap_raw) -> "NetworkOverlapConfig":
    if not isinstance(overlap_raw, dict):
        raise ValueError("network.overlap must be a mapping with tp_overlap, tp_sp_overlap, and cp_overlap")
    required_fields = ("tp_overlap", "tp_sp_overlap", "cp_overlap")
    values = {}
    for field in required_fields:
        if field not in overlap_raw:
            raise ValueError(f"network.overlap missing required field '{field}'")
        try:
            val = float(overlap_raw[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network.overlap.{field} must be numeric") from exc
        if val < 0.0 or val > 1.0:
            raise ValueError(f"network.overlap.{field} must be between 0.0 and 1.0")
        values[field] = val
    return NetworkOverlapConfig(
        tp_overlap=values["tp_overlap"],
        tp_sp_overlap=values["tp_sp_overlap"],
        cp_overlap=values["cp_overlap"],
    )


def _build_network_layout_config(
    dimensions: Sequence[NetworkDimensionLayout],
    faulty_links: Sequence[Tuple[int, int, float]] = (),
    overlap_config: Optional["NetworkOverlapConfig"] = None,
) -> NetworkLayoutConfig:
    parallelism_map: Dict[str, NetworkDimensionLayout] = {}
    for dim in dimensions:
        for pname in dim.parallelisms:
            if pname in parallelism_map:
                raise ValueError(
                    f"parallelism '{pname}' assigned to multiple network dimensions"
                )
            parallelism_map[pname] = dim
    return NetworkLayoutConfig(
        dimensions=tuple(dimensions),
        faulty_links=tuple(faulty_links),
        parallelism_map=parallelism_map,
        overlap_config=overlap_config,
    )


_PARALLELISM_DEFAULTS: Dict[str, object] = {
    "auto": False,
    "pp": 1,
    "mb": 1,
    "tp": 1,
    "cp": 1,
    "tp_sp": False,
}


@dataclass
class TrainParallelismConfig:
    dp: int
    ep: int
    tp_ep: bool

    @classmethod
    def from_dict(cls, train_block: Optional[Dict[str, object]]) -> "TrainParallelismConfig":
        train_block = _require_mapping("parallelism.train", train_block or {})
        dp = _coerce_int(_require_field("parallelism.train", train_block, "dp"), "parallelism.train.dp")
        ep = _coerce_int(_require_field("parallelism.train", train_block, "ep"), "parallelism.train.ep")
        tp_ep = _coerce_bool(_require_field("parallelism.train", train_block, "tp_ep"), "parallelism.train.tp_ep")
        return cls(dp=dp, ep=ep, tp_ep=tp_ep)


@dataclass
class InferenceParallelismConfig:
    replica_count: int
    moe_dp: int

    @classmethod
    def from_dict(cls, inference_block: Optional[Dict[str, object]]) -> "InferenceParallelismConfig":
        inference_block = _require_mapping("parallelism.inference", inference_block or {})
        replica_count = _coerce_int(
            _require_field("parallelism.inference", inference_block, "replica_count"),
            "parallelism.inference.replica_count",
        )
        moe_dp = _coerce_int(
            _require_field("parallelism.inference", inference_block, "moe_dp"),
            "parallelism.inference.moe_dp",
        )
        return cls(replica_count=replica_count, moe_dp=moe_dp)


@dataclass
class MemoryConfig:
    type: str
    scope: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d["type"],
            scope=d["scope"],
        )


@dataclass
class MemoryHierarchyConfig:
    num_levels: int
    mem_hr: list

    @classmethod
    def from_dict(cls, d):
        num_levels = len(d)
        mem_hr = [None] * num_levels
        for level in range(num_levels):
            m = MemoryConfig.from_dict(d["l" + str(level)])
            mem_hr[level] = m
        return cls(
            num_levels=num_levels,
            mem_hr=mem_hr,
        )


def _require_mapping(context: str, value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_field(context: str, data: Dict[str, object], field: str) -> object:
    if field not in data:
        raise ValueError(f"{context}.{field} must be specified")
    return data[field]


def _parse_str_field(context: str, data: Dict[str, object], field: str) -> str:
    value = _require_field(context, data, field)
    return str(value).strip()


def _parse_int_field(context: str, data: Dict[str, object], field: str, *, min_value: Optional[int] = 1) -> int:
    value = _require_field(context, data, field)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{field} must be an integer (got {value!r})") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{context}.{field} must be >= {min_value}")
    return parsed


def _coerce_bool(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{context} must be a boolean (got {value!r})")


def _parse_bool_field(context: str, data: Dict[str, object], field: str) -> bool:
    value = _require_field(context, data, field)
    return _coerce_bool(value, f"{context}.{field}")


@dataclass
class GEMMConfig:
    mode: str
    M: int
    K: int
    N: int
    backward: bool
    gemm_shard_axis: str

    @classmethod
    def from_dict(cls, model_dict: Dict[str, object]) -> "GEMMConfig":
        model_dict = _require_mapping("model_param", model_dict)
        mode_raw = _parse_str_field("model_param", model_dict, "mode")
        mode = mode_raw.upper()
        if mode != "GEMM":
            raise ValueError(f"model_param.mode must be 'GEMM' for GEMM configs (got {mode_raw!r})")
        M = _parse_int_field("model_param", model_dict, "M")
        K = _parse_int_field("model_param", model_dict, "K")
        N = _parse_int_field("model_param", model_dict, "N")
        backward = _coerce_bool(model_dict.get("backward", False), "model_param.backward")
        axis_raw = _parse_str_field("model_param", model_dict, "gemm_shard_axis")
        axis = axis_raw.strip().lower()
        if axis not in {"row", "col"}:
            raise ValueError(
                "model_param.gemm_shard_axis must be 'row' or 'col' "
                f"(got {axis_raw!r})"
            )
        return cls(
            mode=mode,
            M=M,
            K=K,
            N=N,
            backward=backward,
            gemm_shard_axis=axis,
        )
@dataclass
class LLMAttentionConfig:
    attention_type: str
    num_heads: int
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    use_flashattention: bool = False
    attention_tile_size: Optional[int] = None

    @classmethod
    def from_dict(cls, attention_dict: Dict[str, object]) -> "LLMAttentionConfig":
        attention_dict = _require_mapping("model_param.attention", attention_dict)

        attn_type_raw = _parse_str_field("model_param.attention", attention_dict, "attention_type")
        attn_type = attn_type_raw.strip().lower()
        if attn_type == "mla":
            raise NotImplementedError("attention_type='mla' is not yet supported. Please use 'mha' or 'gqa'.")
        if attn_type not in {"mha", "gqa"}:
            raise ValueError(
                f"model_param.attention.attention_type must be either 'mha' or 'gqa' (got {attn_type_raw!r})"
            )

        num_heads = _parse_int_field("model_param.attention", attention_dict, "num_heads")
        head_dim_raw = attention_dict.get("head_dim", None)
        if head_dim_raw is not None:
            try:
                head_dim = int(head_dim_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.head_dim must be an integer when provided (got {head_dim_raw!r})"
                ) from exc
            if head_dim <= 0:
                raise ValueError("model_param.attention.head_dim must be a positive integer")
        else:
            head_dim = None
        kv_heads_raw = attention_dict.get("kv_heads", None)
        if attn_type == "gqa":
            if kv_heads_raw is None:
                raise ValueError(
                    "model_param.attention.kv_heads must be specified when attention_type='gqa'"
                )
            try:
                kv_heads = int(kv_heads_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.kv_heads must be an integer when attention_type='gqa' (got {kv_heads_raw!r})"
                ) from exc
            if kv_heads <= 0:
                raise ValueError("model_param.attention.kv_heads must be a positive integer")
            if num_heads % kv_heads != 0:
                raise ValueError(
                    f"model_param.attention.kv_heads={kv_heads} must divide num_heads={num_heads}"
                )
        else:
            kv_heads = num_heads

        raw_flash = attention_dict.get(
            "use_flashattention",
            attention_dict.get("used_flash_attention", False),
        )
        use_flashattention = _coerce_bool(
            raw_flash,
            "model_param.attention.use_flashattention",
        )

        attention_tile_size = attention_dict.get("attention_tile_size", None)
        if use_flashattention:
            if attention_tile_size is None:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be specified when flash attention is enabled"
                )
            try:
                attention_tile_size = int(attention_tile_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be an integer when flash attention is enabled "
                    f"(got {attention_tile_size!r})"
                ) from exc
            if attention_tile_size <= 0:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be a positive integer when flash attention is enabled"
                )
        else:
            attention_tile_size = None

        return cls(
            attention_type=attn_type,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            use_flashattention=use_flashattention,
            attention_tile_size=attention_tile_size,
        )


@dataclass
class MoEConfig:
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    n_shared_experts: int
    moe_layer_freq: int
    first_k_dense_replace: int

    @classmethod
    def from_dict(
        cls,
        moe_dict: Dict[str, object],
        *,
        validate: bool = True,
        fallback_intermediate_size: Optional[int] = None,
    ) -> "MoEConfig":
        moe_dict = _require_mapping("model_param.moe", moe_dict)
        if not validate:
            def _lenient_int(value: object, default: int) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            if fallback_intermediate_size is None:
                fallback_intermediate_size = 1
            fallback_intermediate_size = _lenient_int(fallback_intermediate_size, 1)

            num_experts = _lenient_int(moe_dict.get("num_experts", 1), 1)
            top_k = _lenient_int(moe_dict.get("top_k", 1), 1)
            moe_intermediate_size = _lenient_int(
                moe_dict.get("moe_intermediate_size", fallback_intermediate_size),
                fallback_intermediate_size,
            )
            n_shared_experts = _lenient_int(moe_dict.get("n_shared_experts", 0), 0)
            moe_layer_freq = _lenient_int(moe_dict.get("moe_layer_freq", 1), 1)
            first_k_dense_replace = _lenient_int(moe_dict.get("first_k_dense_replace", 0), 0)
        else:
            num_experts = _parse_int_field("model_param.moe", moe_dict, "num_experts")
            top_k = _parse_int_field("model_param.moe", moe_dict, "top_k")
            moe_intermediate_size = _parse_int_field("model_param.moe", moe_dict, "moe_intermediate_size")
            n_shared_experts = _coerce_int(
                moe_dict.get("n_shared_experts", 0),
                "model_param.moe.n_shared_experts",
                min_value=0,
            )
            moe_layer_freq = _coerce_int(
                moe_dict.get("moe_layer_freq", 1),
                "model_param.moe.moe_layer_freq",
                min_value=1,
            )
            first_k_dense_replace = _coerce_int(
                moe_dict.get("first_k_dense_replace", 0),
                "model_param.moe.first_k_dense_replace",
                min_value=0,
            )

            if top_k > num_experts:
                raise ValueError("model_param.moe.top_k cannot exceed model_param.moe.num_experts")

        # TODO: Shared experts are modeled as replicated across EP for now.
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            moe_intermediate_size=moe_intermediate_size,
            n_shared_experts=n_shared_experts,
            moe_layer_freq=moe_layer_freq,
            first_k_dense_replace=first_k_dense_replace,
        )


@dataclass
class LLMConfig:
    mode: str
    run_type: str
    model_type: str
    tied_embeddings: bool
    num_layers: int
    hidden_dim: int
    global_batch_size: int
    gradient_accumulation_steps: int
    seq_len: int
    decode_len: Optional[int]
    intermediate_size: Optional[int]
    vocab_size: int
    n_tokens: int
    attention: LLMAttentionConfig
    moe: MoEConfig

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads

    @property
    def head_dim(self) -> int:
        if getattr(self.attention, "head_dim", None) is not None:
            return int(self.attention.head_dim)
        return self.hidden_dim // self.num_heads

    @property
    def use_flashattention(self) -> bool:
        return bool(getattr(self.attention, "use_flashattention", False))

    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1 and self.num_moe_layers > 0

    @property
    def num_experts(self) -> int:
        return self.moe.num_experts

    @property
    def top_k(self) -> int:
        return self.moe.top_k

    @property
    def num_moe_layers(self) -> int:
        return sum(self.moe_layer_mask)

    @property
    def n_shared_experts(self) -> int:
        return self.moe.n_shared_experts

    @property
    def moe_intermediate_size(self) -> int:
        return self.moe.moe_intermediate_size

    @property
    def moe_layer_freq(self) -> int:
        return self.moe.moe_layer_freq

    @property
    def first_k_dense_replace(self) -> int:
        return self.moe.first_k_dense_replace

    @property
    def moe_params_enabled(self) -> bool:
        return self.num_experts > 1 and self.num_moe_layers > 0

    @property
    def moe_layer_mask(self) -> List[bool]:
        if self.num_experts <= 1:
            return [False for _ in range(self.num_layers)]
        mask: List[bool] = []
        for layer_idx in range(self.num_layers):
            if layer_idx < self.first_k_dense_replace:
                mask.append(False)
                continue
            if self.moe_layer_freq <= 0:
                mask.append(False)
                continue
            mask.append(((layer_idx - self.first_k_dense_replace) % self.moe_layer_freq) == 0)
        return mask


    @property
    def grad_accumulation_steps(self) -> int:
        """Backward-compatible alias for gradient accumulation steps."""
        return self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, model_dict: Dict[str, object]) -> "LLMConfig":
        model_dict = _require_mapping("model_param", model_dict)

        mode_raw = _parse_str_field("model_param", model_dict, "mode")
        mode = mode_raw.strip().upper()
        if mode != "LLM":
            raise ValueError(f"model_param.mode must be 'LLM' for LLM configs (got {mode_raw!r})")

        run_type_raw = _parse_str_field("model_param", model_dict, "run_type")
        run_type = run_type_raw.strip().lower()
        if run_type not in {"training", "inference"}:
            raise ValueError(
                f"model_param.run_type must be either 'training' or 'inference' (got {run_type_raw!r})"
            )

        tied_embeddings = _coerce_bool(
            _require_field("model_param", model_dict, "tied_embeddings"),
            "model_param.tied_embeddings",
        )

        model_type_raw = _parse_str_field("model_param", model_dict, "model_type")
        model_type = model_type_raw.strip().lower()
        if model_type in {"glm4", "glm"}:
            model_type = "glm4_moe"
        if model_type not in {"gpt", "llama", "glm4_moe"}:
            raise ValueError(
                "model_param.model_type must be either 'gpt', 'llama', or 'glm4_moe' "
                f"(got {model_type_raw!r})"
            )

        attention = LLMAttentionConfig.from_dict(_require_field("model_param", model_dict, "attention"))

        num_layers = _parse_int_field("model_param", model_dict, "num_layers")
        hidden_dim = _parse_int_field("model_param", model_dict, "hidden_dim")
        global_batch_size = _parse_int_field("model_param", model_dict, "global_batch_size")
        grad_accum_raw = model_dict.get("gradient_accumulation_steps", None)
        if grad_accum_raw is None:
            grad_accum_raw = model_dict.get("gradient_accumulation_step", None)
        if grad_accum_raw is None:
            gradient_accumulation_steps = 1
        else:
            try:
                gradient_accumulation_steps = int(grad_accum_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.gradient_accumulation_steps must be an integer (got {grad_accum_raw!r})"
                ) from exc
            if gradient_accumulation_steps <= 0:
                raise ValueError("model_param.gradient_accumulation_steps must be a positive integer")
        seq_len = _parse_int_field("model_param", model_dict, "seq_len")
        vocab_size = _parse_int_field("model_param", model_dict, "vocab_size")
        intermediate_size = _parse_int_field("model_param", model_dict, "intermediate_size")

        if model_type == "glm4_moe":
            if attention.head_dim is None:
                raise ValueError(
                    "model_param.attention.head_dim must be specified when model_type is 'glm4_moe'"
                )
        elif attention.head_dim is not None:
            if hidden_dim % attention.num_heads != 0:
                raise ValueError(
                    "model_param.hidden_dim must be divisible by attention.num_heads when "
                    "model_param.attention.head_dim is provided for non-GLM models"
                )
            expected_head_dim = hidden_dim // attention.num_heads
            if attention.head_dim != expected_head_dim:
                raise ValueError(
                    "model_param.attention.head_dim must match hidden_dim/num_heads for non-GLM models "
                    f"(expected {expected_head_dim}, got {attention.head_dim})"
                )
        else:
            if hidden_dim % attention.num_heads != 0:
                raise ValueError(
                    "model_param.hidden_dim must be divisible by attention.num_heads when "
                    "model_param.attention.head_dim is not provided"
                )

        decode_len = model_dict.get("decode_len", None)
        if decode_len is not None:
            try:
                decode_len = int(decode_len)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.decode_len must be an integer when provided (got {decode_len!r})"
                ) from exc

        if run_type == "inference" and decode_len is None:
            raise ValueError("model_param.decode_len must be specified when run_type is 'inference'")

        moe_block = model_dict.get("moe", {})
        if moe_block is None:
            moe_block = {}
        moe_candidate = MoEConfig.from_dict(
            moe_block,
            validate=False,
            fallback_intermediate_size=intermediate_size,
        )

        def _count_moe_layers(
            *,
            num_layers: int,
            moe_layer_freq: int,
            first_k_dense_replace: int,
        ) -> int:
            count = 0
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace:
                    continue
                if moe_layer_freq <= 0:
                    continue
                if ((layer_idx - first_k_dense_replace) % moe_layer_freq) == 0:
                    count += 1
            return count

        moe_enabled = (
            moe_candidate.num_experts > 1
            and _count_moe_layers(
                num_layers=num_layers,
                moe_layer_freq=moe_candidate.moe_layer_freq,
                first_k_dense_replace=moe_candidate.first_k_dense_replace,
            )
            > 0
        )
        moe = moe_candidate
        if moe_enabled:
            moe = MoEConfig.from_dict(moe_block)

        return cls(
            mode=mode,
            run_type=run_type,
            model_type=model_type,
            tied_embeddings=tied_embeddings,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            global_batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seq_len=seq_len,
            decode_len=decode_len,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            n_tokens=0,
            attention=attention,
            moe=moe,
        )


@dataclass
class LLMInferenceConfig:
    sample_every: int = -1

    @classmethod
    def from_dict(cls, inference_dict: Optional[Dict[str, object]]) -> "LLMInferenceConfig":
        if not inference_dict:
            return cls(sample_every=-1)
        inference_dict = _require_mapping("inference_param", inference_dict)
        raw = inference_dict.get("sample_every", -1)
        try:
            sample_every = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"inference_param.sample_every must be an integer (got {raw!r})"
            ) from exc
        return cls(sample_every=sample_every)
def _coerce_int(value: object, context: str, *, min_value: Optional[int] = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer (got {value!r})") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{context} must be >= {min_value}")
    return parsed


@dataclass
class SWConfig:
    kernel_launch_overhead: float
    precision: PrecisionConfig
    h2d_bandwidth: float
    dp_zero_stage: int
    full_recomputation: bool
    dp_microbatch: str
    const_mem_offset: float
    grad_acc_overhead: float

    @classmethod
    def from_dict(cls, sw_block: Dict[str, object]) -> "SWConfig":
        sw_block = _require_mapping("sw_param", sw_block)
        precision_spec = _require_field("sw_param", sw_block, "precision")
        precision_config = _parse_precision_block(precision_spec)
        kernel_launch_overhead = float(_require_field("sw_param", sw_block, "kernel_launch_overhead"))
        h2d_bandwidth = float(sw_block.get("h2d_bandwidth", -1))
        dp_zero_stage = _coerce_int(sw_block.get("dp_zero_stage", 0), "sw_param.dp_zero_stage", min_value=0)
        full_recomputation = _coerce_bool(
            sw_block.get("full_recomputation", False),
            "sw_param.full_recomputation",
        )
        dp_microbatch_raw = sw_block.get("dp_microbatch", "every_mb")
        dp_microbatch = str(dp_microbatch_raw).strip().lower()
        if dp_microbatch not in {"every_mb", "last_mb"}:
            raise ValueError("sw_param.dp_microbatch must be 'every_mb' or 'last_mb'")
        const_mem_offset_raw = sw_block.get("const_mem_offset", 0.0)
        if const_mem_offset_raw is None:
            const_mem_offset = 0.0
        else:
            try:
                const_mem_offset = float(const_mem_offset_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sw_param.const_mem_offset must be a float-compatible value (got {const_mem_offset_raw!r})"
                ) from exc
        grad_acc_overhead_raw = sw_block.get("grad_acc_overhead", 0.0)
        if grad_acc_overhead_raw is None:
            grad_acc_overhead = 0.0
        else:
            try:
                grad_acc_overhead = float(grad_acc_overhead_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sw_param.grad_acc_overhead must be a float-compatible value (got {grad_acc_overhead_raw!r})"
                ) from exc
        return cls(
            kernel_launch_overhead=kernel_launch_overhead,
            precision=precision_config,
            h2d_bandwidth=h2d_bandwidth,
            dp_zero_stage=dp_zero_stage,
            full_recomputation=full_recomputation,
            dp_microbatch=dp_microbatch,
            const_mem_offset=const_mem_offset,
            grad_acc_overhead=grad_acc_overhead,
        )


@dataclass
class SchedulingConfig:
    auto: bool
    pp: int
    mb: int
    tp: int
    cp: int
    tp_sp: bool
    train: TrainParallelismConfig
    inference: InferenceParallelismConfig

    @classmethod
    def from_dict(cls, parallelism_block: Optional[Dict[str, object]]) -> "SchedulingConfig":
        if parallelism_block is None:
            parallelism_block = {}
        parallelism_block = _require_mapping("parallelism", parallelism_block)
        params = dict(_PARALLELISM_DEFAULTS)
        params.update(parallelism_block)
        auto = _coerce_bool(params.get("auto", False), "parallelism.auto")
        tp_sp = _coerce_bool(params.get("tp_sp", False), "parallelism.tp_sp")
        pp = _coerce_int(params.get("pp", 1), "parallelism.pp")
        mb = _coerce_int(params.get("mb", 1), "parallelism.mb")
        tp = _coerce_int(params.get("tp", 1), "parallelism.tp")
        cp = _coerce_int(params.get("cp", 1), "parallelism.cp")
        train_cfg = TrainParallelismConfig.from_dict(_require_field("parallelism", parallelism_block, "train"))
        inference_cfg = InferenceParallelismConfig.from_dict(
            _require_field("parallelism", parallelism_block, "inference")
        )
        return cls(
            auto=auto,
            pp=pp,
            mb=mb,
            tp=tp,
            cp=cp,
            tp_sp=tp_sp,
            train=train_cfg,
            inference=inference_cfg,
        )


@dataclass
class FullConfig:
    model_config: object
    sw_config: SWConfig
    tech_config: TechConfig
    power_breakdown: PowerBreakdownConfig
    sch_config: SchedulingConfig
    area_breakdown: AreaBreakdownConfig
    perimeter_breakdown: PerimeterBreakdownConfig
    memory_hierarchy: MemoryHierarchyConfig
    network_layout: NetworkLayoutConfig


@dataclass
class ExecutionBackendAstraCollectives:
    all_gather: str = "auto"
    all_reduce: str = "auto"
    reduce_scatter: str = "auto"
    all_to_all: str = "auto"

    @classmethod
    def from_dict(cls, coll_dict: Optional[Dict[str, object]]) -> "ExecutionBackendAstraCollectives":
        if not coll_dict:
            return cls()
        coll_dict = _require_mapping("execution_backend.astra.collectives", coll_dict)
        return cls(
            all_gather=str(coll_dict.get("all_gather", "auto")),
            all_reduce=str(coll_dict.get("all_reduce", "auto")),
            reduce_scatter=str(coll_dict.get("reduce_scatter", "auto")),
            all_to_all=str(coll_dict.get("all_to_all", "auto")),
        )


@dataclass
class ExecutionBackendAstraSysOptions:
    endpoint_delay: Optional[int] = None
    active_chunks_per_dimension: Optional[int] = None
    preferred_dataset_splits: Optional[int] = None
    collective_arbitration: Optional[str] = None

    @classmethod
    def from_dict(cls, sys_dict: Optional[Dict[str, object]]) -> Optional["ExecutionBackendAstraSysOptions"]:
        if sys_dict is None:
            return None
        sys_dict = _require_mapping("execution_backend.astra.sys_options", sys_dict)
        endpoint_delay = sys_dict.get("endpoint_delay", None)
        active_chunks = sys_dict.get("active_chunks_per_dimension", None)
        preferred_splits = sys_dict.get("preferred_dataset_splits", None)
        collective_arbitration = sys_dict.get("collective_arbitration", None)
        if collective_arbitration is not None:
            collective_arbitration = str(collective_arbitration).strip().lower()
            allowed = {"off", "last_resort", "best_effort", "strict", "on", "true", "false", "0", "1"}
            if collective_arbitration not in allowed:
                raise ValueError(
                    "execution_backend.astra.sys_options.collective_arbitration "
                    f"must be one of {sorted(allowed)} (got {collective_arbitration!r})"
                )
        return cls(
            endpoint_delay=None if endpoint_delay is None else _coerce_int(endpoint_delay, "execution_backend.astra.sys_options.endpoint_delay", min_value=0),
            active_chunks_per_dimension=None if active_chunks is None else _coerce_int(active_chunks, "execution_backend.astra.sys_options.active_chunks_per_dimension", min_value=1),
            preferred_dataset_splits=None if preferred_splits is None else _coerce_int(preferred_splits, "execution_backend.astra.sys_options.preferred_dataset_splits", min_value=1),
            collective_arbitration=collective_arbitration,
        )


@dataclass
class ExecutionBackendAstra:
    backend: str
    mode: str
    collectives: ExecutionBackendAstraCollectives
    sys_options: Optional[ExecutionBackendAstraSysOptions]

    @classmethod
    def from_dict(cls, astra_dict: Optional[Dict[str, object]]) -> "ExecutionBackendAstra":
        astra_dict = _require_mapping("execution_backend.astra", astra_dict or {})
        backend = str(astra_dict.get("backend", "analytical"))
        mode = str(astra_dict.get("mode", "hybrid"))
        collectives = ExecutionBackendAstraCollectives.from_dict(astra_dict.get("collectives"))
        sys_options = ExecutionBackendAstraSysOptions.from_dict(astra_dict.get("sys_options"))
        return cls(
            backend=backend,
            mode=mode,
            collectives=collectives,
            sys_options=sys_options,
        )


@dataclass
class ExecutionBackend:
    model: str
    astra: Optional[ExecutionBackendAstra]

    @classmethod
    def from_dict(cls, backend_dict: Optional[Dict[str, object]]) -> "ExecutionBackend":
        backend_dict = _require_mapping("execution_backend", backend_dict or {})
        model = str(backend_dict.get("model", "analytical"))
        astra_cfg = backend_dict.get("astra", {}) if model == "astra" else None
        astra = ExecutionBackendAstra.from_dict(astra_cfg) if astra_cfg is not None else None
        return cls(model=model, astra=astra)


@dataclass
class InferenceHWConfig:
    kvcache_type: str

    @classmethod
    def from_dict(cls, inference_dict: Optional[Dict[str, object]]) -> "InferenceHWConfig":
        if not inference_dict:
            return cls(kvcache_type="hbm_only")
        inference_dict = _require_mapping("inference", inference_dict)
        return cls(kvcache_type=str(inference_dict.get("kvcache_type", "hbm_only")))


@dataclass
class HWConfig:
    sw_config: SWConfig
    tech_config: TechConfig
    power_breakdown: PowerBreakdownConfig
    sch_config: SchedulingConfig
    area_breakdown: AreaBreakdownConfig
    perimeter_breakdown: PerimeterBreakdownConfig
    memory_hierarchy: MemoryHierarchyConfig
    network_layout: NetworkLayoutConfig
    execution_backend: ExecutionBackend
    inference_config: InferenceHWConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, object]) -> "HWConfig":
        config_dict = _require_mapping("hardware_config", config_dict)
        sw_config = SWConfig.from_dict(_require_field("hardware_config", config_dict, "sw_param"))
        sch_config = SchedulingConfig.from_dict(config_dict.get("parallelism", {}))
        scheduling_for_network = {
            "auto": sch_config.auto,
            "dp": sch_config.train.dp,
            "ep": sch_config.train.ep,
            "pp": sch_config.pp,
            "mb": sch_config.mb,
            "tp": sch_config.tp,
            "cp": sch_config.cp,
            "tp_sp": sch_config.tp_sp,
        }
        network_dimensions, network_faults, network_overlap = _parse_network_layout(
            config_dict.get("network"),
            scheduling_for_network,
        )
        if not network_dimensions:
            raise ValueError("network section must define at least one dimension")
        network_layout_config = _build_network_layout_config(
            network_dimensions,
            network_faults,
            network_overlap,
        )
        tech_config = TechConfig.from_dict(_require_field("hardware_config", config_dict, "tech_param"))

        if "power_breakdown" in config_dict:
            power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
        else:
            power_config = PowerBreakdownConfig(
                TDP=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                network=NetworkPowerConfig(inter_node=1.0, intra_node=1.0)
            )
        if "area_breakdown" in config_dict:
            area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
        else:
            area_config = AreaBreakdownConfig(
                proc_chip_area_budget=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                node_area_budget=1.0, network=NetworkAreaConfig(inter_node=1.0, intra_node=1.0)
            )
        if "perimeter_breakdown" in config_dict:
            perimeter_config = PerimeterBreakdownConfig.from_dict(config_dict["perimeter_breakdown"])
        else:
            perimeter_config = PerimeterBreakdownConfig(DRAM=0.1, inter_node=0.1, intra_node=0.1)

        memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
            _require_field("hardware_config", config_dict, "memory_hierarchy")
        )
        execution_backend = ExecutionBackend.from_dict(config_dict.get("execution_backend", {}))
        inference_config = InferenceHWConfig.from_dict(config_dict.get("inference"))

        return cls(
            sw_config=sw_config,
            tech_config=tech_config,
            power_breakdown=power_config,
            sch_config=sch_config,
            area_breakdown=area_config,
            perimeter_breakdown=perimeter_config,
            memory_hierarchy=memory_hierarchy_config,
            network_layout=network_layout_config,
            execution_backend=execution_backend,
            inference_config=inference_config,
        )

@dataclass
class ModelConfig:
    model_config: object
    inference_config: Optional["LLMInferenceConfig"]


def _convert_scalar_string(value: str):
    try:
        return float(value)
    except ValueError:
        pass

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]
    if not (order and digit):
        return value

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else "B"
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix '{prefix}' while parsing value '{value}'")

    if bit == "b":
        mult = mult / 8  # Capacity is expected in Bytes
    elif bit != "B":
        raise ValueError(f"Unknown type '{bit}' while parsing value '{value}'")

    return digit[0] * mult


def _convert_value(value):
    if isinstance(value, dict):
        convert(value)
        return value
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            return _convert_scalar_string(value)
        except ValueError:
            return value
    return value


def convert(d):
    if not isinstance(d, dict):
        return d
    for key, val in list(d.items()):
        d[key] = _convert_value(val)
    return d


def parse_bandwidth_string(value):
    """Parse bandwidth/size string (e.g., '300 GB', '1986 GB') to bytes.

    This function uses the same logic as convert() to parse bandwidth strings.
    Returns the numeric value if already a number, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(parse_bandwidth_string(item) for item in value)

    if not isinstance(value, str):
        return float(value)

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]

    if not order or not digit:
        # If no units found, try to parse as float
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse bandwidth value: {value}")

    assert len(order) >= 1
    assert len(digit) >= 1

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else 'B'  # Default to Bytes
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix: {prefix} in bandwidth value: {value}")

    if bit == "b":
        mult = mult / 8  # Convert bits to Bytes
    elif bit == "B":
        mult = mult
    else:
        raise ValueError(f"Unknown type: {bit} in bandwidth value: {value}")

    return digit[0] * mult


def parse_config(filename, config_type):
    """Parse a yaml configuration file for this experiment.
    Args:
            filename (str): Path to the configuration file
    Returns:
            FullConfig: Contains dataset, model, optimization, training and
            scheduling configurations
    """
    with open(filename, "r") as f:
        try:
            config_dict = _yaml.safe_load(f)
        except _YAMLError as exc:
            hint = (
                f"Failed to parse YAML config '{filename}'. "
                "Please check indentation and required sections like 'attention' and parameters such as 'moe.num_experts'."
            )
            raise ValueError(hint) from exc
        # print(config_dict)
        convert(config_dict)
    if config_type == "hardware":
        config = HWConfig.from_dict(config_dict)
    elif config_type == "GEMM":
        model_config = GEMMConfig.from_dict(config_dict["model_param"])
        config = ModelConfig(model_config=model_config, inference_config=None)
    elif config_type == "LLM":
        model_config = LLMConfig.from_dict(config_dict["model_param"])
        inference_config = None
        if model_config.run_type == "inference":
            inference_config = LLMInferenceConfig.from_dict(config_dict.get("inference_param"))
        config = ModelConfig(model_config=model_config, inference_config=inference_config)
    else:
        raise ValueError("Invalid config type: {}".format(config_type))
    
    # model_config = ModelConfig(**config_dict["model_param"])
    # sw_config = SWConfig(**config_dict["sw_param"])
    # sch_config = SchedulingConfig(**config_dict["parallelism"])
    # tech_config = TechConfig.from_dict(config_dict["tech_param"])
    # power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
    # area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
    # perimeter_config = PerimeterBreakdownConfig.from_dict(
    #     config_dict["perimeter_breakdown"]
    # )
    # system_config = SystemHierarchyConfig.from_dict(config_dict["system_hierarchy"])
    # memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
    #     config_dict["memory_hierarchy"]
    # )
    # network_topology_config = NetworkTopologyConfig.from_dict(
    #     config_dict["network_topology"]
    # )

    return config


def validate_hw_config(hw_config: HWConfig) -> None:
    backend = getattr(hw_config, "execution_backend", None)
    model = getattr(backend, "model", "analytical") if backend else "analytical"
    network_layout = getattr(hw_config, "network_layout", None)
    if str(model).lower() == "analytical" and network_layout:
        for dim in getattr(network_layout, "dimensions", ()):
            topo = str(getattr(dim, "topology_type", "ring")).lower()
            if topo != "ring":
                raise RuntimeError(
                    "Non-ring network topologies are not supported in analytical mode. "
                    "Only execution_backend.model='astra' (requires a valid AstraSim install) supports non-ring networks."
                )


def validate_model_config(hw_config: HWConfig, model_config: ModelConfig) -> None:
    sch = getattr(hw_config, "sch_config", None)
    if sch is None:
        raise ValueError("hardware parallelism settings are missing")

    pp = sch.pp
    mb = sch.mb
    tp = sch.tp
    cp = sch.cp
    train_dp = sch.train.dp
    train_ep = sch.train.ep

    model = model_config.model_config

    if isinstance(model, GEMMConfig):
        if tp > 1:
            if model.gemm_shard_axis == "row" and (model.K % tp != 0):
                raise ValueError("GEMM row sharding requires K divisible by tp")
            if model.gemm_shard_axis == "col" and (model.N % tp != 0):
                raise ValueError("GEMM col sharding requires N divisible by tp")
        return

    if not isinstance(model, LLMConfig):
        raise ValueError("Unsupported model config type for validation")

    if model.use_moe and model.top_k > model.num_experts:
        raise ValueError("model_param.moe.top_k cannot exceed model_param.moe.num_experts")

    run_type = str(getattr(model, "run_type", "training")).lower()
    if run_type == "inference":
        replica_count = sch.inference.replica_count
        moe_dp = sch.inference.moe_dp
        if cp > 1:
            raise ValueError(
                "Context parallelism (cp) is not supported for LLM inference. "
                "Please set parallelism.cp to 1 for inference runs."
            )
        if mb > 1:
            print(
                f"[WARNING]: LLM inference configured with mb={mb} (>1). \n "
                "Pipeline micro-batching is ill-defined for autoregressive decode and should be avoided."
            )
        if getattr(hw_config.sw_config, "dp_zero_stage", 0) >= 3 and replica_count > 1:
            raise ValueError(
                "ZeRO-3 data parallelism is not supported for inference runs "
                "(dp_zero_stage must be <3 or replica_count=1)."
            )
        if not model.use_moe and moe_dp > 1:
            raise ValueError(
                "parallelism.inference.moe_dp must be 1 when MoE is disabled."
            )
        if model.decode_len is not None and model.decode_len > model.seq_len:
            raise ValueError("model_param.decode_len must be <= seq_len for inference")
    else:
        if cp > 1 and train_ep > 1:
            raise ValueError(
                "Unsupported parallelism combination: cp > 1 with ep > 1 is not allowed. "
                "Set parallelism.cp=1 or parallelism.train.ep=1."
            )

    if (
        model.gradient_accumulation_steps > 1
        and getattr(hw_config.sw_config, "dp_zero_stage", 0) >= 2
    ):
        # ZeRO-2/3 do funky things with DP comms and aren't captured by the current
        # no-DP + DP step model for gradient accumulation.
        raise ValueError(
            "Gradient accumulation steps > 1 is not supported with ZeRO-2/3 (dp_zero_stage >= 2)."
        )

    if model.global_batch_size % model.gradient_accumulation_steps != 0:
        raise ValueError(
            "Global batch size must be divisible by gradient accumulation steps"
        )
    if run_type != "inference":
        batch_size = model.global_batch_size // model.gradient_accumulation_steps
        dp_dense = train_dp * train_ep if model.use_moe else train_dp
        if batch_size % dp_dense != 0:
            if model.use_moe:
                raise ValueError(f"Batch size must be divisible by dp*ep when MoE is enabled: {batch_size} % {dp_dense} != 0")
            raise ValueError(f"Batch size must be divisible by data parallelism degree: {batch_size} % {train_dp} != 0")
        mini_batch = batch_size // dp_dense
        if mini_batch % mb != 0:
            raise ValueError(f"Batch size must be divisible by micro-batch size: {mini_batch} % {mb} != 0")

        if not model.use_moe and train_ep > 1:
            raise ValueError(
                "parallelism.train.ep must be 1 when MoE is disabled. "
                "Set parallelism.train.ep=1 or enable MoE."
            )

    if model.use_moe:
        allow_moe_padding = _env_flag("RAPID_ALLOW_MOE_EXPERT_PADDING")
        if run_type == "inference":
            moe_ranks = tp * max(1, moe_dp)
            if moe_ranks > model.num_experts:
                raise ValueError(
                    "MoE routing group size cannot exceed the number of MoE experts "
                    f"(moe_group={moe_ranks}, tp={tp}, moe_dp={moe_dp})."
                )
            if model.num_experts % moe_ranks != 0:
                if allow_moe_padding:
                    global _MOE_PADDING_WARNED
                    if not _MOE_PADDING_WARNED:
                        print(
                            "[WARNING]: MoE expert count is not divisible by tp*moe_dp for "
                            "inference; padding experts to enable simulation. "
                            "Set RAPID_ALLOW_MOE_EXPERT_PADDING=0 to enforce divisibility."
                        )
                        _MOE_PADDING_WARNED = True
                else:
                    raise ValueError(
                        "Number of MoE experts must be divisible by the MoE routing group size "
                        f"(moe_group={moe_ranks}, tp={tp}, moe_dp={moe_dp})."
                    )
        else:
            tp_sp = bool(getattr(sch, "tp_sp", False))
            if tp > 1 and train_ep > 1 and not tp_sp:
                raise ValueError(
                    "MoE with tp>1 and ep>1 requires sequence parallelism. "
                    "Set parallelism.tp_sp=true."
                )
            moe_ranks = train_ep
            if moe_ranks > model.num_experts:
                raise ValueError(
                    "MoE routing group size cannot exceed the number of MoE experts "
                    f"(moe_group={moe_ranks}, tp={tp}, ep={train_ep})."
                )
            if model.num_experts % moe_ranks != 0:
                raise ValueError(
                    "Number of MoE experts must be divisible by the MoE routing group size "
                    f"(moe_group={moe_ranks}, tp={tp}, ep={train_ep})."
                )
            effective_batch = mini_batch
            if pp > 1:
                effective_batch = mini_batch // mb
            elif dp_dense <= 1:
                effective_batch = batch_size
            seq_per_rank = math.ceil(model.seq_len / max(1, cp))
            tokens_owner = int(effective_batch) * int(seq_per_rank)
            tokens_dispatched = tokens_owner * int(model.top_k)
            if tokens_dispatched % moe_ranks != 0:
                raise ValueError(
                    "MoE routed tokens must divide evenly across the MoE routing group for batched expert GEMMs\n"
                    f"(tokens_owner = effective_batch * seq_per_rank = {effective_batch} * {seq_per_rank})\n"
                    f"(tokens_dispatched = tokens_owner * top_k = {tokens_owner} * {model.top_k})\n"
                    f"(tokens_dispatched={tokens_dispatched} % moe_group={moe_ranks} != 0)"
                )
            tokens_local = tokens_dispatched // moe_ranks
            experts_per_rank = model.num_experts // moe_ranks
            if tokens_local % experts_per_rank != 0:
                raise ValueError(
                    "MoE routed tokens per rank must divide evenly across experts for batched expert GEMMs\n"
                    f"(tokens_owner = effective_batch * seq_per_rank = {effective_batch} * {seq_per_rank})\n"
                    f"(tokens_dispatched = tokens_owner * top_k = {tokens_owner} * {model.top_k})\n"
                    f"(tokens_local = tokens_dispatched // moe_ranks = {tokens_dispatched} // {moe_ranks})\n"
                    f"(tokens_local={tokens_local} % experts_per_rank={experts_per_rank} != 0)\n"
                )
        backend = getattr(hw_config, "execution_backend", None)
        if backend and str(getattr(backend, "model", "")).lower() == "astra":
            astra_cfg = getattr(backend, "astra", None)
            astra_mode = str(getattr(astra_cfg, "mode", "")).lower() if astra_cfg else ""
            if astra_mode == "full_astrasim_flattened":
                raise NotImplementedError(
                    "MoE is not supported with full AstraSim flattened execution."
                )
        if run_type != "inference" and getattr(hw_config.sw_config, "dp_zero_stage", 0) >= 2:
            raise NotImplementedError("MoE with ZeRO-2/3 (dp_zero_stage >= 2) is not supported yet.")
        network_layout = getattr(hw_config, "network_layout", None)
        faulty_links = getattr(network_layout, "faulty_links", ()) if network_layout else ()
        if faulty_links:
            raise ValueError(
                "MoE with faulty links is not supported yet. Please disable faults or MoE."
            )


def validate_configs(hw_config: HWConfig, model_config: ModelConfig) -> None:
    validate_hw_config(hw_config)
    validate_model_config(hw_config, model_config)
