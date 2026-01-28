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
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class CollectiveType(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    PIPELINE = "pipeline"


@dataclass(frozen=True)
class CommSpec:
    """
    Metadata describing a single communication collective.

    Attributes:
        name: Unique identifier used when wiring graph edges.
        kind: Collective type (CollectiveType).
        size_bytes: Total bytes transferred (int >= 0).
        participants: Number of ranks involved (int >= 1).
        interconnect: Logical interconnect label (tp/dp/cp/pp/etc.).
        extra: Free-form metadata (e.g. debug labels).
    """

    name: str
    kind: CollectiveType
    size_bytes: int
    participants: int
    interconnect: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CommSpec.name must be non-empty")
        if not isinstance(self.kind, CollectiveType):
            raise TypeError(f"CommSpec.kind must be a CollectiveType (got {type(self.kind).__name__})")
        if self.size_bytes < 0:
            raise ValueError(f"CommSpec '{self.name}' has negative size_bytes ({self.size_bytes})")
        if self.participants <= 0:
            raise ValueError(f"CommSpec '{self.name}' must have participants >= 1 (got {self.participants})")
        if not self.interconnect:
            raise ValueError(f"CommSpec '{self.name}' missing interconnect label")


@dataclass(frozen=True)
class DirectionTiming:
    """
    Timing information for a single direction (forward/backward) of an operation.

    Attributes:
        compute_time: Pure computation time (seconds).
        comm_time: Time attributed to communication collectives (seconds).
        comm_bytes: Raw communication bytes (int >= 0). CommSpec instances are created
                    later during graph building based on COMMUNICATION_RULES.
        flops: Floating-point operation count (optional).
        memory_accesses: Tuple describing memory accesses per level (optional).
        notes: Misc metadata for debugging/reporting.
    """

    compute_time: float
    comm_time: float = 0.0
    comm_bytes: int = 0
    flops: float = 0.0
    memory_accesses: Mapping[str, float] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.compute_time < 0:
            raise ValueError(f"DirectionTiming compute_time must be >= 0 (got {self.compute_time})")
        if self.comm_time < 0:
            raise ValueError(f"DirectionTiming comm_time must be >= 0 (got {self.comm_time})")
        if self.comm_bytes < 0:
            raise ValueError(f"DirectionTiming comm_bytes must be >= 0 (got {self.comm_bytes})")
        if self.flops < 0:
            raise ValueError(f"DirectionTiming flops must be >= 0 (got {self.flops})")
        for level_bytes in self.memory_accesses.values():
            if level_bytes < 0:
                raise ValueError("DirectionTiming memory_accesses entries must be >= 0")

    def total_time(self) -> float:
        """Return compute + comm time."""
        return self.compute_time + self.comm_time

    def total_comm_bytes(self) -> int:
        """Return raw communication bytes."""
        return self.comm_bytes

    def has_comm(self) -> bool:
        return self.comm_bytes > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict for backwards compatibility (e.g. logging)."""
        return {
            "compute_time": self.compute_time,
            "comm_time": self.comm_time,
            "comm_bytes": self.comm_bytes,
            "flops": self.flops,
            "memory_accesses": dict(self.memory_accesses),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class OperationTiming:
    """
    Timing details for an operation (e.g., QKV GEMM, MLP).

    Attributes:
        name: Logical operation name.
        forward: Forward-direction timing (optional).
        backward: Backward-direction timing (optional).
        metadata: Additional operation-level metadata (e.g. layer index).
    """

    name: str
    forward: Optional[DirectionTiming]
    backward: Optional[DirectionTiming] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OperationTiming.name must be non-empty")
        if self.forward is None and self.backward is None:
            raise ValueError(f"OperationTiming '{self.name}' must have at least one direction")
        if self.forward is not None and not isinstance(self.forward, DirectionTiming):
            raise TypeError("OperationTiming.forward must be a DirectionTiming instance or None")
        if self.backward is not None and not isinstance(self.backward, DirectionTiming):
            raise TypeError("OperationTiming.backward must be a DirectionTiming instance or None")

    def has_backward(self) -> bool:
        return self.backward is not None

    def total_forward_time(self) -> float:
        if self.forward is None:
            raise RuntimeError(f"Operation '{self.name}' does not have forward timing")
        return self.forward.total_time()

    def total_backward_time(self) -> float:
        if self.backward is None:
            raise RuntimeError(f"Operation '{self.name}' does not have backward timing")
        return self.backward.total_time()

    def total_time(self) -> float:
        total = self.forward.total_time() if self.forward is not None else 0.0
        if self.backward is not None:
            total += self.backward.total_time()
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for reporting or artifact dumps."""
        data: Dict[str, Any] = {
            "name": self.name,
        }
        if self.forward is not None:
            data["forward"] = self.forward.to_dict()
        if self.backward is not None:
            data["backward"] = self.backward.to_dict()
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def validate(self, require_backward: bool = False) -> None:
        if require_backward and self.backward is None:
            raise ValueError(f"Operation '{self.name}' is missing backward timing")


@dataclass(frozen=True)
class OperationGroup:
    """
    Aggregates a set of OperationTiming instances (e.g. the GEMMs that form MHA).
    Provides convenience accessors for summary statistics.
    """

    name: str
    operations: Tuple[OperationTiming, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OperationGroup.name must be non-empty")
        if not self.operations:
            raise ValueError(f"OperationGroup '{self.name}' must contain at least one operation")
        for op in self.operations:
            if not isinstance(op, OperationTiming):
                raise TypeError("OperationGroup.operations must contain OperationTiming instances")

    def forward_compute_time(self) -> float:
        return sum(op.forward.compute_time for op in self.operations)

    def forward_comm_time(self) -> float:
        return sum(op.forward.comm_time for op in self.operations)

    def forward_comm_bytes(self) -> int:
        return sum(op.forward.comm_bytes for op in self.operations)

    def forward_total_time(self) -> float:
        return sum(op.total_forward_time() for op in self.operations)

    def backward_compute_time(self) -> float:
        total = 0.0
        for op in self.operations:
            if op.backward is None:
                raise RuntimeError(
                    f"OperationGroup '{self.name}' member '{op.name}' is missing backward timing"
                )
            total += op.backward.compute_time
        return total

    def backward_comm_time(self) -> float:
        total = 0.0
        for op in self.operations:
            if op.backward is None:
                raise RuntimeError(
                    f"OperationGroup '{self.name}' member '{op.name}' is missing backward timing"
                )
            total += op.backward.comm_time
        return total

    def backward_comm_bytes(self) -> int:
        total = 0
        for op in self.operations:
            if op.backward is None:
                raise RuntimeError(
                    f"OperationGroup '{self.name}' member '{op.name}' is missing backward timing"
                )
            total += op.backward.comm_bytes
        return total

    def backward_total_time(self) -> float:
        total = 0.0
        for op in self.operations:
            if op.backward is None:
                raise RuntimeError(
                    f"OperationGroup '{self.name}' member '{op.name}' is missing backward timing"
                )
            total += op.total_backward_time()
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": dict(self.metadata),
        }
