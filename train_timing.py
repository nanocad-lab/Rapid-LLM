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
import os
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Mapping, Sequence, Set
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
import simulate_train_graph as llm_simulation
from llm_execution import ExecutionMode, LLMExecutionDispatcher, apply_overlap_transforms
from simulate_train_graph import Graph
import llm_util
from memory_estimation import MemoryEstimator
from base_timing import TimeCalculation
from itertools import zip_longest  # for element-wise aggregation of memory access lists
from timing_model import CollectiveType, CommSpec, DirectionTiming, OperationTiming, OperationGroup
import yaml

def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}

    
class ParallelismMode(Enum):
    TENSOR = "tensor"
    TENSOR_SEQUENCE = "tensor_sequence"
    CONTEXT = "context"
    TENSOR_CONTEXT_HYBRID = "tensor_context_hybrid"
    SINGLE = "single"

# TODO: verify all 4 of these
SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT = 10
SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT = 20
GELU_FORWARD_FLOPS_PER_ELEMENT = 10
GELU_BACKWARD_FLOPS_PER_ELEMENT = 20
LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT = 7
LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT = 14
LAYER_NORM_FORWARD_MEM_ACCESSES = 2
LAYER_NORM_BACKWARD_MEM_ACCESSES = 4
SOFTMAX_FORWARD_FLOPS_PER_ELEMENT = 7 # exponentiation 4FLOPS, subtract max 1FLOPS, dropout 2FLOPS
SOFTMAX_FORWARD_MEM_ACCESSES = 4
SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT = 4 
SOFTMAX_BACKWARD_MEM_ACCESSES = 3

ALL_REDUCE = CollectiveType.ALL_REDUCE
ALL_GATHER = CollectiveType.ALL_GATHER
REDUCE_SCATTER = CollectiveType.REDUCE_SCATTER
ALL_TO_ALL = CollectiveType.ALL_TO_ALL
PIPELINE = CollectiveType.PIPELINE

# Map each parallelism mode to operation-level collective specs used across the
# metadata pipeline. Each spec records the collective kind, the participant
# scope (tp/cp/seq/etc.), and the interconnect label.
COMM_RULE_DEFAULT_KEY = "__default__"
COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, object]]]]
] = {
    ParallelismMode.TENSOR: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm1': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'qkv_proj': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'output_proj': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.TENSOR_SEQUENCE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None}, # <- dangerous
        'layernorm1': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'qkv_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'output_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.CONTEXT: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'},
        },
        'attention': {'forward': None, 'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'}},
        'output_proj': {'forward': None, 'backward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'}},
        'qkv_proj': {'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'}, 'backward': None},
    },
    ParallelismMode.TENSOR_CONTEXT_HYBRID: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {'forward': None, 'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'}},
        'output_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'cp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'},
        },
        'qkv_proj': {
            'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.SINGLE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
    },
}
# MoE-specific overrides for dispatcher collectives (router stays compute-only).
_MOE_ROUTER_RULE = {
    'forward': None,
    'backward': None,
}
_MOE_DISPATCH_RULE = {
    'forward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
    'backward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
}
_MOE_COMBINE_RULE = {
    'forward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
    'backward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
}
MOE_COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, object]]]]
] = {
    mode: {
        'router': _MOE_ROUTER_RULE,
        'moe_dispatch': _MOE_DISPATCH_RULE,
        'moe_combine': _MOE_COMBINE_RULE,
    }
    for mode in ParallelismMode
}
class GemmType(Enum):
    ATTENTION_SCORE = "attention_score"
    ATTENTION_OUTPUT = "attention_output"
    QKV = "qkv"
    OUT_PROJ = "out_proj"
    FFN1 = "ffn1"
    FFN2 = "ffn2"
    LINEAR_SOFTMAX = "linear_softmax"
    LAYER_NORM_1 = "layer_norm1"


@dataclass(frozen=True)
class GemmShardSpec:
    batch: int
    m: int
    k: int
    n: int
    shard_batch: int
    shard_m: int
    shard_k: int
    shard_n: int
    batch_scale: float

    def output_elements(self) -> int:
        return int(self.shard_batch * self.shard_m * self.shard_n)


class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
# Mode parameter
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )

          
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        overlap_cfg = getattr(getattr(hw_config, "network_layout", None), "overlap_config", None)
        self.tp_overlap = float(overlap_cfg.tp_overlap)
        self.tp_sp_overlap = float(overlap_cfg.tp_sp_overlap)
        self.cp_overlap = float(overlap_cfg.cp_overlap)
        self._generate_graphs = _env_flag("RAPID_VISUALIZE_GRAPHS")
        self.persist_astrasim_artifacts = _env_flag("RAPID_PERSIST_ASTRASIM_ARTIFACTS")
        self._debug_memory = _env_flag("RAPID_DEBUG_MEMORY")
        self._memory_breakdown_debug = None
        self.execution_mode = execution_mode
        self.pipeline_style_recompute = bool(self.full_recomputation)

        self.model_type = self.model.model_type
        self.tied_embeddings = getattr(self.model, "tied_embeddings", True)

        self.memory_capacity_exceeded = False
        self.memory_capacity_violation_gb = 0.0
        self.zero3_ephemeral_peak_bytes = 0.0
        self.pipeline_graph: Optional[Graph] = None
        self.pipeline_root: Optional[Any] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None
        self.transformer_graph: Optional[Graph] = None
        self.transformer_forward_root: Optional[Any] = None
        self.transformer_backward_root: Optional[Any] = None
        self.transformer_graph_moe: Optional[Graph] = None
        self.transformer_forward_root_moe: Optional[Any] = None
        self.transformer_backward_root_moe: Optional[Any] = None
        self.transformer_graph_no_dp: Optional[Graph] = None
        self.transformer_forward_root_no_dp: Optional[Any] = None
        self.transformer_backward_root_no_dp: Optional[Any] = None
        self.transformer_graph_moe_no_dp: Optional[Graph] = None
        self.transformer_forward_root_moe_no_dp: Optional[Any] = None
        self.transformer_backward_root_moe_no_dp: Optional[Any] = None
        self.transformer_analytical_time_forward: Optional[float] = None
        self.transformer_analytical_time_backward: Optional[float] = None
        self.transformer_analytical_time_backward_combined: Optional[float] = None
        self.transformer_astrasim_time_forward: Optional[float] = None
        self.transformer_astrasim_time_backward: Optional[float] = None
        self.transformer_astrasim_time_forward_moe: Optional[float] = None
        self.transformer_astrasim_time_backward_moe: Optional[float] = None
        self.transformer_astrasim_per_rank_forward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_forward_moe: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward_moe: Optional[List[float]] = None
        self.pipeline_astrasim_time: Optional[float] = None
        self.pipeline_astrasim_per_rank: Optional[List[float]] = None
        self.pipeline_graph_no_dp: Optional[Graph] = None
        self.pipeline_root_no_dp: Optional[Any] = None

    def _sequence_parallel_degree(self) -> int:
        """Return tensor-parallel degree used for sequence-parallel collectives.
            return 1 if no sequence parallelism is used.
            return tp if tp-sp is True and no context parallelism is used
            return cp if context parallelism is used
        """
        if self.tp_sp and self.cp == 1: #tensor parallelism only
            return self.tp
        elif self.cp > 1: #context parallelism or cp-tp hybrid parallelism
            return self.cp
        else: 
            return 1

    def get_parallelism_mode(self):
        if self.tp_sp and self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR_SEQUENCE
        elif self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR
        elif self.cp > 1 and self.tp == 1:
            return ParallelismMode.CONTEXT
        elif self.cp > 1 and self.tp > 1:
            return ParallelismMode.TENSOR_CONTEXT_HYBRID
        else:
            return ParallelismMode.SINGLE

    def is_moe_layer(self, layer_idx: int) -> bool:
        mask = getattr(self, "moe_layer_mask", None)
        if not mask:
            return False
        if layer_idx < 0 or layer_idx >= len(mask):
            return False
        return bool(mask[layer_idx])

    @property
    def experts_per_gpu(self) -> int:
        """
        Return the number of MoE experts assigned to each GPU assuming each dp group has all experts.
        """
        if not self.use_moe or self.moe_num_experts <= 1:
            return 0
        moe_ranks = self._moe_routing_group()
        if self.moe_num_experts % moe_ranks != 0:
            if self._moe_allow_padding() and _env_flag("RAPID_ALLOW_MOE_EXPERT_PADDING"):
                if not getattr(self, "_moe_expert_padding_warned", False):
                    warnings.warn(
                        "MoE expert count is not divisible by the routing group; "
                        "padding experts for inference. Set RAPID_ALLOW_MOE_EXPERT_PADDING=0 "
                        "to enforce divisibility.",
                        stacklevel=2,
                    )
                    self._moe_expert_padding_warned = True
                return max(1, math.ceil(self.moe_num_experts / moe_ranks))
            raise ValueError(
                "moe_num_experts must be divisible by the MoE routing group size. "
                f"moe_num_experts={self.moe_num_experts}, moe_group={moe_ranks}, "
                f"tp={self.tp}, ep={self.ep}."
            )
        return max(1, math.ceil(self.moe_num_experts / moe_ranks))

    @experts_per_gpu.setter
    def experts_per_gpu(self, value: int) -> None:
        # Preserve the base class assignment for debugging, but the getter drives usage.
        self._experts_per_gpu_base_value = value


    def _moe_tokens_owner(self, batch_size: int, seq_len: int) -> int:
        seq_divisor = max(1, int(self.cp)) if self.cp > 1 else 1
        return int(batch_size) * int(math.ceil(float(seq_len) / float(seq_divisor)))

    def _moe_tokens_dispatched(self, tokens_owner: int) -> int:
        return int(tokens_owner) * int(self.moe_top_k)

    def _moe_tokens_shared(self, tokens_owner: int) -> int:
        # TODO: Shared experts are replicated across EP for now.
        return int(tokens_owner) * int(self.n_shared_experts)

    def _moe_routing_group(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            return max(1, int(self.tp) * int(self.ep))
        return max(1, int(self.ep))

    def _moe_allow_padding(self) -> bool:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        return run_type == "inference"

    def _moe_tokens_local(self, tokens_dispatched: int) -> int:
        moe_group = self._moe_routing_group()
        return int(math.ceil(float(tokens_dispatched) / float(max(1, moe_group))))

    def _moe_routed_tokens_per_expert(
        self,
        batch_size: int,
        seq_len: int,
        *,
        allow_padding: bool = False,
    ) -> Tuple[int, int, int, int, int]:
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len)
        tokens_dispatched = self._moe_tokens_dispatched(tokens_owner)
        moe_group = self._moe_routing_group()
        if moe_group <= 0:
            moe_group = 1
        if allow_padding:
            tokens_dispatched = int(math.ceil(float(tokens_dispatched) / float(moe_group)) * moe_group)
        else:
            if tokens_dispatched % moe_group != 0:
                raise ValueError(
                    "MoE routed tokens must divide evenly across the MoE routing group for batched expert GEMMs. "
                    f"tokens_dispatched={tokens_dispatched}, moe_group={moe_group}, "
                    f"tp={self.tp}, ep={self.ep}, tokens_owner={tokens_owner}, "
                    f"batch_size={batch_size}, seq_len={seq_len}, top_k={self.moe_top_k}, cp={self.cp}."
                )
        tokens_local = tokens_dispatched // moe_group
        experts_per_rank = self.experts_per_gpu
        if experts_per_rank <= 0:
            raise ValueError(
                "MoE experts_per_rank must be positive when use_moe is enabled. "
                f"experts_per_rank={experts_per_rank}, moe_num_experts={self.moe_num_experts}, "
                f"tp={self.tp}, ep={self.ep}."
            )
        if allow_padding:
            if tokens_local % experts_per_rank != 0:
                tokens_local = int(
                    math.ceil(float(tokens_local) / float(experts_per_rank)) * experts_per_rank
                )
                tokens_dispatched = tokens_local * moe_group
        else:
            if tokens_local % experts_per_rank != 0:
                raise ValueError(
                    "MoE routed tokens per rank must divide evenly across experts for batched expert GEMMs. "
                    f"tokens_local={tokens_local}, experts_per_rank={experts_per_rank} "
                    f"(moe_num_experts={self.moe_num_experts}, moe_group={moe_group}). "
                    f"tokens_owner={tokens_owner}, tokens_dispatched={tokens_dispatched}, "
                    f"batch_size={batch_size}, seq_len={seq_len}, top_k={self.moe_top_k}, cp={self.cp}."
                )
        tokens_per_expert = tokens_local // experts_per_rank
        return tokens_owner, tokens_dispatched, tokens_local, experts_per_rank, tokens_per_expert

    def _param_stats_per_rank(
        self,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
    ) -> Tuple[float, float, float, float, float]:
        """Return detailed per-rank parameter counts used for ZeRO modeling."""

        tp = max(1, self.tp)
        pp = max(1, self.pp)

        ffn_proj_factor = 3 if llm_util.is_llama_style(self.model_type) else 2
        if llm_util.is_glm_style(self.model_type):
            _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                hidden_dim,
                self.num_heads,
                self.kv_heads,
                head_dim=getattr(self, "head_dim", None),
            )
            attention_params = (hidden_dim * (q_size + 2 * kv_size)) + (q_size * hidden_dim)
        else:
            attention_params = 4 * hidden_dim * hidden_dim
        transformer_param_layer = attention_params + intermediate_size * ffn_proj_factor * hidden_dim
        params_per_layer_per_rank = transformer_param_layer / tp

        total_transformer_params = params_per_layer_per_rank * self.num_layers
        if pp == 1:
            transformer_params_local = total_transformer_params
        else:
            layers_per_stage = math.ceil(self.num_layers / pp)
            transformer_params_local = params_per_layer_per_rank * layers_per_stage
            transformer_params_local = min(transformer_params_local, total_transformer_params)

        embedding_params = vocab_size * hidden_dim / tp
        output_params = 0.0 if self.tied_embeddings else embedding_params

        if pp == 1:
            total_params_per_rank = transformer_params_local + embedding_params + output_params
        else:
            # Pipeline splits embeddings/output across stages. Approximate their contribution by spreading across stages.
            total_params_per_rank = transformer_params_local + (embedding_params / pp) + (output_params / pp)

        max_layer_params = max(params_per_layer_per_rank, embedding_params, output_params)
        return (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params,
            output_params,
        )

    def get_kv_size_bytes(self) -> int:
        """Return the total size in bytes of the KV cache."""
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        total_elements = 2 * self.seq_len * self.micro_batch * head_dim * self.kv_heads
        return total_elements * self.precision.kv_cache

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astra":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astra", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode in (
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
            ExecutionMode.FULL_ASTRASIM_FLATTENED,
        ):
            return 'full'
        # Hybrid still uses the analytical pipeline, so keep the hybrid policy.
        return 'hybrid'

    @staticmethod
    def _expand_gemm_descriptor(gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        if len(gemm) == 3:
            return 1, gemm[0], gemm[1], gemm[2]
        if len(gemm) == 4:
            return gemm[0], gemm[1], gemm[2], gemm[3]
        raise ValueError(f"Unsupported GEMM descriptor length: {len(gemm)}")

    def _shard_gemm_descriptor(self, gemm: Tuple[int, ...], gemm_type: Optional[GemmType]) -> GemmShardSpec:
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for GEMM sharding")

        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        shard_batch = batch
        shard_m = m
        shard_k = k
        shard_n = n
        batch_scale = 1.0

        tp = max(1, int(self.tp))
        cp = max(1, int(self.cp))
        mode = self.get_parallelism_mode()

        if mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE):
            if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
                shard_batch = math.ceil(batch / tp)
                batch_scale = 1.0 / tp
            elif gemm_type in (GemmType.QKV, GemmType.FFN1):
                shard_n = math.ceil(n / tp)
            elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
                shard_k = math.ceil(k / tp)
            elif gemm_type == GemmType.LINEAR_SOFTMAX:
                shard_n = math.ceil(n / max(1, tp * cp))
            else:
                raise ValueError(f"Unsupported GEMM type for tensor parallelism: {gemm_type}")
        elif mode == ParallelismMode.CONTEXT:
            shard_m = math.ceil(m / cp)
            if gemm_type not in (
                GemmType.ATTENTION_SCORE,
                GemmType.ATTENTION_OUTPUT,
                GemmType.QKV,
                GemmType.OUT_PROJ,
                GemmType.FFN1,
                GemmType.FFN2,
                GemmType.LINEAR_SOFTMAX,
            ):
                raise ValueError(f"Unsupported GEMM type for context parallelism: {gemm_type}")
        elif mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            shard_m = math.ceil(m / cp)
            if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
                shard_batch = math.ceil(batch / tp)
                batch_scale = 1.0 / tp
            elif gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.LINEAR_SOFTMAX):
                shard_n = math.ceil(n / tp)
            elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
                shard_k = math.ceil(k / tp)
            else:
                raise ValueError(f"Unsupported GEMM type for hybrid parallelism: {gemm_type}")
        elif mode == ParallelismMode.SINGLE:
            pass
        else:
            raise ValueError(f"Unsupported parallelism mode: {mode}")

        return GemmShardSpec(
            batch=batch,
            m=m,
            k=k,
            n=n,
            shard_batch=shard_batch,
            shard_m=shard_m,
            shard_k=shard_k,
            shard_n=shard_n,
            batch_scale=batch_scale,
        )

    def _gemm_output_elements(self, gemm: Tuple[int, ...], gemm_type: Optional[GemmType]) -> int:
        spec = self._shard_gemm_descriptor(gemm, gemm_type)
        return spec.output_elements()

    @classmethod
    def _effective_dims(cls, gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        batch, m, k, n = cls._expand_gemm_descriptor(gemm) 
        return batch, batch * m, k, n
    def _ffn1_output_dim(self, intermediate_size: int) -> int:
        return 2 * intermediate_size if llm_util.is_llama_style(self.model_type) else intermediate_size
    # def sequence
    def get_tensor_reduction_time(
        self,
        total_bytes: int,
        kind: CollectiveType,
        name: str,
        participants: Optional[int] = None,
    ) -> float:
        """Return collective time for tensor-parallel reductions.
           receives total_bytes, which is the total size of the data to be reduced across all participants.
        """
        if not total_bytes:
            return 0.0

        if not participants:
            participants = int(self.tp)

        reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis="tp",
        )
        return reduction_time

    def flash_attention_kernel_forward(self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return time for flash attention kernel."""
        
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_dim // num_heads
        d = head_dim # gemm shape for one head is (seq_len, d) x (d, seq_len)
        q_size = num_heads * head_dim
        Bc = self.attention_tile_size # kv tile size
        Br = min(Bc, d) # q tile size
        Tr = math.ceil(shard_seq / Br) #number of q tiles
        Tc =  math.ceil(shard_seq / Bc) #number of kv tiles
        

        attention_forward_reduction_time = 0
        attention_size_f = 0
        # assuming key and value are preloaded into shared memory before attention computation
        load_kv_bytes = Bc * d * 2 * num_SMs * self.precision.activations #load key and value for one tile from HBM to SRAM
        initial_load_time = self.roofline(0, load_kv_bytes, "flash_attention_initial_load", mem_level=self.num_levels - 1) #assume key and value of one attention head is loaded from HBM to SRAM

        
        # attention score gemm
        load_q_bytes = Br * d * self.precision.activations #load query for one tile assuming k is already in shared memory
        attn_score_time_per_tile = self.get_gemm_time(Br, d, Bc, "attention_score_f",read_bytes_l2=load_q_bytes, flashattn_enable=True)[0] 
        attn_score_time = attn_score_time_per_tile * Tc * Tr #attention score gemm time for one head

        # Softmax time
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        
        # attention output gemm
        output_bytes = Br * d * self.precision.activations #load value for one tile S is already in shared memory
        attn_output_time_per_tile = self.get_gemm_time(Br, Bc, d, "attention_output", read_bytes_l2=output_bytes, write_bytes_l2=output_bytes, flashattn_enable=True)[0] 
        attn_output_time = attn_output_time_per_tile * Tc * Tr #attention output gemm time for one head
        
        
        attn_score_time *= batch_size * num_heads / max(1, self.tp) + self.O
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        attn_output_time *= batch_size * num_heads / max(1, self.tp) + self.O


        attention_forward_gemm_time = initial_load_time + attn_score_time + attn_scale_softmax_time + attn_output_time
        attention_forward_time = attention_forward_gemm_time + attention_forward_reduction_time

        # HBM traffic consists of only reading Q, K, V once and writing output once
        attention_mem = 2 * seq_len * (q_size + d * kv_heads) * self.precision.activations
        
        return attention_forward_time, attention_forward_gemm_time, attention_forward_reduction_time, attention_size_f, attention_mem
    
    
    def flash_attention_kernel_backward(self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return time for flash attention backward kernel."""
        
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_dim // num_heads
        d = head_dim # gemm shape for one head is (seq_len, d) x (d, seq_len)
        Bc = self.attention_tile_size # kv tile size
        Br = min(Bc, d) # q tile size
        Tr = math.ceil(shard_seq / Br) #number of q tiles
        Tc =  math.ceil(shard_seq / Bc) #number of kv tiles
        
        attention_backward_reduction_time = 0
        attention_size_b = 0
        recompute_time = 0
        # assuming key and value are preloaded into shared memory before attention backward computation
        load_kv_bytes = Bc * d * 2 * num_SMs * self.precision.activations #load key and value for one tile from HBM to SRAM
        initial_load_time = self.roofline(0, load_kv_bytes, "flash_attention_initial_load_b", mem_level=self.num_levels - 1) #assume key and value of one attention head is loaded from HBM to SRAM

        
        # attention score recompute
        load_q_bytes = Br * d * self.precision.activations #load query for one tile assuming k is already in shared memory
        attn_score_time_per_tile = self.get_gemm_time(Br, d, Bc, "attention_score_b",read_bytes_l2=load_q_bytes, flashattn_enable=True)[0] #recompute S = QK^T for backward [Br, Bc]
        attn_score_time = attn_score_time_per_tile * Tc * Tr #attention score gemm time for one head
        
        # Softmax recompute time 
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax_recompute", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        
        # dV dP
        load_o_bytes = Br * d * self.precision.activations #load output for one tile assuming v is already in shared memory
        act_dO_time_per_tile = self.get_gemm_time(Bc, Br, d, "attention_output_b", read_bytes_l2=load_o_bytes, flashattn_enable=True)[0] #dV = P^T * dO  [Bc, d]
        act_dP_time_per_tile = self.get_gemm_time(Br, d, Bc, "attention_score_b", read_bytes_l2=0, flashattn_enable=True)[0] #dP = dO * V^T [Br, Bc]
        act_dO_time = act_dO_time_per_tile * Tc * Tr #dV time for one head
        act_dP_time = act_dP_time_per_tile * Tc * Tr #dP time for one head
        
        # compute dS
        elements = Br * Bc
        flops = (SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT ) * elements 
        softmax_time_backward = self.roofline(flops, 1, "attention_scale_softmax_b", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        
        # dQ
        act_dQ_time_per_tile = self.get_gemm_time(Br, Bc, d, "attention_score_b", read_bytes_l2=0, flashattn_enable=True)[0] #dQ = dS * K [Br, d]
        act_dQ_time = act_dQ_time_per_tile * Tc * Tr #dQ time for one head
        
        
        attn_score_time *= batch_size * num_heads / max(1, self.tp) + self.O
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        act_dO_time *= batch_size * num_heads / max(1, self.tp) + self.O
        act_dP_time *= batch_size * num_heads / max(1, self.tp) + self.O
        softmax_time_backward *= batch_size * num_heads / max(1, self.tp)
        act_dQ_time *= batch_size * num_heads / max(1, self.tp) + self.O
        if self.full_recomputation:  #attention recompute is already included in full recomputation   
            recompute_time = 0
        else:
            recompute_time = attn_score_time + attn_scale_softmax_time #selective recomputation only recompute attention score and softmax
        attention_backward_gemm_time = initial_load_time + recompute_time + act_dO_time + act_dP_time + softmax_time_backward + act_dQ_time
        attention_backward_time = attention_backward_gemm_time + attention_backward_reduction_time
        
        
        attention_size_b = self.precision.grad_communication * batch_size * seq_len * hidden_dim * 2 / self.tp # weight gradient of K V need to be reduce scattered  *2 account for both attn key and value
        kind = REDUCE_SCATTER
        participants = self.cp
        axis_hint = "cp"
        if attention_size_b > 0:
            attention_backward_reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=attention_size_b,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label= "attention_backward_reduction",
            axis=axis_hint,
        )
           
        
        
    
        return attention_backward_time, attention_backward_gemm_time, attention_backward_reduction_time, attention_size_b
        
        
    @staticmethod
    def _normalize_gemm_type(gemm_type: Optional[GemmType]) -> Optional[GemmType]:
        if gemm_type is None or isinstance(gemm_type, GemmType):
            return gemm_type
        if isinstance(gemm_type, str):
            try:
                return GemmType(gemm_type)
            except ValueError as exc:
                raise ValueError(f"Unknown GEMM type: {gemm_type!r}") from exc
        raise TypeError(f"Unsupported gemm type specifier: {gemm_type!r}")

    def _grad_accum_time(self, elements: float, name: str) -> float:
        if elements <= 0:
            return 0.0
        if getattr(self, "gradient_accumulation_steps", 1) <= 1:
            return 0.0
        mem_bytes = float(elements) * (
            2.0 * float(self.precision.gradients) + float(self.precision.grad_communication)
        )
        time_s = self.roofline(0, mem_bytes, name=name, mem_level=self.num_levels - 1)
        overhead = getattr(self, "grad_acc_overhead", 0.0) or 0.0
        if overhead:
            time_s += float(overhead)
        return time_s

    def _grad_accum_elems_for_gemm(
        self,
        gemm_type: Optional[GemmType],
        *,
        k: int,
        n: int,
        shard_k: int,
        shard_n: int,
    ) -> int:
        if getattr(self, "gradient_accumulation_steps", 1) <= 1:
            return 0
        if gemm_type is None:
            return int(k * n)
        if gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.LINEAR_SOFTMAX):
            return int(k * shard_n)
        if gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
            return int(shard_k * n)
        return 0
    
    # assuming no context parallelism for now
    def parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_forward(gemm, name, gemm_type) # also return flops and mem accesses
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_forward(gemm, name, gemm_type) # also return flops and mem accesses
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        
    def parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_backward(gemm, name, gemm_type)

        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_backward(gemm, name, gemm_type)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        
    def single_gpu_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.get_gemm_time(m, k, n, name, disable_overhead=True)[0] * batch + self.O
        else :
            gemm_time,_,_, mem_accesses = self.get_gemm_time(m, k, n, name)
        return gemm_time, 0, 0, total_flops, mem_accesses
    
    def single_gpu_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            grad_time_act = self.get_gemm_time(m, k, n, name, disable_overhead=True)[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(k, m, n, name, disable_overhead=True)[0] * batch + self.O
        else :
            grad_time_act = self.get_gemm_time(m, n, k, name)[0]
            grad_time_wt = self.get_gemm_time(k, m, n, name)[0]
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=k,
            n=n,
            shard_k=k,
            shard_n=n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        return gemm_time, 0, 0
        
    def _tensor_context_hybrid_gemm_forward(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
    ) -> Tuple[float, float, float, float, Any]:
        """
        Megatron-LM style TPxCP hybrid forward GEMM behavior.

        Sharding:
        • CP shards the token (sequence) dimension M: shard_m = ceil(M / cp).
        • TP shards either K (row-wise) or N (column-wise), depending on gemm_type:
            - Column-wise (split N): QKV, FFN1, LINEAR_SOFTMAX 
            - Row-wise   (split K): OUT_PROJ, FFN2

        Per-op compute & communication:
        • ATTENTION_SCORE / ATTENTION_OUTPUT:
            -attention gemm shape is (shard_m, k) x (k, n) for each head, should scale with batch * num_heads / tp, number of heads is already multiplied in batch dimension
            - No extra collective here (K/V movement is modeled in QKV).

        • QKV (column-wise over N):
            -  CP all_gather(K,V) so each CP rank holds full-context K/V for attention.

        • OUT_PROJ, FFN2 (row-wise over K):
            - same as tensor parallelism forward gemm with row-wise sharding.:

        • FFN1 (column-wise over N):
            - No TP/CP collective in forward same as in tensor parallelism.

        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)

        participants = 0
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses: Any = []

        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid forward GEMM")
        
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        axis_hint = None

        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
                disable_overhead=True,
            )
            gemm_time = gemm_time * batch * shard_spec.batch_scale + self.O
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
                disable_overhead=True,
            )
            gemm_time = gemm_time * batch * shard_spec.batch_scale + self.O
        elif gemm_type == GemmType.QKV:  # column wise
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
            total_bytes = self.get_kv_size_bytes()  / self.tp # each tp group holds a tp shard of kv for each cp group
            kind = ALL_GATHER
            participants = self.cp # all gather K V for each cp group
            axis_hint = "cp"
        elif gemm_type == GemmType.OUT_PROJ:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * n)
            kind = REDUCE_SCATTER
            participants = self.tp # reduce scatter output activation for each tp group
            axis_hint = "tp"
        elif gemm_type == GemmType.FFN2:  # row wise
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * n)
            kind = REDUCE_SCATTER
            participants = self.tp  # reduce scatter output activation for each tp group
            axis_hint = "tp"
        elif gemm_type == GemmType.FFN1:  # column wise
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis=axis_hint,
        )
        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses
    def _tensor_context_hybrid_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Megatron-LM style TPxCP hybrid backward GEMM behavior.

        Sharding:
        • CP shards tokens (M): shard_m = ceil(M / cp).
        • TP shards either N (column-wise) or K (row-wise) per gemm_type:
            - Column-wise (split N): QKV, FFN1, LINEAR_SOFTMAX (vocab proj)
            - Row-wise   (split K): OUT_PROJ, FFN2
            - Attention inner GEMMs: per-head work scaled by 1/tp.

        Backward rules (compute + comm):
        • Column-wise (QKV / FFN1 / LINEAR_SOFTMAX):
            - dX = dY_i @ W_i^T  → shape [shard_m, K]  → TP reduction on dX (sum):
                use reduce-scatter on TP (keeps token sharding), 
            - dW_i = X^T @ dY_i  → local, no TP comm
            - (QKV only) in CP: dK, dV must be reduce-scattered on CP back to token owners.
                Bytes ≈ get_kv_size_bytes() / tp (TP shard per CP group).

        • ATTENTION_SCORE / ATTENTION_OUTPUT:
            - Backward GEMMs are local on token shard with 1/tp scaling across heads.
            - CP/TP collectives handled in QKV branch (for K/V only), thus no comm here.
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        # size_bytes = 0
        participants = 0
        total_bytes = 0
        reduction_time = 0

        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid backward GEMM")
        
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        axis_hint = None

        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
            )[0] * batch * shard_spec.batch_scale + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
            )[0] * batch * shard_spec.batch_scale + self.O
            total_bytes = self.precision.grad_communication * k * n * batch * 2 / self.tp # weight gradient of K V need to be reduce scattered  *2 account for both attn key and value
            kind = REDUCE_SCATTER
            participants = self.cp
            axis_hint = "cp"
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
            )[0] * batch * shard_spec.batch_scale
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0] * batch * shard_spec.batch_scale
        elif gemm_type == GemmType.QKV:  # column wise
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
            )[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_spec.shard_m * k)
            kind = REDUCE_SCATTER
            participants = self.tp
            axis_hint = "tp"
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
            total_bytes = self.get_kv_size_bytes() / self.tp
            kind = ALL_GATHER
            participants = self.cp
            axis_hint = "cp"
        elif gemm_type == GemmType.FFN2:  # row wise
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
        elif gemm_type == GemmType.FFN1:  # column wise
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
            )[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_spec.shard_m * k)
            kind = REDUCE_SCATTER
            participants = self.tp
            axis_hint = "tp"
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_spec.shard_m * shard_spec.shard_k) * self.cp * self.tp # in tp-cp hybrid parallelism, the linear softmax weight is sharded by both tp and cp
            kind = ALL_GATHER
            participants = self.cp * self.tp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis=axis_hint,
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Tensor-parallel forward GEMM behavior.

        • Multi-Head Attention (MHA):
          - With tensor parallelism, the attention computation is sharded along the head dimension,
            so each TP rank handles a subset of heads. No communication is needed for
            ATTENTION_SCORE / ATTENTION_OUTPUT GEMMs in the forward pass.
            attention gemm time is scaled by batch / tp

        • After attention, before feeding MLP:
          - If TP-only, we all-reduce to gather the full attention output.
          - If TP+SP (sequence parallel enabled), each TP rank keeps only a shard of the sequence
            after layernorm, so we use reduce-scatter instead of all-reduce.
            
        • Sharding rules by gemm_type:
          - QKV, FFN1: **column-wise sharding** (split along output dimension N).
            Each TP rank produces its local output columns independently - no communication needed.
          - OUT_PROJ, FFN2: **row-wise sharding** (split along input dimension K).
            Each TP rank computes partial sums that must be combined across ranks via
            all-reduce or reduce-scatter.
          - LINEAR_SOFTMAX (final logits projection): **column-wise sharding**.
            The output projection weight [hidden_dim, vocab_size] is split by vocab dimension.
            Each TP rank computes logits for its vocab slice, and results are all-gathered
        """
        tp_mode = self.get_parallelism_mode()
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_fwd = ALL_GATHER
        else:
            comm_kind_fwd = ALL_REDUCE if tp_mode == ParallelismMode.TENSOR else REDUCE_SCATTER
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        size_bytes = 0
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel forward GEMM")
        
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
                disable_overhead=True,
            )
            gemm_time = gemm_time * batch * shard_spec.batch_scale + self.O
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp
        elif gemm_type == GemmType.LINEAR_SOFTMAX: #assuming linear softmax is always column wise sharded
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
            
        if size_bytes > 0:
            total_bytes = size_bytes # we already has the total bytes for all reduce not bytes per rank
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_fwd, participants=participants, name=name)


        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses
    
    def _tensor_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        """
        Tensor-parallel backward GEMM behavior.

        We model the time for two backward GEMMs per op:
          • grad wrt activation (dX)
          • grad wrt weight (dW)

        • ATTENTION_SCORE / ATTENTION_OUTPUT
          - Per-rank work scales with batch/tp (each rank handles a subset of heads).

        • Column-wise sharded ops :
          - QKV, FFN1, and LINEAR_SOFTMAX:
            - Local backward GEMMs:
                dX:  [m, k]  via (dY_i @ W_i^T)
                dW:  [k, n/tp] via (X^T @ dY_i)
            - dX is a partial across ranks** → requires tensor reduction
              (all-reduce if TP-only; reduce-scatter if TP+SP).

        • Row-wise sharded ops :
          - OUT_PROJ, FFN2:
            - Local backward GEMMs:
                dX_i: [m, k/tp] via (dY @ W_i^T) - disjoint along K, no cross-rank sum
                dW_i: [k/tp, n] via (X_i^T @ dY)
            -  no tensor reduction on dX for row-wise in backward.

        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        seq_degree = self._sequence_parallel_degree()
        act_bytes = 0
        total_bytes = 0
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_bwd = ALL_REDUCE
        else:
            comm_kind_bwd = ALL_REDUCE if seq_degree == 1 else REDUCE_SCATTER
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel backward GEMM")

        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
            )[0] * batch * shard_spec.batch_scale + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
            )[0] * batch * shard_spec.batch_scale + self.O
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
            )[0]
            act_bytes = math.ceil(self.precision.grad_communication * m * k)
            participants = self.tp
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
            )[0]
            act_bytes = math.ceil(self.precision.grad_communication * m * k)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        reduction_time = 0
        if act_bytes > 0:
            total_bytes = act_bytes #total bytes for all reduce
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_bwd, participants=participants, name=name)


        return gemm_time, reduction_time, total_bytes
    def _context_parallelism_gemm_forward(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
    ) -> Tuple[float, float, float, float, Any]:
        """
        Megatron-LM style context-parallel (CP) forward GEMM behavior.

        • CP shards the token (sequence) dimension M across cp ranks:
            shard_m = ceil(M / cp). Each rank processes a disjoint subset of tokens.
        • Each GEMM is performed locally on the rank’s token slice.
        • Communication is required only when gathering K and V for attention,
          since every rank must hold the full set of keys and values to compute
          attention scores.

        """

        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses: Any = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel forward GEMM")
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
                disable_overhead=True,
            )
            gemm_time = gemm_time * batch + self.O
        elif gemm_type == GemmType.QKV:  # qkv gemm
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
            total_bytes = self.get_kv_size_bytes()
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN1, GemmType.FFN2):
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        if gemm_type == GemmType.QKV:
            kind = ALL_GATHER 
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label=name or "comm",
                axis="cp",
            )

        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses

    def _context_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:

        """
        Megatron-LM style context-parallel (CP) backward GEMM behavior.

        Communication rules in CP (sequence-parallel) backward:
        • QKV projection:
            - Forward did an all-gather(K, V) across cp ranks.
            - Backward must return gradients to token owners:
                    dK, dV → reduce-scatter over cp ranks.
        • Output projection:
            - K V need to be gathered again to compute activation gradients.
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel backward GEMM")
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type == GemmType.ATTENTION_SCORE:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
            )[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
            )[0] * batch + self.O
            total_bytes = self.precision.grad_communication * k * n * batch * 2 # account for both K and V
            kind = REDUCE_SCATTER
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
            )[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
            )[0] * batch + self.O
        elif gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.FFN2):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
            )[0]
            total_bytes = self.get_kv_size_bytes()
            kind = ALL_GATHER
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label=name or "comm",
                axis="cp",
            )
        return gemm_time, reduction_time, total_bytes

    def _scale_mem_accesses(self, mem_accesses: Any, scale: int) -> Dict[str, float]:
        if scale <= 0:
            return {}
        levels = self._mem_levels(mem_accesses)
        if scale == 1:
            return levels
        return {k: v * scale for k, v in levels.items()}

    def _fused_batched_comm_time(
        self,
        total_bytes: int,
        gemm_type: Optional[GemmType],
        *,
        direction: str,
    ) -> float:
        total_bytes = int(math.ceil(float(total_bytes or 0.0)))
        if total_bytes <= 0:
            return 0.0
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            return 0.0
        mode = self.get_parallelism_mode()
        kind = None
        participants = None
        if mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE):
            if direction == "forward" and gemm_type == GemmType.FFN2:
                kind = ALL_REDUCE if mode == ParallelismMode.TENSOR else REDUCE_SCATTER
                participants = self.tp
            elif direction == "backward" and gemm_type == GemmType.FFN1:
                kind = ALL_REDUCE if mode == ParallelismMode.TENSOR else REDUCE_SCATTER
                participants = self.tp
        elif mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            if direction == "forward" and gemm_type == GemmType.FFN2:
                kind = REDUCE_SCATTER
                participants = self.tp
            elif direction == "backward" and gemm_type == GemmType.FFN1:
                kind = REDUCE_SCATTER
                participants = self.tp
        if not kind or not participants or participants <= 1:
            return 0.0
        return self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=f"batched_{gemm_type.value}_{direction}",
            axis="tp",
        )

    def _batched_gemm_forward_compute(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        use_tp: bool = True,
    ) -> Tuple[float, float, int, float, Dict[str, float]]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, 0, 0.0, {}
        if use_tp:
            result = self.parallelism_gemm_forward(
                (m, k, n),
                name,
                gemm_type=gemm_type,
            )
            if len(result) != 5:
                raise ValueError(f"Unsupported parallelism_gemm_forward return length: {len(result)}")
            per_time, per_comm_time, per_comm_bytes, per_flops, per_mem = result
        else:
            per_time, per_comm_time, per_comm_bytes, per_flops, per_mem = self.single_gpu_gemm_forward(
                (m, k, n),
                name,
                gemm_type=gemm_type,
            )
        scale = int(batch)
        comm_bytes = int(math.ceil(float(per_comm_bytes or 0.0))) * scale
        comm_time = per_comm_time * scale
        if use_tp and comm_bytes and scale > 1:
            fused_time = self._fused_batched_comm_time(comm_bytes, gemm_type, direction="forward")
            if fused_time:
                comm_time = fused_time
        return (
            per_time * scale,
            comm_time,
            comm_bytes,
            per_flops * scale,
            self._scale_mem_accesses(per_mem, scale),
        )

    def _batched_gemm_backward_compute(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        use_tp: bool = True,
    ) -> Tuple[float, float, int]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, 0
        if use_tp:
            per_time, per_comm_time, per_comm_bytes = self.parallelism_gemm_backward(
                (m, k, n),
                name,
                gemm_type=gemm_type,
            )
        else:
            per_time, per_comm_time, per_comm_bytes = self.single_gpu_gemm_backward(
                (m, k, n),
                name,
                gemm_type=gemm_type,
            )
        scale = int(batch)
        comm_bytes = int(math.ceil(float(per_comm_bytes or 0.0))) * scale
        comm_time = per_comm_time * scale
        if use_tp and comm_bytes and scale > 1:
            fused_time = self._fused_batched_comm_time(comm_bytes, gemm_type, direction="backward")
            if fused_time:
                comm_time = fused_time
        return per_time * scale, comm_time, comm_bytes

    def get_moe_ffn_f(
        self,
        gemm,
        name,
        gemm_type: Optional[GemmType] = None,
        *,
        batch_size: int,
        seq_len: int,
        allow_padding: Optional[bool] = None,
    ):
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm)
        if allow_padding is None:
            allow_padding = self._moe_allow_padding()
        (
            tokens_owner,
            tokens_dispatched,
            tokens_local,
            experts_per_rank,
            tokens_per_expert,
        ) = self._moe_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
        tokens_shared = self._moe_tokens_shared(tokens_owner)

        m_per_expert = tokens_per_expert * max(1, int(self.cp))
        gemm_override = (experts_per_rank, m_per_expert, k, n)
        use_tp_sharded = bool(getattr(self, "tp_ep", True))
        gemm_time, tp_comm_time, tp_comm_bytes, total_flops, mem_accesses_total = self._batched_gemm_forward_compute(
            gemm_override,
            name,
            gemm_type=gemm_type,
            use_tp=use_tp_sharded,
        )

        if tokens_shared > 0:
            shared_m = tokens_owner * max(1, int(self.cp))
            shared_name = f"{name}_shared" if name else "moe_ffn_shared"
            gemm_override_shared = (int(self.n_shared_experts), shared_m, k, n)
            (
                shared_gemm_time,
                shared_comm_time,
                shared_comm_bytes,
                shared_flops,
                mem_accesses_shared,
            ) = self._batched_gemm_forward_compute(
                gemm_override_shared,
                shared_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += shared_gemm_time
            tp_comm_time += shared_comm_time
            tp_comm_bytes += shared_comm_bytes
            total_flops += shared_flops
            mem_accesses_total = self._combine_mem(mem_accesses_total, mem_accesses_shared)

        return gemm_time, tp_comm_time, tp_comm_bytes, total_flops, mem_accesses_total

    def get_moe_ffn_b(
        self,
        gemm,
        name,
        gemm_type,
        *,
        batch_size: int,
        seq_len: int,
        allow_padding: Optional[bool] = None,
    ):
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm)
        if allow_padding is None:
            allow_padding = self._moe_allow_padding()
        (
            tokens_owner,
            tokens_dispatched,
            tokens_local,
            experts_per_rank,
            tokens_per_expert,
        ) = self._moe_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
        tokens_shared = self._moe_tokens_shared(tokens_owner)

        m_per_expert = tokens_per_expert * max(1, int(self.cp))
        gemm_override = (experts_per_rank, m_per_expert, k, n)
        use_tp_sharded = bool(getattr(self, "tp_ep", True))
        gemm_time, tp_comm_time, tp_comm_bytes = self._batched_gemm_backward_compute(
            gemm_override,
            name,
            gemm_type=gemm_type,
            use_tp=use_tp_sharded,
        )
        if tokens_shared > 0:
            shared_m = tokens_owner * max(1, int(self.cp))
            shared_name = f"{name}_shared" if name else "moe_ffn_shared"
            gemm_override_shared = (int(self.n_shared_experts), shared_m, k, n)
            shared_gemm_time, shared_comm_time, shared_comm_bytes = self._batched_gemm_backward_compute(
                gemm_override_shared,
                shared_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += shared_gemm_time
            tp_comm_time += shared_comm_time
            tp_comm_bytes += shared_comm_bytes

        return gemm_time, tp_comm_time, tp_comm_bytes
        

                
    def get_embedding_f(self, vocab_size, seq_len, hidden_dim):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = vocab_size * hidden_dim * self.precision.activations + seq_len * batch * hidden_dim * self.precision.activations
        embedding_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_f",
            mem_level=self.num_levels - 1,
        ) + self.O
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        return embedding_time + embedding_transfer_time, embedding_mem

    def get_linear_softmax_f(self, gemm):
        """Estimate time for final projection + softmax forward.
            assuming linear softmax gemm always use tensor parallelism sharded by vocab dimension
        """
        _, effective_m, k, n = self._effective_dims(gemm)

        # Previous TP-aware path kept comm time here:
        # gemm_time, reduction_time, size_bytes, _, _ = self._tensor_parallelism_gemm_forward(
        #     gemm, "linear_softmax_f", gemm_type=GemmType.LINEAR_SOFTMAX
        # )
        # Linear softmax is modeled as running on a single device within each TP group.
        gemm_time, _, _, _, _ = self.single_gpu_gemm_forward(gemm, "linear_softmax_f", gemm_type=GemmType.LINEAR_SOFTMAX)

            
        elements = effective_m * n / (self.tp * self.cp) # each tp-cp group holds a shard of the vocab dimension
        point_flop = elements * SOFTMAX_FORWARD_FLOPS_PER_ELEMENT
        point_mem = self.precision.activations * elements * SOFTMAX_FORWARD_MEM_ACCESSES 
        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-f",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (f) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (f) point_time: {:,}\n".format(point_time))

        return gemm_time + point_time, point_mem
    
    def get_linear_softmax_b(self, gemm):


        _, effective_m, k, n = self._effective_dims(gemm)

        # Previous TP-aware path kept comm time here:
        # gemm_time, reduction_time, size_bytes = self._tensor_parallelism_gemm_backward(
        #     gemm, "linear_softmax_b", gemm_type=GemmType.LINEAR_SOFTMAX
        # )
        # Linear softmax is modeled as running on a single device within each TP group.
        gemm_time, _, _ = self.single_gpu_gemm_backward(gemm, "linear_softmax_b", gemm_type=GemmType.LINEAR_SOFTMAX)
        elements = effective_m * n / (self.tp * self.cp) # each tp-cp group holds a shard of the vocab dimension
        point_flop = elements * SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT
        # TODO:
        # same here, unsure if should be precision.activations or precision.stats
        point_mem = self.precision.activations * elements * SOFTMAX_BACKWARD_MEM_ACCESSES

        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (b) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (b) point_time: {:,}\n".format(point_time))

        return gemm_time + point_time
    def get_scale_softmax_f(self, gemm):
        """
        Estimate time for scale + softmax forward.
        total elements is divided by tp and cp since in tp-cp hybrid parallelism,
        """
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)  # +1 for scaling
        mem = self.precision.activations * elements * (SOFTMAX_FORWARD_MEM_ACCESSES )  

        time = self.roofline(
            flops,
            mem,
            name="pointwise-scale-softmax-f",
            mem_level=self.num_levels - 1,
        ) + self.O

        return time
    
    def get_scale_softmax_b(self, gemm):
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT + 1)  # +1 for scaling
        mem = self.precision.activations * elements * (SOFTMAX_BACKWARD_MEM_ACCESSES)  

        time = self.roofline(
            flops,
            mem,
            name="pointwise-scale_softmax-b",
            mem_level=self.num_levels - 1,
        ) +  self.O


        if self.debug:
            print(
                "Scale Softmax (b) flop: {:,}, mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Scale Softmax(b) time: {:,}".format(time))


        return time 
        
    def get_residual_f(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements_m = m
        mode = self.get_parallelism_mode()
        if mode in (
            ParallelismMode.TENSOR_SEQUENCE,
            ParallelismMode.CONTEXT,
            ParallelismMode.TENSOR_CONTEXT_HYBRID,
        ):
            seq_degree = max(1, int(self._sequence_parallel_degree()))
            elements_m = math.ceil(m / seq_degree)
        elements = batch * elements_m * n

        flops = 2 * elements  # add + bias
        mem = self.precision.activations * elements * 3  # read main, read residual, write out
        time = self.roofline(flops, mem, name="pointwise-residual-f", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (f) time: {:,}\n".format(time))

        return time

    def get_residual_b(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements_m = m
        mode = self.get_parallelism_mode()
        if mode in (
            ParallelismMode.TENSOR_SEQUENCE,
            ParallelismMode.CONTEXT,
            ParallelismMode.TENSOR_CONTEXT_HYBRID,
        ):
            seq_degree = max(1, int(self._sequence_parallel_degree()))
            elements_m = math.ceil(m / seq_degree)
        elements = batch * elements_m * n

        flops = elements  # dL/dx = dL/dy passthrough
        mem = self.precision.gradients * elements * 3  # read grad, read forward residual, write grad
        time = self.roofline(flops, mem, name="pointwise-residual-b", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (b) time: {:,}\n".format(time))

        return time

    def get_gelu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * hidden * 2  # read, write

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-f", mem_level=self.num_levels - 1) + 2 * self.O

        if self.debug:
            print(
                "GELU (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (f) time: {:,}\n".format(time))

        return time
    def get_gelu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.gradients * elements * hidden * 3  # read grad, read forward, write grad

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-b", mem_level=self.num_levels - 1) + 3 * self.O

        if self.debug:
            print(
                "GELU (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (b) time: {:,}\n".format(time))
        return time

    def get_swiglu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT + gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        writes = gate_hidden  # SwiGLU output
        mem_bytes = self.precision.activations * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-f", mem_level=self.num_levels - 1) + 2 * self.O
        if self.debug:
            print(
                "SwiGLU (f) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (f) time: {:,}\n".format(time))
        return time

    def get_swiglu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT + 2 * gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        reads += gate_hidden  # upstream gradient
        writes = 2 * gate_hidden  # gradients for gate and up projections
        mem_bytes = self.precision.gradients * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-b", mem_level=self.num_levels - 1) + 3 * self.O
        if self.debug:
            print(
                "SwiGLU (b) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (b) time: {:,}\n".format(time))
        return time

    def get_layernorm_f(self, batch, seq_len, d_model, comm_after=False):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        elif tp_mode == ParallelismMode.TENSOR_SEQUENCE:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        else:
            elements = batch * seq_len * d_model
        compute_flops = elements * LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.stats * elements * LAYER_NORM_FORWARD_MEM_ACCESSES
        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-f",
            mem_level=self.num_levels - 1,
        ) + 3 * self.O
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):  # all-gather after layernorm
            per_rank_bytes = self.precision.stats * elements
            total_bytes = int(math.ceil(per_rank_bytes))
            reduction_time = self.network_model.collective(
                kind=ALL_GATHER,
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label="layernorm_f_all_gather",
                axis="tp",
            )
        else:
            reduction_time = 0.0
            total_bytes = 0

        return compute_time, reduction_time, total_bytes
    
    

    def get_layernorm_b(self, batch, seq_len, d_model, type = Optional):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        else:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.stats * elements * LAYER_NORM_BACKWARD_MEM_ACCESSES

        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O
        compute_time += self._grad_accum_time(2 * d_model, "layernorm_grad_accum")
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID) :
            per_rank_bytes = self.precision.grad_communication * elements
            total_bytes = int(math.ceil(per_rank_bytes))
            reduction_time = self.network_model.collective(
                kind=ALL_GATHER,
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label="layernorm_b_all_gather",
                axis="tp",
            )

        else:
            reduction_time = 0.0
            total_bytes = 0


        return compute_time, reduction_time, total_bytes
    def get_router_f(self, gemm_router, gemm_ffn1, *, batch_size: int, seq_len: int):
        #TODO router overhead for moe is very minimal, can be ignored, implemented for completeness
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len)
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm_router)
        gemm_time = self.get_gemm_time(tokens_owner, k, n, "router_f")[0]
        elements = tokens_owner * n
        flops = elements * SOFTMAX_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * SOFTMAX_FORWARD_MEM_ACCESSES  

        compute_time = gemm_time + self.roofline(flops, mem_bytes, name="pointwise-router-f", mem_level=self.num_levels - 1) + self.O

        return compute_time, 0.0, 0
    def get_router_b(self, gemm_router, gemm_ffn1, *, batch_size: int, seq_len: int):
        #TODO router overhead for moe is very minimal, can be ignored, implemented for completeness
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len)
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm_router)

        grad_time_act = self.get_gemm_time(tokens_owner, n, k, "router_b")[0]
        grad_time_wt = self.get_gemm_time(k, tokens_owner, n, "router_b")[0]
        gemm_time = grad_time_act + grad_time_wt

        elements = tokens_owner * n
        flops = elements * SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * SOFTMAX_BACKWARD_MEM_ACCESSES

        compute_time = gemm_time + self.roofline(flops, mem_bytes, name="pointwise-router-b", mem_level=self.num_levels - 1) + self.O
        return compute_time, 0.0, 0
    
    def get_embedding_b(self, vocab_size, seq_len, hidden_dim):
        batch = self._effective_transformer_batch()
        embedding_mem = vocab_size * hidden_dim * self.precision.gradients + seq_len * batch * hidden_dim * self.precision.gradients
        embedding_mem_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_b",
            mem_level=self.num_levels - 1,
        ) + self.O
        embedding_params = math.ceil((vocab_size * hidden_dim) / max(1, self.tp))
        embedding_mem_time += self._grad_accum_time(embedding_params, "embedding_grad_accum")

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    

    def get_inter_layer_comm_latency_llm(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.pp > 1:
            w_size = self.precision.activations * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.links["pp"].bandwidth + self.links["pp"].latency
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer", mem_level=self.num_levels - 1)
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size

    def get_data_parallel_reduction_sizes(self, d, intermediate_size, *, moe: bool = False):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return 0

        # Calculate sizes only
        if llm_util.is_glm_style(self.model_type):
            _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                d,
                self.num_heads,
                self.kv_heads,
                head_dim=getattr(self, "head_dim", None),
            )
            qkv_params = d * (q_size + 2 * kv_size)
            output_params = q_size * d
        else:
            qkv_params = d * 3 * d
            output_params = d * d
        qkv_size = math.ceil(self.precision.grad_communication * qkv_params)
        output_size = math.ceil(self.precision.grad_communication * output_params)
        if not moe:
            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            ffn1_size = math.ceil(self.precision.grad_communication * ffn1_dim * d)
            ffn2_size = math.ceil(self.precision.grad_communication * intermediate_size * d)
            total_size = qkv_size + output_size + ffn1_size + ffn2_size
            return total_size

        ffn_proj_factor = 3 if llm_util.is_llama_style(self.model_type) else 2
        expert_param_size = ffn_proj_factor * intermediate_size * d
        tp = max(1, int(self.tp))
        moe_group = max(1, int(self._moe_routing_group()))
        routed_experts_per_rank = self.moe_num_experts / moe_group
        shared_experts_per_rank = self.n_shared_experts
        use_tp_sharded = bool(getattr(self, "tp_ep", True))
        if use_tp_sharded:
            routed_params_per_rank = (expert_param_size / tp) * routed_experts_per_rank
        else:
            routed_params_per_rank = expert_param_size * routed_experts_per_rank
        if use_tp_sharded:
            shared_params_per_rank = (expert_param_size / tp) * shared_experts_per_rank
        else:
            shared_params_per_rank = expert_param_size * shared_experts_per_rank
        expert_params_per_rank = routed_params_per_rank + shared_params_per_rank
        router_params_per_rank = d * self.moe_num_experts
        moe_param_size = math.ceil(
            self.precision.grad_communication * (expert_params_per_rank + router_params_per_rank)
        )
        total_size = qkv_size + output_size + moe_param_size
        return total_size

    def get_data_parallel_reduction_llm(self, d, intermediate_size, *, moe: bool = False):
        """Return apply_grad compute time per rank (no communication), honoring ZeRO sharding."""
        if llm_util.is_glm_style(self.model_type):
            _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                d,
                self.num_heads,
                self.kv_heads,
                head_dim=getattr(self, "head_dim", None),
            )
            qkv_params = d * (q_size + 2 * kv_size)
            output_params = q_size * d
        else:
            qkv_params = d * 3 * d
            output_params = d * d
        apply_grad_time = self.apply_grad(int(qkv_params)) # QKV
        apply_grad_time += self.apply_grad(int(output_params)) # Output
        if not moe:
            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            apply_grad_time += self.apply_grad(int(ffn1_dim * d)) # FFN1
            apply_grad_time += self.apply_grad(int(intermediate_size * d)) # FFN2
        else:
            ffn_proj_factor = 3 if llm_util.is_llama_style(self.model_type) else 2
            expert_param_size = ffn_proj_factor * intermediate_size * d
            tp = max(1, int(self.tp))
            moe_group = max(1, int(self._moe_routing_group()))
            routed_experts_per_rank = self.moe_num_experts / moe_group
            shared_experts_per_rank = self.n_shared_experts
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            if use_tp_sharded:
                routed_params_per_rank = (expert_param_size / tp) * routed_experts_per_rank
            else:
                routed_params_per_rank = expert_param_size * routed_experts_per_rank
            if use_tp_sharded:
                shared_params_per_rank = (expert_param_size / tp) * shared_experts_per_rank
            else:
                shared_params_per_rank = expert_param_size * shared_experts_per_rank
            expert_params_per_rank = routed_params_per_rank + shared_params_per_rank
            apply_grad_time += self.apply_grad(int(expert_params_per_rank))
            router_params_per_rank = d * self.moe_num_experts
            apply_grad_time += self.apply_grad(int(router_params_per_rank)) # Router

        grad_shard = self.dp if (self.zero_stage >= 2 and self.dp > 1) else 1
        if grad_shard > 1:
            apply_grad_time /= grad_shard

        return apply_grad_time
    
    def _combine_mem(self, *args):
            combined = {}
            for d in args:
                mem_levels = self._mem_levels(d)
                for k, v in mem_levels.items():
                    combined[k] = combined.get(k, 0.0) + (v or 0.0)
            return combined
        
    def _mem_levels(self, arr):
        # Accept both dict-like and sequence-like inputs since GEMM helpers return lists while
        # aggregated OperationTiming instances already carry dicts. MappingABC/SequenceABC cover
        # any object implementing the mapping/sequence protocol (e.g., dict, defaultdict, numpy arrays).
        if isinstance(arr, MappingABC):
            return {str(k): float(v) for k, v in arr.items()}
        if isinstance(arr, (int, float)):
            return {f"L{i}": (float(arr) if i == self.num_levels - 1 else 0.0) for i in range(self.num_levels)}
        if isinstance(arr, SequenceABC) and not isinstance(arr, (str, bytes)):
            return {f"L{i}": float(v) for i, v in enumerate(arr)}
        return {}

    # TODO TODO:
    # we need a significant refactor here. The comm sizes are ingested in a weird way and never used. Instead we use old precomputed sizes.
    # FIX at some point!
    def compute_all_gemm_and_node_times(
        self,
        batch_size,
        vocab_size,
        hidden_dim,
        seq_len,
        num_heads,
        kv_heads,
        intermediate_size,
        num_SMs,
        use_moe_override: Optional[bool] = None,
    ):
        """Compute latency for all GEMM operations and node breakdown times."""

        use_moe_layer = self.use_moe if use_moe_override is None else bool(use_moe_override)
        gemm_ctx = self
        if use_moe_layer != self.use_moe:
            from types import SimpleNamespace

            gemm_ctx = SimpleNamespace(
                use_moe=use_moe_layer,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
            )

        gemm_shapes = llm_util.process_gemm_shapes(
            gemm_ctx,
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            kv_heads,
            intermediate_size,
            vocab_size,
        )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        gemm_linear = gemm_shapes["linear"]
        gemm_router = gemm_shapes["router"]

        transformer_timings: Dict[str, OperationTiming] = {}
        moe_tokens_owner: Optional[int] = None
        moe_tokens_dispatched: Optional[int] = None
        moe_tokens_local: Optional[int] = None
        moe_tokens_shared: Optional[int] = None

        def _extract_forward(ret: Sequence[Any]) -> Tuple[float, float, float, float, Any]:
            if len(ret) == 5:
                time, reduction, size, flops, mem = ret
                return time, reduction, size, flops, mem
            if len(ret) == 3:
                time, reduction, size = ret
                return time, reduction, size, 0.0, []
            raise ValueError(f"Unsupported return length: {len(ret)}")

        # QKV
        qkv_proj_gemm_f, qkv_proj_reduction_f, qkv_proj_size_f, qkv_proj_flops_f, qkv_proj_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_qkv_proj, "qkv_projection_f", gemm_type=GemmType.QKV)
        )
        qkv_proj_gemm_b, qkv_proj_reduction_b, qkv_proj_size_b = self.parallelism_gemm_backward(
            gemm_qkv_proj, "qkv_projection_b", gemm_type=GemmType.QKV
        )
        qkv_forward = DirectionTiming(
            compute_time=qkv_proj_gemm_f,
            comm_time=qkv_proj_reduction_f,
            comm_bytes=int(qkv_proj_size_f or 0),
            flops=qkv_proj_flops_f,
            memory_accesses=dict(self._mem_levels(qkv_proj_mem_f)),
        )
        qkv_backward = DirectionTiming(
            compute_time=qkv_proj_gemm_b,
            comm_time=qkv_proj_reduction_b,
            comm_bytes=int(qkv_proj_size_b or 0),
        )
        transformer_timings["qkv_proj"] = OperationTiming("qkv_proj", forward=qkv_forward, backward=qkv_backward)

        if not self.flash_attention:
            attn_score_gemm_f, attn_score_reduction_f, attn_score_size_f, attn_score_flops_f, attn_score_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_score, "attention_score_f", gemm_type=GemmType.ATTENTION_SCORE)
            )
            attn_score_gemm_b, attn_score_reduction_b, attn_score_size_b = self.parallelism_gemm_backward(
                gemm_attention_score, "attention_score_b", gemm_type=GemmType.ATTENTION_SCORE
            )
            attn_score_forward = DirectionTiming(
                compute_time=attn_score_gemm_f,
                comm_time=attn_score_reduction_f,
                comm_bytes=int(attn_score_size_f or 0),
                flops=attn_score_flops_f,
                memory_accesses=dict(self._mem_levels(attn_score_mem_f)),
            )
            attn_score_backward = DirectionTiming(
                compute_time=attn_score_gemm_b,
                comm_time=attn_score_reduction_b,
                comm_bytes=int(attn_score_size_b or 0),
            )
            transformer_timings["attention_score"] = OperationTiming(
                "attention_score",
                forward=attn_score_forward,
                backward=attn_score_backward,
            )
            
            
            attention_scale_softmax_f = self.get_scale_softmax_f(gemm=gemm_attention_score)
            attention_scale_softmax_b = self.get_scale_softmax_b(gemm=gemm_attention_score)
            
            transformer_timings["attention_scale_softmax"] = OperationTiming(
                name="attention_scale_softmax",
                forward=DirectionTiming(
                    compute_time=attention_scale_softmax_f,
                    comm_time=0.0,
                    comm_bytes=0,
                ),
                backward=DirectionTiming(
                    compute_time=attention_scale_softmax_b,
                    comm_time=0.0,
                    comm_bytes=0,
                ),
            )

            attn_out_gemm_f, attn_out_reduction_f, attn_out_size_f, attn_out_flops_f, attn_out_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_output, "attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT)
            )
            attn_out_gemm_b, attn_out_reduction_b, attn_out_size_b = self.parallelism_gemm_backward(
                gemm_attention_output, "attention_output_b", gemm_type=GemmType.ATTENTION_OUTPUT
            )
            attn_out_forward = DirectionTiming(
                compute_time=attn_out_gemm_f,
                comm_time=attn_out_reduction_f,
                comm_bytes=int(attn_out_size_f or 0),
                flops=attn_out_flops_f,
                memory_accesses=dict(self._mem_levels(attn_out_mem_f)),
            )
            attn_out_backward = DirectionTiming(
                compute_time=attn_out_gemm_b,
                comm_time=attn_out_reduction_b,
                comm_bytes=int(attn_out_size_b or 0),
            )
            transformer_timings["attention_output"] = OperationTiming(
                "attention_output",
                forward=attn_out_forward,
                backward=attn_out_backward,
            )

            attention_forward_compute = (
                attn_score_forward.compute_time
                + transformer_timings["attention_scale_softmax"].forward.compute_time
                + attn_out_forward.compute_time
            )
            attention_forward_comm = attn_score_forward.comm_time + attn_out_forward.comm_time
            attention_backward_compute = (
                attn_score_backward.compute_time
                + transformer_timings["attention_scale_softmax"].backward.compute_time
                + attn_out_backward.compute_time
            )
            attention_backward_comm = attn_score_backward.comm_time + attn_out_backward.comm_time
            attention_comm_bytes_f = attn_score_size_f + attn_out_size_f
            attention_comm_bytes_b = attn_score_size_b + attn_out_size_b
            attention_mem = self._combine_mem(attn_score_mem_f, attn_out_mem_f)
            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=DirectionTiming(
                    compute_time=attention_forward_compute,
                    comm_time=attention_forward_comm,
                    comm_bytes=int(attention_comm_bytes_f or 0),
                    memory_accesses=dict(self._mem_levels(attention_mem)),
                    flops=(attn_score_flops_f + attn_out_flops_f),
                ),
                backward=DirectionTiming(
                    compute_time=attention_backward_compute,
                    comm_time=attention_backward_comm,
                    comm_bytes=int(attention_comm_bytes_b or 0),
                ),
            )
        else:
            attention_f, attention_gemm_f, attention_reduction_f, attention_size_f, attention_mem = self.flash_attention_kernel_forward(
                batch_size,
                hidden_dim,
                seq_len,
                num_heads,
                kv_heads,
                num_SMs,
            )
            attention_b, attention_gemm_b, attention_reduction_b, attention_size_b = self.flash_attention_kernel_backward(
                batch_size,
                hidden_dim,
                seq_len,
                num_heads,
                kv_heads,
                num_SMs,
            )

            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=DirectionTiming(
                    compute_time=attention_gemm_f,
                    comm_time=attention_reduction_f,
                    comm_bytes=int(attention_size_f or 0),
                    memory_accesses=dict(self._mem_levels(attention_mem)),
                ),
                backward=DirectionTiming(
                    compute_time=attention_gemm_b,
                    comm_time=attention_reduction_b,
                    comm_bytes=int(attention_size_b or 0),
                ),
            )

        out_proj_gemm_f, out_proj_reduction_f, out_proj_size_f, out_proj_flops_f, out_proj_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_output_proj, "output_projection_f", gemm_type=GemmType.OUT_PROJ)
        )
        out_proj_gemm_b, out_proj_reduction_b, out_proj_size_b = self.parallelism_gemm_backward(
            gemm_output_proj, "output_projection_b", gemm_type=GemmType.OUT_PROJ
        )
        transformer_timings["output_proj"] = OperationTiming(
            "output_proj",
            forward=DirectionTiming(
                compute_time=out_proj_gemm_f,
                comm_time=out_proj_reduction_f,
                comm_bytes=int(out_proj_size_f or 0),
                flops=out_proj_flops_f,
                memory_accesses=dict(self._mem_levels(out_proj_mem_f)),
            ),
            backward=DirectionTiming(
                compute_time=out_proj_gemm_b,
                comm_time=out_proj_reduction_b,
                comm_bytes=int(out_proj_size_b or 0),
            ),
        )
        if not use_moe_layer: 
            ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_ffn1, "ffn_f", gemm_type=GemmType.FFN1)
            )
            ffn1_gemm_b, ffn1_reduction_b, ffn1_size_b = self.parallelism_gemm_backward(
                gemm_ffn1, "ffn_b", gemm_type=GemmType.FFN1
            )
            ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_ffn2, "ffn2_f", gemm_type=GemmType.FFN2)
            )
            ffn2_gemm_b, ffn2_reduction_b, ffn2_size_b = self.parallelism_gemm_backward(
                gemm_ffn2, "ffn2_b", gemm_type=GemmType.FFN2
            )

        else: # MOE expert parallelism
            allow_padding = self._moe_allow_padding()
            (
                moe_tokens_owner,
                moe_tokens_dispatched,
                moe_tokens_local,
                _experts_per_rank,
                _tokens_per_expert,
            ) = self._moe_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
            moe_tokens_shared = self._moe_tokens_shared(moe_tokens_owner)
            router_gemm_time_f , router_reduction_f, router_size_f, _, _ = _extract_forward(
                self.get_router_f(gemm_router, gemm_ffn1, batch_size=batch_size, seq_len=seq_len)
            )
            router_gemm_time_b , router_reduction_b, router_size_b= self.get_router_b(
                gemm_router, gemm_ffn1, batch_size=batch_size, seq_len=seq_len
            )
            moe_group = self._moe_routing_group()
            axis = "ep" if moe_group == self.ep else None
            dispatch_fwd_bytes = int(
                math.ceil(self.precision.activations * moe_tokens_dispatched * hidden_dim)
            )
            dispatch_bwd_bytes = int(
                math.ceil(self.precision.grad_communication * moe_tokens_dispatched * hidden_dim)
            )
            dispatch_fwd_time = self.network_model.collective(
                kind=ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label="moe_dispatch_f_all_to_all",
                axis=axis,
            )
            dispatch_bwd_time = self.network_model.collective(
                kind=ALL_TO_ALL,
                size_bytes=dispatch_bwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label="moe_dispatch_b_all_to_all",
                axis=axis,
            )
            ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = _extract_forward(
                self.get_moe_ffn_f(
                    gemm_ffn1,
                    "ffn1_f",
                    gemm_type=GemmType.FFN1,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    allow_padding=allow_padding,
                )
            )
            ffn1_gemm_b, ffn1_reduction_b, ffn1_size_b = self.get_moe_ffn_b(
                gemm_ffn1,
                "ffn1_b",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=seq_len,
                allow_padding=allow_padding,
            )
            combine_fwd_time = self.network_model.collective(
                kind=ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label="moe_combine_f_all_to_all",
                axis=axis,
            )
            combine_bwd_time = self.network_model.collective(
                kind=ALL_TO_ALL,
                size_bytes=dispatch_bwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label="moe_combine_b_all_to_all",
                axis=axis,
            )
            ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = _extract_forward(
                self.get_moe_ffn_f(
                    gemm_ffn2,
                    "ffn2_f",
                    gemm_type=GemmType.FFN2,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    allow_padding=allow_padding,
                )
            )
            ffn2_gemm_b, ffn2_reduction_b, ffn2_size_b = self.get_moe_ffn_b(
                gemm_ffn2,
                "ffn2_b",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=seq_len,
                allow_padding=allow_padding,
            )
        if use_moe_layer:
            transformer_timings["router"] = OperationTiming(
                "router",
                forward=DirectionTiming(
                    compute_time=router_gemm_time_f,
                    comm_time=router_reduction_f,
                    comm_bytes=int(router_size_f or 0),
                ),
                backward=DirectionTiming(
                    compute_time=router_gemm_time_b,
                    comm_time=router_reduction_b,
                    comm_bytes=int(router_size_b or 0),
                ),
            )
            transformer_timings["moe_dispatch"] = OperationTiming(
                "moe_dispatch",
                forward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=dispatch_fwd_time,
                    comm_bytes=int(dispatch_fwd_bytes or 0),
                ),
                backward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=dispatch_bwd_time,
                    comm_bytes=int(dispatch_bwd_bytes or 0),
                ),
            )
            transformer_timings["moe_combine"] = OperationTiming(
                "moe_combine",
                forward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=combine_fwd_time,
                    comm_bytes=int(dispatch_fwd_bytes or 0),
                ),
                backward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=combine_bwd_time,
                    comm_bytes=int(dispatch_bwd_bytes or 0),
                ),
            )
        
        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=DirectionTiming(
                compute_time=ffn1_gemm_f,
                comm_time=ffn1_reduction_f,
                comm_bytes=int(ffn1_size_f or 0),
                flops=ffn1_flops_f,
                memory_accesses=dict(self._mem_levels(ffn1_mem_f)),
            ),
            backward=DirectionTiming(
                compute_time=ffn1_gemm_b,
                comm_time=ffn1_reduction_b,
                comm_bytes=int(ffn1_size_b or 0),
            ),
        )


        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=DirectionTiming(
                compute_time=ffn2_gemm_f,
                comm_time=ffn2_reduction_f,
                comm_bytes=int(ffn2_size_f or 0),
                flops=ffn2_flops_f,
                memory_accesses=dict(self._mem_levels(ffn2_mem_f)),
            ),
            backward=DirectionTiming(
                compute_time=ffn2_gemm_b,
                comm_time=ffn2_reduction_b,
                comm_bytes=int(ffn2_size_b or 0),
            ),
        )

        embedding_f, embedding_mem = self.get_embedding_f(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim)
        embedding_b = self.get_embedding_b(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim)
        transformer_timings["embedding"] = OperationTiming(
            "embedding",
            forward=DirectionTiming(
                compute_time=embedding_f,
                comm_time=0.0,
                comm_bytes=0,
                memory_accesses=dict(self._mem_levels(embedding_mem)),
            ),
            backward=DirectionTiming(
                compute_time=embedding_b,
                comm_time=0.0,
                comm_bytes=0,
            ),
        )

        residual1_f = self.get_residual_f(tensor_shape=gemm_output_proj)
        residual1_b = self.get_residual_b(tensor_shape=gemm_output_proj)
        layernorm1_f, layernorm1_reduction_f, LN1_comm_bytes_f = self.get_layernorm_f(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
        )
        layernorm1_b, layernorm1_reduction_b, LN1_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
            type=GemmType.LAYER_NORM_1,
        )
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=DirectionTiming(
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction_f,
                comm_bytes=int(LN1_comm_bytes_f or 0),
            ),
            backward=DirectionTiming(
                compute_time=layernorm1_b + residual1_b,
                comm_time=layernorm1_reduction_b,
                comm_bytes=int(LN1_comm_bytes_b or 0),
            ),
        )

        residual2_f = self.get_residual_f(tensor_shape=gemm_ffn2)
        residual2_b = self.get_residual_b(tensor_shape=gemm_ffn2)
        layernorm2_f, layernorm2_reduction_f, LN2_comm_bytes_f = self.get_layernorm_f(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
        )
        layernorm2_b, layernorm2_reduction_b, LN2_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=DirectionTiming(
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction_f,
                comm_bytes=int(LN2_comm_bytes_f or 0),
            ),
            backward=DirectionTiming(
                compute_time=layernorm2_b + residual2_b,
                comm_time=layernorm2_reduction_b,
                comm_bytes=int(LN2_comm_bytes_b or 0),
            ),
        )

        ffn1_spec = self._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        ffn1_activation_shape_shared = None
        if use_moe_layer:
            tokens_owner = moe_tokens_owner if moe_tokens_owner is not None else self._moe_tokens_owner(batch_size, seq_len)
            tokens_local = moe_tokens_local if moe_tokens_local is not None else self._moe_tokens_local(
                self._moe_tokens_dispatched(tokens_owner)
            )
            tokens_shared = moe_tokens_shared if moe_tokens_shared is not None else self._moe_tokens_shared(tokens_owner)
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            ffn1_n = ffn1_spec.shard_n if use_tp_sharded else ffn1_spec.n
            ffn1_activation_shape = (tokens_local, ffn1_spec.k, ffn1_n)
            if tokens_shared > 0:
                ffn1_activation_shape_shared = (tokens_shared, ffn1_spec.k, ffn1_n)

        if llm_util.is_llama_style(self.model_type):
            act_f = self.get_swiglu_f(tensor_shape=ffn1_activation_shape)
            act_b = self.get_swiglu_b(tensor_shape=ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_swiglu_f(tensor_shape=ffn1_activation_shape_shared)
                act_b += self.get_swiglu_b(tensor_shape=ffn1_activation_shape_shared)
        else:
            act_f = self.get_gelu_f(tensor_shape=ffn1_activation_shape)
            act_b = self.get_gelu_b(tensor_shape=ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_gelu_f(tensor_shape=ffn1_activation_shape_shared)
                act_b += self.get_gelu_b(tensor_shape=ffn1_activation_shape_shared)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=DirectionTiming(compute_time=act_f, comm_time=0.0, comm_bytes=0),
            backward=DirectionTiming(compute_time=act_b, comm_time=0.0, comm_bytes=0),
        )

        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(gemm=gemm_linear)
        linear_softmax_b = self.get_linear_softmax_b(gemm=gemm_linear)
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=DirectionTiming(
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0,
                memory_accesses=dict(self._mem_levels(linear_softmax_mem)),
            ),
            backward=DirectionTiming(
                compute_time=linear_softmax_b,
                comm_time=0.0,
                comm_bytes=0,
            ),
        )

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        transformer_time_f = (
            transformer_timings["qkv_proj"].total_forward_time()
            + transformer_timings["attention"].total_forward_time()
            + transformer_timings["output_proj"].total_forward_time()
            + mlp_group.forward_total_time()
            + transformer_timings["layernorm1"].total_forward_time()
            + transformer_timings["layernorm2"].total_forward_time()
        )
        transformer_time_b = (
            transformer_timings["qkv_proj"].total_backward_time()
            + transformer_timings["attention"].total_backward_time()
            + transformer_timings["output_proj"].total_backward_time()
            + mlp_group.backward_total_time()
            + transformer_timings["layernorm1"].total_backward_time()
            + transformer_timings["layernorm2"].total_backward_time()
        )
        if use_moe_layer:
            for op_name in ("router", "moe_dispatch", "moe_combine"):
                if op_name in transformer_timings:
                    transformer_time_f += transformer_timings[op_name].total_forward_time()
                    transformer_time_b += transformer_timings[op_name].total_backward_time()
        # Pipeline-style recompute uses explicit recompute nodes, so keep backward time as-is.
        transformer_time_b_combined = transformer_time_b

        node_breakdown = {
            "transformer_time_f": transformer_time_f,
            "transformer_time_b": transformer_time_b,
            "transformer_time_b_combined": transformer_time_b_combined,
            "embedding_f": transformer_timings["embedding"].total_forward_time(),
            "embedding_b": transformer_timings["embedding"].total_backward_time(),
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": transformer_timings["linear_softmax"].total_backward_time(),
        }
        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "transformer_timings.yaml")
            with open(results_path, "w", encoding="utf-8") as results_file:
                yaml.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    sort_keys=True,
                )

        return transformer_timings, node_breakdown




    def _effective_transformer_batch(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            if self.pp > 1:
                return self.micro_batch
            return self.batch_size
        if self.pp > 1:
            return self.micro_batch
        dp_dense = self.dp * self.ep if self.use_moe else self.dp
        if dp_dense > 1:
            return self.mini_batch
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes_dense: float,
        reduction_sizes_moe: float,
        local_comp_dense: float,
        local_comp_moe: float,
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
        transformer_dense_ep_sync_bytes: float = 0.0,
        transformer_moe_ep_sync_bytes: float = 0.0,
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        grad_collective = REDUCE_SCATTER if (self.zero_stage >= 2 and self.dp > 1) else ALL_REDUCE
        metadata = {
            'transformer_dense': {
                'size': reduction_sizes_dense,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp_dense,
            },
            'transformer_moe': {
                'size': reduction_sizes_moe,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp_moe,
            },
            'embedding': {
                'size': embedding_size,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': PIPELINE,
                'participants': 2,
                'interconnect_type': 'pp',
                'local_comp_time': 0
            }
        }
        if transformer_dense_ep_sync_bytes and self.ep > 1:
            metadata['transformer_dense_ep_sync'] = {
                'size': int(math.ceil(transformer_dense_ep_sync_bytes)),
                'type': grad_collective,
                'participants': self.ep,
                'interconnect_type': 'ep',
                'local_comp_time': 0,
            }
        if transformer_moe_ep_sync_bytes and self.ep > 1:
            metadata['transformer_moe_ep_sync'] = {
                'size': int(math.ceil(transformer_moe_ep_sync_bytes)),
                'type': grad_collective,
                'participants': self.ep,
                'interconnect_type': 'ep',
                'local_comp_time': 0,
            }
        if zero2_embedding_gather_bytes:
            metadata['zero2_embedding_gather'] = {
                'size': int(math.ceil(zero2_embedding_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero2_transformer_gather_bytes:
            metadata['zero2_transformer_gather'] = {
                'size': int(math.ceil(zero2_transformer_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero2_softmax_gather_bytes:
            metadata['zero2_softmax_gather'] = {
                'size': int(math.ceil(zero2_softmax_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero3_embedding_gather_bytes:
            metadata['zero3_embedding_gather'] = {
                'size': int(math.ceil(zero3_embedding_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero3_transformer_gather_bytes:
            metadata['zero3_transformer_gather'] = {
                'size': int(math.ceil(zero3_transformer_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                'tp_shard': True,
            }
        if zero3_softmax_gather_bytes:
            metadata['zero3_softmax_gather'] = {
                'size': int(math.ceil(zero3_softmax_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        return metadata

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.links["dp"].bandwidth, self.links["dp"].latency),
            'ep': (self.links["ep"].bandwidth, self.links["ep"].latency),
            'pp': (self.links["pp"].bandwidth, self.links["pp"].latency),
            'tp': (self.links["tp"].bandwidth, self.links["tp"].latency),
            'cp': (self.links["cp"].bandwidth, self.links["cp"].latency),
        }


    def _prepare_execution_graphs(
        self,
        *,
        node_breakdown: Dict[str, float],
        transformer_timings: Dict[str, OperationTiming],
        moe_node_breakdown: Optional[Dict[str, float]] = None,
        moe_transformer_timings: Optional[Dict[str, OperationTiming]] = None,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
        include_pipeline_backward: bool,
        include_transformer_backward: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,  # optional override (decode only)
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Tuple[
        Graph,
        Any,
        Optional[Graph],
        Optional[Any],
        Optional[Graph],
        Optional[Any],
        Optional[Any],
        Optional[Graph],
        Optional[Any],
        Optional[Any],
        Dict[str, Tuple[float, float]],
    ]:
        """Build pipeline/transformer graphs shared across training and inference."""
        need_no_dp_variant = getattr(self, "gradient_accumulation_steps", 1) > 1

        if not include_pipeline_backward and not include_transformer_backward:
            # Forward-only inference: skip training all-reduce bookkeeping.
            reduction_sizes_dense = 0.0
            reduction_sizes_moe = 0.0
            local_comp_dense = 0.0
            local_comp_moe = 0.0
        else:
            reduction_sizes_dense = self.get_data_parallel_reduction_sizes(hidden_dim, intermediate_size, moe=False)
            local_comp_dense = self.get_data_parallel_reduction_llm(hidden_dim, intermediate_size, moe=False)
            reduction_sizes_moe = reduction_sizes_dense
            local_comp_moe = local_comp_dense
            if self.use_moe:
                reduction_sizes_moe = self.get_data_parallel_reduction_sizes(
                    hidden_dim,
                    self.moe_intermediate_size,
                    moe=True,
                )
                local_comp_moe = self.get_data_parallel_reduction_llm(
                    hidden_dim,
                    self.moe_intermediate_size,
                    moe=True,
                )

        # these are used for dp all-reduce/reduce-scatter.
        embedding_size = math.ceil(self.precision.grad_communication * vocab_size * hidden_dim) + math.ceil(self.precision.grad_communication * seq_len * hidden_dim * batch_size)
        softmax_size = math.ceil(self.precision.grad_communication * hidden_dim * vocab_size)
        cross_layer_bytes = self.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, seq_len)[1]

        ep_dense_sync_bytes_dense = 0
        ep_dense_sync_bytes_moe = 0
        if include_transformer_backward and self.use_moe and self.ep > 1:
            if llm_util.is_glm_style(self.model_type):
                _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                    hidden_dim,
                    self.num_heads,
                    self.kv_heads,
                    head_dim=getattr(self, "head_dim", None),
                )
                qkv_params = hidden_dim * (q_size + 2 * kv_size)
                output_params = q_size * hidden_dim
            else:
                qkv_params = hidden_dim * 3 * hidden_dim
                output_params = hidden_dim * hidden_dim
            qkv_size = math.ceil(self.precision.grad_communication * qkv_params)
            output_size = math.ceil(self.precision.grad_communication * output_params)
            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            ffn1_size = math.ceil(self.precision.grad_communication * ffn1_dim * hidden_dim)
            ffn2_size = math.ceil(self.precision.grad_communication * intermediate_size * hidden_dim)
            ep_dense_sync_bytes_dense = int(qkv_size + output_size + ffn1_size + ffn2_size)
            router_size = math.ceil(self.precision.grad_communication * hidden_dim * self.moe_num_experts)
            shared_size = 0
            if self.n_shared_experts > 0:
                ffn_proj_factor = 3 if llm_util.is_llama_style(self.model_type) else 2
                expert_param_size = ffn_proj_factor * self.moe_intermediate_size * hidden_dim
                use_tp_sharded = bool(getattr(self, "tp_ep", True))
                if use_tp_sharded:
                    shared_params_per_rank = (expert_param_size / max(1, self.tp)) * self.n_shared_experts
                else:
                    shared_params_per_rank = expert_param_size * self.n_shared_experts
                shared_size = math.ceil(self.precision.grad_communication * shared_params_per_rank)
            ep_dense_sync_bytes_moe = int(qkv_size + output_size + router_size + shared_size)

        comm_metadata = self._build_comm_metadata(
            reduction_sizes_dense=reduction_sizes_dense,
            reduction_sizes_moe=reduction_sizes_moe,
            local_comp_dense=local_comp_dense,
            local_comp_moe=local_comp_moe,
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
            transformer_dense_ep_sync_bytes=ep_dense_sync_bytes_dense,
            transformer_moe_ep_sync_bytes=ep_dense_sync_bytes_moe,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )
        grad_collective = REDUCE_SCATTER if (self.zero_stage >= 2 and self.dp > 1) else ALL_REDUCE

        parallelism_mode = self.get_parallelism_mode()
        def _clone_comm_rules(rules: Dict[str, Dict[str, Optional[Dict[str, str]]]]) -> Dict[str, Dict[str, Optional[Dict[str, str]]]]:
            cloned: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {}
            for op_name, directions in rules.items():
                cloned[op_name] = {}
                for direction, spec in directions.items():
                    cloned[op_name][direction] = dict(spec) if spec else None
            return cloned

        def _resolve_comm_rules(
            mode: ParallelismMode,
            use_moe_layer: bool,
        ) -> Dict[str, Dict[str, Optional[Dict[str, str]]]]:
            base_rules = COMMUNICATION_RULES.get(mode) or {}
            if not use_moe_layer:
                return base_rules
            overrides = MOE_COMMUNICATION_RULES.get(mode)
            if not overrides:
                return base_rules
            merged = _clone_comm_rules(base_rules)
            for op_name, directions in overrides.items():
                merged.setdefault(op_name, {})
                for direction, spec in directions.items():
                    merged[op_name][direction] = dict(spec) if spec else None
            return merged

        graph_ep = self.ep if self.use_moe else 1
        participants_lookup_base = {
            'tp': int(getattr(self, 'tp', 0) or 0),
            'cp': int(getattr(self, 'cp', 0) or 0),
            'ep': int(graph_ep or 0),
            'dp': int(getattr(self, 'dp', 0) or 0),
            'pp': int(getattr(self, 'pp', 0) or 0),
        }
        group_members = {
            "MLP": ("ffn1", "gelu", "ffn2"),
        }
        def _build_transformer_graph(
            transformer_timings_local: Dict[str, OperationTiming],
            *,
            use_moe_layer: bool,
            ep_dense_sync_bytes: int,
        ) -> Tuple[Optional[Graph], Optional[Any], Optional[Any]]:
            participants_lookup = dict(participants_lookup_base)
            if use_moe_layer:
                participants_lookup["ep"] = int(self._moe_routing_group())
            transformer_operation_entries: List[Dict[str, Any]] = []
            transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}
            rules_by_mode = _resolve_comm_rules(parallelism_mode, use_moe_layer)

            def _register_specs(specs: Sequence[CommSpec]) -> List[str]:
                names: List[str] = []
                for spec in specs:
                    existing = transformer_comm_metadata.get(spec.name)
                    placement = None
                    if spec.extra:
                        placement = spec.extra.get("placement")
                    if existing:
                        if (
                            existing["size"] != spec.size_bytes
                            or existing["type"] != spec.kind
                            or existing["participants"] != spec.participants
                            or existing["interconnect_type"] != spec.interconnect
                        ):
                            raise ValueError(
                                f"Conflicting CommSpec registration for '{spec.name}' "
                                f"(existing={existing}, new={spec})"
                            )
                    else:
                        transformer_comm_metadata[spec.name] = {
                            "size": spec.size_bytes,
                            "type": spec.kind,
                            "participants": spec.participants,
                            "interconnect_type": spec.interconnect,
                        }
                        if placement:
                            transformer_comm_metadata[spec.name]["placement"] = placement
                    names.append(spec.name)
                return names

            def _make_ep_dense_sync_specs(
                op_name: str,
                direction: str,
                total_bytes: int,
            ) -> Tuple[CommSpec, ...]:
                bytes_int = int(math.ceil(float(total_bytes or 0.0)))
                if bytes_int <= 0 or self.ep <= 1:
                    return ()
                name = f"ep_dense_sync_{op_name}_{direction}"
                return (
                    CommSpec(
                        name=name,
                        kind=grad_collective,
                        size_bytes=bytes_int,
                        participants=int(self.ep),
                        interconnect="ep",
                        extra={
                            "scope": "ep",
                            "direction": direction,
                            "op_name": op_name,
                            "parallelism_mode": parallelism_mode.value,
                            "placement": "post",
                        },
                    ),
                )

            def _build_direction_entry(
                op_name: str,
                direction: str,
                timing: Optional[DirectionTiming],
            ) -> Dict[str, Any]:
                if timing is None:
                    return {"duration": 0.0, "reduction": 0.0, "comm_keys": []}
                specs = list(_make_comm_specs(op_name, direction, timing.comm_bytes, rules_by_mode))
                if (
                    include_transformer_backward
                    and direction == "backward"
                    and op_name == "layernorm1"
                    and ep_dense_sync_bytes > 0
                ):
                    specs.extend(_make_ep_dense_sync_specs(op_name, direction, ep_dense_sync_bytes))
                return {
                    "duration": timing.compute_time,
                    "reduction": timing.comm_time,
                    "comm_keys": _register_specs(specs),
                }

            def _make_comm_specs(
                op_name: str,
                direction: str,
                total_bytes: float,
                rules: Dict[str, Dict[str, Optional[Dict[str, str]]]],
            ) -> Tuple[CommSpec, ...]:
                bytes_int = int(math.ceil(float(total_bytes or 0.0)))
                if bytes_int <= 0:
                    return ()
                rule_spec = rules.get(op_name) or rules.get(COMM_RULE_DEFAULT_KEY)
                if not rule_spec:
                    raise ValueError(
                        f"Missing communication rule for op '{op_name}' in mode '{parallelism_mode.value}' "
                        f"with non-zero comm_bytes ({bytes_int})"
                    )
                rule = rule_spec.get(direction)
                if not rule:
                    return ()
                kind = rule.get("kind")
                scope = rule.get("participants")
                if kind is None or not scope:
                    return ()
                if not isinstance(kind, CollectiveType):
                    raise TypeError(
                        f"Comm rule for '{op_name}' must use CollectiveType (got {type(kind).__name__})"
                    )
                kind_enum = kind
                participants = participants_lookup.get(scope, 0)
                if participants <= 1:
                    return ()
                interconnect = rule.get("interconnect", scope)
                count_raw = rule.get("count", 1)
                try:
                    count = int(count_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Comm rule for '{op_name}' has non-integer count ({count_raw!r})"
                    ) from exc
                if count <= 0:
                    return ()
                if count == 1:
                    sizes = [bytes_int]
                else:
                    per_size = bytes_int // count
                    if per_size * count != bytes_int:
                        raise ValueError(
                            f"Comm bytes for '{op_name}' ({bytes_int}) must be divisible by count={count}"
                        )
                    sizes = [per_size for _ in range(count)]
                placement_spec = rule.get("placement", None)
                placements: List[str] = []
                if placement_spec is None:
                    placements = ["post" for _ in range(len(sizes))]
                elif isinstance(placement_spec, str):
                    placements = [placement_spec for _ in range(len(sizes))]
                else:
                    try:
                        placements = list(placement_spec)
                    except TypeError as exc:
                        raise ValueError(
                            f"Comm rule for '{op_name}' has invalid placement ({placement_spec!r})"
                        ) from exc
                    if len(placements) != len(sizes):
                        raise ValueError(
                            f"Comm rule for '{op_name}' placement length mismatch "
                            f"(expected {len(sizes)}, got {len(placements)})"
                        )
                specs: List[CommSpec] = []
                for idx, size in enumerate(sizes):
                    if size <= 0:
                        continue
                    placement = placements[idx] if idx < len(placements) else "post"
                    if placement not in {"pre", "post"}:
                        raise ValueError(
                            f"Comm rule for '{op_name}' has invalid placement '{placement}' "
                            "(expected 'pre' or 'post')"
                        )
                    name = f"{op_name}_{direction}_{kind_enum.value}"
                    if len(sizes) > 1:
                        name = f"{name}_part{idx}"
                    specs.append(
                        CommSpec(
                            name=name,
                            kind=kind_enum,
                            size_bytes=size,
                            participants=participants,
                            interconnect=interconnect,
                            extra={
                                "scope": scope,
                                "direction": direction,
                                "op_name": op_name,
                                "parallelism_mode": parallelism_mode.value,
                                "placement": placement,
                            },
                        )
                    )
                return tuple(specs)

            def _build_group_operation(name: str, members: Tuple[str, ...]) -> OperationTiming:
                ops = tuple(transformer_timings_local[m] for m in members)
                group = OperationGroup(name, ops)
                forward_memory = self._combine_mem(*(op.forward.memory_accesses for op in ops))
                forward_flops = sum(op.forward.flops for op in ops)
                forward_dir = DirectionTiming(
                    compute_time=group.forward_compute_time(),
                    comm_time=group.forward_comm_time(),
                    comm_bytes=group.forward_comm_bytes(),
                    flops=forward_flops,
                    memory_accesses=forward_memory,
                )
                if include_transformer_backward:
                    backward_memory = self._combine_mem(
                        *(op.backward.memory_accesses for op in ops if op.backward is not None)
                    )
                    backward_flops = sum((op.backward.flops if op.backward else 0.0) for op in ops)
                    backward_dir = DirectionTiming(
                        compute_time=group.backward_compute_time(),
                        comm_time=group.backward_comm_time(),
                        comm_bytes=group.backward_comm_bytes(),
                        flops=backward_flops,
                        memory_accesses=backward_memory,
                    )
                else:
                    backward_dir = None
                return OperationTiming(name, forward_dir, backward_dir)

            def _resolve_operation(name: str) -> OperationTiming:
                if name in group_members:
                    return _build_group_operation(name, group_members[name])
                return transformer_timings_local[name]

            if parallelism_mode not in (
                ParallelismMode.CONTEXT,
                ParallelismMode.TENSOR_CONTEXT_HYBRID,
                ParallelismMode.TENSOR,
                ParallelismMode.TENSOR_SEQUENCE,
                ParallelismMode.SINGLE,
            ):
                raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
            op_names: Sequence[str] = ("layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2", "MLP")
            op_names = list(op_names)
            if "optimizer" in transformer_timings_local:
                op_names.append("optimizer")
            if use_moe_layer and "MLP" in op_names:
                mlp_index = op_names.index("MLP")
                for offset, op_name in enumerate(("router", "moe_dispatch")):
                    if op_name not in op_names:
                        if op_name not in transformer_timings_local:
                            raise KeyError(f"Missing transformer timing for operation '{op_name}'")
                        op_names.insert(mlp_index + offset, op_name)
                if "moe_combine" not in op_names:
                    if "moe_combine" not in transformer_timings_local:
                        raise KeyError("Missing transformer timing for operation 'moe_combine'")
                    mlp_index = op_names.index("MLP")
                    op_names.insert(mlp_index + 1, "moe_combine")

            for key in op_names:
                try:
                    op_timing = _resolve_operation(key)
                except KeyError as exc:
                    raise KeyError(f"Missing transformer timing for operation '{key}'") from exc

                entry = {
                    "name": key,
                    "forward": _build_direction_entry(key, "forward", op_timing.forward),
                    "backward": _build_direction_entry(
                        key, "backward", op_timing.backward if include_transformer_backward else None
                    ),
                }

                if not include_transformer_backward:
                    # Ensure backward section is zeroed when excluded.
                    entry["backward"] = {"duration": 0.0, "reduction": 0.0, "comm_keys": []}

                transformer_operation_entries.append(entry)
            transformer_graph: Optional[Graph] = None
            transformer_forward_root: Optional[Any] = None
            transformer_backward_root: Optional[Any] = None

            transformer_comp_times = {
                "transformer": {
                    "gemms": transformer_operation_entries,
                }
            }

            transformer_graph = llm_simulation.Graph(
                mode="transformer",
                dp=self.dp,
                pp=self.pp,
                tp=self.tp,
                cp=self.cp,
                ep=graph_ep,
                comp_times=transformer_comp_times,
                comm_metadata=transformer_comm_metadata,
                misc_metadata={"dp_zero_stage": self.zero_stage},
            )
            transformer_forward_root = transformer_graph.construct_transformer_graph(direction="forward")
            if include_transformer_backward:
                bwd_direction = "backward"
                transformer_backward_root = transformer_graph.construct_transformer_graph(direction=bwd_direction)

            transformer_forward_root = apply_overlap_transforms(
                transformer_forward_root,
                parallelism_mode,
                self.tp_overlap,
                self.tp_sp_overlap,
                self.cp_overlap,
            )
            if include_transformer_backward:
                transformer_backward_root = apply_overlap_transforms(
                    transformer_backward_root,
                    parallelism_mode,
                    self.tp_overlap,
                    self.tp_sp_overlap,
                    self.cp_overlap,
                )
            return transformer_graph, transformer_forward_root, transformer_backward_root

        transformer_graph, transformer_forward_root, transformer_backward_root = _build_transformer_graph(
            transformer_timings,
            use_moe_layer=False,
            ep_dense_sync_bytes=ep_dense_sync_bytes_dense,
        )
        moe_transformer_graph: Optional[Graph] = None
        moe_transformer_forward_root: Optional[Any] = None
        moe_transformer_backward_root: Optional[Any] = None
        moe_layer_mask = list(getattr(self, "moe_layer_mask", []) or [])
        has_moe_layers = bool(self.use_moe and any(moe_layer_mask))
        if has_moe_layers:
            if not moe_transformer_timings or not moe_node_breakdown:
                raise ValueError("MoE layer schedule requires moe_transformer_timings and moe_node_breakdown.")
            (
                moe_transformer_graph,
                moe_transformer_forward_root,
                moe_transformer_backward_root,
            ) = _build_transformer_graph(
                moe_transformer_timings,
                use_moe_layer=True,
                ep_dense_sync_bytes=ep_dense_sync_bytes_moe,
            )

        transformer_graph_no_dp = None
        transformer_forward_root_no_dp = None
        transformer_backward_root_no_dp = None
        moe_transformer_graph_no_dp = None
        moe_transformer_forward_root_no_dp = None
        moe_transformer_backward_root_no_dp = None
        if need_no_dp_variant and include_transformer_backward and self.use_moe and self.ep > 1:
            (
                transformer_graph_no_dp,
                transformer_forward_root_no_dp,
                transformer_backward_root_no_dp,
            ) = _build_transformer_graph(
                transformer_timings,
                use_moe_layer=False,
                ep_dense_sync_bytes=0,
            )
            if has_moe_layers:
                if not moe_transformer_timings or not moe_node_breakdown:
                    raise ValueError("MoE layer schedule requires moe_transformer_timings and moe_node_breakdown.")
                (
                    moe_transformer_graph_no_dp,
                    moe_transformer_forward_root_no_dp,
                    moe_transformer_backward_root_no_dp,
                ) = _build_transformer_graph(
                    moe_transformer_timings,
                    use_moe_layer=True,
                    ep_dense_sync_bytes=0,
                )

        dense_transformer_f = node_breakdown.get('transformer_time_f', 0.0)
        dense_transformer_b = node_breakdown.get('transformer_time_b', 0.0) if include_pipeline_backward else 0.0
        moe_transformer_f = dense_transformer_f
        moe_transformer_b = dense_transformer_b
        if has_moe_layers and moe_node_breakdown:
            moe_transformer_f = moe_node_breakdown.get('transformer_time_f', dense_transformer_f)
            moe_transformer_b = (
                moe_node_breakdown.get('transformer_time_b', dense_transformer_b)
                if include_pipeline_backward
                else 0.0
            )

        comp_times = {
            "embedding_f": node_breakdown.get('embedding_f', 0.0),
            "embedding_b": node_breakdown.get('embedding_b', 0.0) if include_pipeline_backward else 0.0,
            "linear_softmax_f": node_breakdown.get('linear_softmax_f', 0.0),
            "linear_softmax_b": node_breakdown.get('linear_softmax_b', 0.0) if include_pipeline_backward else 0.0,
            "transformer_f": dense_transformer_f,
            "transformer_b": dense_transformer_b,
            "transformer_f_dense": dense_transformer_f,
            "transformer_b_dense": dense_transformer_b,
            "transformer_f_moe": moe_transformer_f,
            "transformer_b_moe": moe_transformer_b,
            "optimizer": self.get_data_parallel_reduction_llm(hidden_dim, intermediate_size),
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        comp_times_no_dp = None
        if need_no_dp_variant:
            comp_times_no_dp = dict(comp_times)
            if include_pipeline_backward and self.use_moe and self.ep > 1:
                gemm_shapes_local = gemm_shapes
                if gemm_shapes_local is None:
                    gemm_shapes_local = llm_util.process_gemm_shapes(
                        self,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        d_model=hidden_dim,
                        num_heads=self.num_heads,
                        kv_heads=self.kv_heads,
                        intermediate_size=intermediate_size,
                        vocab_size=vocab_size,
                    )
                gemm_linear = gemm_shapes_local.get("linear") if gemm_shapes_local else None
                if gemm_linear is not None:
                    comp_times_no_dp["linear_softmax_b"] = self.get_linear_softmax_b(
                        gemm_linear,
                    )
                comp_times_no_dp["embedding_b"] = self.get_embedding_b(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                )
        flattened_mode = self.execution_mode == ExecutionMode.FULL_ASTRASIM_FLATTENED
        pipeline_style_recompute_flag = bool(getattr(self, "full_recomputation", False))
        misc_metadata = {
            "num_batch": self.mb,
            "num_layer": self.num_layers,
            "dp_zero_stage": self.zero_stage,
            "full_recomputation": self.full_recomputation,
            "flattened_mode": flattened_mode,
            "pipeline_style_recompute": pipeline_style_recompute_flag,
            "dp_microbatch_mode": getattr(self, "dp_microbatch", "every_mb"),
            "moe_layer_mask": moe_layer_mask,
        }

        pipeline_graph_obj = llm_simulation.Graph(
            mode="pipeline",
            dp=self.dp,
            pp=self.pp,
            tp=self.tp,
            cp=self.cp,
            ep=graph_ep,
            comp_times=comp_times,
            comm_metadata=comm_metadata,
            misc_metadata=misc_metadata,
        )
        
        graph_root = pipeline_graph_obj.construct_fwd_bwd_graph(
            include_backward=include_pipeline_backward,
            include_optimizer=True
        )
        pipeline_graph_obj_no_dp = None
        graph_root_no_dp = None
        if need_no_dp_variant:
            pipeline_graph_obj_no_dp = llm_simulation.Graph( # if gradient accumulation is used, we need a no-dp variant of the graph for the non-last step, since dp all-reduce is only done on the last step
                mode="pipeline",
                dp=1,
                pp=self.pp,
                tp=self.tp,
                cp=self.cp,
                ep=graph_ep,
                comp_times=comp_times_no_dp or comp_times,
                comm_metadata=comm_metadata,
                misc_metadata=misc_metadata,
            )
            
            graph_root_no_dp = pipeline_graph_obj_no_dp.construct_fwd_bwd_graph(
                include_backward=include_pipeline_backward,
                include_optimizer=False
            )

        
        interconnect_params = self._build_interconnect_params()

        self.transformer_graph_no_dp = transformer_graph_no_dp
        self.transformer_forward_root_no_dp = transformer_forward_root_no_dp
        self.transformer_backward_root_no_dp = transformer_backward_root_no_dp
        self.transformer_graph_moe_no_dp = moe_transformer_graph_no_dp
        self.transformer_forward_root_moe_no_dp = moe_transformer_forward_root_no_dp
        self.transformer_backward_root_moe_no_dp = moe_transformer_backward_root_no_dp

        return (
            pipeline_graph_obj,
            graph_root,
            pipeline_graph_obj_no_dp,
            graph_root_no_dp,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            moe_transformer_graph,
            moe_transformer_forward_root,
            moe_transformer_backward_root,
            interconnect_params,
        )

    def _build_training_graphs_and_memory_data(self):
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads

        # ZeRO data-parallel stages:
        #   stage 0 – identical to DDP (replicated params/grads/optimizer)
        #   stage 1 – shard optimizer state only (communication unchanged)
        #   stage 2 – shard optimizer and gradients (grad RS + one param AG)
        #   stage 3 – shard optimizer, gradients, and parameters (grad RS + two param AG, per-layer materialization)
        (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = self._param_stats_per_rank(hidden_dim, intermediate_size, vocab_size)

        param_bytes = total_params_per_rank * self.precision.parameters
        transformer_param_layer_bytes = params_per_layer_per_rank * self.precision.parameters
        embedding_param_bytes = embedding_params_per_rank * self.precision.parameters
        softmax_param_bytes = output_params_per_rank * self.precision.parameters

        if self.zero_stage == 2 and self.dp > 1:
            zero2_embedding_gather_bytes = embedding_param_bytes
            zero2_transformer_gather_bytes = transformer_param_layer_bytes
            zero2_softmax_gather_bytes = softmax_param_bytes
        else:
            zero2_embedding_gather_bytes = 0.0
            zero2_transformer_gather_bytes = 0.0
            zero2_softmax_gather_bytes = 0.0

        if self.zero_stage >= 3 and self.dp > 1:
            zero3_embedding_gather_bytes = embedding_param_bytes
            zero3_transformer_gather_bytes = transformer_param_layer_bytes
            zero3_softmax_gather_bytes = softmax_param_bytes
        else:
            zero3_embedding_gather_bytes = 0.0
            zero3_transformer_gather_bytes = 0.0
            zero3_softmax_gather_bytes = 0.0

        # zero stage 3 creates a need for materialization of parameters per layer. This is the peak memory requirement.
        # these bytes are 'ephemeral' as they get discarded after, but still need to accounted for in the memory sim.
        self.zero3_ephemeral_peak_bytes = (
            max_layer_params * self.precision.parameters
            if self.zero_stage >= 3 and self.dp > 1
            else 0.0
        )

        num_SMs = self.hw_config.tech_config.core.num_bundles
        transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            seq_len,
            num_heads,
            kv_heads,
            intermediate_size,
            num_SMs,
            use_moe_override=False,
        )
        moe_transformer_timings = None
        moe_node_breakdown = None
        if self.use_moe:
            moe_transformer_timings, moe_node_breakdown = self.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                seq_len,
                num_heads,
                kv_heads,
                self.moe_intermediate_size,
                num_SMs,
                use_moe_override=True,
            )

        mem_estimator = MemoryEstimator(self)
        memory_data = mem_estimator.build_memory_data(
            mode="training",
            batch_size=batch_size,
            seq_len=seq_len,
            zero3_ephemeral_peak_bytes=self.zero3_ephemeral_peak_bytes,
        )

        transformer_act_layer = memory_data.get("activation_mem_per_layer", 0.0)
        transformer_act_layer_inf = memory_data.get("activation_mem_per_layer_inference", 0.0)
        transformer_static_layer = memory_data.get("static_mem_per_layer", 0.0)
        gradient_mem_layer = memory_data.get("gradient_mem_per_layer", 0.0)
        optimizer_mem_layer = memory_data.get("optimizer_mem_per_layer", 0.0)
        weight_memory_layer = memory_data.get("weight_mem_per_layer", 0.0)
        transformer_mem_layer = memory_data.get("total_mem_per_layer", 0.0)
        if self.use_moe:
            dense_layers = max(0, self.num_layers - self.num_moe_layers)
            moe_layers = max(0, self.num_moe_layers)
            total_layers = max(1, dense_layers + moe_layers)
            act_dense = memory_data.get("activation_mem_per_layer_dense", transformer_act_layer)
            act_moe = memory_data.get("activation_mem_per_layer_moe", act_dense)
            act_inf_dense = memory_data.get("activation_mem_per_layer_inference_dense", transformer_act_layer_inf)
            act_inf_moe = memory_data.get("activation_mem_per_layer_inference_moe", act_inf_dense)
            static_dense = memory_data.get("static_mem_per_layer_dense", transformer_static_layer)
            static_moe = memory_data.get("static_mem_per_layer_moe", static_dense)
            grad_dense = memory_data.get("gradient_mem_per_layer_dense", gradient_mem_layer)
            grad_moe = memory_data.get("gradient_mem_per_layer_moe", grad_dense)
            opt_dense = memory_data.get("optimizer_mem_per_layer_dense", optimizer_mem_layer)
            opt_moe = memory_data.get("optimizer_mem_per_layer_moe", opt_dense)
            weight_dense = memory_data.get("weight_mem_per_layer_dense", weight_memory_layer)
            weight_moe = memory_data.get("weight_mem_per_layer_moe", weight_dense)
            total_dense = memory_data.get("total_mem_per_layer_dense", transformer_mem_layer)
            total_moe = memory_data.get("total_mem_per_layer_moe", total_dense)
            transformer_act_layer = (act_dense * dense_layers + act_moe * moe_layers) / total_layers
            transformer_act_layer_inf = (act_inf_dense * dense_layers + act_inf_moe * moe_layers) / total_layers
            transformer_static_layer = (static_dense * dense_layers + static_moe * moe_layers) / total_layers
            gradient_mem_layer = (grad_dense * dense_layers + grad_moe * moe_layers) / total_layers
            optimizer_mem_layer = (opt_dense * dense_layers + opt_moe * moe_layers) / total_layers
            weight_memory_layer = (weight_dense * dense_layers + weight_moe * moe_layers) / total_layers
            transformer_mem_layer = (total_dense * dense_layers + total_moe * moe_layers) / total_layers

        if self._debug_memory:
            layers_per_device = max(1, math.ceil(self.num_layers / max(1, self.pp)))
            to_gib = lambda bytes_val: float(bytes_val) / float(1024 ** 3)
            self._memory_breakdown_debug = {
                "layers_per_device": layers_per_device,
                "zero_stage": self.zero_stage,
                "activation_gib": to_gib(transformer_act_layer * layers_per_device),
                "weight_gib": to_gib(weight_memory_layer * layers_per_device),
                "gradient_gib": to_gib(gradient_mem_layer * layers_per_device),
                "optimizer_gib": to_gib(optimizer_mem_layer * layers_per_device),
                "static_gib": to_gib(transformer_static_layer * layers_per_device),
                "total_layer_gib": to_gib(transformer_mem_layer * layers_per_device),
            }
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                self._memory_breakdown_debug["zero3_ephemeral_gib"] = to_gib(
                    self.zero3_ephemeral_peak_bytes
                )
            extra_static = memory_data.get("extra_static_bytes_per_device", {}) or {}
            if extra_static:
                self._memory_breakdown_debug["extra_static_gib_max"] = to_gib(
                    max(extra_static.values())
                )

        (
            pipeline_graph_obj,
            graph_root,
            pipeline_graph_obj_no_dp,
            graph_root_no_dp,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            moe_transformer_graph,
            moe_transformer_forward_root,
            moe_transformer_backward_root,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            # num_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )

        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.transformer_graph_moe = moe_transformer_graph
        self.transformer_forward_root_moe = moe_transformer_forward_root
        self.transformer_backward_root_moe = moe_transformer_backward_root
        self.transformer_analytical_time_forward = node_breakdown['transformer_time_f']
        # Report backward including recompute overhead when enabled.
        self.transformer_analytical_time_backward_combined = node_breakdown['transformer_time_b_combined']
        self.transformer_analytical_time_backward = self.transformer_analytical_time_backward_combined
        if self.use_moe and moe_node_breakdown and self.num_moe_layers == self.num_layers:
            self.transformer_analytical_time_forward = moe_node_breakdown.get(
                "transformer_time_f",
                self.transformer_analytical_time_forward,
            )
            self.transformer_analytical_time_backward_combined = moe_node_breakdown.get(
                "transformer_time_b_combined",
                self.transformer_analytical_time_backward_combined,
            )
            self.transformer_analytical_time_backward = self.transformer_analytical_time_backward_combined

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params
        self.pipeline_graph_no_dp = pipeline_graph_obj_no_dp
        self.pipeline_root_no_dp = graph_root_no_dp

        return mem_estimator, memory_data

    def calc_time_llm(self):
        """Calculate time for LLM model."""
        mem_estimator, memory_data = self._build_training_graphs_and_memory_data()

        mode = self.execution_mode
        time_fw_bw_no_dp: Optional[float] = None
        if self.gradient_accumulation_steps > 1:
            if not (self.pipeline_graph_no_dp and self.pipeline_root_no_dp):
                raise RuntimeError("Gradient accumulation steps > 1 requires a no-DP pipeline graph")
            transformer_graph_no_dp = self.transformer_graph_no_dp or self.transformer_graph
            transformer_forward_root_no_dp = self.transformer_forward_root_no_dp or self.transformer_forward_root
            transformer_backward_root_no_dp = self.transformer_backward_root_no_dp or self.transformer_backward_root
            moe_transformer_graph_no_dp = self.transformer_graph_moe_no_dp or self.transformer_graph_moe
            moe_transformer_forward_root_no_dp = (
                self.transformer_forward_root_moe_no_dp or self.transformer_forward_root_moe
            )
            moe_transformer_backward_root_no_dp = (
                self.transformer_backward_root_moe_no_dp or self.transformer_backward_root_moe
            )
            dispatcher_no_dp = LLMExecutionDispatcher(
                time_calc=self,
                pipeline_graph=self.pipeline_graph_no_dp,
                pipeline_root=self.pipeline_root_no_dp,
                interconnect_params=self.pipeline_interconnect,
                transformer_graph=transformer_graph_no_dp,
                transformer_forward_root=transformer_forward_root_no_dp,
                transformer_backward_root=transformer_backward_root_no_dp,
                moe_transformer_graph=moe_transformer_graph_no_dp,
                moe_transformer_forward_root=moe_transformer_forward_root_no_dp,
                moe_transformer_backward_root=moe_transformer_backward_root_no_dp,
                no_data_parallel=True,
            )
            try:
                result_no_dp = dispatcher_no_dp.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc
            time_fw_bw_no_dp = result_no_dp.total_time

        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
            moe_transformer_graph=self.transformer_graph_moe,
            moe_transformer_forward_root=self.transformer_forward_root_moe,
            moe_transformer_backward_root=self.transformer_backward_root_moe,
            no_data_parallel=False,
        )
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc
        time_fw_bw = result.total_time


        pipeline_root = result.graph_root
        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = dispatcher.interconnect_params
        memory_root = dispatcher.build_flattened_root_for_memory()
        _, training_peak_gb = mem_estimator.simulate_peak(memory_root, memory_data, mode="training", filename="memory_graph_training") 



        # debug helper. If set, print analytical transformer time and actual transformer time
        if self._generate_graphs:
            print(f"Analytical transformer forward time: {self.transformer_analytical_time_forward:.4f}s")
            print(f"Analytical transformer backward time: {self.transformer_analytical_time_backward:.4f}s")
            if self.transformer_astrasim_time_forward is not None and self.transformer_astrasim_time_backward is not None:
                print(f"Actual transformer forward time: {self.transformer_astrasim_time_forward:.4f}s")
                print(f"Actual transformer backward time: {self.transformer_astrasim_time_backward:.4f}s")

        if self.gradient_accumulation_steps > 1: 
            if time_fw_bw_no_dp is None:
                raise RuntimeError("Missing no-DP timing for gradient accumulation computation")
            self.tot_time = time_fw_bw_no_dp * (self.gradient_accumulation_steps - 1) + time_fw_bw #total time is (time for no-dp steps) + (time for last step with dp)
        else:
            self.tot_time = time_fw_bw
        return self.tot_time

    def estimate_memory_only(self) -> Dict[str, Any]:
        mem_estimator, memory_data = self._build_training_graphs_and_memory_data()

        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
            moe_transformer_graph=self.transformer_graph_moe,
            moe_transformer_forward_root=self.transformer_forward_root_moe,
            moe_transformer_backward_root=self.transformer_backward_root_moe,
            no_data_parallel=False,
        )
        memory_root = dispatcher.build_flattened_root_for_memory()
        _, training_peak_gb = mem_estimator.simulate_peak(
            memory_root,
            memory_data,
            mode="training",
            filename="memory_graph_training",
        )

        return {
            "peak_gb": float(training_peak_gb),
            "capacity_gb": getattr(self, "memory_capacity_per_device_gb", None),
            "headroom_gb": getattr(self, "memory_headroom_gb", None),
            "capacity_exceeded": bool(getattr(self, "memory_capacity_exceeded", False)),
            "capacity_violation_gb": float(getattr(self, "memory_capacity_violation_gb", 0.0) or 0.0),
            "output_dir": self.output_dir,
        }
        
    def _simulate_with_memory( 
        self,
        graph_root: Any,
        memory_data: Dict[str, Any],
        mode: str = "training", #training or inference
        filename: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Run memory-aware simulation and report duration plus peak usage."""

        time_with_memory, peak_mem = self.pipeline_graph.simulate_memory(
            graph_root,
            memory_data,
            mode,
            self.output_dir,
            filename or "memory_graph",
        )
        self.memory_peak_gb = peak_mem

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None and hasattr(self.hw_config, "tech_config"):
            tech_cfg = self.hw_config.tech_config
            if hasattr(tech_cfg, "DRAM"):
                hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - peak_mem
            self.memory_headroom_gb = mem_delta
            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                f"Simulation mode: {mode}",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                f"Simulated peak memory usage(per gpu): {peak_mem:.2f} GiB",
            ]
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                info_lines.append(
                    "ZeRO-3 ephemeral param gather (per gpu): {:.2f} GiB".format(
                        self.zero3_ephemeral_peak_bytes / float(1024 ** 3)
                    )
                )
            if mem_delta < 0:
                info_lines.append(f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.2f} GiB")
                self.memory_capacity_exceeded = True
                self.memory_capacity_violation_gb = max(self.memory_capacity_violation_gb, abs(mem_delta))
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.2f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
            if self._debug_memory:
                breakdown = self._memory_breakdown_debug or {}
                print("[DEEPFLOW] Memory summary (per device):")
                print("  Capacity: {:.2f} GiB".format(hardware_mem_gib))
                print("  Simulated peak usage: {:.2f} GiB".format(peak_mem))
                print("  Headroom: {:.2f} GiB".format(mem_delta))
                if breakdown:
                    print("  Layers per device: {}".format(breakdown.get("layers_per_device")))
                    print("  Breakdown (GiB, approx per device):")
                    # print("    Activations: {:.2f}".format(breakdown.get("activation_gib", 0.0)))
                    print("    Weights: {:.2f}".format(breakdown.get("weight_gib", 0.0)))
                    print("    Gradients: {:.2f}".format(breakdown.get("gradient_gib", 0.0)))
                    print("    Optimizer: {:.2f}".format(breakdown.get("optimizer_gib", 0.0)))
                    print("    Static buffers: {:.2f}".format(breakdown.get("static_gib", 0.0)))
                    # print("    Total (layer-based): {:.2f}".format(breakdown.get("total_layer_gib", 0.0)))
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            if self._debug_memory:
                print("[DEEPFLOW] Memory summary (per device):")
                print("  Capacity: unknown")
                print("  Simulated peak usage: {:.6f} GiB".format(peak_mem))

        return time_with_memory, peak_mem

    def get_time(self):
        return self.tot_time

    def memory_capacity_warning(self) -> Optional[str]:
        violation = max(0.0, getattr(self, "memory_capacity_violation_gb", 0.0))
        if violation > 0.0:
            return (
                f"[WARN] Peak memory exceeds capacity by {violation:.2f} GiB. "
                "Please change parallelism settings for realistic results."
            )
        return None
