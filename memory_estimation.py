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
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

import llm_util


class MemKind(Enum):
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    OPTIMIZER = "optimizer"
    LAYERNORM1 = "layernorm1"
    QKV_PROJ = "qkv_proj"
    ATTENTION = "attention"
    OUTPUT_PROJ = "output_proj"
    LAYERNORM2 = "layernorm2"
    MLP = "MLP"
    ROUTER = "router"
    MOE_DISPATCH = "moe_dispatch"
    MOE_COMBINE = "moe_combine"


_OP_NAME_TO_MEM_KIND: Dict[str, MemKind] = {
    kind.value: kind for kind in MemKind
}


def mem_kind_from_op_name(op_name: str) -> MemKind:
    if op_name in _OP_NAME_TO_MEM_KIND:
        return _OP_NAME_TO_MEM_KIND[op_name]
    raise ValueError(f"Unsupported mem_kind op name: {op_name!r}")


TRANSFORMER_OP_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.LAYERNORM1,
        MemKind.QKV_PROJ,
        MemKind.ATTENTION,
        MemKind.OUTPUT_PROJ,
        MemKind.LAYERNORM2,
        MemKind.MLP,
        MemKind.MOE_DISPATCH,
        MemKind.MOE_COMBINE,
        MemKind.TRANSFORMER,
    }
)


NON_TRANSFORMER_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.EMBEDDING,
        MemKind.SOFTMAX,
        MemKind.OPTIMIZER,
    }
)


class MemoryEstimator:
    """Build memory sizing inputs for graph-based peak memory simulation."""

    def __init__(self, time_calc: Any) -> None:
        self.time_calc = time_calc

    def build_memory_data(
        self,
        *,
        mode: str,
        batch_size: int,
        seq_len: int,
        gemm_shapes: Optional[Dict[str, Any]] = None,
        kv_cache_tokens: Optional[int] = None,
        zero3_ephemeral_peak_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        tc = self.time_calc
        precision = tc.precision

        dp = max(1, int(getattr(tc, "dp", 1)))
        dp_layout = dp
        tp = max(1, int(getattr(tc, "tp", 1)))
        pp = max(1, int(getattr(tc, "pp", 1)))
        cp = max(1, int(getattr(tc, "cp", 1)))
        ep = max(1, int(getattr(tc, "ep", 1)))
        tp_ep = bool(getattr(tc, "tp_ep", True))
        moe_group = ep
        zero_stage = int(getattr(tc, "zero_stage", 0) or 0)
        flash_attention = bool(getattr(tc, "flash_attention", False))
        full_recomputation = bool(getattr(tc, "full_recomputation", False))
        use_moe = bool(getattr(tc, "use_moe", False))
        if mode == "inference":
            dp = 1
            tp_ep = False
            if use_moe and hasattr(tc, "_moe_routing_group"):
                moe_group = max(1, int(tc._moe_routing_group()))
            if cp != 1:
                raise ValueError("Inference memory estimation requires cp=1 (context parallelism is WIP).")
        if use_moe and zero_stage >= 2:
            raise NotImplementedError("MoE memory estimation with ZeRO-2/3 (dp_zero_stage >= 2) is not supported yet.")

        seq_len_eff = float(seq_len) / float(cp)

        (
            transformer_mem_layer_dense,
            transformer_act_layer_dense,
            transformer_act_layer_inf_dense,
            transformer_static_layer_dense,
            gradient_mem_layer_dense,
            optimizer_mem_layer_dense,
            weight_memory_layer_dense,
        ) = llm_util.get_transformer_mem_layer(
            dp=dp,
            tp=tp,
            pp=pp,
            mb=max(1, int(getattr(tc, "mb", 1))),
            batch_size=batch_size,
            hidden_dim=tc.hidden_dim,
            seq_len=seq_len_eff,
            intermediate_size=tc.intermediate_size,
            n_heads=tc.num_heads,
            kv_heads=tc.kv_heads,
            head_dim=getattr(tc, "head_dim", None),
            precision=precision,
            model_type=tc.model_type,
            zero_stage=zero_stage,
            flash_attention=flash_attention,
            full_recomputation=full_recomputation,
        )
        transformer_mem_layer_moe = transformer_mem_layer_dense
        transformer_act_layer_moe = transformer_act_layer_dense
        transformer_act_layer_inf_moe = transformer_act_layer_inf_dense
        transformer_static_layer_moe = transformer_static_layer_dense
        gradient_mem_layer_moe = gradient_mem_layer_dense
        optimizer_mem_layer_moe = optimizer_mem_layer_dense
        weight_memory_layer_moe = weight_memory_layer_dense

        (
            _total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = tc._param_stats_per_rank(tc.hidden_dim, tc.intermediate_size, tc.vocab_size)

        if zero3_ephemeral_peak_bytes is None:
            if zero_stage >= 3 and dp > 1:
                zero3_ephemeral_peak_bytes = max_layer_params * precision.parameters
            else:
                zero3_ephemeral_peak_bytes = 0.0

        extra_static_bytes_per_device: Dict[int, float] = {}

        bytes_per_param_weight = 0.0
        bytes_per_param_grad = 0.0
        bytes_per_param_opt = 0.0
        if params_per_layer_per_rank:
            denom = float(params_per_layer_per_rank)
            bytes_per_param_weight = float(weight_memory_layer_dense) / denom
            if mode == "training":
                bytes_per_param_grad = float(gradient_mem_layer_dense) / denom
                bytes_per_param_opt = float(optimizer_mem_layer_dense) / denom

        per_param_static_bytes = (
            bytes_per_param_weight + bytes_per_param_grad + bytes_per_param_opt
        )
        embedding_static_bytes = float(embedding_params_per_rank) * per_param_static_bytes
        lm_head_static_bytes = float(output_params_per_rank) * per_param_static_bytes

        if use_moe:
            moe_intermediate = int(getattr(tc, "moe_intermediate_size", tc.intermediate_size))
            (
                _,
                transformer_act_layer_moe,
                transformer_act_layer_inf_moe,
                _,
                _,
                _,
                _,
            ) = llm_util.get_transformer_mem_layer(
                dp=dp,
                tp=tp,
                pp=pp,
                mb=max(1, int(getattr(tc, "mb", 1))),
                batch_size=batch_size,
                hidden_dim=tc.hidden_dim,
                seq_len=seq_len_eff,
                intermediate_size=moe_intermediate,
                n_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                head_dim=getattr(tc, "head_dim", None),
                precision=precision,
                model_type=tc.model_type,
                zero_stage=zero_stage,
                flash_attention=flash_attention,
                full_recomputation=full_recomputation,
            )
            ffn_proj_factor = 3 if llm_util.is_llama_style(tc.model_type) else 2
            if llm_util.is_glm_style(tc.model_type):
                _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                    tc.hidden_dim,
                    tc.num_heads,
                    tc.kv_heads,
                    head_dim=getattr(tc, "head_dim", None),
                )
                attention_params = (tc.hidden_dim * (q_size + 2 * kv_size)) + (q_size * tc.hidden_dim)
            else:
                attention_params = 4 * tc.hidden_dim * tc.hidden_dim
            expert_param = ffn_proj_factor * moe_intermediate * tc.hidden_dim
            routed_experts = int(getattr(tc, "moe_num_experts", 1))
            shared_experts = int(getattr(tc, "n_shared_experts", 0))
            routed_experts_per_rank = routed_experts / float(max(1, moe_group))
            # TODO: Shared experts are modeled as fully replicated across EP for now.
            if tp_ep:
                routed_params_per_rank = (expert_param / float(max(1, tp))) * routed_experts_per_rank
            else:
                routed_params_per_rank = expert_param * routed_experts_per_rank
            if tp_ep:
                shared_params_per_rank = (expert_param / float(max(1, tp))) * shared_experts
            else:
                shared_params_per_rank = expert_param * shared_experts
            expert_params_per_rank = routed_params_per_rank + shared_params_per_rank
            router_params_per_rank = tc.hidden_dim * routed_experts
            attention_params_per_rank = attention_params / float(max(1, tp))
            params_per_layer_moe = (
                attention_params_per_rank + expert_params_per_rank + router_params_per_rank
            )
            weight_memory_layer_moe = params_per_layer_moe * bytes_per_param_weight
            if mode == "training":
                gradient_mem_layer_moe = params_per_layer_moe * bytes_per_param_grad
                optimizer_mem_layer_moe = params_per_layer_moe * bytes_per_param_opt
            else:
                gradient_mem_layer_moe = 0.0
                optimizer_mem_layer_moe = 0.0
            transformer_static_layer_moe = weight_memory_layer_moe + gradient_mem_layer_moe + optimizer_mem_layer_moe
        const_mem_offset_bytes = 0.0
        hw_config = getattr(tc, "hw_config", None)
        sw_config = getattr(hw_config, "sw_config", None) if hw_config is not None else None
        if sw_config is not None:
            const_mem_offset_bytes = float(getattr(sw_config, "const_mem_offset", 0.0) or 0.0)

        def _build_rank_layout():
            hw_config = getattr(tc, "hw_config", None)
            layout = getattr(hw_config, "network_layout", None)
            dimensions = getattr(layout, "dimensions", None) if layout is not None else None
            if not dimensions:
                return None

            ep = max(1, int(getattr(tc, "ep", 1)))
            axis_sizes = {"tp": tp, "cp": cp, "ep": ep, "pp": pp, "dp": dp_layout}
            axis_order = []
            for dim in dimensions:
                dim_axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ()) or ()]
                for name in dim_axes:
                    if name not in axis_sizes:
                        raise ValueError(
                            f"Unsupported parallelism axis '{name}' in network layout. "
                            "Supported axes for memory estimation are: tp, cp, ep, pp, dp."
                        )
                    if name not in axis_order:
                        axis_order.append(name)

                declared = int(getattr(dim, "size", 1))
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
            if not axis_order:
                return None
            if tp > 1 and "tp" not in axis_order:
                raise ValueError("Network layout must include 'tp' when tensor parallelism > 1.")
            if cp > 1 and "cp" not in axis_order:
                raise ValueError("Network layout must include 'cp' when context parallelism > 1.")
            if ep > 1 and "ep" not in axis_order:
                raise ValueError("Network layout must include 'ep' when expert parallelism > 1.")
            if pp > 1 and "pp" not in axis_order:
                raise ValueError("Network layout must include 'pp' when pipeline parallelism > 1.")

            axis_strides = {}
            span = 1
            for axis in axis_order:
                axis_strides[axis] = span
                span *= axis_sizes[axis]
            return axis_order, axis_sizes, axis_strides

        def _hw_id_for_rank(stage_id: int, tp_rank: int, layout):
            if layout is None:
                return stage_id * par_degree + tp_rank
            axis_order, axis_sizes, axis_strides = layout
            coords = {}
            tp_size = axis_sizes.get("tp", 1)
            cp_size = axis_sizes.get("cp", 1)
            ep_size = axis_sizes.get("ep", 1)
            if "tp" in axis_order:
                coords["tp"] = tp_rank % tp_size
            if "cp" in axis_order:
                coords["cp"] = (tp_rank // tp_size) % cp_size
            if "ep" in axis_order:
                coords["ep"] = (tp_rank // max(1, tp_size * cp_size)) % ep_size
            if "pp" in axis_order:
                if stage_id < 0 or stage_id >= axis_sizes.get("pp", 1):
                    raise ValueError(f"stage_id {stage_id} is out of range for pp={axis_sizes.get('pp', 1)}")
                coords["pp"] = stage_id % axis_sizes["pp"]

            linear_rank = 0
            for axis in axis_order:
                coord = coords.get(axis, 0)
                size = axis_sizes.get(axis, 1)
                if coord < 0 or coord >= size:
                    raise ValueError(f"Coordinate {coord} for axis '{axis}' is out of range <{size}")
                stride = axis_strides.get(axis)
                if stride is None:
                    raise KeyError(f"Rank layout stride missing for axis '{axis}'")
                linear_rank += coord * stride
            return linear_rank

        par_degree = max(1, tp * cp * ep)
        layout = None
        if embedding_static_bytes or lm_head_static_bytes or const_mem_offset_bytes:
            layout = _build_rank_layout()
        if embedding_static_bytes or lm_head_static_bytes:
            last_stage = max(0, pp - 1)
            for tp_rank in range(par_degree):
                if embedding_static_bytes:
                    hw_id = _hw_id_for_rank(0, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + embedding_static_bytes
                if lm_head_static_bytes:
                    hw_id = _hw_id_for_rank(last_stage, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + lm_head_static_bytes
        if const_mem_offset_bytes:
            for stage_id in range(pp):
                for tp_rank in range(par_degree):
                    hw_id = _hw_id_for_rank(stage_id, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = (
                        extra_static_bytes_per_device.get(hw_id, 0.0) + const_mem_offset_bytes
                    )

        kv_cache_bytes_per_layer = 0.0
        if gemm_shapes is None:
            gemm_shapes_moe = llm_util.process_gemm_shapes(
                tc,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=tc.hidden_dim,
                num_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                intermediate_size=tc.intermediate_size,
                vocab_size=tc.vocab_size,
            )
        else:
            gemm_shapes_moe = gemm_shapes

        gemm_shapes_dense = gemm_shapes_moe
        if use_moe:
            from types import SimpleNamespace

            dense_ctx = SimpleNamespace(
                use_moe=False,
                moe_num_experts=int(getattr(tc, "moe_num_experts", 1)),
                moe_top_k=int(getattr(tc, "moe_top_k", 1)),
                moe_intermediate_size=tc.intermediate_size,
                model_type=tc.model_type,
                head_dim=getattr(tc, "head_dim", None),
            )
            gemm_shapes_dense = llm_util.process_gemm_shapes(
                dense_ctx,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=tc.hidden_dim,
                num_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                intermediate_size=tc.intermediate_size,
                vocab_size=tc.vocab_size,
            )

        gemm_type_map = {
            "qkv_proj": "qkv",
            "attention_score": "attention_score",
            "attention_output": "attention_output",
            "output_proj": "out_proj",
            "ffn1": "ffn1",
            "ffn2": "ffn2",
            "linear": "linear_softmax",
            "router": "ffn1",
        }

        def _build_activation_maps(
            gemm_shapes_local: Dict[str, Any],
            transformer_act_layer_local: float,
            transformer_act_layer_inf_local: float,
            use_moe_local: bool,
        ) -> Tuple[Dict[MemKind, float], Dict[MemKind, float], Dict[MemKind, float]]:
            def _gemm_out_bytes(key: str, gemm_type: str) -> float:
                if key not in gemm_shapes_local:
                    raise RuntimeError(f"Missing GEMM shape for '{key}'")
                elements = tc._gemm_output_elements(gemm_shapes_local[key], gemm_type)
                return float(elements) * precision.activations

            qkv_bytes = _gemm_out_bytes("qkv_proj", gemm_type_map["qkv_proj"])
            attn_out_bytes = _gemm_out_bytes("attention_output", gemm_type_map["attention_output"])
            out_proj_bytes = _gemm_out_bytes("output_proj", gemm_type_map["output_proj"])
            ffn1_bytes = _gemm_out_bytes("ffn1", gemm_type_map["ffn1"])
            ffn2_bytes = _gemm_out_bytes("ffn2", gemm_type_map["ffn2"])
            attn_score_bytes = 0.0
            if not flash_attention:
                attn_score_bytes = _gemm_out_bytes("attention_score", gemm_type_map["attention_score"])

            softmax_out_bytes = 0.0
            if "linear" in gemm_shapes_local:
                softmax_out_bytes = _gemm_out_bytes("linear", gemm_type_map["linear"])
            else:
                tokens = float(batch_size) * float(seq_len_eff)
                vocab_shard = math.ceil(float(tc.vocab_size) / float(max(1, tp * cp)))
                softmax_out_bytes = tokens * float(vocab_shard) * float(precision.activations)

            # TODO: figure out router memory estimation (minor, interacts with sp/cp in non intuitive ways)
        
            # Activation output sizes by operation.
            # LayerNorm outputs have shape (batch, seq, hidden_dim), same as out_proj.
            # When FlashAttention is OFF, we store the softmax output (same shape as
            # attention scores) for backward; this is added to ATTENTION below.
            base_outputs = {
                MemKind.LAYERNORM1: out_proj_bytes,  # (batch, seq, hidden_dim)
                MemKind.QKV_PROJ: qkv_bytes,
                MemKind.ATTENTION: attn_out_bytes,
                MemKind.OUTPUT_PROJ: out_proj_bytes,  # (batch, seq, hidden_dim)
                MemKind.LAYERNORM2: out_proj_bytes,  # (batch, seq, hidden_dim)
                MemKind.MLP: ffn2_bytes,
                MemKind.EMBEDDING: out_proj_bytes,
                MemKind.SOFTMAX: softmax_out_bytes,
                MemKind.OPTIMIZER: 0.0,
                MemKind.MOE_DISPATCH: 0.0,
                MemKind.MOE_COMBINE: 0.0,
            }
            parallelism_mode = None
            try:
                parallelism_mode = tc.get_parallelism_mode()
            except AttributeError:
                parallelism_mode = None
            if parallelism_mode is not None:
                mode_value = str(getattr(parallelism_mode, "value", parallelism_mode)).lower()
                if mode_value.endswith("tensor_sequence"):
                    seq_degree = max(1, int(tc._sequence_parallel_degree()))
                    hidden_full_bytes = (
                        float(batch_size)
                        * float(seq_len)
                        * float(tc.hidden_dim)
                        * float(precision.activations)
                    )
                    hidden_shard_bytes = (
                        float(batch_size)
                        * float(math.ceil(float(seq_len) / float(seq_degree)))
                        * float(tc.hidden_dim)
                        * float(precision.activations)
                    )
                    base_outputs[MemKind.OUTPUT_PROJ] = hidden_shard_bytes
                    base_outputs[MemKind.MLP] = hidden_shard_bytes
                    base_outputs[MemKind.LAYERNORM1] = hidden_full_bytes
                    base_outputs[MemKind.LAYERNORM2] = hidden_full_bytes

            transformer_fallback_bytes = transformer_act_layer_local
            if mode != "training":
                transformer_fallback_bytes = transformer_act_layer_inf_local
            mlp_bytes_base = base_outputs[MemKind.MLP] + ffn1_bytes
            if use_moe_local:
                seq_divisor = cp if cp > 1 else 1
                tokens_owner = float(batch_size) * math.ceil(float(seq_len) / float(seq_divisor))
                moe_scale = 1.0
                if tokens_owner > 0:
                    tokens_dispatched = tokens_owner * float(getattr(tc, "moe_top_k", 1))
                    tokens_local = math.ceil(tokens_dispatched / float(max(1, moe_group)))
                    if mode != "training":
                        experts_per_rank = int(
                            max(1, float(getattr(tc, "moe_num_experts", 1)) / float(max(1, moe_group)))
                        )
                        if experts_per_rank > 0 and (tokens_local % experts_per_rank) != 0:
                            tokens_local = math.ceil(tokens_local / float(experts_per_rank)) * experts_per_rank
                    moe_scale = (float(tokens_local) / float(tokens_owner)) + float(
                        getattr(tc, "n_shared_experts", 0)
                    )
                # TODO: Shared experts are modeled as replicated across EP for now.
                base_outputs[MemKind.MLP] *= moe_scale
                ffn1_bytes *= moe_scale
            mlp_bytes_scaled = base_outputs[MemKind.MLP] + ffn1_bytes
            if use_moe_local:
                transformer_fallback_bytes += float(mlp_bytes_scaled - mlp_bytes_base)
            base_outputs[MemKind.TRANSFORMER] = transformer_fallback_bytes

            persistent_bytes_by_kind = {kind: 0.0 for kind in base_outputs}
            transient_bytes_by_kind = {kind: 0.0 for kind in base_outputs}

            store_outputs = mode == "training" and not full_recomputation
            if store_outputs:
                for kind, bytes_val in base_outputs.items():
                    persistent_bytes_by_kind[kind] = float(bytes_val or 0.0)
                if attn_score_bytes:
                    persistent_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
                persistent_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)
            else:
                for kind, bytes_val in base_outputs.items():
                    transient_bytes_by_kind[kind] = float(bytes_val or 0.0)
                if attn_score_bytes:
                    transient_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
                transient_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)

            return base_outputs, persistent_bytes_by_kind, transient_bytes_by_kind

        _, persistent_bytes_by_kind_dense, transient_bytes_by_kind_dense = _build_activation_maps(
            gemm_shapes_dense,
            transformer_act_layer_dense,
            transformer_act_layer_inf_dense,
            False,
        )
        persistent_bytes_by_kind_moe = persistent_bytes_by_kind_dense
        transient_bytes_by_kind_moe = transient_bytes_by_kind_dense
        if use_moe:
            base_outputs_moe, persistent_bytes_by_kind_moe, transient_bytes_by_kind_moe = _build_activation_maps(
                gemm_shapes_moe,
                transformer_act_layer_moe,
                transformer_act_layer_inf_moe,
                True,
            )
            if mode == "training":
                transformer_act_layer_moe = float(
                    base_outputs_moe.get(MemKind.TRANSFORMER, transformer_act_layer_moe)
                )
            else:
                transformer_act_layer_inf_moe = float(
                    base_outputs_moe.get(MemKind.TRANSFORMER, transformer_act_layer_inf_moe)
                )
        persistent_bytes_by_kind = persistent_bytes_by_kind_dense
        transient_bytes_by_kind = transient_bytes_by_kind_dense

        if mode != "training" and kv_cache_tokens:
            # KV cache stores K and V tensors with shape (kv_heads, head_dim, seq).
            # For GQA/MQA, kv_heads < num_heads, so we must use kv_heads directly
            # rather than deriving from the attention score GEMM (which uses Q heads).
            kv_heads = int(getattr(tc, "kv_heads", tc.num_heads))
            head_dim = getattr(tc, "head_dim", None)
            if head_dim is None:
                head_dim = tc.hidden_dim // tc.num_heads
            kv_heads_per_tp = math.ceil(kv_heads / tp)
            kv_tokens = int(kv_cache_tokens)
            kv_cache_bytes_per_layer = (
                float(batch_size)
                * float(kv_heads_per_tp)
                * float(head_dim)
                * float(kv_tokens)
                * float(precision.kv_cache)
                * 2.0  # K and V
            )

        param_gather_bytes = 0.0
        if mode == "training" and zero3_ephemeral_peak_bytes and dp > 1 and zero_stage >= 3:
            param_gather_bytes = float(zero3_ephemeral_peak_bytes)

        if use_moe:
            transformer_mem_layer_moe = transformer_act_layer_moe + weight_memory_layer_moe

        return {
            "activation_mem_per_layer": transformer_act_layer_dense,
            "activation_mem_per_layer_inference": transformer_act_layer_inf_dense,
            "weight_mem_per_layer": weight_memory_layer_dense,
            "gradient_mem_per_layer": gradient_mem_layer_dense,
            "optimizer_mem_per_layer": optimizer_mem_layer_dense,
            "static_mem_per_layer": transformer_static_layer_dense,
            "total_mem_per_layer": transformer_mem_layer_dense,
            "activation_mem_per_layer_dense": transformer_act_layer_dense,
            "activation_mem_per_layer_moe": transformer_act_layer_moe,
            "activation_mem_per_layer_inference_dense": transformer_act_layer_inf_dense,
            "activation_mem_per_layer_inference_moe": transformer_act_layer_inf_moe,
            "weight_mem_per_layer_dense": weight_memory_layer_dense,
            "weight_mem_per_layer_moe": weight_memory_layer_moe,
            "gradient_mem_per_layer_dense": gradient_mem_layer_dense,
            "gradient_mem_per_layer_moe": gradient_mem_layer_moe,
            "optimizer_mem_per_layer_dense": optimizer_mem_layer_dense,
            "optimizer_mem_per_layer_moe": optimizer_mem_layer_moe,
            "static_mem_per_layer_dense": transformer_static_layer_dense,
            "static_mem_per_layer_moe": transformer_static_layer_moe,
            "total_mem_per_layer_dense": transformer_mem_layer_dense,
            "total_mem_per_layer_moe": transformer_mem_layer_moe,
            "persistent_bytes_by_kind": persistent_bytes_by_kind,
            "transient_bytes_by_kind": transient_bytes_by_kind,
            "persistent_bytes_by_kind_dense": persistent_bytes_by_kind_dense,
            "persistent_bytes_by_kind_moe": persistent_bytes_by_kind_moe,
            "transient_bytes_by_kind_dense": transient_bytes_by_kind_dense,
            "transient_bytes_by_kind_moe": transient_bytes_by_kind_moe,
            "extra_static_bytes_per_device": extra_static_bytes_per_device,
            "kv_cache_bytes_per_layer": kv_cache_bytes_per_layer,
            "zero3_ephemeral_peak_bytes": zero3_ephemeral_peak_bytes,
            "param_gather_bytes": param_gather_bytes,
        }

    def simulate_peak(
        self,
        graph_root: Any,
        memory_data: Dict[str, Any],
        *,
        mode: str,
        filename: Optional[str] = None,
    ) -> Any:
        """Run memory simulation on the provided graph root."""
        def _is_non_flattened(root: Any) -> bool:
            stack = list(root if isinstance(root, (list, tuple)) else [root])
            visited = set()
            while stack:
                current = stack.pop()
                if current is None:
                    continue
                current_id = id(current)
                if current_id in visited:
                    continue
                visited.add(current_id)

                name = getattr(current, "name", "")
                if isinstance(name, str) and name.startswith("transformer_layer"):
                    mem_kind = getattr(current, "mem_kind", None)
                    if mem_kind == MemKind.TRANSFORMER:
                        return True

                children = getattr(current, "children", None)
                if isinstance(children, (list, tuple)):
                    stack.extend(children)
                elif children is not None:
                    stack.append(children)
            return False

        if _is_non_flattened(graph_root):
            raise RuntimeError(
                "Memory simulation requires a flattened graph. "
                "Use LLMExecutionDispatcher.build_flattened_root_for_memory()."
            )
        return self.time_calc._simulate_with_memory(
            graph_root,
            memory_data,
            mode=mode,
            filename=filename,
        )


def estimate_inference_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    return llm_util.estimate_inference_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
        **overrides,
    )


def estimate_training_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    from train_timing import TimeCalculationLLM

    tc = TimeCalculationLLM(exp_hw_config, exp_model_config, mode, output_dir=output_dir)
    return tc.estimate_memory_only()


def estimate_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    run_type = str(getattr(exp_model_config.model_config, "run_type", "training")).lower()
    if run_type == "inference":
        return estimate_inference_memory(
            exp_hw_config,
            exp_model_config,
            mode=mode,
            output_dir=output_dir,
            **overrides,
        )
    return estimate_training_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
    )
