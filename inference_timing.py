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

"""LLM inference prefill time-calculation entry points."""

import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Mapping, Set
from train_timing import (
    LLMExecutionDispatcher,
    TimeCalculationLLM,
    GemmType,
    COMMUNICATION_RULES,
    COMM_RULE_DEFAULT_KEY,
)
from memory_estimation import MemoryEstimator
from simulate_inference_graph import DecodeSample, InferenceConfig, InferenceEngine
import llm_util
import json
from timing_model import CollectiveType, CommSpec, DirectionTiming, OperationTiming, OperationGroup

def convert_prefix(value: float) -> float:
    """Assign SI unit prefixes to numerical values."""
    if value > 1:
        prefixes = ["", "k", "M", "G"]
        index = min(int(math.log10(value) // 3), len(prefixes) - 1)
        scaled_value = value / (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"
    else:
        prefixes = ["", "m", "µ", "n"]
        index = min(int(-(math.log10(value) // 3)), len(prefixes) - 1)
        scaled_value = value * (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"

class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        self._raw_model_config = model_config

    def _build_decode_transformer_results(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        use_moe_layer: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[Dict[str, OperationTiming], Dict[str, float]]:
        """Construct transformer timings + node breakdown for a single decode step."""

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads

        token_bytes = llm_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        intermediate_size = self.moe_intermediate_size if use_moe_layer else self.intermediate_size
        gemm_ctx = self
        if use_moe_layer != self.use_moe:
            gemm_ctx = SimpleNamespace(
                use_moe=use_moe_layer,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
            )
        if gemm_shapes is None or use_moe_layer != self.use_moe:
            gemm_shapes = llm_util.process_decode_gemm_shapes(
                gemm_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        # FlashAttention is not used during single-token (incremental) decoding.


        output_seq_len = 1

        transformer_timings: Dict[str, OperationTiming] = {}

        def _make_forward(
            op_name: str,
            compute_time: float,
            comm_time: float,
            comm_bytes: float,
            *,
            flops: float = 0.0,
            memory: Optional[Mapping[str, float]] = None,
        ) -> DirectionTiming:
            bytes_int = int(math.ceil(float(comm_bytes or 0.0)))
            return DirectionTiming(
                compute_time=compute_time,
                comm_time=comm_time,
                comm_bytes=bytes_int,
                flops=flops,
                memory_accesses=dict(memory) if memory else {},
            )

        # QKV projection
        qkv_proj_time, qkv_proj_reduction, qkv_proj_size, qkv_proj_flops, qkv_proj_mem = self.parallelism_gemm_forward(
            gemm_qkv_proj, "decode_qkv_proj_f", gemm_type=GemmType.QKV
        )
        transformer_timings["qkv_proj"] = OperationTiming(
            "qkv_proj",
            forward=_make_forward(
                "qkv_proj",
                compute_time=qkv_proj_time,
                comm_time=qkv_proj_reduction,
                comm_bytes=qkv_proj_size,
                flops=qkv_proj_flops,
                memory=self._mem_levels(qkv_proj_mem),
            ),
            backward=None,
        )

        # Attention components
        attention_score_time, attention_score_reduction, attention_score_size, attention_score_flops, attention_score_mem = self.parallelism_gemm_forward(
            gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE
        )
        attention_output_time, attention_output_reduction, attention_output_size, attention_output_flops, attention_output_mem = self.parallelism_gemm_forward(
            gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT
        )
        attention_scale_softmax_f = self.get_scale_softmax_f(gemm_attention_score)

        attention_reduction = attention_score_reduction + attention_output_reduction
        attention_comm_bytes = attention_score_size + attention_output_size
        attention_forward_compute = attention_score_time + attention_scale_softmax_f + attention_output_time
        attention_forward_time = attention_forward_compute + attention_reduction
        attention_flops = (attention_score_flops or 0.0) + (attention_output_flops or 0.0)
        attention_mem = self._combine_mem(attention_score_mem, attention_output_mem)

        transformer_timings["attention"] = OperationTiming(
            "attention",
            forward=_make_forward(
                "attention",
                compute_time=attention_forward_compute,
                comm_time=attention_reduction,
                comm_bytes=attention_comm_bytes,
                flops=attention_flops,
                memory=attention_mem,
            ),
            backward=None,
        )

        attention_scale_softmax_op = OperationTiming(
            "attention_scale_softmax",
            forward=_make_forward(
                "attention_scale_softmax",
                compute_time=attention_scale_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
            ),
            backward=None,
        )
        transformer_timings["attention_scale_softmax"] = attention_scale_softmax_op

        # Output projection
        out_proj_time, out_proj_reduction, out_proj_size, out_proj_flops, out_proj_mem = self.parallelism_gemm_forward(
            gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ
        )
        transformer_timings["output_proj"] = OperationTiming(
            "output_proj",
            forward=_make_forward(
                "output_proj",
                compute_time=out_proj_time,
                comm_time=out_proj_reduction,
                comm_bytes=out_proj_size,
                flops=out_proj_flops,
                memory=self._mem_levels(out_proj_mem),
            ),
            backward=None,
        )

        # FFN layers (dense vs MoE)
        router_time_f = 0.0
        router_comm_f = 0.0
        router_bytes_f = 0.0
        dispatch_fwd_time = 0.0
        dispatch_fwd_bytes = 0.0
        combine_fwd_time = 0.0
        moe_tokens_local = None
        moe_tokens_shared = None
        if not use_moe_layer:
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.parallelism_gemm_forward(
                gemm_ffn1, "decode_ffn1_f", gemm_type=GemmType.FFN1
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.parallelism_gemm_forward(
                gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2
            )
        else:
            gemm_router = gemm_shapes.get("router")
            if gemm_router is None:
                raise KeyError("Missing decode GEMM shape for 'router'")
            allow_padding = self._moe_allow_padding()
            (
                tokens_owner,
                tokens_dispatched,
                tokens_local,
                _experts_per_rank,
                _tokens_per_expert,
            ) = self._moe_routed_tokens_per_expert(
                batch_size,
                output_seq_len,
                allow_padding=allow_padding,
            )
            moe_tokens_local = tokens_local
            moe_tokens_shared = self._moe_tokens_shared(tokens_owner)
            router_time_f, router_comm_f, router_bytes_f = self.get_router_f(
                gemm_router,
                gemm_ffn1,
                batch_size=batch_size,
                seq_len=output_seq_len,
            )
            moe_group = self._moe_routing_group()
            axis = None
            dispatch_fwd_bytes = int(
                math.ceil(self.precision.activations * tokens_dispatched * self.hidden_dim)
            )
            dispatch_fwd_time = self.network_model.collective(
                kind=CollectiveType.ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0.0,
                debug_label="decode_moe_dispatch_f_all_to_all",
                axis=axis,
            )
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.get_moe_ffn_f(
                gemm_ffn1,
                "decode_ffn1_f",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.get_moe_ffn_f(
                gemm_ffn2,
                "decode_ffn2_f",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            combine_fwd_time = self.network_model.collective(
                kind=CollectiveType.ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0.0,
                debug_label="decode_moe_combine_f_all_to_all",
                axis=axis,
            )
            transformer_timings["router"] = OperationTiming(
                "router",
                forward=_make_forward(
                    "router",
                    compute_time=router_time_f,
                    comm_time=router_comm_f,
                    comm_bytes=router_bytes_f,
                ),
                backward=None,
            )
            transformer_timings["moe_dispatch"] = OperationTiming(
                "moe_dispatch",
                forward=_make_forward(
                    "moe_dispatch",
                    compute_time=0.0,
                    comm_time=dispatch_fwd_time,
                    comm_bytes=dispatch_fwd_bytes,
                ),
                backward=None,
            )
            transformer_timings["moe_combine"] = OperationTiming(
                "moe_combine",
                forward=_make_forward(
                    "moe_combine",
                    compute_time=0.0,
                    comm_time=combine_fwd_time,
                    comm_bytes=dispatch_fwd_bytes,
                ),
                backward=None,
            )

        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=_make_forward(
                "ffn1",
                compute_time=ffn1_time,
                comm_time=ffn1_reduction,
                comm_bytes=ffn1_size,
                flops=ffn1_flops,
                memory=self._mem_levels(ffn1_mem),
            ),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=_make_forward(
                "ffn2",
                compute_time=ffn2_time,
                comm_time=ffn2_reduction,
                comm_bytes=ffn2_size,
                flops=ffn2_flops,
                memory=self._mem_levels(ffn2_mem),
            ),
            backward=None,
        )

        # GELU/SwiGLU activation
        ffn1_spec = self._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        ffn1_activation_shape_shared = None
        if use_moe_layer and moe_tokens_local is not None:
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            ffn1_n = ffn1_spec.shard_n if use_tp_sharded else ffn1_spec.n
            ffn1_activation_shape = (moe_tokens_local, ffn1_spec.k, ffn1_n)
            if moe_tokens_shared and moe_tokens_shared > 0:
                ffn1_activation_shape_shared = (moe_tokens_shared, ffn1_spec.k, ffn1_n)
        if llm_util.is_llama_style(self.model_type):
            act_f = self.get_swiglu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_swiglu_f(ffn1_activation_shape_shared)
        else:
            act_f = self.get_gelu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_gelu_f(ffn1_activation_shape_shared)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=_make_forward("gelu", compute_time=act_f, comm_time=0.0, comm_bytes=0.0),
            backward=None,
        )

        # Layer norms
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        q_size = self.num_heads * head_dim
        output_proj_shape = (
            batch_size,
            output_seq_len,
            q_size,
            self.hidden_dim,
        )
        residual1_f = self.get_residual_f(output_proj_shape)
        layernorm1_f, layernorm1_reduction, layernorm1_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=_make_forward(
                "layernorm1",
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction,
                comm_bytes=layernorm1_bytes,
            ),
            backward=None,
        )

        ffn2_shape = (
            batch_size,
            output_seq_len,
            intermediate_size,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f, layernorm2_reduction, layernorm2_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=_make_forward(
                "layernorm2",
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction,
                comm_bytes=layernorm2_bytes,
            ),
            backward=None,
        )

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(linear_shape)
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=_make_forward(
                "linear_softmax",
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
                memory=self._mem_levels(linear_softmax_mem),
            ),
            backward=None,
        )

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        # Match exact floating-point operation order from original code
        qkv_proj_forward = qkv_proj_time + qkv_proj_reduction
        attention_forward = attention_score_time + attention_scale_softmax_f + attention_output_time + attention_reduction
        out_proj_forward = out_proj_time + out_proj_reduction
        mha_forward = qkv_proj_forward + attention_forward + out_proj_forward

        ffn1_forward = ffn1_time + ffn1_reduction
        ffn2_forward = ffn2_time + ffn2_reduction
        mlp_forward = ffn1_forward + act_f + ffn2_forward
        if use_moe_layer:
            mlp_forward += router_time_f + dispatch_fwd_time + combine_fwd_time

        layernorm1_forward = residual1_f + layernorm1_f
        layernorm2_forward = residual2_f + layernorm2_f

        transformer_forward = (
            mha_forward
            + mlp_forward
            + layernorm1_forward
            + layernorm1_reduction
            + layernorm2_forward
            + layernorm2_reduction
        )

        node_breakdown = {
            "transformer_time_f": transformer_forward,
            "transformer_time_b": 0.0,
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
        }

        return transformer_timings, node_breakdown


    def prepare_decode_graphs(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        moe_layers_active = bool(self.use_moe and any(getattr(self, "moe_layer_mask", []) or []))
        moe_intermediate = self.moe_intermediate_size
        decode_gemm_shapes_moe = gemm_shapes
        if decode_gemm_shapes_moe is None:
            decode_gemm_shapes_moe = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=moe_intermediate if self.use_moe else self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )
        decode_gemm_shapes_dense = decode_gemm_shapes_moe
        if self.use_moe:
            dense_ctx = SimpleNamespace(
                use_moe=False,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=self.intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
            )
            decode_gemm_shapes_dense = llm_util.process_decode_gemm_shapes(
                dense_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        transformer_timings, node_breakdown = self._build_decode_transformer_results(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            use_moe_layer=False,
            gemm_shapes=decode_gemm_shapes_dense,
        )
        moe_transformer_timings = None
        moe_node_breakdown = None
        if moe_layers_active:
            moe_transformer_timings, moe_node_breakdown = self._build_decode_transformer_results(
                batch_size=batch_size,
                total_seq_len=total_seq_len,
                use_moe_layer=True,
                gemm_shapes=decode_gemm_shapes_moe,
            )

        output_act_bytes = decode_gemm_shapes_dense["qkv_proj"][0] * decode_gemm_shapes_dense["qkv_proj"][1] * self.precision_bytes
        energy = self.calc_energy(transformer_timings, output_act_bytes)

        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "decode_transformer_results.txt")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )

        return self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
            batch_size=batch_size,
            seq_len=1,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
            gemm_shapes=decode_gemm_shapes_moe if self.use_moe else decode_gemm_shapes_dense,
        ), energy
    
    def calc_energy(self, transformer_timings: Dict[str, OperationTiming], cross_layer_comm) -> float:
        """
        Calculate energy consumption based on transformer results.
        """
        if getattr(self, "use_moe", False) and not getattr(self, "_moe_energy_warning_emitted", False):
            warning = (
                "!!! WARNING: MoE energy estimates are not reliable.\n"
                "!!! WARNING: Router/dispatch/combine (and some MoE-specific comms) are not modeled in energy.\n"
                "!!! WARNING: Treat reported energy numbers as lower bounds for MoE runs."
            )
            print(warning)
            self._moe_energy_warning_emitted = True
        # NOTE: MoE energy estimation is incomplete; router/dispatch/combine are not modeled.
        total_flops = 0.0
        total_hbm_bytes = 0.0
        inter_comm_bytes = 0.0  # data parallelism?

        aggregate_groups = {
            "MLP": ("ffn1", "gelu", "ffn2"),
        }
        solo_ops = ("layernorm1", "layernorm2", "embedding", "linear_softmax", "qkv_proj", "attention", "output_proj")

        for members in aggregate_groups.values():
            for name in members:
                op = transformer_timings.get(name)
                if op is None:
                    continue
                total_flops += op.forward.flops
                total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
                inter_comm_bytes += op.forward.comm_bytes

        for name in solo_ops:
            op = transformer_timings.get(name)
            if op is None:
                continue
            total_flops += op.forward.flops
            total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
            inter_comm_bytes += op.forward.comm_bytes

        total_comm_bytes = self.num_layers * inter_comm_bytes + (self.pp - 1) * cross_layer_comm

        energy_per_flop = self.core.nominal_energy_per_flop
        energy_hbm_byte = self.DRAM.dynamic_energy_per_bit * 8
        # TODO: honor per-dimension interconnect energies; currently assumes all comms use dimension 0.
        energy_comm_byte = (self.network.energies_per_bit[0] if self.network.energies_per_bit else 0.0) * 8

        total_energy = (total_flops * energy_per_flop) + \
            (total_hbm_bytes * energy_hbm_byte) + \
                (total_comm_bytes * energy_comm_byte)   
        
        return total_energy



    def calc_time(self) -> Tuple[float, float]:
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        decode_len = self.model.decode_len
        prefill_len = self.seq_len - decode_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads

        total_time = 0.0
        total_energy = 0.0
        prefill_peak_gb = 0.0
        mem_estimator = MemoryEstimator(self)

        if prefill_len <= 0:
            print("Skipping prefill")
            self.pipeline_graph = None
            self.pipeline_root = None
            self.pipeline_interconnect = None
            self.transformer_graph = None
            self.transformer_forward_root = None
            self.transformer_backward_root = None
            self.transformer_graph_moe = None
            self.transformer_forward_root_moe = None
            self.transformer_backward_root_moe = None
            self.transformer_analytical_time_forward = None
            self.transformer_analytical_time_backward = None
        else:
            num_SMs = self.hw_config.tech_config.core.num_bundles
            transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                prefill_len,
                num_heads,
                kv_heads,
                intermediate_size,
                num_SMs,
                use_moe_override=False,
            )
            moe_transformer_timings = None
            moe_node_breakdown = None
            if self.use_moe and any(getattr(self, "moe_layer_mask", []) or []):
                moe_transformer_timings, moe_node_breakdown = self.compute_all_gemm_and_node_times(
                    batch_size,
                    vocab_size,
                    hidden_dim,
                    prefill_len,
                    num_heads,
                    kv_heads,
                    self.moe_intermediate_size,
                    num_SMs,
                    use_moe_override=True,
                )

            output_act_bytes = batch_size * prefill_len * hidden_dim * self.precision_bytes
            total_energy = self.calc_energy(transformer_timings, output_act_bytes)

            head_dim = getattr(self, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_dim // num_heads
            token_bytes = llm_util.kv_cache_token_bytes(
                batch_size=batch_size,
                kv_heads=self.kv_heads,
                head_dim=head_dim,
                precision_bytes=self.precision.kv_cache,
            )

            (
                pipeline_graph,
                pipeline_root,
                _,
                _,
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
                seq_len=prefill_len,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                include_pipeline_backward=False,
                include_transformer_backward=False,
            )

            self.pipeline_graph = pipeline_graph
            self.pipeline_root = pipeline_root
            self.pipeline_interconnect = interconnect_params
            self.transformer_graph = transformer_graph
            self.transformer_forward_root = transformer_forward_root
            self.transformer_backward_root = None
            self.transformer_graph_moe = moe_transformer_graph
            self.transformer_forward_root_moe = moe_transformer_forward_root
            self.transformer_backward_root_moe = moe_transformer_backward_root
            self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
            self.transformer_analytical_time_backward = None

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
            )
            mode = self.execution_mode
            try:
                result = dispatcher.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{exc}. Selected execution mode '{mode.value}'."
                ) from exc

            self.pipeline_graph = dispatcher.pipeline_graph
            self.pipeline_root = result.graph_root
            self.pipeline_interconnect = dispatcher.interconnect_params

            total_time = result.total_time

            prefill_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=prefill_len,
                kv_cache_tokens=prefill_len,
            )
            prefill_root = dispatcher.build_flattened_root_for_memory()
            _, prefill_peak_gb = mem_estimator.simulate_peak(
                prefill_root,
                prefill_memory_data,
                mode="inference",
                filename="memory_graph_prefill",
            )

        decode_peak_gb = prefill_peak_gb
        if decode_len > 0:
            decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=self.seq_len,
                d_model=hidden_dim,
                num_heads=num_heads,
                kv_heads=kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else intermediate_size,
                vocab_size=vocab_size,
                model_type=self.model_type,
            )
            (
                decode_pipeline_graph,
                decode_pipeline_root,
                _,
                _,
                decode_transformer_graph,
                decode_transformer_forward_root,
                decode_transformer_backward_root,
                decode_moe_transformer_graph,
                decode_moe_transformer_forward_root,
                decode_moe_transformer_backward_root,
                decode_interconnect_params,
            ), _ = self.prepare_decode_graphs(
                batch_size=batch_size,
                total_seq_len=self.seq_len,
                gemm_shapes=decode_gemm_shapes,
            )
            decode_dispatcher = LLMExecutionDispatcher(
                time_calc=self,
                pipeline_graph=decode_pipeline_graph,
                pipeline_root=decode_pipeline_root,
                interconnect_params=decode_interconnect_params,
                transformer_graph=decode_transformer_graph,
                transformer_forward_root=decode_transformer_forward_root,
                transformer_backward_root=decode_transformer_backward_root,
                moe_transformer_graph=decode_moe_transformer_graph,
                moe_transformer_forward_root=decode_moe_transformer_forward_root,
                moe_transformer_backward_root=decode_moe_transformer_backward_root,
            )
            decode_memory_root = decode_dispatcher.build_flattened_root_for_memory()
            decode_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=1,
                gemm_shapes=decode_gemm_shapes,
                kv_cache_tokens=self.seq_len,
            )
            original_pipeline_graph = self.pipeline_graph
            try:
                self.pipeline_graph = decode_pipeline_graph
                _, decode_peak_gb = mem_estimator.simulate_peak(
                    decode_memory_root,
                    decode_memory_data,
                    mode="inference",
                    filename="memory_graph_decode",
                )
            finally:
                self.pipeline_graph = original_pipeline_graph

        max_peak_gb = max(prefill_peak_gb, decode_peak_gb)
        self.memory_peak_gb = max_peak_gb

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None and hasattr(self.hw_config, "tech_config"):
            tech_cfg = self.hw_config.tech_config
            if hasattr(tech_cfg, "DRAM"):
                hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - max_peak_gb
            self.memory_headroom_gb = mem_delta
            self.memory_capacity_exceeded = mem_delta < 0
            self.memory_capacity_violation_gb = abs(mem_delta) if mem_delta < 0 else 0.0

            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                "Simulation mode: inference",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                f"Prefill peak memory usage (per gpu): {prefill_peak_gb:.2f} GiB",
                f"Final decode peak memory usage (per gpu): {decode_peak_gb:.2f} GiB",
                f"Max peak memory usage (per gpu): {max_peak_gb:.2f} GiB",
            ]
            if mem_delta < 0:
                info_lines.append(
                    f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.2f} GiB"
                )
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.2f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            self.memory_capacity_exceeded = False
            self.memory_capacity_violation_gb = 0.0

        return total_time, total_energy

    def calc_decode_time(self) -> Tuple[float, List[DecodeSample]]:
        """
        Calculate autoregressive decode phase execution time using sample-based approach.

        Returns:
            float: Total decode phase execution time
        """
        # Get inference sampling configuration
        sample_every = self.model.inference_sample_every
        if sample_every == -1:
            sample_every = 2**31 - 1

        decode_len = self.model.decode_len
        if decode_len == 0:
            print("Skipping decode")
            return 0.0, []

        # Create inference configuration from model parameters
        inference_config = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len - decode_len,
            decode_len=decode_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            use_moe=self.use_moe,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            pp=self.pp,
            tp=self.tp,
            tp_sp=self.tp_sp,
            moe_dp=self.moe_dp,
            sample_every=sample_every,
        )


        # Create inference engine with proper hardware and model configs
        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self._raw_model_config,
            time_calc_cls=TimeCalculationLLMInference,
        )

        # Build decode phase using sample-based approach with real RAPID-LLM integration
        # decode_time, decode_energy, decode_samples = inference_engine._build_decode_graph()
        return inference_engine._build_decode_graph()

    def calc_total_inference_time(self) -> dict:
        """
        Calculate complete inference time including prefill + decode phases.

        Returns:
            dict: Breakdown of inference timing components
        """
        # Calculate prefill time (existing functionality)
        prefill_time, prefill_energy = self.calc_time()

        # Calculate decode time (new functionality)
        decode_time, decode_energy, decode_samples = self.calc_decode_time()
        total_time = prefill_time + decode_time

        time_to_first_token = prefill_time
        if decode_samples:
            time_to_first_token += decode_samples[0].execution_time

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        token_bytes = llm_util.kv_cache_token_bytes(
            batch_size=self._effective_transformer_batch(),
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        prefill_len = self.seq_len - self.model.decode_len
        decode_len = self.model.decode_len
        num_layers = self.num_layers

        prefill_store_bytes = token_bytes * prefill_len * num_layers
        decode_store_bytes = token_bytes * decode_len * num_layers
        decode_fetch_bytes = token_bytes * num_layers * (
            decode_len * (prefill_len + self.seq_len) // 2
        )

        def _to_gib(byte_val: int) -> str:
            gib_val = byte_val / (1024 ** 3)
            if gib_val > 1024:
                tib_val = gib_val / 1024
                return f"{tib_val:.1f} TiB"
            return f"{gib_val:.1f} GiB"

        if decode_samples:
            # do NOT use effective_transformer_batch here
            decode_rates = self._decode_token_rates(decode_samples, decode_len, decode_time, self.batch_size)
        else:
            decode_rates = None

        print(
            f"[prefill] time: {prefill_time:.4f}s, "
            f"[decode] time: {decode_time:.4f}s, "
            f"[total] time: {total_time:.4f}s"
        )
        print(
            f"[kv-cache] prefill_store={_to_gib(prefill_store_bytes)}, "
            f"decode_store={_to_gib(decode_store_bytes)}, "
            f"decode_fetch={_to_gib(decode_fetch_bytes)}"
        )
        
        total_energy = prefill_energy + decode_energy
        print(
            f"[prefill] energy: {convert_prefix(prefill_energy)}J, energy/tok: {convert_prefix(prefill_energy / (self._effective_transformer_batch() * prefill_len))}J",
            f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy / (self._effective_transformer_batch() * decode_len))}J",
            f"[total] energy: {convert_prefix(total_energy)}J, energy/tok: {convert_prefix(total_energy / (self._effective_transformer_batch() * (prefill_len + decode_len)))}J",
        )


        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time,
            "time_to_first_token": time_to_first_token,
            "kv_cache_prefill_store_bytes": prefill_store_bytes,
            "kv_cache_decode_store_bytes": decode_store_bytes,
            "kv_cache_decode_fetch_bytes": decode_fetch_bytes,
            "decode_tokens_per_s": decode_rates,
        }

    @staticmethod
    def _decode_token_rates(
        samples: List[DecodeSample],
        decode_len: int,
        total_decode_time: float,
        batch_size: int,
    ) -> Dict[str, float]:
        if decode_len <= 0:
            return {}

        def token_time_at(step: int) -> float:
            if not samples:
                return 0.0
            if step <= samples[0].step_id:
                return samples[0].execution_time
            for idx in range(1, len(samples)):
                prev = samples[idx - 1]
                curr = samples[idx]
                if step <= curr.step_id:
                    gap = curr.step_id - prev.step_id
                    if gap <= 0:
                        return curr.execution_time
                    ratio = (step - prev.step_id) / gap
                    return prev.execution_time + ratio * (curr.execution_time - prev.execution_time)
            return samples[-1].execution_time

        def safe_rate(token_time: float) -> float:
            if token_time <= 0.0:
                return 0.0
            return 1.0 / token_time

        last_step = max(decode_len - 1, 0)
        mid_step = decode_len // 2

        start_rate = safe_rate(token_time_at(0))
        mid_rate = safe_rate(token_time_at(mid_step))
        end_rate = safe_rate(token_time_at(last_step))

        overall_rate = 0.0
        if total_decode_time > 0.0:
            overall_rate = decode_len / total_decode_time

        return {
            "start": start_rate,
            "midpoint": mid_rate,
            "end": end_rate,
            "midpoint_step": mid_step,
            "overall": overall_rate,
        }



__all__ = ["TimeCalculationLLMInference"]
