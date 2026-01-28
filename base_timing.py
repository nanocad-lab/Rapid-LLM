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
import sys
import config
import shutil
import itertools
from dataclasses import replace
from typing import Dict, NamedTuple, Optional

import util
from hw_component import Core, MemoryHierarchy, Network, DRAM
from astrasim_lib import run_cache_astrasim
from tile import AccessBytes, TiledGEMM, formatBytes
from timing_model import CollectiveType


class LinkInfo(NamedTuple):
    bandwidth: float
    latency: float


def _active_axis_sizes(hw_config, run_type: str) -> Dict[str, int]:
    sch = getattr(hw_config, "sch_config", None)
    if sch is None:
        return {"tp": 1, "cp": 1, "ep": 1, "pp": 1, "dp": 1}
    normalized = str(run_type or "training").lower()
    if normalized == "inference":
        dp = 1
        ep = int(sch.inference.moe_dp)
    else:
        dp = int(sch.train.dp)
        ep = int(sch.train.ep)
    return {
        "tp": int(sch.tp),
        "cp": int(sch.cp),
        "ep": ep,
        "pp": int(sch.pp),
        "dp": dp,
    }


def _refresh_network_layout(hw_config, axis_sizes: Dict[str, int]):
    layout = getattr(hw_config, "network_layout", None)
    if layout is None or not getattr(layout, "dimensions", None):
        return layout
    dimensions = []
    for dim in layout.dimensions:
        axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ()) or ()]
        if not axes:
            dimensions.append(dim)
            continue
        expected = 1
        for axis in axes:
            expected *= max(1, int(axis_sizes.get(axis, 1)))
        size_value = int(getattr(dim, "size", expected) or expected)
        if expected != size_value:
            dimensions.append(replace(dim, size=expected))
        else:
            dimensions.append(dim)
    return config._build_network_layout_config(
        dimensions,
        getattr(layout, "faulty_links", ()) or (),
        getattr(layout, "overlap_config", None),
    )


class Parallelism:
    def __init__(self, exp_config, run_type: str = "training"):
        sch = exp_config.sch_config
        self.pp = sch.pp
        self.mb = sch.mb
        self.tp = sch.tp
        self.cp = sch.cp
        self.tp_sp = sch.tp_sp
        self.run_type = str(run_type or "training").lower()
        if self.run_type == "inference":
            self.dp = 1
            self.ep = int(sch.inference.moe_dp)
            self.tp_ep = False
            self.replica_count = int(sch.inference.replica_count)
            self.moe_dp = int(sch.inference.moe_dp)
        else:
            self.dp = int(sch.train.dp)
            self.ep = int(sch.train.ep)
            self.tp_ep = bool(sch.train.tp_ep)
            self.replica_count = 1
            self.moe_dp = 1


class model_GEMM:
    def __init__(self, exp_config):
        self.M = exp_config.model_config.M
        self.K = exp_config.model_config.K
        self.N = exp_config.model_config.N
        self.backward = exp_config.model_config.backward
        self.gemm_shard_axis = str(exp_config.model_config.gemm_shard_axis).strip().lower()


class model_LLM:
    def __init__(self, exp_config):
        self.global_batch_size = exp_config.model_config.global_batch_size
        self.gradient_accumulation_steps = exp_config.model_config.gradient_accumulation_steps
        self.vocab_size = exp_config.model_config.vocab_size
        self.num_layers = exp_config.model_config.num_layers
        self.hidden_dim = exp_config.model_config.hidden_dim
        self.seq_len = exp_config.model_config.seq_len
        self.decode_len = exp_config.model_config.decode_len
        self.num_heads = exp_config.model_config.num_heads
        self.head_dim = getattr(exp_config.model_config, "head_dim", None)
        self.tied_embeddings = exp_config.model_config.tied_embeddings
        self.model_type = exp_config.model_config.model_type
        self.intermediate_size = exp_config.model_config.intermediate_size
        self.n_tokens = exp_config.model_config.n_tokens
        self.run_type = exp_config.model_config.run_type
        self.attention_type = exp_config.model_config.attention.attention_type
        self.kv_heads = (
            exp_config.model_config.attention.kv_heads
            if hasattr(exp_config.model_config.attention, "kv_heads")
            else None
        )
        self.use_flashattention = getattr(exp_config.model_config.attention, "use_flashattention", False)
        self.attention_tile_size = getattr(exp_config.model_config.attention, "attention_tile_size", None)

        self.moe_num_experts = int(getattr(exp_config.model_config, "num_experts", 1))
        self.moe_top_k = int(getattr(exp_config.model_config, "top_k", 1))
        self.moe_intermediate_size = int(
            getattr(exp_config.model_config, "moe_intermediate_size", self.intermediate_size)
        )
        self.n_shared_experts = int(getattr(exp_config.model_config, "n_shared_experts", 0))
        self.moe_layer_freq = int(getattr(exp_config.model_config, "moe_layer_freq", 1))
        self.first_k_dense_replace = int(getattr(exp_config.model_config, "first_k_dense_replace", 0))
        self.moe_layer_mask = list(getattr(exp_config.model_config, "moe_layer_mask", []))
        self.use_moe = bool(getattr(exp_config.model_config, "use_moe", False))

        inference_cfg = getattr(exp_config, "inference_config", None)
        if str(self.run_type).lower() == "inference":
            if inference_cfg is None:
                raise ValueError("Inference configuration not found for inference run_type")
            self.inference_sample_every = inference_cfg.sample_every
        else:
            self.inference_sample_every = -1

class NetworkModel:
    def __init__(self, hw_config, precision, kernel_overhead, roofline_cb, astra_policy: Optional[str] = None):
        self.hw_config = hw_config
        self.precision = precision
        self.O = kernel_overhead
        self._roofline = roofline_cb
        self._astra_policy = astra_policy or "analytical"

    def _astra_collective(self, kind: str, participants: int, size_bytes: int, axis: Optional[str] = None) -> float:
        part = int(participants)
        if part <= 1 or size_bytes <= 0:
            return 0.0
        byte_count = int(math.ceil(size_bytes))
        axes_filter = [str(axis).lower()] if axis else None
        # for collectives ONLY, we cannot use 2D topologies (Mesh2D, Torus2D, KingMesh2D)
        # transform them to their 1D equivalents (Ring, Mesh, HyperCube (?))
        _, max_sec = run_cache_astrasim(
            self.hw_config,
            comm=kind,
            npus_count=part,
            size_bytes=byte_count,
            axes_filter=axes_filter,
            transform_2d_to_1d=True,
        )
        return float(max_sec)


    def collective(
        self,
        *,
        kind: CollectiveType,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        local_bytes: float = 0.0,
        local_ops: float = 0.0,
        debug_label: str = "",
        axis: Optional[str] = None,
    ) -> float:
        if participants is None:
            return 0.0

        part = int(participants)
        if part <= 1:
            return 0.0

        if size_bytes <= 0:
            return 0.0

        # Pipeline and point-to-point operations now handled through unified path

        if not isinstance(kind, CollectiveType):
            raise TypeError(f"collective kind must be CollectiveType (got {type(kind).__name__})")
        kind_value = kind.value
        collective_ops = {
            CollectiveType.ALL_REDUCE,
            CollectiveType.REDUCE_SCATTER,
            CollectiveType.ALL_GATHER,
            CollectiveType.ALL_TO_ALL,
        }

        network_bytes = float(math.ceil(size_bytes))
        local_bytes_int = int(math.ceil(local_bytes)) if local_bytes else 0

        if self._astra_policy in {"hybrid", "full"}:
            if kind in collective_ops or kind == CollectiveType.PIPELINE:
                # Pipeline uses 2 NPUs for point-to-point, others use part
                npus = 2 if kind == CollectiveType.PIPELINE else part
                axis_filter = axis if kind != CollectiveType.PIPELINE else None
                network_time = self._astra_collective(kind_value, npus, network_bytes, axis_filter)
            else:
                raise ValueError(f"Unsupported collective operation: {kind_value}")
        else:
            network_time = self._analytical_collective(
                kind=kind,
                size_bytes=network_bytes,
                participants=part,
                ib=ib,
                ll=ll,
                debug_label=debug_label,
            )

        # TODO TODO TODO
        # We want to ignore this in the future. Skipping it directly so that we match astrasim for now.
        # FIX THIS (FIGURE OUT IF WE WANT TO SKIP OR NOT)
        local_time = 0.0

        overhead_kinds = {
            CollectiveType.ALL_REDUCE,
            CollectiveType.REDUCE_SCATTER,
            CollectiveType.ALL_GATHER,
        }
        overhead = self.O if (network_bytes and kind in overhead_kinds) else 0.0

        return network_time + local_time + overhead


    def _analytical_collective(
        self,
        *,
        kind: CollectiveType,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        debug_label: str,
    ) -> float:
        if kind == CollectiveType.ALL_REDUCE:
            return self._analytical_all_reduce(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.REDUCE_SCATTER:
            return self._analytical_reduce_scatter(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.ALL_GATHER:
            return self._analytical_all_gather(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.ALL_TO_ALL:
            return self._analytical_all_to_all(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.PIPELINE:
            return self._analytical_point_to_point(size_bytes, ib, ll)
        # Default fallback for unknown patterns
        raise ValueError(f"Unsupported collective operation: {kind.value}")

    def _analytical_all_reduce(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * 2 * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))

        # NOTE: removed overhead because it made it impossible to eliminate network effects even with inf bandwidth and zero latency
        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep") # + self.O
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_reduce_scatter(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))
        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep") + self.O
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_all_gather(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",
        )
        data_transfer = ((size_bytes / ib) + mem_access + ll) * (participants - 1)
        return data_transfer

    def _analytical_all_to_all(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        return ((size_bytes / participants) / ib + ll) * (participants - 1)

    def _analytical_point_to_point(self, size_bytes, ib, ll):
        if size_bytes <= 0:
            return 0.0
        if ib == 0:
            return float("inf")
        return size_bytes / ib + ll


class TimeCalculation:
    def __init__(self, hw_config, model_config, mode, *, astra_policy_override: Optional[str] = None):
# Mode parameter
        

        # Software Parameters
        self.O = hw_config.sw_config.kernel_launch_overhead
        self.precision = hw_config.sw_config.precision
        self.precision_bytes = self.precision.activations
        self.h2d_bandwidth = getattr(hw_config.sw_config, "h2d_bandwidth", -1)
        self.zero_stage = getattr(hw_config.sw_config, "dp_zero_stage", 0)
        self.full_recomputation = getattr(hw_config.sw_config, "full_recomputation", False)
        self.dp_microbatch = getattr(hw_config.sw_config, "dp_microbatch", "every_mb")
        self.grad_acc_overhead = float(getattr(hw_config.sw_config, "grad_acc_overhead", 0.0) or 0.0)
        self.attached = True

        run_type = "training"
        if str(mode).upper() == "LLM":
            model_cfg = getattr(model_config, "model_config", None)
            run_type = str(getattr(model_cfg, "run_type", "training")).lower()
        self.run_type = run_type

        axis_sizes = _active_axis_sizes(hw_config, run_type)
        setattr(hw_config, "active_parallelism", dict(axis_sizes))
        refreshed_layout = _refresh_network_layout(hw_config, axis_sizes)
        if refreshed_layout is not None:
            hw_config.network_layout = refreshed_layout

        # Hardware Parameters
        self.hw_config = hw_config
        self.core = Core(hw_config)
        self.th = self.core.get_throughput()
        self.FMA_dims = self.core.FMA_dims  # (FMA_x, FMA_y)
        self.dataflow = self.core.dataflow

        self.memory_hierarchy = MemoryHierarchy(hw_config, core=self.core)
        self.num_levels = self.memory_hierarchy.num_levels
        self.mem_layer = self.memory_hierarchy.mem_layer
        self.tile_space = None
        self.H2Dbw = self.h2d_bandwidth

        level = 0
        mem_config = hw_config.memory_hierarchy.mem_hr[level]
        self.DRAM = DRAM(hw_config, mem_config, level, self.core)
        self.network = Network(hw_config)

        self.links = {
            "dp": LinkInfo(*self.network.get_link("dp")),
            "ep": LinkInfo(*self.network.get_link("ep")),
            "pp": LinkInfo(*self.network.get_link("pp")),
            "tp": LinkInfo(*self.network.get_link("tp")),
            "cp": LinkInfo(*self.network.get_link("cp")),
        }
        
        # Scheduling Parameters
        par = Parallelism(hw_config, run_type=run_type)
        self.mode = mode
        self.pp = par.pp
        self.mb = par.mb
        self.dp = par.dp
        self.ep = par.ep

        self.tp = par.tp
        self.cp = par.cp

        self.tp_sp = par.tp_sp
        self.tp_ep = par.tp_ep
        self.replica_count = par.replica_count
        self.moe_dp = par.moe_dp
        self.num_workers = self._derive_num_workers(hw_config)

        
        # Statistics Param
        self.tot_flop = 0
        self.tot_mem = 0
        self.tot_time = 0
        self.debug = False
        
        default_policy = 'analytical'
        eb = getattr(hw_config, "execution_backend", None)
        if eb and getattr(eb, "model", "analytical") == "astra":
            default_policy = 'hybrid'
        self._astra_policy = astra_policy_override or default_policy

        self.network_model = NetworkModel(
            hw_config,
            self.precision_bytes,
            self.O,
            self.roofline,
            astra_policy=self._astra_policy,
        )

        model_class = self.get_model_class(mode)
        self.model = model_class(model_config)

        if mode == "GEMM":
            self.M = self.model.M
            self.K = self.model.K
            self.N = self.model.N
            self.gemm_shard_axis = self.model.gemm_shard_axis

        if mode == "LLM":
            self.global_batch_size = self.model.global_batch_size
            self.gradient_accumulation_steps = self.model.gradient_accumulation_steps
            self.batch_size = self.global_batch_size // self.gradient_accumulation_steps
            self.vocab_size = self.model.vocab_size
            self.num_layers = self.model.num_layers
            self.hidden_dim = self.model.hidden_dim
            self.seq_len = self.model.seq_len
            self.num_heads = self.model.num_heads
            self.head_dim = (
                int(self.model.head_dim)
                if getattr(self.model, "head_dim", None) is not None
                else self.hidden_dim // self.num_heads
            )
            self.intermediate_size = self.model.intermediate_size
            self.n_tokens = self.model.n_tokens
            if self.run_type == "inference":
                dp_dense = 1
            else:
                dp_dense = self.dp * self.ep if bool(getattr(self.model, "use_moe", False)) else self.dp
            self.mini_batch = math.ceil(self.batch_size / dp_dense)  # mini-batch size for each data parallel node
            self.micro_batch = math.ceil(self.mini_batch / self.mb) if self.pp > 1 else self.mini_batch
            self.attention_type = self.model.attention_type
            self.flash_attention = getattr(self.model, "use_flashattention", False)
            self.kv_heads = self.model.kv_heads if hasattr(self.model, "kv_heads") else self.num_heads
            self.attention_tile_size = getattr(self.model, "attention_tile_size", None)
            raw_num_experts = getattr(self.model, "moe_num_experts", 1)
            raw_top_k = getattr(self.model, "moe_top_k", 1)
            self.moe_num_experts = max(1, int(raw_num_experts))
            self.moe_top_k = max(1, int(raw_top_k))
            self.moe_intermediate_size = int(
                getattr(self.model, "moe_intermediate_size", self.intermediate_size)
            )
            self.n_shared_experts = int(getattr(self.model, "n_shared_experts", 0))
            self.moe_layer_freq = int(getattr(self.model, "moe_layer_freq", 1))
            self.first_k_dense_replace = int(getattr(self.model, "first_k_dense_replace", 0))
            self.moe_layer_mask = list(getattr(self.model, "moe_layer_mask", []))
            self.use_moe = bool(getattr(self.model, "use_moe", False))
            self.num_moe_layers = sum(self.moe_layer_mask)


    def _derive_num_workers(self, hw_config) -> int:
        layout = getattr(hw_config, "network_layout", None)
        if layout and getattr(layout, "dimensions", None):
            total = 1
            for dim in layout.dimensions:
                try:
                    size = int(dim.size)
                except (TypeError, ValueError):
                    size = 1
                if size < 1:
                    size = 1
                total *= size
            if total >= 1:
                return int(total)

        active = getattr(hw_config, "active_parallelism", None)
        if isinstance(active, dict) and active:
            total = 1
            for axis in ("dp", "pp", "tp", "cp", "ep"):
                total *= max(1, int(active.get(axis, 1) or 1))
            return max(1, total)

        sch = getattr(hw_config, "sch_config", None)
        if sch is None:
            return 1
        total = (
            max(1, int(sch.train.dp))
            * max(1, int(sch.train.ep))
            * max(1, int(sch.pp))
            * max(1, int(sch.tp))
            * max(1, int(sch.cp))
        )
        return max(1, total)
   

    def get_model_class(self, model_type):
        """Return the appropriate model class based on the model type."""
        model_classes = {
            "GEMM": model_GEMM,
            "LLM": model_LLM,
        }
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_classes[model_type]

    def roofline(self, flop, mem_access_, name="", util=1, info=False, mem_level=None, flashattn_enable=False):
        # print("Roofline: entered {}".format(name))

        # Parse mem_access_ into consistent format
        if isinstance(mem_access_, int):
            mem_access = [mem_access_]
        elif isinstance(mem_access_, float):
            mem_access = [int(mem_access_)]
        elif isinstance(mem_access_, list):
            mem_access = mem_access_
        else:
            print(mem_access_)
            print("mem_access_ should be integer or list, wrong input", flush=True)
            sys.exit(0)

        throughput = self.th * util

        # Determine which levels to compute
        if mem_level is not None:
            # Single level mode
            levels_to_compute = [(mem_level, mem_access[0])]
        else:
            # Multi-level mode (original behavior)
            num_level = len(mem_access)
            try:
                if not flashattn_enable:
                    assert mem_access[num_level - 1] > 0, f"mem_access: {mem_access_}"
            except Exception as e:
                print(
                    "{}: Number of accesses to the last level of memory hierarchy cannot be zero:\n {}".format(
                        name, e
                    ),
                    flush=True,
                )
                sys.exit(0)
            levels_to_compute = [(i, mem_access[i]) for i in range(num_level)]

        # Compute roofline time for each level
        times = []
        for level_idx, num_mem in levels_to_compute:
            mem_bw = self.mem_layer[level_idx].get_throughput()
            mem_latency = self.mem_layer[level_idx].get_latency()
            inflection_point = float("inf") if mem_bw == 0 else throughput / mem_bw
            # print(f"Level {level_idx}: mem_bw={mem_bw}, mem_latency={mem_latency}, inflection_point={inflection_point}", flush=True)
            comp_int = float("inf") if num_mem == 0 else flop / num_mem
            if comp_int < inflection_point:  # mem-bound
                level_time = (float("inf") if (mem_bw == 0 or num_mem == 0) else (num_mem / mem_bw)) + mem_latency
            else:  # compute-bound
                level_time = float("inf") if (throughput == 0) else (flop / throughput)

            times.append(level_time)

        max_time = max(times)

        # print("Roofline: exited {}".format(name))
        return max_time
    
    def get_gemm_time(self, dim1, dim2, dim3, name="", 
                    flashattn_enable=False, disable_overhead=False, read_bytes_l2=0, write_bytes_l2=0, original=False):
        # Streaming best selection to avoid building large dicts
        best_time = float("inf")
        best_choice = None  # type: Optional[tuple]
        best_mem_access = None  # type: Optional[tuple]
        best_rw_access = None  # type: Optional[AccessBytes]
        best_gemm = None  # type: Optional[TiledGEMM]
        best_metric = float("inf")

        # Iterate directly over candidates from tile module; no string orders
        for gemm in TiledGEMM.enumerate_candidates(self.core, self.mem_layer, dim1, dim2, dim3, self.precision_bytes, original=False):
            if self.debug:
                print("===============================================================")
                print(f"inner_code: {gemm._inner_code}")
                print("===============================================================")

            GEMM_flop = gemm.GEMM_flop
            rw_accesses = gemm.mem_accesses

            mem_access = rw_accesses.totals() # (L0, L1, L2, DRAM)
            if flashattn_enable:
                mem_access = list(mem_access)
                mem_access[3] = 0 # no HBM accesses
                mem_access[2] = read_bytes_l2 + write_bytes_l2 # explicitly set L2 accesses
                mem_access = tuple(mem_access)
                

            # Adjust shared accesses per effective SMs
            mem_access_per_sm = list(mem_access)
            reuse_M = (dim1 + gemm.l2_M - 1) // gemm.l2_M
            reuse_K = (dim2 + gemm.l2_K - 1) // gemm.l2_K
            reuse_N = (dim3 + gemm.l2_N - 1) // gemm.l2_N
            eff_sm = min(self.core.num_bundle, reuse_M * reuse_K * reuse_N)
            if eff_sm > 0:
                mem_access_per_sm[1] = mem_access_per_sm[1] / eff_sm

            if flashattn_enable:
                sm_util = 0.25 # assume 25% SM utilization for FlashAttention (Dao, 2023)
            else:
                sm_util = 1
                gemm_waves = self.mem_layer[1].calc_waves_per_sm(dim1, dim2, dim3, gemm.l2_M, gemm.l2_K, gemm.l2_N)
                if gemm_waves != -1:
                    full_waves = math.floor(gemm_waves)
                    partial_wave = gemm_waves - full_waves
                    sm_util = (full_waves + partial_wave * partial_wave) / gemm_waves

            GEMM_time = self.roofline(GEMM_flop, mem_access_per_sm, name, util=sm_util, flashattn_enable=flashattn_enable) 
            if flashattn_enable or disable_overhead:
                pass
            else:
                GEMM_time = GEMM_time + self.O
     
            tile_dims = (
                (gemm.l0_M, gemm.l0_K, gemm.l0_N),
                (gemm.l1_M, gemm.l1_K, gemm.l1_N),
                (gemm.l2_M, gemm.l2_K, gemm.l2_N),
            )
            key = (gemm._inner_code, tile_dims)

            # Tie-breaker metric identical to previous selection: hypot(dram, l2)
            
            metric = math.hypot(mem_access[3], mem_access[2])

            if (GEMM_time < best_time) or (GEMM_time == best_time and metric < best_metric):
                best_time = GEMM_time
                best_choice = key
                best_mem_access = mem_access
                best_rw_access = rw_accesses
                best_gemm = gemm
                best_metric = metric

        # best_choice, best_mem_access must be set if there was at least one candidate
        mem_access = best_mem_access  # type: ignore

        if self.debug:
            print(repr(best_gemm))
            print(
                f"{name}: Best Time: {best_time * 1e3:,} ms, Best Inner: {best_choice[0]}, Best Tile: {best_choice[1]}\n"
            )

        # Inner code mapping for loop order (no strings):
        # 0 -> inner 'm' (weight stationary)
        # 1 -> inner 'k' (output stationary)
        # 2 -> inner 'n' (activation stationary)
        best_inner_code = best_choice[0]  # type: ignore[index]
        best_tile_dims = best_choice[1]  # type: ignore[index]
        return best_time, best_inner_code, best_tile_dims, mem_access #, best_rw_access
    
    def get_tile_size(self, lid):
        memory = self.mem_layer[lid]
        memory.calc_tile_dim()
        tile_dim = memory.get_tile_dim()
        return tile_dim, tile_dim, tile_dim

    # Count the number of accesses from level-1 to level
    # input matrix A(dim1, dim2) and B(dim2, dim3)
    # output matrix C(dim1, dim3)
    def get_num_accesses(self, level, dim1, dim2, dim3, tile_dim, num_repeat, name, r):
        # tile1,tile2,tile3 = self.get_tile_size(level-1)
        # print("dim1= ", dim1, "dim2= ", dim2, "dim3 = ", dim3)

        tile1, tile2, tile3 = tile_dim
        # print("BEFORE: level = ", level, "|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        orig_size = tile1 * tile2 + tile1 * tile3 + tile2 * tile3
        short_tile_cond = [0, 0, 0]

        if tile1 > dim1:
            tile1 = dim1
            short_tile_cond[0] = 1
        if tile2 > dim2:
            tile2 = dim2
            short_tile_cond[1] = 1
        if tile3 > dim3:
            tile3 = dim3
            short_tile_cond[2] = 1

        # print("AFTER: level= ", level ,"|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        if short_tile_cond[2] == 0 and (short_tile_cond[0] | short_tile_cond[1]) == 1:
            if level <= 1:
                tile3 = math.floor((orig_size - tile1 * tile2) / (tile1 + tile2))
            else:
                # store bypasses cache, directly goes to memory
                tile3 = math.floor((orig_size - tile1 * tile2) / tile2)
            if tile3 > dim3:
                tile3 = dim3
            # Uncomment if tile3 needs to be pow of 2
            # tile3 = int(math.pow(2, math.floor(math.log2(tile3))))
        elif short_tile_cond[0] == 0 and (short_tile_cond[1] | short_tile_cond[2]) == 1:
            if level <= 1:
                tile1 = math.floor((orig_size - tile3 * tile2) / (tile3 + tile2))
            else:
                # store bypasses cache, directly goes to memory
                tile1 = math.floor((orig_size - tile3 * tile2) / tile2)
            if tile1 > dim1:
                tile1 = dim1
        elif short_tile_cond[1] == 0 and (short_tile_cond[0] & short_tile_cond[2]) == 1:
            if level <= 1:
                tile2 = math.floor((orig_size - tile3 * tile1) / (tile3 + tile1))
            else:
                tile2 = math.floor((orig_size) / (tile1 + tile3))
            if tile2 > dim2:
                tile2 = dim2

        # print("FINAL: level= ", level ,"|tile1= ", tile1, "tile2= ", tile2, "tile3 = ", tile3) ###############

        reload_A = 1
        reload_B = 1
        reload_C = 1

        if tile1 > 0 and tile2 > 0 and tile3 > 0:
            reload_A = math.ceil(dim3 / tile3)
            reload_B = math.ceil(dim1 / tile1)
            # do not access the slow memory on every write,acculmuate in fast memory
            reload_C = 1 if level > 1 else math.ceil(dim2 / tile2)

        num_repeat = r[0] * r[1] * r[2]

        if level == 2:  # access to L1 scratchpad
            num_mem = (
                num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B)
                + r[0] * dim1 * r[2] * dim3 * reload_C
            ) * self.precision_bytes
        else:
            num_mem = (
                num_repeat
                * (
                    dim1 * dim2 * reload_A
                    + dim2 * dim3 * reload_B
                    + dim1 * dim3 * reload_C
                )
                * self.precision_bytes
            )

        # num_mem = num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B + dim1 * dim3 * reload_C) * self.precision_bytes

        if self.debug:
            print(name)
            print(
                "Matrix dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level, dim1, dim2, dim3
                )
            )
            print(
                "Tile dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level - 1, tile1, tile2, tile3
                )
            )
            print(
                "reload_A: {}, reload_B: {}, reload_C: {}".format(
                    reload_A, reload_B, reload_C
                )
            )
            print("num_repeat: {}".format(num_repeat))
            print("Bytes Accessed: {:,}".format(num_mem))
            print("")

        return num_mem, tile1, tile2, tile3

    # This is the main function that captures the memory hierarchy impact
    # on the number of accesses to global memory considering not everything fits in
    # L2 cache and also captures the effect of shared memory
    def GEMM(self, order_dims, tile_dims, name):
        dim1_ = order_dims[0]
        dim2_ = order_dims[1]
        dim3_ = order_dims[2]
        # dim1 = util.power2RoundUp(dim1_)
        # dim2 = util.power2RoundUp(dim2_)
        # dim3 = util.power2RoundUp(dim3_)
        dim1 = dim1_
        dim2 = dim2_
        dim3 = dim3_

        GEMM_flop = dim1 * dim3 * (dim2 + dim2 - 1)
        # dim2 multiply
        # dim2-1 add

        # X1 = self.L2_tile_dim
        # X2 = self.shared_mem_tile_dim
        # X3 = self.reg_tile_dim

        num_accesses = [0] * self.num_levels
        r1, r2, r3 = 1, 1, 1

        num_repeat = 1
        for level in range(self.num_levels - 1, 0, -1):
            repeat = (r1, r2, r3)
            num_accesses[level], tile1, tile2, tile3 = self.get_num_accesses(
                level,
                dim1,
                dim2,
                dim3,
                tile_dims[level - 1],
                num_repeat,
                name,
                repeat,
            )
            try:
                num_repeat *= (
                    math.ceil(dim1 / tile1)
                    * math.ceil(dim2 / tile2)
                    * math.ceil(dim3 / tile3)
                )
                r1 *= math.ceil(dim1 / tile1)
                r2 *= math.ceil(dim2 / tile2)
                r3 *= math.ceil(dim3 / tile3)
            except:
                num_repeat *= 1

            dim1 = tile1 if tile1 != 0 else dim1
            dim2 = tile2 if tile2 != 0 else dim2
            dim3 = tile3 if tile3 != 0 else dim3

            # assume systolic engine can support n x m x n GEMM  (e.g. 8 x 4 x 8 for A100 tensorcore), which is FLOPs_tile = n^2 * (m-1) FLOPs
            # reuse is the number of n x m x n GEMMs that are performed before stationary values (weight, activations, or output) get swapped
            # every n x n output tile:
            #   1. loads nxm activations and mxn weights -> 2 * reuse * n * m accesses
            #   2. performs reuse * FLOPs_tile computations
            #   3. writes back n^2 output elements

            reuse = 1
            dim1 = dim1_
            dim2 = dim2_
            dim3 = dim3_

            if self.dataflow == "none":
                reuse = 1
            elif self.dataflow == "best":
                reuse = max(
                    math.ceil(dim1 / self.FMA_dims[0]),
                    math.ceil(dim3 / self.FMA_dims[0]),
                    math.ceil(dim2 / self.FMA_dims[1]),
                )
            elif self.dataflow == "wst":  # weight stationary
                reuse = math.ceil(dim1 / self.FMA_dims[0])
            elif self.dataflow == "ast":  # activation stationary
                reuse = math.ceil(dim3 / self.FMA_dims[0])
            elif self.dataflow == "ost":  # output stationary
                reuse = math.ceil(dim2 / self.FMA_dims[1])
            else:
                raise NotImplementedError()

            # TODO: make sure to model underutilized systolic array
            num_accesses[0] = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/(2 * self.FMA_dims[0]) * self.precision_bytes

            # num_accesses[0] = (
            #     GEMM_flop
            #     * (
            #         2 * reuse * self.FMA_dims[0] * self.FMA_dims[1]
            #         + self.FMA_dims[0] ** 2
            #     )
            #     / (2 * reuse * self.FMA_dims[0] * (self.FMA_dims[1] - 1))
            #     * self.precision_bytes
            # )
            # num_accesses[0]    = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/self.FMA_width * self.precision_bytes
            # num_accesses[0]    = GEMM_flop * ((2 * reuse + self.FMA_width) / (2 * reuse)) * 1/self.FMA_width * self.precision_bytes

            # TODO: do we still need these in new hierarchical version?
            #  if X3 == 0:
            #    GEMM_smem  = GEMM_rmem
            #    GEMM_rmem  = 0
            #  if X2 == 0:
            #    GEMM_l2mem = GEMM_smem
            #    GEMM_smem = 0
            #  if X1 == 0:
            #    GEMM_gmem  = GEMM_l2mem
            #    GEMM_l2mem = 0

            #  try:
            #    GEMM_l2mem = GEMM_smem
            #    GEMM_smem = 0
            #  if X1 == 0:
            #    GEMM_gmem  = GEMM_l2mem
            #    GEMM_l2mem = 0

        return GEMM_flop, num_accesses

    def get_cf(self, m, k, n):
        """Compute forward GEMM time for a dense matmul with pointwise epilogue."""
        GEMM_time = self.get_gemm_time(m, k, n, "Cf")
        point_time = self._gemm_pointwise_time(m, n, "Cf")
        return GEMM_time[0] + point_time

    def _gemm_pointwise_time(self, m, n, name):
        point_flop = m * n * 5
        # 1: add bias
        # 5: add nonlinearities, there is one more than the number of gates (self.G)
        # 1: pointwise muliply and add
        point_mem = self.precision_bytes * m * n * (3 * 3 + 2 * 2)
        # 3: 3 memory accesses for operands with two inputs and one output
        # 2: 1 for bias add + 1 for pointwise mul
        # 2: 2 memory accesses for operands with one input and one output
        # 1: 5/4 non-linearities per gate

        point_time = (
            self.roofline(point_flop, point_mem, name=f"pointwise_{name}") + 5 * self.O
        )

        if self.debug:
            gigaByte = 1024 * 1024 * 1024
            print(
                "Hidden point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / gigaByte)
                )
            )
            print("Hidden point_time: {:,}\n".format(point_time))

        return point_time

    # Reduction and all-gather time estimation

    def grad_clipping(self, num_params: int) -> float:
        """Gradient clipping (L2 norm + scale) cost."""
        norm_comp = num_params * 2  # square + sum
        clip_comp = num_params * 2  # scale + divide
        clip_mem = num_params * 2 * self.precision.gradients  # read + write

        gradclip_mem = clip_mem
        gradclip_comp = norm_comp + clip_comp

        return self.roofline(gradclip_comp, gradclip_mem, name="pointwise-grad-clipping")

    def apply_grad(self, num_params: int) -> float:
        """Approximate optimizer update cost (Adam/AdamW-style) per parameter tensor."""
        OPT_FLOPS_PER_PARAM = 14.0 # typical for Adam
        bytes_per_param_read = (
            self.precision.parameters
            + self.precision.gradients
            + 2 * self.precision.optimizer_states
            + self.precision.master_parameters
        )
        bytes_per_param_write = (
            self.precision.parameters
            + 2 * self.precision.optimizer_states
            + self.precision.master_parameters
        )

        apply_grad_comp = num_params * OPT_FLOPS_PER_PARAM
        apply_grad_mem = num_params * (bytes_per_param_read + bytes_per_param_write)

        apply_grad_time = self.roofline(apply_grad_comp, apply_grad_mem, name="apply_grad", mem_level=self.num_levels-1)
        clip_time = self.grad_clipping(num_params)
        return apply_grad_time + clip_time



    def get_dist_gemm_forward(self, m, k, n, name, batch=1):
        if self.tp <= 1:
            gemm_time = self.get_gemm_time(m, k, n, name)[0] * batch
            point_time = self._gemm_pointwise_time(m, n, name)
            return gemm_time + point_time, 0.0
        if self.gemm_shard_axis == "row":
            gemm_time = self.get_gemm_time(m, k // self.tp, n, name)[0]
            gemm_time *= batch
            point_time = self._gemm_pointwise_time(m, n, name)
            total_bytes = math.ceil(self.precision_bytes * m * n) * batch
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_REDUCE,
                size_bytes=total_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time + point_time, reduction_time
        if self.gemm_shard_axis == "col":
            gemm_time = self.get_gemm_time(m, k, n // self.tp, name)[0]
            gemm_time *= batch
            point_time = self._gemm_pointwise_time(m, n, name)
            total_bytes = math.ceil(self.precision_bytes * m * n) * batch
            shard_bytes = math.ceil(total_bytes / self.tp)
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_GATHER,
                size_bytes=shard_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time + point_time, reduction_time
        raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")

    def get_dist_gemm_backward(self, m, k, n, name, batch=1):
        if self.tp <= 1:
            grad_act_time, _, _, _ = self.get_gemm_time(m, n, k, name + "_act")
            grad_wt_time, _, _, _ = self.get_gemm_time(k, m, n, name + "_wt")
            return (grad_act_time + grad_wt_time) * batch, 0.0
        if self.gemm_shard_axis == "row":
            grad_wt_time, _, _, _ = self.get_gemm_time(k // self.tp, m, n, name + "_wt")
            grad_act_time, _, _, _ = self.get_gemm_time(m, n, k // self.tp, name + "_act")
            gemm_time = (grad_wt_time + grad_act_time) * batch
            total_bytes = math.ceil(self.precision_bytes * m * k) * batch
            shard_bytes = math.ceil(total_bytes / self.tp)
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_GATHER,
                size_bytes=shard_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time, reduction_time
        if self.gemm_shard_axis == "col":
            grad_wt_time, _, _, _ = self.get_gemm_time(k, m, n // self.tp, name + "_wt")
            grad_act_time, _, _, _ = self.get_gemm_time(m, n // self.tp, k, name + "_act")
            gemm_time = (grad_wt_time + grad_act_time) * batch
            total_bytes = math.ceil(self.precision_bytes * m * k) * batch
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_REDUCE,
                size_bytes=total_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time, reduction_time
        raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")

    def get_data_parallel_reduction(self, k, n, name):
        if not self.dp or self.dp <= 1:
            return 0.0
        if self.gemm_shard_axis == "row":
            param_count = (k // self.tp) * n if self.tp > 1 else k * n
        elif self.gemm_shard_axis == "col":
            param_count = k * (n // self.tp) if self.tp > 1 else k * n
        else:
            raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")
        total_bytes = math.ceil(self.precision_bytes * param_count)
        reduction_time = self.network_model.collective(
            kind=CollectiveType.ALL_REDUCE,
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.links["dp"].bandwidth,
            ll=self.links["dp"].latency,
            debug_label=name or "comm",
        )
        apply_grad_time = self.apply_grad(int(math.ceil(param_count)))
        if self.debug:
            print(f"reduction_time_wt_dp: {reduction_time}")
            print(f"apply_grad_time: {apply_grad_time}")
        return reduction_time + apply_grad_time
