#!/usr/bin/env python3
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


import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config
import memory_estimation

HARDWARE_CONFIG = PROJECT_ROOT / "configs" / "hardware-config" / "a100_80GB.yaml"
TRAINING_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model-config" / "Llama2-7B.yaml"
INFERENCE_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model-config" / "Llama2-7B_inf.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output" / "memory_estimator_smoke"
MODE = "LLM"

SEQ_LEN: Optional[int] = None
DECODE_LEN: Optional[int] = None
BATCH_SIZE: Optional[int] = None
GLOBAL_BATCH_SIZE: Optional[int] = None
NUM_LAYERS: Optional[int] = None
HIDDEN_DIM: Optional[int] = None
INTERMEDIATE_SIZE: Optional[int] = None
VOCAB_SIZE: Optional[int] = None
TP: Optional[int] = None
CP: Optional[int] = None
PP: Optional[int] = None
DP: Optional[int] = None
EP: Optional[int] = None
MB: Optional[int] = None
INFERENCE_REPLICA_COUNT: Optional[int] = None
INFERENCE_MOE_DP: Optional[int] = None
NUM_EXPERTS: Optional[int] = None
TOP_K: Optional[int] = None

FLASH_ATTENTION = "both"
FULL_RECOMPUTATION = "both"
ATTENTION_TILE_SIZE = 128
SKIP_TRAINING = False
SKIP_INFERENCE = False


@dataclass(frozen=True)
class SmokeConfig:
    hardware_config: str = str(HARDWARE_CONFIG)
    training_model_config: str = str(TRAINING_MODEL_CONFIG)
    inference_model_config: str = str(INFERENCE_MODEL_CONFIG)
    mode: str = MODE
    output_dir: str = str(OUTPUT_DIR)
    seq_len: Optional[int] = SEQ_LEN
    decode_len: Optional[int] = DECODE_LEN
    batch_size: Optional[int] = BATCH_SIZE
    global_batch_size: Optional[int] = GLOBAL_BATCH_SIZE
    num_layers: Optional[int] = NUM_LAYERS
    hidden_dim: Optional[int] = HIDDEN_DIM
    intermediate_size: Optional[int] = INTERMEDIATE_SIZE
    vocab_size: Optional[int] = VOCAB_SIZE
    tp: Optional[int] = TP
    cp: Optional[int] = CP
    pp: Optional[int] = PP
    dp: Optional[int] = DP
    ep: Optional[int] = EP
    mb: Optional[int] = MB
    inference_replica_count: Optional[int] = INFERENCE_REPLICA_COUNT
    inference_moe_dp: Optional[int] = INFERENCE_MOE_DP
    flash_attention: str = FLASH_ATTENTION
    full_recomputation: str = FULL_RECOMPUTATION
    attention_tile_size: int = ATTENTION_TILE_SIZE
    num_experts: Optional[int] = NUM_EXPERTS
    top_k: Optional[int] = TOP_K
    skip_training: bool = SKIP_TRAINING
    skip_inference: bool = SKIP_INFERENCE


DEFAULT_CONFIG = SmokeConfig()


def _expand_toggle(value: str):
    if value == "both":
        return [False, True]
    return [value == "on"]


def _set_if_present(obj, attr: str, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _apply_overrides(model_cfg, sched_cfg, cfg: SmokeConfig):
    if cfg.batch_size is not None:
        _set_if_present(model_cfg, "batch_size", int(cfg.batch_size))
        _set_if_present(model_cfg, "global_batch_size", int(cfg.batch_size))
    if cfg.global_batch_size is not None:
        _set_if_present(model_cfg, "global_batch_size", int(cfg.global_batch_size))
        _set_if_present(model_cfg, "batch_size", int(cfg.global_batch_size))

    for key in (
        "seq_len",
        "decode_len",
        "num_layers",
        "hidden_dim",
        "intermediate_size",
        "vocab_size",
    ):
        value = getattr(cfg, key, None)
        if value is not None:
            _set_if_present(model_cfg, key, int(value))

    if cfg.num_experts is not None:
        moe_cfg = getattr(model_cfg, "moe", None)
        if moe_cfg is not None:
            moe_cfg.num_experts = int(cfg.num_experts)
    if cfg.top_k is not None:
        moe_cfg = getattr(model_cfg, "moe", None)
        if moe_cfg is not None:
            moe_cfg.top_k = int(cfg.top_k)
    moe_cfg = getattr(model_cfg, "moe", None)
    if moe_cfg is not None:
        if getattr(model_cfg, "intermediate_size", None) is not None:
            moe_cfg.moe_intermediate_size = int(getattr(model_cfg, "intermediate_size"))
        moe_enabled = bool(
            moe_cfg.num_experts > 1
            or moe_cfg.top_k > 1
            or getattr(moe_cfg, "n_shared_experts", 0) > 0
        )
        if moe_enabled:
            moe_cfg.moe_layer_freq = 1
            moe_cfg.first_k_dense_replace = 0
        else:
            moe_cfg.moe_layer_freq = 1
            moe_cfg.first_k_dense_replace = int(getattr(model_cfg, "num_layers", 1))

    if sched_cfg is not None:
        for key in ("tp", "cp", "pp", "mb"):
            value = getattr(cfg, key, None)
            if value is not None and hasattr(sched_cfg, key):
                setattr(sched_cfg, key, int(value))
        train_cfg = getattr(sched_cfg, "train", None)
        if train_cfg is not None:
            if cfg.dp is not None:
                train_cfg.dp = int(cfg.dp)
            if cfg.ep is not None:
                train_cfg.ep = int(cfg.ep)
        inference_cfg = getattr(sched_cfg, "inference", None)
        if inference_cfg is not None:
            if cfg.inference_replica_count is not None:
                inference_cfg.replica_count = int(cfg.inference_replica_count)
            if cfg.inference_moe_dp is not None:
                inference_cfg.moe_dp = int(cfg.inference_moe_dp)


def _configure_flash_attention(model_cfg, enabled: bool, tile_size: int) -> None:
    attention_cfg = getattr(model_cfg, "attention", None)
    if attention_cfg is None:
        return
    attention_cfg.use_flashattention = bool(enabled)
    if enabled:
        attention_cfg.attention_tile_size = int(tile_size)
    else:
        attention_cfg.attention_tile_size = None


def _configure_full_recompute(hw_cfg, enabled: bool) -> None:
    sw_cfg = getattr(hw_cfg, "sw_config", None)
    if sw_cfg is None:
        return
    if hasattr(sw_cfg, "full_recomputation"):
        sw_cfg.full_recomputation = bool(enabled)


def _load_configs(
    cfg: SmokeConfig,
    *,
    flash_enabled: bool,
    full_recompute: bool,
    model_config_path: Optional[str] = None,
):
    exp_hw_path = os.path.expandvars(os.path.expanduser(cfg.hardware_config))
    model_config_path = model_config_path or cfg.training_model_config
    exp_model_path = os.path.expandvars(os.path.expanduser(model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type=cfg.mode)

    model_cfg = exp_model_config.model_config
    sched_cfg = getattr(exp_hw_config, "sch_config", None)

    _apply_overrides(model_cfg, sched_cfg, cfg)
    _configure_flash_attention(model_cfg, flash_enabled, cfg.attention_tile_size)
    _configure_full_recompute(exp_hw_config, full_recompute)

    config.validate_configs(exp_hw_config, exp_model_config)
    return exp_hw_config, exp_model_config


def _run_inference_case(cfg: SmokeConfig, *, flash_enabled: bool, full_recompute: bool, label: str):
    exp_hw_config, exp_model_config = _load_configs(
        cfg,
        flash_enabled=flash_enabled,
        full_recompute=full_recompute,
        model_config_path=cfg.inference_model_config,
    )
    output_dir = os.path.join(cfg.output_dir, label)
    summary = memory_estimation.estimate_inference_memory(
        exp_hw_config,
        exp_model_config,
        mode=cfg.mode,
        output_dir=output_dir,
    )
    print(f"[inference] {label}")
    print(f"  Prefill peak (per gpu): {summary['prefill_peak_gb']:.2f} GiB")
    print(f"  Final decode peak (per gpu): {summary['decode_peak_gb']:.2f} GiB")
    print(f"  Max peak (per gpu): {summary['max_peak_gb']:.2f} GiB")
    if summary["capacity_gb"] is None:
        print("  Capacity (per gpu): unknown")
    else:
        print(f"  Capacity (per gpu): {summary['capacity_gb']:.2f} GiB")
    if summary["headroom_gb"] is None:
        print("  Headroom: unknown")
    else:
        print(f"  Headroom: {summary['headroom_gb']:.2f} GiB")
    return summary


def _run_training_case(cfg: SmokeConfig, *, flash_enabled: bool, full_recompute: bool, label: str):
    exp_hw_config, exp_model_config = _load_configs(
        cfg,
        flash_enabled=flash_enabled,
        full_recompute=full_recompute,
        model_config_path=cfg.training_model_config,
    )
    output_dir = os.path.join(cfg.output_dir, label)
    summary = memory_estimation.estimate_training_memory(
        exp_hw_config,
        exp_model_config,
        mode=cfg.mode,
        output_dir=output_dir,
    )
    peak = float(summary.get("peak_gb", 0.0) or 0.0)
    print(f"[training] {label}")
    print(f"  Peak (per gpu): {peak:.2f} GiB")
    capacity = summary.get("capacity_gb")
    headroom = summary.get("headroom_gb")
    if capacity is None:
        print("  Capacity (per gpu): unknown")
    else:
        print(f"  Capacity (per gpu): {capacity:.2f} GiB")
    if headroom is None:
        print("  Headroom: unknown")
    else:
        print(f"  Headroom: {headroom:.2f} GiB")
    return peak


def run(cfg: SmokeConfig = DEFAULT_CONFIG) -> int:
    flash_values = _expand_toggle(cfg.flash_attention)
    recompute_values = _expand_toggle(cfg.full_recomputation)

    if cfg.skip_training and cfg.skip_inference:
        raise SystemExit("Both training and inference runs are disabled.")

    training_peaks = {}
    if not cfg.skip_inference:
        for flash_enabled in flash_values:
            label = f"infer_flash_{'on' if flash_enabled else 'off'}"
            _run_inference_case(
                cfg,
                flash_enabled=flash_enabled,
                full_recompute=False,
                label=label,
            )

    if not cfg.skip_training:
        for full_recompute in recompute_values:
            for flash_enabled in flash_values:
                label = (
                    f"train_flash_{'on' if flash_enabled else 'off'}_recompute_"
                    f"{'on' if full_recompute else 'off'}"
                )
                peak = _run_training_case(
                    cfg,
                    flash_enabled=flash_enabled,
                    full_recompute=full_recompute,
                    label=label,
                )
                training_peaks[(flash_enabled, full_recompute)] = peak

        if len(recompute_values) > 1:
            for flash_enabled in flash_values:
                peak_no = training_peaks.get((flash_enabled, False))
                peak_yes = training_peaks.get((flash_enabled, True))
                if peak_no is None or peak_yes is None:
                    continue
                if peak_yes > peak_no + 1e-6:
                    raise RuntimeError(
                        "Full recomputation did not reduce training peak memory "
                        f"(flash={flash_enabled}): no_recompute={peak_no:.3f} GiB, "
                        f"recompute={peak_yes:.3f} GiB."
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
