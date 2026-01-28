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

from itertools import product
from pathlib import Path
from pprint import pformat
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest

from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
MODEL_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"

# Small 7B-ish settings for fast coverage.
HIDDEN_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128
GLM_HEAD_DIM = 128
INTERMEDIATE = 11008
VOCAB_SIZE = 32000
GLOBAL_BATCH = 8
SEQ_LEN_TRAIN = 128
SEQ_LEN_INF = 256
DECODE_LEN = 64
GQA_KV_HEADS = 8
MOE_EXPERTS = 4

BACKEND_OVERRIDES: Dict[str, Dict[str, object]] = {
  "analytical": {"model": "analytical"},
  "hybrid": {"model": "astra", "astra": {"mode": "hybrid"}},
  "hierarchical": {"model": "astra", "astra": {"mode": "full_astrasim_hierarchical"}},
  "flattened": {"model": "astra", "astra": {"mode": "full_astrasim_flattened"}},
}

PARALLELISM_TRAIN: Tuple[Dict[str, object], ...] = (
  {"dp": 1, "tp": 1, "cp": 1, "pp": 1, "mb": 1, "tp_sp": False},
  {"dp": 2, "tp": 1, "cp": 1, "pp": 2, "mb": 2, "tp_sp": False},
  {"dp": 1, "tp": 2, "cp": 1, "pp": 2, "mb": 2, "tp_sp": True},
  {"dp": 1, "tp": 1, "cp": 2, "pp": 2, "mb": 2, "tp_sp": False},
  {"dp": 2, "tp": 2, "cp": 2, "pp": 2, "mb": 2, "tp_sp": True},
)

PARALLELISM_INF: Tuple[Dict[str, object], ...] = (
  {"dp": 1, "tp": 1, "cp": 1, "pp": 1, "mb": 1, "tp_sp": False},
  {"dp": 2, "tp": 1, "cp": 1, "pp": 2, "mb": 2, "tp_sp": False},
  {"dp": 1, "tp": 2, "cp": 1, "pp": 2, "mb": 2, "tp_sp": True},
  {"dp": 2, "tp": 2, "cp": 1, "pp": 2, "mb": 2, "tp_sp": True},
)

MODEL_TYPES = ("gpt", "llama", "glm4_moe")
ATTN_TYPES = ("mha", "gqa")
BACKENDS = tuple(BACKEND_OVERRIDES.keys())


def _parse_noop(_output: str, _spec: ValidationSpec) -> Dict[str, object]:
  return {}


def _build_attention(attention_type: str, head_dim: int) -> Dict[str, object]:
  attention: Dict[str, object] = {
    "attention_type": attention_type,
    "num_heads": NUM_HEADS,
    "head_dim": int(head_dim),
    "use_flashattention": False,
    "attention_tile_size": 128,
  }
  if attention_type == "gqa":
    attention["kv_heads"] = GQA_KV_HEADS
  else:
    attention["kv_heads"] = NUM_HEADS
  return attention


def _build_model_overrides(
  *,
  model_type: str,
  attention_type: str,
  run_type: str,
  use_moe: bool,
  num_layers: int,
) -> Dict[str, object]:
  model_param: Dict[str, object] = {
    "run_type": run_type,
    "model_type": model_type,
    "global_batch_size": GLOBAL_BATCH,
    "gradient_accumulation_steps": 1,
    "seq_len": SEQ_LEN_INF if run_type == "inference" else SEQ_LEN_TRAIN,
    "hidden_dim": HIDDEN_DIM,
    "intermediate_size": INTERMEDIATE,
    "vocab_size": VOCAB_SIZE,
    "num_layers": int(num_layers),
    "attention": _build_attention(
      attention_type,
      GLM_HEAD_DIM if model_type == "glm4_moe" else HEAD_DIM,
    ),
  }
  if run_type == "inference":
    model_param["decode_len"] = DECODE_LEN
  if use_moe:
    model_param["moe"] = {
      "num_experts": MOE_EXPERTS,
      "top_k": 1,
      "moe_intermediate_size": INTERMEDIATE,
      "n_shared_experts": 0,
      "moe_layer_freq": 1,
      "first_k_dense_replace": 0,
    }
  return {"model_param": model_param}


def _build_hw_overrides(
  *,
  backend: str,
  dp: int,
  ep: int,
  tp: int,
  cp: int,
  pp: int,
  mb: int,
  tp_sp: bool,
  run_type: str,
  use_moe: bool,
) -> Dict[str, object]:
  normalized = str(run_type or "training").lower()
  inference_replica = int(dp) if normalized == "inference" else 1
  inference_moe_dp = int(dp) if normalized == "inference" and use_moe else 1
  overrides: Dict[str, object] = {
    "parallelism": {
      "tp": int(tp),
      "tp_sp": bool(tp_sp),
      "cp": int(cp),
      "pp": int(pp),
      "mb": int(mb),
      "train": {"dp": int(dp), "ep": int(ep), "tp_ep": True},
      "inference": {"replica_count": inference_replica, "moe_dp": inference_moe_dp},
    },
    "execution_backend": BACKEND_OVERRIDES[backend],
  }
  if backend == "analytical":
    overrides["network"] = {
      "dimensions": [
        {
          "id": "dim0",
          "topology": {"type": "Ring"},
        }
      ]
    }
  return overrides


def _build_dense_specs(run_type: str) -> List[ValidationSpec]:
  specs: List[ValidationSpec] = []
  parallelism_grid = PARALLELISM_INF if run_type == "inference" else PARALLELISM_TRAIN
  for idx, (model_type, attention_type, backend, par) in enumerate(
    product(MODEL_TYPES, ATTN_TYPES, BACKENDS, parallelism_grid)
  ):
    pp = int(par["pp"])
    num_layers = 2 * pp
    label = (
      f"{run_type}:dense:{model_type}:{attention_type}:{backend}"
      f":dp{par['dp']}:ep1:tp{par['tp']}:cp{par['cp']}"
      f":pp{pp}:mb{par['mb']}:tp_sp{par['tp_sp']}"
    )
    spec = ValidationSpec(
      label=label,
      model_overrides=_build_model_overrides(
        model_type=model_type,
        attention_type=attention_type,
        run_type=run_type,
        use_moe=False,
        num_layers=num_layers,
      ),
      hardware_overrides=_build_hw_overrides(
        backend=backend,
        dp=int(par["dp"]),
        ep=1,
        tp=int(par["tp"]),
        cp=int(par["cp"]),
        pp=pp,
        mb=int(par["mb"]),
        tp_sp=bool(par["tp_sp"]),
        run_type=run_type,
        use_moe=False,
      ),
      metadata={
        "suite": "dense",
        "model_type": model_type,
        "attention_type": attention_type,
        "backend": backend,
        "run_type": run_type,
        "parallelism": dict(par),
      },
      order=idx,
    )
    specs.append(spec)
  return specs


def _build_moe_specs(run_type: str) -> List[ValidationSpec]:
  specs: List[ValidationSpec] = []
  moe_variants: Iterable[Tuple[int, int, int, int, int, int, bool]] = (
    (2, 1, 2, 2, 2, 2, True),
    (2, 2, 2, 1, 2, 2, True),
  )
  for idx, (attention_type, backend, par) in enumerate(
    product(ATTN_TYPES, BACKENDS, moe_variants)
  ):
    # skip flattened backend
    if backend == "flattened":
      continue
    dp, ep, tp, cp, pp, mb, tp_sp = par
    if run_type == "inference" and cp > 1:
      continue
    if cp > 1 and ep > 1:
      continue
    if ep > 1 and tp > 1 and not tp_sp:
      continue
    num_layers = 2 * int(pp)
    label = (
      f"{run_type}:moe:glm4_moe:{attention_type}:{backend}"
      f":dp{dp}:ep{ep}:tp{tp}:cp{cp}:pp{pp}:mb{mb}:tp_sp{tp_sp}"
    )
    spec = ValidationSpec(
      label=label,
      model_overrides=_build_model_overrides(
        model_type="glm4_moe",
        attention_type=attention_type,
        run_type=run_type,
        use_moe=True,
        num_layers=num_layers,
      ),
      hardware_overrides=_build_hw_overrides(
        backend=backend,
        dp=dp,
        ep=ep,
        tp=tp,
        cp=cp,
        pp=pp,
        mb=mb,
        tp_sp=tp_sp,
        run_type=run_type,
        use_moe=True,
      ),
      metadata={
        "suite": "moe",
        "model_type": "glm4_moe",
        "attention_type": attention_type,
        "backend": backend,
        "run_type": run_type,
        "parallelism": {
          "dp": dp,
          "ep": ep,
          "tp": tp,
          "cp": cp,
          "pp": pp,
          "mb": mb,
          "tp_sp": tp_sp,
        },
      },
      order=idx,
    )
    specs.append(spec)
  return specs


def _format_failure(res) -> str:
  spec = res.spec
  block = [
    f"[{spec.label}]",
    f"returncode: {res.returncode}",
    f"error: {res.error}",
    f"duration_s: {res.duration_s:.2f}",
    f"model_config_used: {res.model_config_used}",
    f"hardware_config_used: {res.hardware_config_used}",
    f"metadata: {pformat(spec.metadata)}",
    f"model_overrides: {pformat(spec.model_overrides)}",
    f"hardware_overrides: {pformat(spec.hardware_overrides)}",
    "raw_output:",
    res.raw_output or "<empty>",
  ]
  return "\n".join(block)


def _ensure_unique_labels(specs: Sequence[ValidationSpec]) -> None:
  seen = set()
  dupes = []
  for spec in specs:
    if spec.label in seen:
      dupes.append(spec.label)
    seen.add(spec.label)
  if dupes:
    raise ValueError(f"Duplicate ValidationSpec labels detected: {sorted(set(dupes))}")


TRAIN_SPECS = _build_dense_specs("training") + _build_moe_specs("training")
INF_SPECS = _build_dense_specs("inference") + _build_moe_specs("inference")
ALL_SPECS = TRAIN_SPECS + INF_SPECS
_ensure_unique_labels(ALL_SPECS)
SPEC_LABELS = [spec.label for spec in ALL_SPECS]
XFAIL_LABELS = {
  "training:moe:glm4_moe:mha:hierarchical:dp2:ep2:tp2:cp1:pp2:mb2:tp_spTrue",
  "training:moe:glm4_moe:gqa:hierarchical:dp2:ep2:tp2:cp1:pp2:mb2:tp_spTrue",
}
PARAMS = [
  pytest.param(
    label,
    marks=pytest.mark.xfail(reason="Known failure in hierarchical MoE + ep training", strict=False)
    if label in XFAIL_LABELS
    else (),
    id=label,
  )
  for label in SPEC_LABELS
]


@pytest.fixture(scope="module")
def func_test_results():
  results = run_validation_suite(
    ALL_SPECS,
    base_model_config_path=str(MODEL_BASE),
    base_hardware_config_path=str(HW_BASE),
    result_parser=_parse_noop,
  )
  return {res.spec.label: res for res in results}


@pytest.mark.parametrize("label", PARAMS)
def test_func_test(label, func_test_results):
  result = func_test_results.get(label)
  assert result is not None, f"Missing validation result for {label}"
  if not result.success:
    pytest.fail(_format_failure(result))
