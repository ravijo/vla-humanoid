"""Low-memory GR00T N1.7 fine-tuning for the Unitree G1 (UNITREE_G1 embodiment tag).

Wraps Isaac-GR00T's gr00t/experiment/launch_finetune.py without editing it. Changes:
  1. UNITREE_G1's pre-registered action horizon (50) exceeds the N1.7 base model's maximum (40),
     so it is reset to --action-horizon (default 40 = 0.8 s at 50 Hz).
  2. Frozen parameters are cast to bf16 (stock launcher keeps everything fp32).
  3. --dit-mode controls how much of the 1.08B-param DiT is trained:
       full    : stock behaviour (all of the DiT trainable; ~26 GB of weights/grads/Adam state)
       frozen  : DiT frozen; projector (state/action encoders+decoder) and VL self-attention train
       top<k>  : like frozen, plus the last k DiT blocks trainable (e.g. top4)
       lora    : like frozen, plus LoRA adapters on every Linear in the DiT blocks
     LoRA weights are merged into the base weights whenever a state_dict is taken, so saved
     checkpoints have the stock key layout and load with gr00t/eval/run_gr00t_server.py.
     (Consequence: --resume-from-checkpoint is not supported in lora mode.)

Usage (all unrecognised args go to launch_finetune.py; defaults are the settings used in the README):
  third_party/Isaac-GR00T/.venv/bin/python scripts/finetune_g1.py \
      --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <ds> --embodiment-tag UNITREE_G1 \
      --output-dir <out> --global-batch-size 32 --max-steps 30000
"""

import argparse
import logging
import math
import os
import runpy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

GR00T_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Isaac-GR00T"
G1_TAG = "unitree_g1_full_body_with_waist_height_nav_cmd"

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.model.gr00t_n1d7 import setup as n1d7_setup
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear


# Report real GPU memory use (nvidia-smi only shows the allocator's reserved pool).
import gr00t.experiment.experiment as exp
from transformers import TrainerCallback


class LoRALinear(nn.Module):
    """y = base(x) + scale * (x A^T) B^T. Keeps the wrapped Linear's own parameters intact."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data = self.lora_A.data.to(base.weight.device)
        self.lora_B.data = self.lora_B.data.to(base.weight.device)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale


def _inject_lora(root: nn.Module, rank: int, alpha: float) -> list[str]:
    wrapped = []
    for name, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRALinear(child, rank, alpha))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return wrapped


class SlotDeltaCSL(nn.Module):
    """Wraps a CategorySpecificLinear (one weight slice per embodiment, 32 slots) so that only a
    delta for a single embodiment slot is trained. The full W/b stay frozen (bf16)."""

    def __init__(self, base: nn.Module, slot: int):
        super().__init__()
        self.base = base
        self.slot = slot
        self.slot_dW = nn.Parameter(torch.zeros_like(base.W[slot], dtype=torch.float32))
        self.slot_db = nn.Parameter(torch.zeros_like(base.b[slot], dtype=torch.float32))

    def forward(self, x, cat_ids):
        on = (cat_ids == self.slot).to(x.dtype)  # [B]
        W = self.base.W[cat_ids].to(x.dtype) + on[:, None, None] * self.slot_dW.to(x.dtype)
        b = self.base.b[cat_ids].to(x.dtype) + on[:, None] * self.slot_db.to(x.dtype)
        return torch.bmm(x, W) + b.unsqueeze(1)


def _wrap_category_specific(root: nn.Module, slot: int) -> int:
    n = 0
    for mod in list(root.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, CategorySpecificLinear):
                child.requires_grad_(False)
                setattr(mod, child_name, SlotDeltaCSL(child, slot))
                n += 1
    return n


def _merge_adapters_state_dict_hook(module, state_dict, prefix, local_metadata):
    """Fold trained adapters back into the stock key layout so checkpoints load unmodified:
    LoRA: '<p>.base.weight' + scale * B @ A -> '<p>.weight'
    slot delta: '<p>.base.W'[slot] += dW -> '<p>.W' (same for b)."""
    for key in [k for k in list(state_dict) if k.endswith(".lora_A")]:
        p = key[: -len(".lora_A")]
        A = state_dict.pop(f"{p}.lora_A").float()
        B = state_dict.pop(f"{p}.lora_B").float()
        w = state_dict.pop(f"{p}.base.weight")
        state_dict[f"{p}.weight"] = w.float() + _ADAPTER["lora_scale"] * (B @ A)
        if f"{p}.base.bias" in state_dict:
            state_dict[f"{p}.bias"] = state_dict.pop(f"{p}.base.bias")
    for key in [k for k in list(state_dict) if k.endswith(".slot_dW")]:
        p = key[: -len(".slot_dW")]
        slot = _ADAPTER["slot"]
        W = state_dict.pop(f"{p}.base.W").float().clone()
        b = state_dict.pop(f"{p}.base.b").float().clone()
        W[slot] += state_dict.pop(f"{p}.slot_dW").float()
        b[slot] += state_dict.pop(f"{p}.slot_db").float()
        state_dict[f"{p}.W"], state_dict[f"{p}.b"] = W, b
    return state_dict


_ADAPTER = {"lora_scale": 1.0, "slot": None}


def _postprocess_model(model, args, slot):
    ah = model.action_head
    log = logging.getLogger("finetune_g1")
    _ADAPTER["lora_scale"] = args.lora_alpha / args.lora_rank
    _ADAPTER["slot"] = slot
    use_hook = False
    if args.dit_mode != "full":
        ah.model.requires_grad_(False)
    if args.dit_mode.startswith("top"):
        k = int(args.dit_mode[3:])
        for blk in ah.model.transformer_blocks[-k:]:
            blk.requires_grad_(True)
    if args.dit_mode == "lora":
        wrapped = _inject_lora(ah.model.transformer_blocks, args.lora_rank, args.lora_alpha)
        use_hook = True
        log.info(f"DiT LoRA r={args.lora_rank} alpha={args.lora_alpha} on {len(wrapped)} Linear layers")
    if args.vlsa_mode != "full":
        ah.vl_self_attention.requires_grad_(False)
    if args.vlsa_mode == "lora":
        wrapped = _inject_lora(ah.vl_self_attention.transformer_blocks, args.lora_rank, args.lora_alpha)
        use_hook = True
        log.info(f"VL-self-attn LoRA on {len(wrapped)} Linear layers")
    if args.projector_mode == "slot":
        n = _wrap_category_specific(ah, slot)
        use_hook = True
        log.info(f"Projector: training only embodiment slot {slot} in {n} category-specific layers")
    if use_hook:
        model._register_state_dict_hook(_merge_adapters_state_dict_hook)
    if args.frozen_bf16:
        for p in model.parameters():
            if not p.requires_grad and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[finetune_g1] dit_mode={args.dit_mode} trainable={train:,} ({100 * train / total:.2f}%)")
    print(f"[finetune_g1] dit_mode={args.dit_mode} trainable={train:,} ({100 * train / total:.2f}%)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dit-mode", default="lora")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--vlsa-mode", default="lora", choices=["full", "frozen", "lora"])
    ap.add_argument("--projector-mode", default="slot", choices=["full", "slot"])
    ap.add_argument("--action-horizon", type=int, default=40)
    ap.add_argument("--no-frozen-bf16", dest="frozen_bf16", action="store_false")
    args, rest = ap.parse_known_args()
    assert args.dit_mode in ("full", "frozen", "lora") or (
        args.dit_mode.startswith("top") and args.dit_mode[3:].isdigit()
    ), f"bad --dit-mode {args.dit_mode}"

    sys.path.insert(0, str(GR00T_DIR))
    os.chdir(GR00T_DIR)

    # DeepSpeed is installed but only used for multi-GPU runs. Merely importing it needs a CUDA
    # toolkit (CUDA_HOME/bin/nvcc), and accelerate/transformers import it whenever the package is
    # present. Hide it when no toolkit is set up, so a driver-only machine can still train.
    if not os.environ.get("CUDA_HOME"):
        import accelerate.utils as au
        import accelerate.utils.imports as ai
        import accelerate.utils.other as ao
        import transformers.integrations.deepspeed as td
        import transformers.utils.import_utils as tu

        for mod in (au, ai, ao, td, tu):
            if hasattr(mod, "is_deepspeed_available"):
                mod.is_deepspeed_available = lambda: False

    MODALITY_CONFIGS[G1_TAG]["action"].delta_indices = list(range(args.action_horizon))

    pipeline_cls = next(
        c for c in vars(n1d7_setup).values() if isinstance(c, type) and hasattr(c, "_create_model")
    )
    orig_create = pipeline_cls._create_model

    def create_model(self):
        model = orig_create(self)

        # Slot 25 is shared with the real-G1 pretraining data, so the G1 projector starts pretrained.
        slot = EMBODIMENT_TAG_TO_PROJECTOR_INDEX[G1_TAG]
        return _postprocess_model(model, args, slot)

    pipeline_cls._create_model = create_model

    class MemCallback(TrainerCallback):
        def on_train_begin(self, targs, state, control, model=None, **kw):
            by = {}
            for p in model.parameters():
                k = (str(p.dtype), p.requires_grad, str(p.device))
                by[k] = by.get(k, 0) + p.numel() * p.element_size()
            for k, v in sorted(by.items()):
                print(f"[mem] params dtype={k[0]} trainable={k[1]} device={k[2]}: {v / 2**30:.2f} GiB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, targs, state, control, **kw):
            if state.global_step in (1, 2, 5) or state.global_step % 200 == 0:
                print(
                    f"[mem] step {state.global_step}: allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )

    base_trainer = exp.Gr00tTrainer

    class Trainer(base_trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(MemCallback())

    exp.Gr00tTrainer = Trainer

    sys.argv = [str(GR00T_DIR / "gr00t/experiment/launch_finetune.py"), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
