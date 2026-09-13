"""Checkpoint loading across stage and precision boundaries.

Post-training loads a checkpoint that pretraining wrote, and the two stages do
not necessarily run at the same precision (pretraining uses NVFP4 for speed;
the RL stages use BF16 because ratio/KL objectives are far more numerically
fragile than a plain cross-entropy). That transition has two traps, both hit
for real:

  1. TE's `_extra_state`. A model built with precision="nvfp4"/"fp8" uses
     Transformer Engine linears, which add one pickled `_extra_state` entry per
     linear holding FP8/FP4 scaling metadata. A BF16 model uses plain
     nn.Linear and has none, so a strict load fails with 4 "unexpected keys"
     per block that are not a corrupt checkpoint at all.
  2. TE refuses to UNPICKLE that state unless
     NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 is set before TE is imported. The
     failure only occurs on the load path, so a cold start looks perfectly
     healthy and every resume dies.

`load_weights` handles (1) explicitly and loudly: quantization state may be
dropped when the target does not want it, but a missing or unexpected
PARAMETER is still an error, because that is the case that silently yields a
randomly-initialised layer.
"""

from __future__ import annotations

import os

# Must precede any import that pulls in Transformer Engine. See (2) above.
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

import torch

_QUANT_SUFFIX = "_extra_state"


def load_weights(model, state_dict: dict, strict_params: bool = True) -> dict:
    """Load `state_dict` into `model`, tolerating only quantization-state drift.

    Returns a report dict. Raises if real parameters are missing/unexpected,
    since a silently unloaded layer is indistinguishable from a bad checkpoint
    until many training hours later.
    """
    target_keys = set(model.state_dict().keys())
    dropped = [k for k in state_dict
               if k.endswith(_QUANT_SUFFIX) and k not in target_keys]
    filtered = {k: v for k, v in state_dict.items() if k not in set(dropped)}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if not k.endswith(_QUANT_SUFFIX)]
    unexpected = [k for k in unexpected if not k.endswith(_QUANT_SUFFIX)]
    if strict_params and (missing or unexpected):
        raise RuntimeError(
            f"parameter mismatch loading checkpoint: missing={missing[:8]} "
            f"(n={len(missing)}), unexpected={unexpected[:8]} (n={len(unexpected)}). "
            "These are real weights, not quantization metadata -- loading anyway "
            "would leave layers randomly initialised."
        )
    return {"dropped_quant_state": len(dropped),
            "missing": missing, "unexpected": unexpected}


def load_stage_checkpoint(path: str, model, map_location="cpu") -> dict:
    """Load a previous stage's checkpoint: WEIGHTS ONLY, never the optimizer.

    Each stage starts a fresh optimizer on purpose. Adam moments accumulated
    under a cross-entropy objective are meaningless under a policy-gradient one,
    and carrying them across is a good way to get a first RL step that undoes
    the fine-tune.
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = ck.get("model", ck)
    report = load_weights(model, sd)
    report["step"] = ck.get("step")
    report["cfg"] = ck.get("cfg")
    return report
