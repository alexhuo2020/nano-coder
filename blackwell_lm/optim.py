"""Optimizer wrapper that keeps FP32 master weights.

WHY THIS EXISTS. Transformer Engine's NVFP4 path requires bfloat16 parameters
(its quantizer rejects fp32 outright: "RHT is only supported for bfloat16
input"). But bf16 cannot hold the optimizer's state, because it cannot represent
the updates. bf16 keeps 7 mantissa bits, so the gap between representable values
at magnitude x is 2**(floor(log2 x) - 7); an AdamW step is roughly lr. Anything
smaller than about half that gap rounds straight back to where it started:

    |w| ~ 0.03  ->  spacing 1.22e-4
      lr 3e-5  (pretrain floor)   0.25 x spacing   mostly lost
      lr 1e-5  (SFT)              0.08 x spacing   lost
      lr 1e-6  (GRPO)             0.008 x spacing  entirely lost
      lr 5e-7  (agentic RL)       0.004 x spacing  entirely lost

This was found the expensive way. Pretraining ran 657,000 steps with all five
RMSNorm gains still bit-exactly 1.0 -- they had never moved once -- while the
embedding was silently discarding 44% of its updates. Nothing raised, and the
loss still fell, because the weight matrices happened to sit just above the
resolution limit. The post-training stages run at 3x-60x lower learning rates,
so without this most of the model simply cannot move. MEASURED on a small
model: at lr 1e-5 bf16-in-place updates 8.2% of elements versus 44.9% with
masters; at lr 1e-6 it is 2.0% versus 15.0%. Not literally zero -- spacing
scales with magnitude, so the smallest weights still move -- but a stage running
on 2% of its parameters looks like the far more famous "zero update steps"
failure and would be misdiagnosed for a long time.

TWO POLICY CHOICES, both learned from the live run:

  * 1-D PARAMETERS (normalisation gains) ARE FROZEN by default. Unfreezing them
    mid-run was measurably harmful: their optimizer `exp_avg` had accumulated
    657k steps of gradients whose updates were being thrown away, so the moment
    updates could land that stale momentum discharged -- gains went 1.0 to mean
    0.67/0.70/2.17 in 39k steps and held-out loss rose 0.052. A fixed unit gain
    is a legitimate design, and these are ~0.006% of parameters.
  * The masters are the REAL weights; the bf16 params are a compute-only copy
    refreshed after each step. Checkpoint the masters, or a resume silently
    rounds the model back down.

Measured overhead: ~3% of step time, ~1GB for an 85M model.
"""

from __future__ import annotations

import torch


class MasterWeightOptimizer:
    """AdamW over FP32 masters, mirroring into bf16 params after each step.

    Presents the small slice of the torch.optim.Optimizer surface the training
    scripts actually use, so it can be dropped in where AdamW was.
    """

    def __init__(self, model, lr: float, betas=(0.9, 0.95), weight_decay: float = 0.0,
                 grad_clip: float | None = 1.0, freeze_1d: bool = True, fused: bool = True):
        if freeze_1d:
            for _n, p in model.named_parameters():
                if p.ndim == 1:
                    p.requires_grad_(False)
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("no trainable parameters left after freezing 1-D tensors")
        self.master = [p.detach().float().clone() for p in self.params]
        for q in self.master:
            q.requires_grad_(True)
            q.grad = torch.zeros_like(q)      # preallocated once, reused every step
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas,
                                     weight_decay=weight_decay, fused=fused)

    # ---------------------------------------------------------------- stepping

    def step(self):
        """Copy bf16 grads up, clip, step, and mirror masters back down."""
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            # A None grad means that tensor took no part in the loss. Filling it
            # with zeros keeps the fused foreach path usable and is equivalent to
            # the parameter simply not moving this step.
            grads = [torch.zeros_like(p) if p.grad is None else p.grad
                     for p in self.params]
        torch._foreach_copy_([q.grad for q in self.master], grads)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.master, self.grad_clip)
        self.opt.step()
        with torch.no_grad():
            torch._foreach_copy_(self.params, self.master)

    def zero_grad(self):
        # set_to_none=False on BOTH: the master grads are preallocated once in
        # __init__ and reused, so nulling them would leave step() copying into
        # None on the very next iteration. (It did; the test caught it.)
        self.opt.zero_grad(set_to_none=False)
        self.model.zero_grad(set_to_none=False)

    # ------------------------------------------------------------------- state

    @property
    def param_groups(self):
        return self.opt.param_groups

    def set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def state_dict(self):
        return {"opt": self.opt.state_dict(),
                "master": [q.detach().cpu() for q in self.master]}

    def load_state_dict(self, sd, model_params_as_fallback: bool = True):
        """Restore masters and moments.

        A checkpoint written before master weights existed has no "master" key;
        seeding from the bf16 weights is then exact, since bf16 -> fp32 is a
        lossless upcast. Optimizer state saved for MORE parameters than are now
        trainable is remapped by position rather than discarded -- dropping Adam
        moments mid-run spikes the loss for hundreds of steps.
        """
        if "master" in sd and len(sd["master"]) == len(self.master):
            with torch.no_grad():
                torch._foreach_copy_(self.master,
                                     [t.to(self.master[0].device) for t in sd["master"]])
            restored = "masters restored from checkpoint"
        elif model_params_as_fallback:
            with torch.no_grad():
                torch._foreach_copy_(self.master, self.params)
            restored = "masters seeded from bf16 weights (lossless upcast)"
        else:
            raise KeyError("checkpoint has no master weights")

        raw = sd.get("opt", sd)
        if "param_groups" in raw:
            n = len(self.master)
            if len(raw["param_groups"][0]["params"]) != n:
                all_p = list(self.model.parameters())
                keep = [i for i, q in enumerate(all_p) if q.requires_grad]
                state = {new: raw["state"][old] for new, old in enumerate(keep)
                         if old in raw["state"]}
                raw = {"state": state,
                       "param_groups": [{**raw["param_groups"][0],
                                         "params": list(range(n))}]}
                restored += f"; optimizer state remapped to {len(state)}/{n} tensors"
            self.opt.load_state_dict(raw)
        return restored


def frozen_report(model) -> str:
    """One line describing what is trainable, for the run log."""
    tr = [n for n, p in model.named_parameters() if p.requires_grad]
    fr = [n for n, p in model.named_parameters() if not p.requires_grad]
    n_fr = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return (f"{len(tr)} trainable tensors, {len(fr)} frozen 1-D gains "
            f"({n_fr/max(1,n_all)*100:.3f}% of params); fp32 masters active")
