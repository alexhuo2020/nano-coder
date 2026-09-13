"""Group-relative policy optimisation, with the corrections that a previous run
in this lineage needed before it would train at all.

Four things here are not the textbook version, each because the textbook version
measurably failed:

  1. SEQUENCE-LEVEL RATIO (GSPO). The importance ratio is
     exp(mean_t(logp_new - logp_old)) over the completion, not a per-token
     ratio. Per-token ratios on long completions give a product that is
     numerically hopeless and an update dominated by whichever token happened
     to move most.
  2. THE KL IS LENGTH-NORMALISED TOO. When the ratio was normalised by length
     and the KL was not, the KL term was ~1748 against an objective of order 1
     and accounted for 69.94 of a 69.93 total loss -- i.e. the policy gradient
     was numerically absent. Both terms must be per-token or neither.
  3. THE OBJECTIVE IS COMPUTED IN FP32. In bf16 the Schulman k3 estimator came
     out NEGATIVE (about -1/1024), which is impossible for a KL and was pure
     accumulated rounding.
  4. ZERO-VARIANCE GROUPS ARE DROPPED, NOT DIVIDED. If every sample in a group
     scores the same, the advantage is 0/0. Guarding with a small epsilon
     silently turns rounding noise into a training signal, so such groups are
     skipped and counted instead.

Also DAPO's Clip-Higher: the upper clip is looser than the lower, which keeps
low-probability-but-correct tokens from being clipped away and is what stops
entropy collapsing early.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GRPOConfig:
    group_size: int = 8
    clip_low: float = 0.2          # DAPO Clip-Higher: asymmetric on purpose
    clip_high: float = 0.28
    kl_beta: float = 0.02
    update_epochs: int = 2         # >1 so the clipping is actually reachable
    max_new_tokens: int = 256


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8):
    """(reward - mean) / std within the group, or None if there is no spread.

    Returning None rather than a guarded division is the point: a group where
    every sample scored identically carries no preference information, and
    dividing its zero spread by an epsilon manufactures a large advantage out of
    floating-point noise.
    """
    r = rewards.float()
    std = r.std(unbiased=False)
    if not torch.isfinite(std) or std.item() < 1e-6:
        return None
    return (r - r.mean()) / (std + eps)


def sequence_ratio(new_logprobs, old_logprobs, valid):
    """GSPO's sequence-level ratio, length-normalised: exp(mean_t delta)."""
    n = valid.sum(-1).clamp(min=1)
    delta = ((new_logprobs - old_logprobs) * valid).sum(-1) / n
    return torch.exp(delta), n


def k3_kl(new_logprobs, ref_logprobs, valid):
    """Schulman's k3 estimator, per token, always >= 0.

    k3 = exp(r) - r - 1 with r = log pi_ref - log pi_new. Unlike the naive
    (logp_new - logp_ref) mean, k3 is a proper non-negative KL estimate, which
    also makes it a usable health check: a negative value means numerics, not
    a real divergence.
    """
    r = (ref_logprobs - new_logprobs).clamp(-20, 20)
    per_token = torch.exp(r) - r - 1.0
    n = valid.sum(-1).clamp(min=1)
    return (per_token * valid).sum(-1) / n


def grpo_objective(new_logprobs, old_logprobs, ref_logprobs, valid, advantages,
                   cfg: GRPOConfig):
    """Returns (loss, metrics). Everything in fp32; see the module docstring."""
    new_logprobs = new_logprobs.float()
    old_logprobs = old_logprobs.float()
    valid = valid.float()
    adv = advantages.float()

    ratio, n_tok = sequence_ratio(new_logprobs, old_logprobs, valid)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip_low, 1.0 + cfg.clip_high) * adv
    policy = -torch.min(unclipped, clipped).mean()

    if ref_logprobs is None:
        kl = torch.zeros((), device=new_logprobs.device)
    else:
        kl = k3_kl(new_logprobs, ref_logprobs.float(), valid).mean()

    loss = policy + cfg.kl_beta * kl
    frac_clipped = ((ratio < 1.0 - cfg.clip_low) | (ratio > 1.0 + cfg.clip_high))
    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio.detach().mean(),
        "frac_clipped": frac_clipped.float().mean().detach(),
        "mean_completion_len": n_tok.float().mean().detach(),
    }
    return loss, metrics
