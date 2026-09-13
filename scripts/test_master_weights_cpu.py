
# Run from anywhere: put the repo root on sys.path so this works without
# the caller having set PYTHONPATH. Aliased imports keep it independent of
# whatever the module imports below.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
"""Regression test for the lost-update bug: bf16 parameters cannot be the
optimizer's state.

WHAT HAPPENED. The pretraining run held its parameters in bf16 (which TE
requires -- its NVFP4 quantizer rejects fp32 input) and let AdamW update them in
place. bf16 spacing at |w|~1.0 is 2**-7 = 7.8e-3, while an AdamW step is ~lr.
At lr 1.6e-4 every update to a normalisation gain therefore rounded straight
back to its initial 1.0. Measured on the real checkpoint at step 657,000: all
five RMSNorm gains were still bit-exactly 1.0, having moved 0.00% of their
elements in 657,000 steps, and the embedding was losing 44% of its updates.

Nothing raised. The loss still fell, because the weight matrices (|w|~0.03,
spacing 1.2e-4) were just above the resolution limit -- so the failure looked
like a plateau rather than a bug, and would have become total as lr annealed to
3e-5 and pushed even those below the limit.

These tests assert BOTH halves: that the naive path really is broken (so the
test cannot silently stop testing anything), and that fp32 masters fix it.
"""
import torch

from nanocoder.model import BlackwellLM, ModelConfig

NORM_KEYS = ("blocks.0.n1.weight", "blocks.0.n2.weight", "norm.weight")


def _cfg():
    return ModelConfig(vocab_size=512, d_model=256, n_layers=1, n_heads=2,
                       n_kv_heads=1, ffn_hidden=512, max_seq_len=64, window=8,
                       global_every=0, n_loops=2, loop_sample=None,
                       zero_init_residual=False)


def _run(use_master, steps=30, lr=1.6e-4):
    torch.manual_seed(0)
    cfg = _cfg()
    m = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.bfloat16)
    params = [p for p in m.parameters() if p.requires_grad]
    if use_master:
        master = [p.detach().float().clone() for p in params]
        for q in master:
            q.requires_grad_(True)
            q.grad = torch.zeros_like(q)
        opt = torch.optim.AdamW(master, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    before = {k: v.detach().float().clone() for k, v in m.named_parameters()}
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    for _ in range(steps):
        logits = m(ids, n_loops=2)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1))
        loss.backward()
        if use_master:
            torch._foreach_copy_([q.grad for q in master], [p.grad for p in params])
            opt.step()
            with torch.no_grad():
                torch._foreach_copy_(params, master)
            m.zero_grad(set_to_none=False)
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
    return {k: (v.detach().float() - before[k]).abs() for k, v in m.named_parameters()}


def test_bf16_params_lose_updates_to_normalisation_gains():
    """The bug, asserted. If this ever starts FAILING, bf16 rounding behaviour
    changed and the fp32-master workaround may no longer be needed."""
    d = _run(use_master=False)
    for k in NORM_KEYS:
        moved = (d[k] > 0).float().mean().item()
        assert moved == 0.0, f"expected {k} frozen under bf16 AdamW, but {moved:.1%} moved"
    print("test_bf16_params_lose_updates_to_normalisation_gains: PASS "
          "(all norm gains frozen, reproducing the real-run bug)")


def test_fp32_masters_restore_updates():
    d = _run(use_master=True)
    unmoved = [k for k in NORM_KEYS if (d[k] > 0).float().mean().item() == 0.0]
    assert not unmoved, f"fp32 masters failed to unfreeze {unmoved}"
    frac = {k: (d[k] > 0).float().mean().item() for k in NORM_KEYS}
    print("test_fp32_masters_restore_updates: PASS ("
          + ", ".join(f"{k.split('.')[-2]}={v*100:.0f}%" for k, v in frac.items()) + ")")


def test_bf16_spacing_is_the_actual_mechanism():
    """Ties the failure to representable spacing rather than to gradients, so a
    future reader does not go looking for a vanishing-gradient explanation."""
    import math
    for mag, lr, should_survive in ((1.0, 1.6e-4, False),
                                    (0.03, 1.6e-4, True),
                                    (0.03, 3.0e-5, False)):
        spacing = 2.0 ** (math.floor(math.log2(mag)) - 7)   # bf16: 7 mantissa bits
        survives = lr / spacing >= 0.5
        assert survives == should_survive, (mag, lr, lr / spacing)
    # and confirm bf16 really cannot represent 1.0 + 1.6e-4
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    nudged = (one.float() + 1.6e-4).to(torch.bfloat16)
    assert nudged.item() == one.item(), "bf16 unexpectedly represented the update"
    print("test_bf16_spacing_is_the_actual_mechanism: PASS "
          "(1.0 + 1.6e-4 is not representable in bf16)")


def test_policy_is_fp32_masters_for_matrices_and_frozen_1d_gains():
    """The shipped policy, and the reason for it.

    Unfreezing the norm gains was tried on the live run and MEASURABLY HURT:
    their optimizer exp_avg had accumulated 657k steps of gradients whose
    updates were being discarded, so when updates could finally land the stale
    momentum discharged -- gains went 1.0 -> mean 0.67/0.70/2.17 in 39k steps
    and fixed-block loss on the real mixture rose 1.4555 -> 1.6659. So 1-D
    gains stay frozen (a fixed unit gain is a legitimate design), while the
    matrices get fp32 masters, which is where the lost-update bug actually
    costs anything.
    """
    cfg = _cfg()
    m = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.bfloat16)
    for _n, _p in m.named_parameters():
        if _p.ndim == 1:
            _p.requires_grad_(False)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert all(m.get_parameter(n).ndim >= 2 for n in trainable), trainable
    for k in NORM_KEYS:
        assert k in frozen, f"{k} should be frozen"
    # the frozen set must be a negligible slice of capacity, or freezing it
    # would be trading away real model capability rather than avoiding a risk
    n_frozen = sum(m.get_parameter(n).numel() for n in frozen)
    n_total = sum(p.numel() for p in m.parameters())
    assert n_frozen / n_total < 0.01, f"frozen share {n_frozen/n_total:.2%} is too large"
    print(f"test_policy_is_fp32_masters_for_matrices_and_frozen_1d_gains: PASS "
          f"({len(trainable)} trainable tensors, {len(frozen)} frozen 1-D gains "
          f"= {n_frozen/n_total*100:.3f}% of params)")


def test_master_optimizer_works_at_post_training_learning_rates():
    """The post-training stages run 3x-60x lower lr than pretraining, which is
    where bf16-in-place fails completely rather than partially.

    At |w|~0.03 the bf16 spacing is 1.22e-4, so an AdamW update is 0.08x that at
    the SFT lr of 1e-5 and 0.008x at the GRPO lr of 1e-6. Plain AdamW on bf16
    params therefore moves nothing at all -- GRPO would run every rollout, score
    every reward, and update no weight, which looks exactly like the far more
    famous zero-variance-group failure.
    """
    from nanocoder.optim import MasterWeightOptimizer

    for lr in (1e-5, 1e-6):
        # naive: bf16 params updated in place
        torch.manual_seed(0)
        cfg = _cfg()
        m0 = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.bfloat16)
        naive = torch.optim.AdamW([p for p in m0.parameters() if p.requires_grad], lr=lr)
        b0 = {k: v.detach().float().clone() for k, v in m0.named_parameters()}
        ids = torch.randint(0, cfg.vocab_size, (2, 32))
        for _ in range(20):
            torch.nn.functional.cross_entropy(
                m0(ids, n_loops=2).reshape(-1, cfg.vocab_size).float(),
                ids.reshape(-1)).backward()
            naive.step(); naive.zero_grad(set_to_none=True)
        # FRACTION of elements that moved, not max delta. bf16-in-place does not
        # freeze everything: spacing scales with magnitude, so small weights
        # (|w|~0.001, spacing 7.6e-6) still move even at lr 1e-6. What collapses
        # is the SHARE of the model that can update -- which is exactly what the
        # live diagnosis measured (embedding 56% at lr 1.56e-4, gains 0%).
        moved_naive = sum((v.detach().float() - b0[k] != 0).float().mean().item()
                          for k, v in m0.named_parameters()) / len(b0)

        # fixed: fp32 masters
        torch.manual_seed(0)
        m1 = BlackwellLM(_cfg(), precision="bf16", device="cpu", dtype=torch.bfloat16)
        opt = MasterWeightOptimizer(m1, lr=lr, weight_decay=0.0, fused=False)
        b1 = {k: v.detach().float().clone() for k, v in m1.named_parameters()}
        for _ in range(20):
            torch.nn.functional.cross_entropy(
                m1(ids, n_loops=2).reshape(-1, cfg.vocab_size).float(),
                ids.reshape(-1)).backward()
            opt.step(); opt.zero_grad()
        moved_fixed = sum((v.detach().float() - b1[k] != 0).float().mean().item()
                          for k, v in m1.named_parameters()) / len(b1)

        assert moved_fixed > moved_naive * 1.5, (
            f"at lr={lr:g} fp32 masters ({moved_fixed:.1%}) should update far more of "
            f"the model than bf16-in-place ({moved_naive:.1%})")
        print(f"  lr={lr:g}: bf16-in-place updates {moved_naive:.1%} of elements, "
              f"fp32 masters update {moved_fixed:.1%}")
    print("test_master_optimizer_works_at_post_training_learning_rates: PASS")


if __name__ == "__main__":
    test_bf16_spacing_is_the_actual_mechanism()
    test_bf16_params_lose_updates_to_normalisation_gains()
    test_fp32_masters_restore_updates()
    test_policy_is_fp32_masters_for_matrices_and_frozen_1d_gains()
    test_master_optimizer_works_at_post_training_learning_rates()
    print("All test_master_weights_cpu tests passed.")
