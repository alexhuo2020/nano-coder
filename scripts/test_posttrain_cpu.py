
# Run from anywhere: put the repo root on sys.path so this works without
# the caller having set PYTHONPATH. Aliased imports keep it independent of
# whatever the module imports below.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
"""CPU tests for the post-training stack: chat format, generation, GRPO maths,
SFT masking and cross-precision checkpoint loading. No GPU, no TE, no network.

These target the failures that are SILENT -- a wrong prompt format, a mask off
by one, a KL that swamps the objective. Each of those produced a run that looked
healthy and learned nothing.
"""
import torch

from nanocoder import chat
from nanocoder.checkpoint import load_weights
from nanocoder.generate import completion_logprobs, generate, make_cache
from nanocoder.grpo import (GRPOConfig, group_advantages, grpo_objective,
                               k3_kl, sequence_ratio)
from nanocoder.model import BlackwellLM, ModelConfig
from nanocoder.sft import masked_cross_entropy
from nanocoder.tokenizer import EOS, train_tokenizer

CORPUS = ["def f(x):\n    return x + 1\n", "### User\nhi\n\n### Assistant\nhello\n\n"] * 80


def _tok():
    return train_tokenizer(CORPUS, vocab_size=512, out_path="/tmp/test_pt_tok.json")


def _cfg(**kw):
    base = dict(vocab_size=512, d_model=256, n_layers=2, n_heads=2, n_kv_heads=1,
                ffn_hidden=512, max_seq_len=128, window=8, global_every=0,
                n_loops=2, loop_sample=None, zero_init_residual=False)
    base.update(kw)
    return ModelConfig(**base)


def _model(cfg=None):
    return BlackwellLM(cfg or _cfg(), precision="bf16", device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------- chat format

def test_prompt_is_exact_prefix_of_trained_sequence():
    """THE invariant. If the RL rollout prompt is not a token-for-token prefix
    of what SFT trained on, the policy is asked to continue a format it never
    saw -- which in a previous run produced a permanent zero reward that looked
    like an RL problem for days."""
    tok = _tok()
    eos = tok.token_to_id(EOS)
    msgs = chat.user_turn("write a function", system="sys text")
    full = msgs + [{"role": chat.ASSISTANT, "content": "def f():\n    pass"}]

    prompt = chat.tokenize_prompt(tok, msgs)
    ids, mask, _ = chat.tokenize_conversation(tok, full, eos)
    assert ids[:len(prompt)] == prompt, "rollout prompt is not a prefix of the trained sequence"
    assert sum(mask[:len(prompt)]) == 0, "prompt tokens are marked trainable"
    assert sum(mask[len(prompt):]) == len(ids) - len(prompt), "assistant body not fully trained"
    print("test_prompt_is_exact_prefix_of_trained_sequence: PASS "
          f"(prompt {len(prompt)} tok, completion {len(ids)-len(prompt)} tok)")


def test_assistant_turn_is_eos_terminated_and_trained():
    tok = _tok()
    eos = tok.token_to_id(EOS)
    ids, mask, _ = chat.tokenize_conversation(
        tok, chat.user_turn("q") + [{"role": chat.ASSISTANT, "content": "a"}], eos)
    assert ids[-1] == eos, "assistant turn does not end in EOS"
    assert mask[-1] == 1, "EOS is not trained on, so the model never learns to stop"
    print("test_assistant_turn_is_eos_terminated_and_trained: PASS")


def test_headers_are_never_trainable():
    """The model must not learn to emit the User header -- that lets it
    fabricate its own turns and answer itself."""
    tok = _tok()
    eos = tok.token_to_id(EOS)
    msgs = [{"role": chat.USER, "content": "AAA"},
            {"role": chat.ASSISTANT, "content": "BBB"},
            {"role": chat.USER, "content": "CCC"},
            {"role": chat.ASSISTANT, "content": "DDD"}]
    ids, mask, _ = chat.tokenize_conversation(tok, msgs, eos)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "User" not in trained and "Assistant" not in trained, trained
    assert "BBB" in trained and "DDD" in trained, trained
    assert "AAA" not in trained and "CCC" not in trained, trained
    print("test_headers_are_never_trainable: PASS")


# ------------------------------------------------------------------ generation

def test_cache_slots_scale_with_loops():
    """A looped model needs one cache slot per APPLICATION. Sizing by n_layers
    silently reuses another application's keys (or IndexErrors)."""
    m = _model(_cfg(n_layers=2, n_loops=3))
    assert len(make_cache(m, 3)) == 6, len(make_cache(m, 3))
    print("test_cache_slots_scale_with_loops: PASS (2 blocks x 3 loops = 6 slots)")


def test_generate_stops_at_eos_and_masks_after():
    torch.manual_seed(0)
    m = _model()
    eos = 7
    toks, valid, _ = generate(m, [1, 2, 3], max_new_tokens=12, eos_id=eos,
                              num_return_sequences=4, n_loops=2)
    assert toks.shape[0] == 4 and valid.shape == toks.shape
    for r in range(toks.shape[0]):
        row, v = toks[r].tolist(), valid[r].tolist()
        if eos in row:
            i = row.index(eos)
            assert bool(v[i]), "the terminating EOS must be valid"
            assert not any(v[i + 1:]), "tokens after EOS are marked valid"
    print("test_generate_stops_at_eos_and_masks_after: PASS")


def test_generate_refuses_logprobs_from_truncated_sampling():
    """Log-probs from a top-p/temperature-modified distribution do not match the
    policy GRPO's ratio assumes, and the bias is invisible in the loss."""
    m = _model()
    for kw in (dict(top_p=0.9), dict(temperature=0.7), dict(greedy=True)):
        try:
            generate(m, [1, 2], max_new_tokens=2, return_logprobs=True, **kw)
        except ValueError as e:
            assert "importance ratio" in str(e) or "untruncated" in str(e)
        else:
            raise AssertionError(f"expected refusal for {kw}")
    print("test_generate_refuses_logprobs_from_truncated_sampling: PASS")


def test_completion_logprobs_alignment():
    """Off-by-one here trains the model to predict the CURRENT token. Checked
    against an explicit per-position computation."""
    torch.manual_seed(1)
    m = _model()
    seq = torch.randint(0, 512, (2, 11))
    plen = 5
    got = completion_logprobs(m, plen, seq, n_loops=2)
    assert got.shape == (2, seq.shape[1] - plen), got.shape
    with torch.no_grad():
        logits = m(seq, n_loops=2)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    for b in range(2):
        for j in range(seq.shape[1] - plen):
            pos = plen + j
            want = lsm[b, pos - 1, seq[b, pos]]
            assert torch.allclose(got[b, j], want, atol=1e-5), (b, j, got[b, j], want)
    print("test_completion_logprobs_alignment: PASS")


def test_incremental_decode_matches_full_under_a_window():
    """The decode path must apply the sliding window. Without it a windowed
    layer generates with context it never saw in training -- and the existing
    KV-cache test cannot catch this because it runs with window=0."""
    torch.manual_seed(2)
    cfg = _cfg(window=4, global_every=0, n_loops=1, n_layers=1, max_seq_len=32)
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full = m(ids, n_loops=1)
        cache = make_cache(m, 1)
        outs = []
        for t in range(ids.shape[1]):
            step, cache = m(ids[:, t:t + 1], cache=cache, n_loops=1)
            outs.append(step)
        inc = torch.cat(outs, dim=1)
    diff = (full - inc).abs().max().item()
    assert diff < 2e-4, f"windowed decode diverges from windowed full recompute: {diff}"
    print(f"test_incremental_decode_matches_full_under_a_window: PASS (max diff {diff:.2e})")


# ----------------------------------------------------------------- GRPO maths

def test_group_advantages_hand_computed():
    r = torch.tensor([0.0, 0.5, 1.0, 0.5])
    a = group_advantages(r)
    mean, std = r.mean(), r.std(unbiased=False)
    assert torch.allclose(a, (r - mean) / (std + 1e-8), atol=1e-6)
    assert abs(a.mean().item()) < 1e-6, "advantages must be mean-zero"
    print(f"test_group_advantages_hand_computed: PASS ({[round(x, 3) for x in a.tolist()]})")


def test_zero_variance_group_is_dropped_not_divided():
    """Every-sample-identical groups carry no preference information. An epsilon
    guard would turn their rounding noise into a training signal."""
    for r in ([0.0, 0.0, 0.0], [1.0, 1.0], [0.5, 0.5, 0.5, 0.5]):
        assert group_advantages(torch.tensor(r)) is None, r
    assert group_advantages(torch.tensor([0.0, 1.0])) is not None
    print("test_zero_variance_group_is_dropped_not_divided: PASS")


def test_k3_kl_is_nonnegative_and_zero_at_identity():
    torch.manual_seed(3)
    new = torch.randn(4, 6) - 2.0
    valid = torch.ones(4, 6)
    assert torch.allclose(k3_kl(new, new, valid), torch.zeros(4), atol=1e-6)
    ref = new + torch.randn(4, 6) * 0.5
    kl = k3_kl(new, ref, valid)
    assert (kl >= 0).all(), kl
    print(f"test_k3_kl_is_nonnegative_and_zero_at_identity: PASS (mean {kl.mean():.4f})")


def test_ratio_and_kl_are_both_length_normalised():
    """The bug this encodes: a length-normalised ratio with an unnormalised KL
    made the KL ~1748 against an objective of order 1, so the policy gradient
    was numerically absent. Doubling the length must leave both unchanged."""
    new, old = torch.zeros(1, 5), torch.full((1, 5), -0.1)
    v = torch.ones(1, 5)
    r_short, _ = sequence_ratio(new, old, v)
    kl_short = k3_kl(new, old, v)

    new2, old2 = torch.zeros(1, 10), torch.full((1, 10), -0.1)
    v2 = torch.ones(1, 10)
    r_long, _ = sequence_ratio(new2, old2, v2)
    kl_long = k3_kl(new2, old2, v2)
    assert torch.allclose(r_short, r_long, atol=1e-6), (r_short, r_long)
    assert torch.allclose(kl_short, kl_long, atol=1e-6), (kl_short, kl_long)
    print("test_ratio_and_kl_are_both_length_normalised: PASS "
          f"(ratio {r_short.item():.4f} at both lengths)")


def test_invalid_tokens_excluded_from_ratio():
    """Padding past EOS must not move the ratio."""
    new = torch.zeros(1, 6)
    old = torch.tensor([[-0.1, -0.1, -0.1, 5.0, 5.0, 5.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    r, n = sequence_ratio(new, old, valid)
    assert n.item() == 3
    assert torch.allclose(r, torch.exp(torch.tensor(0.1)), atol=1e-6), r
    print("test_invalid_tokens_excluded_from_ratio: PASS")


def test_clip_higher_is_asymmetric_and_engages():
    cfg = GRPOConfig(clip_low=0.2, clip_high=0.28, kl_beta=0.0)
    valid = torch.ones(2, 4)
    adv = torch.tensor([1.0, -1.0])
    new = torch.zeros(2, 4)
    old = torch.full((2, 4), -1.0)
    loss, m = grpo_objective(new, old, None, valid, adv, cfg)
    assert torch.isfinite(loss)
    assert m["frac_clipped"].item() > 0.0, m
    assert cfg.clip_high > cfg.clip_low, "Clip-Higher must be asymmetric"
    print("test_clip_higher_is_asymmetric_and_engages: PASS "
          f"(ratio {m['ratio_mean']:.3f}, clipped {m['frac_clipped']:.2f})")


def test_kl_penalty_moves_loss_in_the_right_direction():
    valid = torch.ones(2, 4)
    adv = torch.tensor([1.0, -1.0])
    new = torch.zeros(2, 4)
    old = torch.zeros(2, 4)
    ref = torch.full((2, 4), -0.5)
    l0, _ = grpo_objective(new, old, ref, valid, adv, GRPOConfig(kl_beta=0.0))
    l1, m1 = grpo_objective(new, old, ref, valid, adv, GRPOConfig(kl_beta=1.0))
    assert m1["kl"].item() > 0, m1
    assert l1.item() > l0.item(), (l0.item(), l1.item())
    print("test_kl_penalty_moves_loss_in_the_right_direction: PASS "
          f"(kl {m1['kl']:.4f}, loss {l0.item():.4f} -> {l1.item():.4f})")


# ------------------------------------------------------------------ SFT losses

def test_masked_cross_entropy_matches_hand_computation():
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3], [0, 4, 1]])
    mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    got = masked_cross_entropy(logits, targets, mask)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    want = -(lsm[0, 1, 2] + lsm[0, 2, 3] + lsm[1, 2, 1]) / 3.0
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    print(f"test_masked_cross_entropy_matches_hand_computation: PASS ({got.item():.4f})")


def test_masked_loss_is_invariant_to_padding():
    """Normalising by masked-token count (not batch tokens) is what makes the
    reported loss comparable across differently-shaped batches."""
    torch.manual_seed(5)
    logits = torch.randn(1, 3, 5)
    targets = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    base = masked_cross_entropy(logits, targets, mask)
    pad_logits = torch.cat([logits, torch.randn(1, 4, 5)], dim=1)
    pad_targets = torch.cat([targets, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(1, 4)], dim=1)
    assert torch.allclose(base, masked_cross_entropy(pad_logits, pad_targets, pad_mask),
                          atol=1e-6)
    print("test_masked_loss_is_invariant_to_padding: PASS")


# ------------------------------------------------------- checkpoint transition

def test_quant_state_dropped_but_missing_params_raise():
    """Loading an NVFP4 pretrain checkpoint into a BF16 RL model must tolerate
    TE's _extra_state and nothing else."""
    m = _model()
    sd = dict(m.state_dict())
    sd["blocks.0.attn.qkv.inner._extra_state"] = torch.zeros(3)
    rep = load_weights(m, sd)
    assert rep["dropped_quant_state"] == 1, rep
    assert not rep["missing"] and not rep["unexpected"], rep

    bad = dict(m.state_dict())
    removed = next(k for k in bad if k.endswith("mlp.down.inner.weight"))
    bad.pop(removed)
    try:
        load_weights(m, bad)
    except RuntimeError as e:
        assert "randomly initialised" in str(e) or "missing" in str(e)
    else:
        raise AssertionError("a missing real parameter must raise")
    print("test_quant_state_dropped_but_missing_params_raise: PASS")


if __name__ == "__main__":
    test_prompt_is_exact_prefix_of_trained_sequence()
    test_assistant_turn_is_eos_terminated_and_trained()
    test_headers_are_never_trainable()
    test_cache_slots_scale_with_loops()
    test_generate_stops_at_eos_and_masks_after()
    test_generate_refuses_logprobs_from_truncated_sampling()
    test_completion_logprobs_alignment()
    test_incremental_decode_matches_full_under_a_window()
    test_group_advantages_hand_computed()
    test_zero_variance_group_is_dropped_not_divided()
    test_k3_kl_is_nonnegative_and_zero_at_identity()
    test_ratio_and_kl_are_both_length_normalised()
    test_invalid_tokens_excluded_from_ratio()
    test_clip_higher_is_asymmetric_and_engages()
    test_kl_penalty_moves_loss_in_the_right_direction()
    test_masked_cross_entropy_matches_hand_computation()
    test_masked_loss_is_invariant_to_padding()
    test_quant_state_dropped_but_missing_params_raise()
    print("All test_posttrain_cpu tests passed.")
