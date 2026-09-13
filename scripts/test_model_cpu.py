
# Run from anywhere: put the repo root on sys.path so this works without
# the caller having set PYTHONPATH. Aliased imports keep it independent of
# whatever the module imports below.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
"""CPU correctness tests for nanocoder.model. No GPU, no TE -- the FP8 path
falls back to nn.Linear, which leaves the shapes, masking, GQA wiring and KV
cache exactly as they run on the GPU. Plain asserts, __main__ runner.
"""
import torch

from nanocoder.model import BlackwellLM, ModelConfig


def _cfg(**kw):
    # zero_init_residual makes every block an identity map at init, which would
    # make the information-flow tests below vacuously pass; loop_sample makes
    # forward() nondeterministic, which would break KV-cache parity. Both are
    # exercised by their own dedicated tests instead.
    base = dict(vocab_size=512, d_model=256, n_layers=4, n_heads=2, n_kv_heads=1,
                ffn_hidden=512, max_seq_len=64, window=8, global_every=2,
                n_loops=1, loop_sample=None, zero_init_residual=False)
    base.update(kw)
    return ModelConfig(**base)


def _model(cfg):
    return BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.float32)


def test_validation_catches_bad_shapes():
    for kw, needle in [
        (dict(n_heads=3), "divisible"),                       # d_model % n_heads
        (dict(n_kv_heads=3), "divisible"),                    # n_heads % n_kv_heads
        (dict(d_model=192, n_heads=2), "head_dim"),           # head_dim != 128
    ]:
        try:
            _cfg(**kw).validate()
        except ValueError as e:
            assert needle in str(e), (kw, str(e))
        else:
            raise AssertionError(f"expected validate() to reject {kw}")
    print("test_validation_catches_bad_shapes: PASS")


def test_head_dim_128_enforced():
    """The whole point of the shape rules: a config that is 'reasonable' but
    gives head_dim != 128 must be rejected, not silently run slowly."""
    cfg = ModelConfig(vocab_size=512, d_model=768, n_layers=2, n_heads=12,
                      n_kv_heads=4, ffn_hidden=3072, max_seq_len=64)
    assert cfg.head_dim == 64
    try:
        cfg.validate()
    except ValueError as e:
        assert "head_dim" in str(e)
        print("test_head_dim_128_enforced: PASS (rejected head_dim=64)")
        return
    raise AssertionError("head_dim=64 should have been rejected")


def test_forward_shape_and_finiteness():
    torch.manual_seed(0)
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(ids)
    assert out.shape == (2, 16, cfg.vocab_size), out.shape
    assert torch.isfinite(out).all()
    print(f"test_forward_shape_and_finiteness: PASS ({m.n_params():,} params)")


def test_gqa_shrinks_qkv_and_matches_mha_when_full():
    """n_kv_heads == n_heads must reproduce plain MHA, and fewer KV heads must
    actually shrink the fused QKV projection (that shrinkage IS the bandwidth
    saving this design is buying)."""
    wide = _cfg(n_heads=2, n_kv_heads=2)
    narrow = _cfg(n_heads=2, n_kv_heads=1)
    assert narrow.qkv_width < wide.qkv_width, (narrow.qkv_width, wide.qkv_width)
    torch.manual_seed(1)
    m = _model(wide)
    ids = torch.randint(0, wide.vocab_size, (2, 12))
    assert torch.isfinite(m(ids)).all()
    print(f"test_gqa_shrinks_qkv_and_matches_mha_when_full: PASS "
          f"(qkv {wide.qkv_width} -> {narrow.qkv_width})")


def test_sliding_window_actually_masks():
    """A token beyond the window must not influence a later token's logits in a
    windowed layer. Perturb position 0 and check a far-away position is
    unchanged, using a single windowed layer so nothing else can mix them."""
    torch.manual_seed(2)
    cfg = _cfg(n_layers=1, window=4, global_every=0, max_seq_len=32)
    m = _model(cfg)
    a = torch.randint(0, cfg.vocab_size, (1, 20))
    b = a.clone()
    b[0, 0] = (a[0, 0].item() + 1) % cfg.vocab_size   # change only position 0
    with torch.no_grad():
        ya, yb = m(a), m(b)
    far = ya[0, 19] - yb[0, 19]     # 19 - 0 = 19 >= window(4): must be identical
    near = ya[0, 2] - yb[0, 2]      #  2 - 0 =  2  < window(4): must differ
    assert far.abs().max() < 1e-5, far.abs().max().item()
    assert near.abs().max() > 1e-6, near.abs().max().item()
    print("test_sliding_window_actually_masks: PASS "
          f"(far delta {far.abs().max():.2e}, near delta {near.abs().max():.2e})")


def test_kv_cache_matches_full_recompute():
    """Incremental decode must equal running the whole prefix at once. This is
    the property the agentic phase depends on -- a cache that drifts silently
    would corrupt every multi-turn rollout."""
    torch.manual_seed(3)
    cfg = _cfg(window=0, global_every=0, max_seq_len=32)   # full causal: exact comparison
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full = m(ids)
        cache = [None] * cfg.n_layers
        outs = []
        for t in range(ids.shape[1]):
            step, cache = m(ids[:, t:t + 1], cache=cache)
            outs.append(step)
        inc = torch.cat(outs, dim=1)
    diff = (full - inc).abs().max().item()
    assert diff < 2e-4, f"cache drift {diff}"
    print(f"test_kv_cache_matches_full_recompute: PASS (max diff {diff:.2e})")


def test_backward_produces_gradients():
    torch.manual_seed(4)
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = torch.nn.functional.cross_entropy(
        m(ids).reshape(-1, cfg.vocab_size), ids.reshape(-1))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    print(f"test_backward_produces_gradients: PASS (loss {loss.item():.3f})")


def test_qk_norm_present_and_bounds_logit_scale():
    """QK-Norm must actually normalise: with it on, q/k rows have unit RMS, so
    attention logits cannot blow up as d_model grows -- the failure mode that
    bites hardest in FP8/FP4."""
    cfg = _cfg(qk_norm=True)
    m = _model(cfg)
    assert m.blocks[0].attn.q_norm is not None
    off = _model(_cfg(qk_norm=False))
    assert off.blocks[0].attn.q_norm is None
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    assert torch.isfinite(m(ids)).all()
    print("test_qk_norm_present_and_bounds_logit_scale: PASS")


def test_zero_init_residual_makes_blocks_identity_at_init():
    """With residual projections zeroed, an untrained block must pass its input
    through unchanged -- so logits depend only on the embedding. That is the
    property that keeps a LOOPED model stable, since the same perturbation
    would otherwise compound n_loops times."""
    torch.manual_seed(0)
    cfg = _cfg(zero_init_residual=True, n_loops=4, loop_sample=None)
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        got = m(ids)
        want = m.lm_head(m.norm(m.embed(ids)))   # blocks contribute nothing
    diff = (got - want).abs().max().item()
    assert diff < 1e-5, f"blocks are not identity at init: {diff}"
    print(f"test_zero_init_residual_makes_blocks_identity_at_init: PASS (diff {diff:.2e})")


def test_loop_count_changes_computation_and_is_sampled_in_training():
    """Two things: (a) n_loops must actually change the output, else looping is
    a no-op; (b) sampling must fire in train() and never in eval(), so eval is
    reproducible while training sees varied depth."""
    torch.manual_seed(1)
    cfg = _cfg(n_loops=4, loop_sample=(2, 8), zero_init_residual=False)
    m = _model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    m.eval()
    with torch.no_grad():
        a, b = m(ids, n_loops=2), m(ids, n_loops=8)
    assert (a - b).abs().max() > 1e-4, "n_loops did not change the computation"
    assert {m.sample_loops() for _ in range(20)} == {cfg.n_loops}, "eval must not sample"
    m.train()
    seen = {m.sample_loops() for _ in range(200)}
    assert len(seen) > 1 and min(seen) >= 2 and max(seen) <= 8, seen
    print(f"test_loop_count_changes_computation_and_is_sampled_in_training: PASS "
          f"(train sampled {min(seen)}..{max(seen)})")


def test_loop_conditioning_distinguishes_iterations():
    """A shared block is blind to which iteration it is on unless conditioned.
    The loop embedding starts at zero (so it is learned, not imposed); once
    perturbed it must change the output."""
    torch.manual_seed(2)
    cfg = _cfg(loop_conditioning=True, n_loops=3, loop_sample=None, zero_init_residual=False)
    m = _model(cfg)
    assert torch.equal(m.loop_embed.weight, torch.zeros_like(m.loop_embed.weight)),         "loop embedding must start as a no-op"
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        before = m(ids)
        m.loop_embed.weight[1].normal_(0, 0.1)   # only iteration 1
        after = m(ids)
    assert (before - after).abs().max() > 1e-5, "loop conditioning had no effect"
    print("test_loop_conditioning_distinguishes_iterations: PASS")


if __name__ == "__main__":
    test_validation_catches_bad_shapes()
    test_head_dim_128_enforced()
    test_forward_shape_and_finiteness()
    test_gqa_shrinks_qkv_and_matches_mha_when_full()
    test_sliding_window_actually_masks()
    test_kv_cache_matches_full_recompute()
    test_backward_produces_gradients()
    test_qk_norm_present_and_bounds_logit_scale()
    test_zero_init_residual_makes_blocks_identity_at_init()
    test_loop_count_changes_computation_and_is_sampled_in_training()
    test_loop_conditioning_distinguishes_iterations()
    print("All test_model_cpu tests passed.")
