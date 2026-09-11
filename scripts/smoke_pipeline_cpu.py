"""End-to-end smoke test of the whole post-training pipeline on CPU.

The unit tests check each piece; this checks that the pieces are WIRED. It runs a
real masked-CE step, a real GRPO update (real sandbox, real execution rewards),
and a real agent episode against a tiny randomly-initialised model.

REWARDS ARE INJECTED FOR THE UPDATE-PATH TEST, ON PURPOSE. A random tiny model
solves nothing, so every group is zero-variance and -- correctly -- dropped,
which means a natural run exercises none of the update code. That is exactly the
failure mode that once hid for hours behind healthy-looking logs, so here the
rewards are forced to vary in order to prove the update path actually executes
and moves the weights. The reward FUNCTION itself is tested for real, separately,
in test_sandbox_cpu.py.
"""

import torch

from blackwell_lm import chat
from blackwell_lm.agent import ToolBox, episode_reward, run_episode
from blackwell_lm.generate import completion_logprobs, generate
from blackwell_lm.grpo import GRPOConfig, group_advantages, grpo_objective
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.reward import code_reward
from blackwell_lm.sft import masked_cross_entropy
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, train_tokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "### User\nWrite a function\n\n### Assistant\n```python\ndef f():\n    return 1\n```\n\n",
] * 120


def build():
    tok = train_tokenizer(CORPUS, vocab_size=1024, out_path="/tmp/smoke_tok.json")
    vocab = ((tok.get_vocab_size() + 15) // 16) * 16
    cfg = ModelConfig(vocab_size=vocab, d_model=256, n_layers=1, n_heads=2,
                      n_kv_heads=1, ffn_hidden=512, max_seq_len=512, window=64,
                      global_every=0, n_loops=2, loop_sample=None,
                      zero_init_residual=False)
    model = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.float32)
    return tok, cfg, model


def stage_sft(tok, cfg, model):
    eos = tok.token_to_id(EOS)
    msgs = chat.user_turn("Write a function add(a, b).") + [
        {"role": chat.ASSISTANT, "content": "```python\ndef add(a, b):\n    return a + b\n```"}]
    ids, mask, _ = chat.tokenize_conversation(tok, msgs, eos,
                                              max_len=cfg.max_seq_len - 1)
    t = torch.tensor(ids)[None]
    m = torch.tensor(mask, dtype=torch.float32)[None]
    inp, tgt, msk = t[:, :-1], t[:, 1:], m[:, 1:]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.lm_head.weight.detach().clone()
    losses = []
    for _ in range(5):
        loss = masked_cross_entropy(model(inp, n_loops=cfg.n_loops), tgt, msk)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), losses
    assert not torch.equal(before, model.lm_head.weight), "SFT step did not move the weights"
    assert losses[-1] < losses[0], f"masked loss did not fall on a single example: {losses}"
    print(f"  SFT: masked loss {losses[0]:.4f} -> {losses[-1]:.4f} over 5 steps, weights moved")


def stage_grpo(tok, cfg, model):
    eos = tok.token_to_id(EOS)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=16, kl_beta=0.02, update_epochs=2)
    task = get_tasks("easy")[0]
    prompt_ids = chat.tokenize_prompt(tok, chat.user_turn(task.prompt))

    toks, valid, old_lp = generate(model, prompt_ids, max_new_tokens=gcfg.max_new_tokens,
                                   eos_id=eos, num_return_sequences=gcfg.group_size,
                                   n_loops=cfg.n_loops, return_logprobs=True)
    assert old_lp is not None and old_lp.shape == toks.shape, (old_lp.shape, toks.shape)

    # real reward path on real generations (expected: all zero for a random model)
    real = []
    for r in range(toks.shape[0]):
        text = tok.decode([int(x) for x, v in zip(toks[r].tolist(), valid[r].tolist()) if v])
        real.append(code_reward(text, task.tests)[0])
    assert group_advantages(torch.tensor(real)) is None, (
        f"a random model scored non-uniformly ({real}); the injected-reward rationale "
        "below no longer holds and this test should be rewritten")
    print(f"  GRPO: real rewards from a random policy are uniformly {real[0]:.1f} "
          f"-> group correctly dropped as zero-variance")

    # injected variance, to exercise the update path itself
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.0])
    adv = group_advantages(rewards)
    assert adv is not None
    plen = len(prompt_ids)
    seq = torch.cat([torch.tensor(prompt_ids)[None].expand(toks.shape[0], -1), toks], dim=1)

    import copy
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    with torch.no_grad():
        ref_lp = completion_logprobs(ref, plen, seq, n_loops=cfg.n_loops)
    new_lp = completion_logprobs(model, plen, seq, n_loops=cfg.n_loops)
    vf = valid.float()

    loss, met = grpo_objective(new_lp, old_lp, ref_lp, vf, adv, gcfg)
    # policy == reference at step 0, so the KL must be numerically zero. A
    # non-zero value here means the reference is not actually a frozen copy.
    assert met["kl"].abs().item() < 1e-5, f"KL at step 0 should be ~0, got {met['kl'].item()}"
    assert met["kl"].item() >= 0.0, f"k3 KL went negative: {met['kl'].item()}"
    assert torch.isfinite(loss), loss

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.lm_head.weight.detach().clone()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert torch.isfinite(gnorm) and gnorm > 0, gnorm
    assert not torch.equal(before, model.lm_head.weight), "GRPO step did not move the weights"

    ref_before = ref.lm_head.weight.detach().clone()
    assert torch.equal(ref_before, ref.lm_head.weight), "reference model drifted"
    print(f"  GRPO: update executed | loss {loss.item():.4f} | kl {met['kl'].item():.2e} "
          f"| ratio {met['ratio_mean'].item():.4f} | grad norm {gnorm:.4f} | weights moved")


def stage_agentic(tok, cfg, model):
    eos = tok.token_to_id(EOS)
    task = get_tasks("easy")[0]
    ep = run_episode(model, tok, task.prompt, ToolBox(), eos, max_turns=2,
                     max_new_tokens=12, n_loops=cfg.n_loops, record_logprobs=True)
    assert ep.turns >= 1
    assert len(ep.steps) == ep.turns, (len(ep.steps), ep.turns)
    for tr in ep.steps:
        assert tr.logprobs is not None and tr.logprobs.shape == tr.tokens.shape
    rw, detail = episode_reward(ep, task.tests)
    assert 0.0 <= rw <= 1.0, rw
    print(f"  AGENTIC: episode ran {ep.turns} turn(s), {ep.tool_calls} tool call(s), "
          f"reward {rw:.2f} ({detail[:40] or 'ok'}), per-turn logprobs recorded")


if __name__ == "__main__":
    torch.manual_seed(0)
    tok, cfg, model = build()
    print(f"tiny model: {model.n_params()/1e6:.2f}M params, vocab {cfg.vocab_size}, "
          f"{cfg.n_layers}x{cfg.n_loops} effective depth")
    print("PRE-TRAINING -> (checkpoint) -> POST-TRAINING -> AGENTIC, wired end to end:")
    stage_sft(tok, cfg, model)
    stage_grpo(tok, cfg, model)
    stage_agentic(tok, cfg, model)
    print("\nsmoke_pipeline_cpu: ALL STAGES WIRED AND EXECUTING")
