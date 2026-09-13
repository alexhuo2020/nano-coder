"""Incremental generation and log-probability scoring.

Used by every post-training stage, so the correctness properties here are
load-bearing three times over. Two choices are worth stating up front:

  * NO PADDING. A group rollout generates G continuations of ONE prompt, so the
    prompts are identical and no left-padding is needed. Padded batches with
    per-row offsets are a classic source of silently-wrong RoPE positions and
    off-by-one masks; the API is shaped to avoid the situation instead of
    handling it.
  * ROLLOUTS ARE ON-POLICY BY DEFAULT (temperature 1.0, top_p 1.0). Truncated
    sampling makes the behaviour policy differ from the model, which biases the
    importance ratio that GRPO's update is built on. `generate` therefore
    refuses to hand back log-probs for a truncated distribution rather than
    letting a subtly wrong ratio through.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_cache(model, n_loops: int):
    """A fresh KV cache: one slot per BLOCK APPLICATION, not per unique block.

    A looped model applies the same weights n_loops times and each application
    attends over its own history, so the cache has n_layers*n_loops slots. Sizing
    this by n_layers works only at n_loops=1 and then IndexErrors -- or worse,
    silently reuses another application's keys.
    """
    return [None] * (model.cfg.n_layers * n_loops)


def _filter(logits, temperature: float, top_p: float):
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        keep = (torch.softmax(srt, dim=-1).cumsum(-1) - torch.softmax(srt, dim=-1)) < top_p
        keep[..., 0] = True                       # never drop the argmax
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, idx, keep)
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits


@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    eos_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    n_loops: int | None = None,
    device=None,
    return_logprobs: bool = False,
    greedy: bool = False,
):
    """Returns (tokens [B, L], valid_mask [B, L], logprobs [B, L] or None).

    `valid_mask` is 1 for real generated tokens INCLUDING the terminating EOS,
    and 0 for everything produced after a sequence finished. Scoring anything
    past EOS would train the model on its own padding.
    """
    if return_logprobs and (top_p < 1.0 or temperature != 1.0 or greedy):
        raise ValueError(
            "return_logprobs=True requires an untruncated, unit-temperature, "
            "sampled distribution (temperature=1.0, top_p=1.0, greedy=False). "
            "Log-probs taken from a modified distribution do not match the "
            "policy that GRPO's importance ratio assumes, and the resulting "
            "bias is invisible in the loss curve."
        )
    was_training = model.training
    model.eval()
    dev = device or next(model.parameters()).device
    loops = n_loops if n_loops is not None else model.cfg.n_loops
    B = num_return_sequences

    ids = torch.tensor(prompt_ids, dtype=torch.long, device=dev)[None].expand(B, -1).contiguous()
    cache = make_cache(model, loops)
    logits, cache = model(ids, cache=cache, n_loops=loops)
    nxt = logits[:, -1, :].float()

    toks, lps = [], []
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_new_tokens):
        if greedy:
            t = nxt.argmax(-1)
        else:
            probs = torch.softmax(_filter(nxt, temperature, top_p), dim=-1)
            t = torch.multinomial(probs, 1).squeeze(-1)
        if return_logprobs:
            # log-prob under the UNMODIFIED policy, which is what the ratio needs
            lps.append(F.log_softmax(nxt, dim=-1).gather(-1, t[:, None]).squeeze(-1))
        toks.append(torch.where(done, torch.zeros_like(t), t))
        if eos_id is not None:
            done = done | (t == eos_id)
            if bool(done.all()):
                break
        step_logits, cache = model(t[:, None], cache=cache, n_loops=loops)
        nxt = step_logits[:, -1, :].float()

    tokens = torch.stack(toks, dim=1) if toks else torch.zeros(B, 0, dtype=torch.long, device=dev)
    # valid = not yet finished BEFORE this position, so the EOS itself is valid
    if eos_id is not None and tokens.numel():
        finished_before = (tokens == eos_id).cumsum(1) - (tokens == eos_id).long()
        valid = finished_before == 0
    else:
        valid = torch.ones_like(tokens, dtype=torch.bool)
    logprobs = torch.stack(lps, dim=1) if (return_logprobs and lps) else None
    if was_training:
        model.train()
    return tokens, valid, logprobs


def completion_logprobs(model, prompt_len: int, sequences, n_loops: int | None = None):
    """Teacher-forced log-probs of the COMPLETION tokens under the current policy.

    `sequences` is [B, prompt_len + gen_len]. Returns [B, gen_len]: entry (b, j)
    is log p(sequences[b, prompt_len + j] | everything before it). Kept
    differentiable -- this is the term GRPO backprops through.
    """
    loops = n_loops if n_loops is not None else model.cfg.n_loops
    logits = model(sequences, n_loops=loops)
    # predict position i+1 from position i, so the completion's first token is
    # predicted by the last PROMPT position
    logits = logits[:, prompt_len - 1:-1, :]
    targets = sequences[:, prompt_len:]
    # fp32 for the log-softmax: in bf16 the k3 KL estimator went NEGATIVE in a
    # previous run in this lineage, which is impossible for a KL and came purely
    # from accumulated precision loss in this exact reduction.
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[..., None]).squeeze(-1)
