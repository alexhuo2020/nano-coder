"""Supervised fine-tuning data: instruction/chat corpora -> (ids, targets, mask).

Two properties this module exists to guarantee:

  * LOSS IS MASKED TO ASSISTANT CONTENT. Training on the user's turns teaches the
    model to write the questions, which at generation time shows up as it
    happily inventing a '### User' turn and answering itself.
  * RIGHT-PADDING IS SAFE HERE, and is used. Under causal attention a real token
    never attends to a later pad, and the loss mask zeroes the pads, so
    right-padding cannot leak. (Generation is the opposite case -- there a padded
    batch shifts RoPE positions per row, which is why `generate` refuses to pad
    and replicates a single prompt instead.)

Formats are normalised on the way in, because instruction datasets disagree
about field names and a converter that silently yields empty content produces a
run that trains on nothing but headers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

from blackwell_lm.chat import (ASSISTANT, DEFAULT_SYSTEM, SYSTEM, TOOL, USER,
                                tokenize_conversation)


@dataclass
class SFTSource:
    name: str
    hf_path: str
    weight: float
    fmt: str                      # "messages" | "instruction" | "prompt_completion"
    hf_config: str | None = None
    split: str = "train"


# Probed for unauthenticated streaming access; see MIN_LIVE_WEIGHT below for what
# happens when one of these turns out to be gated.
DEFAULT_SFT_MIXTURE = [
    SFTSource("opencoder-sft", "OpenCoder-LLM/opc-sft-stage2", 0.55, "instruction",
              hf_config="educational_instruct"),
    SFTSource("evol-codealpaca", "theblackcat102/evol-codealpaca-v1", 0.30, "instruction"),
    SFTSource("tulu3", "allenai/tulu-3-sft-mixture", 0.15, "messages"),
]

MIN_LIVE_WEIGHT = 0.50


def _to_messages(rec, fmt: str):
    """Normalise one record into a message list, or None if unusable."""
    if fmt == "messages":
        msgs = rec.get("messages") or rec.get("conversations")
        if not msgs:
            return None
        out = []
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            role = {"human": USER, "gpt": ASSISTANT, "assistant": ASSISTANT,
                    "user": USER, "system": SYSTEM}.get(role, role)
            if role not in (SYSTEM, USER, ASSISTANT) or not content:
                return None
            out.append({"role": role, "content": content})
        return out if any(m["role"] == ASSISTANT for m in out) else None

    if fmt == "instruction":
        q = rec.get("instruction") or rec.get("question") or rec.get("prompt")
        a = rec.get("output") or rec.get("response") or rec.get("completion")
        extra = rec.get("input") or ""
        if not q or not a:
            return None
        if extra:
            q = f"{q}\n\n{extra}"
        return [{"role": SYSTEM, "content": DEFAULT_SYSTEM},
                {"role": USER, "content": q},
                {"role": ASSISTANT, "content": a}]

    if fmt == "prompt_completion":
        q, a = rec.get("prompt"), rec.get("completion")
        if not q or not a:
            return None
        return [{"role": SYSTEM, "content": DEFAULT_SYSTEM},
                {"role": USER, "content": q},
                {"role": ASSISTANT, "content": a}]
    return None


def stream_sft(sources: List[SFTSource], seed: int = 0) -> Iterator[list]:
    """Yields message lists, sampling live sources by weight."""
    from datasets import load_dataset

    live, iters = [], {}
    for s in sources:
        try:
            ds = load_dataset(s.hf_path, s.hf_config, split=s.split, streaming=True)
            iters[s.name] = iter(ds)
            live.append(s)
        except Exception as e:
            print(f"[sft] source {s.name!r} unavailable ({type(e).__name__}); "
                  f"redistributing its {s.weight:.0%}", flush=True)
    if not live:
        raise RuntimeError("no SFT sources could be opened")
    got = sum(s.weight for s in live)
    print(f"[sft] live sources: {[s.name for s in live]} "
          f"({got:.0%} of intended weight)", flush=True)
    if got < MIN_LIVE_WEIGHT:
        raise RuntimeError(
            f"only {got:.0%} of the SFT mixture is reachable (< {MIN_LIVE_WEIGHT:.0%}). "
            "Fine-tuning on the remnant would quietly change what the model is being "
            "taught; authenticate to the Hub (HF_TOKEN) or fix the source list."
        )

    rng = random.Random(seed)
    names = [s.name for s in live]
    weights = [s.weight for s in live]
    by_name = {s.name: s for s in live}
    while True:
        s = by_name[rng.choices(names, weights=weights, k=1)[0]]
        try:
            rec = next(iters[s.name])
        except StopIteration:
            from datasets import load_dataset as _ld
            iters[s.name] = iter(_ld(s.hf_path, s.hf_config, split=s.split, streaming=True))
            continue
        except Exception:
            continue
        msgs = _to_messages(rec, s.fmt)
        if msgs:
            yield msgs


def sft_batches(msg_stream: Iterator[list], tok, eos_id: int, batch_size: int,
                max_len: int, device: str = "cuda", min_trainable: int = 8):
    """Collates conversations into right-padded (ids, targets, loss_mask).

    Conversations whose assistant content is truncated away by `max_len` are
    dropped: a sequence with an all-zero mask contributes nothing but still
    occupies a batch slot, and enough of them make the loss look artificially
    stable because it is averaging over fewer and fewer real tokens.
    """
    import torch

    pending = []
    n_dropped_trunc = 0
    for msgs in msg_stream:
        ids, mask, truncated = tokenize_conversation(tok, msgs, eos_id,
                                                     max_len=max_len + 1)
        if sum(mask[1:]) < min_trainable:
            continue
        # A TRUNCATED TOOL TRAJECTORY IS WORSE THAN NO TRAJECTORY. Truncation
        # keeps the head, so a retry demo (wrong fix -> failing tests ->
        # correct fix) that overruns max_len keeps the WRONG fix, still has
        # ample trainable tokens, and teaches the model to write a bad edit
        # and stop -- the exact behaviour these demos exist to remove. A
        # single long instruction answer is unaffected: head-truncating prose
        # is harmless, so only conversations containing a TOOL turn are
        # dropped.
        if truncated and any(m.get("role") == TOOL for m in msgs):
            n_dropped_trunc += 1
            if n_dropped_trunc in (1, 10, 100, 1000):
                print(f"[sft] dropped {n_dropped_trunc} tool trajectories that "
                      f"exceeded max_len ({max_len}); raise --max-len if this "
                      f"grows", flush=True)
            continue
        pending.append((ids, mask))
        if len(pending) < batch_size:
            continue
        L = max(len(i) for i, _ in pending)
        n = len(pending)
        a_ids = np.zeros((n, L), dtype=np.int64)
        a_msk = np.zeros((n, L), dtype=np.float32)
        for r, (i, m) in enumerate(pending):
            a_ids[r, :len(i)] = i
            a_msk[r, :len(m)] = m
        pending = []
        t_ids = torch.from_numpy(a_ids).to(device)
        t_msk = torch.from_numpy(a_msk).to(device)
        # targets are inputs shifted by one; the mask follows the TARGET
        yield t_ids[:, :-1].contiguous(), t_ids[:, 1:].contiguous(), t_msk[:, 1:].contiguous()


def masked_cross_entropy(logits, targets, mask):
    """Mean CE over masked positions only.

    Normalised by the number of MASKED tokens, not by the batch's token count:
    dividing by all tokens would make the loss depend on how much padding and
    prompt happened to be in the batch, so the same model would report different
    losses on differently-shaped batches.
    """
    import torch.nn.functional as F

    per = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1), reduction="none",
    ).view_as(targets)
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom
