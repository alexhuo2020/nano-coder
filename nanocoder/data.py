"""Streaming data for pretraining: a weighted source mixture, fill-in-the-middle
applied to code, and packing into fixed-length blocks.

Two deliberate choices worth stating, because both are places this could quietly
be wrong:

  * The mixture is CODE-HEAVY (60% by default). A prior model in this lineage
    was trained on 20% code and produced fluent prose that was not valid code in
    any language; the diagnosis was code-token starvation, not architecture.
  * Documents are packed EOS-separated, and attention is NOT masked across
    document boundaries. This is what nanoGPT/GPT-2 do and it trains fine. The
    stricter alternative (a per-document BlockMask through FlexAttention) is a
    real improvement but adds a data-dependent mask to every step; it is left as
    a follow-up rather than shipped unvalidated.

Everything streams. Nothing here materialises a corpus in memory, because the
target corpora are tens of GB.
"""

from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass
class Source:
    name: str
    hf_path: str
    weight: float
    text_key: str = "text"
    hf_config: str | None = None
    split: str = "train"
    is_code: bool = False       # only code documents get the FIM transform


# 60/25/15 code/web/math. The code share is the whole point; see module docstring.
#
# NOTE ON SOURCE CHOICE: the obvious code corpora (the-stack-v2, the-stack-smol,
# starcoderdata, tiny-codes) are ALL gated on the Hub and fail without a token.
# Probed unauthenticated, these are the ones that actually stream:
DEFAULT_MIXTURE = [
    Source("codeparrot", "codeparrot/codeparrot-clean", 0.50, text_key="content", is_code=True),
    Source("rosetta", "christopher/rosetta-code", 0.10, text_key="code", is_code=True),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 0.25, hf_config="sample-10BT"),
    Source("finemath", "HuggingFaceTB/finemath", 0.15, hf_config="finemath-4plus"),
]

# If gated/broken sources drop the realised code share below this, refuse to run.
# Graceful degradation is the WRONG behaviour here: a previous model in this
# lineage trained on a mixture that quietly lost its code share and produced
# fluent prose that was not valid code in any language. Failing loudly costs
# minutes; failing silently costs the whole run.
MIN_CODE_WEIGHT = 0.40


def stream_mixture(sources: List[Source], seed: int = 0) -> Iterator[tuple[str, bool]]:
    """Yields (text, is_code), sampling sources by weight. A source that fails to
    load is dropped and its weight redistributed, so one dead dataset does not
    take the run down -- it is logged by the caller rather than raised."""
    from datasets import load_dataset

    live, iters = [], {}
    for s in sources:
        try:
            ds = load_dataset(s.hf_path, s.hf_config, split=s.split, streaming=True)
            iters[s.name] = iter(ds)
            live.append(s)
        except Exception as e:
            print(f"[data] source {s.name!r} unavailable ({type(e).__name__}); "
                  f"redistributing its {s.weight:.0%} weight", flush=True)
    if not live:
        raise RuntimeError("no data sources could be opened")

    total = sum(s.weight for s in live)
    code_share = sum(s.weight for s in live if s.is_code) / total
    print(f"[data] live sources: {[s.name for s in live]} | realised code share "
          f"{code_share:.0%}", flush=True)
    if code_share < MIN_CODE_WEIGHT:
        raise RuntimeError(
            f"realised code share is {code_share:.0%}, below MIN_CODE_WEIGHT="
            f"{MIN_CODE_WEIGHT:.0%}. Gated/failed sources: "
            f"{[s.name for s in sources if s.name not in {l.name for l in live}]}. "
            "Training on this mixture would repeat the known code-starvation failure. "
            "Authenticate to the Hub (HF_TOKEN) or add an accessible code source."
        )

    rng = random.Random(seed)
    names = [s.name for s in live]
    weights = [s.weight for s in live]
    by_name = {s.name: s for s in live}
    while True:
        name = rng.choices(names, weights=weights, k=1)[0]
        src = by_name[name]
        try:
            rec = next(iters[name])
        except StopIteration:
            iters[name] = iter(load_dataset(src.hf_path, src.hf_config,
                                            split=src.split, streaming=True))
            continue
        except Exception:
            continue
        text = rec.get(src.text_key)
        if isinstance(text, str) and text:
            yield text, src.is_code


def prefetch(it: Iterator, depth: int = 16) -> Iterator:
    """Run `it` on a daemon thread, buffering `depth` items ahead.

    Worth it because the producer is tokenization + HTTP streaming while the
    consumer is the GPU: without overlap they serialise, which measured as a 16%
    throughput loss versus feeding the same model pre-generated tokens. The
    Rust tokenizer releases the GIL, so a thread (not a process) is enough and
    avoids pickling a live HTTP stream.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def run():
        try:
            for item in it:
                q.put(item)
        except BaseException as e:      # surface producer errors on the consumer
            q.put(e)
        finally:
            q.put(DONE)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def packed_blocks(
    text_stream: Iterator[tuple[str, bool]],
    tok,
    seq_len: int,
    eos_id: int,
    fim_ids: tuple[int, int, int] | None = None,
    fim_rate: float = 0.5,
    seed: int = 0,
    tokenize_batch: int = 64,
) -> Iterator[np.ndarray]:
    """Tokenize, optionally FIM-transform code, and emit exactly seq_len+1 token
    blocks (the +1 gives input/target pairs without re-reading).

    Documents are tokenized in batches: `encode_batch` runs the Rust tokenizer
    across threads, which is markedly faster than a Python loop of `encode`.
    """
    from nanocoder.tokenizer import fim_transform

    rng = random.Random(seed)
    buf: List[int] = []
    pending_text: List[str] = []
    pending_code: List[bool] = []

    def flush():
        nonlocal buf, pending_text, pending_code
        if not pending_text:
            return
        for enc, is_code in zip(tok.encode_batch(pending_text), pending_code):
            ids = enc.ids
            if is_code and fim_ids is not None:
                ids = fim_transform(ids, *fim_ids, rate=fim_rate, rng=rng)
            buf.extend(ids)
            buf.append(eos_id)
        pending_text, pending_code = [], []

    for text, is_code in text_stream:
        pending_text.append(text)
        pending_code.append(is_code)
        if len(pending_text) < tokenize_batch:
            continue
        flush()
        while len(buf) >= seq_len + 1:
            yield np.asarray(buf[: seq_len + 1], dtype=np.int32)
            buf = buf[seq_len:]          # keep 1 token of overlap for the target shift


def batches(block_iter: Iterator[np.ndarray], batch_size: int, device: str = "cuda"):
    """Collates blocks into (input_ids, targets) on device."""
    import torch

    pending: List[np.ndarray] = []
    for blk in block_iter:
        pending.append(blk)
        if len(pending) == batch_size:
            arr = torch.from_numpy(np.stack(pending)).to(device, non_blocking=True).long()
            pending = []
            yield arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
