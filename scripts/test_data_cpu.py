"""CPU tests for the data pipeline. Uses a tiny locally-trained tokenizer and a
synthetic document stream -- no network, no GPU. These check the properties that
would silently corrupt a multi-day run if wrong.
"""
import random

import numpy as np

from blackwell_lm.data import batches, packed_blocks, prefetch
from blackwell_lm.tokenizer import (EOS, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX,
                                    fim_transform, train_tokenizer)

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "class Thing:\n    def __init__(self):\n        self.x = 1\n",
    "The quick brown fox jumps over the lazy dog. " * 5,
] * 60


def _tok():
    return train_tokenizer(CORPUS, vocab_size=1024, out_path="/tmp/test_tok.json")


def test_blocks_are_exact_length_and_shifted():
    """Every block must be seq_len+1 so inputs/targets shift without re-reading.
    An off-by-one here silently trains the model to predict the current token."""
    tok = _tok()
    eos = tok.token_to_id(EOS)
    stream = ((t, False) for t in CORPUS * 4)
    blocks = [b for _, b in zip(range(6), packed_blocks(stream, tok, 32, eos))]
    assert len(blocks) == 6
    for b in blocks:
        assert b.shape == (33,), b.shape
        assert b.dtype == np.int32
    print("test_blocks_are_exact_length_and_shifted: PASS")


def test_documents_are_eos_separated():
    """Packing must insert EOS between documents, or the model learns to run one
    file straight into the next with no boundary signal."""
    tok = _tok()
    eos = tok.token_to_id(EOS)
    stream = ((t, False) for t in CORPUS * 4)
    joined = np.concatenate([b for _, b in zip(range(8), packed_blocks(stream, tok, 32, eos))])
    assert (joined == eos).sum() > 0, "no EOS found in packed output"
    print(f"test_documents_are_eos_separated: PASS ({(joined == eos).sum()} EOS in 8 blocks)")


def test_fim_only_applied_to_code_and_preserves_content():
    """FIM must rearrange, never drop. The middle must land last so ordinary
    next-token prediction teaches infilling."""
    tok = _tok()
    pid, mid, sid = (tok.token_to_id(t) for t in (FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX))
    ids = tok.encode(CORPUS[1]).ids
    out = fim_transform(list(ids), pid, mid, sid, rate=1.0, rng=random.Random(0))
    assert len(out) == len(ids) + 3, (len(out), len(ids))
    assert sorted(x for x in out if x not in (pid, mid, sid)) == sorted(ids), "FIM lost content"
    assert mid in out and out.index(mid) < len(out) - 1
    # rate=0 must be a no-op
    assert fim_transform(list(ids), pid, mid, sid, rate=0.0, rng=random.Random(0)) == ids
    print("test_fim_only_applied_to_code_and_preserves_content: PASS")


def test_batches_shape_and_target_shift():
    import torch
    tok = _tok()
    eos = tok.token_to_id(EOS)
    stream = ((t, False) for t in CORPUS * 8)
    blk = packed_blocks(stream, tok, 16, eos)
    inp, tgt = next(batches(blk, batch_size=3, device="cpu"))
    assert inp.shape == tgt.shape == (3, 16), (inp.shape, tgt.shape)
    assert torch.equal(inp[:, 1:], tgt[:, :-1]), "targets are not the inputs shifted by one"
    print("test_batches_shape_and_target_shift: PASS")


def test_prefetch_is_transparent_and_propagates_errors():
    """Prefetching must not change the data -- same items, same order -- and a
    producer exception must surface on the consumer rather than hanging."""
    src = list(range(50))
    assert list(prefetch(iter(src), depth=4)) == src

    def boom():
        yield 1
        raise ValueError("producer failed")
    try:
        list(prefetch(boom(), depth=2))
    except ValueError as e:
        assert "producer failed" in str(e)
    else:
        raise AssertionError("producer exception was swallowed")
    print("test_prefetch_is_transparent_and_propagates_errors: PASS")


if __name__ == "__main__":
    test_blocks_are_exact_length_and_shifted()
    test_documents_are_eos_separated()
    test_fim_only_applied_to_code_and_preserves_content()
    test_batches_shape_and_target_shift()
    test_prefetch_is_transparent_and_propagates_errors()
    print("All test_data_cpu tests passed.")
