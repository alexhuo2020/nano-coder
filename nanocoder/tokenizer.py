"""A code-aware byte-level BPE tokenizer, plus the fill-in-the-middle transform
that makes a code model able to infill rather than only continue.

Three things a general-purpose tokenizer (GPT-2 BPE) gets wrong for code, all
addressed here:

  1. INDENTATION. GPT-2 frequently spends one token per space, so a 16-space
     indent costs 16 tokens. Explicit whitespace-run tokens are added to the
     vocabulary so common indents are single tokens. This is pure throughput:
     the same source file becomes fewer tokens, at zero hardware cost.
  2. INFILLING. Real code assistance is "complete the middle", not "continue
     the end". Fill-in-the-middle (Bavarian et al. 2022) is a data transform
     plus three sentinel tokens; `fim_transform` below implements the SPM/PSM
     split.
  3. UNSEEN IDENTIFIERS. Byte-level fallback means an arbitrary identifier or
     binary blob can always be encoded, never dropped to <unk>.

Vocabulary is deliberately small (32,768). The embedding/lm_head backward is
bandwidth-bound, and bandwidth is the weak axis of this hardware -- a large
vocabulary is one of the few places a small model can accidentally spend a
double-digit percentage of its step time.
"""

from __future__ import annotations

import random
from typing import Iterable, List

EOS = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_PAD = "<|fim_pad|>"
SPECIALS = [EOS, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD]

# Runs that dominate real source: 2/4/8/12/16/24/32 spaces and tab groups.
# Guaranteed present rather than left to chance in BPE merges.
WHITESPACE_RUNS = (
    [" " * n for n in (2, 3, 4, 6, 8, 12, 16, 20, 24, 32)]
    + ["\t" * n for n in (1, 2, 3, 4)]
    + ["\n" * n for n in (2, 3)]
)


def train_tokenizer(corpus: Iterable[str], vocab_size: int = 32768, out_path: str = "tokenizer.json"):
    """Trains a byte-level BPE. `corpus` is an iterable of text chunks (stream
    it; do not materialise a corpus in memory)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel with add_prefix_space=False: code is not natural language, and a
    # forced leading space corrupts indentation-sensitive text.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # byte fallback
        show_progress=True,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    # Add the indent tokens after training so they exist regardless of what the
    # merge search happened to find.
    tok.add_tokens(WHITESPACE_RUNS)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str = "tokenizer.json"):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def fim_transform(ids: List[int], prefix_id: int, middle_id: int, suffix_id: int,
                  rate: float = 0.5, spm_rate: float = 0.5,
                  rng: random.Random | None = None) -> List[int]:
    """With probability `rate`, rewrite a document into fill-in-the-middle form.

    Two orderings, both used in the literature; `spm_rate` picks between them:
      PSM: <pre> prefix <suf> suffix <mid> middle
      SPM: <suf> suffix <pre> prefix <mid> middle
    Training on both makes the model robust to how a caller frames the request.
    The middle always comes last, so ordinary next-token prediction teaches
    infilling with no change to the loss.
    """
    rng = rng or random
    if len(ids) < 8 or rng.random() >= rate:
        return ids
    a, b = sorted(rng.sample(range(1, len(ids) - 1), 2))
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]
    if not middle:
        return ids
    if rng.random() < spm_rate:
        return [suffix_id] + suffix + [prefix_id] + prefix + [middle_id] + middle
    return [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle


def token_efficiency(tok, text: str) -> float:
    """Characters per token -- the number that decides how much real source fits
    in a fixed context and how many tokens a fixed corpus costs. Higher is
    better; a code-aware tokenizer should clearly beat a general one on code."""
    n = len(tok.encode(text).ids)
    return len(text) / max(n, 1)
