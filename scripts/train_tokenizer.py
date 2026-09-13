
# Run from anywhere: put the repo root on sys.path so this works without
# the caller having set PYTHONPATH. Aliased imports keep it independent of
# whatever the module imports below.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
"""Train the code-aware tokenizer on a sample of the real pretraining mixture.

Trained on the SAME mixture the model will see (code-heavy), because a tokenizer
fitted to the wrong distribution silently wastes context on every subsequent
step -- and unlike most mistakes here, it is baked in before training starts.
"""
import argparse
import itertools
import time

from nanocoder.data import DEFAULT_MIXTURE, stream_mixture
from nanocoder.tokenizer import WHITESPACE_RUNS, train_tokenizer, token_efficiency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--docs", type=int, default=200_000, help="documents to fit on")
    p.add_argument("--out", default="tokenizer.json")
    a = p.parse_args()

    t0 = time.perf_counter()
    stream = stream_mixture(DEFAULT_MIXTURE)
    held_out = []

    def corpus():
        for i, (text, is_code) in enumerate(itertools.islice(stream, a.docs)):
            if i < 200:
                held_out.append(text)     # never trained on; used for the report below
            if i % 20_000 == 0 and i:
                print(f"  ...{i:,} docs ({time.perf_counter()-t0:.0f}s)", flush=True)
            yield text

    print(f"training {a.vocab_size}-token BPE on up to {a.docs:,} docs from the real mixture")
    tok = train_tokenizer(corpus(), vocab_size=a.vocab_size, out_path=a.out)

    present = sum(1 for w in WHITESPACE_RUNS if tok.token_to_id(w) is not None)
    print(f"\nsaved {a.out} | vocab {tok.get_vocab_size():,} "
          f"| whitespace-run tokens {present}/{len(WHITESPACE_RUNS)}")

    if held_out:
        sample = "\n".join(held_out)
        eff = token_efficiency(tok, sample)
        print(f"held-out efficiency: {eff:.2f} chars/token")
        try:
            from tokenizers import Tokenizer
            g = Tokenizer.from_pretrained("gpt2")
            ge = token_efficiency(g, sample)
            print(f"GPT-2 baseline     : {ge:.2f} chars/token  -> {eff/ge:.2f}x better")
        except Exception as e:
            print(f"(gpt2 comparison unavailable: {type(e).__name__})")


if __name__ == "__main__":
    main()
