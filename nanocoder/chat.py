"""Conversation format for post-training: one source of truth, shared by SFT,
GRPO rollouts and agentic episodes.

WHY THIS IS ITS OWN MODULE. In a previous model in this lineage, SFT rendered
prompts with role markers while the RL rollout path called
`tokenizer.encode(prompt_text)` on the bare question. The policy was therefore
asked, at RL time, to continue text in a format it had never been trained on.
Nothing crashed, rewards were simply always zero, and the run looked like an
RL-tuning problem for days. Every stage here imports the SAME functions so that
divergence is impossible rather than merely unlikely.

NO NEW SPECIAL TOKENS. The tokenizer is already trained and published (the
pretraining run is consuming it right now); adding chat tokens would change the
vocabulary and silently invalidate the checkpoint. So roles are plain text that
the existing BPE handles, and EOS -- which the model has seen as a document
separator since pretraining step 1 -- terminates assistant turns.

TOKENIZED PIECEWISE, DELIBERATELY. BPE applied to a concatenated string does
not generally equal BPE applied to its pieces, so a mask built piecewise cannot
be laid over a whole-string tokenization. Rather than fight that, piecewise IS
the canonical encoding here: `tokenize_conversation` (training) and
`tokenize_prompt` (inference) share one helper, so training and rollout
sequences agree by construction.
"""

from __future__ import annotations

SYSTEM, USER, ASSISTANT, TOOL = "system", "user", "assistant", "tool"

HEADER = {
    SYSTEM: "### System\n",
    USER: "### User\n",
    ASSISTANT: "### Assistant\n",
    TOOL: "### Tool\n",
}
TURN_END = "\n\n"

DEFAULT_SYSTEM = (
    "You are a coding assistant. Answer with correct, runnable Python code in a "
    "```python fenced block."
)


def _pieces(messages, add_generation_prompt: bool):
    """Flattens a conversation into (text, is_assistant_content) pieces.

    `is_assistant_content` is what drives loss masking: the model is trained to
    produce assistant bodies, never the headers that prompt them (learning to
    emit '### Assistant' would let it fabricate its own turns).
    """
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        out.append((HEADER[role], False))
        if role == ASSISTANT:
            out.append((content, True))
            out.append((TURN_END, True))      # train the turn boundary too
        else:
            out.append((content + TURN_END, False))
    if add_generation_prompt:
        out.append((HEADER[ASSISTANT], False))
    return out


def render(messages, add_generation_prompt: bool = False) -> str:
    """Human-readable rendering. For inspection and logging -- NOT the thing
    that gets tokenized (see the module docstring on piecewise encoding)."""
    return "".join(t for t, _ in _pieces(messages, add_generation_prompt))


def _encode_pieces(tok, pieces, eos_id: int):
    ids, mask = [], []
    for text, trainable in pieces:
        piece = tok.encode(text).ids
        ids.extend(piece)
        mask.extend([1 if trainable else 0] * len(piece))
    return ids, mask


def tokenize_conversation(tok, messages, eos_id: int, max_len: int | None = None):
    """Returns (ids, loss_mask) with 1s only on assistant content.

    An EOS is appended after each assistant turn and IS trained on, because a
    model that never learns to stop generates until it hits the token limit --
    which at RL time reads as a wrong answer for a reason unrelated to its
    reasoning.
    """
    ids, mask = [], []
    for m in messages:
        p = _pieces([m], add_generation_prompt=False)
        i, k = _encode_pieces(tok, p, eos_id)
        ids.extend(i)
        mask.extend(k)
        if m["role"] == ASSISTANT:
            ids.append(eos_id)
            mask.append(1)
    truncated = max_len is not None and len(ids) > max_len
    if truncated:
        ids, mask = ids[:max_len], mask[:max_len]
    # `truncated` is reported because head-truncation is SAFE for a single
    # long answer and UNSAFE for a multi-turn tool trajectory: the tail is
    # where the outcome lives. A retry demo (wrong fix -> failing tests ->
    # correct fix) cut short keeps the wrong fix, still has plenty of
    # trainable tokens, and teaches the model to write a bad edit and stop.
    return ids, mask, truncated


def tokenize_prompt(tok, messages) -> list[int]:
    """Prompt ids ending in the assistant header, ready for generation.

    Uses the same piecewise path as training, which is the whole point: this is
    the function every rollout must call instead of encoding a bare question.
    """
    ids, _ = _encode_pieces(tok, _pieces(messages, add_generation_prompt=True), 0)
    return ids


def user_turn(question: str, system: str | None = DEFAULT_SYSTEM):
    msgs = [] if system is None else [{"role": SYSTEM, "content": system}]
    msgs.append({"role": USER, "content": question})
    return msgs
