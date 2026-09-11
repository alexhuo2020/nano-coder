"""Serve the 85M checkpoint behind an Anthropic /v1/messages endpoint.

This exists so a real client -- Claude Code included -- can drive OUR model with
no Anthropic model in the loop. Nothing here calls out to any API; the only
weights involved are the local checkpoint's.

THE CONTEXT PROBLEM IS THE POINT, NOT AN OBSTACLE TO HIDE. Our checkpoint has a
2,048-token context. A measurement against a recording server showed Claude
Code's first request is ~18,850 tokens (~15,100 of it tool schemas), so the
prompt does not fit and cannot be made to fit by tuning. Rather than silently
truncating and returning plausible-looking garbage -- which would make a
context failure look like a model failure -- this server REPORTS the overflow:
it returns an error when the prompt cannot fit, unless --truncate is passed,
and every request's token accounting is logged.

Truncation, when enabled, keeps the TAIL of the conversation and drops the
head, because the user's actual request is at the end. That ordering is a
deliberate choice: dropping the tail instead would discard the question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
sys.path.insert(0, "/home/ubuntu/bnano")

import torch

from blackwell_lm import chat
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.generate import generate
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.tokenizer import EOS, load_tokenizer

LOCK = threading.Lock()          # one GPU, one decode at a time
STATE: dict = {}
STATS = {"requests": 0, "overflow": 0, "served": 0}


def load(ckpt: str, tok_path: str, context: int | None = None):
    """`context` overrides the trained max_seq_len.

    This is safe here for an architectural reason, not by luck: RoPE is a
    NON-PERSISTENT buffer rebuilt from config, so no stored weight has a shape
    tied to the old length; and every attention application is a 1,024-token
    sliding window (window=1024, global_every_loop=0), so no relative offset
    beyond 1,024 ever occurs no matter how long the sequence is. Measured on
    held-out text: CE on tokens PAST the 2,048 training length is 1.26 at
    4,096 and 1.16 at 8,192 -- no degradation, where a broken extrapolation
    would show CE in the tens.
    """
    tok = load_tokenizer(tok_path)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_d = dict(ck["cfg"])
    if context:
        cfg_d["max_seq_len"] = context
    cfg = ModelConfig(**cfg_d)
    model = BlackwellLM(cfg, precision="bf16", device="cuda",
                        dtype=torch.bfloat16)
    load_stage_checkpoint(ckpt, model)
    model.eval()
    STATE.update(tok=tok, model=model, cfg=cfg,
                 eos=tok.token_to_id(EOS), ctx=cfg.max_seq_len)
    print(f"[shim] loaded {os.path.basename(ckpt)}  d_model={cfg.d_model} "
          f"n_loops={cfg.n_loops} context={cfg.max_seq_len}", flush=True)


def flatten_system(sysf) -> str:
    if isinstance(sysf, list):
        return "\n".join(b.get("text", "") for b in sysf
                         if isinstance(b, dict))
    return sysf or ""


def flatten_content(content) -> str:
    """Anthropic content blocks -> plain text. Tool results are rendered as
    text because this model has no native tool-call channel; its tool protocol
    is the JSON convention it was trained on."""
    if isinstance(content, str):
        return content
    parts = []
    for blk in content or []:
        if not isinstance(blk, dict):
            parts.append(str(blk))
            continue
        t = blk.get("type")
        if t == "text":
            parts.append(blk.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(blk.get("content")))
        elif t == "tool_use":
            parts.append(json.dumps({"name": blk.get("name"),
                                     "args": blk.get("input")}))
    return "\n".join(p for p in parts if p)


def to_chat(body) -> list:
    msgs = []
    sys_text = flatten_system(body.get("system"))
    tools = body.get("tools") or []
    if tools:
        # The client's schemas are summarised to NAMES ONLY. Sending 15k tokens
        # of JSON Schema to a 2k-context model is not a tradeoff, it is a
        # guaranteed overflow; names at least preserve which tools exist.
        names = ", ".join(t.get("name", "?") for t in tools)
        sys_text = (sys_text + "\n\nAvailable tools: " + names).strip()
    if sys_text:
        msgs.append({"role": chat.SYSTEM, "content": sys_text})
    for m in body.get("messages") or []:
        role = {"user": chat.USER, "assistant": chat.ASSISTANT,
                "system": chat.SYSTEM}.get(m.get("role"), chat.USER)
        msgs.append({"role": role, "content": flatten_content(m.get("content"))})
    return msgs


def run(body, truncate: bool, max_new: int):
    tok, model, cfg = STATE["tok"], STATE["model"], STATE["cfg"]
    ctx = STATE["ctx"]
    msgs = to_chat(body)
    ids = chat.tokenize_prompt(tok, msgs)   # a list[int], NOT a tensor
    n = len(ids)

    budget = ctx - max_new
    over = n - budget
    if over > 0:
        STATS["overflow"] += 1
        if not truncate:
            return None, {"n_prompt": n, "budget": budget, "over": over}
        # Keep the TAIL: the request sits at the end of the conversation, so
        # dropping the head loses context while dropping the tail would lose
        # the question itself.
        #
        # (This first read `ids[..., -budget:]`, tensor syntax on a list. It
        # raised TypeError inside the handler, which reached the client as a
        # bare ECONNRESET -- a truncation bug wearing the costume of a network
        # failure.)
        ids = ids[-budget:]
        n = len(ids)

    with LOCK:
        t0 = time.time()
        toks, valid, _ = generate(model, ids, max_new_tokens=max_new,
                                  eos_id=STATE["eos"], temperature=0.7,
                                  num_return_sequences=1,
                                  n_loops=cfg.n_loops)
        dt = time.time() - t0
    text = tok.decode([int(t) for t, v in
                       zip(toks[0].tolist(), valid[0].tolist()) if v])
    STATS["served"] += 1
    return text, {"n_prompt": n, "budget": budget, "over": max(0, over),
                  "seconds": round(dt, 2)}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    truncate = False
    max_new = 256

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._json(200, {"ok": True, "stats": STATS,
                         "context": STATE.get("ctx")})

    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(ln) or b"{}")
        except json.JSONDecodeError:
            body = {}
        STATS["requests"] += 1
        i = STATS["requests"]

        text, info = run(body, self.truncate, self.max_new)
        fits = "FITS" if not info["over"] else f"OVER by {info['over']:,}"
        print(f"[shim] req {i}: prompt {info['n_prompt']:,} tok "
              f"(budget {info['budget']:,}) -> {fits}"
              + (f", {info.get('seconds')}s" if text is not None else ""),
              flush=True)

        if text is None:
            # A real error, not a fabricated answer. 400 with an explicit
            # message so the failure is attributable to context, not quality.
            self._json(400, {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": (
                    f"prompt is {info['n_prompt']} tokens; this model's context "
                    f"is {STATE['ctx']} and {self.max_new} are reserved for the "
                    f"reply, leaving {info['budget']}. Over by {info['over']}. "
                    f"Pass --truncate to serve a head-truncated prompt.")}})
            return

        if body.get("stream"):
            self._sse(text, info)
        else:
            self._json(200, {
                "id": f"msg_{i}", "type": "message", "role": "assistant",
                "model": body.get("model") or "blackwell-nanogpt-85m",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": info["n_prompt"],
                          "output_tokens": len(text) // 4}})

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, text, info):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def ev(kind, obj):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(obj)}\n\n"
                             .encode())
            self.wfile.flush()

        ev("message_start", {"type": "message_start", "message": {
            "id": "msg_stream", "type": "message", "role": "assistant",
            "model": "blackwell-nanogpt-85m", "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": info["n_prompt"], "output_tokens": 0}}})
        ev("content_block_start", {"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "text",
                                                     "text": ""}})
        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta",
                                             "text": text}})
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": "end_turn"},
                             "usage": {"output_tokens": len(text) // 4}})
        ev("message_stop", {"type": "message_stop"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ubuntu/bnano/agent_repo_ab.pt")
    ap.add_argument("--tokenizer", default="/home/ubuntu/bnano/tokenizer.json")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--context", type=int, default=None,
                    help="override the trained max_seq_len; measured safe to "
                         "8192 on this architecture (see load())")
    ap.add_argument("--truncate", action="store_true",
                    help="serve an over-long prompt by dropping its HEAD "
                         "instead of returning an error")
    a = ap.parse_args()
    load(a.ckpt, a.tokenizer, a.context)
    H.truncate, H.max_new = a.truncate, a.max_new
    print(f"[shim] listening on 0.0.0.0:{a.port}  truncate={a.truncate}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
