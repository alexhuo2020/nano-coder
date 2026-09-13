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
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
# The repo root, whether that is the GPU box's /home/ubuntu/bnano or a local
# checkout: resolve it from this file rather than hardcoding one machine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir("/home/ubuntu/bnano"):
    sys.path.insert(0, "/home/ubuntu/bnano")

import torch

from nanocoder import chat
from nanocoder.checkpoint import load_stage_checkpoint
from nanocoder.generate import generate
from nanocoder.model import BlackwellLM, ModelConfig
from nanocoder.tokenizer import EOS, load_tokenizer

import cli_adapter
import codex_adapter


def codex_adapter_cwd(body, fallback: str) -> str:
    """Codex states the cwd in its `instructions` / environment context.

    Getting this wrong is the same silent failure as on the Anthropic path:
    file operations run against a directory that does not exist on the
    client, and the error looks like the model's fault.
    """
    text = body.get("instructions") or ""
    for item in body.get("input") or []:
        c = item.get("content")
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    text += "\n" + (blk.get("text") or "")
    return cli_adapter.extract_cwd(text, fallback)

SEP = "\n---\n"                  # separator for the --dump-prompt rendering
DEVICE = ["cuda"]                # set from --device before the model loads
CHAT_PASSTHROUGH = [False]       # set from --chat-passthrough
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
    # CPU runs in float32, not bfloat16: bf16 matmuls on CPU fall back to a
    # slow emulated path, and the 85M model fits in RAM in fp32 regardless.
    # Transformer Engine is CUDA-only and its import is already optional in
    # model.py, so the low-precision layers degrade to nn.Linear here.
    dev = DEVICE[0]
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = BlackwellLM(cfg, precision="bf16" if dev == "cuda" else "bf16",
                        device=dev, dtype=dtype)
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


# A repo task names the thing it is about: a file, the tests, or the act of
# fixing. A conversational question does none of those. The bar is deliberately
# LOW -- any hit means agent mode -- because the agent path is the one that
# works, and mistaking a repo task for chat would break it, whereas mistaking
# chat for a repo task only reproduces today's behaviour.
_REPO_TASK = re.compile(
    r"\.py\b|\btests?\b|\brepo\b|\bfix\b|\bfailing\b|\bpytest\b|\bfunction\b|"
    r"\bimplement\b|\bdebug\b|\bassert\b|\brun_tests\b|\bsolution\b",
    re.IGNORECASE)


def looks_like_a_repo_task(body) -> bool:
    """True if this conversation is (or has become) a repository task.

    Any tool traffic already in the transcript settles it: the agent loop is
    under way and must keep its own system prompt, or the model would lose the
    convention mid-episode.
    """
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") in (
                        "tool_use", "tool_result"):
                    return True
    for m in body.get("messages") or []:
        if m.get("role") != "user":
            continue
        text = m.get("content")
        if not isinstance(text, str):
            text = " ".join(b.get("text", "") for b in (text or [])
                            if isinstance(b, dict))
        if _REPO_TASK.search(text or ""):
            return True
    return False


def to_chat(body, claude_code: bool = False) -> list:
    """Build the model's message list from an Anthropic request.

    In claude_code mode the client's system prompt and 27 tool schemas are
    REPLACED by the model's own trained system prompt (render_system_prompt(),
    ~500 tokens, four tools). That is the whole point: the measured failure was
    out-of-distribution prompting, not capability, so the fix is to hand the
    model the distribution it was trained on and translate at the boundary.
    Only the user's actual request and the tool transcript carry through.
    """
    msgs = []
    sys_text = flatten_system(body.get("system"))
    tools = body.get("tools") or []

    if claude_code:
        from nanocoder.mcp_tools import render_system_prompt
        if CHAT_PASSTHROUGH[0] and not looks_like_a_repo_task(body):
            # A plain question gets the CHAT system prompt the instruction SFT
            # used, not the agent one. Without this the agent prompt is
            # injected on every request -- its first line is "You are a coding
            # agent working in a repository" and its first example is a
            # read_file call -- so "who are you" is answered by reading
            # solution.py. The model is not confused; it is doing exactly what
            # the prompt and its training say to do.
            msgs.append({"role": chat.SYSTEM, "content": chat.DEFAULT_SYSTEM})
        else:
            msgs.append({"role": chat.SYSTEM,
                         "content": render_system_prompt()})
    else:
        if tools:
            # Outside claude_code mode the schemas are summarised to NAMES
            # ONLY. Sending 15k tokens of JSON Schema to this model is not a
            # tradeoff, it is a guaranteed overflow; names at least preserve
            # which tools exist.
            names = ", ".join(t.get("name", "?") for t in tools)
            sys_text = (sys_text + "\n\nAvailable tools: " + names).strip()
        if sys_text:
            msgs.append({"role": chat.SYSTEM, "content": sys_text})

    for m in body.get("messages") or []:
        raw_role = m.get("role")
        role = {"user": chat.USER, "assistant": chat.ASSISTANT,
                "system": chat.SYSTEM}.get(raw_role, chat.USER)
        if claude_code:
            # DROP system-role MESSAGES. Claude Code sends its tool/agent
            # scaffolding as a `role: "system"` entry in `messages`, not in the
            # `system` field -- measured at 1,862 tokens of "Available agent
            # types ...". Substituting body["system"] alone therefore left the
            # prompt at 2,102 tokens of mostly client bookkeeping, and the
            # model answered with unrelated prose. It is not user content and
            # the model has no use for it.
            if raw_role == "system":
                continue
            content = flatten_content_cc(m.get("content"), role)
        else:
            content = flatten_content(m.get("content"))
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def flatten_content_cc(content, role):
    """Like flatten_content, but renders tool traffic in the MODEL's format.

    An assistant tool_use block becomes the fenced ```tool {...}``` payload the
    model was trained to emit, and a tool_result becomes a plain TOOL turn. A
    transcript that looked like Claude Code's wire format would be as
    out-of-distribution as the system prompt was.
    """
    if isinstance(content, str):
        return content
    parts = []
    for blk in content or []:
        if not isinstance(blk, dict):
            parts.append(str(blk))
            continue
        t = blk.get("type")
        if t == "text":
            txt = blk.get("text", "")
            # Drop the CLI's injected <system-reminder> scaffolding: it is
            # client bookkeeping, not the user's request, and it is a large
            # fraction of the prompt.
            txt = re.sub(r"<system-reminder>.*?</system-reminder>", "", txt,
                         flags=re.DOTALL).strip()
            if txt:
                parts.append(txt)
        elif t == "tool_use":
            payload = cli_adapter.cli_call_to_model(blk.get("name"),
                                                    blk.get("input"))
            parts.append("```tool\n" + json.dumps(payload) + "\n```")
        elif t == "tool_result":
            raw = flatten_content(blk.get("content"))
            # Tool RESULTS must match training too, not just tool calls.
            # pytest's real output is nothing like the "3/3 tests passed" the
            # model's own run_tests returns; given the raw form it looped
            # read -> test -> read -> test and never wrote a fix (0/10 CLI
            # trials). Line numbering gets stripped for the same reason.
            norm = cli_adapter.normalise_test_output(raw)
            parts.append(norm if norm is not None
                         else cli_adapter.strip_line_numbers(raw))
    return "\n".join(p for p in parts if p)


def run(body, truncate: bool, max_new: int, claude_code: bool = False):
    tok, model, cfg = STATE["tok"], STATE["model"], STATE["cfg"]
    ctx = STATE["ctx"]
    msgs = to_chat(body, claude_code=claude_code)
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
                                  eos_id=STATE["eos"], temperature=H.temperature,
                                  num_return_sequences=1,
                                  n_loops=cfg.n_loops)
        dt = time.time() - t0
    text = tok.decode([int(t) for t, v in
                       zip(toks[0].tolist(), valid[0].tolist()) if v])
    STATS["served"] += 1
    return text, {"n_prompt": n, "budget": budget, "over": max(0, over),
                  "seconds": round(dt, 2),
                  "prompt_text": SEP.join(
                      f"[{m['role']}] {m['content']}" for m in msgs)}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    truncate = False
    max_new = 256
    claude_code = False
    cwd = os.getcwd()
    dump_prompt = False
    # 0.2, not the 0.7 that used to be hardcoded here and not the 1.0 the RL
    # rollouts use. MEASURED on 120 episodes/tier, same checkpoint:
    #   T=1.0  mutate 52.5%  stub  7.5%
    #   T=0.5  mutate 74.2%  stub 25.0%
    #   T=0.2  mutate 76.7%  stub 31.7%
    # T=1.0 is correct for RL EXPLORATION and badly wrong for SOLVING; every
    # earlier figure in this project, including a reported 0/240 "capability
    # ceiling", was taken at 1.0.
    temperature = 0.2

    def log_message(self, *a):
        pass

    def handle_one_request(self):
        """Swallow client-side disconnects.

        Claude Code opens connections it then closes without sending (probes,
        keep-alives), and http.server prints a full ConnectionResetError
        traceback for each one. That looks exactly like a server crash to
        anyone reading the console, and it is not: the request that matters is
        logged separately by do_POST.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

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

        # Codex CLI speaks the OpenAI Responses API, not Anthropic's. Route on
        # the path rather than on a flag so ONE server (and one loaded model,
        # on one GPU) can serve both clients.
        if self.path.rstrip("/").endswith("/responses"):
            self._codex(body, i)
            return

        text, info = run(body, self.truncate, self.max_new, self.claude_code)
        if self.dump_prompt:
            print("[shim] --- rendered prompt ---" + SEP
                  + (info.get("prompt_text") or "")[:2500]
                  + SEP + "[shim] --- end prompt ---", flush=True)
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

        if self.claude_code:
            cwd = cli_adapter.extract_cwd(flatten_system(body.get("system")),
                                          self.cwd)
            resp, cli_tool, model_tool = cli_adapter.build_response(
                text, cwd, body.get("model") or "nano-coder-85m",
                f"msg_{i}", info["n_prompt"])
            if model_tool:
                print(f"[shim]   model called {model_tool!r} -> "
                      f"{cli_tool or 'UNMAPPABLE (returned as text)'}",
                      flush=True)
            else:
                print(f"[shim]   no tool call; replied with "
                      f"{len(text or '')} chars of text", flush=True)
            if body.get("stream"):
                self._sse_blocks(resp)
            else:
                self._json(200, resp)
            return

        if body.get("stream"):
            self._sse(text, info)
        else:
            self._json(200, {
                "id": f"msg_{i}", "type": "message", "role": "assistant",
                "model": body.get("model") or "nano-coder-85m",
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

    def _codex(self, body, i):
        """Serve one OpenAI Responses-API turn for Codex CLI."""
        from nanocoder.mcp_tools import render_system_prompt

        turns = codex_adapter.input_to_turns(body)
        msgs = [{"role": chat.SYSTEM, "content": render_system_prompt()}]
        for role, txt in turns:
            msgs.append({"role": {"user": chat.USER,
                                  "assistant": chat.ASSISTANT,
                                  "tool": chat.TOOL}[role], "content": txt})

        tok, model, cfg = STATE["tok"], STATE["model"], STATE["cfg"]
        ids = chat.tokenize_prompt(tok, msgs)
        budget = STATE["ctx"] - self.max_new
        n = len(ids)
        if n > budget:
            STATS["overflow"] += 1
            ids = ids[-budget:]
            n = budget
        with LOCK:
            t0 = time.time()
            toks, valid, _ = generate(model, ids, max_new_tokens=self.max_new,
                                      eos_id=STATE["eos"], temperature=H.temperature,
                                      num_return_sequences=1,
                                      n_loops=cfg.n_loops)
            dt = time.time() - t0
        text = tok.decode([int(t) for t, v in
                           zip(toks[0].tolist(), valid[0].tolist()) if v])
        STATS["served"] += 1

        cwd = codex_adapter_cwd(body, self.cwd)
        events, codex_tool, model_tool = codex_adapter.build_events(
            text, cwd, body.get("model") or "nano-coder-85m",
            cli_adapter.split_model_output)
        print(f"[shim] codex req {i}: prompt {n:,} tok "
              f"(budget {budget:,}) -> {'FITS' if n < budget else 'TRUNCATED'}"
              f", {dt:.2f}s", flush=True)
        if model_tool:
            print(f"[shim]   model called {model_tool!r} -> "
                  f"{codex_tool or 'UNMAPPABLE (returned as text)'}", flush=True)
        else:
            print(f"[shim]   no tool call; replied with {len(text or '')} "
                  f"chars of text", flush=True)
        if self.dump_prompt:
            print("[shim] --- rendered prompt ---" + SEP
                  + SEP.join(f"[{m['role']}] {m['content']}" for m in msgs)[:2500]
                  + SEP + "[shim] --- end prompt ---", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for kind, obj in events:
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(obj)}\n\n"
                             .encode())
            self.wfile.flush()

    def _sse_blocks(self, resp):
        """Stream a prebuilt response, including tool_use blocks.

        tool_use arguments go out as input_json_delta on a content block whose
        `input` starts empty -- sending the populated object in
        content_block_start is accepted by some clients and ignored by others,
        which shows up as a tool call with no arguments.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def ev(kind, obj):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(obj)}\n\n"
                             .encode())
            self.wfile.flush()

        head = dict(resp)
        head["content"] = []
        head["stop_reason"] = None
        ev("message_start", {"type": "message_start", "message": head})
        for idx, blk in enumerate(resp["content"]):
            if blk["type"] == "text":
                ev("content_block_start",
                   {"type": "content_block_start", "index": idx,
                    "content_block": {"type": "text", "text": ""}})
                ev("content_block_delta",
                   {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": blk["text"]}})
            else:
                ev("content_block_start",
                   {"type": "content_block_start", "index": idx,
                    "content_block": {"type": "tool_use", "id": blk["id"],
                                      "name": blk["name"], "input": {}}})
                ev("content_block_delta",
                   {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(blk["input"])}})
            ev("content_block_stop",
               {"type": "content_block_stop", "index": idx})
        ev("message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": resp["stop_reason"]},
                             "usage": resp["usage"]})
        ev("message_stop", {"type": "message_stop"})

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
            "model": "nano-coder-85m", "content": [],
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
    ap.add_argument("--claude-code", action="store_true",
                    help="translate to/from Claude Code's protocol: substitute "
                         "the model's OWN trained system prompt for the "
                         "client's, and emit real tool_use blocks so the CLI "
                         "actually executes the model's calls")
    ap.add_argument("--cwd", default=os.getcwd(),
                    help="fallback working directory for resolving the "
                         "relative paths the model emits")
    ap.add_argument("--chat-passthrough", action="store_true",
                    help="answer non-repo questions as CHAT instead of forcing "
                         "the agent prompt. Without it the agent system prompt "
                         "is injected on every request, so 'who are you' is "
                         "answered by reading solution.py -- the model is "
                         "following the prompt it was given.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu runs the 85M model in fp32 with no GPU and no "
                         "Transformer Engine (its import is already optional)")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="sampling temperature (default 0.2, measured best; "
                         "see the table on H.temperature)")
    ap.add_argument("--dump-prompt", action="store_true",
                    help="log the prompt actually rendered for the model; the "
                         "fastest way to see that a client's scaffolding is "
                         "still leaking in")
    ap.add_argument("--truncate", action="store_true",
                    help="serve an over-long prompt by dropping its HEAD "
                         "instead of returning an error")
    a = ap.parse_args()
    DEVICE[0] = a.device
    CHAT_PASSTHROUGH[0] = a.chat_passthrough
    load(a.ckpt, a.tokenizer, a.context)
    H.truncate, H.max_new = a.truncate, a.max_new
    H.claude_code, H.cwd = a.claude_code, a.cwd
    H.dump_prompt = a.dump_prompt
    H.temperature = a.temperature
    if a.claude_code:
        print(f"[shim] claude-code mode: model's own system prompt, "
              f"tool_use translation, cwd fallback {a.cwd}", flush=True)
    print(f"[shim] listening on 0.0.0.0:{a.port}  truncate={a.truncate}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
