"""A fake Anthropic /v1/messages endpoint that records what Claude Code sends.

WHY THIS EXISTS. Before serving an 85M model to a coding CLI, the question that
decides the whole exercise is how big a request the CLI actually makes. Our
checkpoint has a 2,048-token context; if the system prompt plus tool schemas
exceed that, the model is truncated out of its own instructions and nothing
downstream matters. That is a measurement, not a guess, and it needs no GPU --
only something that speaks the wire protocol and writes down what arrives.

It answers every request with a fixed, valid reply, so the CLI proceeds far
enough to reveal its real prompt. Run it, then:

    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 ANTHROPIC_AUTH_TOKEN=dummy \
      claude -p "write hello world"
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured")
os.makedirs(OUT, exist_ok=True)
N = [0]


def approx_tokens(s: str) -> int:
    """~4 chars/token. Deliberately crude: we are asking whether the prompt is
    2k or 20k, and no tokenizer subtlety changes that answer."""
    return max(1, len(s) // 4)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # our own logging below

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        self._json(200, {"ok": True})

    def do_POST(self):
        raw = self._read()
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}

        N[0] += 1
        path = os.path.join(OUT, f"req_{N[0]:03d}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"path": self.path, "headers": dict(self.headers),
                       "body": body}, fh, indent=2)

        # --- what did it actually send? ---
        sys_field = body.get("system")
        if isinstance(sys_field, list):
            sys_text = "".join(b.get("text", "") for b in sys_field
                               if isinstance(b, dict))
        else:
            sys_text = sys_field or ""
        tools = body.get("tools") or []
        tools_text = json.dumps(tools)
        msgs = body.get("messages") or []
        msgs_text = json.dumps(msgs)

        print(f"\n=== request {N[0]}  {self.path}", flush=True)
        print(f"  model requested : {body.get('model')}", flush=True)
        print(f"  stream          : {body.get('stream')}", flush=True)
        print(f"  max_tokens      : {body.get('max_tokens')}", flush=True)
        print(f"  system prompt   : {len(sys_text):>9,} chars  "
              f"~{approx_tokens(sys_text):>7,} tokens", flush=True)
        print(f"  tools           : {len(tools):>9} defs   "
              f"~{approx_tokens(tools_text):>7,} tokens", flush=True)
        if tools:
            print("    " + ", ".join(t.get("name", "?") for t in tools)[:300],
                  flush=True)
        print(f"  messages        : {len(msgs):>9} msgs   "
              f"~{approx_tokens(msgs_text):>7,} tokens", flush=True)
        total = approx_tokens(sys_text) + approx_tokens(tools_text) + \
            approx_tokens(msgs_text)
        print(f"  TOTAL PROMPT    : ~{total:,} tokens   "
              f"(our model's context is 2,048 -> "
              f"{'FITS' if total <= 2048 else f'OVER BY {total - 2048:,}'})",
              flush=True)
        print(f"  saved -> {path}", flush=True)

        if body.get("stream"):
            self._sse()
        else:
            self._json(200, {
                "id": "msg_probe", "type": "message", "role": "assistant",
                "model": body.get("model") or "probe",
                "content": [{"type": "text", "text": "probe-ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": total, "output_tokens": 3},
            })

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def ev(kind, obj):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(obj)}\n\n"
                             .encode())
            self.wfile.flush()

        ev("message_start", {"type": "message_start", "message": {
            "id": "msg_probe", "type": "message", "role": "assistant",
            "model": "probe", "content": [], "stop_reason": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}})
        ev("content_block_start", {"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "text", "text": ""}})
        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta",
                                             "text": "probe-ok"}})
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": "end_turn"},
                             "usage": {"output_tokens": 3}})
        ev("message_stop", {"type": "message_stop"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"probe server on http://127.0.0.1:{port}  (captured -> {OUT})",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
