"""Reduce a captured client request to the measurements, dropping its content.

WHY. serve/probe_server.py records what a coding CLI actually sends, which is
how the ~18,850-token figure in this project was measured rather than guessed.
But the raw capture contains the client's VERBATIM system prompt and all of its
tool schemas -- a commercial product's internal prompt -- plus a session id.
None of that is ours to publish, and none of it is needed: every claim made
from the capture is about SIZES.

So the published artefact keeps the shape and the numbers and drops the text.
Anyone can regenerate the full capture locally with probe_server.py.
"""
from __future__ import annotations

import io
import json
import sys


def summarise(path: str) -> dict:
    d = json.load(io.open(path, encoding="utf-8"))
    body = d.get("body", {})

    sysf = body.get("system")
    if isinstance(sysf, list):
        blocks = [len(b.get("text", "")) for b in sysf if isinstance(b, dict)]
    else:
        blocks = [len(sysf or "")]

    tools = body.get("tools") or []
    msgs = body.get("messages") or []

    approx = lambda n: max(1, n // 4)          # ~4 chars/token, crude on purpose
    sys_chars = sum(blocks)
    tools_chars = len(json.dumps(tools))
    msgs_chars = len(json.dumps(msgs))

    return {
        "_note": ("Structural summary only. The client's system prompt and tool "
                  "schemas are its own product content and are deliberately NOT "
                  "reproduced here; regenerate a full capture locally with "
                  "serve/probe_server.py if you need it."),
        "path": d.get("path"),
        "model": body.get("model"),
        "stream": body.get("stream"),
        "max_tokens": body.get("max_tokens"),
        "system_prompt": {"blocks": len(blocks), "chars": sys_chars,
                          "approx_tokens": approx(sys_chars)},
        "tools": {"count": len(tools),
                  "names": sorted(t.get("name", "?") for t in tools),
                  "chars": tools_chars, "approx_tokens": approx(tools_chars)},
        "messages": {"count": len(msgs), "chars": msgs_chars,
                     "approx_tokens": approx(msgs_chars)},
        "approx_total_tokens": approx(sys_chars + tools_chars + msgs_chars),
    }


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src
    s = summarise(src)
    io.open(out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(s, indent=2) + "\n")
    print(f"{src} -> {out}: {s['approx_total_tokens']:,} approx tokens "
          f"({s['tools']['count']} tools), content dropped")
