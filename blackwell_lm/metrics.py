"""Durable metrics for every training stage.

WHY THIS IS A MODULE AND NOT THREE COPIES. Pretraining's loss history lived only
in a log file on ephemeral spot storage; two reclaims destroyed about two thirds
of the curve before anyone looked, and the checkpoint carried model/opt/step and
no loss at all. That was fixed for pretraining -- and then SFT ran, completed,
and had its GATE RESULTS erased by the next reclaim, because the same fix had
not been carried across. The weights survived; the evidence did not.

The rule this encodes: anything you would be annoyed to lose must be written
BOTH to a JSONL file that is mirrored to S3 and into the checkpoint itself. A
checkpoint that reaches S3 and a result that does not is an asymmetry that keeps
biting.
"""

from __future__ import annotations

import json
import os
import time


def append(path: str, row: dict) -> None:
    """Append one JSON line. Cheap enough to call at every log interval."""
    row = {"t": time.time(), **row}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def load(path: str) -> list[dict]:
    """Read back a metrics file, tolerating a torn final line.

    A process killed mid-write (spot reclaim) can leave a partial last line;
    discarding it is right, because the alternative is refusing to read any of
    the history that did survive.
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue          # torn tail from an interrupted write
    return out


def publish(local: str, uri: str | None, log=print) -> None:
    """Mirror a metrics file to S3. Never fatal -- losing the mirror is bad, but
    killing a training run over it is worse."""
    if not uri or not os.path.exists(local):
        return
    try:
        import boto3
        bucket, _, key = uri[len("s3://"):].partition("/")
        boto3.client("s3").upload_file(local, bucket, key)
        log(f"metrics mirrored -> {uri}")
    except Exception as e:
        log(f"WARNING: metrics mirror failed ({e!r}); the local file is still there")


def metrics_uri_for(checkpoint_uri: str | None, name: str = "metrics.jsonl") -> str | None:
    """Put metrics next to the checkpoint they describe, so the two travel
    together and a reader does not have to guess which run a file belongs to."""
    if not checkpoint_uri:
        return None
    return f"{checkpoint_uri.rsplit('/', 1)[0]}/metrics/{name}"
