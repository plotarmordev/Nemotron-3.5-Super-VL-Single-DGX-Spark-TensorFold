#!/usr/bin/env python3
"""Smoke test for the running server: one text and one image chat completion, greedy, thinking off.

Prints each reply with its decode speed and token_sha256, TensorFold's digest of the reply's token ids: two builds
that are exact give the same digest. The image is drawn here, so the test needs no file and no network; the server
must load the image tower (VISION=1, the default).
API_URL (default http://127.0.0.1:$PORT, PORT default 8888) and MODEL (default: the first id /v1/models lists) pick
the server; TENSORFOLD_API_KEY, when set in the environment (.env is not read), supplies its key (the first, if it
lists several).
Exits 1 when a request fails or a reply is empty.
"""
import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib

API_URL = os.environ.get("API_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8888')}").rstrip("/")
API_KEY = os.environ.get("TENSORFOLD_API_KEY", "").split(",")[0].strip()


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(API_URL + path, data, headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def png(width, height, pixel):
    """An RGB PNG; pixel(x, y) gives each pixel's (r, g, b)."""
    rows = b"".join(b"\0" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def chat(model, label, content, max_tokens):
    r = request("/v1/chat/completions", {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = (r["choices"][0]["message"].get("content") or "").strip()
    t = r["tensorfold"]
    print(f"{label}: {r['usage']['prompt_tokens']} prompt + {r['usage']['completion_tokens']} tokens, "
          f"{t['tokens_per_second']:.1f} tok/s, token_sha256 {t['token_sha256']}")
    print(f"  {text!r}")
    return bool(text)


def main():
    model = os.environ.get("MODEL") or request("/v1/models")["data"][0]["id"]
    # left half red, right half blue
    image = png(448, 448, lambda x, y: (220, 30, 30) if x < 224 else (30, 60, 220))
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    ok = chat(model, "text", "Name the three primary colours of light, in one short sentence.", 48)
    ok &= chat(model, "image", [
        {"type": "text", "text": "Which two colours fill this picture, and where is each? One sentence."},
        {"type": "image_url", "image_url": {"url": url}},
    ], 48)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as e:
        print(f"smoke: HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except (OSError, KeyError, ValueError) as e:
        print(f"smoke: {e}", file=sys.stderr)
        sys.exit(1)
