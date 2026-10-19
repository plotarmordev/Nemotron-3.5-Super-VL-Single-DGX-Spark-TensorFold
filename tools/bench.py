#!/usr/bin/env python3
"""Measures the running server as the README's Measured table was measured, so two builds can be paired.

  tools/bench.py decode [PROMPT...]
      count, prose, code, image, multilingual and lang_en/es/fr/de/it/ja (or the PROMPTs named) drafted: a warmup
      and 3 measured requests each, the median of the server's decode tok/s; count with "draft": false the same
      way, and one "draft": false reply for every other prompt
  tools/bench.py streams
      1, 2, 4 and 8 short prompts at once: a warmup and 2 measured runs, the median aggregate tok/s; each stream's
      reply against the same prompt alone
  tools/bench.py ttft
      cold prompts of 2,048, 8,192 and 16,384 tokens, 32 tokens "draft": false: a warmup and 3 each, the median time
      to first token. Start the server for it with
      VISION=0 ./start.sh restart --context 18432 --parallel 1 --segments 1 --prompt-cache-gib 0
  tools/bench.py image-ttft
      cold prompts with 1 and 4 generated 2048 x 1664 PNGs, 32 tokens "draft": false: a warmup and 3 each.
      Start with --vision --vision-image-tokens 13312 --vision-max-images 4 --context 18432 --parallel 1
      --prompt-cache-gib 0; cached rows are refused. Cold means uncached prompt rows, not a cold server.

Every request is greedy, thinking off and streamed. Before a timed request, a hottest thermal zone above
TF_HEAT_HIGH (85) is let cool to TF_HEAT_LOW (75), so the server's own thermal guard does not pause it; both TTFT modes
cool to TF_HEAT_LOW before every request. Each wait prints one "cooling" line on stderr. Each result ends with the reply's
token_sha256: builds that are exact print the same. API_URL, MODEL and TENSORFOLD_API_KEY (from the environment; .env
is not read) pick the server as for tools/smoke.py. Run it on the server's machine, whose thermal zones it reads.
"""
import base64
import functools
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import prompts

API_URL = os.environ.get("API_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8888')}").rstrip("/")
API_KEY = os.environ.get("TENSORFOLD_API_KEY", "").split(",")[0].strip()
HEAT_HIGH = float(os.environ.get("TF_HEAT_HIGH", "85"))
HEAT_LOW = float(os.environ.get("TF_HEAT_LOW", "75"))
REPS = 3
DECODE = ("count", "prose", "code", "image", "multilingual",
          "lang_en", "lang_es", "lang_fr", "lang_de", "lang_it", "lang_ja")
SHORT = ("code", "prose", "reasoning", "repetition", "count", "multilingual", "tool_json", "gpu_explain")


def request(path, body=None):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(API_URL + path, data, headers), timeout=1800)


@functools.cache
def model():
    if name := os.environ.get("MODEL"):
        return name
    with request("/v1/models") as r:
        return json.load(r)["data"][0]["id"]


def tokenize(text, special):
    with request("/tokenize", {"prompt": text, "add_special_tokens": special}) as r:
        return json.load(r)["tokens"]


def hottest():
    temps = [int(z.read_text()) / 1000 for z in Path("/sys/class/thermal").glob("thermal_zone*/temp")]
    if not temps:
        raise RuntimeError("no thermal zones in /sys/class/thermal: run bench.py on the server's machine")
    return max(temps)


def cool(always=False):
    t = hottest()
    if t > HEAT_LOW and (always or t > HEAT_HIGH):
        print(f"cooling: {t:.0f} °C > TF_HEAT_LOW {HEAT_LOW:.0f} °C", file=sys.stderr, flush=True)
        while hottest() > HEAT_LOW:
            time.sleep(1)


def streamed(choice):
    """The text a chunk carries, tool calls included."""
    delta = choice.get("delta") or {}
    calls = "".join((c.get("function") or {}).get(k) or ""
                    for c in delta.get("tool_calls") or [] for k in ("name", "arguments"))
    return (choice.get("text") or "") + (delta.get("reasoning_content") or "") + (delta.get("content") or "") + calls


def generate(path, body, gate=None):
    """One streamed request: the server's own figures, and when the client saw the first and last text."""
    body = {"model": model(), "stream": True, "stream_options": {"include_usage": True}, **body}
    if gate:
        gate.wait(timeout=60)
    first = last = None
    final = {}
    with request(path, body) as r:
        for raw in r:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(f"server error: {chunk['error']}")
            final.update({k: chunk[k] for k in ("usage", "tensorfold") if chunk.get(k)})
            if any(streamed(c) for c in chunk.get("choices", [])):
                last = time.perf_counter()
                first = first or last
    usage, server = final["usage"], final["tensorfold"]
    return {"tokens": usage["completion_tokens"], "cached": usage["prompt_tokens_details"]["cached_tokens"],
            "sha256": server["token_sha256"], "tps": server["tokens_per_second"],
            "ttft": server["time_to_first_token"], "first": first, "last": last}


def chat(prompt, draft=True):
    body = {"messages": prompt["messages"], "max_tokens": prompt.get("max_tokens", prompts.MAX_TOKENS),
            "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    if "tools" in prompt:
        body["tools"] = prompt["tools"]
    if not draft:
        body["draft"] = False
    return body


def timed(body):
    """A warmup and REPS measured requests: the median decode tok/s, the reply's hash, whether all replies agreed."""
    runs = []
    for _ in range(REPS + 1):
        cool()
        runs.append(generate("/v1/chat/completions", body))
    return statistics.median(r["tps"] for r in runs[1:]), runs[0]["sha256"], len({r["sha256"] for r in runs}) == 1


def row(name, label, tps, stable, sha, tail=""):
    print(f"{name:13} {label:14} {tps:7.2f} tok/s  {'stable' if stable else 'UNSTABLE'}  {sha}{tail}")


def decode(*names):
    chosen = prompts.prompts()
    for name in names or DECODE:
        prompt = chosen[name]
        tps, sha, stable = timed(chat(prompt))
        if name == "count":
            off_tps, off, off_stable = timed(chat(prompt, draft=False))
        else:
            off = generate("/v1/chat/completions", chat(prompt, draft=False))["sha256"]
        row(name, "drafted", tps, stable, sha, '  = "draft": false' if off == sha else '  != "draft": false')
        if name == "count":
            row(name, '"draft": false', off_tps, off_stable, off)


def together(bodies):
    cool()
    gate = threading.Barrier(len(bodies))
    out = [None] * len(bodies)

    def run(i):
        try:
            out[i] = generate("/v1/chat/completions", bodies[i], gate)
        except Exception as e:  # re-raised below, on the main thread
            gate.abort()
            out[i] = e

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors := [r for r in out if isinstance(r, Exception)]:
        raise errors[0]
    return out


def streams():
    chosen = prompts.prompts()
    short = [chosen[name] for name in SHORT]
    solo = [generate("/v1/chat/completions", chat(p))["sha256"] for p in short]
    for n in (1, 2, 4, 8):
        bodies = [chat(p) for p in short[:n]]
        together(bodies)
        runs = [together(bodies) for _ in range(2)]
        aggregate = statistics.median(sum(r["tokens"] - 1 for r in run) / (max(r["last"] for r in run) -
                                                                          min(r["first"] for r in run)) for run in runs)
        equal = sum(r["sha256"] == solo[i] for run in runs for i, r in enumerate(run))
        print(f"{n} streams  {aggregate:7.2f} tok/s aggregate  {equal}/{2 * n} replies equal to the prompt alone")


def ttft():
    body = tokenize(prompts.corpus(), special=False)

    def cold(width, tag):
        ids = (tokenize(f"Cold prefill {width}-{tag}.\n", special=True) + body)[:width]
        cool(always=True)
        return generate("/v1/completions", {"prompt": ids, "temperature": 0, "max_tokens": 32, "draft": False})

    cold(2048, "warmup")
    for width in (2048, 8192, 16384):
        runs = [cold(width, rep) for rep in range(REPS)]
        print(f"{width:6} tokens  TTFT {statistics.median(r['ttft'] for r in runs):.3f} s  "
              f"cached {sum(r['cached'] for r in runs)}  {runs[0]['sha256']}")


def image_ttft():
    url = "data:image/png;base64," + base64.b64encode(prompts.png(2048, 1664)).decode()
    image = {"type": "image_url", "image_url": {"url": url}}
    for count in (1, 4):
        content = [image] * count + [{"type": "text", "text": "Describe the shapes and colours in these images."}]
        body = chat({"messages": [{"role": "user", "content": content}], "max_tokens": 32}, draft=False)
        runs = []
        for rep in range(REPS + 1):
            cool(always=True)
            result = generate("/v1/chat/completions", body)
            if result["cached"]:
                raise RuntimeError("image-ttft needs --prompt-cache-gib 0: the server reused prompt rows")
            if rep:
                runs.append(result)
                print(f"{count} images  run {rep}  TTFT {result['ttft']:.3f} s  "
                      f"cached {result['cached']}  token_sha256 {result['sha256']}")
        stable = len({r["sha256"] for r in runs}) == 1
        print(f"{count} images  median TTFT {statistics.median(r['ttft'] for r in runs):.3f} s  "
              f"{'stable' if stable else 'UNSTABLE'}  token_sha256 {runs[0]['sha256']}")


MODES = {"decode": decode, "streams": streams, "ttft": ttft, "image-ttft": image_ttft}

if __name__ == "__main__":
    mode, names = (sys.argv[1] if len(sys.argv) > 1 else None), sys.argv[2:]
    if mode not in MODES or (names and mode != "decode") or not set(names) <= set(DECODE):
        sys.exit(__doc__)
    try:
        request("/health").close()  # reach the server before any wait for cooling
        MODES[mode](*names)
    except urllib.error.HTTPError as e:
        sys.exit(f"bench: HTTP {e.code}: {e.read().decode(errors='replace')}")
    except (OSError, KeyError, ValueError, RuntimeError) as e:
        sys.exit(f"bench: {e}")
