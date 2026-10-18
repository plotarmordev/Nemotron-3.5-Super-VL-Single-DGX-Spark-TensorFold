# Nemotron 3.5 Super VL on one DGX Spark

[![license: MIT](https://img.shields.io/badge/license-MIT-0969da?style=flat-square)](#license)
[![platform: DGX Spark GB10](https://img.shields.io/badge/platform-DGX_Spark_GB10-76b900?style=flat-square)](#quickstart)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76b900?style=flat-square)](#quickstart)
[![model: Nemotron 3.5 Super VL NVFP4](https://img.shields.io/badge/model-Nemotron_3.5_Super_VL_NVFP4-8250df?style=flat-square)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
[![engine: TensorFold 1.1.0](https://img.shields.io/badge/engine-TensorFold_1.1.0-57606a?style=flat-square)](#engine-changes-now-in-tensorfold-110)
[![status: +20% on one workload](https://img.shields.io/badge/status-%2B20%25_on_one_workload-a16207?style=flat-square)](#stock-vs-this-recipe)

**NVIDIA's 120B vision-language model on one DGX Spark: 20% faster than stock TensorFold on a one-request counting
test, with the same replies.**

This recipe serves Nemotron 3.5 Super VL from NVIDIA's NVFP4 checkpoint with stock
[TensorFold](https://github.com/ashhart/TensorFold) 1.1.0's native CUDA engine, which now ships the 16 engine changes
this recipe used to patch in. In every check we ran, its replies were token for token the same as stock TensorFold
1.0.12's. You get an OpenAI-compatible API with text and images, up to 8 requests at once and a 131,072-token window
each. The weights are not in this repository: you download them from
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4).

| I want to... | Use | What you get |
|---|---|---|
| Serve the model on my Spark | [Quickstart](#quickstart) ![Ready to try](https://img.shields.io/badge/ready_to_try-0969da?style=flat-square) | API on port 8888, text and images |
| Check the speed claims | [`tools/bench.py`](tools/bench.py) ![Benchmark tools](https://img.shields.io/badge/benchmark_tools-1a7f37?style=flat-square) | The More measurements table on your machine, with reply hashes |
| See what changed in the engine | [Engine changes](#engine-changes-now-in-tensorfold-110) ![In TensorFold 1.1.0](https://img.shields.io/badge/in_TensorFold_1.1.0-1a7f37?style=flat-square) | 16 changes, merged upstream and shipped in TensorFold 1.1.0 |

## Stock vs this recipe

Measured with [the Grill](https://github.com/plotarmordev/thegrill) on one DGX Spark: greedy, thinking off, 400
tokens of counting from 1 to 1,000 per request, one request at a time (Grill 0.6.0) and 4 or 8 at once (Grill
0.8.0, combined tok/s). Stock ran before and after the recipe.

| | Stock TensorFold 1.0.12 | This recipe | Stock again |
|---|---:|---:|---:|
| 1 at a time, median tok/s (8 runs each) | 75.43 | **90.96** | 75.83 |
| Change | | **+20.6%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift +0.5% |
| 4 at once, median combined tok/s | 150.14 | **196.26** | 149.50 |
| Change | | **+26.8%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift −2.5% |
| 8 at once, median combined tok/s | 177.65 | **239.95** | 168.89 |
| Change | | **+41.0%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift −2.6% |

Every measured reply was the same text in both deployments: 24 one at a time (and its 8 warm-ups), 96 at 4 at
once, 192 at 8 at once.

<details>
<summary>What this number doesn't show</summary>

- **One workload:** one counting prompt, sent to every request when several run at once. Counting is very
  predictable, so speculative decoding (drafting) does most of the work. Quote "+20% one at a time, +27% at 4 and
  +41% at 8 on this workload", not a general speedup.
- **Not covered:** images, long prompts, more than 8 requests at once.
- **Different settings:** the recipe also serves a 131,072-token window where stock serves 16,384, so the gains
  can't be credited to the patches alone; More measurements compares both at the same settings.
- **Statistics:** one at a time, +20.56% over the first stock run (95% interval +20.43 to +20.69) and +19.99% over
  the second (+19.87 to +20.10). At 4 at once, +26.84% (+10.25 to +45.94) and +30.09% (+12.07 to +50.99); at 8,
  +41.04% (+23.33 to +61.29) and +44.78% (+26.59 to +65.59). Each change is the Grill's estimate over all 8 runs
  (log scale), not the ratio of the medians shown. The intervals are model-based and may be too narrow for runs
  made one after another.
- **Wider spread when several run at once:** at 4 at once the recipe's runs ranged from 142 to 221 combined tok/s
  (most near 173 or 220), and the engines' thermal guard paused each deployment about 30 times (2 or 4 s each)
  during these runs. Those intervals are much wider than one at a time; the gains are still above zero against
  both stock runs.
- **Startup tuning:** at its first start the engine times its drafting options and saves the result; the one at
  a time runs reused a saved table. Fresh starts of an earlier build counted at 92.2 to 95.9 tok/s one at a time
  (7 starts).
- **Run details:** 8 runs per deployment, all complete: 32 of 32 requests (12,800 tokens) one at a time, 128 of
  128 (51,200 tokens) at 4 at once and 256 of 256 (102,400 tokens) at 8, for stock, the recipe and stock again.

</details>

<details>
<summary>More measurements</summary>

Stock TensorFold 1.0.12 built from source with the same toolchain, against this recipe's build, on one DGX Spark,
both at the recipe's settings. Medians of 3 requests after a warmup, server-reported decode tok/s, 256-token replies
(400 for counting). Every reply's token hash equals stock's.

| Measure | Stock | Recipe | Change |
|---|---:|---:|---:|
| Counting, drafted | 75.86 | 92.72 | +22% |
| Short story, drafted | 29.28 | 34.52 | +18% |
| Python code, drafted | 43.91 | 51.42 | +17% |
| Describing an image, drafted | 59.88 | 71.88 | +20% |
| English essay, drafted | 29.73 | 35.22 | +18% |
| Spanish essay, drafted | 31.62 | 35.16 | +11% |
| French essay, drafted | 29.22 | 33.34 | +14% |
| German essay, drafted | 29.44 | 33.11 | +12% |
| Italian essay, drafted | 33.21 | 36.83 | +11% |
| Japanese essay, drafted | 38.50 | 43.50 | +13% |
| Counting, `"draft": false` | 18.49 | 21.44 | +16% |
| 1 request, total tok/s | 43.93 | 51.46 | +17% |
| 2 requests at once, total tok/s | 41.05 | 47.32 | +15% |
| 4 requests at once, total tok/s | 57.10 | 64.97 | +14% |
| 8 requests at once, total tok/s | 68.69 | 87.57 | +28% |
| First token, 2,048-token prompt | 1.22 s | 0.91 s | −26% |
| First token, 8,192-token prompt | 4.45 s | 3.08 s | −31% |
| First token, 16,384-token prompt | 13.25 s | 8.96 s | −32% |

- **Drafting off:** counting, a multilingual prompt and the six essays were also sent with `"draft": false` and
  gave the same hashes. Concurrent requests equalled their solo runs.
- **First token:** cold prompts on a server with `--context 18432 --prompt-cache-gib 0`, the host cooled to 75 °C
  before each. The thermal guard still paused every 16k prompt (stock 6 s, recipe 4 s), and that row includes the
  pauses.
- **Second Spark:** a broader harness gave stock's tokens on 10 greedy prompts (one with an image), seeded sampled
  prose and code, 1/2/4/8 concurrent requests and resumed prompts.
- **Reproduce:** `tools/bench.py decode` (add a prompt name, e.g. `decode count`, for one prompt), `streams`, and
  `ttft` on a server restarted with
  `VISION=0 ./start.sh restart --context 18432 --parallel 1 --segments 1 --prompt-cache-gib 0`.

</details>

## Quickstart

You need a DGX Spark with the NVIDIA driver and a systemd user session, the Hugging Face CLI (`hf`), about 70 GiB
of disk, and about 115 GiB of free memory at start.

```bash
git clone https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold.git
cd Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --revision 82d93699acd3be9e4617c53360fe5b3f5dbeb1b3 --local-dir ~/models/nemotron-3.5-super-vl-nvfp4
./start.sh        # first run downloads the engine and checks the checkpoint
tools/smoke.py    # one text and one image request
./stop.sh
```

The API is at `http://<spark>:8888/v1` with model `nemotron-3.5-super-vl`. Logs:
`journalctl --user -u nemotron-super-vl -f`. The server listens on all interfaces: set `TENSORFOLD_API_KEY` to
require a key, or `HOST=127.0.0.1` to keep it local.

<details>
<summary>Settings</summary>

Set them in the environment or in `.env` (`KEY=value` lines; the environment wins), or pass `tensorfold serve`
flags after `./start.sh`. All of them are in [`scripts/config.sh`](scripts/config.sh).

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT` | `131072` | Window per request (stock: 16,384) |
| `PARALLEL` | `auto` | Requests decoded together, up to 8 |
| `VISION` | `1` | Load the image tower (~2.2 GiB) |
| `THINKING` | `1` | Reason before answering; a request's `enable_thinking` wins |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | empty | Override the drafting depth; `15` / `0.5` measured +10% on counting and −2.5% on code (earlier build) |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | Where the API listens |
| `MODEL_DIR` | `~/models/nemotron-3.5-super-vl-nvfp4` | The checkpoint folder |
| `TF_HEAT_HIGH` / `TF_HEAT_LOW` | `85` / `75` | Thermal guard (°C): a prompt chunk waits while the board is hotter |

`./start.sh restart --context 65536` checks new arguments before it stops the running server.

</details>

<details>
<summary>Memory</summary>

At the default window the engine takes 74.18 GiB for the model and its first request. At start it also sets aside the
other 7 requests' attention caches (7.9 GiB of pages they lease as they grow, plus 264 MiB each) and 7.8 GiB of its
16 GiB prompt cache; 12.2 GiB (a tenth of RAM) stays reserved. A started server left about 20 GiB of memory
available, and with 8 requests filled to 112k tokens each at least 13 GiB. With less free at start the engine keeps a
smaller cache, then fewer requests; its startup line says what it kept.

</details>

<details>
<summary>A Spark without GitHub</summary>

Copy the release archive and its checksum from
[TensorFold v1.1.0](https://github.com/ashhart/TensorFold/releases/tag/v1.1.0) to the Spark, then point the recipe at
them:

```bash
TF_ARCHIVE=~/tensorfold-1.1.0-linux-aarch64.tar.gz ./start.sh   # its .sha256 beside it
```

`scripts/prepare.sh` checks the archive against its `.sha256` and the pinned `TF_SHA256` before it unpacks it.

</details>

### Try it with your coding agent

Paste this into Claude Code, Codex, Cursor or whichever agent you use, on the DGX Spark:

```text
Help me serve Nemotron 3.5 Super VL with https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
Read its README first and use only the repository's own scripts.

1. Check for an NVIDIA GB10, the hf CLI, 70 GiB of disk and ~115 GiB of free memory.
   Stop and tell me if anything is missing.
2. Download the checkpoint at the README's pinned revision, then run ./start.sh. The first run takes a while.
3. Run tools/smoke.py and show me both replies.

Don't stop other GPU work or change settings I didn't ask for.
```

## Engine changes, now in TensorFold 1.1.0

This recipe used to carry 16 engine changes as patches on TensorFold 1.0.12. They were merged upstream, with the
maintainer's follow-ups, and ship in TensorFold 1.1.0, so the recipe now runs the stock release. Each keeps
TensorFold's exactness contract: drafted replies equal undrafted ones, concurrent requests equal solo runs, and
resumed prompts equal fresh ones.

- **Decode:** latent top-k with one thread per expert and NVFP4 experts at decode sizes
  ([#999](https://github.com/ashhart/TensorFold/pull/999)); shared-expert plans built once at load and the shared
  expert on a side stream beside the routed experts ([#1002](https://github.com/ashhart/TensorFold/pull/1002)), with
  the `--segments 2` fix ([#996](https://github.com/ashhart/TensorFold/pull/996)).
- **Drafting:** verify windows of up to 32 rows (31 drafts a round) and copied chains that start at 15 tokens
  ([#1006](https://github.com/ashhart/TensorFold/pull/1006)); a draft head over the embedded 49,152-token draft
  vocabulary, chosen per request from its recent tokens ([#1011](https://github.com/ashhart/TensorFold/pull/1011));
  `--mtp-drafts` / `--mtp-confidence` on CUDA ([#997](https://github.com/ashhart/TensorFold/pull/997)).
- **Prompts:** FP8 and NVFP4 tiles, fused quantization and share sums, and a tiled router
  ([#1008](https://github.com/ashhart/TensorFold/pull/1008)).

<details>
<summary>Known limits</summary>

- **Images:** replies match stock's tokens, but image prompt speed was not optimized and only one image prompt was
  measured.
- **Two Sparks:** splitting the model over two GB10s is
  [upstream's setup](https://github.com/ashhart/TensorFold/blob/v1.1.0/docs/recipes/nemotron-3.5-super-vl.md#run-on-two-gb10s),
  not this recipe.
- **Thermal guard:** long prompts can pause while the board cools (40–74 s each on 112k-token prompts); replies
  are unchanged.

</details>

## License

The recipe is [MIT](LICENSE). Its scripts follow
[Mia's AI Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), whose notice
`LICENSE` keeps. TensorFold (Apache-2.0) and the model, under NVIDIA's
[OpenMDW License 1.1](https://openmdw.ai/license/1-1/), are downloaded, never redistributed.

[Credits](CREDITS.md) · [Settings](scripts/config.sh) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
