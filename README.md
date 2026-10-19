# Nemotron 3.5 Super VL on one DGX Spark

[![license: MIT](https://img.shields.io/badge/license-MIT-0969da?style=flat-square)](#license)
[![platform: DGX Spark GB10](https://img.shields.io/badge/platform-DGX_Spark_GB10-76b900?style=flat-square)](#quickstart)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76b900?style=flat-square)](#quickstart)
[![model: Nemotron 3.5 Super VL NVFP4](https://img.shields.io/badge/model-Nemotron_3.5_Super_VL_NVFP4-8250df?style=flat-square)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
[![engine: TensorFold 1.1.0 + 2 patches](https://img.shields.io/badge/engine-TensorFold_1.1.0_%2B_2_patches-57606a?style=flat-square)](#carried-patches)
[![status: +28% one at a time, +22% at 4, +53% at 8, images 28% sooner](https://img.shields.io/badge/status-%2B28%25_one_at_a_time%2C_%2B22%25_at_4%2C_%2B53%25_at_8%2C_images_28%25_sooner-a16207?style=flat-square)](#stock-vs-this-recipe)

**NVIDIA's 120B vision-language model on one DGX Spark: 28% faster than stock TensorFold 1.0.12 one request at a
time, 22% with 4 at once and 53% with 8 on a counting test, and an image prompt's first word 28% sooner than on
stock TensorFold 1.1.0, with the same replies.**

This recipe serves Nemotron 3.5 Super VL from NVIDIA's NVFP4 checkpoint with
[TensorFold](https://github.com/ashhart/TensorFold) 1.1.0's native CUDA engine, which now ships the 16 engine changes
this recipe used to patch in, plus [2 patches](#carried-patches) it carries until a TensorFold release ships them.
In every check we ran, its replies were token for token the same as stock TensorFold 1.0.12's and 1.1.0's. You get an
OpenAI-compatible API with text and images, up to 8 requests at once and a 131,072-token window each. The weights
are not in this repository: you download them from
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4).

| I want to... | Use | What you get |
|---|---|---|
| Serve the model on my Spark | [Quickstart](#quickstart) ![Ready to try](https://img.shields.io/badge/ready_to_try-0969da?style=flat-square) | API on port 8888, text and images |
| Check the speed claims | [`tools/bench.py`](tools/bench.py) ![Benchmark tools](https://img.shields.io/badge/benchmark_tools-1a7f37?style=flat-square) | The More measurements table on your machine, with reply hashes |
| See what changed in the engine | [Engine changes](#engine-changes-now-in-tensorfold-110) ![In TensorFold 1.1.0](https://img.shields.io/badge/in_TensorFold_1.1.0-1a7f37?style=flat-square) | 16 changes, merged upstream and shipped in TensorFold 1.1.0 |

## Stock vs this recipe

Measured with [the Grill](https://github.com/plotarmordev/thegrill) 0.9.0 on one DGX Spark, a different one from
this table's earlier numbers and from More measurements below: greedy, thinking off, 400 tokens of counting from 1
to 1,000 per request, one request at a time and 4 or 8 at once (combined tok/s). Stock TensorFold 1.0.12 ran before
and after this recipe, each at its own defaults.

| | Stock TensorFold 1.0.12 | This recipe | Stock again |
|---|---:|---:|---:|
| 1 at a time, median tok/s (8 runs each) | 71.5 | **95.1** | 74.0 |
| Change | | **+28%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift +3.4% |
| 4 at once, median combined tok/s | 146.5 | **165.3** | 123.8 |
| Change | | **+22%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift −15.5% |
| 8 at once, median combined tok/s | 157.6 | **247.6** | 142.4 |
| Change | | **+53%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift −10.7% |

Every reply in all three deployments was the same text: 24 one at a time, 96 at 4 at once, 192 at 8 at once, and
the warm-ups (8, 32 and 64).

<details>
<summary>What this number doesn't show</summary>

- **One workload:** one counting prompt, sent to every request when several run at once. Counting is very
  predictable, so speculative decoding (drafting) does most of the work. Quote "+28% one at a time, +22% at 4 and
  +53% at 8 on this workload", not a general speedup.
- **Identical requests flatter the concurrent rows:** identical prompts sent together usually decode in step, and each
  shared round then reads about one request's experts. When one request gets a round ahead they no longer do, and the
  wave runs about 20% slower. Prompts that differ (tested with a different first word per request) lose that bonus;
  More measurements sends a different prompt per request (+15% at 4 at once, +30% at 8).
- **Not covered:** images and long prompts (More measurements times their first token), more than 8 requests at once.
- **Whole deployments:** stock ran at its defaults (a 16,384-token window), the recipe at its own (131,072 tokens),
  both with images on, so this compares deployments, not engines; More measurements compares the engines at the same
  settings.
- **Statistics:** one at a time, +32.73% over the first stock run (95% interval +32.38 to +33.07) and +28.41% over
  the second (+28.39 to +28.42). At 4 at once, +21.83% (+10.98 to +33.74) and +44.15% (+31.26 to +58.30). At 8,
  +53.03% (+39.36 to +68.05) and +71.37% (+54.20 to +90.45). Each quoted change is the smaller of the two. Each is
  the Grill's estimate over all 8 runs (log scale), not the ratio of the medians shown. The intervals are
  model-based and may be too narrow for runs made one after another.
- **Stock drift:** stock's second run was 3.4% faster one at a time, 15.5% slower at 4 at once and 10.7% slower at 8
  (at 8 the Grill finds no clear shift). At 4 at once the thermal guard paused its prompts 22 times, against 3 in
  its first run: that explains the drift and inflates the +44% over it, so the quoted +22% is over the first run.
- **Heat pauses and spread:** the engines' thermal guard (85/75 °C) paused prompts for 2 s each: 14 times in stock's
  first run (3 at 4 at once, 11 at 8), 39 in the recipe's (8 and 31) and 50 in stock's second (22 and 28), never one
  at a time. At 4 at once the recipe ran at two levels, 203 combined tok/s in 3 of its 8 runs and 162 to 165 in 5.
  At 8 it ran at 247.5 to 247.7 in 7 runs and 211 in one; stock moved between about 138, 157 and 175 in its first
  run, and 127 and 157 in its second.
- **Startup tuning:** at its first start the engine times its drafting options and saves the result; which table a
  start finds can move counting by a few percent.
- **Run details:** 8 runs per deployment, all complete: 32 of 32 requests (12,800 tokens; 8 of them warm-ups) one at
  a time, 128 of 128 (51,200 tokens; 32 warm-ups) at 4 at once and 256 of 256 (102,400 tokens; 64 warm-ups) at 8,
  for stock, the recipe and stock again.

</details>

<details>
<summary>More measurements</summary>

Stock TensorFold 1.0.12's release binary against this recipe's build (TensorFold 1.1.0 with its 2 patches), on one
DGX Spark, both at the recipe's settings. Two runs of each, alternated (stock, recipe, stock, recipe), each from a
cool, idle box; a run is three parts (decode, requests at once, first token), each on a fresh server that times its
own drafting options. The mean of the two runs' medians of 3 requests after a warmup, server-reported decode tok/s,
256-token replies (400 for counting). All 532 replies in these runs and the image runs below have the same token
hash as stock's.

| Measure | Stock 1.0.12 | This recipe | Change |
|---|---:|---:|---:|
| Counting, drafted | 77.83 | 99.40 | +28% |
| Short story, drafted | 30.02 | 35.23 | +17% |
| Python code, drafted | 44.84 | 53.91 | +20% |
| Describing an image, drafted | 61.38 | 71.97 | +17% |
| English essay, drafted | 30.29 | 35.70 | +18% |
| Spanish essay, drafted | 32.38 | 35.87 | +11% |
| French essay, drafted | 29.84 | 33.39 | +12% |
| German essay, drafted | 30.06 | 33.42 | +11% |
| Italian essay, drafted | 33.73 | 37.48 | +11% |
| Japanese essay, drafted | 39.41 | 44.67 | +13% |
| Counting, `"draft": false` | 18.98 | 22.00 | +16% |
| 1 request, total tok/s | 43.46 | 53.53 | +23% |
| 2 requests at once, total tok/s | 42.86 | 48.21 | +12% |
| 4 requests at once, total tok/s | 58.05 | 66.48 | +15% |
| 8 requests at once, total tok/s | 69.77 | 90.81 | +30% |
| First token, 2,048-token prompt | 1.20 s | 0.90 s | −25% |
| First token, 8,192-token prompt | 4.40 s | 3.07 s | −30% |
| First token, 16,384-token prompt | 12.20 s | 8.99 s | −26% |

First token with images, in seconds, against both stock releases: 1.1.0 is the one patch 0001 fixes (it runs the
image tower twice per image), 1.0.12 the one the table above compares against. Two runs of each build, alternated.

| Measure | Stock 1.0.12 | Stock 1.1.0 | This recipe | Change vs 1.1.0 | Change vs 1.0.12 |
|---|---:|---:|---:|---:|---:|
| First token, 1 image | 2.76 | 3.17 | 2.28 | −28% | −18% |
| First token, 4 images | 16.43 | 18.68 | 14.98 | −20% | −9% |
| First token, 4 images, less heat waits | 8.43 | 10.63 | 6.98 | −34% | −17% |

- **Drafting off:** every drafted prompt above, and a multilingual one, was also sent with `"draft": false` and gave
  the same hash. Concurrent requests equalled their solo runs.
- **Heat pauses:** decode tok/s counts from the first token, so the thermal guard's pauses on prompts don't touch
  it. A pause in a timed concurrent run or a 2k or 8k prompt would have sent the run back; none fell there.
- **First token:** cold prompts on a server with `--context 18432 --prompt-cache-gib 0`, the host cooled to 75 °C
  before each. The guard paused every 16k prompt (stock 4 to 6 s, the recipe 4 s), and that row includes the
  pauses. Less the pauses the guard logged: 7.20 s against 4.99 s (−31%).
- **Images:** 2048 x 1664 PNGs, 1 or 4 in one message, 32 tokens without drafting, the host cooled to 75 °C before
  each request. No 1-image request paused, but every 4-image request, on every build, waited 8 s for the board to
  cool (one stock 1.1.0 request 12 s); the last row takes out the waits the guard logged. All three builds gave the
  same replies.
- **Reproduce:** `tools/bench.py decode` (add a prompt name, e.g. `decode count`, for one prompt), `streams`, and
  `ttft` on a server restarted with
  `VISION=0 ./start.sh restart --context 18432 --parallel 1 --segments 1 --prompt-cache-gib 0`. The first token
  with 1 and 4 images: `image-ttft` on a server restarted with
  `./start.sh restart --vision-image-tokens 13312 --vision-max-images 4 --context 18432 --parallel 1 --prompt-cache-gib 0`.

</details>

## Quickstart

You need a DGX Spark with the NVIDIA driver and a systemd user session, the Hugging Face CLI (`hf`), about 70 GiB
of disk, and about 115 GiB of free memory at start. Building the patched engine also needs git, xz and CUDA 13's
`nvcc` (on `PATH`, or set `NVCC`); `scripts/prepare.sh` fetches the pinned Zig 0.17.0 itself (or set `ZIG`).

```bash
git clone https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold.git
cd Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --revision 82d93699acd3be9e4617c53360fe5b3f5dbeb1b3 --local-dir ~/models/nemotron-3.5-super-vl-nvfp4
./start.sh        # first run downloads and builds the engine (~6 min) and checks the checkpoint
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
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | empty | Override the drafting depth; `15` / `0.5` suited counting and repetition on the patched 1.0.12, not yet measured on 1.1.0 |
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
<summary>A Spark without GitHub or the build tools</summary>

Build on any aarch64 Linux machine that has them, then copy the engine over (`spark` is the Spark's SSH host):

```bash
scripts/prepare.sh --engine-only
ssh spark mkdir -p opt/tensorfold
rsync -a ~/opt/tensorfold/tensorfold-1.1.0-* spark:opt/tensorfold/
```

On the Spark, `./start.sh` checks the copied engine against its `SHA256SUMS` and uses it as it is. To build on the
Spark itself without GitHub, copy the release archive and its `.sha256` from
[TensorFold v1.1.0](https://github.com/ashhart/TensorFold/releases/tag/v1.1.0) and a TensorFold clone there, and set
`TF_ARCHIVE` and `TF_SRC`; `scripts/prepare.sh` checks them against the pinned `TF_SHA256` and `TF_COMMIT`.

</details>

### Try it with your coding agent

Paste this into Claude Code, Codex, Cursor or whichever agent you use, on the DGX Spark:

```text
Help me serve Nemotron 3.5 Super VL with https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
Read its README first and use only the repository's own scripts.

1. Check for an NVIDIA GB10, the hf CLI, git, xz, CUDA 13 nvcc, 70 GiB of disk and ~115 GiB of free memory.
   Stop and tell me if anything is missing.
2. Download the checkpoint at the README's pinned revision, then run ./start.sh. The first run takes a while.
3. Run tools/smoke.py and show me both replies.

Don't stop other GPU work or change settings I didn't ask for.
```

## Carried patches

The recipe builds TensorFold 1.1.0 with these changes before a TensorFold release ships them. Each is also an
upstream pull request; [`patches/PATCHES.md`](patches/PATCHES.md) tracks its pull request, status and checks, and
the release that ships a change replaces its patch.

- **Images:** a prompt with images runs the RADIO tower once per image, not twice
  ([0001](patches/0001-nemotron-cuda-a-prompt-pass-with-images-runs-the-RAD.patch),
  [#1057](https://github.com/ashhart/TensorFold/pull/1057)).
- **Several requests at once:** their shared windows run the latent shared expert on the side stream again
  ([0002](patches/0002-nemotron-cuda-shared-windows-of-several-sequences-ru.patch),
  [#1058](https://github.com/ashhart/TensorFold/pull/1058)).

## Engine changes, now in TensorFold 1.1.0

This recipe used to carry 16 engine changes as patches on TensorFold 1.0.12. They were merged upstream, with the
maintainer's follow-ups, and ship in TensorFold 1.1.0, the release the recipe builds on. Each keeps TensorFold's
exactness contract: drafted replies equal undrafted ones, concurrent requests equal solo runs, and resumed prompts
equal fresh ones.

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

- **Images:** replies match stock's tokens. Image speed was measured on one image prompt (decode) and one picture
  size (first token with 1 and 4 images), and every 4-image prompt paused about 8 s for heat.
- **Two Sparks:** splitting the model over two GB10s is
  [upstream's setup](https://github.com/ashhart/TensorFold/blob/v1.1.0/docs/recipes/nemotron-3.5-super-vl.md#run-on-two-gb10s),
  not this recipe.
- **Thermal guard:** long prompts can pause while the board cools (40–74 s each on 112k-token prompts); replies
  are unchanged.

</details>

## License

The recipe is [MIT](LICENSE). Its scripts follow
[Mia's AI Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), whose notice
`LICENSE` keeps. The patches modify TensorFold and are Apache-2.0 ([`patches/LICENSE`](patches/LICENSE),
[`patches/NOTICE`](patches/NOTICE)). TensorFold (Apache-2.0) and the model, under NVIDIA's
[OpenMDW License 1.1](https://openmdw.ai/license/1-1/), are downloaded, never redistributed.

[Credits](CREDITS.md) · [Patches](patches/PATCHES.md) · [Settings](scripts/config.sh) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
