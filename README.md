# Nemotron 3.5 Super VL on one DGX Spark

[![license: MIT](https://img.shields.io/badge/license-MIT-0969da?style=flat-square)](#license)
[![platform: DGX Spark GB10](https://img.shields.io/badge/platform-DGX_Spark_GB10-76b900?style=flat-square)](#quickstart)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76b900?style=flat-square)](#quickstart)
[![model: Nemotron 3.5 Super VL NVFP4](https://img.shields.io/badge/model-Nemotron_3.5_Super_VL_NVFP4-8250df?style=flat-square)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
[![engine: TensorFold 1.0.12 + 18 patches](https://img.shields.io/badge/engine-TensorFold_1.0.12_%2B_18_patches-57606a?style=flat-square)](#what-the-patches-change)
[![status: +20% on one workload](https://img.shields.io/badge/status-%2B20%25_on_one_workload-a16207?style=flat-square)](#stock-vs-this-recipe)

**NVIDIA's 120B vision-language model on one DGX Spark: 20% faster than stock TensorFold on a one-request counting
test, with the same replies.**

This recipe serves Nemotron 3.5 Super VL from NVIDIA's NVFP4 checkpoint with
[TensorFold](https://github.com/ashhart/TensorFold)'s native CUDA engine plus 18 patches. In every check we ran,
replies were token for token the same as stock's. You get an OpenAI-compatible API with text and images, up to 8
requests at once and a 131,072-token window each. The weights are not in this repository: you download them from
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4).

| I want to... | Use | What you get |
|---|---|---|
| Serve the model on my Spark | [Quickstart](#quickstart) ![Ready to try](https://img.shields.io/badge/ready_to_try-0969da?style=flat-square) | API on port 8888, text and images |
| Check the speed claims | [`tools/bench.py`](tools/bench.py) ![Benchmark tools](https://img.shields.io/badge/benchmark_tools-1a7f37?style=flat-square) | The More measurements table on your machine, with reply hashes |
| See what changed in the engine | [`patches/`](patches) ![Same tokens as stock](https://img.shields.io/badge/same_tokens_as_stock-1a7f37?style=flat-square) | 18 patches (16 engine changes, 2 docs) against TensorFold `v1.0.12` |

## Stock vs this recipe

Measured with [the Grill](https://github.com/plotarmordev/thegrill) 0.6.0 on one DGX Spark: one chat request at a
time, greedy, thinking off, 400 tokens of counting from 1 to 1,000. Stock ran before and after the recipe.

| | Stock TensorFold 1.0.12 | This recipe | Stock again |
|---|---:|---:|---:|
| Median tok/s (8 runs each) | 75.43 | **90.96** | 75.83 |
| Change | | **+20.6%** ![Measured faster](https://img.shields.io/badge/measured_faster-1a7f37?style=flat-square) | stock drift +0.5% |

Every reply was the same text in both deployments: the 24 measured and the 8 warm-ups.

<details>
<summary>What this number doesn't show</summary>

- **One workload:** one request at a time on one counting prompt. Counting is very predictable, so speculative
  decoding (drafting) does most of the work. Quote it as "+20% on this workload", not as a general speedup.
- **Not covered:** images, long prompts, concurrent requests.
- **Different settings:** the recipe also serves a 131,072-token window where stock serves 16,384, so the +20%
  can't be credited to the patches alone; More measurements compares both at the same settings.
- **Statistics:** +20.56% over the first stock run (95% interval +20.43 to +20.69) and +19.99% over the second
  (+19.87 to +20.10). The intervals are model-based and may be too narrow for runs made one after another.
- **Startup tuning:** at its first start the engine times its drafting options and saves the result; both runs
  reused a saved table. Fresh starts of an earlier build counted at 92.2 to 95.9 tok/s (7 starts).
- **Run details:** 8 runs per deployment, all complete; 32 of 32 requests (12,800 tokens) for stock and for the
  recipe.

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
of disk, and about 115 GiB of free memory at start. Building the engine also needs git,
[Zig 0.17.0](https://ziglang.org/download/) and CUDA 13's `nvcc` on `PATH` (or set `ZIG` and `NVCC`).

```bash
git clone https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold.git
cd Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --revision 82d93699acd3be9e4617c53360fe5b3f5dbeb1b3 --local-dir ~/models/nemotron-3.5-super-vl-nvfp4
./start.sh        # first run checks the checkpoint and builds the engine (~2 min)
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

At the default window the engine takes 74.03 GiB for the model and its first request, then 1.39 GiB per extra
request and a 16 GiB prompt cache; 12.2 GiB (a tenth of RAM) stays reserved. With 8 requests filled to 112k tokens
each, an earlier build kept at least 21 GiB of memory available. With less than ~115 GiB free at start the engine
keeps a smaller cache, and below ~99 GiB fewer requests.

</details>

<details>
<summary>A Spark without GitHub or the build tools</summary>

Build on any aarch64 Linux machine that has them, then copy the engine over (`spark` is the Spark's SSH host):

```bash
scripts/prepare.sh --engine-only
ssh spark mkdir -p opt/tensorfold
rsync -a ~/opt/tensorfold/tensorfold-1.0.12-* spark:opt/tensorfold/
```

On the Spark, `./start.sh` checks the copied engine against its `SHA256SUMS` and uses it as it is.

</details>

### Try it with your coding agent

Paste this into Claude Code, Codex, Cursor or whichever agent you use, on the DGX Spark:

```text
Help me serve Nemotron 3.5 Super VL with https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold
Read its README first and use only the repository's own scripts.

1. Check for an NVIDIA GB10, the hf CLI, zig 0.17.0, CUDA 13 nvcc, 70 GiB of disk and ~115 GiB of free memory.
   Stop and tell me if anything is missing.
2. Download the checkpoint at the README's pinned revision, then run ./start.sh. The first run takes a while.
3. Run tools/smoke.py and show me both replies.

Don't stop other GPU work or change settings I didn't ask for.
```

## What the patches change

Every patch keeps TensorFold's exactness contract: drafted replies equal undrafted ones, concurrent requests equal
solo runs, and resumed prompts equal fresh ones. Each patch's message has its own measurements.

<details>
<summary>The 18 patches</summary>

| # | Change | Helps |
|---|---|---|
| 1 | Latent top-k: one thread per expert | decode |
| 2 | NVFP4 experts at decode sizes: 32 columns a warp, loads in flight | decode |
| 3 | Shared-expert plans built once at load; also fixes `--segments 2` replies | decode |
| 4 | Shared expert on a side stream beside the routed experts | decode |
| 5 | Draft head over the embedded 49,152-token draft vocabulary | drafting |
| 6 | `--mtp-drafts` / `--mtp-confidence` on CUDA | opt-in |
| 7–9 | Verify windows of up to 32 rows (31 drafts a round) | drafting |
| 10 | Copied chains start at 15 tokens and grow with what they keep | repeated text |
| 11 | Each request picks the subset or full draft head from its recent tokens | drafting outside English |
| 12–16 | Prompt passes: FP8 and NVFP4 tiles, fused quantization and share sums, tiled router | prompts |
| 17–18 | TensorFold's docs and changelog for the above | docs |

</details>

<details>
<summary>Known limits</summary>

- **Images:** replies match stock's tokens, but image prompt speed was not optimized and only one image prompt was
  measured.
- **Two Sparks:** splitting the model over two GB10s is
  [upstream's setup](https://github.com/ashhart/TensorFold/blob/v1.0.12/docs/recipes/nemotron-3.5-super-vl.md#run-on-two-gb10s),
  not this recipe; the patches were checked there for identical tokens only.
- **Tests:** running the shared expert on a side stream has no byte-for-byte test of its own; end-to-end token
  identity covers it.
- **Thermal guard:** long prompts can pause while the board cools (36–56 s each on 112k-token prompts); replies
  are unchanged.
- **Builds:** the engine's bytes depend on the folder it was built in (one compiler-generated symbol), so compare
  builds by their replies' `token_sha256`.

</details>

## License

The recipe is [MIT](LICENSE). Its scripts follow
[Mia's AI Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), whose notice
`LICENSE` keeps. The patches modify TensorFold and are Apache-2.0 ([`patches/LICENSE`](patches/LICENSE),
[`patches/NOTICE`](patches/NOTICE)). The model is under NVIDIA's [OpenMDW License 1.1](https://openmdw.ai/license/1-1/)
and is downloaded, never redistributed.

[Credits](CREDITS.md) · [Patches](patches) · [Settings](scripts/config.sh) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
