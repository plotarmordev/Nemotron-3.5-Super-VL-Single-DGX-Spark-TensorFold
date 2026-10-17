# Nemotron 3.5 Super VL on one DGX Spark — TensorFold 1.0.12

Serves NVIDIA's Nemotron 3.5 Super VL 120B-A12B from its NVFP4 checkpoint with
[TensorFold](https://github.com/ashhart/TensorFold)'s native CUDA engine on one DGX Spark. Eighteen engine patches
(16 changes and their docs) make it faster; on every check below, its replies are token for token stock 1.0.12's.
The recipe's design (the start/stop/prepare scripts, the settings and the checks) follows
[Mia's AI Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold); see
[`CREDITS.md`](CREDITS.md).

- Checkpoint: [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
  at revision `82d93699acd3be9e4617c53360fe5b3f5dbeb1b3` (about 70 GiB: NVFP4 experts, FP8 Mamba, attention and head
  projections, bf16 router, latent projections and embeddings), read as stored
- Engine: TensorFold 1.0.12 built from source with the series in [`patches/`](patches) (one native binary: no
  Python, no Docker); with no `.patch` files there, the stock release archive
- Serving: OpenAI-compatible API on port 8888, text and images, up to 8 concurrent requests (streams) of 131,072
  tokens each, as a systemd user unit with TensorFold's thermal guard on
- MTP drafting: the model's multi-token-prediction head proposes tokens and the full model checks them, so replies
  do not change

One GB10 holds the whole model; splitting it over two GB10s is
[upstream's setup](https://github.com/ashhart/TensorFold/blob/v1.0.12/docs/recipes/nemotron-3.5-super-vl.md#run-on-two-gb10s),
outside this recipe.

## Start

The Spark needs the NVIDIA driver, curl, tar, python3, a systemd user manager, the Hugging Face CLI (`hf`) and about
70 GiB of disk for the checkpoint, plus room for the engine. Building the engine also needs git, zig 0.17.0 and
CUDA 13's `nvcc`. Zig comes as `zig-aarch64-linux-0.17.0.tar.xz` from [ziglang.org](https://ziglang.org/download/):
unpack it and put its `zig` on `PATH`, or set `ZIG`. `nvcc` is taken from `PATH`, else `/usr/local/cuda/bin/nvcc`,
or from `NVCC`.

Download the checkpoint into a plain folder (`MODEL_DIR`, default `~/models/nemotron-3.5-super-vl-nvfp4`):

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --revision 82d93699acd3be9e4617c53360fe5b3f5dbeb1b3 --local-dir ~/models/nemotron-3.5-super-vl-nvfp4
```

Then:

```bash
./start.sh        # prepares the engine and checks the checkpoint when needed, starts the server, runs a smoke test
tools/smoke.py    # one text and one image request (images need VISION=1): reply, tok/s, token_sha256
./stop.sh         # stops it (running requests are cut off) and confirms the GPU is empty
./start.sh restart --context 65536   # new arguments are checked before the running server is stopped
```

The first `./start.sh` runs `scripts/prepare.sh`. It checks the sha256 of the checkpoint's 111 files (all but the
README and two pictures) against [`checkpoint.sha256`](checkpoint.sha256), reading all of it in about a minute; later
starts skip that. It unpacks the stock release, checked against its `.sha256` and the digest pinned in
`scripts/config.sh`, and builds the patched engine from TensorFold's tag `v1.0.12` in about two minutes.

Logs: `journalctl --user -u nemotron-super-vl -f`. The unit outlives your login session only with lingering on
(`loginctl enable-linger`).

`token_sha256` is TensorFold's hash of a reply's token ids. On this recipe and on stock 1.0.12, `tools/smoke.py` gives
text `43d6b95020ec8af8d4be7538dde6f4891dbcd17d0494a2904bb3f4b1e046187e` and image
`42f3a0826af6d636888429cf57b5d50aad25a8e4255c7a027cd069c056f08d98`.

### A Spark without GitHub or the toolchain

Build on an aarch64 Linux machine that has them, then copy the engine folders over (`spark` stands for the Spark's
SSH host; `~/opt/tensorfold` is the default `TF_HOME`):

```bash
scripts/prepare.sh --engine-only     # the release and the patched engine, no checkpoint
ssh spark mkdir -p opt/tensorfold
rsync -a ~/opt/tensorfold/tensorfold-1.0.12-* spark:opt/tensorfold/
```

On the Spark, `prepare.sh` checks each folder against its `SHA256SUMS` and uses it as it is. A Spark with the
toolchain but without GitHub can instead build from the release archive (`TF_ARCHIVE`, its `.sha256` beside it) and a
local TensorFold clone (`TF_SRC`).

## Memory

At the default 131,072-token window the patched engine reports 74.03 GiB for the model and its first stream (stock:
74.20). From 116 GiB available it kept 27.6 GiB of room after its reserve of a tenth of RAM (12.2 GiB): 7 more
streams at 1.39 GiB each (stock: 1.34) and a 16 GiB prompt cache. At the engine's default 16,384 tokens it reports
72.93 GiB and 0.40 GiB a stream (stock: 73.16 and 0.35). A stream's caches are sized for the whole window when it
starts, so they do not grow as it fills.

An earlier build of the series (the same 1.39 GiB a stream, 0.48 GiB more for the model) ran eight concurrent
111,989-token prompts (85% of the window), all alive at once. MemAvailable went from 38.4 GiB at ready to a low of
21 GiB, about 2.4 GiB a stream. The prompt cache then held 9.5 GiB of the 15.4 GiB its own log gives as its cap. With
that cache full, about 15 GiB would stay available (an estimate: the measured low less the cache's remaining 5.9 GiB).

All 8 streams and the full cache need about 115 GiB available at start (`MIN_FREE_GIB`, below which `start.sh`
warns). With less, the engine keeps a smaller cache, and under about 99 GiB fewer streams. `start.sh` refuses to
start while any other process is on the GPU.

## Settings

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment or `.env`
(`KEY=value` lines; the environment wins), or with `tensorfold serve` flags after `./start.sh`. The checkpoint
revision and the release are pins, not settings: `MODEL_REPO`, `MODEL_REVISION`, `TF_VERSION`, `TF_SHA256` and
`TF_COMMIT` change together in `scripts/config.sh`, with `checkpoint.sha256` and `patches/`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTEXT` | `131072` | prompt + reply window per stream (the engine's default is 16,384; an earlier build of the series decoded count and prose at the same speed at both) |
| `PARALLEL` | `auto` | streams decoded together; `auto`: up to 8, fewer if they do not fit |
| `VISION` | `1` | load the RADIO image tower (~2.2 GiB) and accept image parts |
| `THINKING` | `1` | reason before answering by default; a request's `enable_thinking` wins |
| `DRAFTS` | `1` | MTP drafting; `0` passes `--no-drafts` |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | empty | `--mtp-drafts` / `--mtp-confidence`, passed only when set; empty: the engine's depth rule |
| `PROMPT_CACHE_GIB` | empty | kept prompt states; empty: 16 GiB, less if the streams leave less |
| `SERVED_NAME` | `nemotron-3.5-super-vl` | the model id in `/v1/models` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `TF_HEAT_HIGH` / `TF_HEAT_LOW` | `85` / `75` | a prompt chunk waits while the hottest thermal zone is above the high band, until it is at or under the low one (°C) |
| `MODEL_DIR` | `~/models/nemotron-3.5-super-vl-nvfp4` | the checkpoint folder |
| `TF_HOME` | `~/opt/tensorfold` | where the release is unpacked and the patched engine built |
| `TF_ARCHIVE` | empty | a local release archive instead of the GitHub download |
| `TF_REPO` / `TF_SRC` | GitHub / empty | the source the patched build clones (`TF_SRC`: a local clone) |
| `ZIG` / `NVCC` | `zig` / `nvcc` on `PATH` (`nvcc` else `/usr/local/cuda/bin/nvcc`) | the build's zig 0.17.0 and CUDA 13 `nvcc` |
| `UNIT` | `nemotron-super-vl` | the systemd user unit |
| `STOP_TIMEOUT` | `30` | seconds after SIGTERM before the server is killed (fixed in the unit at start) |
| `RUNTIME_MAX` | `infinity` | a wall-clock limit on the server's life |
| `MIN_FREE_GIB` | `115` | the memory `start.sh` wants available |
| `WAIT_TIMEOUT` | `900` | seconds `start.sh` waits for the server to answer (a new build's first start takes up to ~2 min more: it times its verify windows once) |
| `PREPARE` | `auto` | `start.sh` runs `scripts/prepare.sh` when needed; `1` always, `0` never |
| `STATE_DIR` | `~/.cache/nemotron-super-vl-spark` | locks, the source build and what `prepare.sh` last left ready |

**MTP option for count- and repetition-heavy work:** `MTP_DRAFTS=15 MTP_CONFIDENCE=0.5`. On an earlier build of the
series (warmup + 3 reps, every reply equal to stock) it gave count +10.1% (84.81 → 93.40 tok/s), prose −0.6% and
code −2.5% against the engine's own rule, so it is not the default. It has not been measured on this build.

Any `TENSORFOLD_*` variable, and the engine's own `TF_*` switches, in the environment reach the server. With
`HOST=0.0.0.0` the API is open to anyone on the network, as the server warns at start: set `TENSORFOLD_API_KEY` to
require a key, or use `HOST=127.0.0.1` to keep the API on this machine. `start.sh` reads the key from the environment
or `.env`; `tools/smoke.py` and `tools/bench.py` read it only from the environment, so export it for them.

## What differs from stock

`patches/` holds the engine changes as a `git format-patch` series against TensorFold tag `v1.0.12`, applied in
file-name order with `git am`; `scripts/prepare.sh` builds it into `TF_HOME/tensorfold-1.0.12-patched-<series
hash>/`. A build's bytes depend on the folder it was built in (nvcc names one GLM kernel's symbols after the source
path), so compare builds by their source tree and their replies' `token_sha256`. Every change keeps TensorFold's
exactness contract: drafted replies equal `"draft": false` ones, concurrent streams equal solo runs, and resumed
prompts equal fresh ones. Each patch's message gives the effect measured when it was made; [Measured](#measured)
gives the whole series against stock.

| # | Change | Speeds up |
| --- | --- | --- |
| 1 | The latent top-k ranks every expert on a thread of its own | decode |
| 2 | The checkpoint's NVFP4 experts at decode-sized calls: 32 columns a warp, four steps of loads in flight | decode |
| 3 | A decode window's latent shared-expert plans are built once, at load; a prompt segment's buffers share the owner's | decode; also a fix: on an 8,186-token prompt `--segments 2` now gives `--segments 1`'s reply, which 1.0.12's did not |
| 4 | On one GPU the latent shared expert runs on a side stream beside the routed experts | decode |
| 5 | The MTP head drafts over the embedded draft vocabulary's FP8 output rows | drafting |
| 6 | `--mtp-drafts` and `--mtp-confidence` set a lone stream's MTP depth | nothing unless set (flags only) |
| 7 | The Mamba scan steps kept and window rows in stages of 32 | nothing alone; needed by 8 |
| 8 | Super verifies windows of up to 32 rows (31 MTP drafts a round) | drafting |
| 9 | Verify windows of 17 to 32 rows take the decode-sized NVFP4 experts | drafting |
| 10 | A lone stream's copied and Recall chains start at 15 tokens and grow with what copies keep | drafting on repeated text |
| 11 | A stream drafts on the 49,152-token subset head while its last 16 context tokens (the prompt's, until 16 are emitted) are draft ids, else on the whole head | drafting, most of all outside English |
| 12 | Prompt passes run the FP8 layers in 128-row tiles | prompts |
| 13 | Prompt passes run the NVFP4 experts in 128-pair by 128-column tiles | prompts |
| 14 | Prompt up kernels write down's NVFP4 rows | prompts |
| 15 | One GPU's prompt down kernels sum the expert width shares into bf16 | prompts |
| 16 | Router logits of passes over 32 rows run on 128-row by 128-expert register tiles | prompts |
| 17 | Docs: Super VL's memory and speed on one GB10 against 1.0.12 | nothing (docs) |
| 18 | CHANGELOG: the decode, `--mtp-*`, prompt and memory changes | nothing (docs) |

The recipe also serves with other settings than stock's:

| Setting | Here | Stock 1.0.12 |
| --- | --- | --- |
| Window | `--context 131072`, up to 8 streams (see Memory) | 16,384 tokens |
| Images | `--vision` | off (upstream's Super VL recipe turns it on) |
| Address and name | `0.0.0.0:8888`, `nemotron-3.5-super-vl` | `127.0.0.1:8080`, the checkpoint folder's name |
| Thermal guard | `TF_HEAT_HIGH=85 TF_HEAT_LOW=75`: replies unchanged, long prompts can wait (36–56 s each on the 111,989-token prompts in Memory) | none |

## Measured

Paired on one DGX Spark: TensorFold 1.0.12 built from source with the same toolchain and build commands, without
patches (stock), against this recipe's build of the series (patched; patches 17–18, docs only, were reworded after
the measurements). Both served as the recipe serves them (`--vision --context 131072`, up to 8 streams, a 16 GiB
prompt cache, thermal guard 85/75). For each set of measures stock ran first, then the patched build, which booted
fresh: its cached window costs (below) were deleted first. Every request was greedy, thinking off and streamed. Each
prompt had a warmup and 3 measured requests; the table gives the medians of the server's decode tok/s. Every reply's
token hash equals stock's. Count, multilingual and the six language prompts were also sent with `"draft": false` and
gave the same hashes. Each prompt's replies were the same across warmup and reps, and concurrent streams equal their
solo runs.

| Measure | Stock | Patched | Change |
| --- | ---: | ---: | ---: |
| count, drafted | 75.86 | 92.72 | +22.2% |
| prose, drafted | 29.28 | 34.52 | +17.9% |
| code, drafted | 43.91 | 51.42 | +17.1% |
| image, drafted | 59.88 | 71.88 | +20.0% |
| multilingual, drafted | 37.94 | 42.27 | +11.4% |
| en / es / fr / de / it / ja, drafted | 29.73 / 31.62 / 29.22 / 29.44 / 33.21 / 38.50 | 35.22 / 35.16 / 33.34 / 33.11 / 36.83 / 43.50 | +18.5 / +11.2 / +14.1 / +12.5 / +10.9 / +13.0% |
| count, `"draft": false` | 18.49 | 21.44 | +16.0% |
| 1 / 2 / 4 / 8 streams, aggregate | 43.93 / 41.05 / 57.10 / 68.69 | 51.46 / 47.32 / 64.97 / 87.57 | +17.1 / +15.3 / +13.8 / +27.5% |
| cold TTFT, 2,048 / 8,192 / 16,384 tokens (s) | 1.216 / 4.447 / 13.254 | 0.905 / 3.078 / 8.958 | −25.6 / −30.8 / −32.4% |

- Prompts: count is counting from 1 to 1,000 (400 tokens); prose a short story; code a Python function with tests;
  image a description of a generated picture; multilingual a translation into five languages; en to ja the same
  essay request in six languages. Replies are 256 tokens unless noted, `ignore_eos`.
- Window costs: at its first start, the engine times each verify-window width on this GPU and caches the table in
  `~/.cache/tensorfold/window-costs/`. The table sets how many drafts a round tries, so count's speed depends on it.
  This boot drafted count in 31 rounds, as stock did. Fresh boots of an earlier build drafted it in 30 rounds in 6 of
  7 boots (95.6–95.9 tok/s) and in 31 once (92.2).
- Streams: n requests of the short prompts (code, prose, reasoning, repetition, count, multilingual, tool_json,
  gpu_explain, the first n) sent at once; a warmup and 2 measured runs. Aggregate tok/s is every stream's tokens
  after its first over the span from the first streamed token of any stream to the last.
- Cold TTFT: separate servers at `--context 18432 --parallel 1 --segments 1 --prompt-cache-gib 0` without the image
  tower. Each request was a 32-token `"draft": false` completion of a prefix of a synthetic field log, with nothing
  cached; a warmup, then 3 per size, the host cooled to 75 °C or less before each. The thermal guard still paused
  every 16,384-token request inside its prompt (stock 6 s, patched 4 s a request), and that row includes the waits;
  the 2,048- and 8,192-token requests did not wait.
- On a second DGX Spark, a broader harness compared a build of the same source tree with the stock release (its
  bytes differ from this build's only in the kernel symbol nvcc names after the source path). No reply differed: 10
  greedy prompts (one with an image), drafted and `"draft": false`, seeded sampled prose and code, 1/2/4/8 streams and
  resumed prompts.

[`tools/bench.py`](tools/bench.py) reruns these measures against the running server with the same prompts
([`tools/prompts.py`](tools/prompts.py)): `decode` for the drafted rows and count's `"draft": false` (`decode count`
for one prompt), `streams`, and `ttft` on a server restarted with `VISION=0 ./start.sh restart --context 18432
--parallel 1 --segments 1 --prompt-cache-gib 0`. It prints each reply's `token_sha256`, so two builds can be compared
hash for hash. The recipe does not build the stock column's engine.

### Grill verdict

Measured with [the Grill](https://github.com/plotarmordev/thegrill) 0.6.0, which compares whole deployments; its
collector (the measuring client) was built from source `8ab41f34808825566c46c2d5cd52197a565a29ee` and has sha256
`4e39540f8f9e19f682a6bbfb7d34276994988842094d92b84a3d8db0b112e6fa`. Workload `portable-v1`: one chat request at a
time, greedy, thinking off, 400 tokens of counting from 1 to 1,000; throughput includes the prompt. One DGX Spark,
client on the same host, in the order stock (A), this recipe (B), stock again (A2), 8 runs each, all complete; A and
B each finished 32 of 32 requests (12,800 tokens).

| Deployment | Median tok/s | Range of 8 runs |
| --- | ---: | --- |
| A: stock 1.0.12 release, `--name super --vision` | 75.43 | 75.4 – 75.6 |
| B: this recipe (`./start.sh`, defaults) | **90.96** | 90.9 – 91.1 |
| A2: stock again | 75.83 | 75.7 – 75.9 |

Verdict **measured faster**: B is +20.56% over A (95% confidence interval +20.43 to +20.69) and +19.99% over A2
(+19.87 to +20.10); stock drifted +0.48% (+0.36 to +0.59) between A and A2. The Grill states its limits: the
intervals are model-based and may be too narrow for sequential runs; a whole-deployment comparison does not attribute
the gain to one factor (B also serves the recipe's 131,072-token window); and the counting prompt is so predictable
that drafting dominates it. Every reply in A and B, the 24 measured and the 8 warm-ups, was the same text. Both
deployments loaded window costs cached by earlier boots; B's table drafts count in 31 rounds. This is single-request
throughput on one counting prompt; it does not exercise images, long prompts or concurrency (see Measured above).

## License

MIT, see [`LICENSE`](LICENSE) (with Mia's AI Lab's notice). The patches in `patches/` modify TensorFold and are
Apache-2.0: see [`patches/LICENSE`](patches/LICENSE) and [`patches/NOTICE`](patches/NOTICE) (TensorFold code from
before 0.6.0 keeps its MIT notice, [`patches/LICENSES/MIT.txt`](patches/LICENSES/MIT.txt)). The model weights are
under NVIDIA's [OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/).
