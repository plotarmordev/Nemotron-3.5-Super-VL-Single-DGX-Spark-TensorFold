# Credits

This repository is scripts and patches. Almost everything that makes it work was built by others.

## Recipe design

- **[Mia's AI Lab](https://github.com/MiaAI-Lab)**: the recipe's design, from her
  [Qwen3.8 Flash Next single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):
  the `start.sh` / `stop.sh` / `scripts/prepare.sh` split, settings in `scripts/config.sh` overridable from the
  environment or `.env` (its parser is hers), the prepared-state marker that reruns preparation only when something
  changed, the restart that checks new arguments before stopping anything, the load heartbeat, the smoke test and
  the README's layout. The MIT notice of her work is kept in [`LICENSE`](LICENSE).

## Model

- **[NVIDIA Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)** by
  [NVIDIA](https://www.nvidia.com/): the model's design, training and evaluations, its RADIO vision tower and the
  NVFP4 checkpoint served here (ModelOpt export). Released under the
  **[OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/)**, which governs any use of the weights. The
  weights are not part of this repository; `checkpoint.sha256` only lists their files' digests.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and
  the TensorFold contributors (Apache License 2.0 from v0.6.0; earlier code keeps its MIT notice): the native CUDA
  engine for Nemotron 3.5 Super VL, including the NVFP4 and FP8 kernels, the latent MoE, MTP drafting with exact
  verification, the RADIO image path, the thermal guard and the OpenAI-compatible server.
- The release archive carries TensorFold's own third-party notices (`THIRD_PARTY_NOTICES.md` and `LICENSES/`), among
  them [xgrammar](https://github.com/mlc-ai/xgrammar) (Apache 2.0) with dlpack (Apache 2.0) and picojson (BSD-2-Clause)
  in `lib/libtfgrammar.so`.

## Engine changes

- The Nemotron 3.5 Super CUDA changes this recipe first carried as patches are by plotarmordev and shipped in
  TensorFold 1.1.0 ([#996](https://github.com/ashhart/TensorFold/pull/996),
  [#997](https://github.com/ashhart/TensorFold/pull/997), [#999](https://github.com/ashhart/TensorFold/pull/999),
  [#1002](https://github.com/ashhart/TensorFold/pull/1002), [#1006](https://github.com/ashhart/TensorFold/pull/1006),
  [#1008](https://github.com/ashhart/TensorFold/pull/1008), [#1011](https://github.com/ashhart/TensorFold/pull/1011)),
  with the maintainer's follow-ups, under TensorFold's Apache 2.0 license.
- The `.patch` files in `patches/` are by plotarmordev: changes to TensorFold v1.1.0's Nemotron CUDA engine, each
  also an upstream pull request ([`patches/PATCHES.md`](patches/PATCHES.md)), under TensorFold's Apache 2.0 license.

## Toolchain and runtime

- **[Zig](https://ziglang.org/)** (MIT) 0.17.0 and **[CUDA](https://developer.nvidia.com/cuda-toolkit)** 13's
  `nvcc` (NVIDIA's license terms): building the patched engine from source.
- **[systemd](https://systemd.io/)** (LGPL 2.1): the user unit the server runs in.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory): the machine this recipe targets.

If you believe something here is missing or credited wrongly, please say so.
