# Carried patches

The recipe serves TensorFold [v1.1.0](https://github.com/ashhart/TensorFold/releases/tag/v1.1.0)
(`76aafcf274e60a5e31a59ce145b2f71f7b366e9d`) with these patches, applied in order by `git am`
([`scripts/prepare.sh`](../scripts/prepare.sh)). Each is the branch of an upstream pull request, rebased on v1.1.0
with `git format-patch`; the recipe drops it when a TensorFold release ships it.

| Patch | What it does | Upstream | Status | Verification |
|---|---|---|---|---|
| [0001](0001-nemotron-cuda-a-prompt-pass-with-images-runs-the-RAD.patch) | A prompt pass with images runs the RADIO tower once: v1.1.0 calls `Vision.begin` twice, so every image was encoded twice before the first chunk. | [#1057](https://github.com/ashhart/TensorFold/pull/1057) at `e2357b58` | Open | The tower runs once per image in a served image prompt; image replies token for token the same as stock v1.1.0's |
| [0002](0002-nemotron-cuda-shared-windows-of-several-sequences-ru.patch) | Shared windows of several streams run the latent shared expert on the side stream again, beside the routed experts, as a lone stream already does. | [#1058](https://github.com/ashhart/TensorFold/pull/1058) at `33d0db58` | Open | `tf-cuda-test shared-fork`: lone and shared windows of 2, 4 and 8 streams byte-equal to one stream; replies at 1 to 8 requests at once the same as stock v1.1.0's |

`LICENSE`, `NOTICE` and `LICENSES/MIT.txt` are copied from v1.1.0; the NOTICE adds one line naming the changes.

## Rules

- **In:** a patch is our upstream pull request's branch rebased cleanly on the pinned commit, with that pull request
  linked here. Before a patched build replaces the one serving, TensorFold's host suites pass on the patched tree
  (`zig build test`, `test-golden`, `fuzz-cortheon`, `test-cuda-host` and `test-bolt-host`), and so do the GPU checks
  the patch names (the Verification column), and its replies are token for token the same as the stock release's
  (`tools/bench.py decode` hashes).
- **Out:** when a TensorFold release ships the change, the recipe moves to that release and deletes the patch in the
  same commit (a patch upstream already carries is never applied twice), measures again and updates the README.
- **Build:** `scripts/prepare.sh` builds the series into `tensorfold-$TF_VERSION-patched-<series hash>` beside the
  stock release; its `BUILD` file records the commit, source tree, series, binary sha256 and toolchain. The binary's
  bytes depend on the folder it was built in, so compare builds by their replies' `token_sha256`.
