#!/usr/bin/env bash
# Prepare everything needed to serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 with TensorFold on one
# DGX Spark:
#   1. preflight checks (tools, GPU driver)
#   2. the stock TensorFold $TF_VERSION release: downloaded from GitHub, or copied from TF_ARCHIVE on a node without
#      GitHub access, checked against its .sha256 and TF_SHA256, unpacked into $TF_HOME
#   3. only with patches/*.patch (patches/PATCHES.md): TensorFold's source at tag v$TF_VERSION (from TF_REPO, or a
#      local clone in TF_SRC, checked against TF_COMMIT), the series applied with git am, built with zig
#      $ZIG_VERSION (ZIG, or the pinned release fetched into $TF_HOME and checked against ZIG_SHA256) and CUDA 13's
#      nvcc for GB10 (sm_121) into the release's folder layout beside it, named by the series' hash; its BUILD file
#      records the commit, the source tree, the series, the binary's sha256 and the toolchain. A folder built
#      elsewhere and copied into TF_HOME is used as it is, once it matches its SHA256SUMS: the node then needs neither
#      the toolchain nor the source.
#   4. the checkpoint in MODEL_DIR: every file's sha256 against ./checkpoint.sha256 (~70 GiB read, about a minute),
#      then `tensorfold info`
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Usage: scripts/prepare.sh [--skip-hash] [--rebuild] [--engine-only]
#   --skip-hash     only check that the checkpoint's files are there, without reading them
#   --rebuild       build the patched engine again even if this series was built before
#   --engine-only   stop after step 3: build the engine on a machine without the checkpoint, to copy it to a Spark
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh

SKIP_HASH=0; REBUILD=0; ENGINE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-hash) SKIP_HASH=1 ;;
    --rebuild) REBUILD=1 ;;
    --engine-only) ENGINE_ONLY=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

mapfile -t PATCHES < <(patch_files)
mkdir -p "$STATE_DIR" "$TF_HOME"
exec 9>"$STATE_DIR/.prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it: pgrep -af prepare.sh"

# ---------------------------------------------------------------- 1. preflight
log "Preflight checks"
[[ "$(uname -m)" == aarch64 ]] || die "this recipe runs the linux-aarch64 release (a DGX Spark); this machine is $(uname -m)"
for tool in curl sha256sum tar python3 systemd-run; do
  command -v "$tool" >/dev/null || die "$tool is not installed"
done
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found: start.sh needs it"
elif ! nvidia-smi -L; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi

# An unpacked engine folder is intact when every file matches its SHA256SUMS.
intact() { [[ -f "$1/SHA256SUMS" ]] && (cd "$1" && sha256sum -c --quiet --strict SHA256SUMS >/dev/null 2>&1); }
tmp_dirs=()
trap 'rm -rf -- "${tmp_dirs[@]}"' EXIT

# ---------------------------------------------------------------- 2. stock release
ARCHIVE="tensorfold-$TF_VERSION-linux-aarch64.tar.gz"
if intact "$STOCK_DIR"; then
  log "TensorFold $TF_VERSION release already in $STOCK_DIR"
else
  if [[ ! -f "$TF_HOME/$ARCHIVE" || ! -f "$TF_HOME/$ARCHIVE.sha256" ]]; then
    if [[ -n "$TF_ARCHIVE" ]]; then
      [[ -f "$TF_ARCHIVE" && -f "$TF_ARCHIVE.sha256" ]] || die "TF_ARCHIVE=$TF_ARCHIVE: need the archive and its .sha256 beside it"
      log "Copying $TF_ARCHIVE"
      cp -- "$TF_ARCHIVE" "$TF_HOME/$ARCHIVE"
      cp -- "$TF_ARCHIVE.sha256" "$TF_HOME/$ARCHIVE.sha256"
    else
      url="${TF_REPO%.git}/releases/download/v$TF_VERSION/$ARCHIVE"
      log "Downloading $url"
      { curl -fL --retry 3 -o "$TF_HOME/$ARCHIVE.sha256" "$url.sha256" &&
          curl -fL --retry 3 -o "$TF_HOME/$ARCHIVE" "$url"; } || {
        rm -f -- "$TF_HOME/$ARCHIVE" "$TF_HOME/$ARCHIVE.sha256"
        die "download failed; without GitHub access, copy the archive and its .sha256 to this node and set TF_ARCHIVE"
      }
    fi
  fi
  if ! (cd "$TF_HOME" && sha256sum -c --quiet --strict "$ARCHIVE.sha256"); then
    rm -f -- "$TF_HOME/$ARCHIVE" "$TF_HOME/$ARCHIVE.sha256"
    die "$ARCHIVE does not match its .sha256: removed it from $TF_HOME; run scripts/prepare.sh again to fetch it again"
  fi
  sum=$(cut -d' ' -f1 "$TF_HOME/$ARCHIVE.sha256")
  [[ "$sum" == "$TF_SHA256" ]] || die "$ARCHIVE is $sum, not the pinned TF_SHA256 $TF_SHA256"
  unpack=$(mktemp -d "$TF_HOME/.unpack.XXXXXX"); tmp_dirs+=("$unpack")
  tar -xzf "$TF_HOME/$ARCHIVE" -C "$unpack"
  intact "$unpack/${STOCK_DIR##*/}" || die "the unpacked $ARCHIVE does not match its own SHA256SUMS"
  rm -rf -- "$STOCK_DIR"
  mv -- "$unpack/${STOCK_DIR##*/}" "$STOCK_DIR"
  log "Unpacked TensorFold $TF_VERSION ($sum) into $STOCK_DIR"
fi

# ---------------------------------------------------------------- 3. patched engine
if (( ${#PATCHES[@]} == 0 )); then
  log "No patches/*.patch: serving the stock release"
elif [[ $REBUILD -eq 0 ]] && intact "$ENGINE_DIR"; then
  log "Patched engine already built in $ENGINE_DIR"
else
  command -v git >/dev/null || die "git is not installed (needed to apply patches/)"
  if [[ -z "$ZIG" ]]; then
    zig_name="zig-aarch64-linux-$ZIG_VERSION"
    ZIG="$TF_HOME/$zig_name/zig"
    if [[ ! -x "$ZIG" ]]; then
      command -v xz >/dev/null || die "xz is not installed (needed to unpack zig)"
      url="https://ziglang.org/download/$ZIG_VERSION/$zig_name.tar.xz"
      log "Downloading $url"
      unpack=$(mktemp -d "$TF_HOME/.unpack.XXXXXX"); tmp_dirs+=("$unpack")
      curl -fL --retry 3 -o "$unpack/$zig_name.tar.xz" "$url" ||
        die "download failed; without access to ziglang.org, set ZIG to a zig $ZIG_VERSION"
      sum=$(sha256sum "$unpack/$zig_name.tar.xz" | cut -d' ' -f1)
      [[ "$sum" == "$ZIG_SHA256" ]] || die "$zig_name.tar.xz is $sum, not the pinned ZIG_SHA256 $ZIG_SHA256"
      tar -xJf "$unpack/$zig_name.tar.xz" -C "$unpack"
      rm -rf -- "${ZIG%/zig}"
      mv -- "$unpack/$zig_name" "${ZIG%/zig}"
    fi
  fi
  [[ -x "$ZIG" ]] || die "ZIG=$ZIG is not an executable"
  [[ -x "$NVCC" ]] || die "nvcc not found at $NVCC: put CUDA 13's nvcc on PATH or set NVCC"
  zig_version=$("$ZIG" version)
  nvcc_version=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
  [[ "$zig_version" == "$ZIG_VERSION" ]] || die "$ZIG is zig $zig_version; the build needs zig $ZIG_VERSION (set ZIG)"
  [[ "$nvcc_version" == 13.* ]] || die "$NVCC is CUDA ${nvcc_version:-unknown}; the build needs CUDA 13's nvcc (set NVCC)"
  PATCHES_HASH=$(patches_hash)
  log "Building TensorFold $TF_VERSION with ${#PATCHES[@]} patch(es), series $PATCHES_HASH"
  build="$STATE_DIR/build"
  rm -rf -- "$build"
  # file:// so a local clone can be shallow too
  src=$TF_REPO; [[ -n "$TF_SRC" ]] && src="file://$(readlink -f "$TF_SRC")"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "v$TF_VERSION" "$src" "$build/src"
  head=$(git -C "$build/src" rev-parse HEAD)
  [[ "$head" == "$TF_COMMIT" ]] || die "tag v$TF_VERSION in $src is $head, not the pinned TF_COMMIT $TF_COMMIT"
  git -C "$build/src" -c user.name=nemotron-super-vl-spark -c user.email=recipe@localhost am --quiet "${PATCHES[@]/#/$PWD/}" ||
    die "the series does not apply to v$TF_VERSION (left in $build/src for git am --show-current-patch)"
  # A glibc 2.28 baseline, GB10's sm_121 only, and the release's optimization and stripping.
  # nvcc names one kernel's symbols after the source path, so the build always runs in this same folder: the same
  # series gives the same binary sha256 on this machine.
  (
    cd "$build/src"
    "$ZIG" build fatbins -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline \
      -Dnvcc="$NVCC" -Dsm=121 --prefix "$build/out" -j1
    "$ZIG" build native -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline \
      -Doptimize=fast -Dstrip=true -Dversion="$TF_VERSION" \
      -Dfatbins="$build/out/fatbin" --prefix "$build/out" -j1
  )
  # The release's folder (grammar library, notices) with the built binary in place of its own.
  stage=$(mktemp -d "$TF_HOME/.patched.XXXXXX"); tmp_dirs+=("$stage")
  cp -a -- "$STOCK_DIR/." "$stage/"
  install -m 755 "$build/out/native/bin/tensorfold-native" "$stage/bin/tensorfold-native"
  binary_sha=$(sha256sum "$stage/bin/tensorfold-native" | cut -d' ' -f1)
  {
    echo "tensorfold=$TF_VERSION commit=$TF_COMMIT tree=$(git -C "$build/src" rev-parse 'HEAD^{tree}')"
    echo "patches=$PATCHES_HASH ${PATCHES[*]##*/}"
    echo "binary_sha256=$binary_sha"
    echo "zig=$zig_version nvcc=$nvcc_version"
  } > "$stage/BUILD"
  (cd "$stage" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > "$stage/SHA256SUMS")
  rm -rf -- "$ENGINE_DIR"
  mv -- "$stage" "$ENGINE_DIR"
  chmod 755 "$ENGINE_DIR"
  log "Built $ENGINE_DIR: binary $binary_sha, patches $PATCHES_HASH"
fi
"$TF_BIN" --version
if [[ $ENGINE_ONLY -eq 1 ]]; then
  log "Engine ready in $ENGINE_DIR (--engine-only: the checkpoint is not checked)"
  exit 0
fi

# ---------------------------------------------------------------- 4. checkpoint
[[ -f "$MODEL_DIR/config.json" ]] ||
  die "no checkpoint in $MODEL_DIR (config.json missing); fetch it with: hf download $MODEL_REPO --revision $MODEL_REVISION --local-dir $MODEL_DIR"
files=$(wc -l < checkpoint.sha256)
if [[ $SKIP_HASH -eq 1 ]]; then
  missing=$(cut -d' ' -f3- checkpoint.sha256 | while IFS= read -r f; do [[ -f "$MODEL_DIR/$f" ]] || echo "$f"; done)
  [[ -z "$missing" ]] || die "missing from $MODEL_DIR: $(echo "$missing" | paste -sd' ')"
  log "Checkpoint: all $files files present in $MODEL_DIR (--skip-hash: not hashed)"
else
  log "Hashing the checkpoint's $files files ($(du -shL "$MODEL_DIR" | cut -f1)) against checkpoint.sha256"
  rm -f -- "$CHECKPOINT_STAMP"   # a copy that fails is not verified, whatever an earlier pass found
  (cd "$MODEL_DIR" && sha256sum -c --quiet --strict "$OLDPWD/checkpoint.sha256") ||
    die "$MODEL_DIR differs from $MODEL_REPO@$MODEL_REVISION (see above)"
  checkpoint_id > "$CHECKPOINT_STAMP"
  log "Checkpoint: all $files files match $MODEL_REPO@$MODEL_REVISION"
fi
"$TF_BIN" info "$MODEL_DIR"

prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."
