#!/usr/bin/env bash
# Prepare everything needed to serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 with TensorFold on one
# DGX Spark:
#   1. preflight checks (tools, GPU driver)
#   2. the TensorFold $TF_VERSION release: downloaded from GitHub, or copied from TF_ARCHIVE on a node without GitHub
#      access, checked against its .sha256 and TF_SHA256, unpacked into $TF_HOME
#   3. the checkpoint in MODEL_DIR: every file's sha256 against ./checkpoint.sha256 (~70 GiB read, about a minute),
#      then `tensorfold info`
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Usage: scripts/prepare.sh [--skip-hash]
#   --skip-hash     only check that the checkpoint's files are there, without reading them
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh

SKIP_HASH=0
for arg in "$@"; do
  case "$arg" in
    --skip-hash) SKIP_HASH=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

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

# ---------------------------------------------------------------- 2. release
ARCHIVE="tensorfold-$TF_VERSION-linux-aarch64.tar.gz"
if intact "$ENGINE_DIR"; then
  log "TensorFold $TF_VERSION release already in $ENGINE_DIR"
else
  if [[ ! -f "$TF_HOME/$ARCHIVE" || ! -f "$TF_HOME/$ARCHIVE.sha256" ]]; then
    if [[ -n "$TF_ARCHIVE" ]]; then
      [[ -f "$TF_ARCHIVE" && -f "$TF_ARCHIVE.sha256" ]] || die "TF_ARCHIVE=$TF_ARCHIVE: need the archive and its .sha256 beside it"
      log "Copying $TF_ARCHIVE"
      cp -- "$TF_ARCHIVE" "$TF_HOME/$ARCHIVE"
      cp -- "$TF_ARCHIVE.sha256" "$TF_HOME/$ARCHIVE.sha256"
    else
      url="https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION/$ARCHIVE"
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
  intact "$unpack/${ENGINE_DIR##*/}" || die "the unpacked $ARCHIVE does not match its own SHA256SUMS"
  rm -rf -- "$ENGINE_DIR"
  mv -- "$unpack/${ENGINE_DIR##*/}" "$ENGINE_DIR"
  log "Unpacked TensorFold $TF_VERSION ($sum) into $ENGINE_DIR"
fi
"$TF_BIN" --version

# ---------------------------------------------------------------- 3. checkpoint
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
