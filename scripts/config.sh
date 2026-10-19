# Shared settings for start.sh, stop.sh and scripts/*.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh`, or set in ./.env: KEY=value lines, read here (never run as a script); a variable
# already set in the environment wins over the file. .env is yours, not the repository's.
# shellcheck shell=bash disable=SC2034   # the scripts that source this file use its settings
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi

# Pins, not settings: what checkpoint.sha256, the engine release and the patch series are made for. Change them
# together, here.
MODEL_REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
MODEL_REVISION=82d93699acd3be9e4617c53360fe5b3f5dbeb1b3
TF_VERSION=1.1.0
TF_SHA256=8f18033fca9df0c56c19afb2f11b55175151ff886c2603b01c0390bf48773e59   # the release archive
TF_COMMIT=76aafcf274e60a5e31a59ce145b2f71f7b366e9d                           # what tag v$TF_VERSION must point at
ZIG_VERSION=0.17.0
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8   # zig-aarch64-linux-$ZIG_VERSION.tar.xz

# The checkpoint: a plain folder (e.g. `hf download ... --local-dir`), checked against ./checkpoint.sha256, which
# lists the sha256 of every file the revision has except its README and pictures.
MODEL_DIR="${MODEL_DIR:-$HOME/models/nemotron-3.5-super-vl-nvfp4}"

# The engine: TensorFold's native release (one binary, no Python or Docker), unpacked into TF_HOME. With
# patches/*.patch the same tag is built from source with them, into its own folder beside the release.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_HOME="${TF_HOME:-$HOME/opt/tensorfold}"
TF_ARCHIVE="${TF_ARCHIVE:-}"   # a local copy of the release archive (its .sha256 beside it), for nodes without GitHub
TF_SRC="${TF_SRC:-}"           # a local TensorFold clone to take tag v$TF_VERSION from instead of TF_REPO
ZIG="${ZIG:-}"                 # a zig $ZIG_VERSION to build with; empty: the pinned release, fetched into TF_HOME
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"           # CUDA 13
# The recipe's own TF_* names. Every other TF_* and TENSORFOLD_* variable in the environment is the engine's, and
# start.sh passes it to the server.
RECIPE_TF_VARS=(TF_VERSION TF_SHA256 TF_COMMIT TF_REPO TF_HOME TF_ARCHIVE TF_SRC)

SERVED_NAME="${SERVED_NAME:-nemotron-3.5-super-vl}"   # the model id clients see in /v1/models (tensorfold --name)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Serving defaults (./start.sh arguments come after them and win): a 131,072-token window, streams as many as fit
# up to 8, images on, and the engine's own MTP depth rule (no --mtp-* flags). See README, Memory and Measured.
CONTEXT="${CONTEXT:-131072}"           # prompt + reply window per stream (engine default 16,384)
PARALLEL="${PARALLEL:-auto}"           # streams; auto: up to 8, fewer if they do not fit
VISION="${VISION:-1}"                  # 1: load the RADIO image tower (~2.2 GiB more) and accept image parts
THINKING="${THINKING:-1}"              # 1: reason before answering by default; a request's enable_thinking wins
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"   # kept prompt states; empty: 16 GiB, less if the streams leave less
# MTP drafting, with exact verification. DRAFTS=0 serves without it (--no-drafts). MTP_DRAFTS (drafts a round) and
# MTP_CONFIDENCE (a chain stops before a draft under it) are passed only when set.
# MTP_DRAFTS=15 MTP_CONFIDENCE=0.5 suited count- and repetition-heavy work on the patched 1.0.12; not yet measured on
# 1.1.0 (README, Settings).
DRAFTS="${DRAFTS:-1}"
MTP_DRAFTS="${MTP_DRAFTS:-}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-}"

# Each prompt chunk waits while the hottest thermal zone is above TF_HEAT_HIGH, until it is at or under TF_HEAT_LOW
# (degrees Celsius); the reply's tokens do not change. The engine reads them, so they are exported.
export TF_HEAT_HIGH="${TF_HEAT_HIGH:-85}"
export TF_HEAT_LOW="${TF_HEAT_LOW:-75}"

UNIT="${UNIT:-nemotron-super-vl}"         # the server's systemd user unit
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"        # seconds the server gets after SIGTERM before SIGKILL (set at start)
RUNTIME_MAX="${RUNTIME_MAX:-infinity}"    # wall-clock limit on the server's life (systemd RuntimeMaxSec)
# Memory available at start known to be enough for all 8 streams and the full prompt cache (starts from 116 and 109
# GiB both kept them); see README, Memory.
MIN_FREE_GIB="${MIN_FREE_GIB:-115}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/nemotron-super-vl-spark}"   # locks, the source build, what prepare.sh left

# Colours only on a terminal.
_c() { if [[ -t "$1" ]]; then printf '\033[%sm' "$2"; fi; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

# The engine patch series, in the order git am applies it (see patches/PATCHES.md).
patch_files() { local p; for p in patches/*.patch; do [[ -e "$p" ]] && echo "$p"; done; return 0; }
patches_hash() { patch_files | xargs -r cat | sha256sum | cut -c1-12; }

STOCK_DIR="$TF_HOME/tensorfold-$TF_VERSION-linux-aarch64"
if [[ -n "$(patch_files)" ]]; then ENGINE_DIR="$TF_HOME/tensorfold-$TF_VERSION-patched-$(patches_hash)"; else ENGINE_DIR="$STOCK_DIR"; fi
TF_BIN="$ENGINE_DIR/bin/tensorfold-native"

# Where to reach a server bound to HOST:PORT from this machine: a wildcard bind answers on loopback.
api_url() {
  local host=$HOST
  [[ "$host" == 0.0.0.0 || "$host" == "::" ]] && host=127.0.0.1
  [[ "$host" == *:* ]] && host="[$host]"
  echo "http://$host:$PORT"
}
# The running unit's address (start.sh ends its description with it) and its first API key, if it has one: the
# server's own, whatever this shell's settings say now.
unit_url() { local d; d=$(systemctl --user show -p Description --value "$UNIT"); echo "${d##* }"; }
unit_key() { systemctl --user show -p Environment --value "$UNIT" | tr ' ' '\n' | sed -n 's/^TENSORFOLD_API_KEY=//p' | cut -d, -f1; }
# curl to the running unit's API with its key, read from stdin so that it never shows on a command line.
api() {
  local key; key=$(unit_key)
  { [[ -z "$key" ]] || printf 'header = "Authorization: Bearer %s"\n' "$key"; } | curl -s -K - "$@"
}

# The PIDs of every compute process on the GPU (one GB10, one at a time).
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '; }

# What scripts/prepare.sh last left ready (it writes this line to PREPARED_MARKER when it succeeds); start.sh runs
# prepare.sh again whenever the current line differs: another release, new patches, another checkpoint.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local bin=missing model=missing
  [[ -x "$TF_BIN" ]] && bin=$(sha256sum "$TF_BIN" | cut -c1-12)
  [[ -f "$MODEL_DIR/config.json" ]] && model=present
  echo "engine=$ENGINE_DIR($bin) model=$MODEL_DIR@$MODEL_REVISION($model)"
}
# A full hash pass of MODEL_DIR against this manifest leaves this line in CHECKPOINT_STAMP.
CHECKPOINT_STAMP="$STATE_DIR/checkpoint-verified"
checkpoint_id() { echo "$MODEL_DIR $(sha256sum < checkpoint.sha256 | cut -c1-12)"; }
