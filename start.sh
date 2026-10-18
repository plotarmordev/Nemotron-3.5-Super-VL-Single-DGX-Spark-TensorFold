#!/usr/bin/env bash
# Serve NVIDIA Nemotron 3.5 Super VL (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) with TensorFold on one
# DGX Spark, end to end: runs scripts/prepare.sh when the engine or the checkpoint is not ready yet (first run, or
# after settings change), launches `tensorfold-native serve` as the systemd user unit $UNIT, waits until
# the OpenAI API answers, then runs a smoke test. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold serve args]
#   ./start.sh                         # scripts/config.sh defaults: 131,072-token window, up to 8 streams, images on
#                                      # (if the server already runs, says so and leaves it alone)
#   ./start.sh restart                 # stop the running server (./stop.sh), then start it again, e.g. to apply
#                                      # changed settings; the new arguments are checked before stopping
#   ./start.sh restart --context 32768
#   VISION=0 THINKING=0 ./start.sh restart
# Extra arguments come after the defaults, so they win (the last value of a flag counts).
# Settings, from the environment or ./.env (KEY=value lines): CONTEXT, PARALLEL, VISION, THINKING, DRAFTS,
#      MTP_DRAFTS, MTP_CONFIDENCE, PROMPT_CACHE_GIB, SERVED_NAME, PORT, HOST, MODEL_DIR, UNIT, TF_HEAT_HIGH,
#      TF_HEAT_LOW, STOP_TIMEOUT, RUNTIME_MAX, MIN_FREE_GIB (see scripts/config.sh); TENSORFOLD_* (e.g.
#      TENSORFOLD_API_KEY) and the engine's other TF_* (passed to the server); PREPARE (auto | 1 | 0);
#      WAIT_TIMEOUT (seconds, default 900)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

# The serve arguments: scripts/config.sh's defaults first, then the command line's (the engine keeps the last value).
SERVE_ARGS=(--host "$HOST" --port "$PORT" --name "$SERVED_NAME" --context "$CONTEXT" --parallel "$PARALLEL")
[[ -n "$PROMPT_CACHE_GIB" ]] && SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE_ARGS+=(--no-drafts)
[[ -n "$MTP_DRAFTS" ]] && SERVE_ARGS+=(--mtp-drafts "$MTP_DRAFTS")
[[ -n "$MTP_CONFIDENCE" ]] && SERVE_ARGS+=(--mtp-confidence "$MTP_CONFIDENCE")
SERVE_ARGS+=("$@")
# The effective value of a flag (its last occurrence, as --flag value or --flag=value).
arg_value() {
  local flag=$1 value="" i
  for (( i = 0; i < ${#SERVE_ARGS[@]}; i++ )); do
    case "${SERVE_ARGS[i]}" in
      "$flag") value="${SERVE_ARGS[i + 1]:-}" ;;
      "$flag="*) value="${SERVE_ARGS[i]#*=}" ;;
    esac
  done
  echo "$value"
}
# The address the server will listen on, after any --host or --port given here; whether it loads the image tower.
HOST=$(arg_value --host); PORT=$(arg_value --port)
URL=$(api_url)
images=0; [[ " ${SERVE_ARGS[*]} " == *" --vision "* ]] && images=1
ctl() { systemctl --user "$@"; }
running() { ctl is-active --quiet "$UNIT"; }
served_name() {
  api --max-time 5 "$(unit_url)/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

# ---------------------------------------------------------------- banner and progress
B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
printf '\n%s  TensorFold %s · Nemotron 3.5 Super VL on one DGX Spark%s\n' "$M" "$TF_VERSION" "$R"
printf '%s  recipe design after Mia'"'"'s AI Lab (see CREDITS.md)%s\n' "$D" "$R"
printf '%s  %s · %s tokens · %s streams%s · port %s%s\n\n' "$D" "$MODEL_REPO" "$(arg_value --context)" \
  "$(arg_value --parallel)" "$( (( images )) && echo ' · images')" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

mkdir -p "$STATE_DIR"
exec 8>"$STATE_DIR/.start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"

# ---------------------------------------------------------------- already running?
if [[ "$MODE" == start ]] && running; then
  log "$UNIT is already running (model: $(served_name || echo "not answering yet"), $(unit_url)): nothing to do."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
# scripts/prepare.sh runs whenever what it last prepared differs from now: the first run, another release or
# checkpoint. PREPARE=1 forces it, PREPARE=0 skips it. A checkpoint already hashed is not hashed again.
step 1 "Setup: engine and checkpoint"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  prepare_args=()
  [[ "$(checkpoint_id)" == "$(cat "$CHECKPOINT_STAMP" 2>/dev/null)" ]] && prepare_args=(--skip-hash)
  log "Not ready yet: running scripts/prepare.sh ${prepare_args[*]}"
  ./scripts/prepare.sh "${prepare_args[@]}"
else
  log "Ready: $ENGINE_DIR and $MODEL_DIR${PREPARE:+ (PREPARE=$PREPARE)}"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
[[ -x "$TF_BIN" ]] || die "$TF_BIN missing: $why install it"
[[ -f "$MODEL_DIR/config.json" ]] || die "no checkpoint in $MODEL_DIR: $why check it"

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: arguments, previous server, GPU, port, memory"
# The engine's own parser, given a model path that cannot exist: it refuses bad arguments (exit 2) before it looks
# for the model, and good ones reach the lookup, which fails (exit 1) naming the path. So a typo, or a binary that
# does not run, fails here, before anything is stopped, and nothing touches the GPU.
problem=$("$TF_BIN" serve /dev/null/argument-check "${SERVE_ARGS[@]}" 2>&1) && rc=0 || rc=$?
problem=${problem##*$'\n'}
[[ $rc == 1 && "$problem" == *"/dev/null/argument-check is neither"* ]] ||
  die "tensorfold serve rejects the arguments (exit $rc: ${problem#tensorfold serve: error: }); nothing was changed"
if [[ "$MODE" == restart ]] && running; then          # after the setup and the checks: down only while restarting
  ./stop.sh
fi
ctl reset-failed "$UNIT" 2>/dev/null || true         # a previous run that failed leaves its unit behind
others=$(gpu_pids) || die "nvidia-smi failed: is the NVIDIA driver working?"
[[ -z "$others" ]] || die "another process is on the GPU: $(ps -o pid=,args= -p "$(paste -sd, <<< "$others")" | cut -c1-120)"
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
# The engine's own room is what is available after the load less a tenth of RAM (see MIN_FREE_GIB).
avail_gib=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if (( avail_gib >= MIN_FREE_GIB )); then
  log "Arguments OK, GPU idle, port $PORT free, ${avail_gib} GiB memory available"
else
  warn "only ${avail_gib} GiB memory available (~${MIN_FREE_GIB} is known to be enough for all 8 streams and the full prompt cache): the engine may keep a smaller cache or fewer streams; stop other work, or lower CONTEXT / PARALLEL / PROMPT_CACHE_GIB"
fi

# The engine's switches from the environment reach the server: TENSORFOLD_* and every TF_* but the recipe's own.
ENV_ARGS=()
while read -r name; do ENV_ARGS+=(--setenv="$name"); done < <(
  compgen -e | grep -E '^(TENSORFOLD|TF)_' | grep -vxF -f <(printf '%s\n' "${RECIPE_TF_VARS[@]}") || true)

# ---------------------------------------------------------------- 3. launch
step 3 "Launch: systemd user unit $UNIT"
log "$TF_BIN serve $MODEL_DIR ${SERVE_ARGS[*]}"
launched=$(date +%s.%N)   # to the microsecond: a restart's stop lines fall in the same second
# systemd stops the unit with SIGTERM, its default, and SIGKILL after TimeoutStopSec. The description ends with the
# API's address, which stop.sh and a later start.sh read back (unit_url).
systemd-run --user --quiet --unit="$UNIT" --description="TensorFold $TF_VERSION: $SERVED_NAME at $URL" \
  --property=TimeoutStopSec="$STOP_TIMEOUT" --property=RuntimeMaxSec="$RUNTIME_MAX" \
  "${ENV_ARGS[@]}" "$TF_BIN" serve "$MODEL_DIR" "${SERVE_ARGS[@]}"

# ---------------------------------------------------------------- 4. load, with the server's log and a heartbeat
# What TensorFold 1.1.0 holds on a GB10 at the default settings (131,072-token window, 8 streams, images), measured:
# ~92 GiB, the model with its first stream (74.18) plus the other streams' attention pages and the prompt cache's
# slots, set aside at start. The first start of a new build also times its verify windows (~2 min), cached for later
# starts.
step 4 "Loading: ~92 GiB onto the GPU at the default settings (a new build's first start: ~2 min more)"
# journalctl is the background job, so killing it ends the whole pipeline (no orphaned `journalctl -f`)
journalctl --user -u "$UNIT" --since "@$launched" -f -o cat > >(sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT

# GPU memory the server holds so far (GiB)
loaded_gib() {
  local pid
  pid=$(ctl show -p MainPID --value "$UNIT")
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v pid="$pid" '$1 == pid { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── last server log lines ──%s\n' "$D" "$R"
  journalctl --user -u "$UNIT" --since "@$launched" -n 40 -o cat --no-pager 2>&1 | sed 's/^/  │ /'
  die "$1"
}

start=$SECONDS
next_beat=15
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do   # /health never needs the API key
  running || fail "the server exited (status $(ctl show -p ExecMainStatus --value "$UNIT")) before it was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); it is still running: journalctl --user -u $UNIT -f"
  if (( SECONDS - start >= next_beat )); then
    printf '  %s⋯ %ss elapsed, %s GiB on the GPU%s\n' "$D" "$((SECONDS - start))" "$(loaded_gib)" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
log "Server answered after $((SECONDS - start))s with $(loaded_gib) GiB on the GPU"

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion, thinking off"
SERVED=$(served_name || echo "$SERVED_NAME")
if smoke=$(api --max-time 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}], \"chat_template_kwargs\": {\"enable_thinking\": false}}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); print(r["usage"]["completion_tokens"], "tokens, token_sha256", r["tensorfold"]["token_sha256"][:16])' 2>/dev/null); then
  log "OK: $smoke"
else
  warn "the smoke test request failed (a board above TF_HEAT_HIGH holds prompts until it cools to TF_HEAT_LOW); the server is still running (journalctl --user -u $UNIT -f)"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Logs     journalctl --user -u $UNIT -f
    Check    API_URL=$URL tools/smoke.py
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
