#!/usr/bin/env bash
# Stop the server that ./start.sh started (its systemd user unit), then confirm the GPU is empty. The server gets
# SIGTERM and the stop timeout start.sh gave the unit before SIGKILL; requests still running are cut off (it does
# not drain them).
# Usage: ./stop.sh      Env: UNIT (see scripts/config.sh)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

if systemctl --user is-active --quiet "$UNIT"; then
  log "Stopping $UNIT (SIGTERM, up to $(systemctl --user show -p TimeoutStopUSec --value "$UNIT"))"
  systemctl --user stop "$UNIT"
else
  log "$UNIT is not running: nothing to stop"
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# The driver frees the memory a moment after the process ends.
for _ in {1..30}; do
  pids=$(gpu_pids) || die "nvidia-smi failed: cannot confirm the GPU is empty"
  [[ -n "$pids" ]] || { log "The GPU is empty"; exit 0; }
  sleep 1
done
die "the GPU still runs: $(ps -o pid=,args= -p "$(paste -sd, <<< "$pids")" | cut -c1-120)"
