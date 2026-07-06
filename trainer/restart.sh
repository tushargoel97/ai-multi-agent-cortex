#!/usr/bin/env bash
# Safely (re)start the host-side MLX trainer on port 8200.
#
# Frees the port by killing ONLY the process LISTENING on it (the old trainer) —
# never Docker's port-proxies or the daemon (that's what `kill -9 $(lsof -ti :8200)`
# without -sTCP:LISTEN did wrong). Then starts uvicorn.
#
# Usage:
#   ./restart.sh              # start / restart the trainer
#   ./restart.sh --reload     # + auto-reload on code changes (dev)
#   TRAINER_PORT=8300 ./restart.sh
set -euo pipefail

PORT="${TRAINER_PORT:-8200}"
cd "$(dirname "$0")"

# Only the LISTENER on this port — not clients or Docker's port forwarders.
pids="$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$pids" ]; then
  echo "Stopping existing trainer on :${PORT} (pid: ${pids//$'\n'/ })…"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  # Give it up to ~5s to exit gracefully.
  for _ in 1 2 3 4 5; do
    sleep 1
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1 || break
  done
  # Force only the specific stubborn PID(s), if any survived.
  still="$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$still" ]; then
    echo "Force-killing stubborn pid: ${still//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -9 $still 2>/dev/null || true
    sleep 1
  fi
fi

echo "Starting trainer on :${PORT}…"
exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
