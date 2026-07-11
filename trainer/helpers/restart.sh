#!/usr/bin/env bash
set -euo pipefail

PORT="${TRAINER_PORT:-8200}"
TRAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAINER_DIR"

pids="$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$pids" ]; then
  echo "Stopping existing trainer on :${PORT} (pid: ${pids//$'\n'/ })..."
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    sleep 1
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1 || break
  done
  still="$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$still" ]; then
    echo "Force-killing stubborn pid: ${still//$'\n'/ }"
    kill -9 $still 2>/dev/null || true
    sleep 1
  fi
fi

echo "Starting trainer on :${PORT}..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
