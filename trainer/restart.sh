#!/usr/bin/env bash
set -euo pipefail

TRAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TRAINER_DIR/helpers/restart.sh" "$@"
