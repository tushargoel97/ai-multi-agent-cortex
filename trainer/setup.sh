#!/usr/bin/env bash
# Compatibility entry point. The implementation lives in trainer/helpers/.
set -euo pipefail

TRAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TRAINER_DIR/helpers/setup.sh" "$@"
