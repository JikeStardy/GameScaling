#!/usr/bin/env bash
# Activate the dedicated serving environment before invoking this script.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GAMESCALING_API_KEY_FILE:?Set GAMESCALING_API_KEY_FILE to a private token file}"
exec "${GAMESCALING_PYTHON:-python}" -m gamescaling.server \
  --model-path "${1:?Usage: bash scripts/serve_transformers.sh /path/to/model}" \
  --host "${GAMESCALING_HOST:-127.0.0.1}" \
  --port "${GAMESCALING_PORT:-8000}" \
  --api-key-file "$GAMESCALING_API_KEY_FILE"
