#!/usr/bin/env bash
# Activate the dedicated serving environment before invoking this script.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GAMESCALING_API_KEY_FILE:?Set GAMESCALING_API_KEY_FILE to a private token file}"
extra_args=()
if [[ -n "${GAMESCALING_KERNEL_BUILD_DIRECTORY:-}" ]]; then
  extra_args+=(--kernel-build-directory "$GAMESCALING_KERNEL_BUILD_DIRECTORY")
fi
# The final expansion also supports empty arrays under Bash 3.2's nounset.
exec "${GAMESCALING_PYTHON:-python}" -m gamescaling.server \
  --model-path "${1:?Usage: bash scripts/serve_transformers.sh /path/to/model}" \
  --host "${GAMESCALING_HOST:-127.0.0.1}" \
  --port "${GAMESCALING_PORT:-8000}" \
  --api-key-file "$GAMESCALING_API_KEY_FILE" \
  --attention-implementation "${GAMESCALING_ATTENTION:-eager}" \
  --stop-implementation "${GAMESCALING_STOP_IMPLEMENTATION:-legacy}" \
  --vision-cache-mib "${GAMESCALING_VISION_CACHE_MIB:-0}" \
  --gdn-implementation "${GAMESCALING_GDN_IMPLEMENTATION:-reference}" \
  --prefill-implementation "${GAMESCALING_PREFILL_IMPLEMENTATION:-reference}" \
  --decode-implementation "${GAMESCALING_DECODE_IMPLEMENTATION:-eager}" \
  ${extra_args[@]+"${extra_args[@]}"}
