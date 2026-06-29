#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIRS=(
  "data/frames"
  "data/coordinates"
  "raw-mask-data/frames"
  "raw-mask-data/coordinates"
  "raw-mask-data/mask"
  "raw-mask-data/masked_frames"
)

for relative_dir in "${TARGET_DIRS[@]}"; do
  target_dir="$SCRIPT_DIR/$relative_dir"

  if [[ ! -d "$target_dir" ]]; then
    mkdir -p "$target_dir"
    printf 'Created missing directory: %s\n' "$relative_dir"
    continue
  fi

  find "$target_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  printf 'Cleared: %s\n' "$relative_dir"
done
