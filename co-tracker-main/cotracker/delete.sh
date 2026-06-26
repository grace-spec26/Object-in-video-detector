#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_DIRS=(
  "${PROJECT_ROOT}/data/frames"
  "${PROJECT_ROOT}/data/coordinates"
  "${PROJECT_ROOT}/raw-mask-data/frames"
  "${PROJECT_ROOT}/raw-mask-data/coordinates"
  "${PROJECT_ROOT}/raw-mask-data/mask"
  "${PROJECT_ROOT}/raw-mask-data/masked_frames"
)

delete_contents() {
  local dir="$1"
  local abs_dir

  mkdir -p "${dir}"
  abs_dir="$(cd "${dir}" && pwd)"

  case "${abs_dir}" in
    "${PROJECT_ROOT}"/*) ;;
    *)
      echo "Refusing to delete outside project root: ${abs_dir}" >&2
      exit 1
      ;;
  esac

  find "${abs_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  echo "Cleared: ${abs_dir}"
}

for dir in "${TARGET_DIRS[@]}"; do
  delete_contents "${dir}"
done

echo "Done."
