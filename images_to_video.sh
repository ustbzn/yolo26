#!/usr/bin/env bash
set -euo pipefail

# Default args:
#   1) input dir   (default: runs/detect/predict5)
#   2) output mp4  (default: <input>/out.mp4)
#   3) fps         (default: 25)
#   4) size        (optional, WIDTHxHEIGHT like 1920x1080)
INPUT_DIR="${1:-runs/detect/predict37}"
OUTPUT_VIDEO="${2:-${INPUT_DIR%/}/debug37.mp4}"
FPS="${3:-10}"
SIZE="${4:-800x600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/images_to_video.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Error: not found ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input folder not found: ${INPUT_DIR}" >&2
  echo "Usage: $0 [input_dir] [output_video] [fps] [size]" >&2
  echo "Example: $0 runs/detect/predict5 runs/detect/predict5/out.mp4 25 1920x1080" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "Error: python/python3 not found in PATH" >&2
  exit 1
fi

CMD=("${PY_BIN}" "${PY_SCRIPT}" --input "${INPUT_DIR}" --output "${OUTPUT_VIDEO}" --fps "${FPS}")
if [[ -n "${SIZE}" ]]; then
  CMD+=(--size "${SIZE}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
