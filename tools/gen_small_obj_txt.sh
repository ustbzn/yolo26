#!/usr/bin/env bash
set -euo pipefail

IMG_ROOT="/pcpt/pcpt/project/zengliang/obs_data/small_goal_half/images"
DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dataset"
TRAIN_OUT="${DATASET_DIR}/small_obj_train.txt"
VAL_OUT="${DATASET_DIR}/small_obj_val.txt"

find "${IMG_ROOT}/train" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | sort > "${TRAIN_OUT}"
find "${IMG_ROOT}/val" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | sort > "${VAL_OUT}"

echo "Done."
wc -l "${TRAIN_OUT}" "${VAL_OUT}"
