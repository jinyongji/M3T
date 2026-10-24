#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "$KIT/.." && pwd -P)"
[[ $# -eq 0 ]] || { printf 'Usage: bash prepare.sh\n'; exit 2; }
source "$KIT/guard.sh"
guard_benchmark
guard_workspace
[[ ! -d "$BASE/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016" && ! -d "$BASE/dataset/交付物/工业物体跟踪数据集-v0.9-20261017" ]] || {
  printf 'Use a separate advanced-v1 upload directory to preserve the existing datasets.\n'; exit 2;
}
exec 9>>"$BASE/dataset/中间结果文件/run_state_advanced_v1/render.lock"
flock -n 9 || { printf 'The advanced-v1 controller is running; wait before extracting again.\n'; exit 2; }
cd -- "$BASE"
printf 'Preparing the independent advanced-v1 dataset (already 900x600).\n'
bash "$BASE/start-advanced-v1.0-20261018.sh" --extract-only
printf '\nPrepared. Select the allocated GPU UUID, then run bash %q check, smoke, full.\n' "$KIT/render.sh"
