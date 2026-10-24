#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${1:-/home/jyj/autodl-tmp/industrial-render-900x600}"
BASE="$(cd -- "$BASE" && pwd -P)"
NAME='工业同时运动序列服务器渲染包-v0.9-20261017.tar.gz'
EXPECTED='615401ecc2fe06dfaf0dda81fafb1eb9855e513ec759e8d5d6bc516837d0b5b2'
PYTHON='/home/jyj/autodl-tmp/m3t-user/env/bin/python'
[[ -f "$BASE/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016/cameras/camera.json" ]] || {
  printf 'Existing 900x600 dataset is missing below %s/dataset. Check the directory.\n' "$BASE"; exit 2;
}
[[ -x "$PYTHON" ]] || { printf 'Existing user Python is missing: %s\n' "$PYTHON"; exit 2; }
command -v flock >/dev/null || { printf 'The system flock utility is required to protect the v0.9 inputs.\n'; exit 2; }
mkdir -p -- "$BASE/dataset/中间结果文件/run_state_v09"
exec 9>"$BASE/dataset/中间结果文件/run_state_v09/render.lock"
flock -n 9 || { printf 'The v0.9 renderer is active. Wait for it before preparing its input files again.\n'; exit 2; }
cd -- "$BASE"
printf 'Verifying both uploaded parts...\n'
printf '%s  %s\n' \
  '30e620dd505e9de97be77ecbbe54cf4e9169d9860224f8315a7c28531ac34af6' "$NAME.part01" \
  '217df3140fabfb594b8693880784f4387561d1cc2acdb4751dc7d629f59a6985' "$NAME.part02" | sha256sum -c -
if [[ -f "$NAME" ]] && printf '%s  %s\n' "$EXPECTED" "$NAME" | sha256sum -c -; then
  printf 'The merged archive is already verified.\n'
else
  printf 'Joining parts in order...\n'
  cat -- "$NAME.part01" "$NAME.part02" > "$NAME.joining"
  printf '%s  %s\n' "$EXPECTED" "$NAME.joining" | sha256sum -c -
  mv -- "$NAME.joining" "$NAME"
fi
printf 'Extracting the v0.9 input files beside the existing dataset...\n'
tar --no-same-owner -xzf "$NAME" -C "$BASE"
"$PYTHON" "$KIT/apply_compat.py" "$BASE/dataset"
printf '\nPrepared. Next: check-only, smoke-test, then background rendering using dataset/run-v0.9.sh.\n'
