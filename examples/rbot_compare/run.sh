#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${M3T_RBOT_COMPARE_RUNS:-/home/jyj/autodl-tmp/M3T-RBOT-comparison-runs}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == build ]]; then exec bash "$KIT/build.sh" "$@"; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1
case "$MODE" in
  status)
    OUTPUT="${1:-$(cat "$RUNS/latest_run.txt")}"; exec python3 -B "$KIT/compare_rbot.py" status "$OUTPUT";;
  smoke) OPTIONS=(--smoke);;
  full) OPTIONS=(--frames 1001);;
  resume) OUTPUT="${1:?Usage: bash run.sh resume EXISTING_OUTPUT}"; OPTIONS=(--resume); [[ $# -eq 1 ]] || exit 2; shift;;
  *) printf 'Usage: bash run.sh build|smoke|full|status|resume OUTPUT\n'; exit 2;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || { printf 'Select one GPU UUID first.\n'; exit 2; }
if [[ "$MODE" != resume ]]; then OUTPUT="$RUNS/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"; fi
mkdir -p -- "$OUTPUT"
printf '%s\n' "$OUTPUT" > "$RUNS/latest_run.txt"
if [[ "$MODE" == smoke ]]; then
  python3 -B -u "$KIT/compare_rbot.py" --output "$OUTPUT" "${OPTIONS[@]}" "$@" 2>&1 | tee "$OUTPUT/launcher.log"
else
  nohup python3 -B -u "$KIT/compare_rbot.py" --output "$OUTPUT" "${OPTIONS[@]}" "$@" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
  printf 'Batch launched. PID: %s\nOutput: %s\nLog: %s/launcher.log\n' "$!" "$OUTPUT" "$OUTPUT"
fi
