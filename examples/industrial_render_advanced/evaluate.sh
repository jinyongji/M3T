#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "$KIT/.." && pwd -P)"
RUNS="${M3T_ADVANCED_RUNS:-/home/jyj/autodl-tmp/M3T-industrial-advanced-runs}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1
case "$MODE" in
  queue-status)
    OUTPUT="${1:-$(cat "$RUNS/latest_queue.txt")}"
    if [[ -f "$OUTPUT/queue_status.json" ]]; then cat "$OUTPUT/queue_status.json"; fi
    for file in "$OUTPUT/queue.log" "$OUTPUT/smoke/launcher.log" "$OUTPUT/full/launcher.log"; do
      if [[ -f "$file" ]]; then printf '\n%s\n' "$file"; tail -n 12 "$file"; fi
    done
    exit 0;;
  status)
    OUTPUT="${1:-$(cat "$RUNS/latest_run.txt")}"
    if [[ -f "$OUTPUT/run_manifest.json" ]]; then cat "$OUTPUT/run_manifest.json"; fi
    if [[ -f "$OUTPUT/launcher.log" ]]; then tail -n 15 "$OUTPUT/launcher.log"; fi
    exit 0;;
  smoke) OPTIONS=(--smoke --frames 140);;
  full) OPTIONS=(--frames 500);;
  after-render)
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
      printf 'Select exactly one allocated GPU UUID first.\n'; exit 2;
    }
    OUTPUT="$RUNS/after_render_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p -- "$OUTPUT"
    nohup python3 -B -u "$KIT/after_render.py" --root "$BASE/dataset" --output "$OUTPUT" "$@" \
      > "$OUTPUT/queue.log" 2>&1 < /dev/null &
    printf 'Queue launched. PID: %s\nOutput: %s\nLog: %s/queue.log\n' "$!" "$OUTPUT" "$OUTPUT"
    exit 0;;
  resume)
    OUTPUT="${1:?Usage: bash evaluate.sh resume EXISTING_OUTPUT}"
    OPTIONS=(--resume)
    [[ $# -eq 1 ]] || { printf 'Resume accepts only its output directory.\n'; exit 2; }
    shift;;
  *) printf 'Usage: bash evaluate.sh after-render|queue-status|smoke|full [--focus-only]|status|resume OUTPUT\n'; exit 2;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'Select your allocated GPU UUID first.\n'; exit 2; }
if [[ "$MODE" != resume ]]; then OUTPUT="$RUNS/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"; fi
mkdir -p -- "$OUTPUT"
printf '%s\n' "$OUTPUT" > "$RUNS/latest_run.txt"
if [[ "$MODE" == smoke ]]; then
  python3 -u "$KIT/evaluate.py" --root "$BASE/dataset" --output "$OUTPUT" "${OPTIONS[@]}" "$@" 2>&1 | tee "$OUTPUT/launcher.log"
else
  nohup python3 -u "$KIT/evaluate.py" --root "$BASE/dataset" --output "$OUTPUT" "${OPTIONS[@]}" "$@" \
    >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
  printf 'Batch launched. PID: %s\nOutput: %s\n' "$!" "$OUTPUT"
fi
