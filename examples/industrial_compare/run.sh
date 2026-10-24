#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${M3T_COMPARE_RUNS:-/home/jyj/autodl-tmp/M3T-industrial-comparison-runs}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
# Fix CPU parallelism for all algorithms and the software OpenGL renderer.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1
case "$MODE" in
  status)
    OUTPUT="${1:-}"
    if [[ -z "$OUTPUT" ]]; then OUTPUT="$(cat "$RUN_ROOT/latest_run.txt")"; fi
    if [[ -f "$OUTPUT/run_manifest.json" ]]; then
      cat "$OUTPUT/run_manifest.json"
    else
      printf 'No manifest yet; read the startup error below.\n'
    fi
    printf '\nRecent log (%s):\n' "$OUTPUT"
    if [[ -f "$OUTPUT/launcher.log" ]]; then tail -n 15 "$OUTPUT/launcher.log"; fi
    exit 0;;
  resume)
    OUTPUT="${1:?Usage: bash run.sh resume /absolute/run/directory}"
    [[ $# -eq 1 ]] || { printf 'Resume accepts only a run directory.\n'; exit 2; }
    exec python3 "$KIT/resume.py" "$OUTPUT";;
  smoke) OPTIONS=(--scene scene_01 --condition 01_normal --object obj_000001 --frames 20);;
  full) OPTIONS=(--frames 500);;
  *) printf 'Usage: bash run.sh smoke|full [options]|status [RUN_DIR]|resume RUN_DIR\n'; exit 2;;
esac
OUTPUT="$RUN_ROOT/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "$OUTPUT"
printf '%s\n' "$OUTPUT" > "$RUN_ROOT/latest_run.txt"
if [[ "$MODE" == smoke ]]; then
  python3 -u "$KIT/compare.py" --output "$OUTPUT" "${OPTIONS[@]}" "$@" 2>&1 | tee "$OUTPUT/launcher.log"
else
  nohup python3 -u "$KIT/compare.py" --output "$OUTPUT" "${OPTIONS[@]}" "$@" \
    > "$OUTPUT/launcher.log" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
  printf 'Batch launched. PID: %s\nOutput: %s\nLog: %s/launcher.log\n' "$PID" "$OUTPUT" "$OUTPUT"
fi
