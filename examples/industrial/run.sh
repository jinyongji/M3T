#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${M3T_SOURCE:-/home/jyj/Projects/M3T}"
RUN_ROOT="${M3T_INDUSTRIAL_RUNS:-/home/jyj/autodl-tmp/M3T-industrial-runs}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
case "$MODE" in
  status)
    OUTPUT="${1:-}"
    if [[ -z "$OUTPUT" ]]; then OUTPUT="$(cat "$RUN_ROOT/latest_run.txt")"; fi
    if [[ -f "$OUTPUT/run_manifest.json" ]]; then cat "$OUTPUT/run_manifest.json"; fi
    printf '\nRecent log (%s):\n' "$OUTPUT"
    if [[ -f "$OUTPUT/launcher.log" ]]; then tail -n 12 "$OUTPUT/launcher.log"; fi
    exit 0;;
  smoke) OPTIONS=(--scene scene_01 --motion camera_motion --condition 01_normal --object obj_000001 --frames 20);;
  full) OPTIONS=(--frames 500);;
  resume)
    OUTPUT="${1:?Usage: bash run.sh resume /absolute/run/directory}"
    shift
    [[ $# -eq 0 ]] || { printf 'Resume accepts only the existing run directory.\n'; exit 2; }
    # Recover the exact original selection; shell never evaluates saved strings.
    exec python3 "$KIT/resume.py" "$OUTPUT";;
  *) printf 'Usage: bash run.sh smoke|full|status [RUN_DIR]|resume RUN_DIR\n'; exit 2;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'Set CUDA_VISIBLE_DEVICES to your allocated GPU UUID.\n'; exit 2; }
OUTPUT="$RUN_ROOT/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "$OUTPUT"
printf '%s\n' "$OUTPUT" > "$RUN_ROOT/latest_run.txt"
if [[ "$MODE" == smoke ]]; then
  python3 -u "$KIT/run.py" --source "$SOURCE" --output "$OUTPUT" "${OPTIONS[@]}" "$@" 2>&1 | tee "$OUTPUT/launcher.log"
else
  nohup python3 -u "$KIT/run.py" --source "$SOURCE" --output "$OUTPUT" "${OPTIONS[@]}" "$@" \
    > "$OUTPUT/launcher.log" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
  printf 'Batch launched. PID: %s\nOutput: %s\nLog: %s/launcher.log\n' "$PID" "$OUTPUT" "$OUTPUT"
fi
