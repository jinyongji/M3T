#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${TLESS_RUNS:-/home/jyj/autodl-tmp/M3T-TLESS-runs}"
ACTION="${1:-status}"
if [[ "$ACTION" == build ]]; then exec bash "$KIT/build_tracking.sh"; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON="${TLESS_PYTHON:-/home/jyj/autodl-tmp/m3t-user/env/bin/python3}"
[[ -x "$PYTHON" ]] || { printf 'Missing T-LESS Python: %s\n' "$PYTHON"; exit 2; }
unset PYTHONHOME PYTHONPATH
if [[ "$ACTION" != status ]]; then "$PYTHON" -B "$KIT/tracking_python.py"; fi
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNS"
case "$ACTION" in
 prepare) exec "$PYTHON" -B "$KIT/tracking_run.py" prepare;;
 smoke|full)
   OUTPUT="$RUNS/${ACTION}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
   mkdir -p "$OUTPUT"; printf '%s\n' "$OUTPUT" > "$RUNS/latest_run.txt"
   if [[ "$ACTION" == smoke ]]; then
     "$PYTHON" -B "$KIT/tracking_run.py" run --smoke --output "$OUTPUT" 2>&1 | tee "$OUTPUT/launcher.log"
   else
     nohup "$PYTHON" -B "$KIT/tracking_run.py" run --output "$OUTPUT" > "$OUTPUT/launcher.log" 2>&1 < /dev/null &
     printf 'Full launched. PID: %s\nOutput: %s\nLog: %s/launcher.log\n' "$!" "$OUTPUT" "$OUTPUT"
   fi;;
 resume)
   [[ $# == 2 ]] || { printf 'Usage: run_tracking.sh resume /absolute/run/path\n'; exit 1; }
   OUTPUT="$2"; [[ -f "$OUTPUT/run_manifest.json" ]] || exit 1
   nohup "$PYTHON" -B "$KIT/tracking_run.py" run --resume --output "$OUTPUT" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
   printf 'Resume PID: %s\nLog: %s/launcher.log\n' "$!" "$OUTPUT";;
 status)
   OUTPUT="${2:-$(cat "$RUNS/latest_run.txt" 2>/dev/null || true)}"
   [[ -n "$OUTPUT" && -d "$OUTPUT" ]] || { printf 'No run yet.\n'; exit 0; }
   "$PYTHON" -B -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1])/"run_manifest.json";x=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({k:x.get(k) for k in ["status","pid","output","expected_jobs","job","current","completed_evaluated_frames","error"]},indent=2));print("completed_jobs=",len(x.get("completed_jobs",[])))' "$OUTPUT"
   tail -n 12 "$OUTPUT/launcher.log";;
 *) printf 'Usage: run_tracking.sh build|prepare|smoke|full|status [output]|resume output\n'; exit 1;;
esac
