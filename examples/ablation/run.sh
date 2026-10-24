#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
if [[ "$ACTION" == build ]]; then exec bash "$KIT/build.sh"; fi
[[ $# -ge 2 ]] || { printf 'Usage: run.sh build | smoke|full|status DATASET | resume OUTPUT\nDatasets: rbot industrial_base industrial_advanced\n'; exit 2; }
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON=/home/jyj/autodl-tmp/m3t-user/env/bin/python3
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "$ACTION" == resume ]]; then
  OUTPUT="$2"
  LANE="$("$PYTHON" -B -c 'import json,sys;from pathlib import Path;print(json.loads((Path(sys.argv[1])/"run_manifest.json").read_text())["identity"]["launch"]["lane"])' "$OUTPUT")"
else LANE="$2"; fi
case "$LANE" in rbot) GPU=1;; industrial_base) GPU=4;; industrial_advanced) GPU=7;; *) printf 'Unknown dataset: %s\n' "$LANE"; exit 2;; esac
RUNS="${M3T_ABLATION_RUNS:-/home/jyj/autodl-tmp/M3T-ablation-runs}/$LANE"
mkdir -p "$RUNS"
if [[ "$ACTION" == status ]]; then
  OUTPUT="${3:-$(cat "$RUNS/latest_run.txt" 2>/dev/null || true)}"
  [[ -n "$OUTPUT" && -d "$OUTPUT" ]] || { printf 'No run yet.\n'; exit 0; }
  "$PYTHON" -B -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1])/"run_manifest.json";x=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({k:x.get(k) for k in ["status","pid","current","job","expected_jobs","expected_reference_jobs","completed_evaluated_frames","error"]},ensure_ascii=False,indent=2));print("completed_jobs=",len(x.get("completed_jobs",[])));print("output=",sys.argv[1])' "$OUTPUT"
  tail -n 15 "$OUTPUT/launcher.log"; exit 0
fi
[[ "$ACTION" == smoke || "$ACTION" == full || "$ACTION" == resume ]] || { printf 'Unknown action\n'; exit 2; }
"$PYTHON" -B -c 'import numpy; print("NumPy:",numpy.__version__)'
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 2
OPTIONS=()
if [[ "$ACTION" == resume ]]; then OPTIONS=(--resume); else
  OUTPUT="$RUNS/${ACTION}_$(date -u +%Y%m%dT%H%M%SZ)_$$"; mkdir -p "$OUTPUT"
  if [[ "$ACTION" == smoke ]]; then
    OPTIONS=(--smoke)
    PREVIOUS_SMOKE="$(cat "$RUNS/latest_smoke.txt" 2>/dev/null || true)"
    if [[ -n "$PREVIOUS_SMOKE" && -f "$PREVIOUS_SMOKE/run_manifest.json" ]]; then OPTIONS+=(--reuse-model-run "$PREVIOUS_SMOKE"); fi
    printf '%s\n' "$OUTPUT" > "$RUNS/latest_smoke.txt"
  else
    SMOKE="$(cat "$RUNS/latest_smoke.txt" 2>/dev/null || true)"
    [[ -n "$SMOKE" && -f "$SMOKE/regression_gate.json" ]] || { printf 'Complete smoke regression first.\n'; exit 2; }
    "$PYTHON" -B -c 'import json,sys;from pathlib import Path;x=json.loads((Path(sys.argv[1])/"regression_gate.json").read_text());sys.exit(0 if x.get("passed") else "Smoke regression requires review")' "$SMOKE"
    OPTIONS=(--smoke-run "$SMOKE")
  fi
fi
printf '%s\n' "$OUTPUT" > "$RUNS/latest_run.txt"
nvidia-smi > "$OUTPUT/gpu_at_launch.txt"
ps -u jyj -o pid,ppid,etime,pcpu,pmem,args > "$OUTPUT/user_processes_at_launch.txt"
nohup "$PYTHON" -B "$KIT/ablation_run.py" --lane "$LANE" --output "$OUTPUT" "${OPTIONS[@]}" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
printf 'Launched %s on physical GPU %s (%s). PID: %s\nOutput: %s\nLog: %s/launcher.log\n' "$ACTION" "$GPU" "$CUDA_VISIBLE_DEVICES" "$!" "$OUTPUT" "$OUTPUT"
