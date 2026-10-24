#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
if [[ "$ACTION" == build ]]; then exec bash "$KIT/build_rgb.sh"; fi
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON=/home/jyj/autodl-tmp/m3t-user/env/bin/python3
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUNS=/home/jyj/autodl-tmp/M3T-TLESS-RGB-runs
mkdir -p "$RUNS"
if [[ "$ACTION" == status ]]; then
  OUTPUT="${2:-$(cat "$RUNS/latest_run.txt" 2>/dev/null || true)}"
  [[ -n "$OUTPUT" && -d "$OUTPUT" ]] || { printf 'No RGB run yet.\n'; exit 0; }
  "$PYTHON" -B -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1])/"run_manifest.json";x=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({k:x.get(k) for k in ["status","pid","current","job","expected_jobs","expected_reference_jobs","completed_evaluated_frames","error"]},ensure_ascii=False,indent=2));print("output=",sys.argv[1])' "$OUTPUT"
  tail -n 15 "$OUTPUT/launcher.log";exit 0
fi
OPTIONS=()
if [[ "$ACTION" == resume ]]; then
  [[ $# -ge 2 ]] || exit 2
  OUTPUT="$2";GPU="${3:-3}"
  STAGE="$("$PYTHON" -B -c 'import json,sys;from pathlib import Path;print(json.loads((Path(sys.argv[1])/"run_manifest.json").read_text())["identity"]["launch"]["stage"])' "$OUTPUT")"
  OPTIONS=(--resume)
else
  [[ "$ACTION" == smoke || "$ACTION" == pilot || "$ACTION" == full ]] || { printf 'Usage: run_rgb.sh build | smoke|pilot|full [GPU] | status [OUTPUT] | resume OUTPUT [GPU]\n';exit 2; }
  GPU="${2:-3}";STAGE="$ACTION"
  OUTPUT="$RUNS/${ACTION}_$(date -u +%Y%m%dT%H%M%SZ)_$$";mkdir -p "$OUTPUT"
  if [[ "$ACTION" == smoke ]]; then printf '%s\n' "$OUTPUT" > "$RUNS/latest_smoke.txt";else
    SMOKE="$(cat "$RUNS/latest_smoke.txt" 2>/dev/null || true)"
    [[ -n "$SMOKE" && -f "$SMOKE/regression_gate.json" ]] || { printf 'Pass RGB smoke first.\n';exit 2; }
    "$PYTHON" -B -c 'import json,sys;from pathlib import Path;x=json.loads((Path(sys.argv[1])/"regression_gate.json").read_text());sys.exit(0 if x.get("passed") else "B0 regression requires review")' "$SMOKE"
    OPTIONS=(--smoke-run "$SMOKE")
  fi
fi
[[ "$GPU" =~ ^[0-7]$ ]] || { printf 'Expected physical GPU 0..7.\n';exit 2; }
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 2
nvidia-smi > "$OUTPUT/gpu_at_launch.txt"
ps -u jyj -o pid,ppid,etime,pcpu,pmem,args > "$OUTPUT/user_processes_at_launch.txt"
printf '%s\n' "$OUTPUT" > "$RUNS/latest_run.txt"
nohup "$PYTHON" -B "$KIT/rgb_run.py" --stage "$STAGE" --output "$OUTPUT" "${OPTIONS[@]}" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
printf 'Launched RGB %s on physical GPU %s. PID: %s\nOutput: %s\n' "$STAGE" "$GPU" "$!" "$OUTPUT"
