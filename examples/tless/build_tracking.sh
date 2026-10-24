#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OLD="${M3T_COMPARISON_KIT:-/home/jyj/Projects/industrial-tracker-comparison}"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON="${TLESS_PYTHON:-/home/jyj/autodl-tmp/m3t-user/env/bin/python3}"
[[ -x "$PYTHON" ]] || { printf 'Missing T-LESS Python: %s\n' "$PYTHON"; exit 2; }
unset PYTHONHOME PYTHONPATH
"$PYTHON" -B "$KIT/tracking_python.py"
exec 9>"$KIT/.build.lock"
flock -n 9 || { printf 'Another T-LESS build/evaluation is active.\n'; exit 1; }
"$PYTHON" -B "$KIT/tracking_build.py" generate --kit "$OLD"
for METHOD in rbgt srt3d icg_rgb m3t_original_rgb ours; do
  case "$METHOD" in rbgt) ENGINE=RBGT;; srt3d) ENGINE=SRT3D;; icg_rgb) ENGINE=ICG;; m3t_original_rgb) ENGINE=M3T;; ours) ENGINE=OURS;; esac
  cmake -S "$KIT" -B "$KIT/build/$METHOD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBENCH_METHOD="$ENGINE" -DCOMPARISON_KIT="$OLD" -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
  cmake --build "$KIT/build/$METHOD" --target evaluate_tless --parallel "${M3T_BUILD_JOBS:-2}"
  "$PYTHON" -B "$KIT/tracking_build.py" record --kit "$OLD" --method "$METHOD"
done
printf 'Build complete. Next: bash run_tracking.sh smoke\n'
