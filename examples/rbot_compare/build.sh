#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OLD="${M3T_COMPARISON_KIT:-/home/jyj/Projects/industrial-tracker-comparison}"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
exec 9>"$KIT/.build.lock"
flock -n 9 || { printf 'This RBOT kit is already building or evaluating.\n'; exit 1; }
python3 -B "$KIT/verify_payload.py"
python3 -B "$KIT/rbot_build.py" prepare --kit "$OLD"
for METHOD in rbgt srt3d icg_rgb m3t_original_rgb ours; do
  case "$METHOD" in rbgt) ENGINE=RBGT;; srt3d) ENGINE=SRT3D;; icg_rgb) ENGINE=ICG;; m3t_original_rgb) ENGINE=M3T;; ours) ENGINE=OURS;; esac
  cmake -S "$KIT" -B "$KIT/build/$METHOD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBENCH_METHOD="$ENGINE" -DCOMPARISON_KIT="$OLD" \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
  cmake --build "$KIT/build/$METHOD" --target evaluate_rbot_comparison --parallel "${M3T_BUILD_JOBS:-2}"
  python3 -B "$KIT/rbot_build.py" record --kit "$OLD" --method "$METHOD"
done
printf 'RBOT five-method build complete. Next: bash run.sh smoke\n'
