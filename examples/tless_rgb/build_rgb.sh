#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON=/home/jyj/autodl-tmp/m3t-user/env/bin/python3
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>"$KIT/.build.lock"
flock -n 9 || { printf 'RGB build/evaluation already active.\n'; exit 2; }
exec 8>/home/jyj/Projects/tless-tracker-comparison/.build.lock
flock -s -n 8 || { printf 'Original T-LESS build is active.\n'; exit 2; }
"$PYTHON" -B "$KIT/rgb_build.py" generate
cmake -S "$KIT" -B "$KIT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPARISON_KIT=/home/jyj/Projects/industrial-tracker-comparison -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
cmake --build "$KIT/build" --target evaluate_tless_rgb test_rgb_geometry --parallel "${M3T_BUILD_JOBS:-2}"
"$KIT/build/test_rgb_geometry"
"$PYTHON" -B "$KIT/rgb_build.py" record
printf 'RGB build complete. Run smoke before pilot/full.\n'
