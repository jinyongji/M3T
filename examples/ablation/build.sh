#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OLD="${M3T_COMPARISON_KIT:-/home/jyj/Projects/industrial-tracker-comparison}"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
PYTHON=/home/jyj/autodl-tmp/m3t-user/env/bin/python3
unset PYTHONHOME PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" -B -c 'import sys,numpy; print("Python:",sys.executable,"NumPy:",numpy.__version__)'
exec 9>"$KIT/.build.lock"
flock -n 9 || { printf 'An ablation build/evaluation is active.\n'; exit 1; }
"$PYTHON" -B "$KIT/ablation_build.py" generate --kit "$OLD"
cmake -S "$KIT" -B "$KIT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPARISON_KIT="$OLD" -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
cmake --build "$KIT/build" --target evaluate_ablation --parallel "${M3T_BUILD_JOBS:-2}"
"$PYTHON" -B "$KIT/ablation_build.py" record --kit "$OLD"
printf 'Build complete. Run smoke for each dataset before full.\n'
