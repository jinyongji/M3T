#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
python3 "$KIT/verify_payload.py"
METHODS=(rbgt srt3d icg_rgb m3t_original_rgb ours)
if [[ $# -gt 0 ]]; then METHODS=("$@"); fi
for METHOD in "${METHODS[@]}"; do
  case "$METHOD" in
    rbgt) ENGINE=RBGT;; srt3d) ENGINE=SRT3D;; icg_rgb) ENGINE=ICG;;
    m3t_original_rgb) ENGINE=M3T;; ours) ENGINE=OURS;;
    *) printf 'Unknown method: %s\n' "$METHOD"; exit 2;;
  esac
  python3 "$KIT/provenance.py" --prepare "$METHOD"
  cmake -S "$KIT" -B "$KIT/build/$METHOD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBENCH_METHOD="$ENGINE" \
    -DBASELINE_ROOT="$KIT/sources/upstream" -DCURRENT_M3T_ROOT="$KIT/sources/current_m3t" \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
  cmake --build "$KIT/build/$METHOD" --target evaluate_industrial_baseline --parallel "${M3T_BUILD_JOBS:-2}"
  python3 "$KIT/provenance.py" "$METHOD"
done
printf 'Build complete. After rendering/publishing finishes, run bash run.sh smoke\n'
