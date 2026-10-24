#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${M3T_SOURCE:-/home/jyj/Projects/M3T}"
source /home/jyj/autodl-tmp/m3t-user/user-env.sh
source /home/jyj/autodl-tmp/m3t-deps/env.sh
if [[ -f "$KIT/payload_manifest.json" ]]; then
  python3 "$KIT/install.py" "$SOURCE"
fi
cmake -S "$SOURCE" -B "$SOURCE/build-linux" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_AZURE_KINECT=OFF -DUSE_REALSENSE=OFF -DUSE_GTEST=OFF \
  -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" -DOpenCV_DIR="$OpenCV_DIR"
cmake --build "$SOURCE/build-linux" --target evaluate_industrial_dataset --parallel "${M3T_BUILD_JOBS:-2}"
printf 'Build complete. Next: select your allocated GPU and run bash run.sh smoke\n'
