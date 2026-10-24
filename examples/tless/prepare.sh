#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/jyj/autodl-tmp/m3t-user/env/bin/python3
[[ -x "$PYTHON" ]] || { printf 'Existing rootless Python environment is missing.\n'; exit 1; }
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME -u PYTHONPATH "$PYTHON" -B -u "$KIT/prepare_tless.py" "$@"
