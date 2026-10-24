#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${INDUSTRIAL_RENDER_BASE:-$(dirname -- "$KIT")}"
LAUNCHER="$BASE/dataset/run-v0.9.sh"
MODE="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
[[ $# -eq 0 ]] || { printf 'Usage: bash render.sh check|smoke|full|status|stop|install\n'; exit 2; }
[[ -f "$LAUNCHER" ]] || { printf 'Run prepare.sh first. Missing: %s\n' "$LAUNCHER"; exit 2; }
case "$MODE" in
  status) exec bash "$LAUNCHER" --status;;
  stop) exec bash "$LAUNCHER" --stop;;
  check) OPTIONS=(--check-only);;
  smoke) OPTIONS=(--smoke-test);;
  full) OPTIONS=(--background);;
  install) OPTIONS=(--install-only);;
  *) printf 'Usage: bash render.sh check|smoke|full|status|stop|install\n'; exit 2;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'Set CUDA_VISIBLE_DEVICES to your allocated GPU UUID before rendering.\n'; exit 2; }
[[ -f "$BASE/dataset/中间结果文件/compat_v09_receipt.json" ]] || { printf 'Run prepare.sh to install the 900x600 compatibility patch first.\n'; exit 2; }
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME -u PYTHONPATH \
  bash "$LAUNCHER" "${OPTIONS[@]}" --gpu 0
