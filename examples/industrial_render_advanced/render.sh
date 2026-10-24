#!/usr/bin/env bash
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "$KIT/.." && pwd -P)"
ROOT="$BASE/dataset"
LAUNCHER="$ROOT/run-advanced-v1.sh"
MODE="${1:-status}"
[[ $# -le 1 ]] || { printf 'Usage: bash render.sh check|smoke|full|status|stop\n'; exit 2; }
[[ -f "$LAUNCHER" ]] || { printf 'Run bash %q first.\n' "$KIT/prepare.sh"; exit 2; }
CLEAN=(env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME -u PYTHONPATH
       -u LIBGL_ALWAYS_SOFTWARE -u LP_NUM_THREADS -u OMP_NUM_THREADS -u OMP_DYNAMIC)
case "$MODE" in
  status) exec "${CLEAN[@]}" bash "$LAUNCHER" --status;;
  stop) exec "${CLEAN[@]}" bash "$LAUNCHER" --stop;;
  check|smoke|full) ;;
  *) printf 'Usage: bash render.sh check|smoke|full|status|stop\n'; exit 2;;
esac
source "$KIT/guard.sh"
guard_benchmark
guard_workspace
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'Set CUDA_VISIBLE_DEVICES to the allocated GPU UUID first.\n'; exit 2; }
case "$MODE" in
  check)
    "${CLEAN[@]}" bash "$LAUNCHER" --check-only --gpu 0
    BLENDER="$ROOT/中间结果文件/runtime_advanced_v1/extracted/blender-4.5.4-linux-x64/blender"
    OUTPUT="$ROOT/中间结果文件/run_state_advanced_v1/independent_audit"
    mkdir -p -- "$OUTPUT"
    "${CLEAN[@]}" "$BLENDER" --background --factory-startup --disable-autoexec --python-exit-code 1 \
      --python "$KIT/audit_inputs.py" -- --root "$ROOT" --output "$OUTPUT/inputs.json"
    "${CLEAN[@]}" "$BLENDER" --background --factory-startup --disable-autoexec --python-exit-code 1 \
      --python "$KIT/audit_scenes.py" -- --root "$ROOT" --output "$OUTPUT/scenes.json"
    printf 'Independent audits passed: %s\n' "$OUTPUT";;
  smoke) exec "${CLEAN[@]}" bash "$LAUNCHER" --smoke-test --gpu 0;;
  full) exec "${CLEAN[@]}" bash "$LAUNCHER" --background --stage all --gpu 0;;
esac
