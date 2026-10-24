#!/usr/bin/env bash
# CPU affinity and advisory priorities are inherited by the detached controller
# and its Blender/FFmpeg children. They do not reserve CPUs from other users.
set -Eeuo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-status}"
[[ $# -le 1 ]] || { printf 'Usage: bash render_limited.sh prepare|check|smoke|full|status|stop\n'; exit 2; }
case "$MODE" in
  status|stop) exec bash "$KIT/render.sh" "$MODE";;
  prepare) ENTRY=("$KIT/prepare.sh");;
  check|smoke|full) ENTRY=("$KIT/render.sh" "$MODE");;
  *) printf 'Usage: bash render_limited.sh prepare|check|smoke|full|status|stop\n'; exit 2;;
esac
CPUS="${ADVANCED_RENDER_CPUS:-}"
[[ "$CPUS" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || {
  printf 'Set ADVANCED_RENDER_CPUS to the selected logical CPU list, including the chosen cores only.\n'; exit 2;
}
for utility in taskset nice ionice; do
  command -v "$utility" >/dev/null || { printf 'Missing system command: %s\n' "$utility"; exit 2; }
done
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Rendering CPU list: %s; nice: 15; idle I/O class (advisory, may slow rendering).\n' "$CPUS"
exec taskset --cpu-list "$CPUS" nice -n 15 ionice -c 3 bash "${ENTRY[@]}"
