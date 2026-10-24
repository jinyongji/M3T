#!/usr/bin/env bash
# Shared with prepare.sh/render.sh; descriptors survive the detached controller.
guard_benchmark() {
  command -v flock >/dev/null || { printf 'Missing system flock command.\n'; exit 2; }
  case "${ADVANCED_ALLOW_CONCURRENT:-0}" in
    1)
      printf 'Concurrent rendering enabled. Use a separate GPU; CPU and disk load can affect benchmark timings.\n'
      return 0;;
    0) ;;
    *) printf 'ADVANCED_ALLOW_CONCURRENT must be 0 or 1.\n'; exit 2;;
  esac
  local compare_cache="${M3T_COMPARE_CACHE:-/home/jyj/autodl-tmp/M3T-industrial-comparison-cache}"
  mkdir -p -- "$compare_cache"
  exec 8>>"$compare_cache/.cache.lock"
  flock -s -n 8 || {
    printf 'Five-method evaluation is active. Wait, or set ADVANCED_ALLOW_CONCURRENT=1 when rendering on a separate GPU.\n'; exit 2;
  }
}
guard_workspace() {
  mkdir -p -- "$BASE/dataset/中间结果文件/run_state_advanced_v1"
  exec 7>>"$BASE/dataset/中间结果文件/run_state_advanced_v1/wrapper.lock"
  flock -n 7 || { printf 'An advanced-v1 preparation/render task is active in this workspace.\n'; exit 2; }
}
