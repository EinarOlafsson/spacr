#!/usr/bin/env bash
# Print this machine's memory every INTERVAL seconds until stopped.
#
#   tools/ci_memory_telemetry.sh INTERVAL SAMPLES_FILE
#
# WHY (items 43 and 288, 2026-09-15). Qt shard 0 lost its GitHub runner twice
# in one day -- "The runner has received a shutdown signal" at [83%] after 71
# minutes and at [96%] after 89, inside a 180-minute budget. A runner that is
# lost runs no later step and uploads nothing, and a machine killed for memory
# cannot log that it was. So every sample goes to STDOUT as one MEMORY line:
# that is the step's live log, which GitHub keeps up to the moment the runner
# vanished. A fuller block (free, df, the twelve largest processes) goes to
# SAMPLES_FILE for the artifact of a run that survives.
#
# THE OTHER WAY A WORKER DISAPPEARS. tests/conftest.py ends any pytest process
# whose RSS passes SPACR_TEST_MEMORY_GB (default 6) with exit 3, and xdist
# logs that exactly like a segfault ("node down: Not properly terminated").
# This script cannot see an exit status, so each MEMORY line carries every
# pytest process's RSS beside the guard's ceiling, and a pytest process that
# is gone at the next sample is logged with the RSS it last had.
#
# Reads /proc and procps only: no sudo, no Python, nothing to install. Stops
# promptly on SIGTERM, taking its sleep with it, so it never holds the step's
# output open after the step is done.
set -u

interval=${1:?usage: ci_memory_telemetry.sh INTERVAL SAMPLES_FILE}
samples=${2:?usage: ci_memory_telemetry.sh INTERVAL SAMPLES_FILE}
guard_gb=${SPACR_TEST_MEMORY_GB:-6}
mkdir -p "$(dirname "$samples")"

sleeper=""
trap '[ -n "$sleeper" ] && kill "$sleeper" 2>/dev/null; exit 0' TERM INT

declare -A previous=()

while :; do
  now=$(date -u +%H:%M:%S)
  read -r total available swap_total swap_free < <(
    awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2}
         /^SwapTotal:/ {st = $2} /^SwapFree:/ {sf = $2}
         END {printf "%d %d %d %d\n", t / 1024, a / 1024, st / 1024, sf / 1024}' \
      /proc/meminfo
  )
  oom_kills=$(awk '/^oom_kill / {print $2}' /proc/vmstat 2>/dev/null)
  pressure=$(awk '/^full/ {for (i = 2; i <= NF; i++) if ($i ~ /^avg10=/) {
                 sub("avg10=", "", $i); print $i}}' /proc/pressure/memory 2>/dev/null)
  disk_free=$(df -Pm / 2>/dev/null | awk 'NR == 2 {print $4}')
  largest=$(ps -eo rss=,pid=,comm= --sort=-rss 2>/dev/null | head -n 4 |
            awk '{printf "%s[%s]=%dMiB ", $3, $2, $1 / 1024}')

  # pytest controllers and xdist workers: a python whose command line names
  # pytest, or execnet's worker bootstrap (sys.stdin.readline).
  declare -A current=()
  pytest_rss=""
  while read -r pid rss; do
    current[$pid]=$rss
    pytest_rss+="${rss},"
  done < <(
    ps -eo pid=,rss=,comm=,args= 2>/dev/null |
      awk '$3 ~ /^python/ && ($0 ~ /pytest/ || $0 ~ /sys\.stdin\.readline/) {
             printf "%s %d\n", $1, $2 / 1024}'
  )
  for pid in "${!previous[@]}"; do
    if [ -z "${current[$pid]+set}" ]; then
      echo "MEMORY $now pytest process $pid ended; last seen at" \
        "${previous[$pid]}MiB RSS (tests/conftest.py's memory guard ends a" \
        "pytest process at ${guard_gb} GB with exit 3 and says so on stderr)"
    fi
  done
  previous=()
  for pid in "${!current[@]}"; do
    previous[$pid]=${current[$pid]}
  done
  unset current

  echo "MEMORY $now" \
    "used=$((total - available))MiB/${total}MiB" \
    "available=${available}MiB" \
    "swap_used=$((swap_total - swap_free))MiB/${swap_total}MiB" \
    "psi_full_avg10=${pressure:-n/a}" \
    "oom_kills=${oom_kills:-n/a}" \
    "disk_free=${disk_free:-n/a}MiB" \
    "pytest_rss=[${pytest_rss%,}]MiB guard=${guard_gb}GB" \
    "top: ${largest}"
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    free -m
    df -Pm / /tmp
    ps -eo pid,rss,etimes,args --sort=-rss | head -n 13 | cut -c1-200
  } >> "$samples" 2>&1
  sleep "$interval" &
  sleeper=$!
  wait "$sleeper"
  sleeper=""
done
