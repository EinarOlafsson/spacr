#!/usr/bin/env bash
# Take a turn on the GPU, politely, and hand it back.
#
# The maintainer's rule, 2026-09-25, for every session on this machine:
#   * start a GPU job only after the GPU has been idle for 6 minutes;
#   * when several GPU jobs are queued, leave 10 minutes between them, so
#     another session gets a chance at the card in the gap;
#   * tell the other sessions what is running.
# Native Windows uses gpu_turn.ps1, GPU Engine counters, and a shared
# ProgramData queue with a global mutex; the same idle/gap defaults apply.
#
# "Idle" means no CUDA compute process on any GPU. The desktop (Xorg,
# gnome-shell, browsers) always holds graphics contexts and keeps the
# utilisation counter at 20-50 %, so utilisation is useless as a signal;
# compute processes are what another session's work looks like.
# Chrome never blocks the queue, even when nvidia-smi lists it as compute.
#
#   tools/gpu_turn.sh <label> <command...>
#   tools/gpu_turn.sh 507-cellpose3-timing python tools/some_probe.py
#
# State lives in ~/.spacr/gpu/ so every checkout and every session sees the
# same queue:
#   lock          flock(1) target; one holder at a time
#   holder        who holds it now (label, pid, start, checkout)
#   last_release  epoch seconds of the last turn handed back
#   log           one line per wait, start and finish
#
# Tunables (seconds): GPU_IDLE_S (360), GPU_GAP_S (600), GPU_POLL_S (15).
#
# Priority (maintainer, 2026-10-25): jobs whose label starts with a prefix in
# GPU_PRIORITY_PREFIXES (default "plaque-") go first. A job without priority
# that holds the queue while still waiting hands the lock back as soon as a
# priority job is queued, and takes it again once no priority job is queued
# or running. Priority jobs are found from the process table (any copy of
# gpu_turn.sh with a priority label that uses this same queue directory), so
# frozen older copies count too. A job
# that has already started is never stopped.
set -uo pipefail

case "${OSTYPE:-}" in
    msys*|cygwin*)
        windows_gpu_script="${BASH_SOURCE[0]//\\//}"
        if [[ "$OSTYPE" == cygwin* && "$windows_gpu_script" == /* ]]; then
            windows_gpu_script=$(cygpath -w "$windows_gpu_script") || exit 1
            windows_gpu_script="${windows_gpu_script//\\//}"
        fi
        exec "${SYSTEMROOT:-C:/Windows}/System32/WindowsPowerShell/v1.0/powershell.exe" \
            -NoProfile -ExecutionPolicy Bypass -File "${windows_gpu_script%/*}/gpu_turn.ps1" "$@"
        ;;
esac

IDLE_S=${GPU_IDLE_S:-360}
GAP_S=${GPU_GAP_S:-600}
POLL_S=${GPU_POLL_S:-15}
DIR="$HOME/.spacr/gpu"
mkdir -p "$DIR"
LOG="$DIR/log"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <label> <command...>" >&2
    exit 2
fi
LABEL="$1"; shift

say() { echo "$(date -Is) [$LABEL] $*" | tee -a "$LOG" >&2; }

PRIORITY_PREFIXES=${GPU_PRIORITY_PREFIXES-plaque-}

is_priority() {
    local prefix
    for prefix in $PRIORITY_PREFIXES; do
        [[ "$1" == "$prefix"* ]] && return 0
    done
    return 1
}

same_queue() {
    local environment entry_variable home=""
    environment=()
    mapfile -d '' -t environment < "$1/environ" 2>/dev/null || return 1
    for entry_variable in "${environment[@]}"; do
        [[ "$entry_variable" == HOME=* ]] && home=${entry_variable#HOME=}
    done
    [[ -n "$home" && "$home/.spacr/gpu" -ef "$DIR" ]]
}

priority_queued() {
    local entry argv i
    [[ -n "$PRIORITY_PREFIXES" ]] || return 1
    for entry in /proc/[0-9]*; do
        [[ "${entry#/proc/}" == "$$" ]] && continue
        argv=()
        mapfile -d '' -t argv < "$entry/cmdline" 2>/dev/null || continue
        for (( i = 0; i + 1 < ${#argv[@]}; i++ )); do
            if [[ "${argv[i]}" == *gpu_turn.sh ]]; then
                if is_priority "${argv[i+1]}" && same_queue "$entry"; then
                    return 0
                fi
                break
            fi
        done
    done
    return 1
}

compute_pids() {
    local pid executable
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                 | tr -d ' ' | grep -v '^$'); do
        executable=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        executable=${executable% (deleted)}
        case "${executable##*/}" in
            chrome|chromium|chromium-browser|google-chrome|google-chrome-stable)
                continue ;;
        esac
        tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -Fxq -- '--type=gpu-process' \
            && continue
        echo "$pid"
    done
}

exec 9>"$DIR/lock"
say "queued (pid $$, $(pwd))"
flock 9
say "holds the queue; waiting for ${IDLE_S}s idle and ${GAP_S}s since the last turn"

idle_since=$(date +%s)
while true; do
    if ! is_priority "$LABEL" && priority_queued; then
        say "hands the queue to a waiting priority job"
        flock -u 9
        while priority_queued; do
            sleep "$POLL_S"
        done
        flock 9
        say "holds the queue again; waiting for ${IDLE_S}s idle and ${GAP_S}s since the last turn"
        idle_since=$(date +%s)
        continue
    fi
    now=$(date +%s)
    busy=$(compute_pids)
    if [[ -n "$busy" ]]; then
        idle_since=$now
    fi
    last=$(cat "$DIR/last_release" 2>/dev/null || echo 0)
    if (( now - idle_since >= IDLE_S && now - last >= GAP_S )); then
        break
    fi
    sleep "$POLL_S"
done

printf 'label=%s pid=%s start=%s checkout=%s\n' \
    "$LABEL" "$$" "$(date -Is)" "$(pwd)" > "$DIR/holder"
say "START: $*"
"$@"
rc=$?
date +%s > "$DIR/last_release"
: > "$DIR/holder"
say "FINISH rc=$rc"
flock -u 9
exit $rc
