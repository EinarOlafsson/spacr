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
# "Idle" means no non-display process using more than GPU_BUSY_MIB (4096 MiB,
# i.e. 4 GB, maintainer policy 2026-11-06) of VRAM on any GPU. The desktop
# (Xorg, gnome-shell, browsers) always holds graphics contexts and keeps the
# utilisation counter at 20-50 %, so utilisation is useless as a signal;
# big compute processes are what another session's work looks like. A CPU-bound
# job that merely holds a small CUDA context (a few hundred MiB) is not busy:
# it once blocked the exclusive queue for over 2 hours. Chrome and the Mullvad
# VPN window never block the queue, even when nvidia-smi lists them as compute.
#
#   tools/gpu_turn.sh <label> <command...>
#   tools/gpu_turn.sh 507-cellpose3-timing python tools/some_probe.py
#
# State lives in ~/.spacr/gpu/ so every checkout and every session sees the
# same queue:
#   lock          flock(1) target; one holder at a time
#   lock.d        the same lock where flock(1) is missing (macOS): an atomic
#                 mkdir holding the holder's pid; a dead holder's is reclaimed
#   holder        who holds it now (label, pid, start, checkout)
#   last_release  epoch seconds of the last turn handed back
#   log           one line per wait, start and finish
#
# Tunables (seconds): GPU_IDLE_S (360), GPU_GAP_S (600), GPU_POLL_S (15).
# GPU_BUSY_MIB (4096): VRAM above which a non-display process counts as busy.
#
# Concurrent jobs (maintainer, 2026-11-18): a job that declares its need,
#   tools/gpu_turn.sh --vram-gb N <label> <command...>
# skips the exclusive queue, the idle wait and the gap, and starts as soon as
#   * free VRAM (nvidia-smi) minus the needs other declared jobs announced in
#     the last GPU_RESERVE_S seconds (120) is at least N + GPU_VRAM_MARGIN_GB
#     (2) GB,
#   * the GPU is below GPU_MAX_TEMP_C (80) C, and
#   * there is no ram_guard ALERT ($HOME/.claude/ram_guard/ALERT).
# It refuses to start (exit 3) unless power.limit is GPU_POWER_LIMIT_W (280) W,
# the cap set after the 2026-11-16 crashes. Every start is logged with the free
# VRAM, the temperature and the jobs already on the card. Nothing running is
# ever stopped, and a job without --vram-gb keeps the exclusive turn above.
#
# Priority (maintainer, 2026-10-25): jobs whose label starts with a prefix in
# GPU_PRIORITY_PREFIXES (default "plaque-") go first. A job without priority
# that holds the queue while still waiting hands the lock back as soon as a
# priority job is queued, and takes it again once no priority job is queued
# or running. Priority jobs are found from the process table (any copy of
# gpu_turn.sh with a priority label that uses this same queue directory), so
# frozen older copies count too. A job
# that has already started is never stopped.
#
# macOS (2026-10-26): there is no flock(1), no nvidia-smi and no /proc, and
# /bin/bash is 3.2. The queue takes the mkdir lock above through the same
# lock_queue/unlock_queue the priority hand-back uses, priority_queued answers
# no because there is no process table to read (and so never reaches the
# bash 4.4 mapfile below), and the lock and the gap are the only coordination.
# Timestamps use a date(1) format GNU and BSD date both accept.
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

VRAM_GB=""
if [[ "${1:-}" == "--vram-gb" ]]; then
    VRAM_GB="${2:-}"
    shift 2 || true
    if ! [[ "$VRAM_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "usage: $0 [--vram-gb N] <label> <command...> (N in GB)" >&2
        exit 2
    fi
fi
if [[ $# -lt 2 ]]; then
    echo "usage: $0 [--vram-gb N] <label> <command...>" >&2
    exit 2
fi
LABEL="$1"; shift

stamp() { date +%Y-%m-%dT%H:%M:%S%z; }

say() { echo "$(stamp) [$LABEL] $*" | tee -a "$LOG" >&2; }

HOLDS_MKDIR_LOCK=0

lock_queue() {
    if command -v flock >/dev/null 2>&1; then
        flock 9
        return
    fi
    local holder
    while ! mkdir "$DIR/lock.d" 2>/dev/null; do
        holder=$(cat "$DIR/lock.d/pid" 2>/dev/null || true)
        if [[ -n "$holder" ]] && ! kill -0 "$holder" 2>/dev/null; then
            rm -rf "$DIR/lock.d"
            continue
        fi
        sleep "$POLL_S"
    done
    echo "$$" > "$DIR/lock.d/pid"
    HOLDS_MKDIR_LOCK=1
}

unlock_queue() {
    if command -v flock >/dev/null 2>&1; then
        flock -u 9
        return
    fi
    if (( HOLDS_MKDIR_LOCK )); then
        rm -rf "$DIR/lock.d"
        HOLDS_MKDIR_LOCK=0
    fi
}

trap '(( HOLDS_MKDIR_LOCK )) && rm -rf "$DIR/lock.d"' EXIT

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
    [[ -d /proc/self ]] || return 1
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
    local pid mem executable threshold=${GPU_BUSY_MIB:-4096}
    while IFS=, read -r pid mem; do
        [[ -n "$pid" ]] || continue
        # A memory reading that is not a number (e.g. [N/A]) stays busy.
        if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem <= threshold )); then
            continue
        fi
        executable=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        executable=${executable% (deleted)}
        case "${executable##*/}" in
            chrome|chromium|chromium-browser|google-chrome|google-chrome-stable|mullvad-gui)
                continue ;;
        esac
        tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -Fxq -- '--type=gpu-process' \
            && continue
        echo "$pid"
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
             | tr -d ' ' | grep -v '^$')
}

gpu_reading() {
    nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
}

reserved_gb() {
    local file now total=0 pid when need
    now=$(date +%s)
    for file in "$DIR"/reserve/*; do
        [[ -f "$file" ]] || continue
        read -r pid when need < "$file" || continue
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$file"
            continue
        fi
        (( now - when < ${GPU_RESERVE_S:-120} )) || continue
        total=$(awk -v a="$total" -v b="$need" 'BEGIN{print a + b}')
    done
    echo "$total"
}

co_running() {
    local apps holder
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
           | tr -d ' ' | awk -F, 'NF==2 && $1 != "" {printf "%s%s:%sMiB", sep, $1, $2; sep=" "}')
    holder=$(cat "$DIR/holder" 2>/dev/null || true)
    echo "compute=[${apps:-none}] turn_holder=[${holder:-none}]"
}

# Run a declared job after checking the current power cap and available capacity.
run_concurrent() {
    local limit free temp spare margin max_temp power alert why last_why=""
    limit=${GPU_POWER_LIMIT_W:-280}
    margin=${GPU_VRAM_MARGIN_GB:-2}
    max_temp=${GPU_MAX_TEMP_C:-80}
    alert=${GPU_RAM_ALERT:-$HOME/.claude/ram_guard/ALERT}
    mkdir -p "$DIR/reserve"
    exec 8>"$DIR/vram.lock"
    say "queued concurrently (pid $$, $(pwd)); needs ${VRAM_GB} GB + ${margin} GB margin"
    while true; do
        flock 8
        power=$(gpu_reading power.limit)
        if ! awk -v p="$power" -v l="$limit" 'BEGIN{exit !(p != "" && p + 0 == l + 0)}'; then
            say "REFUSED: --vram-gb needs the GPU power limit at ${limit} W; nvidia-smi reports '${power:-nothing}' W. Set it (sudo nvidia-smi -pl ${limit}) and try again."
            exit 3
        fi
        free=$(gpu_reading memory.free)
        temp=$(gpu_reading temperature.gpu)
        spare=$(awk -v f="${free:-0}" -v r="$(reserved_gb)" 'BEGIN{printf "%.1f", f / 1024 - r}')
        if [[ ! -e "$alert" ]] && [[ -n "$temp" ]] \
           && awk -v s="$spare" -v n="$VRAM_GB" -v m="$margin" -v t="$temp" -v x="$max_temp" \
                  'BEGIN{exit !(s >= n + m && t < x)}'; then
            echo "$$ $(date +%s) $VRAM_GB" > "$DIR/reserve/$$"
            say "START concurrent: needs ${VRAM_GB} GB, free ${spare} GB (after reservations), ${temp} C, power ${power} W; $(co_running): $*"
            flock -u 8
            break
        fi
        why="free ${spare} GB, ${temp:-?} C$([[ -e "$alert" ]] && echo ', ram_guard ALERT')"
        if [[ "$why" != "${last_why:-}" ]]; then
            say "waits: needs ${VRAM_GB} + ${margin} GB below ${max_temp} C; now ${why}"
            last_why=$why
        fi
        flock -u 8
        sleep "$POLL_S"
    done
    # Item 1011: the declared VRAM becomes the child's budget (spaCR's gpu_vram_budget_gb).
    SPACR_GPU_VRAM_GB="$VRAM_GB" "$@"
    rc=$?
    rm -f "$DIR/reserve/$$"
    say "FINISH concurrent rc=$rc"
    exit $rc
}

if [[ -n "$VRAM_GB" ]]; then
    run_concurrent "$@"
fi

exec 9>"$DIR/lock"
say "queued (pid $$, $(pwd))"
lock_queue
say "holds the queue; waiting for ${IDLE_S}s idle and ${GAP_S}s since the last turn"

idle_since=$(date +%s)
while true; do
    if ! is_priority "$LABEL" && priority_queued; then
        say "hands the queue to a waiting priority job"
        unlock_queue
        while priority_queued; do
            sleep "$POLL_S"
        done
        lock_queue
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
    "$LABEL" "$$" "$(stamp)" "$(pwd)" > "$DIR/holder"
say "START: $*"
"$@"
rc=$?
date +%s > "$DIR/last_release"
: > "$DIR/holder"
say "FINISH rc=$rc"
unlock_queue
exit $rc
