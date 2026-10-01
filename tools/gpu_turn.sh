#!/usr/bin/env bash
# Take a turn on the GPU, politely, and hand it back.
#
# The maintainer's rule, 2026-09-25, for every session on this machine:
#   * start a GPU job only after the GPU has been idle for 6 minutes;
#   * when several GPU jobs are queued, leave 10 minutes between them, so
#     another session gets a chance at the card in the gap;
#   * tell the other sessions what is running.
#
# "Idle" means no CUDA compute process on any GPU. The desktop (Xorg,
# gnome-shell, browsers) always holds graphics contexts and keeps the
# utilisation counter at 20-50 %, so utilisation is useless as a signal;
# compute processes are what another session's work looks like.
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
set -uo pipefail

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

compute_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -v '^$' || true
}

exec 9>"$DIR/lock"
say "queued (pid $$, $(pwd))"
flock 9
say "holds the queue; waiting for ${IDLE_S}s idle and ${GAP_S}s since the last turn"

idle_since=$(date +%s)
while true; do
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
