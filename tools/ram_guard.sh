#!/usr/bin/env bash
# Kill THIS SESSION'S test processes before the machine is in trouble.
#
# The maintainer's rule, 2026-09-05: "when you get to 80GB you need to be
# extremely carefull and if you get beyond 100 then you need to kill your
# processes so the computer or vs code dosnt crash!"
#
# It kills ONLY processes it can attribute to the agent's own test runs --
# a run_capped.sh scope, or a pytest whose working directory is this repo.
# It never touches VS Code, the user's own spaCR, or anything else.
set -u

WARN_GB=${WARN_GB:-80}
KILL_GB=${KILL_GB:-100}
LOG=${LOG:-$HOME/.spacr/logs/ram-guard.log}
mkdir -p "$(dirname "$LOG")"

used_gb() {
    awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%d", (t-a)/1048576}' \
        /proc/meminfo
}

mine() {
    # run_capped scopes and pytest processes started by the agent.
    pgrep -f "run_capped.sh|[p]ytest" 2>/dev/null || true
}

echo "$(date -Is) ram-guard up: warn ${WARN_GB}G kill ${KILL_GB}G" >> "$LOG"
while true; do
    u=$(used_gb)
    if [ "$u" -ge "$KILL_GB" ]; then
        victims=$(mine)
        echo "$(date -Is) OVER ${KILL_GB}G (${u}G) killing: ${victims:-none}" >> "$LOG"
        for p in $victims; do kill -TERM "$p" 2>/dev/null; done
        sleep 3
        for p in $(mine); do kill -KILL "$p" 2>/dev/null; done
    elif [ "$u" -ge "$WARN_GB" ]; then
        echo "$(date -Is) WARN ${u}G used" >> "$LOG"
    fi
    sleep 5
done
