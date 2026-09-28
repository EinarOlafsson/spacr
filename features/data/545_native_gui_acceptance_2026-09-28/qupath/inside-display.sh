#!/usr/bin/env bash
set -euo pipefail
base=/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation/qupath
printf '%s\n' "$DISPLAY" > "$base/private-display.txt"
cat /proc/self/cgroup > "$base/cgroup.txt"
relative=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
cat "/sys/fs/cgroup$relative/memory.max" > "$base/memory.max"
cat "/sys/fs/cgroup$relative/memory.swap.max" > "$base/memory.swap.max"
"$base/QuPath/bin/QuPath" -q -Dqupath.startup.script="$base/interop.groovy" > "$base/gui.log" 2>&1 &
app_pid=$!
printf '%s\n' "$app_pid" > "$base/app.pid"
trap 'kill "$app_pid" 2>/dev/null || true; wait "$app_pid" 2>/dev/null || true' EXIT
for count in $(seq 1 150); do
    if [[ -f "$base/GUI_READY" ]]; then
        import -window root "$base/private-display.png"
        break
    fi
    if ! kill -0 "$app_pid" 2>/dev/null; then break; fi
    sleep 1
done
wait "$app_pid"
trap - EXIT
test -s "$base/gui-receipt.json"
test -s "$base/private-display.png"
