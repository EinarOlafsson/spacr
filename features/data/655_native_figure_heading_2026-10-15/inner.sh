#!/bin/bash
set -eu
out=${N655_SCRATCH:?}
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 QT_ACCESSIBILITY=1 QT_QPA_PLATFORM=xcb GDK_BACKEND=x11
export CUDA_VISIBLE_DEVICES='' SPACR_NO_GL=1 LIBGL_ALWAYS_SOFTWARE=1
export GSETTINGS_BACKEND=memory
export XDG_RUNTIME_DIR="$out/runtime" XDG_CONFIG_HOME="$out/config" XDG_CACHE_HOME="$out/cache"
export XDG_DATA_HOME="$out/data" XDG_STATE_HOME="$out/state"
export PULSE_SERVER="unix:$out/no-audio" SPEECHD_ADDRESS="unix_socket:$out/no-speech"
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$out/orca-prefs"
chmod 700 "$XDG_RUNTIME_DIR"
unset AT_SPI_BUS_ADDRESS NO_AT_BRIDGE DBUS_STARTER_ADDRESS WAYLAND_DISPLAY
/usr/bin/orca --disable=speech --disable=braille --user-prefs="$out/orca-prefs" --debug-file="$out/orca-debug.log" > "$out/orca.log" 2>&1 &
orca_pid=$!
app_pid=''
trap 'kill "$orca_pid" ${app_pid:-} 2>/dev/null || true; wait "$orca_pid" ${app_pid:-} 2>/dev/null || true' EXIT
for n in $(seq 1 50); do
    if rg -q 'Starting ATSPI registry' "$out/orca-debug.log" 2>/dev/null; then break; fi
    if ! kill -0 "$orca_pid" 2>/dev/null; then exit 2; fi
    sleep .1
done
/home/olafsson/anaconda3/envs/spacr/bin/python "$out/heading-app.py" > "$out/app.log" 2>&1 &
app_pid=$!
for n in $(seq 1 100); do
    if rg -q '^READY ' "$out/app.log"; then break; fi
    if ! kill -0 "$app_pid" 2>/dev/null; then exit 3; fi
    sleep .1
done
/usr/bin/timeout 18s /usr/bin/python3 "$out/read-heading.py" "$app_pid" > "$out/reader.json"
sleep 1
