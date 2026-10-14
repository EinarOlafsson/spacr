#!/bin/bash
set -u
out=/mnt/wd4tb/scratch/n655-external-reader-20261014
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 QT_ACCESSIBILITY=1 QT_QPA_PLATFORM=xcb
export GDK_BACKEND=x11
export CUDA_VISIBLE_DEVICES='' SPACR_NO_GL=1 LIBGL_ALWAYS_SOFTWARE=1
export XDG_CONFIG_HOME="$out/config" XDG_CACHE_HOME="$out/cache" XDG_RUNTIME_DIR="$out/runtime"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
unset AT_SPI_BUS_ADDRESS NO_AT_BRIDGE DBUS_STARTER_ADDRESS WAYLAND_DISPLAY
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
/home/olafsson/anaconda3/envs/spacr/bin/python "$out/${SPACR_PROBE_APP:-control}.py" > "$out/${SPACR_PROBE_APP:-control}.log" 2>&1 &
child=$!
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
for n in $(seq 1 100); do
  if rg -q '^READY ' "$out/${SPACR_PROBE_APP:-control}.log"; then break; fi
  sleep .1
done
/usr/bin/timeout 18s /usr/bin/python3 "$out/read-bus.py" "$child" "${SPACR_PROBE_EXPECTED:-External reader control button}" > "$out/${SPACR_PROBE_APP:-control}-reader.json"
result=$?
exit "$result"
