#!/bin/bash
set -eu
out=${N655_SCRATCH:?}
export GDK_BACKEND=x11
export XDG_RUNTIME_DIR="$out/runtime" XDG_CONFIG_HOME="$out/config" XDG_CACHE_HOME="$out/cache"
export SPACR_PROBE_APP=spacr-app SPACR_PROBE_EXPECTED='High contrast'
cd "${N655_SOURCE_ROOT:?}"
case "${1:?}" in
 atspi) runner=probe.sh ;;
 orca) runner=orca-probe.sh ;;
 *) exit 2 ;;
esac
tools/run_capped.sh 4G /usr/bin/timeout 35s /usr/bin/dbus-run-session -- /usr/bin/xvfb-run -a -s '-screen 0 1400x1000x24' "$out/$runner"
