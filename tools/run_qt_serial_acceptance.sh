#!/usr/bin/env bash
set -euo pipefail

# This runner is called only inside the hosted 12 GiB systemd scope. Refuse
# an uncapped process rather than reporting an invalid acceptance result.
cgroup=$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)
if [[ -z "$cgroup" ]]; then
    echo "N47 requires a cgroup v2 memory scope" >&2
    exit 2
fi
memory_dir="/sys/fs/cgroup${cgroup%/}"
if [[ ! -r "$memory_dir/memory.max" || ! -r "$memory_dir/memory.swap.max" ]]; then
    echo "N47 cannot read the hard memory and swap caps" >&2
    exit 2
fi
memory_max=$(<"$memory_dir/memory.max")
swap_max=$(<"$memory_dir/memory.swap.max")
if [[ "$memory_max" != "12884901888" || "$swap_max" != "0" ]]; then
    echo "N47 needs MemoryMax=12G and MemorySwapMax=0; got $memory_max/$swap_max" >&2
    exit 2
fi
if [[ "${SPACR_TEST_MEMORY_GB:-}" != "10.8" || "${CUDA_VISIBLE_DEVICES-unset}" != "" ]]; then
    echo "N47 needs the 10.8 GiB pytest guard and hidden CUDA" >&2
    exit 2
fi
if [[ -z "${SPACR_QT_SERIAL_RSS_JOURNAL:-}" ]]; then
    echo "N47 needs a durable RSS journal path" >&2
    exit 2
fi
if [[ "${QT_QPA_PLATFORM:-}" != "xcb" || -z "${DISPLAY:-}" ]]; then
    echo "N47 needs an X display; offscreen cannot measure the real Home pixels" >&2
    exit 2
fi

cd "${GITHUB_WORKSPACE:?}"
source_sha=$(git rev-parse HEAD)
if [[ "$source_sha" != "${GITHUB_SHA:?}" ]]; then
    echo "N47 checkout does not match dispatched source: $source_sha != $GITHUB_SHA" >&2
    exit 2
fi
read -r host_total_kib host_available_kib < <(
    awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2}
         END {print total+0, available+0}' /proc/meminfo
)
if (( host_total_kib < 14 * 1024 * 1024 || host_available_kib < 12 * 1024 * 1024 )); then
    echo "N47 host lacks room for the 12 GiB scope: MemTotal=${host_total_kib}KiB MemAvailable=${host_available_kib}KiB" >&2
    exit 2
fi
unset SPACR_PYTEST_FILE_SHARD_INDEX SPACR_PYTEST_FILE_SHARD_COUNT PYTEST_ADDOPTS
echo "N47 serial source=$source_sha host.MemTotal=${host_total_kib}KiB host.MemAvailable=${host_available_kib}KiB cgroup=$cgroup memory.max=$memory_max memory.swap.max=$swap_max guard=${SPACR_TEST_MEMORY_GB}GiB"
echo "N47 native core_pattern=$(</proc/sys/kernel/core_pattern) core_limit_blocks=$(ulimit -c) python_command=$(command -v python)"
python tools/can_this_display_be_measured.py
exec python - tests/qt -v --tb=short -p no:randomly \
    -p tools.pytest_plugins.qt_serial_rss_journal \
    -o faulthandler_timeout=900 --timeout=1200 --timeout-method=thread <<'PY'
import sys

from PySide6.QtWidgets import QApplication

app = QApplication([])
if app.platformName() != "xcb":
    raise SystemExit("N47 did not construct a real X-backed QApplication")
app.setApplicationName("pytest-qt-qapp")

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
PY
