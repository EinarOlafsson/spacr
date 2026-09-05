#!/usr/bin/env bash
# Run a command under a HARD memory cap the kernel enforces.
#
# WHY THIS EXISTS, and why the obvious alternative does not work. spaCR's
# test suite twice took this developer's whole machine down -- on 2026-09-04
# VS Code died, and the second time the kernel's OOM killer took gnome-shell
# with it and logged the user out. A single pytest process had reached 92 GB.
#
# The first attempt at a safeguard was a poller: a daemon that read
# /proc/meminfo every three seconds and killed the largest offender. It fired
# five times and STILL lost, because a process going from nothing to ninety
# gigabytes outruns a three-second poll, and because killing one process then
# waiting ten seconds is far too polite when several are climbing at once.
#
# A cgroup is not a poller. `MemoryMax` is checked by the kernel on every
# allocation, so the process that asks for too much is the process that dies,
# immediately, and nothing else on the machine notices.
#
#   tools/run_capped.sh 8G python -m pytest -q tests/qt/some_test.py
#   CAP=16G tools/run_capped.sh python -m pytest -q tests/qt/
#
# The first argument is the cap when it looks like one (8G, 512M); otherwise
# $CAP is used, defaulting to 8G. `MemorySwapMax=0` matters: without it a
# runaway process swaps instead of dying and takes the machine down slowly
# rather than quickly.
set -uo pipefail

CAP="${CAP:-8G}"
if [[ "${1:-}" =~ ^[0-9]+[MG]$ ]]; then
    CAP="$1"; shift
fi

if [[ $# -eq 0 ]]; then
    echo "usage: $0 [8G] <command...>" >&2
    exit 2
fi

exec systemd-run --user --scope --quiet \
    -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    -- "$@"
