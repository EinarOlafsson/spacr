#!/bin/bash
set -eu
export N655_SOURCE_ROOT=/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export N655_SCRATCH=/mnt/wd4tb/scratch/home-b-n655-native-figure-actions-20261015
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$N655_SCRATCH"
cd "$N655_SCRATCH"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/native-accessibility.lock "$N655_SOURCE_ROOT/tools/run_capped.sh" 2G /usr/bin/time -v -o "$N655_SCRATCH/time.txt" /usr/bin/timeout 90s /usr/bin/dbus-run-session -- /usr/bin/xvfb-run -a -s '-screen 0 1000x800x24' /bin/bash "$N655_SCRATCH/inner.sh"
