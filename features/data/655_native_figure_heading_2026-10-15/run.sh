#!/bin/bash
set -eu
export N655_SOURCE_ROOT=/mnt/wd4tb/spacr-worktrees/codex-home-b-n655-native-heading-20261015
export N655_SCRATCH=/mnt/wd4tb/scratch/home-b-n655-native-heading-20261015
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TMPDIR="$N655_SCRATCH"
cd "$N655_SOURCE_ROOT"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 4G /usr/bin/time -v -o "$N655_SCRATCH/time.txt" /usr/bin/timeout 40s /usr/bin/dbus-run-session -- /usr/bin/xvfb-run -a -s '-screen 0 1200x900x24' /bin/bash "$N655_SCRATCH/inner.sh"
