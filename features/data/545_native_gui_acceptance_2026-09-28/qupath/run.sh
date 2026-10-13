#!/usr/bin/env bash
set -euo pipefail
base=/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation/qupath
mkdir -p "$base/home" "$base/tmp" "$base/prefs" "$base/cache" "$base/config"
export HOME="$base/home" TMPDIR="$base/tmp" XDG_CACHE_HOME="$base/cache" XDG_CONFIG_HOME="$base/config"
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAVA_TOOL_OPTIONS="-Xmx2g -XX:ActiveProcessorCount=2 -Dprism.order=sw -Djava.awt.headless=false -Duser.home=$base/home -Djava.io.tmpdir=$base/tmp -Djava.util.prefs.userRoot=$base/prefs"
exec /mnt/wd4tb/scratch/spacr-completion/worktree/tools/run_capped.sh 4G timeout --signal=TERM --kill-after=10s 180s xvfb-run -a -e "$base/xvfb.log" -s '-screen 0 1280x900x24 -nolisten tcp' bash "$base/inside-display.sh"
