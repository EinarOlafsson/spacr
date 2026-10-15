#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015 || exit 90
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
printf 'LOCK_ACQUIRED\n' > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-audit-help.log
tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/audit-three-tab-api.py actual >> /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-audit-help.log 2>&1
rc=$?
printf '%s\n' "$rc" > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-audit.exit
if [ "$rc" -eq 0 ]; then
 tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_help_search_index.py >> /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-audit-help.log 2>&1
 rc=$?
 printf '%s\n' "$rc" > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-help.exit
fi
printf '%s\n' "$rc" > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-audit-help.exit
exit "$rc"
