#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/n448-n440-n421-runtime-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=$TASK_SCRATCH/consumer-tmp XDG_CONFIG_HOME=$TASK_SCRATCH/consumer-config XDG_CACHE_HOME=$TASK_SCRATCH/consumer-cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/consumer-pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
tools/run_capped.sh 4G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p pytest_timeout tests/qt/test_n448_complete_curation_status.py tests/qt/test_n448_curation_status_locales.py tests/qt/test_n421_legacy_measurement_output_translations.py -q --tb=short --junitxml="$TASK_SCRATCH/consumers.xml" --basetemp="$TASK_SCRATCH/consumer-pytest-tmp" -o cache_dir="$TASK_SCRATCH/consumer-pytest-cache" > "$TASK_SCRATCH/consumers.log" 2>&1
TASK_RC=$?
printf '%s\n' "$TASK_RC" > "$TASK_SCRATCH/consumers.exit"
tools/run_capped.sh 4G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/home-b-n440-source-picker-20261015/check-compiled-consumer.py > "$TASK_SCRATCH/n440-compiled-consumer.json" 2> "$TASK_SCRATCH/n440-compiled-consumer.log"
TASK_CONSUMER_RC=$?
printf '%s\n' "$TASK_CONSUMER_RC" > "$TASK_SCRATCH/n440-compiled-consumer.exit"
if [ "$TASK_RC" -ne 0 ]; then exit "$TASK_RC"; fi
exit "$TASK_CONSUMER_RC"
