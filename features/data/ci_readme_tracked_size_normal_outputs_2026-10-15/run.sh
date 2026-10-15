#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SPACR_NO_GL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages"
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
out=/mnt/wd4tb/scratch/n43-readme-size-maintenance-20261015/run
python_bin=/home/olafsson/anaconda3/envs/spacr/bin/python
trap 'rc=$?; printf "%s\n" "$rc" > "$out/overall.exit"' EXIT
"$python_bin" "$out/check-preservation.py" initial > "$out/initial-preservation.log" 2>&1
printf 'build\n' > "$out/phase"
tools/run_capped.sh 8G "$python_bin" "$out/guarded-command.py" build > "$out/build.log" 2>&1
printf '0\n' > "$out/build.exit"
"$python_bin" "$out/check-preservation.py" after-build > "$out/after-build-preservation.log" 2>&1
printf 'check\n' > "$out/phase"
tools/run_capped.sh 8G "$python_bin" "$out/guarded-command.py" build --check > "$out/check.log" 2>&1
printf '0\n' > "$out/check.exit"
printf 'original-eleven-cases\n' > "$out/phase"
tools/run_capped.sh 8G "$python_bin" "$out/guarded-command.py" pytest -q --disable-warnings --junitxml="$out/tests.xml" tests/test_the_light_install_numbers_are_honest.py::test_the_checkout_size_in_the_readme_matches_the_tree tests/test_documentation_i18n.py::test_old_readme_measurement_reviews_are_retained_without_becoming_current tests/test_readme_is_built_from_images.py::test_each_translation_has_the_english_block_structure > "$out/tests.log" 2>&1
printf '0\n' > "$out/tests.exit"
"$python_bin" "$out/check-preservation.py" final > "$out/final-preservation.log" 2>&1
printf 'terminal-pass\n' > "$out/phase"
