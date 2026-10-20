#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-tooltip-shortcuts-contracts-20261015
exec > "$out/final.log" 2>&1
trap 'printf "%s\n" "$?" > "$out/final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$out/pytest-offline.py" -q tests/test_settings_tooltip_quality.py::test_real_default_claims_have_no_unrecorded_drift tests/qt/test_make_masks_shortcuts_otsu_and_object_edits.py::test_the_shortcuts_sit_right_of_the_three_views tests/qt/test_make_masks_shortcuts_otsu_and_object_edits.py::test_every_hint_is_on_the_screen_with_its_keys_and_its_sentence --junitxml="$out/final.xml"
