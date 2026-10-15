#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
out=/mnt/wd4tb/scratch/home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$out/tmp"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 2G /usr/bin/time -v -o "$out/artifact-checks.time" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_model_cards_are_current.py tests/test_the_model_zoo_has_a_page_of_its_own.py -k 'not how_to_read_fits_the_card and not card_links_resolve and not family_defaults_match_the_zoo_markers and not the_app_the_readme_and_the_cards_open_a_family_on_the_same_version and not a_candidate_never_becomes_a_family_default' --basetemp="$out/pytest-artifacts" --junitxml="$out/artifact-checks.xml" > "$out/artifact-checks.log" 2>&1
result=$?
printf '%s\n' "$result" > "$out/artifact-checks.exit"
exit "$result"
