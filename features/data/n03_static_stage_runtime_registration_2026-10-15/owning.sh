#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n03-stage-invariant-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n03-stage-invariant-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p xdist.plugin -n 0 tests/qt/test_home_v2.py::test_every_app_has_a_stage_and_it_is_written_down_once tests/qt/test_home_v2.py::test_declared_stable_registration_survives_maturity_while_undeclared_defaults_alpha tests/qt/test_634_organism_alpha_pages.py::test_the_species_pages_are_registered_under_their_own_switch tests/qt/test_634_organism_alpha_pages.py::test_toxoplasma_shows_with_every_alpha_switch_off tests/qt/test_app_maturity.py::test_a_module_that_declares_nothing_lands_on_alpha tests/qt/test_classify_public_maturity_route.py -q --tb=short --junitxml="$TASK_SCRATCH/owning.xml" --basetemp="$TASK_SCRATCH/owning-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/owning.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/owning.exit"
exit "$TASK_RC"
