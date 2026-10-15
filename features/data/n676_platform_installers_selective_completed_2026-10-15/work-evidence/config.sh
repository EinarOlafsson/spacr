#!/usr/bin/env bash
# Selective visual repair only. Root supplies the actual frozen combined K commit.
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
N=/media/carruthers/mnt3/claude/n676_stages
WORK=$N/work/lesson04-selective-20261015
STAGE=${SELECTIVE_STAGE:-$N/host/stage-04-selective-20261015}
export SELECTIVE_STAGE=$STAGE
LESSON=04_platform_installers
TAG=lesson04-selective-20261015
BASELINE_CANDIDATE=$N/host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e
ORIGINAL=/media/carruthers/mnt3/claude/n676_stages/host/stage-04-r23
PY=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/render-py312/bin/python
SP=/home/carruthers/anaconda3/envs/spacr/bin/python
APPROVED_CARD_EVIDENCE=$WORK/scene10-doctor-card-visual-acceptance.json
APPROVED_CARD_SHA256=0c1efb1d98a00577558ba114c104cb19fddaadfc8c785372ee49d77d5577aee2
export ROOT N WORK STAGE LESSON TAG BASELINE_CANDIDATE ORIGINAL APPROVED_CARD_EVIDENCE APPROVED_CARD_SHA256
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export SPACR_TUTORIAL_WORKSPACE=$STAGE PYTHONPATH=$ROOT:$ROOT/tools:$ROOT/tools/tutorials:$ROOT/tools/tutorials/authoring/tools
