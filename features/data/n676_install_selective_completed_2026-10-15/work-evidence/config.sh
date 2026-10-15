#!/usr/bin/env bash
# Selective visual repair only. Root supplies the actual frozen combined K commit.
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
N=/media/carruthers/mnt3/claude/n676_stages
WORK=$N/work/lesson02-selective-20261015
STAGE=${SELECTIVE_STAGE:-$N/host/stage-02-selective-r2-20261015}
export SELECTIVE_STAGE=$STAGE
LESSON=02_install_spacr
TAG=lesson02-selective-20261015
BASELINE_CANDIDATE=$N/host/stage-23-r60-20261015/release-candidate-append-7ew0avek
ORIGINAL=/media/carruthers/mnt3/codex/scratch/tutorials-agent-20261008/tutorials-requested-live-r1
PY=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/render-py312/bin/python
SP=/home/carruthers/anaconda3/envs/spacr/bin/python
APPROVED_CARD_EVIDENCE=$WORK/scene07-card-visual-acceptance.json
APPROVED_CARD_SHA256=f99db28091367731527ba42085ba6a77d57123eca020890cc2d2abbef3fb5198
export ROOT N WORK STAGE LESSON TAG BASELINE_CANDIDATE ORIGINAL APPROVED_CARD_EVIDENCE APPROVED_CARD_SHA256
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export SPACR_TUTORIAL_WORKSPACE=$STAGE PYTHONPATH=$ROOT:$ROOT/tools:$ROOT/tools/tutorials:$ROOT/tools/tutorials/authoring/tools
