#!/bin/bash
# One spaCR Mask + Measure run per plate, as a Slurm array job, in the
# Apptainer image. Copy this file, edit the four variables below and the
# #SBATCH lines for your cluster, then:
#
#   sbatch --array=0-$(( $(wc -l < plates.txt) - 1 )) spacr_slurm.sh
#
# plates.txt holds one plate folder per line. mask.csv and measure.csv are
# settings files as the GUI or any earlier run writes them into
# <plate>/settings/; the src of each is overridden per task below, so one pair
# serves every plate.
#
# For a CPU-only partition drop the --gres line and the --nv flag and point
# SIF at the CPU image; nothing else changes.
#
#SBATCH --job-name=spacr
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=spacr_%A_%a.log
set -euo pipefail

SIF=${SIF:-$HOME/containers/spacr-cuda.sif}
PLATES=${PLATES:-plates.txt}
MASK_SETTINGS=${MASK_SETTINGS:-mask.csv}
MEASURE_SETTINGS=${MEASURE_SETTINGS:-measure.csv}
GPU_FLAG=${GPU_FLAG:---nv}

plate=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$PLATES")
if [ -z "$plate" ]; then
    echo "no plate on line $(( SLURM_ARRAY_TASK_ID + 1 )) of $PLATES" >&2
    exit 1
fi

# Resolve both settings files before Mask starts, including symlink targets.
# They can live in separate folders outside Apptainer's automatic binds.
for settings in "$MASK_SETTINGS" "$MEASURE_SETTINGS"; do
    if [ ! -f "$settings" ]; then
        printf 'settings file not found: %s\n' "$settings" >&2
        exit 1
    fi
done
MASK_SETTINGS=$(realpath -e -- "$MASK_SETTINGS")
MEASURE_SETTINGS=$(realpath -e -- "$MEASURE_SETTINGS")

# The plate folder and both settings folders are bound at the same paths
# they have on the host, so the settings files are visible inside the image.
# $HOME is bound by Apptainer already, and with it ~/.cellpose/models.
binds="$plate,$(dirname "$MASK_SETTINGS"),$(dirname "$MEASURE_SETTINGS")"

# Every worker spaCR starts is capped at the CPUs Slurm granted, not at the
# node's core count.
run() {
    apptainer run $GPU_FLAG --bind "$binds" "$SIF" \
        spacr-run "$1" --settings "$2" --set "src=$plate" \
        --set "n_jobs=${SLURM_CPUS_PER_TASK:-1}"
}

run mask "$MASK_SETTINGS"
run measure "$MEASURE_SETTINGS"
