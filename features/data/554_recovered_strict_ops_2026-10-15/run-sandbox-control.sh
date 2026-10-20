#!/bin/bash
set -euo pipefail
status=0
/home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/sandbox-control.py > /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/sandbox-control.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/sandbox-control.exit
exit "$status"
