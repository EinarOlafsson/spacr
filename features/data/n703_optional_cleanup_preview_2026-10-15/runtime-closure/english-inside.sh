#!/bin/bash
trap 'status=$?; echo "$status" > /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/english-labels.exit' EXIT
/home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/english-labels.py
