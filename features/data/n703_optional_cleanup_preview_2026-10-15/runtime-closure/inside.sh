#!/bin/bash
set -e
trap 'status=$?; echo "$status" > /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/final66-run.exit' EXIT
/usr/bin/time -v /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/scoped-runtime.py
/home/olafsson/anaconda3/envs/spacr/bin/python tools/i18n_caption_arrivals.py --refresh
/home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_n703_optional_cleanup_dialog.py::test_actual_inventory_retention_reason_and_keep_button_use_reviewed_locale -n 0 --junitxml=/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/dialog-locales.xml
