#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015
set +e
if [ "$(cat /mnt/wd4tb/scratch/n698-web-localization-20261015/runtime.exit)" != 0 ]; then exit 44; fi
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /mnt/wd4tb/scratch/n698-web-runtime-20261014/bin/python -m pytest -q --basetemp=/mnt/wd4tb/scratch/n698-web-localization-20261015/pytest tests/test_web_mask_runtime_locales_n698.py tests/test_web_caption_extraction_n698.py --junitxml=/mnt/wd4tb/scratch/n698-web-localization-20261015/callbacks.xml > /mnt/wd4tb/scratch/n698-web-localization-20261015/callbacks.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-web-localization-20261015/callbacks.exit
if [ "$status" != 0 ]; then exit "$status"; fi
/home/olafsson/anaconda3/envs/spacr/bin/python - <<'XML'
from pathlib import Path
import xml.etree.ElementTree as ET
r=ET.parse('/mnt/wd4tb/scratch/n698-web-localization-20261015/callbacks.xml')
assert not list(r.iter('skipped')) and not list(r.iter('failure')) and not list(r.iter('error'))
assert sum('test_web_mask_runtime_locales_n698' in x.attrib.get('classname','') for x in r.iter('testcase'))==27
XML
if [ "$?" != 0 ]; then exit 45; fi
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n698-web-localization-20261015/audit-arrivals.py > /mnt/wd4tb/scratch/n698-web-localization-20261015/audit-arrivals.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-web-localization-20261015/audit-arrivals.exit
exit "$status"
