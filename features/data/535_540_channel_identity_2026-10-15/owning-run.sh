#!/bin/bash
set -euo pipefail
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /bin/bash /mnt/wd4tb/scratch/measured-channel-identity-20261015/owning-body.sh
