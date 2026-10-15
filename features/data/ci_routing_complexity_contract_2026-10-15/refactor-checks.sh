#!/bin/bash
exec flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /bin/bash /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/refactor-checks-inner.sh
