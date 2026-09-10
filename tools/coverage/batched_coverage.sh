#!/bin/bash
# Batched full-suite branch coverage, so one dead worker costs one batch.
#
# WHY BATCHED (instruction 310 section 13). Two -n 6 runs each lost workers to
# 'node down', and each loss takes that worker's coverage data with it while its
# tests still count toward the pass/fail summary -- so the report understates
# complete modules. Here every batch writes its .coverage.* to disk before the
# next begins, so a death costs that batch and not the run.
#
# Fresh processes also bound the six-fold deceleration HANDOFF 3d records: the
# suite slows as leaked widgets, live QThreads and unclosed figures accumulate,
# and a new interpreter per batch resets that.
#
# The two files that SEGFAULT under offscreen Qt are excluded by name; a
# segfault kills the process outright, so no timeout can rescue them.
set -uo pipefail
# WT and OUT are overridable, because these were absolute paths on ONE machine
# and the script silently measured nothing anywhere else -- `cd "$WT" || exit 2`
# is the whole diagnostic you got. Defaults keep the original behaviour.
WT=${WT:-/mnt/firecuda2/codex/covwt}
OUT=${OUT:-/mnt/firecuda2/codex/covscratch/batched}
BATCH=${BATCH:-60}
mkdir -p "$OUT"
cd "$WT" || exit 2

# IMPORT GATE. A run measured on a HEAD that does not import reports hundreds
# of failures and a coverage figure for a package that never loaded -- 211 in
# batch 1 on 2026-08-30, all one missing name, split across a commit boundary.
# Forty minutes of measurement to learn something one import would have said.
if ! QT_QPA_PLATFORM=offscreen python -c "
import importlib, pkgutil, spacr, sys
bad = []
for m in pkgutil.walk_packages(spacr.__path__, 'spacr.'):
    try: importlib.import_module(m.name)
    except Exception as e: bad.append(f'{m.name}: {type(e).__name__}: {e}')
if bad:
    print('HEAD DOES NOT IMPORT -- refusing to measure it:', file=sys.stderr)
    for b in bad[:10]: print('  ' + b, file=sys.stderr)
    sys.exit(1)
" 2>&1; then
  echo "ABORTED: HEAD does not import cleanly. Fix that before measuring."
  exit 3
fi
echo "import gate: every spacr module loads"

python -c "
import spacr, sys
assert '$WT' in spacr.__file__, 'WRONG TREE: ' + spacr.__file__
print('measuring:', spacr.__file__)
" || exit 3

export PYTHONPATH="$WT"
export CUDA_VISIBLE_DEVICES=""
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export NUMBA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export COVERAGE_FILE="$WT/.coverage"

find tests -name 'test_*.py' \
  ! -path 'tests/test_regression_screen_layout.py' \
  ! -path 'tests/qt/test_layout_drops.py' | sort > "$OUT/files.txt"
echo "test files: $(wc -l < "$OUT/files.txt")  batch size: $BATCH"

rm -f "$OUT"/batch_* 2>/dev/null
split -l "$BATCH" -d -a 4 "$OUT/files.txt" "$OUT/batch_"

n=0
for b in "$OUT"/batch_*; do
  n=$((n+1))
  echo "=== batch $n ($(wc -l < "$b") files) $(date +%H:%M:%S) ==="
  nice -n 10 timeout 3600 python -m pytest $(tr '\n' ' ' < "$b") \
      -q -p no:randomly --no-header --timeout=600 --timeout-method=thread \
      -n 4 --cov=spacr --cov-branch --cov-report= --cov-append \
      -rf \
      >> "$OUT/batch_${n}.log" 2>&1
  echo "  exit=$? :: $(tail -1 "$OUT/batch_${n}.log" | tr -d '\r')"
done

echo "=== combining $(date +%H:%M:%S) ==="
python -m coverage combine 2>&1 | tail -2
python -m coverage json -o "$OUT/batched.json" 2>&1 | tail -2
echo "DONE $(date -Iseconds)"
