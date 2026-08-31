#!/bin/bash
# batched_coverage.sh's measurement, without pytest-cov or pytest-timeout.
#
# WHY THIS EXISTS. The original drives coverage through pytest-cov (`--cov`)
# and bounds hangs with pytest-timeout (`--timeout`). Neither plugin is
# installed in every environment that has spaCR -- on this host NO conda env
# has them -- and pytest answers a missing plugin with exit code 4 and
# "unrecognized arguments", per batch, in about a second. The parent script
# then prints "No data to combine" and "DONE", and a run that measured
# NOTHING looks like a run that finished. That is the same shape as the
# import gate the parent already guards against, one layer out: the thing
# that failed is the measurement, and the exit status belonged to `tail`.
#
# So: `coverage run --parallel-mode` instead of --cov, shell `timeout`
# instead of --timeout, and concurrent batch PROCESSES instead of xdist
# workers. Parallel-mode gives each process its own data file, which is what
# makes concurrency safe here and is also the thing 310 A42 records going
# wrong when it is missing.
#
#   WT=/path/to/frozen/worktree OUT=/path/for/artifacts JOBS=4 \
#       bash tools/coverage/batched_coverage_nocovplugin.sh
set -uo pipefail
WT=${WT:?set WT to a frozen worktree}
OUT=${OUT:?set OUT to an artifact directory}
BATCH=${BATCH:-60}
JOBS=${JOBS:-4}
PY=${PY:-python}
mkdir -p "$OUT"
cd "$WT" || exit 2

# IMPORT GATE, kept verbatim in intent from the parent: a HEAD that does not
# import yields hundreds of failures and a figure for a package that never
# loaded.
if ! QT_QPA_PLATFORM=offscreen "$PY" -c "
import importlib, pkgutil, spacr, sys
bad = []
for m in pkgutil.walk_packages(spacr.__path__, 'spacr.'):
    try: importlib.import_module(m.name)
    except Exception as e: bad.append(f'{m.name}: {type(e).__name__}: {e}')
if bad:
    print('HEAD DOES NOT IMPORT -- refusing to measure it:', file=sys.stderr)
    for b in bad[:10]: print('  ' + b, file=sys.stderr)
    sys.exit(1)
"; then
  echo "ABORTED: HEAD does not import cleanly."; exit 3
fi
"$PY" -c "
import spacr, sys
assert '$WT' in spacr.__file__, 'WRONG TREE: ' + spacr.__file__
print('measuring:', spacr.__file__)
" || exit 3

# PLUGIN GATE, the lesson this script was written for. Refuse to start rather
# than emit 42 identical usage errors and call it DONE.
"$PY" -c "import coverage" 2>/dev/null || { echo "ABORTED: coverage missing"; exit 4; }

export PYTHONPATH="$WT"
export CUDA_VISIBLE_DEVICES=""
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export NUMBA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export COVERAGE_FILE="$OUT/.coverage"
export COVERAGE_RCFILE="$WT/.coveragerc"

find tests -name 'test_*.py' \
  ! -path 'tests/test_regression_screen_layout.py' \
  ! -path 'tests/qt/test_layout_drops.py' | sort > "$OUT/files.txt"
echo "test files: $(wc -l < "$OUT/files.txt")  batch $BATCH  jobs $JOBS"

rm -f "$OUT"/batch_* "$OUT"/.coverage* 2>/dev/null
split -l "$BATCH" -d -a 4 "$OUT/files.txt" "$OUT/batch_"

run_one() {
  local b="$1" n; n=$(basename "$b")
  nice -n 10 timeout 3600 "$PY" -m coverage run --branch --parallel-mode \
      -m pytest $(tr '\n' ' ' < "$b") -q -p no:randomly --no-header -rf \
      > "$OUT/${n}.log" 2>&1
  echo "  ${n} exit=$? :: $(tail -1 "$OUT/${n}.log" | tr -d '\r')"
}
export -f run_one
export OUT PY

ls "$OUT"/batch_* | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "=== combining $(date +%H:%M:%S) ==="
"$PY" -m coverage combine 2>&1 | tail -2
"$PY" -m coverage json -o "$OUT/batched.json" 2>&1 | tail -2
"$PY" -m coverage report --include='*/spacr/ml.py,*/spacr/io.py,*/spacr/plot.py,*/spacr/timelapse.py,*/spacr/sequencing.py' 2>&1 | tail -10
echo "DONE $(date -Iseconds)"
