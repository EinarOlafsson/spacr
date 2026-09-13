"""Find assertions that a refactor quietly emptied out.

    python tools/which_assertions_went_vacuous.py [--accessor SPEC] [PATHS...]

WHY THIS EXISTS. When a refactor moves a value somewhere else, the tests that
read the OLD place split into two populations, and only one of them is visible:

    `assert X in sheet`      goes RED    -- and gets fixed that afternoon
    `assert X not in sheet`  goes GREEN  -- and is never thought about again

They are the same assertion with opposite polarity. Instruction 380 moved the
composed stylesheet from the QApplication onto each top-level window, and
commit 94f590e0b repaired the first of those two lines in
`tests/qt/test_field_fade.py` while leaving the second, two hundred lines
above it in the same file, asserting that a marker is absent from the empty
string. Four such assertions were found on 2026-09-13, in three files, two
days after the move -- none of them by reading.

WHAT IT MEASURES. Every call to the named accessor is recorded against the
line in `tests/` that led to it, with a count of how often the value came back
FALSY and how often it came back truthy. A site that is only ever falsy is a
site whose `==` and `in` checks are satisfied by emptiness rather than by the
property they name.

    always full     the test installed the value itself; nothing to see
    mixed           it is real in some tests and empty in others
    only ever empty the candidates -- read each one, they are not all defects

  ONLY-EVER-EMPTY IS A CANDIDATE, NOT A VERDICT, and the tool cannot close
  that gap. A test may assert emptiness ON PURPOSE:
  `tests/qt/test_widget_qss_is_complete.py` asserts `not app.styleSheet()`
  because 380 makes a non-empty application sheet foreign by construction.
  That site is correct and appears in this column. Read the assertion.

  IT ALSO CANNOT SEE A SITE NO TEST REACHED. A file that errors during
  collection, a test skipped for a missing backend, a branch not taken --
  none of them call the accessor, so none of them appear at all. Absence
  from this table is not evidence.

THE PROBE PERTURBS A LARGE RUN. TREAT EVERY FAILURE UNDER IT AS SUSPECT.
Measured 2026-09-13 over 66 theme files in one process:

    with this probe       16 failed, 1991 passed, 6 skipped
    without it, same
    files and same order       0 failed, 2007 passed, 6 skipped

Those 16 are not real. They do not reproduce when the affected file is run
alone under the probe (28 passed both ways), nor in a five-file subset chosen
from the likely interactors (312 passed under the probe), and the probe adds no
measurable time -- one file measured 30.34 s under it against 31.06 s without.
So it is not slowness and it is not a simple two-file interaction; the
mechanism was NOT localised, and saying so is more useful than a guess.

  WHAT THAT MEANS IN PRACTICE. Use this tool to produce the SITE TABLE, which
  is what it is for and which was verified correct against a second probe on
  the write side. Do NOT use its pass/fail result for anything. When it reports
  a failure, re-run the same files without `-p vacuous_plugin` before believing
  it -- and if a site appears only in a run that also failed, check whether the
  failing test is the one that would have reached it.

THE COUNTER IS A NAMED DICT AND THAT IS DELIBERATE. The first version of this
measurement stored `[full, empty]` as a two-slot list and the reporting code
unpacked it as `(empty, full)`, so every site was reported inverted: it
claimed 41 of 55 sites were vacuous when the truth was 11, and the 41 were the
healthy ones. It was caught only because a second probe on the write side
disagreed -- 72,979 characters installed and never cleared, against a read
probe insisting the value was always empty. A key cannot be unpacked
backwards.

:param --accessor: ``module:Class.method``, the zero-argument reader to watch.
    Defaults to ``PySide6.QtWidgets:QApplication.styleSheet``.
:param PATHS: what to hand pytest. Defaults to ``tests``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Written to a temporary directory and loaded with ``-p``. It has to be a
#: file on disk rather than a fixture: the patch must be in place before the
#: first test imports anything, and ``pytest_configure`` from a ``-p`` plugin
#: is the earliest hook that is available without editing a conftest.
PLUGIN = '''
"""Record, per call site in tests/, whether {accessor} came back empty."""
import atexit, json, os, traceback

OUT = os.environ["VACUOUS_OUT"]
SITES = {{}}


def pytest_configure(config):
    try:
        import importlib
        module = importlib.import_module({module!r})
        owner = getattr(module, {owner!r})
    except Exception:
        return
    original = getattr(owner, {attr!r})

    def probe(self):
        value = original(self)
        for frame in traceback.extract_stack()[:-1][::-1]:
            if os.sep + "tests" + os.sep in frame.filename:
                key = "%s:%d" % (frame.filename, frame.lineno)
                seen = SITES.setdefault(key, {{"full": 0, "empty": 0}})
                seen["full" if value else "empty"] += 1
                break
        return value

    setattr(owner, {attr!r}, probe)
    atexit.register(lambda: open(OUT, "w").write(json.dumps(SITES, indent=1)))
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accessor",
                    default="PySide6.QtWidgets:QApplication.styleSheet")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("paths", nargs="*", default=["tests"])
    a = ap.parse_args(argv)

    try:
        module, qualified = a.accessor.split(":", 1)
        owner, attr = qualified.rsplit(".", 1)
    except ValueError:
        print(f"--accessor must be module:Class.method, not {a.accessor!r}",
              file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="vacuous-"))
    (work / "vacuous_plugin.py").write_text(
        PLUGIN.format(accessor=a.accessor, module=module, owner=owner,
                      attr=attr))
    out = work / "sites.json"

    env = dict(os.environ)
    env["VACUOUS_OUT"] = str(out)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(work), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The GPU is hidden for the same reason `tests/test_perf_guard.py` hides
    # it: a CUDA context makes the run mean something different, and nothing
    # here is about CUDA.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("SPACR_TEST_MEMORY_GB", "8")

    print(f"watching {a.accessor} over {' '.join(a.paths)}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", *a.paths, "-q",
         "-p", "no:cacheprovider", "-p", "vacuous_plugin", "-m", "not gpu"],
        cwd=ROOT, env=env, timeout=a.timeout)

    if not out.exists():
        print("the plugin wrote nothing: the accessor was never called, or "
              "pytest died before any test ran", file=sys.stderr)
        return 1

    sites = json.loads(out.read_text())
    empty_only, mixed, full = [], [], []
    for key, seen in sites.items():
        name = key.replace(str(ROOT) + os.sep, "")
        if not seen["full"]:
            empty_only.append((name, seen))
        elif seen["empty"]:
            mixed.append((name, seen))
        else:
            full.append((name, seen))

    print(f"\n  sites {len(sites)}   always full {len(full)}   "
          f"mixed {len(mixed)}   only ever empty {len(empty_only)}")
    for title, rows in (("ONLY EVER EMPTY -- read each one", empty_only),
                        ("MIXED", mixed)):
        if not rows:
            continue
        print(f"\n  {title}:")
        for name, seen in sorted(rows):
            print(f"    full={seen['full']:5d} empty={seen['empty']:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
