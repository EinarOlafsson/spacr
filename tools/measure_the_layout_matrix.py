#!/usr/bin/env python
"""Drive the layout matrix and write the policy artifact (359, part 3).

THE MEASUREMENT IS NOT HERE, and that is the point. It lives in
``tests/qt/test_the_layout_matrix.py``, where ``tests/qt/conftest.py``'s
fifteen autouse fixtures and the session one that fills
``theme._WIDGET_QSS`` are in force.

This was first written the other way round -- a standalone script with a
hand-rolled qtbot -- and it DISAGREED WITH THE SWEEP, reporting two clipped
captions on every module including ones that do not contain the button it
named. A generator that disagrees with the suite produces a policy the
suite cannot defend, and the disagreement is invisible until a user meets
it. So this is a driver, and the number written to the artifact is the
number the suite measures.

    python tools/measure_the_layout_matrix.py
    python tools/measure_the_layout_matrix.py --out /tmp/policy.json

The measurement takes several minutes: 24 modules x 4 locales x 2 scales,
each walking a ladder of up to fifteen widths.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Where the wheel expects it. `spacr.qt.layout_policy` reads this name
#: through `importlib.resources`, and `MANIFEST.in` and `package_data`
#: both list it.
DEFAULT_OUT = REPO / "spacr" / "resources" / "layout_policy.json"

#: The measurement builds every module screen four times over; the default
#: 4 GB test ceiling ends it part way through.
MEMORY_GB = "12"


def main(argv=None) -> int:
    """Run the measurement under pytest and report where it landed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where to write the artifact")
    parser.add_argument("--memory-gb", default=MEMORY_GB,
                        help="the test-guard ceiling for the measuring run")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    environment = dict(os.environ)
    environment.update({
        "SPACR_MEASURE_LAYOUT": str(out),
        "SPACR_TEST_MEMORY_GB": str(args.memory_gb),
        "QT_QPA_PLATFORM": environment.get("QT_QPA_PLATFORM", "offscreen"),
        # A GPU is not part of this measurement and an idle one is shared
        # with other work on the machine this runs on.
        "CUDA_VISIBLE_DEVICES": "",
    })
    command = [sys.executable, "-m", "pytest",
               "tests/qt/test_the_layout_matrix.py",
               "-q", "-p", "no:cacheprovider", "-m", "not gpu", "-s"]
    finished = subprocess.run(command, cwd=str(REPO), env=environment)

    # A NON-ZERO EXIT IS NOT A FAILURE TO MEASURE. The artifact is written
    # before the test's own checks run, so a module that no width shows
    # whole leaves both a red test naming it and a policy that has already
    # raised its recommendation to the widest rung. Say which happened
    # rather than letting the caller read the exit code as "no file".
    if out.exists():
        print(f"\nartifact: {out}")
        if finished.returncode:
            print("the measurement found a case no width fixes -- see the "
                  "assertion above; the artifact is still written, with "
                  "that case counted at the widest rung tried")
    else:
        print("\nno artifact was written")
    return finished.returncode


if __name__ == "__main__":
    raise SystemExit(main())
