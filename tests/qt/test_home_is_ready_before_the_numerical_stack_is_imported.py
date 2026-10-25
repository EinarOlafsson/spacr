"""Home reports ready with no numerical stack, except the ambient kernels' numba.

The installed-artifact smoke probe samples sys.modules at Home's readiness
record. The start-up warm imports pandas, scipy, scikit-image and numba on a
worker, so it holds them back until Home has been ready for a moment.

One import is allowed before then, by the maintainer's decision: the ambient
backdrop warms its Numba kernels before the first paint
(``app._start_kernel_warmup_before_first_paint``), so numba may already be
imported at readiness -- but only when that warm-up imported it. Every heavy
package's first import is traced in the child: the importing thread and its
stack. The warm-up starts two compile threads, ``spacr-grain-compile`` (the
packed grain scatter) and ``spacr-satin-compile`` (the satin columns), and
either may reach numba first. numba passes only when it was imported on one
of those two threads through ``ambient._import_numba``; scipy passes only
when numba's own package imported it on that same path. pandas and scikit-image, and numba or
scipy imported from any other thread or path, still fail the check, as does a
heavy package whose first import could not be traced.

Run in fresh processes, several times, because the old failure was a race.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent("""
    import json, sys, threading, traceback

    HEAVY = ("pandas", "scipy", "skimage", "numba")


    class FirstImports:
        \"\"\"Note the thread and stack of each heavy package's first import.\"\"\"

        def __init__(self):
            self.origins = {}

        def find_spec(self, fullname, path=None, target=None):
            if fullname in HEAVY and fullname not in self.origins:
                frames = traceback.StackSummary.extract(
                    traceback.walk_stack(sys._getframe(1)), limit=120,
                    lookup_lines=False)
                self.origins[fullname] = {
                    "thread": threading.current_thread().name,
                    "stack": [frame.filename.replace("\\\\", "/") + ":"
                              + frame.name for frame in frames],
                }
            return None


    FIRST = FirstImports()
    sys.meta_path.insert(0, FIRST)

    from PySide6.QtWidgets import QApplication
    import spacr.qt as spacr_qt
    from spacr.qt import timing
    from spacr.qt.first_run import mark_tour_seen
    mark_tour_seen()
    seen = {}


    def observe(entry):
        if entry.get("detail") != "__home__" or seen:
            return
        seen["heavy"] = {name: FIRST.origins.get(name)
                         for name in HEAVY if name in sys.modules}
        seen["ready"] = True
        QApplication.instance().quit()


    timing.subscribe_readiness(observe)
    code = spacr_qt.run(["--no-setup"])
    sys.stdout.write("\\nRESULT " + json.dumps({"code": code, **seen}) + "\\n")
""")

#: The threads ``ambient._start_kernel_warmup_before_first_paint`` starts.
WARMUP_THREADS = ("spacr-grain-compile", "spacr-satin-compile")

#: The function every allowed numba import goes through.
WARMUP_IMPORTER = "/spacr/qt/widgets/ambient.py:_import_numba"


def _from_the_warmup(origin) -> bool:
    """Whether a traced first import came from the ambient kernel warm-up."""
    return bool(origin) and origin["thread"] in WARMUP_THREADS and any(
        frame.endswith(WARMUP_IMPORTER) for frame in origin["stack"])


def _through_numba(origin) -> bool:
    """Whether a traced first import was made by numba's own package."""
    return bool(origin) and any("/numba/" in frame
                                for frame in origin["stack"])


def _disallowed(heavy) -> dict:
    """The heavy packages at readiness that the warm-up does not account for.

    :param heavy: ``{package: origin or None}`` from the child.
    :returns: ``{package: why}`` for each one not allowed.
    """
    refused = {}
    for name, origin in heavy.items():
        if origin is None:
            refused[name] = "imported, but its first import was not traced"
        elif name == "numba" and _from_the_warmup(origin):
            continue
        elif (name == "scipy" and _from_the_warmup(origin)
              and _through_numba(origin)):
            continue
        else:
            ours = [frame for frame in origin["stack"]
                    if "/spacr/" in frame or "/numba/" in frame]
            refused[name] = (f"thread {origin['thread']!r}, from "
                             f"{(ours or origin['stack'])[:8]}")
    return refused


def _once(tmp_path, index):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="", SPACR_TIMING="1", SPACR_TIMING_IMPORTS="0",
               HOME=str(tmp_path / str(index)),
               XDG_CONFIG_HOME=str(tmp_path / str(index)),
               SPACR_HOME=str(tmp_path / str(index) / "spacr"),
               PYTHONPATH=os.pathsep.join([str(ROOT)] + sys.path))
    if os.name == "nt":
        # The child has no conftest: on Windows HOME moves nothing and Qt's
        # store is the registry, so the app would start from (and write to)
        # the developer's real profile. Portable mode keeps both in tmp_path.
        env.update(USERPROFILE=str(tmp_path / str(index)),
                   SPACR_PORTABLE=str(tmp_path / str(index)))
    (tmp_path / str(index)).mkdir()
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT ")]
    if not lines:
        return {"error": done.stderr[-800:]}
    return json.loads(lines[-1][7:])


def test_home_is_ready_before_the_warm_imports_the_numerical_stack(tmp_path):
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda i: _once(tmp_path, i), range(6)))
    summary = [{"code": result.get("code"), "ready": result.get("ready"),
                "error": result.get("error"),
                "refused": _disallowed(result.get("heavy") or {})}
               for result in results]
    assert summary == [{"code": 0, "ready": True, "error": None,
                        "refused": {}}] * 6


def test_only_the_warmup_path_lets_numba_and_its_scipy_through():
    warmup = {"thread": "spacr-grain-compile",
              "stack": ["/x/spacr/qt/widgets/ambient.py:_import_numba",
                        "/x/spacr/qt/widgets/ambient.py:_compiled_kernel"]}
    satin = {"thread": "spacr-satin-compile",
             "stack": ["/x/spacr/qt/widgets/ambient.py:_import_numba",
                       "/x/spacr/qt/widgets/ambient.py:_warm"]}
    scipy_by_numba = {"thread": "spacr-satin-compile",
                      "stack": ["/site/numba/__init__.py:<module>",
                                "/x/spacr/qt/widgets/ambient.py:_import_numba"]}
    elsewhere = {"thread": "MainThread",
                 "stack": ["/x/spacr/qt/widgets/ambient.py:_import_numba"]}
    other_thread = {"thread": "spacr-data-libraries",
                    "stack": ["/x/spacr/qt/widgets/ambient.py:_import_numba"]}
    other_path = {"thread": "spacr-grain-compile",
                  "stack": ["/x/spacr/qt/app.py:_warm_data_libraries"]}
    assert _disallowed({"numba": warmup, "scipy": scipy_by_numba}) == {}
    assert _disallowed({"numba": satin}) == {}
    assert set(_disallowed({"numba": other_thread})) == {"numba"}
    assert set(_disallowed({"numba": elsewhere})) == {"numba"}
    assert set(_disallowed({"numba": other_path})) == {"numba"}
    assert set(_disallowed({"scipy": warmup})) == {"scipy"}
    assert set(_disallowed({"pandas": warmup, "skimage": warmup})) == {
        "pandas", "skimage"}
    assert set(_disallowed({"numba": None})) == {"numba"}
