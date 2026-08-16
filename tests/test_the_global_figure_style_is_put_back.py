"""A test that restyles every figure must restyle them back.

THE INVARIANT, in the user's terms: the figure spaCR draws is the figure the
user asked for. Nothing else the application did earlier in the session may
decide where its axes sit or how big its text is.

``spacr.figure_style.apply`` exists to make one style global -- the user picks
a font and a DPI once and every plot of the run inherits it. That is right for
the application and wrong for a test, because a test that calls it is choosing
the style for every test that runs after it, in every other file. The settings
it writes are the ones that decide layout: ``figure.autolayout``,
``figure.dpi``, the font family, the colour cycle.

It has already cost a day. ``figure.autolayout`` left on by
``tests/test_figure_style.py`` re-laid-out the axes of a figure built in
``tests/test_toxo_volcano_join_contract.py``, so
``test_the_legend_fits_inside_the_figure`` failed in a full run and passed on
its own -- and the report named the volcano test, which was innocent.

This guard runs each such module in its own pytest session and compares
matplotlib's global settings before and after. It names the module that
leaked, which is the whole point: the alternative is a failure reported
against whichever unrelated test drew the short straw.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent

#: Written as call syntax, so a module that merely names one of these in prose
#: -- ``test_api_i18n_extractor.py`` counts the public API by name -- is not
#: dragged into a nested pytest session for nothing.
GLOBAL_STYLE_CALLS = (
    "figure_style.apply(",
    "apply_figure_style(",
)

#: The nested session. Reports the rcParams the module changed and did not put
#: back, one per line, then exits with pytest's own status.
GUARD = """
import sys

import matplotlib
import pytest


class RcParamsGuard:

    def pytest_sessionstart(self, session):
        self.before = {k: repr(v) for k, v in matplotlib.rcParams.items()}

    def pytest_sessionfinish(self, session, exitstatus):
        after = {k: repr(v) for k, v in matplotlib.rcParams.items()}
        for key in sorted(set(self.before) | set(after)):
            if self.before.get(key) != after.get(key):
                print("LEAKED\t%s\t%s\t%s"
                      % (key, self.before.get(key), after.get(key)))


sys.exit(pytest.main(
    ["-q", "-p", "no:randomly", "-p", "no:cacheprovider", sys.argv[1]],
    plugins=[RcParamsGuard()]))
"""


def _imports_a_global_style(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.name == Path(__file__).name:
        return False
    if "from spacr.figure_style import" in text and "apply" in text:
        return True
    return any(token in text for token in GLOBAL_STYLE_CALLS)


def _modules_that_apply_a_global_style() -> list:
    """Every test module that pushes a style into matplotlib's globals.

    Discovered rather than listed, so the module somebody adds tomorrow is
    guarded without anyone having to remember this file exists.
    """
    found = [path for path in sorted(TESTS_ROOT.rglob("test_*.py"))
             if _imports_a_global_style(path)]
    return found


def test_the_scan_finds_the_module_that_applies_a_style():
    """A guard that silently matches nothing guards nothing."""
    modules = _modules_that_apply_a_global_style()
    assert modules, (
        "no test module calls figure_style.apply / apply_figure_style any "
        "more; either the API was renamed and this guard is now blind, or "
        "the tests for it were deleted")


@pytest.mark.parametrize(
    "module",
    [pytest.param(p, id=p.name) for p in _modules_that_apply_a_global_style()])
def test_it_leaves_matplotlib_as_it_found_it(module):
    """Run the module on its own and check the global style came back.

    A nested session rather than an in-process check, because the leak this
    catches is one test module deciding the style for the NEXT one -- which is
    only visible across a whole session, and is invisible from inside the
    session doing the leaking.
    """
    finished = subprocess.run(
        [sys.executable, "-c", GUARD, str(module.relative_to(REPO_ROOT))],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)

    leaked = [line.split("\t")[1:] for line in finished.stdout.splitlines()
              if line.startswith("LEAKED\t")]
    if finished.returncode not in (0, 5) and not leaked:
        pytest.skip(
            f"{module.name} does not pass on its own in this environment, so "
            f"it cannot be asked about leaks (exit {finished.returncode})")
    assert not leaked, (
        f"{module.name} left matplotlib's global drawing style changed. "
        "Every figure drawn after it in the same session inherits this, "
        "including figures built by other files:\n"
        + "\n".join(f"  {key}: {before} -> {after}"
                    for key, before, after in leaked)
        + "\n\nWrap the tests that apply a style in `matplotlib.rc_context()` "
          "so the style goes back when the test ends.")
