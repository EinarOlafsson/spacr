"""Resolving a settings default must not drag a heavy library in with it.

A settings default is consulted while a panel is being laid out, on the main
thread, before anything has been run. Whatever it imports, the user waits for.

`spacr.settings._outlier_criteria` reached four pairs of strings through
`spacr.outlier_filter`, which imports pandas -- 211 ms of the time it took to
open the Regression screen, spent to read a constant. This pins the fix, and
it is the same shape as `test_validating_a_queue_pulls_no_torch_or_cellpose`
in tests/test_batch.py and as the ContextVar that keeps the key sweep from
importing PySide6: a settings default is a QUESTION ABOUT NAMES AND VALUES,
and answering it should not load machinery.

Run in a CHILD INTERPRETER, because the answer is about what a fresh process
ends up holding. In this one pandas is imported long before the test starts,
so an in-process assertion would be vacuous whatever the code did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.child_env import child_env

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Libraries no settings default has any business importing.
#:
#: PySide6 IS DELIBERATELY NOT HERE. `set_default_plot_data_from_db` consults
#: the saved graph-type preference, which lives in QSettings, so it imports Qt
#: -- and that is item 293 working as intended: a headless pipeline run is
#: meant to honour the preference too. It also costs nothing in the case this
#: file is about, because a process laying out a settings panel has loaded Qt
#: long before it asks for a default.
#:
#: The case where it DOES cost something -- `validate._known_setting_keys`
#: calling every default with `{}` in a headless process purely to learn key
#: names -- is guarded separately, by the `_READ_THE_PREFERENCE_STORE`
#: ContextVar and its own test. Listing PySide6 here would duplicate that
#: guard and contradict it at the same time.
_HEAVY = ("pandas", "torch", "cellpose", "matplotlib", "sklearn", "skimage")

#: The defaults every app screen resolves when it is opened.
_DEFAULTERS = (
    "get_perform_regression_default_settings",
    "set_default_settings_preprocess_generate_masks",
    "get_measure_crop_settings",
    "set_default_plot_data_from_db",
)


def _modules_after(statements):
    """Which heavy modules a fresh interpreter holds after ``statements``.

    :param statements: Python source run in the child, after importing spacr.
    :returns: the names from :data:`_HEAVY` the child ended up holding.
    """
    code = (
        "import json, sys\n"
        "from spacr import settings as S\n"
        f"{statements}\n"
        "print(json.dumps([m for m in %r if m in sys.modules]))\n" % (_HEAVY,)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=300, env=child_env(pythonpath=str(REPO_ROOT)))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_importing_the_settings_module_imports_nothing_heavy():
    """The module that describes every setting is names and values."""
    assert _modules_after("pass") == []


@pytest.mark.parametrize("name", _DEFAULTERS)
def test_resolving_one_screens_defaults_imports_nothing_heavy(name):
    """Opening a screen must not cost a library the screen has not used yet."""
    held = _modules_after(f"S.{name}({{}})")

    assert held == [], (
        f"{name}({{}}) left {held} imported. A settings default runs while a "
        f"panel is laid out, so this is time the user watches pass before "
        f"anything has been run.")


def test_the_criteria_have_exactly_one_definition():
    """`outlier_filter` re-exports them; it does not keep a second copy.

    Two copies of the same four pairs is how one of them drifts. The old
    `_outlier_criteria` carried a hand-written fallback identical to the real
    tuple, and nothing compared them -- so a criterion added to one and not
    the other would have gone missing from the panel with no test to notice.
    """
    from spacr import _outlier_criteria, outlier_filter, settings

    assert outlier_filter.CRITERIA is _outlier_criteria.CRITERIA
    assert settings._outlier_criteria() is _outlier_criteria.CRITERIA
    assert "cell_area" in dict(_outlier_criteria.CRITERIA)
