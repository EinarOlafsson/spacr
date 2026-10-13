"""Building a settings panel must not import the plotting stack.

`get_setting_dependencies` reads REGRESSION_SETTINGS_USED to decide which
widgets on a panel apply to each other. It used to import that from
:mod:`spacr.ml`, which imports :mod:`spacr.plot`, which imports torch, cv2
and IPython -- so LOOKING UP A DICT OF STRINGS cost 2.2 seconds and 900 MB,
on the GUI thread, every time any module was opened.

tests/qt/test_gui_responsiveness.py already asserts the panel does not import
the Tk interface, and it caught this. This file is the narrower statement: the
tables the GUI reads are pure data and must stay reachable without the fit.
"""
from __future__ import annotations

import subprocess
import sys

HEAVY = ("torch", "cv2", "IPython", "spacr.ml", "spacr.plot", "matplotlib")


def _imported_by(code: str) -> set:
    """The heavy modules present in sys.modules after running ``code``.

    IN A SUBPROCESS, because an import cannot be undone: any earlier test in
    the same process that touched spacr.ml would leave torch in sys.modules
    and this would pass for the wrong reason.
    """
    # The sentinel matters: with a bare join, "nothing was imported" and
    # "the script printed nothing at all" are the same empty string, and the
    # test would pass on a subprocess that never ran the code.
    script = (
        "import sys\n" + code + "\n"
        "print('HEAVY:' + ','.join(m for m in %r if m in sys.modules))\n"
        % (HEAVY,))
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next((l for l in out.stdout.splitlines()
                 if l.startswith("HEAVY:")), None)
    assert line is not None, f"the probe did not run: {out.stdout[-2000:]}"
    return {name for name in line[len("HEAVY:"):].split(",") if name}


def test_the_backend_tables_are_reachable_without_the_fit():
    heavy = _imported_by(
        "from spacr.regression_spec import REGRESSION_SETTINGS_USED, "
        "REGRESSION_TYPES\n"
        "assert 'ols' in REGRESSION_TYPES\n"
        "assert REGRESSION_SETTINGS_USED['lasso']\n")

    assert heavy == set(), f"importing the spec pulled in {sorted(heavy)}"


def test_the_dependency_rules_do_not_import_the_plotting_stack():
    heavy = _imported_by(
        "from spacr.settings import get_setting_dependencies\n"
        "assert len(get_setting_dependencies()) > 10\n")

    assert heavy == set(), (
        f"building the dependency rules imported {sorted(heavy)} -- this is "
        f"the 2.2s/900MB cost every module open used to pay to look up a "
        f"dict of strings")


def test_the_refit_settings_do_not_import_the_plotting_stack():
    """The re-fit dialog is built on the GUI thread the moment the user
    right-clicks a plot."""
    heavy = _imported_by(
        "from spacr.refit import prune_for_type\n"
        "settings, reset = prune_for_type({'alpha': 0.3}, 'ols')\n"
        "assert settings['alpha'] == 1.0 and reset\n")

    assert heavy == set(), f"pruning imported {sorted(heavy)}"


def test_ml_still_re_exports_every_name():
    """Moved-and-forgotten would break `from spacr.ml import
    REGRESSION_TYPES` in a dozen callers and in every notebook anyone has
    written."""
    from spacr import ml, regression_spec

    for name in ("REGRESSION_TYPES", "REGRESSION_SETTINGS_USED",
                 "RUN_LEVEL_SETTINGS", "UNSUPPORTED_REGRESSION_TYPES",
                 "NO_P_VALUE_TYPES", "_MODEL_LEVEL_DEFAULTS",
                 "_RUN_LEVEL_DEFAULTS"):
        assert getattr(ml, name) is getattr(regression_spec, name), name


def test_there_is_one_copy_of_each_table():
    """A re-export is one object seen twice. A second literal in ml.py would
    be a table that drifts, which is exactly what the spec exists to stop."""
    import ast
    import pathlib

    import spacr

    source = pathlib.Path(spacr.__file__).with_name("ml.py").read_text()
    assigned = {
        target.id
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)}

    for name in ("REGRESSION_TYPES", "REGRESSION_SETTINGS_USED",
                 "_MODEL_LEVEL_DEFAULTS", "_RUN_LEVEL_DEFAULTS"):
        assert name not in assigned, (
            f"ml.py assigns {name} as well as importing it from the spec")
