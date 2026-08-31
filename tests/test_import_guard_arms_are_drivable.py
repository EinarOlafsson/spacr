"""An ``except ImportError`` arm is never unreachable by construction.

Instruction 288's audit filed every remaining uncovered site as "unreachable by
construction", and concluded the count could not fall without deleting code.
Import-guard arms were misfiled into that set. They are conditional on the
ENVIRONMENT, not on the source -- which makes them look unreachable on a
machine where the dependency happens to be installed -- but a test can always
make the import fail, so they are testable items rather than pins.

THE TECHNIQUE, and it is the general answer for the whole class: swap
``builtins.__import__`` for one that raises ``ImportError`` for exactly one
module name, drive the call, restore. It needs no uninstall, no venv, and no
``sys.modules`` surgery that a later import can undo, and it works whether or
not the package is present -- which is the point, because otherwise the test
passes for one reason on one machine and another reason on another.

``adjustText`` is INSTALLED in the environment this was written in. That is
deliberate: if the arm can be driven with the package present, the arm's
reachability does not depend on the machine at all.

ARC-CHECKED. This is exactly the shape that produces a vacuous pass -- a test
that asserts a fallback happened against a function that returned the same
thing for an unrelated reason (a missing column, an empty frame, a raise
earlier in the call). ``pytest.raises(ImportError)`` in particular will happily
catch an ImportError from somewhere else entirely, so each test below pins the
MESSAGE the arm itself writes, and the arcs were confirmed reached with
``tools/coverage/check_arcs.py`` rather than assumed.
"""
from __future__ import annotations

import builtins
import contextlib

import pandas as pd
import pytest


@contextlib.contextmanager
def import_blocked(name: str):
    """Make ``import <name>`` raise ImportError for the duration.

    Only the exact name and its submodules are blocked; everything else
    imports normally, so the code under test still reaches the arm by its
    ordinary route rather than dying somewhere earlier for an unrelated
    reason.
    """
    real = builtins.__import__

    def fake(module, globals=None, locals=None, fromlist=(), level=0):
        if module == name or module.startswith(name + "."):
            raise ImportError(f"blocked for the test: {name}")
        return real(module, globals, locals, fromlist, level)

    builtins.__import__ = fake
    try:
        yield
    finally:
        builtins.__import__ = real


def test_the_blocker_blocks_only_what_it_names():
    """The instrument itself, before anything is measured with it.

    A blocker that caught too much would make every test below pass for the
    wrong reason, and a blocker that caught nothing would make them all
    vacuous. Both failure modes are silent, so both are checked here.
    """
    with import_blocked("adjustText"):
        with pytest.raises(ImportError):
            import adjustText  # noqa: F401
        import json  # noqa: F401  - an unrelated import still works
        assert json.dumps({"ok": True}) == '{"ok": true}'
    import adjustText  # noqa: F401  - and the block is lifted afterwards


def test_the_volcano_annotation_arm_fires_when_adjusttext_is_missing():
    """spacr/plot.py:7435 -- reachable with adjustText INSTALLED.

    The arm re-raises with an install instruction, which is the whole reason
    it exists: the bare ImportError from the failed import names the module
    but not what the user should do about it.
    """
    from spacr.plot import volcano_plot

    frame = pd.DataFrame({
        "gene": [f"g{i}" for i in range(6)],
        "lfc": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "p": [0.001, 0.01, 0.5, 0.01, 0.001, 0.0001],
    })

    with import_blocked("adjustText"):
        with pytest.raises(ImportError) as caught:
            # annotate_max IS REQUIRED to reach the arm. With no thresholds
            # and no cap, plot.py:7426 zeroes `eligible` and the import is
            # never executed -- the first version of this test passed
            # `annotate=True` alone, drove nothing, and failed loudly with
            # DID NOT RAISE. That is the failure this file is about, caught on
            # itself.
            volcano_plot(frame, fold_change_col="lfc", p_value_col="p",
                         name_col="gene", annotate=True, annotate_max=3)

    message = str(caught.value)
    assert "adjustText" in message
    assert "pip install adjustText" in message, (
        "the arm exists to add the install instruction; catching a bare "
        f"ImportError from anywhere else would also pass: {message!r}")


def test_the_same_call_succeeds_when_the_import_is_not_blocked():
    """The control, and the thing that makes the test above mean something.

    Without this, a volcano_plot that raised ImportError for an unrelated
    reason -- a bad column name, a missing optional at module scope -- would
    satisfy the assertion above and the arm would never have run.
    """
    import matplotlib
    matplotlib.use("Agg")
    from spacr.plot import volcano_plot

    frame = pd.DataFrame({
        "gene": [f"g{i}" for i in range(6)],
        "lfc": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "p": [0.001, 0.01, 0.5, 0.01, 0.001, 0.0001],
    })
    result = volcano_plot(frame, fold_change_col="lfc", p_value_col="p",
                          name_col="gene", annotate=True, annotate_max=3)
    assert result is not None
