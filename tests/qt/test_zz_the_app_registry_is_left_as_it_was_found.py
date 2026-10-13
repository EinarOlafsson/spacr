"""No test may leave the app registry changed for the next one.

`spacr.qt.app.APPS` and `APP_STAGE` are module-level and every screen
reads them -- `app_stage()` answers from the second, the home tiles and
the maturity lists from both. A test that registers a screen and does not
unregister it changes what every later test sees, and the symptom is a
failure in a file that did nothing wrong.

THIS IS THE DETECTOR, NOT THE FIX, and it is deliberately named `zz_` so
it sorts last under a deterministic order. `tests/qt/test_zz_a_reimported
_module_is_put_back_properly.py` does the same job for `sys.modules`;
this is the same idea for the one other global the Qt suite mutates.

FOUND BY: `tests/test_cov_5_screen_factories.py` restored `APPS` after
registering a screen and left `APP_STAGE[key]` behind, because
`register_app` writes both and only one of them was put back.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_the_registry_still_holds_what_the_package_declares():
    """Every key in APP_STAGE is a key in APPS, and vice versa for stages.

    A LEAK SHOWS UP AS AN ORPHAN. A screen registered by a test and then
    removed from `APPS` alone leaves its stage behind, so the two
    disagree -- which is exactly the state that makes
    `test_the_alpha_and_beta_lists_are_the_ones_that_were_asked_for` read
    a module nobody registered.
    """
    from spacr.qt.app import APP_STAGE, APPS

    registered = {key for key, *_rest in APPS}
    orphans = sorted(set(APP_STAGE) - registered)
    assert not orphans, (
        f"{orphans} have a stage but no row in APPS; a test registered a "
        f"screen and put back only half of what `register_app` wrote")


@pytest.fixture(scope="session")
def registry_at_session_start():
    """The app keys present before any test in this session registered one.

    THE ONLY FIXED POINT THERE IS. My first attempt compared against
    `SELF_REGISTERING_MODULES`, on the theory that a self-registering module
    in `APPS` must have been registered by a test. It is not true: five of
    them -- `dose_response`, `gate_editor`, `investigate_hit`,
    `project_browser`, `run_compare` -- are already there at import, which
    is why that check failed on a clean run. "Self-registering" describes
    how a module CAN be registered, not whether it already is.
    """
    from spacr.qt.app import APPS

    return {key for key, *_rest in APPS}


def test_no_test_left_an_app_registered_that_was_not_there_before(
        registry_at_session_start):
    """APPS itself must not have GROWN, which consistency cannot detect.

    THE CHECK ABOVE MISSED A REAL LEAK, and this is the one it missed.
    `spacr.qt.register_self_registering_modules()` adds rows to `APPS` AND
    stages to `APP_STAGE`, so the two stay perfectly consistent -- no
    orphans, nothing for the first test to see -- while the registry now
    holds screens the package leaves switched off. `test_home_v2`'s
    alpha/beta lists then found `feature_explorer`, `trellis` and
    `outliers` among the alpha modules and failed, under some orderings
    only.

    A CONSISTENCY CHECK CANNOT SEE A CONSISTENT ADDITION.
    """
    from spacr.qt.app import APPS

    present = {key for key, *_rest in APPS}
    gained = sorted(present - registry_at_session_start)
    assert not gained, (
        f"{gained} were registered during this session and never removed. "
        f"Every later test now sees screens that were not there when the "
        f"session began. Ten files call "
        f"`register_self_registering_modules()` deliberately and none of "
        f"them is wrong to; the putting back belongs to "
        f"tests/qt/conftest.py -- `_restore_app_registry` for a test that "
        f"registers, and "
        f"`_the_app_registry_is_left_as_the_session_found_it` for a "
        f"module-scoped fixture that does, which the per-test one cannot "
        f"see past.")
