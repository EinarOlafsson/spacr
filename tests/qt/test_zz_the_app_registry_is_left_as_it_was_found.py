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
