"""A settings-driven fold must still find its form after its row is dropped.

A module whose page IS a settings form reads that form out of ``APP_META``
-- ``defaults_module`` and ``entry``. ``unregister_app`` pops APP_META along
with the registry row, so dropping the row of a settings-driven fold opens
its page on nothing at all: the button works, the tab appears, and every
setting the module has is gone. It fails silently, which is why it is
asserted here rather than left to be noticed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module                          # noqa: E402
from spacr.qt.screens.map_barcodes import build_settings_screen  # noqa: E402

#: The folded modules whose page is a generated settings form, with the
#: least number of rows each must still offer. The floor is deliberately
#: low -- this guards "the form vanished", not the exact row count, which
#: moves whenever a setting is added.
SETTINGS_FOLDS = {
    "barcode_qc": 5,
    "anndata_export": 5,
    "illumination": 5,
    "activation": 5,
}


@pytest.mark.parametrize("key,floor", sorted(SETTINGS_FOLDS.items()))
def test_a_folded_settings_page_still_has_its_settings(qapp, key, floor):
    """The page opens on a real form, not on an empty one."""
    screen = build_settings_screen(key, None)
    rows = screen._settings_model.collect()
    assert len(rows) >= floor, (
        f"{key}: the folded page offers {len(rows)} settings. Its row is "
        f"gone and APP_META went with it, so the form has nothing to read.")


@pytest.mark.parametrize("key", sorted(SETTINGS_FOLDS))
def test_the_premise_holds_these_really_are_folded(qapp, key):
    """If a key gets a row back, this file is testing the wrong thing."""
    assert key not in {row[0] for row in app_module.APPS}


def test_the_sweep_row_is_gone(qapp):
    """Regression's parameter sweep was the last settings-driven fold."""
    assert "parameter_sweep" not in {row[0] for row in app_module.APPS}
