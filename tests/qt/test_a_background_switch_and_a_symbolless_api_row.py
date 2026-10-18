"""Two setting lookups the settings panel makes for every row.

A per-organelle background-removal switch is filed under the organelle its
name ends with, whatever word sits between. A setting whose API row names
its module but no public symbol gets no anchor, rather than a link to the
top of a page that never mentions it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import settings_model as sm  # noqa: E402
from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("key, owner", [
    ("remove_background_organelleb", "organelleb"),
    ("remove_background_x_organellec", "organellec"),
])
def test_a_background_switch_belongs_to_the_organelle_it_ends_with(key, owner):
    assert sm.object_of_setting(key) == owner


def test_a_setting_whose_api_row_names_no_symbol_has_no_anchor():
    module, symbol, _exact = SETTING_API_TARGETS["annotated_classes"]
    assert symbol == ""
    assert sm._anchor_inside("annotated_classes", module.removeprefix("spacr.")) == ""
