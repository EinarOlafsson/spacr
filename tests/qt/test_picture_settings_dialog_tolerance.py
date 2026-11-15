"""The picture settings dialog opens, and prices its cap, whatever it is built from.

A decimal cap from a settings file is normalised to a live whole count; a
panel that offers no cap at all still opens with every other setting; and
API tooltips that cannot be installed cost a hover, not the dialog.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QSpinBox                          # noqa: E402

import spacr.picture_settings as ps                             # noqa: E402
import spacr.qt.screens.settings_model as settings_model        # noqa: E402
from spacr.qt.widgets import picture_settings_dialog as psd     # noqa: E402

pytestmark = pytest.mark.qt

#: Every "<n> objects is ..." sentence a tooltip currently carries.
_COST = re.compile(r"([\d,]+) objects is")


def _costs(label) -> list:
    return _COST.findall(label.toolTip())


def _dialog(qtbot, **kwargs):
    made = psd.PictureSettingsDialog(**kwargs)
    qtbot.addWidget(made)
    return made


def test_the_picture_dialog_tolerates_odd_caps_and_missing_decoration(
        qtbot, monkeypatch, subtests):
    """Cap normalisation and live pricing, a capless panel, and failing API tooltips."""
    with subtests.test(msg="a decimal or a default cap is a live whole count"):
        for values, start in (({"cap": 12.5}, 12), (None, 2000)):
            kwargs = {"values": values} if values else {}
            dialog = _dialog(qtbot, mode=ps.LOAD_IMAGES, **kwargs)
            assert isinstance(dialog._editors["cap"], QSpinBox)
            assert dialog.values()["cap"] == start
            assert type(dialog.values()["cap"]) is int
            assert _costs(dialog._labels["cap"]) == [f"{start:,}"]
            dialog._editors["cap"].setValue(30)
            assert _costs(dialog._labels["cap"]) == ["30"]
            assert dialog.values()["cap"] == 30
            assert type(dialog.values()["cap"]) is int

    with subtests.test(msg="a panel that offers no cap still opens"):
        real = ps.categories

        def without_cap():
            return tuple((title, tuple(k for k in keys if k != "cap"))
                         for title, keys in real())

        with monkeypatch.context() as patch:
            patch.setattr(psd, "categories", without_cap)
            capless = _dialog(qtbot, mode=ps.LOAD_IMAGES)
        ordinary = _dialog(qtbot, mode=ps.LOAD_IMAGES)
        assert "cap" not in capless.values()
        assert capless.tab_of("cap") == ""
        assert "Which cells" in capless.tab_titles()
        assert capless.values()["score_column"] == ordinary.values()["score_column"]
        assert set(ordinary.values()) - set(capless.values()) == {"cap"}
        capless._say_what_the_cap_costs()
        assert capless._cap_help is None

    with subtests.test(msg="API tooltips that cannot be installed cost only the decoration"):
        def boom(*_args, **_kwargs):
            raise RuntimeError("the API metadata could not be read")

        with monkeypatch.context() as patch:
            patch.setattr(settings_model, "install_api_tooltips", boom)
            dialog = _dialog(qtbot, mode="png")
        assert dialog.mode() == "png"
        assert "object_type" in dialog.values()
        assert dialog._labels["object_type"].toolTip() != ""
