"""The picture dialog builds itself out of whatever the defaults turn out to be.

Its starting values come from ``set_annotate_default_settings``, which is
shared with the headless annotator and can both raise and answer with
something that is not a mapping. Neither may stop the dialog opening -- a
user who cannot open the picture settings cannot change the picture, and the
per-key defaults the dialog owns are enough to draw every control.

The same tolerance applies to the API tooltips: decorating the help is a
nice-to-have layered on top of the plain descriptions, so a failure there
costs a hover, not a dialog.

The per-entry help on the annotation-method dropdown is asserted here too,
because it is the only place five picking strategies have room to explain
themselves, and it has to survive an entry that carries no user data.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                       # noqa: E402
from PySide6.QtWidgets import QComboBox, QLineEdit  # noqa: E402

from spacr.picture_settings import PICKING_HELP     # noqa: E402
from spacr.qt.widgets import picture_settings_dialog as psd   # noqa: E402

pytestmark = pytest.mark.qt


def test_defaults_survive_a_shared_default_builder_that_raises(monkeypatch):
    """A broken shared default table must not empty the picture settings."""
    import spacr.settings as settings_module

    def boom(_settings):
        raise KeyError("annotate defaults are unavailable")

    monkeypatch.setattr(settings_module, "set_annotate_default_settings", boom)

    defaults = psd.picture_defaults()

    from spacr.picture_settings import OWN_DEFAULTS
    assert set(OWN_DEFAULTS) <= set(defaults)
    assert defaults["object_type"] == OWN_DEFAULTS["object_type"]


def test_defaults_survive_a_shared_default_builder_that_answers_a_list(
        monkeypatch):
    """A non-mapping answer is discarded rather than iterated as one."""
    import spacr.settings as settings_module

    monkeypatch.setattr(settings_module, "set_annotate_default_settings",
                        lambda _settings: ["not", "a", "mapping"])

    defaults = psd.picture_defaults()

    from spacr.picture_settings import OWN_DEFAULTS
    assert set(OWN_DEFAULTS) <= set(defaults)


def test_picking_help_is_not_attached_to_a_control_that_is_not_a_dropdown(
        qtbot):
    """The helper is a no-op where there are no entries to explain."""
    line = QLineEdit()
    qtbot.addWidget(line)

    psd._attach_picking_help(line, "cell_picking")

    assert line.toolTip() == ""


def test_a_dropdown_entry_with_no_stored_value_is_still_explained(qtbot):
    """The visible label identifies the method when no item data was set."""
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("rank — the most confident cells")      # no item data
    combo.addItem("nothing_like_a_method")

    psd._attach_picking_help(combo, "cell_picking")

    assert combo.itemData(0, Qt.ToolTipRole) == PICKING_HELP["rank"]
    assert combo.itemData(1, Qt.ToolTipRole) is None


def test_the_dialog_opens_when_its_api_tooltips_cannot_be_installed(
        qtbot, monkeypatch):
    """A dialog that cannot decorate its help is still a working dialog."""
    import spacr.qt.screens.settings_model as settings_model

    def boom(*_args, **_kwargs):
        raise RuntimeError("the API metadata could not be read")

    monkeypatch.setattr(settings_model, "install_api_tooltips", boom)

    dialog = psd.PictureSettingsDialog(mode="png")
    qtbot.addWidget(dialog)

    assert dialog.mode() == "png"
    values = dialog.values()
    assert "object_type" in values
    assert dialog._labels["object_type"].toolTip()
