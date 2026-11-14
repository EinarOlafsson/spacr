"""A fold button when the artwork, the host or the maturity is not what it was.

Folding a module takes its row out of the app registry, so everything the
button says about itself afterwards comes from the host screen that kept it.
Three of those handovers can go wrong quietly:

* a host module that will not import takes the whole inventory with it unless
  it is skipped,
* a key with no icon shipped for it draws as an empty square nobody can name,
* and a stage restated after the row is gone has to move the widget-local
  ``:checked`` fill as well as the property the hover rule reads, or a switch
  hovers in one colour and lights in another.
"""
from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets import fold_strip                          # noqa: E402
from spacr.qt.widgets.fold_strip import (                        # noqa: E402
    FOLD_HOST_MODULES, FoldButton, folded_modules,
)


@pytest.mark.parametrize("before,after", [("dark", "light"), ("light", "dark")])
def test_preferences_reinks_existing_fold_icons_without_resizing_them(
        qtbot, qapp, before, after):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import app as app_module, preferences

    owner = QWidget()
    qtbot.addWidget(owner)
    owner.refresh_theme = types.MethodType(app_module.MainWindow.refresh_theme, owner)
    owner.refresh_language = lambda: None
    preferences.set_theme(before)
    preferences.apply_preferences_to_app(qapp)
    button = FoldButton("train_cellpose", parent=owner, checkable=True)
    button.setChecked(True)
    geometry = (button.size(), button.iconSize())
    original = button.icon().pixmap(button.iconSize()).toImage()

    preferences.set_theme(after)
    preferences.apply_preferences_to_app(qapp)
    preferences._refresh_owner_window(owner)
    expected = app_module._icon_for_app("train_cellpose").pixmap(button.iconSize()).toImage()
    actual = button.icon().pixmap(button.iconSize()).toImage()
    assert original != expected
    assert actual == expected
    assert (button.size(), button.iconSize()) == geometry
    assert button.isChecked() and button.app_key == "train_cellpose"


class TestOneBrokenHostDoesNotEmptyTheInventory:

    def test_a_host_that_will_not_import_is_skipped_not_fatal(self, monkeypatch):
        """The other hosts' folded modules still come back.

        ``folded_modules`` is walked while a screen is being built and the
        hosts import this module back, so an import that raises is a live
        possibility rather than a hypothetical. Letting it out would take
        down the screen that was only asking what is folded into it.
        """
        intact = folded_modules()
        assert intact, "no folded modules to speak of; the fixture is stale"
        broken = FOLD_HOST_MODULES[0]

        # UNREADABLE, NOT UNIMPORTABLE. `folded_modules` reads each host's
        # declarations out of its source and imports nothing -- importing
        # every host pulled pandas and scipy into the process before Home had
        # painted. So the failure to simulate is a host whose source cannot
        # be located or parsed, which is what `_host_declarations` guards.
        real_declarations = fold_strip._host_declarations

        def refusing(name, *args, **kwargs):
            if name == broken:
                return None
            return real_declarations(name, *args, **kwargs)

        monkeypatch.setattr(fold_strip, "_host_declarations", refusing)
        surviving = folded_modules()

        assert surviving, "one broken host emptied the whole inventory"
        assert all(host != broken for _n, _d, _s, host in surviving.values())
        kept = {key for key, entry in intact.items() if entry[3] != broken}
        assert set(surviving) == kept


class TestAKeyWithNoArtwork:

    def test_the_button_falls_back_to_the_module_initial(self, qtbot,
                                                         monkeypatch):
        """An iconless fold is still identifiable on the masthead.

        The button carries no label, so with neither icon nor initial it is
        an empty square -- indistinguishable from every other fold on the
        strip and from a button that failed to build.
        """
        monkeypatch.setattr(fold_strip.iconset, "app_icon",
                            lambda key: None)
        button = FoldButton("timelapse")
        qtbot.addWidget(button)

        assert button.icon().isNull()
        assert button.text() == "T"
        assert button.accessibleName().lower().startswith("t")
        assert button.toolTip()
