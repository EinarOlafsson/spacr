"""settings_model's two module-level guards: import-time, and load-bearing.

Both run while the module is being imported, which is why nothing had
run them: by the time a test can call anything, the import has already
succeeded the ordinary way.

They matter more than most guards precisely because of when they run. A
module-level exception is not a degraded feature -- it is an ImportError
that takes the whole settings system, and therefore every module screen,
with it. Each of these turns a missing optional piece into a smaller
capability instead.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest

pytestmark = pytest.mark.qt

MODULE = "spacr.qt.screens.settings_model"


def _reimport(monkeypatch=None):
    """Re-import settings_model from scratch and return it."""
    for name in [n for n in list(sys.modules) if n.startswith(MODULE)]:
        del sys.modules[name]
    return importlib.import_module(MODULE)


@pytest.fixture(autouse=True)
def _restore_the_module():
    """Put the ORIGINAL module objects back, not a fresh import.

    These tests deliberately import a crippled settings_model, and a
    re-import afterwards is NOT enough to undo it: importing again builds
    a NEW module object with new classes, while everything that did
    `from ... import X` earlier still holds the old one. Two classes then
    claim the same name and identity checks elsewhere start failing --
    which is exactly what happened, in
    test_a_module_screen_is_built_once_not_twice.

    Saving the module objects and putting them back leaves the session
    exactly as it was found.
    """
    saved = {name: module for name, module in sys.modules.items()
             if name == MODULE or name.startswith(MODULE + ".")}
    try:
        yield
    finally:
        for name in [n for n in list(sys.modules)
                     if n == MODULE or n.startswith(MODULE + ".")]:
            del sys.modules[name]
        sys.modules.update(saved)


class TestTheTrainingBasisInventory:
    """`_ALL_BASIS_SETTINGS` is the set of keys the basis owns."""

    def test_the_inventory_is_populated_on_a_normal_install(self):
        module = _reimport()
        assert module._ALL_BASIS_SETTINGS, (
            "the basis inventory is empty on a healthy install")
        assert isinstance(module._ALL_BASIS_SETTINGS, set)

    def test_a_missing_training_basis_leaves_an_empty_inventory(self,
                                                               monkeypatch):
        """THE UNCOVERED GUARD -- and an empty set is the right fallback.

        The inventory exists so `refresh_training_basis_enablement`
        cannot switch a control back on that something else disabled for
        its own reasons. With no inventory, it owns NOTHING and therefore
        re-enables nothing: the conservative answer.

        A `None` here would be worse than an empty set -- every `in`
        test against it would raise, and they run on every settings
        refresh.
        """
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if (name == "spacr.training_basis"
                    or "training_basis" in (fromlist or ())):
                raise ImportError("training_basis is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        module = _reimport()
        assert module._ALL_BASIS_SETTINGS == set()

    def test_the_inventory_supports_membership_either_way(self,
                                                          monkeypatch):
        """Whichever branch ran, callers do `key in _ALL_BASIS_SETTINGS`."""
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if (name == "spacr.training_basis"
                    or "training_basis" in (fromlist or ())):
                raise ImportError("training_basis is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        crippled = _reimport()
        assert ("anything" in crippled._ALL_BASIS_SETTINGS) is False


class TestTheAlphabetChipRegistration:
    """The theme seam is present in every real launch -- but not asserted."""

    def test_a_theme_that_will_not_take_the_block_does_not_stop_the_import(
            self, monkeypatch):
        """THE UNCOVERED GUARD.

        Registering a QSS block is decoration. If the theme refuses it,
        the alphabet chips lose their styling -- and the settings system
        still has to import, because every module screen is built from
        it. Raising here would mean a chip's colour taking the whole
        application down.
        """
        from spacr.qt import theme

        def refuse(*_a, **_k):
            raise RuntimeError("the theme registry is unavailable")

        monkeypatch.setattr(theme, "register_widget_qss", refuse)
        module = _reimport()
        assert module is not None
        # and the thing the block was for still exists
        assert hasattr(module, "_AlphabetSelect")

    def test_the_block_is_registered_on_a_healthy_launch(self):
        """The other side: the chip QSS really is installed."""
        from spacr.qt import theme

        _reimport()
        assert "SettingAlphabetChip" in theme._WIDGET_QSS, (
            "the alphabet chip block is no longer registered at import")
