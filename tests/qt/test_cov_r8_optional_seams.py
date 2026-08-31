"""Optional seams: a napari bridge that will not import, and two Qt shims.

Each of these runs only when an optional dependency is absent or when Qt
hands over an object shaped differently from the usual one. All carry an
inert `# pragma: no cover`.

The rule they share is the same one the rest of spaCR follows for
optional features: a missing extra is REPORTED, in the pane the user is
looking at, and it does not take the screen down.
"""
from __future__ import annotations

import builtins

import pytest

pytestmark = pytest.mark.qt


class TestTheNapariBridge:
    """`open_in_napari` needs an optional extra that may not be there."""

    @pytest.fixture()
    def screen(self, qtbot, tmp_path):
        from spacr.qt.screens.make_masks import NapariBridgeScreen

        widget = NapariBridgeScreen()
        qtbot.addWidget(widget)
        return widget

    def test_a_bridge_that_will_not_import_is_reported_not_raised(
            self, screen, tmp_path, monkeypatch):
        """THE UNCOVERED BLOCK.

        `spacr[napari]` is an extra. Without it the import fails, and the
        screen has to say so in its own status pane -- a traceback out of
        a button handler tells the user nothing about which extra to
        install.
        """
        mask = tmp_path / "mask.npy"
        mask.write_bytes(b"\x00")
        monkeypatch.setattr(screen, "mask_path", lambda: str(mask))
        monkeypatch.setattr(screen, "image_path", lambda: "")

        said = []
        monkeypatch.setattr(screen, "say", said.append)

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "napari_bridge" in name or "napari_bridge" in (fromlist or ()):
                raise ImportError("no module named 'napari'")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert screen.open_in_napari() is None
        assert said, "the failure was silent"
        assert "napari" in said[-1].lower()

    def test_no_mask_chosen_is_reported_before_any_import(self, screen,
                                                          monkeypatch):
        """The cheaper refusal comes first, so nothing is imported at all."""
        monkeypatch.setattr(screen, "mask_path", lambda: "")
        said = []
        monkeypatch.setattr(screen, "say", said.append)
        assert screen.open_in_napari() is None
        assert said == ["Choose a mask file first."]

    def test_a_mask_path_that_is_not_a_file_is_refused(self, screen,
                                                       tmp_path, monkeypatch):
        monkeypatch.setattr(screen, "mask_path",
                            lambda: str(tmp_path / "absent.npy"))
        said = []
        monkeypatch.setattr(screen, "say", said.append)
        assert screen.open_in_napari() is None
        assert said == ["Choose a mask file first."]


class TestTheRegressionBackendFieldShims:
    """Two places where Qt may hand over something unexpected."""

    @pytest.fixture()
    def field(self, qtbot):
        from spacr.qt.screens.settings_model import _RegressionBackendField

        widget = _RegressionBackendField()
        qtbot.addWidget(widget)
        return widget

    def test_a_combo_whose_view_has_gone_falls_through(self, field,
                                                       monkeypatch):
        """`combo.view()` raises once the C++ half is destroyed.

        The filter still has to answer -- it is called for every event
        the combo receives, including during teardown.
        """
        from PySide6.QtCore import QEvent, QObject

        class _Gone:
            def view(self):
                raise RuntimeError("Internal C++ object already deleted.")

        monkeypatch.setattr(field, "combo", _Gone())
        assert field.eventFilter(QObject(), QEvent(QEvent.Type.None_)) in (
            True, False)

    def test_no_combo_at_all_falls_through(self, field, monkeypatch):
        from PySide6.QtCore import QEvent, QObject

        monkeypatch.setattr(field, "combo", None)
        assert field.eventFilter(QObject(), QEvent(QEvent.Type.None_)) in (
            True, False)

    def test_an_event_with_only_pos_is_still_read(self, field):
        """`event.position()` is Qt6; `event.pos()` is the older spelling.

        The hover handler is given whatever Qt delivers, and an event
        object without `position()` must not take the popup down with an
        AttributeError.
        """
        from PySide6.QtCore import QPoint

        seen = []

        class _OldStyleEvent:
            @staticmethod
            def pos():
                return QPoint(3, 4)

        class _View:
            @staticmethod
            def indexAt(point):
                seen.append(point)

                class _Index:
                    @staticmethod
                    def isValid():
                        return False
                return _Index()

        field._hover_popup_row(_View(), _OldStyleEvent())
        assert seen == [QPoint(3, 4)], (
            "the older event spelling was not read")

    def test_a_modern_event_is_read_through_position(self, field):
        from PySide6.QtCore import QPoint, QPointF

        seen = []

        class _ModernEvent:
            @staticmethod
            def position():
                return QPointF(7.0, 9.0)

        class _View:
            @staticmethod
            def indexAt(point):
                seen.append(point)

                class _Index:
                    @staticmethod
                    def isValid():
                        return False
                return _Index()

        field._hover_popup_row(_View(), _ModernEvent())
        assert seen == [QPoint(7, 9)]
