"""Offscreen construction and visual-contract smoke tests for every module.

Most screen tests instantiate one screen class directly.  That leaves two
important integration gaps:

* a new entry can be added to :data:`spacr.qt.app.APPS` but fail only through
  the real ``MainWindow`` factory; and
* a QWidget method name can accidentally override a Qt virtual.  The former
  ``TrainCompareScreen.metric()`` did exactly that: every ordinary unit test
  passed, while showing the screen made Qt call it with a paint-device
  argument and could terminate the process.

This module deliberately shows and grabs every registered screen using the
offscreen platform.  For the shared settings screen it also checks the two
visual/help contracts users see on every row: linked help lives on the label,
and the label/API-dot wrapper paints the section's dark-gray surface rather
than an opaque black rectangle.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QPoint
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from spacr.qt.app import APPS, MainWindow
from spacr.qt.button_roles import action_role, install_button_roles
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.theme import DARK_PALETTE
from spacr.qt.widgets.info_link import InfoLink


class _FactoryHost(QObject):
    """Stand-in ``self`` for the unbound ``MainWindow._build_screen``.

    ``_build_screen`` hands ``self._on_*`` bound methods to ``connect()``, so
    this host has to answer for every one of them.  Listing them by hand made
    the whole parametrized sweep go red every time a new signal was wired —
    ``remote_submit_requested`` landed in 1b02e8ec and took all 17 AppScreen
    cases down, six hours after this file was last touched, with nothing
    actually wrong with the product.

    So unknown attributes are resolved against ``MainWindow`` itself instead:
    anything the real window defines is stubbed as a no-op, and anything it
    does *not* define still raises ``AttributeError``.  That is the only case
    worth failing on here — it is a connect() to a slot that does not exist,
    which the shipped MainWindow would hit just as hard.
    """

    def _snapshot_current_screen_settings(self):
        # Not a signal slot: the Queue screen calls this one through
        # `wire_add_current` and needs a real (app_key, settings) pair back,
        # so a no-op stub will not do.
        return "mask", {}

    def __getattr__(self, name):
        # Only reached when normal lookup fails, so QObject's own attributes
        # and the override above still win.  `vars()` rather than `getattr`:
        # MainWindow's Qt base classes carry hundreds of inherited methods,
        # and silently stubbing one of those would hide a genuine mistake.
        if callable(vars(MainWindow).get(name)):
            return lambda *_args, **_kwargs: None
        raise AttributeError(
            f"MainWindow._build_screen wired {name!r}, but MainWindow does "
            f"not define it")


def _setting_row_contract(screen: AppScreen, qapp) -> None:
    """Check label help, API links and the rendered section background."""
    rows = [
        (section, label, field)
        for section in screen._settings_sections
        for label, field in section._row_widgets
    ]
    assert rows, f"{screen.app_key}: the settings screen has no rows"

    for _section, label, field in rows:
        key = field.property("settingKey")
        assert key, f"{screen.app_key}: a setting field has no settingKey"
        assert isinstance(label, QLabel), (
            f"{screen.app_key}.{key}: setting label is not a QLabel")
        # AppScreen uses the clickable HoverTooltip rather than Qt's native
        # tooltip (the native popup vanishes before its API link can be
        # clicked). The linked HTML therefore lives in label metadata/map.
        html = (
            label.property("apiTooltipHtml")
            or screen._html_tip_map.get(label, "")
        )
        assert html, (
            f"{screen.app_key}.{key}: help is missing from the label")
        assert "href=" in html, (
            f"{screen.app_key}.{key}: label help has no API link")
        wrapper = label.parentWidget()
        assert wrapper is not None
        assert wrapper.objectName() == "SettingLabelWithInfo", (
            f"{screen.app_key}.{key}: label/API wrapper is missing")
        links = wrapper.findChildren(InfoLink)
        assert len(links) == 1, (
            f"{screen.app_key}.{key}: expected one API dot, got {len(links)}")
        assert links[0].url().startswith(("https://", "http://"))

    # Render one representative row per module.  Structural QSS assertions
    # alone missed the original black rectangle: only the composed screenshot
    # proves a transparent wrapper actually shows its SectionCard through.
    section, label, _field = rows[0]
    section.show()
    section.set_expanded(True)
    qapp.processEvents()
    wrapper = label.parentWidget()
    image = screen.grab().toImage()
    origin = wrapper.mapTo(screen, QPoint(0, 0))
    points = (
        (1, 1),
        (max(1, wrapper.width() - 2), 1),
        (1, max(1, wrapper.height() - 2)),
        (max(1, wrapper.width() - 2), max(1, wrapper.height() - 2)),
    )
    expected = QColor(DARK_PALETTE["surface"]).name()
    observed = [
        QColor(image.pixel(origin.x() + x, origin.y() + y)).name()
        for x, y in points
    ]
    assert observed == [expected] * len(points), (
        f"{screen.app_key}: setting-label background is {observed}, "
        f"expected its container color {expected}")


@pytest.mark.parametrize(
    "app_key",
    [key for key, _name, _description, _section in APPS],
)
def test_every_registered_module_constructs_shows_and_renders(
        qtbot, qt_theme_applied, app_key):
    """Exercise the real screen factory and one composed dark-theme frame."""
    install_button_roles(qt_theme_applied)
    host = _FactoryHost()
    screen = MainWindow._build_screen(host, app_key)
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()
    qt_theme_applied.processEvents()

    build_errors = [
        label.text() for label in screen.findChildren(QLabel)
        if label.text().startswith("Failed to build settings")
    ]
    assert not build_errors, f"{app_key}: {build_errors}"
    frame = screen.grab()
    assert not frame.isNull(), f"{app_key}: screen did not render"
    # A few dense tools declare a minimum width larger than 1200. Qt correctly
    # honours that contract when resize(1200, 720) is requested.
    assert frame.width() >= 1200 and frame.height() >= 720

    # Every requested semantic action is tagged after its Show/Polish event.
    for button in screen.findChildren(QPushButton):
        expected_role = action_role(button.text())
        if expected_role is not None:
            assert button.property("buttonActionRole") == expected_role, (
                f"{app_key}: {button.text()!r} is not styled as "
                f"{expected_role}")

    if isinstance(screen, AppScreen):
        _setting_row_contract(screen, qt_theme_applied)


def _slots_build_screen_reaches_for() -> list:
    """Every ``self._on_*`` / ``self._snapshot_*`` name ``_build_screen`` uses.

    Read off the bytecode rather than listed here: ``co_names`` holds exactly
    the attribute names the function loads, so the day a new
    ``connect(self._on_…)`` lands, this list grows with it and the host below
    is asked about the new name without anyone remembering to say so.
    """
    return sorted(
        name for name in MainWindow._build_screen.__code__.co_names
        if name.startswith("_on_") or name.startswith("_snapshot_"))


def test_factory_host_answers_every_slot_build_screen_wires():
    """The host must satisfy the real factory, and only the real factory.

    This replaces a test that asserted ``_on_train_requested`` resolves and an
    invented name does not — both of which were true of the *hand-written*
    four-name host as well, so it pinned nothing.  The regression that
    actually happened is the missing name: ``_on_remote_submit_requested``
    was wired into every AppScreen branch and the host had never heard of it,
    which took all 17 AppScreen cases red.  Asking for the whole set off the
    bytecode is the assertion that would have caught it.
    """
    host = _FactoryHost()
    wired = _slots_build_screen_reaches_for()
    # Guards the guard: if `_build_screen` is ever rewritten so it no longer
    # reaches for `self._on_*` by name (a dispatch table, say), `wired` goes
    # empty and the loop below asserts nothing at all.
    assert "_on_remote_submit_requested" in wired, (
        "the AppScreen branch no longer wires _on_remote_submit_requested by "
        "name — re-read _build_screen and re-point this test")
    for name in wired:
        assert callable(getattr(host, name)), \
            f"_FactoryHost cannot answer {name}, which _build_screen wires"
    # The Queue screen needs a real pair back, not a no-op stub.
    assert host._snapshot_current_screen_settings() == ("mask", {})
    # And the fallback must not paper over a `connect()` to nothing: a name
    # MainWindow never defines is a wiring bug, and still explodes.
    with pytest.raises(AttributeError):
        host._on_slot_main_window_does_not_have


def test_training_runs_does_not_override_qwidget_metric():
    """Qt must retain its ``QPaintDevice.metric(enum)`` virtual method."""
    from spacr.qt.screens.train_compare import TrainCompareScreen

    assert "metric" not in TrainCompareScreen.__dict__
