"""Paint hooks that run on every field frame do not redo their work.

A field theme repaints every widget above the backdrop on every frame, sixty
times a second. Measured offscreen at 2560x1440 on 2026-10-12, a Mask
Generation field frame cost 1.36-1.52 times a Home one, and part of the
difference was paint hooks: the field fade drew two antialiased rounded
rectangles with seventeen-stop gradients under every field (160 us a field
against 23 us for the cached pixmap, loaded machine), it was asked about the
paint event of every widget in the application, and it, the switches and
the splitter handles built a ``QSettings`` per paint to find out which theme
was on (21 store reads a Mask frame, 50-90 us each).

These tests pin what replaced that work: the rasterised field chrome, the
theme resolved once per preference application, the fade installed on the
fields only, and the splash roles every palette carries derived once.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QRegion
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget

from spacr.qt import preferences as prefs
from spacr.qt import theme as theme_mod
from spacr.qt.widgets import field_fade as ff


@pytest.fixture
def clean_caches(qapp):
    """Start and finish with empty caches, whatever ran before."""
    ff.invalidate_field_fade()
    yield
    ff.invalidate_field_fade()


def _direct(widget, theme_name) -> QImage:
    """The chrome as the hook drew it before the cache: straight onto the image."""
    chrome = theme_mod.field_chrome(theme_name)
    radius = float(chrome["radius"])
    inset = ff.FIELD_BORDER_PX / 2.0
    rect = QRectF(widget.rect()).adjusted(inset, inset, -inset, -inset)
    fill_colour, fill_alpha = chrome["fill"]
    line_colour, line_alpha = chrome["border"]
    image = QImage(widget.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(ff._ramped(fill_colour, fill_alpha,
                                       rect.left(), rect.right())))
    painter.drawRoundedRect(rect, radius, radius)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QBrush(ff._ramped(line_colour, line_alpha,
                                          rect.left(), rect.right())),
                        ff.FIELD_BORDER_PX))
    painter.drawRoundedRect(rect, radius, radius)
    painter.end()
    return image


def _cached(widget, theme_name) -> QImage:
    image = QImage(widget.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    ff.paint_field_fade(widget, painter, theme_name)
    painter.end()
    return image


@pytest.mark.parametrize("theme_name", ["dark", "light", "glass"])
@pytest.mark.parametrize("size", [(240, 32), (97, 27), (3, 3)])
def test_the_cached_chrome_is_the_pixels_the_direct_paint_drew(
        qapp, clean_caches, theme_name, size):
    field = QLineEdit()
    field.resize(*size)
    assert _cached(field, theme_name) == _direct(field, theme_name)
    assert _cached(field, theme_name) == _direct(field, theme_name), (
        "the second paint comes from the cache and must not differ")


def test_one_pixmap_per_size_and_state_and_the_cap_drops_them_all(
        qapp, clean_caches, monkeypatch):
    first = ff._chrome_pixmap(120, 24, 1.0, 4.0, ("#101010", 1.0),
                              ("#202020", 1.0))
    assert ff._chrome_pixmap(120, 24, 1.0, 4.0, ("#101010", 1.0),
                             ("#202020", 1.0)) is first
    focused = ff._chrome_pixmap(120, 24, 1.0, 4.0, ("#101010", 1.0),
                                ("#4c8dff", 1.0))
    assert focused is not first
    sharp = ff._chrome_pixmap(120, 24, 2.0, 4.0, ("#101010", 1.0),
                              ("#202020", 1.0))
    assert (sharp.width(), sharp.height()) == (240, 48)
    assert sharp.devicePixelRatio() == 2.0
    assert len(ff._CHROME) == 3

    monkeypatch.setattr(ff, "_CHROME_CAP", 3)
    ff._chrome_pixmap(60, 24, 1.0, 4.0, ("#101010", 1.0), ("#202020", 1.0))
    assert len(ff._CHROME) == 1, "the cap drops the cache wholesale"


def test_a_field_frame_reads_no_preference_store(qapp, clean_caches,
                                                monkeypatch):
    reads = []
    real = prefs._settings

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(prefs, "_settings", counting)
    field = QLineEdit()
    field.resize(200, 30)
    _cached(field, None)
    first = len(reads)
    assert first >= 1, "the first paint resolves the theme"
    for _frame in range(5):
        _cached(field, None)
    assert len(reads) == first, (
        f"{len(reads) - first} preference reads in five field frames")


def test_invalidation_drops_the_theme_and_the_chrome(qapp, clean_caches):
    field = QLineEdit()
    field.resize(80, 24)
    _cached(field, None)
    assert ff._CHROME and theme_mod._PAINTING_THEME[1] is not None
    ff.invalidate_field_fade()
    assert not ff._CHROME
    assert theme_mod._PAINTING_THEME == [None, None]


def test_a_palette_change_resolves_the_theme_again(qapp, clean_caches,
                                                   monkeypatch):
    answers = iter(["light", "dark"])
    monkeypatch.setattr(prefs, "resolve_effective_theme",
                        lambda: next(answers))
    saved = QApplication.palette()
    try:
        assert theme_mod._painting_theme() == "light"
        assert theme_mod._painting_theme() == "light", "kept between paints"
        changed = QApplication.palette()
        changed.setColor(changed.ColorRole.Window, QColor("#123456"))
        QApplication.setPalette(changed)
        assert theme_mod._painting_theme() == "dark"
        assert theme_mod._painting_palette() == theme_mod.palette_for("dark")
    finally:
        QApplication.setPalette(saved)


def test_an_unreadable_store_paints_dark_and_is_not_remembered(
        qapp, clean_caches, monkeypatch):
    def refuse():
        raise RuntimeError("no store")

    monkeypatch.setattr(prefs, "resolve_effective_theme", refuse)
    assert theme_mod._painting_theme() == "dark"
    assert theme_mod._PAINTING_THEME == [None, None]
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    assert theme_mod._painting_theme() == "light"


def test_without_an_application_the_key_is_none(monkeypatch):
    monkeypatch.setattr(theme_mod.QGuiApplication, "instance",
                        staticmethod(lambda: None))
    assert theme_mod._application_palette_key() is None


def test_the_switch_and_the_handle_paint_without_reading_the_store(
        qtbot, clean_caches, monkeypatch):
    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter
    from spacr.qt.widgets.height_grip import HeightGrip
    from spacr.qt.widgets.toggle import Toggle

    host = QWidget()
    qtbot.addWidget(host)
    switch = Toggle("Masks", host)
    switch.resize(120, 28)
    split = CollapsibleSplitter(Qt.Horizontal, host)
    split.addWidget(QWidget())
    split.addWidget(QWidget())
    split.resize(200, 80)
    grip = HeightGrip(QWidget(host), 20, 400, host)
    grip.resize(80, 12)
    widgets = (switch, split.handle(1), grip)
    for widget in widgets:
        _render(widget)
    reads = []
    real = prefs._settings

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(prefs, "_settings", counting)
    for _frame in range(3):
        for widget in widgets:
            _render(widget)
    assert reads == []


def _render(widget) -> None:
    image = QImage(widget.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    widget.render(painter, QPoint(0, 0), QRegion(widget.rect()),
                  QWidget.RenderFlags(QWidget.DrawChildren))
    painter.end()


def _filters_paint(widget) -> bool:
    """Whether the fade hook is asked about ``widget``'s paint events."""
    calls = []
    real = ff._filter.eventFilter

    def spy(obj, event):
        if event.type() == ff.QEvent.Type.Paint:
            calls.append(obj)
        return real(obj, event)

    ff._filter.eventFilter = spy
    try:
        _render(widget)
    finally:
        del ff._filter.eventFilter
    return widget in calls


def test_only_fields_are_filtered_and_uninstall_lets_them_go(qtbot, qapp,
                                                             clean_caches):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.gil_priority import _application_event_hub

    before = QLineEdit()
    qtbot.addWidget(before)
    before.resize(120, 24)
    ff.uninstall_field_fade(qapp)
    try:
        assert ff.install_field_fade(qapp) is True
        hub = _application_event_hub(qapp, create=False)
        kinds = {kind for kind, chain in hub._by_kind.items()
                 if any(entry[0]() is ff._filter for entry in chain)}
        assert kinds == {ff.QEvent.Type.Polish}, (
            "the hook must not be asked about every paint in the application")
        assert _filters_paint(before), "a field that existed at install"
        after = QLineEdit()
        qtbot.addWidget(after)
        after.resize(120, 24)
        label = QLabel("not a field")
        qtbot.addWidget(label)
        label.resize(120, 24)
        assert _filters_paint(after), "a field polished after install"
        assert not _filters_paint(label)
        assert ff.install_field_fade(qapp) is False, "idempotent"
        assert _filters_paint(after)
        hook = ff._filter
        assert ff.uninstall_field_fade(qapp) is True
        calls = []
        hook.eventFilter = lambda obj, event: (
            event.type() == ff.QEvent.Type.Paint and calls.append(obj)) or False
        _render(after)
        assert calls == []
    finally:
        ff.uninstall_field_fade(qapp)


def test_the_splash_roles_are_derived_once_per_surface(monkeypatch):
    derived = []
    real = theme_mod.splash_dim_alpha

    def counting(*args, **kwargs):
        derived.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(theme_mod, "splash_dim_alpha", counting)
    monkeypatch.setattr(theme_mod, "_SPLASH_ROLES", {})
    palette = {"fg": "#eeeeee", "bg": "#101010"}
    first = theme_mod._splash_roles(palette)
    second = theme_mod._splash_roles(palette)
    assert first == second and first is not second
    assert len(derived) == 1
    first["splash_bg"] = "#ff0000"
    assert theme_mod._splash_roles(palette)["splash_bg"] == "#101010", (
        "a caller's edit must not reach the cache")
    monkeypatch.setattr(theme_mod, "_SPLASH_ROLES",
                        {(str(i), "#000000"): {} for i in range(256)})
    theme_mod._splash_roles(palette)
    assert list(theme_mod._SPLASH_ROLES) == [("#eeeeee", "#101010")]
