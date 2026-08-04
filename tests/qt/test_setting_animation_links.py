"""Qt contracts for purple setting-animation links and their popup."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QToolButton, QWidget

from spacr.qt.screens.settings_model import (
    install_api_tooltips,
    refresh_api_tooltips,
)
from spacr.qt.widgets.animation_link import (
    AnimationLink,
    AnimationPopup,
    SettingLinkStack,
)
from spacr.qt.widgets.info_link import InfoLink


def _setting_row(qtbot, key: str):
    root = QWidget()
    qtbot.addWidget(root)
    form = QFormLayout(root)
    label = QLabel(key.replace("_", " ").title(), root)
    field = QSpinBox(root)
    form.addRow(label, field)
    install_api_tooltips(root, "umap", {field: key})
    root.resize(520, 120)
    root.show()
    qtbot.waitExposed(root)
    return root, label, field


def test_mapped_setting_stacks_purple_dot_above_teal_dot(qtbot):
    root, label, _field = _setting_row(qtbot, "n_neighbors")
    animation = root.findChild(AnimationLink)
    api = root.findChild(InfoLink)
    stack = root.findChild(SettingLinkStack)

    assert animation is not None
    assert api is not None
    assert stack is not None
    assert animation._colours[0] == "#9B009B"
    assert animation.size() == api.size()
    assert animation.parentWidget() is stack
    assert api.parentWidget() is stack
    assert animation.geometry().center().y() < api.geometry().center().y()

    wrapper = label.parentWidget()
    animation_centre = animation.mapTo(wrapper, animation.rect().center()).y()
    api_centre = api.mapTo(wrapper, api.rect().center()).y()
    label_centre = label.mapTo(wrapper, label.rect().center()).y()
    assert abs(((animation_centre + api_centre) / 2.0) - label_centre) <= 1.0


def test_unmapped_setting_keeps_api_dot_without_empty_stack(qtbot):
    root, _label, _field = _setting_row(qtbot, "setting_without_animation")
    assert len(root.findChildren(InfoLink)) == 1
    assert not root.findChildren(AnimationLink)
    assert not root.findChildren(SettingLinkStack)


def test_animation_caption_retranslates_with_setting_help(qtbot):
    root, _label, field = _setting_row(qtbot, "n_neighbors")
    animation = root.findChild(AnimationLink)

    refresh_api_tooltips(root, "sv")
    assert animation.toolTip().startswith("Visa animation för")
    assert animation.accessibleName() == animation.toolTip()
    assert field.toolTip() == ""

    refresh_api_tooltips(root, "ko")
    assert "애니메이션 보기" in animation.toolTip()
    assert animation.accessibleName() == animation.toolTip()


def test_click_starts_square_animation_and_hiding_stops_it(qtbot):
    root, _label, _field = _setting_row(qtbot, "n_neighbors")
    animation = root.findChild(AnimationLink)

    qtbot.mouseClick(animation, Qt.LeftButton)
    popup = AnimationPopup.instance()
    qtbot.waitUntil(popup.isVisible)
    view = popup.animation_view()
    assert view.frame_count() > 1
    assert view.width() == view.height() == AnimationPopup.DISPLAY_SIZE
    assert view.is_playing()
    assert not popup.findChildren(QToolButton)

    popup.hide()
    qtbot.waitUntil(lambda: not view.is_playing())
    root.close()


def test_the_clicked_dot_shows_the_same_zoom_the_hover_does(qtbot):
    """The dot used to open a BIGGER window with a SMALLER illustration.

    ``QMovie`` can only scale a GIF, so the 300-pixel popup showed the raw
    frame — content covering about a third of the square — while a 220-pixel
    hover showed the zoomed one at three quarters. Same asset, same rule.
    """
    from spacr.qt.widgets import animation_zoom as az

    root, _label, _field = _setting_row(qtbot, "n_neighbors")
    link = root.findChild(AnimationLink)

    qtbot.mouseClick(link, Qt.LeftButton)
    popup = AnimationPopup.instance()
    qtbot.waitUntil(popup.isVisible)

    frame = az.from_qimage(popup.animation_view().pixmap().toImage())
    measured = az.content_extent([frame])
    raw = az.source_content_extent(str(link.animation().path))
    assert az.MIN_FILL <= measured <= az.MAX_FILL, (
        f"the clicked popup shows {measured:.1%} of its square")
    assert measured > raw, (
        f"no zoom was applied: {measured:.1%} of the square vs {raw:.1%} raw")
    popup.hide()
    root.close()


def test_an_undecodable_asset_shows_the_error_card_instead_of_raising(
        qtbot, tmp_path):
    """The failure branch moved from ``QMovie.isValid`` to the zoom loader.

    A missing or corrupt GIF must still end in the error card — a hover that
    raises out of a click handler takes the event loop with it.
    """
    from types import SimpleNamespace

    root, _label, _field = _setting_row(qtbot, "n_neighbors")
    link = root.findChild(AnimationLink)
    broken = tmp_path / "broken.gif"
    broken.write_bytes(b"not a gif at all")
    # A distinct slug on purpose: the player keeps the frames it already has
    # when the same slug is loaded again, so reusing the real one would have
    # this test measure the PREVIOUS animation still on screen.
    corrupt = SimpleNamespace(
        slug="deliberately_broken", title="Broken", path=broken)

    popup = AnimationPopup.instance()
    popup.show_animation(link, corrupt)

    assert popup.isVisible()
    assert not popup.animation_view().isVisible()
    assert popup.animation_view().frame_count() == 0
    assert popup._error.isVisible()
    assert "could not be loaded" in popup._error.text()
    popup.hide()
    root.close()
