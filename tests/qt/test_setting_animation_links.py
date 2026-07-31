"""Qt contracts for purple setting-animation links and their popup."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie
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


def test_click_starts_square_movie_and_hiding_stops_it(qtbot):
    root, _label, _field = _setting_row(qtbot, "n_neighbors")
    animation = root.findChild(AnimationLink)

    qtbot.mouseClick(animation, Qt.LeftButton)
    popup = AnimationPopup.instance()
    qtbot.waitUntil(popup.isVisible)
    assert popup._movie is not None
    assert popup._movie.isValid()
    assert popup._movie.scaledSize().width() == popup._movie.scaledSize().height()
    assert popup._movie.state() in {QMovie.Running, QMovie.Paused}
    assert not popup.findChildren(QToolButton)

    popup.hide()
    qtbot.waitUntil(lambda: popup._movie is None)
    root.close()
