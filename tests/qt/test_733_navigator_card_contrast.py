"""Navigator cards keep readable icons and labels on their own surface in every theme."""
from collections import Counter

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication

from spacr.qt import app
from spacr.qt.theme import contrast_ratio
from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def _hex(color):
    """Return an opaque ``#rrggbb`` for a sampled pixel."""
    return '#{:02x}{:02x}{:02x}'.format(color.red(), color.green(), color.blue())


def _pixels(image, rect):
    """Read every composited pixel inside ``rect`` of a window grab."""
    return [_hex(image.pixelColor(x, y))
            for y in range(rect.top(), rect.bottom() + 1)
            for x in range(rect.left(), rect.right() + 1)]


def _ink(pixels, surface, minimum):
    """Return the share of pixels reaching ``minimum`` and the best ratio against ``surface``."""
    ratios = [contrast_ratio(pixel, surface) for pixel in set(pixels)]
    counts = Counter(pixels)
    strong = sum(counts[pixel] for pixel in counts if contrast_ratio(pixel, surface) >= minimum)
    return strong / len(pixels), max(ratios)


@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('backdrop', [False, True], ids=['backdrop_off', 'backdrop_on'])
def test_card_icons_and_labels_contrast_with_the_surface_behind_them(
        qtbot, protocol_profile, theme, backdrop):
    """Icons reach 3:1 and labels 4.5:1 against the colour actually composited behind them."""
    from spacr.qt.preferences import apply_preferences_to_app

    qapp = QApplication.instance()
    palette, sheet = qapp.palette(), qapp.styleSheet()
    signature = getattr(qapp, '_spacr_preferences_style_signature', None)
    protocol_profile.set_theme(theme)
    protocol_profile.set_ambient_enabled(backdrop)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    try:
        window._tour_timer.stop()
        window._consent_timer.stop()
        window.resize(1366, 768)
        window.show()
        qtbot.waitExposed(window)
        apply_preferences_to_app(qapp)
        page = window._startup
        page.path_edit.setFocus()
        qtbot.keyClicks(page.path_edit, 'mask')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: {'mask', 'make_masks'} <= {card._key for card in page.field.cards},
                        timeout=5000)
        qtbot.wait(200)
        image = window.grab().toImage()
        for card in page.field.cards:
            if card._key not in ('mask', 'make_masks'):
                continue
            box = QRect(card.mapTo(window, card.rect().topLeft()), card.size()).adjusted(4, 4, -4, -4)
            surface = Counter(_pixels(image, box)).most_common(1)[0][0]
            icon = QRect(card.icon.mapTo(window, card.icon.rect().topLeft()), card.icon.size())
            title = QRect(card.title.mapTo(window, card.title.rect().topLeft()), card.title.size())
            share, best = _ink(_pixels(image, icon.intersected(box)), surface, 3.0)
            assert best >= 3.0 and share >= 0.02, (theme, backdrop, card._key, surface, share, best)
            share, best = _ink(_pixels(image, title.intersected(box)), surface, 4.5)
            assert best >= 4.5 and share >= 0.01, (theme, backdrop, card._key, surface, share, best)
    finally:
        window.close()
        qapp.setPalette(palette)
        qapp.setStyleSheet(sheet)
        qapp._spacr_preferences_style_signature = signature
