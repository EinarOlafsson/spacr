"""The settings column must never be a black box. Measured, not asserted.

This regressed three times. Each time the report was the same sentence --
"there is a black box behind the settings categories" -- and each time the
fix swept one more container transparent, which made it worse, because the
thing *behind* the containers had no colour of its own. With the ambient
animation switched off there was nothing painting the page at all, so what
showed through was the blanket ``QWidget { background-color: bg }`` from
``_window_block``, and in the dark theme ``bg`` is literally ``#000000``.

So this file does not check a property of the code. It renders a real
:class:`AppScreen` onto a magenta page and counts pixels, because every
earlier fix passed the code-shaped checks that existed at the time.

Two traps are baked in, both of which produced a confidently wrong answer
before:

1. ``theme.stylesheet()`` MUST be applied to the QApplication first.
   Without it, every widget renders in Qt's default palette -- a uniform
   (239, 239, 239) -- and the probe reports a clean page for a screen that
   is black in the real app.

2. The page must start as a colour nothing else uses, not as transparent.
   Filling with 0 and counting "black" pixels scores an *unpainted* region
   as clean, and unpainted is exactly the failure: in the real window the
   thing underneath is ``bg``, which is ``#000000``.

Hence magenta. A magenta pixel means nobody painted it, which in the real
window is the black box. A (0, 0, 0) pixel means something painted the
window colour. Both are failures and both are counted.
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QScrollArea, QWidget

#: Magenta: not in any palette, so it can only be the untouched page.
UNPAINTED = QColor(255, 0, 255)

#: Every module the black box was reported on, plus the ones that share
#: the settings-column construction with them. `mask` and `timelapse` are
#: the widest forms (190 and 204 keys); `cellpose_masks` is among the
#: narrowest, and a short form leaves more page showing, which is where
#: the hole was most visible.
MODULES = (
    "mask", "timelapse", "motility", "measure", "ml_analyze", "classify",
    "map_barcodes", "regression", "external_masks", "illumination",
    "train_cellpose", "cellpose_masks", "umap", "activation",
    "barcode_qc", "model_compare", "model_zoo", "control_chart",
)

#: Below this the column is sound. Not zero: a handful of samples land on
#: antialiased glyph edges and on the focus ring, which are legitimately
#: dark. The failures this guards against were 44.4% and 45.9%, so the
#: gap between passing and failing is three orders of magnitude and the
#: exact threshold is not load-bearing.
TOLERANCE_PCT = 0.5


def _sample_settings_column(app_key: str, ambient: bool) -> tuple[float, float]:
    """Render the screen and return (unpainted %, pure-black %).

    :param ambient: the backdrop animation. ``False`` is the case that
        actually broke -- with it on, the ambient widget paints the page
        and hides the hole, which is why this was reported by users and
        not caught here.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    # trap 1 -- the palette + QSS come from the `qt_theme_applied` fixture
    # the tests below depend on. Without it every widget renders in Qt's
    # default (239, 239, 239) and this probe reports a clean page.
    app = QApplication.instance()

    was = preferences.get_ambient_enabled()
    preferences.set_ambient_enabled(ambient)
    try:
        screen = AppScreen(app_key)
        screen.resize(1600, 1000)
        screen.show()
        for _ in range(10):
            app.processEvents()

        box = next((w for w in screen.findChildren(QScrollArea)
                    if w.objectName() == "SettingsBox"), None)
        if box is None:
            pytest.skip(f"{app_key} has no SettingsBox")

        page = QImage(screen.size(), QImage.Format_ARGB32)
        page.fill(UNPAINTED)                       # trap 2 -- see module docstring
        screen.render(page, QPoint(), screen.rect(),
                      QWidget.RenderFlag.DrawChildren)

        top_left = box.mapTo(screen, QPoint(0, 0))
        unpainted = black = total = 0
        for y in range(top_left.y() + 4, top_left.y() + box.height() - 4, 4):
            for x in range(top_left.x() + 4, top_left.x() + box.width() - 4, 4):
                colour = page.pixelColor(x, y)
                total += 1
                if colour == UNPAINTED:
                    unpainted += 1
                elif colour.red() == colour.green() == colour.blue() == 0:
                    black += 1
        screen.deleteLater()
        return 100.0 * unpainted / total, 100.0 * black / total
    finally:
        preferences.set_ambient_enabled(was)


@pytest.mark.parametrize("app_key", MODULES)
def test_the_settings_column_is_not_a_black_box(qt_theme_applied, app_key):
    """With the backdrop off, the settings column still has a colour.

    This is the exact configuration that was reported. At the commit
    before the fix, `mask` scored 44.4% unpainted here.
    """
    unpainted, black = _sample_settings_column(app_key, ambient=False)
    assert unpainted + black < TOLERANCE_PCT, (
        f"{app_key}: {unpainted:.1f}% of the settings column is unpainted "
        f"and {black:.1f}% is the window colour. Unpainted is the bug -- in "
        f"the real window `bg` (#000000) shows through, which is the black "
        f"box. Do not fix this by making one more container transparent; "
        f"that is what caused it. The page itself needs a colour."
    )


@pytest.mark.parametrize("app_key", MODULES[:6])
def test_the_settings_column_is_not_a_black_box_with_the_backdrop_on(
        qt_theme_applied, app_key):
    """And with the backdrop on, which is the default.

    Fewer modules: the ambient widget paints the page in this mode, so
    this direction has never been the one that broke. It is here so that
    a future change to the page colour cannot fix the off case by
    breaking the on case.
    """
    unpainted, black = _sample_settings_column(app_key, ambient=True)
    assert unpainted + black < TOLERANCE_PCT, (
        f"{app_key}: {unpainted:.1f}% unpainted, {black:.1f}% window colour "
        f"with the ambient backdrop enabled"
    )
