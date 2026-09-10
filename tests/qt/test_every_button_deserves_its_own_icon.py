"""Counted, not eyeballed: how many buttons still draw the default icon.

Instruction 355, verbatim: "make an intuitive, elegant logo for all buttons
that are currently using the default logo, dor example diagnostics in
regression".

The number was measured before anything was changed, by booting a
``MainWindow`` offscreen, opening all 36 registered module screens, and
comparing every ``QAbstractButton``'s icon against the puzzle piece
:func:`spacr.qt.iconset.icon` answers an unknown name with. Of the 208
buttons that carry an icon at all, **seven** were wearing the fallback, from
four distinct causes:

===========================  ==========================================
key / semantic name          where it drew the puzzle piece
===========================  ==========================================
``import_images``            dock row + Import's fold button
``explain_cv``               dock row + Classify's fold button
``regression_diagnostics``   dock row + Regression's fold button
``trash``                    Run History's "Clear all"
===========================  ==========================================

Three of those four are fixed and the count is **two**. The remaining two
buttons are both ``regression_diagnostics``, and they are left on the
fallback ON PURPOSE -- see :class:`TestWhatIsStillOwed`. Instruction 355's
own rule is that the module must be guessable from its mark, and no shipped
artwork and no Font Awesome 5 glyph draws "residuals and influence"; a
wrong-but-present icon is worse than the fallback, which at least reads as
"nobody has chosen one yet".

These tests walk the resolvers and the real widgets the GUI uses, so a new
module registered without artwork fails them the day it is added rather than
whenever somebody next looks at the dock.
"""
from __future__ import annotations

import itertools
import os
import re

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QAbstractButton

from spacr.qt import iconset
from spacr.qt.theme import THEMES

#: Edge the icons are rasterised at for the silhouette comparisons. Larger
#: than the 26 px the dock draws at and the 16 px the instruction calls the
#: floor, so two marks that genuinely differ cannot come out equal purely
#: for lack of pixels.
RENDER_PX = 48

#: The dock draws at 26 px and the fold strip is not much larger, but the
#: instruction sets the bar at 16: "An icon that only works at 64 is a
#: picture, not an icon." Every claim about legibility below is measured
#: here, not at RENDER_PX.
SMALL_PX = 16

#: How many buttons wore the fallback before this instruction, measured as
#: described in the module docstring. Quoted so the numbers in the prose and
#: the numbers in the assertions cannot drift apart.
FALLBACK_BUTTONS_BEFORE = 7

#: And after. Zero: `regression_diagnostics.png` was drawn on 2026-09-02,
#: which was the last one.
FALLBACK_BUTTONS_AFTER = 0

#: Nothing is owed artwork any more. Kept as a name rather than deleted
#: because the two assertions below read better saying "nothing is missing"
#: than comparing against an empty list literal in three places.
STILL_OWED = None

#: What was fixed, and with what. ``explain_cv`` borrows a bundled PNG
#: through :data:`spacr.qt.iconset.SHARED_ICON_ASSETS`; the other two are
#: Font Awesome glyphs named in :data:`spacr.qt.iconset._NAME_TO_GLYPH`,
#: which is the same route ``align``, ``data_manager`` and ``train_cellpose``
#: already take when no bundled PNG says the right thing.
FIXED = {
    "explain_cv": "ml_analyze.png",
    "import_images": "fa5s.images",
    "trash": "fa5s.trash",
}


def render(icon, px: int = RENDER_PX):
    """``icon`` painted onto transparency as a ``(px, px, 4)`` array."""
    image = QImage(px, px, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        icon.paint(painter, 0, 0, px, px)
    finally:
        painter.end()
    buffer = image.constBits()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(px, px, 4).copy()


def coverage(pixels) -> float:
    """Fraction of the square the artwork actually inks."""
    return float((pixels[:, :, 3] > 8).mean())


def _shares_artwork_on_purpose(first: str, second: str) -> bool:
    """Whether these two keys are recorded as drawing one picture.

    `iconset.SHARED_ICON_ASSETS` is the table where a deliberate sharing is
    written down WITH ITS REASON, so consulting it is how this file tells a
    decision from an accident.
    """
    shared = getattr(iconset, "SHARED_ICON_ASSETS", {})
    for key, other in ((first, second), (second, first)):
        alias = shared.get(key)
        if alias and iconset.bundled_icon_path(other) == os.path.join(
                iconset.RESOURCE_DIR, alias):
            return True
    return False


def silhouette_difference(first, second) -> float:
    """Fraction of pixels where two icons disagree about being there."""
    delta = np.abs(first[:, :, 3].astype(int) - second[:, :, 3].astype(int))
    return float((delta > 64).mean())


def every_key():
    """Every key the GUI resolves to an icon: tiles AND folded modules.

    The folded ones are the point. Instruction 318 folded 33 modules onto 11
    mastheads, and all three of the keys measured on the fallback were folded
    children -- a walk of ``APPS`` alone finds none of them.
    """
    from spacr.qt.app import APPS, folded_children

    keys = {row[0] for row in APPS}
    for children in folded_children().values():
        keys.update(children)
    return sorted(keys)


@pytest.fixture(scope="module")
def fallback(qapp):
    """The puzzle piece an unfiled key falls through to, as pixels.

    Taken from the resolver itself rather than from a hardcoded glyph name,
    so it stays correct if the fallback is ever redrawn.
    """
    return render(iconset.icon("__no_key_will_ever_be_named_this__"))


def is_fallback(icon, fallback_pixels) -> bool:
    """True when ``icon`` paints the shared fallback, or paints nothing."""
    if icon is None or icon.isNull():
        return True
    pixels = render(icon)
    if coverage(pixels) <= 0.0:
        return True
    return silhouette_difference(pixels, fallback_pixels) < 0.02


# ===========================================================================
# 1. The count, on the resolver every surface goes through
# ===========================================================================

class TestTheCount:
    """The measurement instruction 355 asked for, as an assertion."""

    def test_only_one_key_in_the_whole_registry_lacks_a_mark(self, qapp,
                                                             fallback):
        """Three keys fell through here; one still does, and it is named.

        ``_icon_for_app`` is what the dock row, the Home tile and the
        command palette all call, so this one loop covers every surface a
        module key is drawn on. It failed at three before ``explain_cv``
        and ``import_images`` were mapped.
        """
        from spacr.qt.app import _icon_for_app

        missing = sorted(key for key in every_key()
                         if is_fallback(_icon_for_app(key), fallback))
        assert missing == [], (
            f"keys drawing the shared fallback: {missing}. Give the new one "
            f"a line in iconset._NAME_TO_GLYPH or artwork named for the key, "
            f"and say why in one clause.")

    def test_the_bare_resolver_agrees_with_the_app_one(self, qapp, fallback):
        """A fold button calls ``app_icon`` bare, and must not differ.

        ``_icon_for_app`` consults ``spacr.qt.app._ICON_OVERRIDES``;
        ``iconset.app_icon`` on its own does not. A key mapped only in the
        first gets one picture in the dock and the puzzle piece on its fold
        button, which is exactly the split ``SHARED_ICON_ASSETS`` exists to
        stop. Both of this instruction's key fixes are on this side of the
        line for that reason.
        """
        missing = sorted(key for key in every_key()
                         if is_fallback(iconset.app_icon(key), fallback))
        assert missing == []

    def test_no_semantic_glyph_name_the_source_asks_for_is_unmapped(self):
        """``icon("trash")`` returned the puzzle piece for Run History.

        Harvested from the source rather than listed by hand: a new
        ``icon("something")`` at a new call site is caught the day it is
        written, which is the only way this stays true. Run History's
        "Clear all" was the one miss -- the single button in the GUI that
        throws away recorded runs, wearing the artwork of every unfiled key.
        """
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(iconset.__file__))))
        call = re.compile(
            r"(?:iconset\.)?(?:accent_icon|contrast_icon|icon)\("
            r"\s*[\"']([a-z][a-z_0-9]*)[\"']")
        asked = set()
        for folder, _dirs, files in os.walk(os.path.join(root, "spacr", "qt")):
            for name in files:
                if not name.endswith(".py"):
                    continue
                with open(os.path.join(folder, name), encoding="utf-8") as fh:
                    asked.update(call.findall(fh.read()))
        unmapped = sorted(asked - set(iconset._NAME_TO_GLYPH))
        assert not unmapped, (
            f"call sites ask iconset.icon() for names it has no glyph for, "
            f"so those buttons draw the puzzle piece: {unmapped}")


# ===========================================================================
# 2. The same count, on real buttons
# ===========================================================================

class TestTheRealButtons:
    """Resolvers are not buttons. These are the widgets a user clicks."""

    def test_no_dock_row_draws_the_puzzle_piece(self, qtbot, fallback):
        """The dock is 56 rows plus 34 indented fold children.

        Three of those rows drew the fallback when this file was written and
        one survived until `regression_diagnostics.png` was drawn. Built
        rather than reasoned about, because the dock sets its icon from
        ``_icon_for_app`` in two separate places -- once for a module row and
        once for a fold child -- and only the second one produced the misses.
        """
        from spacr.qt.app import Sidebar

        dock = Sidebar()
        qtbot.addWidget(dock)
        wearing = [btn for btn in dock.findChildren(QAbstractButton)
                   if not btn.icon().isNull()
                   and is_fallback(btn.icon(), fallback)]
        assert wearing == [], (
            f"dock rows on the fallback: "
            f"{[b.property('moduleNameSource') or b.text() for b in wearing]}")

    @pytest.mark.parametrize("key", ["import_images", "explain_cv"])
    def test_a_fold_button_now_carries_a_picture_not_a_letter(self, qtbot,
                                                             fallback, key):
        """A ``FoldButton`` with no icon falls back to its own initial.

        So the failure this fixes had two shapes depending on which branch
        ran: the puzzle piece when ``app_icon`` returned the glyph fallback
        (which is what happened -- it is never null), and a bare capital
        letter if it had returned nothing. Both are checked, because a fix
        that swapped one for the other would be no fix.
        """
        from spacr.qt.widgets.fold_strip import FoldButton

        button = FoldButton(key)
        qtbot.addWidget(button)
        assert not button.icon().isNull()
        assert not is_fallback(button.icon(), fallback)
        assert button.text() == "", (
            f"{key} fell through to its initial instead of an icon")


# ===========================================================================
# 3. Each new mark, judged by the instruction's own four rules
# ===========================================================================

class TestTheNewMarksAreUsable:
    """"It must read at 16 px, survive re-inking, and differ in shape.\""""

    @pytest.mark.parametrize("theme", THEMES)
    @pytest.mark.parametrize("key", ["import_images", "explain_cv"])
    def test_it_inks_a_real_shape_at_sixteen_pixels(self, qapp, theme, key):
        """Not "an icon exists" -- "an icon is visible in the slot it lives in".

        The floor is 4 %: at 16 px that is ten inked pixels, below which
        there is no silhouette left to recognise. The two marks measure
        23 % (``ml_analyze.png``) and 67 % (``fa5s.images``).
        """
        pixels = render(iconset.app_icon(key, theme=theme), SMALL_PX)
        assert coverage(pixels) > 0.04

    @pytest.mark.parametrize("theme", THEMES)
    @pytest.mark.parametrize("key", ["import_images", "explain_cv"])
    def test_it_is_not_the_fallback_in_any_theme(self, qapp, fallback, theme,
                                                 key):
        assert silhouette_difference(
            render(iconset.app_icon(key, theme=theme)), fallback) > 0.2

    def test_the_borrowed_png_re_inks_for_the_theme(self):
        """``explain_cv`` draws a bundled PNG, and those bake their colour.

        ``Sidebar.refresh_icons`` rebuilds on a theme change precisely
        because a QIcon cannot be recoloured after the fact, so the mark has
        to come out different per theme or one theme gets an invisible one.
        ``ml_analyze.png`` clears the contrast floor in all four themes --
        16.1 dark, 15.8 light, 11.3 cell, 9.7 glass against 3.0 required.
        """
        path = iconset.bundled_icon_path("explain_cv")
        assert path is not None and os.path.isfile(path)
        for theme in THEMES:
            assert iconset.icon_contrast(path, theme) >= \
                iconset.MIN_ICON_CONTRAST, theme
        assert (iconset.icon_ink_color(path, "light")
                != iconset.icon_ink_color(path, "dark"))

    @pytest.mark.parametrize("host,children", [
        ("classify_merged", ["classifier_evaluation", "explain_cv",
                             "activation", "train_compare",
                             "feature_explorer"]),
        ("foreign", ["import_images", "convert", "external_masks"]),
        # REGRESSION'S STRIP WAS NOT COVERED, which is how a mark could have
        # been drawn that agreed with `profiler` on 90.2 % of the square and
        # nothing would have said so. `regression_diagnostics` sits here.
        ("regression", ["volcano_explorer", "hit_list", "methods_export",
                        "investigate_hit", "profiler",
                        "regression_diagnostics"]),
    ])
    def test_no_two_marks_on_one_fold_strip_are_the_same_shape(
            self, qapp, host, children):
        """A fold strip is where these sit side by side at 26 px.

        The bar is 10 % of the square disagreeing, which is well under the
        tightest pair that already ships on these two strips (Format
        Converter against External Masks, at 21 %) -- this test is here to
        catch a duplicate, not to relitigate the existing artwork.
        """
        keys = [host] + list(children)
        drawn = {k: render(iconset.app_icon(k, theme="dark"), SMALL_PX)
                 for k in keys}
        for first, second in itertools.combinations(keys, 2):
            if _shares_artwork_on_purpose(first, second):
                # A RECORDED SHARING IS NOT A DUPLICATE. `investigate_hit`
                # draws `hit_list.png` because the mark names A HIT rather
                # than a tile, and `SHARED_ICON_ASSETS` says so with its
                # reason. What this test is for is the sharing nobody
                # decided.
                continue
            assert silhouette_difference(drawn[first], drawn[second]) > 0.10, (
                f"{first} and {second} are one picture at {SMALL_PX} px")


# ===========================================================================
# 4. The borrowings, declared rather than coincidental
# ===========================================================================

class TestTheBorrowingIsWrittenDown:
    """``ml_analyze.png`` is retired artwork, and the alias says so."""

    def test_explain_cv_says_out_loud_whose_picture_it_draws(self):
        assert iconset.SHARED_ICON_ASSETS["explain_cv"] == "ml_analyze.png"
        resolved = iconset.bundled_icon_path("explain_cv")
        assert os.path.basename(resolved) == "ml_analyze.png"

    def test_the_file_it_borrows_is_claimed_by_no_live_gui_key(self, qapp):
        """The reason this is a retirement and not a collision.

        ``ml_analyze`` is still a CLI key, but the ML and CV classify tiles
        merged into ``classify_merged`` and that screen draws
        ``classify.png``. Measured over all 61 registered-plus-folded keys:
        exactly one resolves to ``ml_analyze.png``, and it is Explain CV
        Model. If somebody re-registers Explain CV's neighbour under this
        file, this fails and the alias needs rethinking.
        """
        from spacr.qt.app import _ICON_OVERRIDES

        users = [key for key in every_key()
                 if (iconset.bundled_icon_path(key, _ICON_OVERRIDES.get(key))
                     or "").endswith("ml_analyze.png")]
        assert users == ["explain_cv"], users

    def test_dedicated_artwork_would_retire_the_alias(self, tmp_path,
                                                      monkeypatch):
        """``<key>.png`` outranks the alias, so drawing one needs no code."""
        (tmp_path / "ml_analyze.png").write_bytes(b"")
        (tmp_path / "explain_cv.png").write_bytes(b"")
        monkeypatch.setattr(iconset, "RESOURCE_DIR", str(tmp_path))
        assert os.path.basename(
            iconset.bundled_icon_path("explain_cv")) == "explain_cv.png"

    @pytest.mark.parametrize("name,glyph", sorted(
        (n, g) for n, g in FIXED.items() if g.startswith("fa5s.")))
    def test_the_glyph_choices_are_the_ones_recorded(self, name, glyph):
        """The prose above names a glyph; the table must still hold it."""
        assert iconset._NAME_TO_GLYPH[name] == glyph


# ===========================================================================
# 5. The debt
# ===========================================================================

class TestWhatIsStillOwed:
    """Regression's Diagnostics -- the button the maintainer named."""

    def test_the_last_one_was_drawn_rather_than_borrowed(self):
        """Regression's Diagnostics -- the button the maintainer named.

        It stayed on the fallback while every candidate was worse than no
        icon: `outliers.png` is the live Outliers module's mark and taking
        it makes two modules one picture; `dose_response.png` is a sigmoid
        near-identical at 16 px to `profiler.png`, which sits BESIDE
        Diagnostics on Regression's strip; and Font Awesome's `stethoscope`
        says "diagnostics" the way a gear says "settings", which this
        instruction rules out.

        So it was drawn: a dashed zero line, six residuals scattered either
        side of it, and one point ringed as influential -- which is what a
        residual plot is, and what "Diagnostics" does. Measured against the
        artwork that already ships: 8.9 % ink at 26 px where the bundled set
        runs 3-5 % at 16, contrast 15.8 light and 16.1 dark against a floor
        of 3.0, and it re-inks per theme like every other mask.
        """
        path = iconset.bundled_icon_path("regression_diagnostics")
        assert path is not None and os.path.isfile(path)
        for theme in THEMES:
            assert iconset.icon_contrast(path, theme) >= \
                iconset.MIN_ICON_CONTRAST, theme
        assert (iconset.icon_ink_color(path, "light")
                != iconset.icon_ink_color(path, "dark"))

    def test_the_debt_is_closed_and_it_was_seven(self):
        """The headline number, so a regression cannot hide behind prose."""
        assert FALLBACK_BUTTONS_AFTER == 0
        assert FALLBACK_BUTTONS_BEFORE - FALLBACK_BUTTONS_AFTER == 7
        assert STILL_OWED is None
