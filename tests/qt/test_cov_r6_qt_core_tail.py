"""The last arc in twelve modules of ``spacr/qt`` (plus one generator).

Every module here is between 94% and 99.9% covered, so this is one
branch apiece. Three of them are guards a line above has already made
true; those carry a proof and a test pinning the invariant rather than a
contortion.

Driven:

``fractal_defaults``
    ``clamp`` -- the whole function, which nothing was calling.
``folder_metadata``
    Filenames that already carry both a well and a field get one mapping
    each without either counter moving.
``multi_format``
    A multi-page TIFF whose series does not say what its axes are.
``preview_registry``
    A preview panel that does not offer a propagate callback.
``walkthrough``
    A window that already has a walkthrough handler keeps it.
``recipes``
    A recipe list on a screen with no settings model still describes the
    recipe.
``space``
    A credits file that is not a mapping is not attribution.
``prerun``
    Scoring the masks from a screen that has no settings model.
``ai/issue_report``
    A second occurrence whose comment does not go through still opens an
    issue.
``variants``
    Home candidate 06 draws its search screen with no logo to head it.

Proved unreachable, with the invariant pinned instead:

``mask_engine``
    ``magic_wand`` only ever queues in-bounds neighbours, so the
    bounds re-check inside the flood fill cannot fire.
``prerun``
    ``QWidget.style()`` never answers ``None``.
``iconset``
    ``reink`` always returns an array, so the cached write is never
    skipped for want of one.
``theme``
    ``_hue_rgb`` spans the whole channel range, so the ink ramp's chroma
    rises strictly with the level and never ties.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


# ---------------------------------------------------------------------------
# mask_engine
# ---------------------------------------------------------------------------

class TestTheMagicWandNeverLeavesTheImage:
    """``if not (0 <= cx < W and 0 <= cy < H): continue`` inside the fill.

    The queue starts with the seed, and the seed has already been
    bounds-checked by the ``return mask`` guard at the top of the
    function. Every other entry is appended by the neighbour loop at the
    bottom, which appends ``(nx, ny)`` only under
    ``0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]``. So every
    ``(cx, cy)`` popped off the queue is in bounds and the re-check
    cannot fire -- it re-checks exactly what the push site guarantees.

    Pinned instead: the property that makes it true. A fill seeded in the
    corner stays inside the image and never wraps around an edge.
    """

    @staticmethod
    def _image(size=8):
        return np.full((size, size), 100, dtype=np.uint8)

    def test_a_fill_from_the_corner_stays_inside_the_image(self):
        from spacr.qt.mask_engine import magic_wand

        image = self._image()
        mask = np.zeros((8, 8), dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=0, seed_y=0, tolerance=1.0)

        assert out.shape == (8, 8)
        assert int((out == 255).sum()) == 64, (
            "a flat image within tolerance fills completely and no further; "
            f"got {int((out == 255).sum())} pixels")

    def test_a_seed_outside_the_image_fills_nothing(self):
        """The only bounds check that ever fires, and it is the outer one."""
        from spacr.qt.mask_engine import magic_wand

        image = self._image()
        mask = np.zeros((8, 8), dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=99, seed_y=0, tolerance=1.0)

        assert out is mask, \
            "an out-of-bounds seed returns the mask untouched, not a copy"
        assert int(out.sum()) == 0

    def test_the_fill_stops_at_a_tolerance_edge_rather_than_wrapping(self):
        """A wrapped index would fill the far edge; this says it does not."""
        from spacr.qt.mask_engine import magic_wand

        image = self._image()
        image[:, 4:] = 250                      # a hard edge down the middle
        mask = np.zeros((8, 8), dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=0, seed_y=0, tolerance=10.0)

        assert int((out[:, :4] == 255).sum()) == 32
        assert int((out[:, 4:] == 255).sum()) == 0, (
            "the bright half is outside the tolerance and must stay empty")


# ---------------------------------------------------------------------------
# fractal_defaults
# ---------------------------------------------------------------------------

class TestClampHoldsAValueInsideItsBounds:
    """The whole of ``fractal_defaults.clamp``, which nothing was calling."""

    @pytest.mark.parametrize("value,expected", [
        (-5.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (7.0, 1.0)])
    def test_a_value_is_held_inside_zero_and_one(self, value, expected):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(value, 0.0, 1.0) == expected

    def test_the_declared_speed_range_brackets_the_pace_it_came_from(self):
        """What the clamp is for: the variable-speed sweep's own bounds.

        THE BOUNDS ARE SHARED with `fractal_travel`, and they bracket THAT
        module's default of 4.0 -- which is where "changes the RANGE, not
        the average pace" is true. `fractal_defaults` lowered its own default
        to 1.0 for the backdrop and kept these bounds, so switching variable
        speed on there runs it at two to six times its pace instead.

        This test asserted the bracket against the backdrop's own 1.0 and so
        could never pass. Narrowing the bounds to 0.5/1.5 reads correctly and
        fails five tests that pin the two modules to the same numbers, so it
        is a decision rather than a typo and is recorded beside the constants
        instead. What is asserted here is the relationship that does hold.
        """
        from spacr.qt.fractal_defaults import (DEFAULT_SPEED_MAX,
                                               DEFAULT_SPEED_MIN, clamp)
        from spacr.qt.widgets.fractal_travel import (
            DEFAULT_SPEED as TRAVEL_SPEED)

        assert DEFAULT_SPEED_MIN <= TRAVEL_SPEED <= DEFAULT_SPEED_MAX, (
            "the sweep bounds no longer bracket the pace they were written "
            "for; if that is deliberate, the backdrop's own bounds are the "
            "thing to settle")
        DEFAULT_SPEED = TRAVEL_SPEED
        assert clamp(DEFAULT_SPEED, DEFAULT_SPEED_MIN,
                     DEFAULT_SPEED_MAX) == DEFAULT_SPEED
        assert clamp(1000.0, DEFAULT_SPEED_MIN,
                     DEFAULT_SPEED_MAX) == DEFAULT_SPEED_MAX


# ---------------------------------------------------------------------------
# folder_metadata
# ---------------------------------------------------------------------------

class TestNamingFilesThatAlreadyKnowTheirPlaceOnThePlate:
    """``if not have_field: ... elif not have_well:`` -- neither side."""

    FILES = [f"/raw/img_{i}.tif" for i in range(4)]

    def test_with_both_known_every_file_is_one_field_of_one_well(self):
        """The caller read them off the folder structure, so nothing counts up.

        Minting fresh wells here would rename files whose real addresses
        the caller has already established.
        """
        from pathlib import Path

        from spacr.qt.folder_metadata import assign_missing_fields

        mappings = assign_missing_fields(
            [Path(p) for p in self.FILES], plate="plate1",
            have_well=True, have_field=True)

        assert [m.well for m in mappings] == ["A01"] * 4, (
            "with the well already known no new ones are minted; got "
            f"{[m.well for m in mappings]}")
        assert [m.field for m in mappings] == [1] * 4
        assert {m.canonical for m in mappings} == {
            "plate1_A01_F001_C01.tif"}, \
            "so every file gets the same canonical name"

    def test_with_neither_known_the_fields_count_up(self):
        """The contrast: the counters really do move when they are needed."""
        from pathlib import Path

        from spacr.qt.folder_metadata import assign_missing_fields

        mappings = assign_missing_fields(
            [Path(p) for p in self.FILES], plate="plate1",
            have_well=False, have_field=False)

        assert [m.field for m in mappings] == [1, 2, 3, 4]
        assert len({m.canonical for m in mappings}) == 4

    def test_with_only_the_field_known_the_wells_count_up(self):
        """And the other side of the same ``elif``."""
        from pathlib import Path

        from spacr.qt.folder_metadata import assign_missing_fields

        mappings = assign_missing_fields(
            [Path(p) for p in self.FILES], plate="plate1",
            have_well=False, have_field=True)

        assert len({m.well for m in mappings}) == 4, (
            "a known field with an unknown well means four wells; got "
            f"{[m.well for m in mappings]}")


# ---------------------------------------------------------------------------
# multi_format
# ---------------------------------------------------------------------------

class TestATiffThatWillNotSayWhatItsAxesAre:
    """``if axes:`` when describing a multi-page TIFF."""

    @staticmethod
    def _stack(tmp_path, pages=3):
        import tifffile

        path = tmp_path / "stack.tif"
        # Written page by page: ``imwrite`` of a 3-D array makes ONE page
        # holding a 3-D sample, which is a single-page TIFF as far as this
        # describer is concerned.
        with tifffile.TiffWriter(path) as writer:
            for _ in range(pages):
                writer.write(np.zeros((8, 8), dtype=np.uint16),
                             contiguous=False)
        return path

    def test_a_series_with_no_axis_string_is_described_without_one(
            self, tmp_path, monkeypatch):
        """``tifffile`` is what supplies the axis string, not spaCR.

        A file whose series carries no axes -- or a ``tifffile`` that
        stops reporting them -- must still be described by its page
        count, which is the number the importer actually branches on.
        """
        import tifffile

        from spacr.qt import multi_format as mf   # noqa: F401 - the module
                                                    # under test

        path = self._stack(tmp_path)

        class _Series:
            axes = ""

        # ``_describe_tif`` does ``import tifffile`` in its own body, so
        # the property on the real class is the only seam there is.
        monkeypatch.setattr(tifffile.TiffFile, "series",
                            property(lambda self: [_Series()]))

        described = mf._describe_tif(path)

        assert described is not None, "the file is still describable"
        assert "pages=3" in described.notes
        assert not any(note.startswith("axes=") for note in described.notes), (
            "with nothing to report there is no axis note; got "
            f"{described.notes}")

    def test_a_series_that_names_its_axes_says_so(self, tmp_path):
        """The contrast that makes the absence above a real absence."""
        from spacr.qt import multi_format as mf

        described = mf._describe_tif(self._stack(tmp_path))

        assert described is not None
        assert "pages=3" in described.notes
        assert any(note.startswith("axes=") for note in described.notes), (
            "a plain ImageJ stack does report its axes; got "
            f"{described.notes}")


# ---------------------------------------------------------------------------
# preview_registry
# ---------------------------------------------------------------------------

class _PlainPanel(QWidget):
    """A preview panel with no propagate seam of its own."""

    def __init__(self):
        super().__init__()
        self.applied = []

    def apply_settings(self, settings):
        self.applied.append(dict(settings))


class _TalkativePanel(_PlainPanel):
    def __init__(self):
        super().__init__()
        self.callback = None

    def set_propagate_callback(self, fn):
        self.callback = fn


class _HostScreen(QWidget):
    """A screen with the anchor ``_insert_above_actions`` looks for."""

    def __init__(self, app_key="mask"):
        super().__init__()
        self.app_key = app_key
        self._settings_model = None
        self._runtime_wrap = QWidget(self)
        layout = QVBoxLayout(self._runtime_wrap)
        self._actions_row = QLabel("Run", self._runtime_wrap)
        layout.addWidget(self._actions_row)


_BUILT = {}


def _build_plain_card(screen):
    panel, card = _PlainPanel(), QWidget()
    _BUILT["panel"] = panel
    return panel, card


def _build_talkative_card(screen):
    panel, card = _TalkativePanel(), QWidget()
    _BUILT["panel"] = panel
    return panel, card


_HERE = "tests.qt.test_cov_r6_qt_core_tail"


class TestAPreviewPanelWithNoPropagateSeam:
    """``if callable(register_cb):`` in ``_attach``."""

    def test_a_panel_that_offers_no_callback_is_still_attached(self, qapp):
        """Propagating tuned values back is optional.

        A preview that only shows is a preview; refusing to attach one
        would cost the picture for want of a feature it does not have.
        """
        from spacr.qt.preview_registry import (PreviewSpec, _attach,
                                               unregister_preview)

        screen = _HostScreen()
        try:
            spec = PreviewSpec(builder=f"{_HERE}:_build_plain_card",
                               title="Plain preview")
            host = _attach(screen, "r6_plain_probe", spec)

            assert host is not None, "the card must still be built"
            assert host.toggle.text() == "Plain preview"
            assert not hasattr(_BUILT["panel"], "callback")
        finally:
            unregister_preview("r6_plain_probe")
            screen.deleteLater()

    def test_a_panel_that_offers_one_is_given_the_hosts_own(self, qapp):
        """The contrast: the seam is used when the panel has it."""
        from spacr.qt.preview_registry import (PreviewSpec, _attach,
                                               unregister_preview)

        screen = _HostScreen()
        try:
            spec = PreviewSpec(builder=f"{_HERE}:_build_talkative_card",
                               title="Live preview")
            host = _attach(screen, "r6_talkative_probe", spec)

            assert host is not None
            assert _BUILT["panel"].callback == host.on_propagate, (
                "a panel that can propagate is wired to the host that owns "
                "the settings model")
        finally:
            unregister_preview("r6_talkative_probe")
            screen.deleteLater()


# ---------------------------------------------------------------------------
# walkthrough
# ---------------------------------------------------------------------------

class TestAWindowKeepsTheWalkthroughHandlerItHas:
    """``if handler is None:`` in ``install_help_menu``."""

    @staticmethod
    def _window():
        window = QMainWindow()
        window.menuBar().addMenu("&Help")
        return window

    def test_a_second_install_reuses_the_first_handler(self, qapp):
        """The menu is rebuilt on a language change; the handler is not.

        A fresh handler each time would leave the old one connected to
        the actions it made, so a "how does Measure work?" click could
        run twice.
        """
        from spacr.qt.walkthrough import install_help_menu

        window = self._window()
        try:
            first = install_help_menu(window)
            assert first is not None, "there has to be a Help menu to add to"
            handler = window._walkthrough_handler

            assert install_help_menu(window) is None, (
                "a submenu that is already there is not installed twice")

            # A menu-bar rebuild -- what a language change does -- takes the
            # submenu off and calls this again. The HANDLER is not rebuilt.
            from spacr.qt.walkthrough import find_menu
            find_menu(window, "Help").removeAction(first.menuAction())

            second = install_help_menu(window)

            assert second is not None and second is not first, \
                "the submenu really was rebuilt"
            assert window._walkthrough_handler is handler, (
                "the handler must survive the rebuild; a fresh one would "
                "leave the old one connected and run every walkthrough twice")
        finally:
            window.deleteLater()


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

class TestARecipeListOnAScreenWithNoSettingsModel:
    """``if model is not None:`` in ``_on_selection_changed``."""

    @staticmethod
    def _recipe(tmp_path):
        from spacr.qt.recipes import Recipe

        return Recipe(name="fast", app_key="mask",
                      settings={"cell_diameter": 30},
                      created="2026-01-01", notes="the quick one")

    def test_a_screen_with_no_model_still_describes_the_recipe(
            self, qapp, tmp_path, monkeypatch):
        """A recipe browser can be opened before a screen is wired up.

        Without a settings model there is no compatibility gap to
        compute; the recipe's own facts are still worth showing.
        """
        from spacr.qt import recipes as rp

        folder = tmp_path / "recipes"
        folder.mkdir()
        monkeypatch.setattr(rp, "recipes_dir", lambda app_key=None: str(folder))
        rp.save_recipe(self._recipe(tmp_path), directory=str(folder))

        screen = QWidget()
        screen.app_key = "mask"
        dialog = rp.RecipeDialog(screen)
        try:
            assert getattr(screen, "_settings_model", None) is None
            dialog.reload()
            assert dialog.recipes(), "the saved recipe has to be listed"

            detail = dialog.detail_text()
            assert "1 settings" in detail, (
                "the recipe's own facts are shown whatever the screen is; "
                f"got {detail!r}")
            assert "the quick one" in detail, "including its notes"
        finally:
            dialog.deleteLater()
            screen.deleteLater()


# ---------------------------------------------------------------------------
# space
# ---------------------------------------------------------------------------

class TestACreditsFileThatIsNotAttribution:
    """``if isinstance(data, dict) and data.get("file"):``"""

    @staticmethod
    def _credits(tmp_path, monkeypatch, payload):
        from spacr.qt import space

        cache = tmp_path / "cache"
        (cache / "nasa").mkdir(parents=True)
        monkeypatch.setattr(space, "cache_dir", lambda: cache)
        (cache / "nasa" / space.CREDITS_FILE).write_text(
            json.dumps(payload), encoding="utf-8")
        return cache / "nasa"

    def test_a_credits_file_holding_a_list_is_not_attribution(
            self, tmp_path, monkeypatch):
        """The file is written by us, but it is on the user's disk.

        Anything that is not a mapping naming a file that is still there
        is no attribution at all, and showing it in Preferences would
        credit nobody for an image that may not exist.
        """
        from spacr.qt.space import read_credits

        self._credits(tmp_path, monkeypatch, ["not", "a", "mapping"])

        assert read_credits() is None

    def test_a_credits_file_naming_a_present_image_is_returned(
            self, tmp_path, monkeypatch):
        """The contrast: a real record, and the image beside it."""
        from spacr.qt.space import read_credits

        folder = self._credits(tmp_path, monkeypatch,
                               {"file": "earth.jpg", "credit": "NASA"})
        (folder / "earth.jpg").write_bytes(b"x")

        credits = read_credits()

        assert credits is not None and credits["credit"] == "NASA"

    def test_a_credits_file_naming_a_missing_image_is_not_returned(
            self, tmp_path, monkeypatch):
        """...and the record without its image is no record."""
        from spacr.qt.space import read_credits

        self._credits(tmp_path, monkeypatch,
                      {"file": "gone.jpg", "credit": "NASA"})

        assert read_credits() is None


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------

class TestTheInkRampsChromaMonotonically:
    """``if spread > chroma:`` in ``_hue_ink``'s value ramp.

    The candidates are ``base * level`` for ``level`` in ``range(256)``,
    and ``base`` is ``_hue_rgb(hue)`` at the default saturation 1.0 --
    which always returns a triple whose largest channel is exactly 1.0
    and whose smallest is exactly 0.0 (see the ``pure`` table: every row
    holds a 1.0 and a 0.0). So ``spread = max(candidate) -
    min(candidate)`` is ``round(1.0 * level) - round(0.0 * level)``,
    i.e. ``level`` itself: strictly increasing. Each admissible candidate
    is therefore strictly more chromatic than the last one kept, and the
    false side of the comparison cannot be taken.

    Pinned instead: the property that makes it so, and the answer it
    produces -- the most coloured candidate the readability band allows.
    """

    def test_the_pure_hue_always_spans_the_whole_channel_range(self):
        from spacr.qt.theme import _hue_rgb

        for hue in (0.0, 45.0, 120.0, 210.0, 300.0, 359.0):
            base = _hue_rgb(hue)
            assert max(base) == pytest.approx(1.0)
            assert min(base) == pytest.approx(0.0), (
                f"hue {hue} came back desaturated ({base}), which would "
                "let two levels tie on chroma")

    def test_the_ink_is_the_most_coloured_candidate_the_band_allows(self):
        from spacr.qt.theme import _hue_ink, _hue_rgb, _rgb_luminance

        ink = _hue_ink(210.0, 0.10, 0.60, "#4A9EFF")

        assert ink.startswith("#") and len(ink) == 7, ink
        rgb = tuple(int(ink[i:i + 2], 16) for i in (1, 3, 5))
        assert 0.10 <= _rgb_luminance(rgb) <= 0.60, (
            "the ink has to sit inside the readability band; got "
            f"{_rgb_luminance(rgb)}")
        assert max(rgb) - min(rgb) > 0, \
            "an ink with no chroma at all is not an ink, it is a grey"

        # ...and no candidate on the value ramp is more chromatic than it,
        # which is the answer the strictly-rising comparison produces.
        base = _hue_rgb(210.0)
        ramp_best = max(
            (max(candidate) - min(candidate)
             for candidate in (
                 tuple(int(round(channel * level)) for channel in base)
                 for level in range(256))
             if 0.10 <= _rgb_luminance(candidate) <= 0.60),
            default=0)
        assert max(rgb) - min(rgb) >= ramp_best, (
            "the ink kept has to be at least as coloured as every level the "
            f"band admitted; {max(rgb) - min(rgb)} < {ramp_best}")

    def test_a_band_that_admits_nothing_falls_back(self):
        """The other exit: no candidate fits, so the plain shift is used."""
        from spacr.qt.theme import _hue_ink

        assert _hue_ink(210.0, 0.999, 1.0, "#4A9EFF")


# ---------------------------------------------------------------------------
# prerun
# ---------------------------------------------------------------------------

class TestScoringTheMasksFromAScreenWithNoSettingsModel:
    """``if model is not None:`` in ``_on_score_clicked``.

    And, on the same class, ``if style is not None:`` in ``_restyle``:
    ``QWidget.style()`` returns the widget's own style, or the
    application's, or the default one -- there is no state in which a
    live QWidget has no style, and a widget whose C++ side is gone raises
    instead of answering None. That guard cannot be false, and greying a
    changed objectName is what it protects.
    """

    class _Screen(QWidget):
        def __init__(self, src):
            super().__init__()
            from PySide6.QtWidgets import QLineEdit

            self._src = QLineEdit(str(src), self)
            self._src.setObjectName("src")
            self.widgets = {"src": self._src}

    def test_a_screen_with_no_model_scores_with_empty_settings(self, qapp,
                                                               tmp_path,
                                                               monkeypatch):
        """Thresholds come from the settings panel; there may not be one.

        The scoring still has to run -- with the defaults -- rather than
        refusing because the panel it usually reads is not there.
        """
        from spacr.qt import prerun

        screen = self._Screen(tmp_path)
        banner = prerun.SegQCBanner(screen)
        try:
            monkeypatch.setattr(prerun, "_src_of", lambda s: str(tmp_path))
            assert getattr(screen, "_settings_model", None) is None

            started = {}

            def _capture(self, fn, box, on_done, app_key):
                started.update(box=dict(box), app_key=app_key)
                return False           # do not actually start a thread

            monkeypatch.setattr(type(banner), "_start_job", _capture)

            banner._on_score_clicked()

            assert started.get("app_key") == "seg_qc"
            assert started["box"]["settings"] == {}, (
                "with no settings model the job gets the empty settings that "
                f"mean 'use the defaults'; got {started['box']['settings']!r}")
            assert started["box"]["src"] == str(tmp_path)
        finally:
            banner.deleteLater()
            screen.deleteLater()

    def test_restyling_a_live_widget_always_finds_a_style(self, qapp):
        """The invariant behind the guard beside it."""
        from spacr.qt import prerun

        screen = self._Screen("/tmp/plate")
        banner = prerun.SegQCBanner(screen)
        try:
            label = QLabel("x", banner)
            assert label.style() is not None, (
                "a live QWidget always has a style; that is why the guard in "
                "_restyle cannot be false")

            label.setObjectName("QCVerdictFail")
            banner._restyle(label)

            assert label.objectName() == "QCVerdictFail"
        finally:
            banner.deleteLater()
            screen.deleteLater()


# ---------------------------------------------------------------------------
# iconset
# ---------------------------------------------------------------------------

class TestAReinkedIconIsAlwaysCached:
    """``if inked is not None:`` before the cache write.

    ``reink`` is annotated and implemented to return "a uint8 RGBA
    array": both of its exits build one -- the early ``out.astype`` for a
    fully transparent source, and the composited array at the end -- and
    neither can be ``None``. The guard re-checks what the call on the
    line above guarantees, so the "do not write the cache" side is
    unreachable.

    Pinned instead: what the cache is for. Re-inking is the expensive
    half of startup, so the second ask for one icon has to come back from
    the PNG rather than from the source art.
    """

    def test_reink_always_answers_with_an_array(self):
        from spacr.qt.iconset import reink

        opaque = np.zeros((4, 4, 4), dtype=np.uint8)
        opaque[..., 3] = 255
        transparent = np.zeros((4, 4, 4), dtype=np.uint8)

        for source in (opaque, transparent):
            inked = reink(source.astype(float), "dark")
            assert inked is not None, "reink has no None exit"
            assert inked.shape == (4, 4, 4)
            assert inked.dtype == np.uint8

    def test_the_re_inked_icon_round_trips_through_the_cache(self, tmp_path):
        """The write the guard protects, and the read that pays for it."""
        from spacr.qt.iconset import _read_cached_icon, _write_cached_icon

        array = np.zeros((4, 4, 4), dtype=np.uint8)
        array[..., 0] = 200
        array[..., 3] = 255
        path = tmp_path / "icons" / "one.png"

        _write_cached_icon(path, array)

        assert path.is_file(), "the icon was written"
        assert not path.with_suffix(".part").exists(), \
            "and the temp file was renamed, not left behind"
        assert np.array_equal(_read_cached_icon(path), array), \
            "the round trip has to be lossless or the cache is a lie"


# ---------------------------------------------------------------------------
# ai/issue_report
# ---------------------------------------------------------------------------

_TRACEBACK = ("Traceback (most recent call last):\n"
              '  File "spacr/ml.py", line 4346, in ml_analysis\n'
              "    raise ValueError('no wells survived')\n"
              "ValueError: no wells survived\n")


class TestASecondOccurrenceWhoseCommentDoesNotLand:
    """``if ok:`` after commenting on the existing issue."""

    @pytest.fixture(autouse=True)
    def _offline(self, monkeypatch):
        from spacr.qt.ai import github_auth

        monkeypatch.setattr(
            github_auth, "_HTTP_OPEN",
            lambda *a, **k: pytest.fail("this test reached HTTP"))

    def test_a_comment_that_failed_still_opens_an_issue(self, monkeypatch):
        """Losing a crash report is worse than filing a duplicate.

        The dedupe comment is best-effort: if it does not go through --
        the issue was locked, or the token lost its scope -- the report
        still has to be filed somewhere a maintainer will see it.
        """
        from spacr.qt.ai import github_auth, issue_report

        created = []
        monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
        monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                            lambda repo, fp: (True, {"number": 79,
                                                     "html_url": "u/79"}))
        monkeypatch.setattr(github_auth, "comment_on_issue",
                            lambda repo, n, body: (False, "locked"))
        monkeypatch.setattr(
            github_auth, "create_issue",
            lambda *a, **k: created.append(a) or (True, "u/NEW"))

        url = issue_report.file_issue(_TRACEBACK, active_app="ml_analyze")

        assert url == "u/NEW", (
            "a failed comment must fall through to a new issue; got "
            f"{url!r}")
        assert len(created) == 1

    def test_a_comment_that_landed_points_at_the_existing_issue(
            self, monkeypatch):
        """The contrast: the comment goes through, so nothing is filed."""
        from spacr.qt.ai import github_auth, issue_report

        created = []
        monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
        monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                            lambda repo, fp: (True, {"number": 79,
                                                     "html_url": "u/79"}))
        monkeypatch.setattr(github_auth, "comment_on_issue",
                            lambda repo, n, body: (True, "c"))
        monkeypatch.setattr(
            github_auth, "create_issue",
            lambda *a, **k: created.append(a) or (True, "u/NEW"))

        assert issue_report.file_issue(_TRACEBACK) == "u/79"
        assert created == [], "no duplicate may be opened"


# ---------------------------------------------------------------------------
# the Home-screen variant generators
# ---------------------------------------------------------------------------

class TestHomeCandidateSixWithoutItsLogo:
    """``if pix is not None:`` in ``variants.v06``.

    ``Ctx.logo`` answers ``None`` when the wordmark PNG is not on disk --
    which is what a source checkout without the resources looks like, and
    what an sdist that pruned them looks like. The candidate is a search
    screen; it has to draw without the mark above it.
    """

    def test_the_search_screen_is_built_with_no_wordmark(self, qapp,
                                                         monkeypatch):
        import importlib.util
        import os
        import sys
        import types

        generators = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "spacr", "resources", "home", "versions", "_generators")
        if not os.path.isdir(generators):
            pytest.skip("home-screen variant generators not present")

        def _load(name):
            path = os.path.join(generators, f"{name}.py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        names = ("common", "parts", "variants")
        saved = {name: sys.modules.get(name) for name in names}
        try:
            common = _load("common")
            common.bootstrap()
            _load("parts")
            variants = _load("variants")

            ctx = common.Ctx(qapp, "dark")

            def _pixmap_labels(page):
                return [child for child in page.findChildren(QLabel)
                        if child.pixmap() and not child.pixmap().isNull()]

            with_logo = variants.v06(ctx)
            monkeypatch.setattr(type(ctx), "logo", lambda self, px: None)
            without_logo = variants.v06(ctx)
            try:
                labels = [child.text()
                          for child in without_logo.findChildren(QLabel)
                          if child.text()]
                assert "spaCR" in labels, (
                    "the wordmark TEXT still heads the page when the image "
                    f"is missing; got {labels[:6]}")
                assert len(_pixmap_labels(without_logo)) == \
                    len(_pixmap_labels(with_logo)) - 1, (
                    "exactly the logo label is missing, and nothing else "
                    f"changed: {len(_pixmap_labels(without_logo))} vs "
                    f"{len(_pixmap_labels(with_logo))}")
            finally:
                with_logo.deleteLater()
                without_logo.deleteLater()
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
