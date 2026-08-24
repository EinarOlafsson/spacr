"""``set_dark_style`` → ``apply_theme``: the rename, finished.

The Tk GUI called its styling helper from ~40 places across ``gui.py``,
``gui_core.py``, ``gui_utils.py`` and ``gui_elements.py`` itself, and those
modules sat in single digits of test coverage. The rename was therefore
additive: ``apply_theme`` was the name, and ``set_dark_style`` kept working
and kept returning exactly what its callers destructured.

THE ALIAS AND ITS TESTS ARE GONE. ``gui_elements`` is deleted, so there is no
``set_dark_style`` left to forward, no signature to match against
``apply_theme``, and no Tk palette dict to destructure — the five tests that
asserted those things had no subject to point at. What replaced them is
narrower and true of the code that ships: the old name appears nowhere, and
the palette that survived the rename is ``spacr.qt.theme.palette_for``, which
answers every theme with the same keys.

The two tests about naming — no "dark/light mode" language now that there are
four themes, and ``color_blind_mode`` being a different thing — are unchanged.
"""
from __future__ import annotations

import pathlib

import spacr

_PACKAGE_ROOT = pathlib.Path(spacr.__file__).parent


class TestRenameIsFinished:
    def test_the_old_name_appears_nowhere(self):
        """A forwarding alias only helps while something forwards to it."""
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "set_dark_style" in text:
                offenders.append(str(path.relative_to(_PACKAGE_ROOT)))
        assert not offenders, (
            f"set_dark_style is back in {offenders}; the helper it named was "
            "deleted with the Tk interface")

    def test_the_scan_would_notice(self):
        """The scan above passes trivially if it reads no files."""
        scanned = list(_PACKAGE_ROOT.rglob("*.py"))
        assert len(scanned) > 100, f"only scanned {len(scanned)} files"


class TestThePaletteThatSurvived:
    """`palette_for` is what callers destructure now, and every theme
    answers with the same keys — a theme missing one renders a widget with
    an empty colour string, which Qt draws black on black."""

    EXPECTED_KEYS = {
        "bg", "fg", "fg_dim", "fg_muted", "accent", "border",
        "surface", "surface_alt", "error", "warning", "success", "info",
    }

    def test_every_theme_answers_with_the_same_keys(self):
        from spacr.qt.theme import THEMES, palette_for

        assert len(THEMES) >= 3, "the rename was made because there are many"
        reference = set(palette_for(THEMES[0]))
        assert self.EXPECTED_KEYS.issubset(reference)
        for theme in THEMES[1:]:
            assert set(palette_for(theme)) == reference, (
                f"theme {theme!r} is missing "
                f"{sorted(reference - set(palette_for(theme)))}")

    def test_the_themes_are_not_all_the_same_palette(self):
        """One palette repeated under four names would satisfy the test above."""
        from spacr.qt.theme import palette_for

        assert palette_for("dark")["bg"] != palette_for("light")["bg"]

    def test_every_colour_is_a_colour(self):
        from spacr.qt.theme import THEMES, palette_for

        for theme in THEMES:
            for key, value in palette_for(theme).items():
                assert isinstance(value, str) and value.strip(), (
                    f"{theme}.{key} is {value!r}, which Qt reads as no colour")


class TestNaming:
    def test_no_user_facing_mode_language_left(self):
        """"Dark mode" stopped being accurate at four themes."""
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for phrase in ("dark mode", "light mode"):
                if phrase in text:
                    offenders.append(f"{path.name}: {phrase!r}")
        assert not offenders, offenders

    def test_colour_blind_mode_is_left_alone(self):
        """`color_blind_mode` is a different thing; "mode" is correct."""
        from spacr.qt.preferences import VALID_CB_MODES, get_color_blind_mode
        assert "off" in VALID_CB_MODES
        assert callable(get_color_blind_mode)
