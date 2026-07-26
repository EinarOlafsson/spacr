"""``set_dark_style`` → ``apply_theme``: the rename, and its alias.

The Tk GUI calls the styling helper from ~40 places across
``gui.py``, ``gui_core.py``, ``gui_utils.py`` and ``gui_elements.py``
itself, and those modules sit in single digits of test coverage. The
rename is therefore additive: ``apply_theme`` is the name, and
``set_dark_style`` still works and still returns exactly what its
callers destructure.
"""
from __future__ import annotations

import inspect

import pytest


def _gui_elements():
    try:
        import spacr.gui_elements as ge
    except Exception as exc:
        if "DisplayConnection" in type(exc).__name__ or "Xauthority" in str(exc):
            pytest.skip(f"spacr.gui_elements needs a display: {exc}")
        raise
    return ge


class TestRename:
    def test_apply_theme_is_the_name(self):
        ge = _gui_elements()
        assert callable(ge.apply_theme)
        assert ge.apply_theme.__name__ == "apply_theme"

    def test_set_dark_style_still_exists_and_forwards(self):
        ge = _gui_elements()
        assert callable(ge.set_dark_style)
        assert "apply_theme" in ge.set_dark_style.__doc__

    def test_alias_forwards_to_whatever_apply_theme_is(self, monkeypatch):
        """Forwarding, not a rebinding — so patching one patches both."""
        ge = _gui_elements()
        seen = {}

        def fake(style, **kwargs):
            seen["style"] = style
            seen["kwargs"] = kwargs
            return {"bg_color": "#123456"}

        monkeypatch.setattr(ge, "apply_theme", fake)
        out = ge.set_dark_style("STYLE", font_size=9)
        assert out == {"bg_color": "#123456"}
        assert seen["style"] == "STYLE"
        assert seen["kwargs"] == {"font_size": 9}

    def test_signatures_match(self):
        ge = _gui_elements()
        sig = inspect.signature(ge.apply_theme)
        assert list(sig.parameters) == [
            "style", "parent_frame", "containers", "widgets",
            "font_family", "font_size", "bg_color", "fg_color",
            "active_color", "inactive_color",
        ]


class TestAliasBehaviour:
    """Both names must return the palette dict callers destructure."""

    EXPECTED_KEYS = {
        "font_loader", "font_family", "font_size", "font_sizes",
        "bg_color", "fg_color", "active_color", "inactive_color",
        "border_color", "muted_color", "success_color", "warning_color",
        "error_color", "spacing",
    }

    def test_set_dark_style_returns_what_its_callers_expect(self, tk_root):
        from tkinter import ttk
        ge = _gui_elements()
        out = ge.set_dark_style(ttk.Style())
        assert self.EXPECTED_KEYS.issubset(out)
        assert out["bg_color"] == "#000000"
        assert out["fg_color"] == "#ffffff"
        assert set(out["spacing"]) == {"xs", "sm", "md", "lg", "xl"}

    def test_both_names_return_the_same_thing(self, tk_root):
        from tkinter import ttk
        ge = _gui_elements()
        assert ge.set_dark_style(ttk.Style()) == ge.apply_theme(ttk.Style())

    def test_custom_colours_still_pass_through_the_alias(self, tk_root):
        from tkinter import ttk
        ge = _gui_elements()
        ge._cached_dark_style = None
        try:
            out = ge.set_dark_style(ttk.Style(), parent_frame=tk_root,
                                    bg_color="#101010", fg_color="#eeeeee")
            assert out["bg_color"] == "#101010"
            assert out["fg_color"] == "#eeeeee"
        finally:
            ge._cached_dark_style = None


class TestCallSitesUseTheNewName:
    """The modules this task owns should say `apply_theme`."""

    @pytest.mark.parametrize("module", ["gui", "gui_core", "app_make_masks"])
    def test_no_stale_call_sites(self, module):
        import pathlib
        import spacr
        path = pathlib.Path(spacr.__file__).parent / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        assert "set_dark_style" not in source, \
            f"{module}.py still calls set_dark_style"
        assert "apply_theme" in source

    def test_no_user_facing_mode_language_left(self):
        """"Dark mode" stopped being accurate at three themes."""
        import pathlib
        import spacr
        root = pathlib.Path(spacr.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for phrase in ("dark mode", "light mode"):
                if phrase in text:
                    # theme.py names the retired phrase on purpose, to
                    # explain why it is retired.
                    if path.name == "theme.py":
                        continue
                    offenders.append(f"{path.name}: {phrase!r}")
        assert not offenders, offenders

    def test_colour_blind_mode_is_left_alone(self):
        """`color_blind_mode` is a different thing; "mode" is correct."""
        from spacr.qt.preferences import VALID_CB_MODES, get_color_blind_mode
        assert "off" in VALID_CB_MODES
        assert callable(get_color_blind_mode)
