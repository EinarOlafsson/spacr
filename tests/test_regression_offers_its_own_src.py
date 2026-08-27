"""Regression writes where it is told, and creates at most one folder.

The engine has honoured a caller's `src` for a while; the PANEL never
offered one, so the override could not be reached from the application at
all -- and the validator errored about a missing `src` on runs that then
completed normally, telling the user to point it at a folder of images,
which is not what regression reads.

The creation rule, as asked: "if scr dosn exist then make the last folder
if the folder holding that folder dosnt exist fall back to auto src".
"""

from __future__ import annotations

import os

import pytest

from spacr.ml import resolve_regression_src


@pytest.fixture
def automatic(tmp_path):
    folder = tmp_path / "beside_the_counts"
    folder.mkdir()
    return str(folder)


def test_empty_means_automatic(automatic):
    """Every run that exists today passes nothing and must not move."""
    for value in (None, "", "   "):
        path, how = resolve_regression_src(value, automatic)
        assert path == automatic
        assert how == "automatic"


def test_a_folder_that_exists_is_used(tmp_path, automatic):
    wanted = tmp_path / "somewhere_else"
    wanted.mkdir()
    path, how = resolve_regression_src(str(wanted), automatic)
    assert path == str(wanted)
    assert "as given" in how


def test_one_missing_level_is_created(tmp_path, automatic):
    wanted = tmp_path / "leaf"
    assert not wanted.exists()
    path, how = resolve_regression_src(str(wanted), automatic)
    assert path == str(wanted)
    assert wanted.is_dir(), "the leaf was not created"
    assert "created" in how


def test_two_missing_levels_build_nothing(tmp_path, automatic):
    """One missing level is a folder to make; two is a typo, and building a
    tree would turn a typo into a directory the run succeeds into."""
    wanted = tmp_path / "missing" / "deep"
    path, how = resolve_regression_src(str(wanted), automatic)
    assert path == automatic, "it wrote where the user asked, having not made it"
    assert not (tmp_path / "missing").exists(), "a tree was built"
    assert "does not exist" in how and automatic in how


def test_the_fallback_says_why(tmp_path, automatic):
    """Silently writing somewhere other than where the user asked is the
    failure this rule exists to avoid."""
    _path, how = resolve_regression_src(str(tmp_path / "a" / "b"), automatic)
    assert how != "automatic"
    assert "typo" in how


def test_a_file_where_a_folder_was_asked_for_falls_back(tmp_path, automatic):
    target = tmp_path / "not_a_folder.txt"
    target.write_text("x")
    path, how = resolve_regression_src(str(target), automatic)
    assert path == automatic
    assert "not a folder" in how


def test_a_tilde_resolves_before_the_decision(automatic):
    path, _how = resolve_regression_src("~", automatic)
    assert path == os.path.abspath(os.path.expanduser("~"))
    assert "~" not in path


def test_dot_dot_resolves_before_the_decision(tmp_path, automatic):
    inner = tmp_path / "one" / "two"
    inner.mkdir(parents=True)
    path, _how = resolve_regression_src(str(inner / ".." / ".."), automatic)
    assert path == str(tmp_path)
    assert ".." not in path


# --- the validator ---------------------------------------------------------


def test_regression_no_longer_errors_about_a_missing_src():
    from spacr.validate import _check_src

    assert _check_src({"count_data": ["/tmp/a.csv"]}, "regression", ()) == []
    assert _check_src({"count_data": ["/tmp/a.csv"], "src": ""},
                      "regression", ()) == []


def test_a_module_that_needs_one_still_errors():
    """The behaviour preprocess_generate_masks depends on."""
    from spacr.validate import _check_src

    problems = _check_src({}, "mask", ())
    assert problems and "missing" in problems[0].message


def test_the_exemption_is_declared_not_guessed():
    from spacr.validate import DERIVES_ITS_OWN_SRC

    assert "regression" in DERIVES_ITS_OWN_SRC
    assert "mask" not in DERIVES_ITS_OWN_SRC


# --- the setting -----------------------------------------------------------


def test_the_default_is_empty():
    from spacr.settings import get_perform_regression_default_settings

    assert get_perform_regression_default_settings({})["src"] == ""


def test_a_supplied_value_survives_the_defaults():
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({"src": "/tmp/mine"})
    assert settings["src"] == "/tmp/mine"


def test_the_panel_offers_it(qtbot):
    """The whole point: the engine's override was unreachable from the app."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    widgets = screen._settings_model._widgets
    assert "src" in widgets, "the override still cannot be reached"


def test_the_panel_control_is_the_same_kind_every_other_module_uses(qtbot):
    """Not a bespoke widget for one module: `src` on Mask and Measure is the
    same control, and a path arrives by drag-and-drop the same way."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    kinds = {}
    for key in ("regression", "mask"):
        screen = AppScreen(key)
        qtbot.addWidget(screen)
        kinds[key] = type(screen._settings_model._widgets["src"]).__name__
    assert kinds["regression"] == kinds["mask"], kinds


def test_the_panel_opens_with_it_empty(qtbot):
    """Empty means automatic, so a user who touches nothing runs exactly
    where they ran before."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    control = screen._settings_model._widgets["src"]
    assert control.text() == ""
    assert not (screen._settings_model.collect().get("src") or "")
