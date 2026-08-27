"""Regression output-directory resolution and settings-panel coverage."""

from __future__ import annotations

import os

import pytest

from spacr.ml import resolve_regression_src


@pytest.fixture
def automatic(tmp_path):
    """Return an existing directory representing the automatic output root."""
    folder = tmp_path / "beside_the_counts"
    folder.mkdir()
    return str(folder)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_output_directory_uses_the_automatic_location(value, automatic):
    path, message = resolve_regression_src(value, automatic)

    assert path == automatic
    assert message == "automatic"


def test_existing_output_directory_is_used(tmp_path, automatic):
    requested = tmp_path / "regression_outputs"
    requested.mkdir()

    path, message = resolve_regression_src(str(requested), automatic)

    assert path == str(requested)
    assert message == f"Regression output directory: {requested}."


def test_one_missing_directory_component_is_created(tmp_path, automatic):
    requested = tmp_path / "new_output_root"

    path, message = resolve_regression_src(str(requested), automatic)

    assert path == str(requested)
    assert requested.is_dir()
    assert "Created regression output directory" in message


def test_missing_parent_uses_automatic_location_without_creating_a_tree(
        tmp_path, automatic):
    requested = tmp_path / "missing_parent" / "output_root"

    path, message = resolve_regression_src(str(requested), automatic)

    assert path == automatic
    assert not requested.parent.exists()
    assert str(requested.parent) in message
    assert automatic in message


def test_file_path_uses_automatic_location(tmp_path, automatic):
    requested = tmp_path / "not_a_directory.txt"
    requested.write_text("data", encoding="utf-8")

    path, message = resolve_regression_src(str(requested), automatic)

    assert path == automatic
    assert "is not a directory" in message
    assert automatic in message


def test_directory_creation_error_is_reported(
        tmp_path, automatic, monkeypatch):
    from spacr import ml

    requested = tmp_path / "new_output_root"

    def unavailable(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ml.os, "mkdir", unavailable)
    path, message = resolve_regression_src(str(requested), automatic)

    assert path == automatic
    assert "Permission denied" in message
    assert automatic in message


def test_user_home_is_resolved_before_validation(automatic):
    path, _message = resolve_regression_src("~", automatic)

    assert path == os.path.abspath(os.path.expanduser("~"))
    assert "~" not in path


def test_relative_parent_segments_are_resolved_before_validation(
        tmp_path, automatic):
    inner = tmp_path / "one" / "two"
    inner.mkdir(parents=True)

    path, _message = resolve_regression_src(
        str(inner / ".." / ".."), automatic)

    assert path == str(tmp_path)
    assert ".." not in path


def test_regression_accepts_a_blank_src_during_preflight():
    from spacr.validate import _check_src

    assert _check_src({"count_data": ["/tmp/a.csv"]}, "regression", ()) == []
    assert _check_src(
        {"count_data": ["/tmp/a.csv"], "src": ""}, "regression", ()) == []


def test_regression_preflight_accepts_an_existing_output_directory(tmp_path):
    from spacr.validate import _check_src

    assert _check_src({"src": str(tmp_path)}, "regression", ()) == []


def test_regression_preflight_accepts_one_missing_directory_component(
        tmp_path):
    from spacr.validate import _check_src

    requested = tmp_path / "new_output_root"

    assert _check_src({"src": str(requested)}, "regression", ()) == []
    assert not requested.exists(), "preflight must not create output"


def test_regression_preflight_warns_when_the_parent_is_missing(tmp_path):
    from spacr.validate import WARNING, _check_src

    requested = tmp_path / "missing_parent" / "output_root"
    problems = _check_src({"src": str(requested)}, "regression", ())

    assert len(problems) == 1
    assert problems[0].severity == WARNING
    assert str(requested.parent) in problems[0].message
    assert "beside the first count table" in problems[0].fix


def test_regression_preflight_warns_for_a_file_output_path(tmp_path):
    from spacr.validate import WARNING, _check_src

    requested = tmp_path / "file.txt"
    requested.write_text("data", encoding="utf-8")
    problems = _check_src({"src": str(requested)}, "regression", ())

    assert len(problems) == 1
    assert problems[0].severity == WARNING
    assert "not a directory" in problems[0].message


def test_modules_that_require_a_source_still_reject_a_missing_src():
    from spacr.validate import _check_src

    problems = _check_src({}, "mask", ())

    assert problems
    assert "missing" in problems[0].message


def test_regression_default_output_directory_is_blank():
    from spacr.settings import get_perform_regression_default_settings

    assert get_perform_regression_default_settings({})["src"] == ""


def test_supplied_output_directory_survives_default_resolution():
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({"src": "/tmp/custom"})

    assert settings["src"] == "/tmp/custom"


def test_regression_panel_exposes_an_app_specific_output_control(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    model = screen._settings_model

    assert "src" in model._widgets
    assert model._label_for("src") == "Output directory"
    assert model._widgets["src"].text() == ""
    tooltip = model.plain_tooltip_for("src")
    assert "Output root for regression results" in tooltip
    assert "first count table" in tooltip


def test_regression_and_mask_use_distinct_src_tooltips(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    regression = AppScreen("regression")
    mask = AppScreen("mask")
    qtbot.addWidget(regression)
    qtbot.addWidget(mask)

    regression_text = regression._settings_model.plain_tooltip_for("src")
    mask_text = mask._settings_model.plain_tooltip_for("src")
    assert regression_text != mask_text
    assert "Output root for regression results" in regression_text
    assert "raw images" in mask_text
