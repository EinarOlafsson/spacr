"""A Classify run in annotation mode always knows its annotation column.

Issue #185: a Classify (ML) run started from the GUI raised
``dataset_mode='annotation' needs annotation_column set`` inside
``generate_ml_scores``. The Classify screen hides ``annotation_column`` -- the
Classes editor names the column of each class -- and its default is None, but
the ML family read only ``annotation_column``. So a run whose classes named
the column still reached the backend without one, and a run with no column
anywhere was started instead of being refused at the screen.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import classify as classify_module
from spacr.settings import set_default_classify


def _merged_settings(**changes):
    settings = set_default_classify(settings={})
    settings.update(classifier_family="ml", dataset_mode="annotation",
                    src=["/tmp/spacr-issue-185-plate"])
    settings.update(changes)
    return settings


def test_the_screen_default_has_no_annotation_column():
    """The precondition: the hidden setting is empty on the Classify screen."""
    assert not _merged_settings().get("annotation_column")


def test_the_ml_family_trains_on_the_column_the_classes_name(monkeypatch):
    seen = {}
    monkeypatch.setattr("spacr.ml.generate_ml_scores",
                        lambda settings: seen.update(settings) or "ran")
    settings = _merged_settings(classes={
        "nc": {"column": "test", "value": 1},
        "pc": {"column": "test", "value": 2},
    })

    assert classify_module.classify(settings) == "ran"
    assert seen["annotation_column"] == "test"


def test_a_run_with_no_column_anywhere_is_reported_before_it_starts():
    from spacr.validate import ERROR, validate_settings

    settings = _merged_settings(classes={})
    assert classify_module._annotation_labels_missing(settings,
                                                      "classify_merged")
    problems = [p for p in validate_settings(settings, "classify_merged")
                if p.severity == ERROR and p.setting == "annotation_column"]
    assert problems
    assert problems[0].message == classify_module._ANNOTATION_COLUMN_MISSING


def test_metadata_mode_and_a_named_column_are_not_refused():
    missing = classify_module._annotation_labels_missing
    assert not missing(_merged_settings(dataset_mode="metadata", classes={}),
                       "classify_merged")
    assert not missing(_merged_settings(annotation_column="test",
                                        classes={}), "classify_merged")


def test_the_backend_still_refuses_a_scripted_call_without_a_column(
        monkeypatch, tmp_path):
    import pandas as pd

    from spacr import ml

    monkeypatch.setattr("spacr.io._read_and_merge_data",
                        lambda *a, **k: (pd.DataFrame({"prcfo": ["a"]}), None))
    with pytest.raises(ValueError, match="needs annotation_column"):
        ml.generate_ml_scores(_merged_settings(classes={},
                                               src=[str(tmp_path)]))


def test_the_classify_screen_refuses_the_run_and_says_why(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.screens import app_screen

    screen = app_screen.AppScreen("classify_merged")
    qtbot.addWidget(screen)
    monkeypatch.setattr(screen, "_wait_for_the_entry_import",
                        lambda override: False)
    monkeypatch.setattr(app_screen, "resolve_pipeline_entry",
                        lambda key: classify_module.classify)
    monkeypatch.setattr(screen, "_claim_the_project", lambda settings: True)
    monkeypatch.setattr(screen, "_confirm_ram_guard", lambda settings: True)
    monkeypatch.setattr(app_screen, "make_thread", lambda *args: pytest.fail(
        "a run with no annotation column started a worker"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warnings.append(args))

    screen._on_run(override=_merged_settings(classes={}))

    assert len(warnings) == 1
    assert "annotation column" in str(warnings[0][2])


def test_the_screen_message_is_the_one_validate_reports():
    """One wording, so the console line and the dialog say the same thing."""
    import ast
    import inspect
    import textwrap

    from spacr.qt.screens.app_screen import AppScreen

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(AppScreen._annotation_labels_named)))
    literals = [node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "tr" and node.args
                and isinstance(node.args[0], ast.Constant)]
    assert classify_module._ANNOTATION_COLUMN_MISSING in literals
