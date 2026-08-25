"""A re-fit predicts where it would write, or admits that it cannot.

The sentence "this will not overwrite what you are looking at" is only true if
the destination is asked of the folder rule itself. When there is no count
table to derive a source from, or the folder rule cannot reach the disk, the
honest answer is no prediction at all -- a guessed path in that sentence is
worse than no sentence.

Seeding a re-fit from a finished run has the same shape: the settings file
beside the results is read if it parses, and a file that does not parse is
passed over so the search continues instead of the dialog failing to open.
"""
from __future__ import annotations

import os

import pytest

from spacr import refit


def test_no_count_table_means_no_predicted_destination():
    """Without count data there is no source folder to predict from."""
    assert refit.destination({}) is None
    assert refit.destination({"count_data": ""}) is None


def test_an_empty_count_table_list_means_no_predicted_destination():
    """An empty list is the same absence as no key at all."""
    assert refit.destination({"count_data": []}) is None


def test_an_unreachable_results_folder_means_no_prediction(monkeypatch,
                                                           tmp_path):
    """A folder rule that cannot reach the disk yields no prediction.

    The caller prints the prediction in a sentence about not overwriting the
    current run; raising here would take the whole dialog down instead.
    """
    import spacr.ml as ml

    def _explode(root, kind):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(ml, "_next_results_folder", _explode, raising=False)

    assert refit.destination(
        {"count_data": str(tmp_path / "counts.csv")}) is None


def test_a_reachable_results_folder_is_predicted(monkeypatch, tmp_path):
    """The prediction comes from the folder rule, not from a local guess."""
    import spacr.ml as ml

    monkeypatch.setattr(ml, "_next_results_folder",
                        lambda root, kind: os.path.join(root, f"{kind}_1"),
                        raising=False)

    predicted = refit.destination({"count_data": [str(tmp_path / "counts.csv")],
                                   "regression_type": "ols"})

    assert predicted == str(tmp_path / "results" / "ols_1")


def test_no_results_path_has_no_settings_to_read():
    """A run that was never located has no settings file beside it."""
    assert refit.settings_of_run(None) is None


def test_an_unparseable_settings_file_is_passed_over(monkeypatch, tmp_path):
    """A settings file that will not parse does not stop the search.

    The dialog opens with defaults instead of failing; the alternative is a
    re-fit that cannot be started at all because one stale CSV is malformed.
    """
    import spacr.utils as utils

    run = tmp_path / "regression_1"
    run.mkdir()
    (run / refit.SETTINGS_NAMES[0]).write_text("not,a,settings,table\n")

    def _explode(path):
        raise ValueError("malformed settings")

    monkeypatch.setattr(utils, "load_settings", _explode, raising=False)

    assert refit.settings_of_run(str(run)) is None


def test_a_parseable_settings_file_is_returned(monkeypatch, tmp_path):
    """The same search does return the settings when the file parses."""
    import spacr.utils as utils

    run = tmp_path / "regression_1"
    run.mkdir()
    (run / refit.SETTINGS_NAMES[0]).write_text("key,value\n")

    monkeypatch.setattr(utils, "load_settings",
                        lambda path: {"regression_type": "ols"},
                        raising=False)

    assert refit.settings_of_run(str(run)) == {"regression_type": "ols"}
