"""A re-fit predicts where it would write, or admits that it cannot (``spacr.refit``).

``destination`` asks the regression folder rule itself and answers ``None`` when
there is no count table, the counts are remote without a local ``src``, or the
rule cannot reach the disk. ``settings_of_run`` reads the settings file beside a
finished run and passes over one that does not parse.
"""
from __future__ import annotations

import os

from spacr import refit


def test_destination_is_the_folder_rule_answer_or_none(subtests, monkeypatch,
                                                        tmp_path):
    """Missing, remote and unreachable sources give None; others the rule's path."""
    import spacr.ml as ml

    counts = str(tmp_path / "counts.csv")
    dsn = os.environ.get("SPACR_TEST_POSTGRES_DSN", "postgresql:///remote_counts")
    real = [
        ("no count_data key", {}, None),
        ("empty count_data string", {"count_data": ""}, None),
        ("empty count_data list", {"count_data": []}, None),
        ("remote counts without src", {"count_data": [dsn],
                                       "regression_type": "ols"}, None),
        ("remote counts with placeholder src", {
            "count_data": [dsn], "regression_type": "ols", "src": "path"}, None),
        ("remote counts with remote src", {
            "count_data": [dsn], "regression_type": "ols",
            "src": "postgresql:///not_a_folder"}, None),
        ("remote counts with a local src", {
            "count_data": [dsn], "regression_type": "ols",
            "src": str(tmp_path)}, str(tmp_path / "results" / "ols")),
    ]
    for case_id, settings, expected in real:
        with subtests.test(msg=case_id):
            assert refit.destination(settings) == expected

    monkeypatch.setattr(ml, "_next_results_folder",
                        lambda root, kind: os.path.join(root, f"{kind}_1"),
                        raising=False)
    ruled = [
        ("local counts list and a named type",
         {"count_data": [counts], "regression_type": "ols"},
         str(tmp_path / "results" / "ols_1")),
        ("no type means auto", {"count_data": counts},
         str(tmp_path / "results" / "auto_1")),
        ("guide permutation names its own kind",
         {"count_data": counts, "regression_type": "ols",
          "analysis_mode": "guide_permutation"},
         str(tmp_path / "results" / "guide_permutation_1")),
    ]
    for case_id, settings, expected in ruled:
        with subtests.test(msg=case_id):
            assert refit.destination(settings) == expected

    def _explode(root, kind):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(ml, "_next_results_folder", _explode, raising=False)
    with subtests.test(msg="unreachable results folder"):
        assert refit.destination({"count_data": counts}) is None


def test_settings_of_a_run_are_read_back_or_passed_over(subtests, monkeypatch,
                                                        tmp_path):
    """A parseable settings file is returned; none or a broken one gives None."""
    import spacr.utils as utils

    run = tmp_path / "regression_1"
    run.mkdir()
    settings_file = run / refit.SETTINGS_NAMES[0]
    settings_file.write_text("key,value\n", encoding="utf-8")
    table = run / "results.csv"
    table.write_text("feature\n", encoding="utf-8")
    nested = tmp_path / "other" / "deep" / "run"
    nested.mkdir(parents=True)
    (tmp_path / "other" / "settings").mkdir()
    (tmp_path / "other" / "settings" / "settings.csv").write_text(
        "key,value\n", encoding="utf-8")

    monkeypatch.setattr(utils, "load_settings",
                        lambda path: {"read": os.path.relpath(path, tmp_path)},
                        raising=False)
    cases = [
        ("no results path", None, None),
        ("the run folder", str(run),
         {"read": os.path.join("regression_1", refit.SETTINGS_NAMES[0])}),
        ("the results table inside the run", str(table),
         {"read": os.path.join("regression_1", refit.SETTINGS_NAMES[0])}),
        ("settings/ two levels up", str(nested),
         {"read": os.path.join("other", "settings", "settings.csv")}),
        ("a folder with no settings anywhere near", str(tmp_path / "nothing"),
         None),
    ]
    for case_id, path, expected in cases:
        with subtests.test(msg=case_id):
            assert refit.settings_of_run(path) == expected

    def _explode(path):
        raise ValueError("malformed settings")

    monkeypatch.setattr(utils, "load_settings", _explode, raising=False)
    with subtests.test(msg="an unparseable settings file is passed over"):
        assert refit.settings_of_run(str(run)) is None
