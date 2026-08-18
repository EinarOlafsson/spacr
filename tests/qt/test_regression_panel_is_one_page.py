"""Instruction 135, the panel half: the regression settings as one page.

The maintainer walked every category on 2026-08-17 and said what each one
needed. This file pins the four answers that live in
``spacr/qt/screens/settings_model.py``:

* THE SECTIONS ARE THE SEVEN THEY ASKED FOR, in order, and "Additional
  Settings" is empty. "Regression plot can be removed" and "Runtime and
  reliability should be removed and go to prefgerences/general" -- and a
  section removed from the layout without its keys being hidden does not
  disappear, it reappears under the ungrouped heading nobody chose.

* THE PERMUTATION TEST SAYS WHAT IT DOES. "Permutation test is good it just
  needs a text box at the top briefly explaining what it does." Written from
  ``spacr.guide_permutation`` rather than from the reputation of permutation
  tests, so the assertions below name the lines of the code it describes.

* THE CSV BUTTON READS THE INPUT CSVs. "for the filter column there is an SQL
  buton this should be a csv buton that can read the input csvs, the
  dependent variable should have a simmilr CSV version of this buton." The
  SQL button opened ``measurements.db``, which is not where these columns
  live, so it could only ever offer names the run would not find.

* WITH NO CSV LOADED THE BUTTON SAYS SO. An empty list of choices shown as
  though it were the answer is the failure ``spacr.columns`` exists to
  prevent, and it is the one a picker falls into by default.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import (            # noqa: E402
    _APP_CATEGORY_SPECS,
    _APP_COMBO_OPTIONS,
    _APP_ESSENTIAL_EXTRAS,
    _APP_HIDDEN_KEYS,
    _CsvColumnField,
    CSV_COLUMN_SOURCES,
    SECTION_EXPLAINERS,
    SettingsWidgets,
    category_tooltip,
    essential_keys,
    explainer_width,
    has_csv_column_picker,
    has_section_explainer,
    permutation_test_explainer,
    regression_model_explainer,
    resolve_default_settings,
    section_explainer,
)

#: Exactly what "HOW TO KNOW IT WORKED" asks for: "The regression panel has
#: exactly these sections, in this order".
EXPECTED_SECTIONS = [
    "Input Tables",
    "Controls & Filters",
    "Plate & Batch Correction",
    "Response",
    "Model & Inference",
    "Estimator Tuning",
    "Permutation Test",
]


@pytest.fixture()
def panel(qtbot):
    """A built regression settings model, widgets and all."""
    model = SettingsWidgets("regression")
    model.build_sections()
    return model


def _sections(model):
    return [name for name, _rows in model.build_sections()]


def _csv(path, header, *rows):
    """Write a CSV and return its path."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(row + "\n")
    return path


# ---------------------------------------------------------------------------
# 1. Seven sections, and nothing left over
# ---------------------------------------------------------------------------

def test_the_panel_has_the_seven_sections_the_maintainer_named(panel):
    assert _sections(panel) == EXPECTED_SECTIONS


def test_nothing_falls_into_the_ungrouped_bucket(panel):
    """"Other" is Qt's name for the absence of a heading.

    Deleting "Regression Plots" and "Runtime & Reliability" from the layout
    without hiding their keys would move fourteen settings here, which looks
    like a new section and is the opposite of what removing one means.
    """
    assert "Other" not in _sections(panel)


def test_the_deleted_sections_keep_their_values_rather_than_dropping_them():
    """Hidden is not absent (INVARIANTS 6).

    `parameter_sweep` does `settings.setdefault("regression_qc", False)` so a
    hundred-trial sweep does not pay ~5.8 s and ~19 figures per trial. Drop
    the key and there is nothing for it to set; hide it and the sweep is
    unchanged while the panel stops asking.
    """
    model = SettingsWidgets("regression")
    model.build_sections()
    collected = model.collect()
    defaults = resolve_default_settings("regression")
    for key in _APP_HIDDEN_KEYS["regression"]:
        assert key not in model._widgets, f"{key} still has a control"
        if key in defaults:
            # Hidden, and therefore at the value the MODULE declares rather
            # than at some widget's idea of it.
            assert collected[key] == defaults[key], key
    # `regression_qc` is the one that must survive as a key rather than being
    # retired with its section: `parameter_sweep.py` sets it False so a
    # hundred-trial sweep does not pay ~5.8 s and ~19 figures per trial.
    assert defaults["regression_qc"] is True
    assert collected["regression_qc"] is True


def test_the_hidden_set_names_runtime_keys_this_module_does_not_declare():
    """Naming an absent key costs nothing and is the point.

    `on_error`, `on_error_attempts`, `on_error_backoff` and `random_seed` are
    shared runtime defaults this module happens not to declare today. The day
    one of them arrives it must not appear on the panel the instruction just
    cleared, and remembering to add it then is exactly what does not happen.
    """
    hidden = _APP_HIDDEN_KEYS["regression"]
    declared = set(resolve_default_settings("regression"))
    assert {"on_error", "random_seed"} <= hidden
    assert not {"on_error", "random_seed"} & declared


def test_the_new_settings_are_placed_where_the_instruction_says():
    """Placement is asserted on the LAYOUT, not on the rendered panel.

    The keys are declared by `spacr.settings` in a slice this file does not
    own, and a layout token naming a key the module has not got yet is
    dropped in silence -- which is what makes landing the two halves in
    either order safe, and what would hide a misplacement until they met.
    """
    layout = dict(_APP_CATEGORY_SPECS["regression"])
    assert {"p_threshold_alpha", "p_threshold_kind"} <= set(
        layout["Model & Inference"])
    assert {"group_lasso_lambda", "rra_alpha", "rra_permutations"} <= set(
        layout["Estimator Tuning"])
    assert {"count_grna_column", "count_value_column"} <= set(
        layout["Input Tables"])


def test_the_raw_or_adjusted_choice_is_a_closed_alphabet():
    """Two readings and no third, so it cannot be a box to type "adj" into."""
    assert _APP_COMBO_OPTIONS["regression"]["p_threshold_kind"] == [
        "adjusted", "raw"]


def test_the_essentials_still_name_a_group_that_exists():
    """"@Controls & Plate Design" stopped existing when the sections merged.

    An unresolvable token is dropped rather than raised on, so the first-run
    disclosure silently lost its whole controls group instead of failing.
    """
    assert _APP_ESSENTIAL_EXTRAS["regression"][0] == "@Controls & Filters"
    # The MODULE's layout, not the shared category map: "@Controls & Filters"
    # is a Qt regroup and `spacr.settings.categories` has never heard of it.
    keys = essential_keys("regression")
    assert "positive_control" in keys and "negative_control" in keys
    assert "regression_type" in keys and "dependent_variable" in keys


def test_model_and_inference_help_describes_the_merged_section():
    """One dict, two "MODEL & INFERENCE" keys: Python keeps the last.

    The first was left behind when "Significance & Hit Calling" merged in, so
    it had never rendered -- the kind of dead entry a reader edits by mistake.
    """
    text = category_tooltip("regression", "Model & Inference")
    assert "what counts as a hit" in text
    assert "multiple-testing correction" in text


# ---------------------------------------------------------------------------
# 2. The two boxes, and which section opens with which
# ---------------------------------------------------------------------------

def test_the_permutation_box_describes_what_the_code_does():
    """Every clause is a line of `guide_freedman_lane_test`."""
    text = permutation_test_explainer()
    flat = " ".join(text.split())
    # Marginal, not conditional -- the module's docstring is explicit that it
    # "does not claim to estimate a simultaneous conditional coefficient".
    assert "marginal association" in flat
    assert "conditional coefficients" in flat
    # The permutation is Freedman-Lane and it is restricted to the block.
    assert "Freedman-Lane" in flat
    assert "WITHIN each block" in flat
    assert "plateID" in flat
    # p = (exceedances + 1) / (n_permutations + 1), and |perm| >= |observed|.
    assert "1/(permutations + 1)" in flat
    assert "two-sided" in flat
    # Eligibility and the per-threshold correction family.
    assert "guide_min_wells" in flat and "guide_presence_threshold" in flat
    assert "its own family" in flat


def test_the_permutation_box_is_shorter_than_the_model_box():
    """Shorter than 132's model box, and one paragraph, as 135 asks."""
    permutation = permutation_test_explainer()
    model = regression_model_explainer("mixed")
    assert len(permutation.split()) < len(model.split())
    # One paragraph: a heading line, then one block of prose.
    body = permutation.split("\n", 1)[1]
    assert body.strip()
    assert "\n\n" not in body.strip()


def test_neither_box_needs_a_horizontal_scrollbar():
    """The widget does not soft-wrap, so the text arrives pre-wrapped."""
    for line in permutation_test_explainer().splitlines():
        assert len(line) <= explainer_width(), repr(line)


def test_each_section_gets_the_box_that_belongs_to_it():
    assert SECTION_EXPLAINERS["regression"] == (
        "Model & Inference", "Permutation Test")
    assert has_section_explainer("regression", "Model & Inference")
    assert has_section_explainer("regression", "Permutation Test")
    assert not has_section_explainer("regression", "Estimator Tuning")
    # Another module's section of the same name gets nothing.
    assert not has_section_explainer("mask", "Model & Inference")
    assert section_explainer("regression", "Estimator Tuning") == ""


def test_the_model_box_follows_the_selection_and_the_other_does_not():
    """The model box states the formula for the CURRENT choice.

    The permutation box does not take settings: the eight controls under it
    change the test's size, its seed and its support cutoff, not its meaning.
    """
    mixed = section_explainer("regression", "Model & Inference",
                              {"regression_type": "mixed", "level": "both"})
    guide = section_explainer("regression", "Model & Inference",
                              {"regression_type": "ols", "level": "grna"})
    assert "MODEL: mixed" in mixed and "MODEL: ols" in guide
    assert mixed != guide
    assert section_explainer("regression", "Permutation Test",
                             {"regression_type": "ols"}) == \
        section_explainer("regression", "Permutation Test")
    # No settings at all is the panel's state before anything is read.
    assert "MODEL: auto" in section_explainer(
        "regression", "Model & Inference")


# ---------------------------------------------------------------------------
# 3. The CSV button
# ---------------------------------------------------------------------------

def test_both_column_settings_get_a_csv_field_not_a_bare_text_box(panel):
    for key in ("dependent_variable", "filter_column"):
        widget = panel._widgets[key]
        assert isinstance(widget, _CsvColumnField), key
        assert widget.button.text() == "CSV"
        # The row contract: every settings field carries its own key, and
        # `attach_api_tooltip` is what puts it there.
        assert widget.property("settingKey") == key


def test_the_screen_is_told_not_to_add_the_sql_button_as_well():
    """Two buttons disagreeing about which file a column comes from is worse
    than the one wrong button this replaces."""
    assert has_csv_column_picker("regression", "filter_column")
    assert has_csv_column_picker("regression", "dependent_variable")
    # Every other module keeps its measurements.db picker for the same key.
    assert not has_csv_column_picker("ml_analyze", "filter_column")
    assert not has_csv_column_picker("regression", "batch_column")


def test_with_no_csv_chosen_the_button_says_so_rather_than_offering_nothing(
        panel):
    """The failure `spacr.columns` exists to prevent, at the button."""
    field = panel._widgets["dependent_variable"]
    said = []
    field.set_reporter(said.append)
    field.set_chooser(lambda choices, current: pytest.fail(
        "the chooser was opened with no CSV loaded"))
    assert field.input_paths() == []
    assert field.pick() is None
    assert len(said) == 1
    assert "no input CSV was given" in said[0]
    # It names the setting the user has to change, not just "column".
    assert "dependent_variable=" in said[0]


def test_a_path_that_is_not_a_csv_is_reported_as_unread_not_as_no_columns(
        panel, tmp_path):
    """A missing file cannot answer the question; a missing column can."""
    junk = tmp_path / "scores.csv"
    junk.write_bytes(b"")
    panel._widgets["paired_data"].set_value(
        [{"score": str(junk), "count": None}])
    field = panel._widgets["dependent_variable"]
    said = []
    field.set_reporter(said.append)
    assert field.pick() is None
    assert "none of scores.csv could be read as a CSV" in said[0]
    # NOT "no column of that name": an unreadable file cannot answer the
    # question at all, and a caller told "column not found" about a file that
    # was never parsed goes looking in the wrong place.
    assert "could not be checked" in said[0]


def test_the_columns_offered_come_from_the_score_csv(panel, tmp_path):
    score = _csv(tmp_path / "scores.csv",
                 ["prc", "pred", "plateID", "columnID"], "w1,0.5,p1,c1")
    count = _csv(tmp_path / "counts.csv", ["prc", "grna", "count"], "w1,g1,3")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": str(count)}])

    field = panel._widgets["dependent_variable"]
    seen = {}

    def chooser(choices, current):
        seen["choices"] = list(choices)
        seen["current"] = current
        return "plateID"

    field.set_chooser(chooser)
    assert field.pick() == "plateID"
    # The SCORE file only: `dependent_variable` is a column of that and of
    # nothing else, so offering `grna` would offer a name the run cannot use.
    assert seen["choices"] == ["prc", "pred", "plateID", "columnID"]
    assert seen["current"] == "pred"
    # ...and the pick lands in the setting, not only in the dialog.
    assert field.get_value() == "plateID"
    assert panel.collect()["dependent_variable"] == "plateID"


def test_the_filter_column_is_offered_both_sides_of_the_paired_table(
        panel, tmp_path):
    """`ml.clean_controls` filters the scores and `ml.process_reads` the
    counts, so one side would be half the answer."""
    score = _csv(tmp_path / "scores.csv", ["prc", "pred", "columnID"],
                 "w1,0.5,c1")
    count = _csv(tmp_path / "counts.csv", ["prc", "grna", "rowID"], "w1,g1,r1")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": str(count)}])

    field = panel._widgets["filter_column"]
    assert field.input_paths() == [str(score), str(count)]
    offered = []
    field.set_chooser(lambda choices, current: offered.extend(choices) or None)
    field.pick()
    assert offered == ["prc", "pred", "columnID", "grna", "rowID"]


def test_one_file_shared_by_two_plates_is_read_once(panel, tmp_path):
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    other = _csv(tmp_path / "scores_b.csv", ["prc", "pred"], "w2,0.6")
    panel._widgets["paired_data"].set_value([
        {"score": str(score), "count": None},
        {"score": str(score), "count": None},
        {"score": str(other), "count": None},
    ])
    field = panel._widgets["dependent_variable"]
    assert field.input_paths() == [str(score), str(other)]


def test_the_picker_reads_the_header_row_and_never_the_body(
        panel, tmp_path, monkeypatch):
    """A score CSV is hundreds of megabytes and this runs on the GUI thread.

    Asserted twice, because a spy alone proves only that the argument was
    passed: the file's body has more fields than its header, which the C
    parser refuses to tokenize. Getting the header back at all is proof that
    the body was never read.
    """
    import pandas as pd

    path = tmp_path / "scores.csv"
    path.write_text("prc,pred\nw1,0.5,EXTRA,FIELDS,HERE\n", encoding="utf-8")
    panel._widgets["paired_data"].set_value(
        [{"score": str(path), "count": None}])

    calls = []
    real = pd.read_csv

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)

    field = panel._widgets["dependent_variable"]
    offered = []
    field.set_chooser(lambda choices, current: offered.extend(choices) or None)
    field.set_reporter(lambda message: pytest.fail(message))
    field.pick()

    assert offered == ["prc", "pred"]
    assert calls, "the picker did not read the CSV at all"
    assert all(kwargs.get("nrows") == 0 for kwargs in calls), calls


def test_the_prompt_offers_the_near_miss_when_the_typed_name_is_absent(
        panel, tmp_path):
    """`predictions` for `pred` is the typo the whole message exists for."""
    score = _csv(tmp_path / "scores.csv", ["prc", "prediction"], "w1,0.5")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": None}])
    field = panel._widgets["dependent_variable"]
    field.set_value("predictions")

    prompts = []
    field.choose = lambda choices, current, prompt="": prompts.append(prompt)
    field.pick()
    assert "No response column 'predictions'" in prompts[0]
    assert "Did you mean 'prediction'?" in prompts[0]


def test_the_prompt_counts_the_columns_when_the_name_is_already_right(
        panel, tmp_path):
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": None}])
    field = panel._widgets["dependent_variable"]
    prompts = []
    field.choose = lambda choices, current, prompt="": prompts.append(prompt)
    field.pick()
    assert prompts[0] == "2 column(s) in the input CSVs:"


def test_cancelling_the_chooser_leaves_the_typed_name_alone(panel, tmp_path):
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": None}])
    field = panel._widgets["dependent_variable"]
    field.set_value("mine")
    field.set_chooser(lambda choices, current: None)
    assert field.pick() is None
    assert field.get_value() == "mine"


def test_a_pick_announces_itself_like_any_other_setting(panel, tmp_path):
    """`value_changed` is the first signal the dependency wiring looks for,
    so a rule gated on this setting must re-evaluate on a pick."""
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": None}])
    field = panel._widgets["dependent_variable"]
    fired = []
    field.value_changed.connect(lambda: fired.append(1))
    field.set_chooser(lambda choices, current: "prc")
    field.pick()
    assert fired


def test_the_panel_can_write_a_value_back_into_the_csv_field(panel):
    """Live Preview's "Propagate settings" pushes values back through here."""
    assert panel.set_value_for_key("dependent_variable", "score_mean")
    assert panel._widgets["dependent_variable"].get_value() == "score_mean"
    assert panel.collect()["dependent_variable"] == "score_mean"


def test_an_empty_box_reads_as_none_not_as_an_empty_string(panel):
    field = panel._widgets["filter_column"]
    field.set_value(None)
    assert field.get_value() is None
    assert field.text() == ""


def test_the_count_columns_get_the_same_button_as_the_others(panel):
    """"also if the count columns ar not found id like similar behaviour, i
    think these are hardcoded and dont have a settins." They have settings
    now, and the button that goes with them."""
    for key in ("count_grna_column", "count_value_column"):
        assert isinstance(panel._widgets[key], _CsvColumnField), key


def test_the_count_columns_read_the_count_csv_and_not_the_score_one(
        panel, tmp_path):
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    count = _csv(tmp_path / "counts.csv", ["prc", "grna", "count"], "w1,g1,3")
    panel._widgets["paired_data"].set_value(
        [{"score": str(score), "count": str(count)}])
    field = panel._widgets["count_grna_column"]
    assert field.input_paths() == [str(count)]
    offered = []
    field.set_chooser(lambda choices, current: offered.extend(choices) or None)
    field.pick()
    assert offered == ["prc", "grna", "count"]


def test_the_count_columns_are_declared_against_the_count_csvs():
    """"also if the count columns ar not found id like similar behaviour."

    The settings themselves belong to another slice; the source they are read
    from belongs here, and declaring it early is free -- a key with no widget
    never reaches `_widget_for` at all.
    """
    sources = CSV_COLUMN_SOURCES["regression"]
    assert sources["count_grna_column"].roles == ("count",)
    assert sources["count_value_column"].roles == ("count",)
    assert sources["dependent_variable"].roles == ("score",)
    assert sources["filter_column"].roles == ("score", "count")


def test_a_fixed_list_of_paths_is_accepted_as_well_as_a_callable(tmp_path):
    """The widget is usable outside the settings panel that built it."""
    score = _csv(tmp_path / "s.csv", ["a", "b"], "1,2")
    field = _CsvColumnField(key="x", default="a", paths=[str(score), ""],
                            what="column")
    assert field.input_paths() == [str(score)]
    field.set_chooser(lambda choices, current: choices[-1])
    assert field.pick() == "b"
    assert field.get_value() == "b"


def test_no_paths_at_all_is_not_a_crash(tmp_path):
    field = _CsvColumnField(key="x")
    said = []
    field.set_reporter(said.append)
    assert field.input_paths() == []
    assert field.pick() is None
    assert "no input CSV was given" in said[0]


def test_the_legacy_flat_lists_still_answer_the_picker(panel, tmp_path):
    """A settings CSV written before `paired_data` carries `score_data`.

    `ml.normalize_regression_input_pairs` migrates them positionally at run
    time; the picker has to read them where they are, or a reloaded run gets
    a button that says nothing is loaded while the run itself finds files.
    """
    score = _csv(tmp_path / "scores.csv", ["prc", "pred"], "w1,0.5")
    panel._widgets["paired_data"].set_value([])
    panel._defaults["score_data"] = [str(score)]
    field = panel._widgets["dependent_variable"]
    assert field.input_paths() == [str(score)]
    assert os.path.isfile(field.input_paths()[0])
