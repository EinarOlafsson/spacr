"""Three classes must not be a special case that two hid.

Instruction 236 D13: "ANNOTATIONS OR MEASUREMENTS, AND AN ARBITRARY NUMBER
OF CLASSES, in both modules. Two classes must not be a special case that
three breaks."

WHAT THE TABULAR CLASSIFIER ACTUALLY IS. `ml_analysis` fits ONE binary
model: one arm is `negative_control`, the other `positive_control`, and
every remaining row of the input is scored afterwards by that model. That
is the point of a screen -- the unknown population is what the scores are
for -- and it is not a defect.

WHAT WAS. With three or more classes in the column, two were trained on and
the rest were scored by a model that had never seen them, and NOTHING SAID
SO. A column holding 1, 2 and 3 produced a full score table, binary
metrics, and no sign anywhere that a third of the labelled data had not
taken part.

Both controls take a LIST, so classes can be pooled into the two arms
deliberately. That is the answer to "an arbitrary number of classes" on
this path, and it is now said in the same breath as the warning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _three_classes(rows=240, features=8, seed=0):
    """A separable three-class frame with the identity the splitter needs."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"cell_channel_1_f{i}" for i in range(features)])
    labels = np.array([0, 1, 2] * (rows // 3))
    frame["columnID"] = [f"c{1 + label}" for label in labels]
    frame["rowID"] = [f"r{1 + i % 8}" for i in range(rows)]
    frame["plateID"] = "plate1"
    frame["fieldID"] = [f"f{1 + i % 4}" for i in range(rows)]
    frame["object_label"] = [str(i) for i in range(rows)]
    for level in range(3):
        frame.loc[labels == level, frame.columns[:features]] += 2.5 * level
    frame.index = [
        f"plate1_{frame['rowID'][i]}_{frame['columnID'][i]}_"
        f"{frame['fieldID'][i]}_o{i}" for i in range(rows)]
    return frame


def _fit(frame, positive, negative="c1"):
    from spacr.ml import ml_analysis

    return ml_analysis(
        frame.copy(), channel_of_interest=1, location_column="columnID",
        positive_control=positive, negative_control=negative,
        n_repeats=1, top_features=5, n_estimators=12,
        model_type="random_forest", n_jobs=1,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False, verbose=False)


class TestAClassNobodyNamed:
    def test_it_is_named_out_loud(self, capsys):
        """THE DEFECT: a third of the labelled data sat out the fit and
        nothing on screen or on disk said which third."""
        _fit(_three_classes(), "c3")
        said = capsys.readouterr().out
        assert "c2" in said
        assert "not in the training set" in said

    def test_the_message_says_what_to_do_about_it(self, capsys):
        """A warning that names a problem and no remedy is a warning people
        learn to scroll past."""
        _fit(_three_classes(), "c3")
        said = capsys.readouterr().out.lower()
        assert "list" in said
        assert "positive_control" in said and "negative_control" in said

    def test_two_classes_say_nothing(self, capsys):
        """The ordinary case must stay quiet, or the warning is noise."""
        frame = _three_classes()
        frame = frame[frame["columnID"] != "c2"]
        _fit(frame, "c3")
        assert "not in the training set" not in capsys.readouterr().out

    def test_pooling_the_classes_silences_it(self, capsys):
        """Both controls take a list, and naming every class is what "an
        arbitrary number of classes" means on this path."""
        _fit(_three_classes(), ["c2", "c3"])
        assert "not in the training set" not in capsys.readouterr().out

    def test_the_unnamed_class_is_still_scored(self):
        """The warning must not have turned into a filter. Scoring the
        unknown population is the whole purpose of the fit."""
        output, _figures = _fit(_three_classes(), "c3")
        scored = output[0]
        assert set(scored["columnID"].unique()) == {"c1", "c2", "c3"}
        assert len(scored) == 240


class TestPoolingIsARealOption:
    def test_a_pooled_arm_trains_on_both_of_its_classes(self):
        """`positive_control=['c2', 'c3']` has to mean 160 training rows,
        not 80 -- otherwise the list is accepted and half-ignored."""
        from spacr.ml import ml_analysis

        frame = _three_classes()
        one, _ = _fit(frame, "c3")
        both, _ = _fit(frame, ["c2", "c3"])
        assert len(both[4]) + len(both[5]) > len(one[4]) + len(one[5])

    def test_the_pooled_fit_still_separates(self):
        """A pooled arm is a real classifier, not a shape that merely
        runs."""
        output, _figures = _fit(_three_classes(), ["c2", "c3"])
        metrics = output[8]
        assert float(metrics.loc["accuracy"].iloc[0]) > 0.6

    def test_the_warning_cannot_break_a_run(self):
        """IT SITS ABOVE THE GUARD THAT REFUSES A DUPLICATED COLUMN, where
        `df[location_column]` is a DataFrame rather than a Series. It
        raised AttributeError there and masked the guard's own message --
        which is the one the user needed, and the one that turned ten
        auto-filed sklearn tracebacks into a sentence. A cosmetic line must
        never be able to do that."""
        frame = _three_classes()
        frame["dup"] = frame["columnID"]
        frame.columns = ["columnID" if c == "dup" else c
                         for c in frame.columns]
        with pytest.raises(ValueError, match="columns named 'columnID'"):
            _fit(frame, "c3")

    def test_a_control_naming_nothing_still_fails_by_name(self):
        """The pooling must not have loosened the guard that turned ten
        auto-filed sklearn tracebacks into one sentence."""
        with pytest.raises(ValueError):
            _fit(_three_classes(), ["c9", "c8"])


class TestTheOtherHalfCountsItsClasses:
    """The CV classifier defines classes with a mapping of name to
    {column, value}, which is arbitrary-N by construction. The tabular half
    above is binary by construction. The two are different modules and the
    difference is worth pinning, because it is the thing a reader of the
    settings panel would most reasonably get wrong."""

    def test_the_cv_classes_setting_is_a_mapping_not_a_pair(self):
        from spacr.settings import expected_types

        assert dict in _as_tuple(expected_types["classes"])

    def test_its_help_says_it_is_per_class(self):
        from spacr.settings import tooltips

        said = tooltips["classes"].lower()
        assert "class name" in said or "each class" in said


def _as_tuple(declared):
    return declared if isinstance(declared, tuple) else (declared,)
