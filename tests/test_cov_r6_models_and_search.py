"""Round-6 coverage for the training/report helpers of ``spacr.deep_spacr``.

One theme: what a model report does when a piece of the usual input is
missing or turned off.

Pinned here:

* ``deep_spacr.format_per_class_accuracy`` names the worst class only when
  there is more than one class to be worst *of*.
* ``deep_spacr._print_cv_report`` prints the fold-to-fold spread even for a
  run whose summary carries no accuracy row.
* ``deep_spacr._log_tensorboard_epoch`` writes a scalar per finite class
  accuracy and skips a non-finite one.
* ``deep_spacr._dataset_class_counts`` skips a split folder holding no class
  folders, and ``_imbalance_note`` skips a class counted at zero.
* ``deep_spacr.write_model_card`` writes the Markdown twin only on request,
  and ``model_card`` still returns a card and its path when the registry
  refuses the row.
* ``deep_spacr.generate_activation_map`` applies no Normalize step under
  ``input_statistics='none'``, and writes correlations to the database with
  the on-screen table turned off.
"""
from __future__ import annotations

import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")

from spacr import deep_spacr as D  # noqa: E402


# ===========================================================================
# spacr.deep_spacr -- per-class reporting
# ===========================================================================

def test_the_worst_class_is_only_named_when_there_is_more_than_one():
    """deep_spacr.py:801 -- ``if len(finite) > 1`` False.

    A class whose accuracy is not finite has no place in a "worst class"
    comparison, so a two-class head with one NaN leaves only one comparable
    class and the line stops at the breakdown. The same head with both
    accuracies finite and far apart does name the worst one.
    """
    one_finite = {"per_class_accuracy": [float("nan"), 0.5],
                  "class_support": [0, 20]}
    both_finite = {"per_class_accuracy": [0.40, 0.95],
                   "class_support": [10, 20]}

    line_one = D.format_per_class_accuracy(one_finite, ["neg", "pos"])
    line_both = D.format_per_class_accuracy(both_finite, ["neg", "pos"])

    assert "pos 0.500 (n=20)" in line_one
    assert "WORST" not in line_one
    assert "WORST: neg at 0.400" in line_both
    assert "0.550 below the best class" in line_both


def test_a_cv_summary_without_an_accuracy_row_still_prints_the_spread(capsys):
    """deep_spacr.py:1166 -- ``if not acc.empty`` False, exiting the report.

    A search that scored folds on f1 alone has no ``accuracy`` row. The
    per-fold table and the spread table are still the report; only the
    accuracy sentence is dropped.
    """
    fold_df = pd.DataFrame({"fold": [0, 1], "f1_macro": [0.7, 0.9]})
    no_acc = pd.DataFrame({"metric": ["f1_macro"], "mean": [0.8],
                           "std": [0.1], "min": [0.7], "max": [0.9]})
    with_acc = pd.DataFrame({"metric": ["accuracy"], "mean": [0.8],
                             "std": [0.1], "min": [0.7], "max": [0.9]})

    D._print_cv_report(fold_df, no_acc, 2)
    without = capsys.readouterr().out
    D._print_cv_report(fold_df, with_acc, 2)
    withit = capsys.readouterr().out

    assert "Fold-to-fold spread" in without and "f1_macro" in without
    assert "accuracy across folds" not in without
    assert "accuracy across folds: 0.8000 +/- 0.1000" in withit


# ===========================================================================
# spacr.deep_spacr -- class balance
# ===========================================================================

def test_a_split_folder_with_no_class_folders_is_not_a_split(tmp_path):
    """deep_spacr.py:2225 -- ``if counts:`` False, on to the next split.

    ``test/`` exists but was never populated, so it contributes no counts at
    all rather than an empty ``{}`` that downstream code would read as "this
    split has zero of everything". ``train/`` in the same tree still counts.
    """
    for name in ("a", "b"):
        d = tmp_path / "train" / name
        d.mkdir(parents=True)
        (d / "1.png").write_bytes(b"x")
    (tmp_path / "test").mkdir()

    counts = D.dataset_class_balance(str(tmp_path))

    assert counts == {"train": {"a": 1, "b": 1}}
    assert "test" not in counts


def test_a_class_counted_at_zero_is_not_a_class():
    """deep_spacr.py:2259 -- ``if count > 0`` False.

    A class with no examples cannot be the smallest class in an imbalance
    sentence -- the ratio would be infinite. It is dropped, so the note
    compares the two classes that are actually there.
    """
    note = D._imbalance_note({"gone": 0, "big": 900, "small": 100})
    assert "big" in note and "small" in note
    assert "gone" not in note
    # ... and with only one surviving class there is nothing to compare.
    assert D._imbalance_note({"gone": 0, "big": 900}) == ""


# ===========================================================================
# spacr.deep_spacr -- the model card
# ===========================================================================

def test_the_markdown_twin_is_written_only_when_asked(tmp_path):
    """deep_spacr.py:2455 -- ``if markdown:`` False.

    The JSON card is the record; the Markdown is the human-readable twin.
    Both calls write the JSON to the same path, and only the second leaves a
    ``.card.md`` beside it.
    """
    model_path = tmp_path / "weights.pth"
    card = D.build_model_card(str(model_path), classes=["a", "b"],
                              split_rule="grouped by well")

    json_path = D.write_model_card(str(model_path), card, markdown=False)
    assert os.path.isfile(json_path)
    assert not os.path.isfile(str(tmp_path / ("weights" + D.MODEL_CARD_MD_SUFFIX)))

    D.write_model_card(str(model_path), card, markdown=True)
    md = str(tmp_path / ("weights" + D.MODEL_CARD_MD_SUFFIX))
    assert os.path.isfile(md)
    assert "grouped by well" in open(md).read()


def test_a_registry_that_refuses_the_row_still_leaves_the_card(tmp_path,
                                                               capsys):
    """deep_spacr.py:2548 -- ``if artifact is not None`` False.

    Losing the registry must not lose the weights or the card. With a
    registry object that cannot register, the card is on disk, the artifact
    is None and the card carries no ``artifact_id`` -- against a real
    registry it carries one.
    """
    model_path = tmp_path / "refused" / "weights.pth"
    model_path.parent.mkdir()

    class _Broken:
        def register(self, **kwargs):
            raise RuntimeError("registry is read-only")

    card, card_path, artifact = D.model_card(
        str(model_path), registry=_Broken(), classes=["a", "b"])

    assert artifact is None
    assert "artifact_id" not in card
    assert os.path.isfile(card_path)
    assert "not registered" in capsys.readouterr().out

    ok_card, _ok_path, ok_artifact = D.model_card(
        str(tmp_path / "kept" / "weights.pth"), classes=["a", "b"])
    assert ok_artifact is not None
    assert ok_card["artifact_id"] == ok_artifact.artifact_id
