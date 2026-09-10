"""The two things the maintainer asked for when only one class is annotated.

Requested 2026-09-05, verbatim: "The suggest button should train an xgboost
model on your annotated images (if only one class randomly choose the same
number of images as is annotated for the other class) if there is class
imbalance use the class with fewer."

Both are properties of the FIT, so they live in ``retrain_round`` rather
than in the button. ``spacr.suggest`` deliberately does not fit a model --
it reads the probabilities this writes -- and building a second fit in the
GUI layer to hold these two rules would have produced two models with two
answers and no way to tell which the reviewer was looking at.

WHAT IS ACTUALLY BEING PROTECTED HERE is not that the numbers come out
right; it is that the round SAYS WHAT IT DID. A downsampled fit and a
reweighted one give different probabilities from the same crops, and
``suggest_from_scores`` sorts on those probabilities. A card that says
"balanced" without saying how describes two different models, and a card
that does not mention invented negatives describes a model that does not
exist.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import active_learning as al

pytest.importorskip("sklearn")

from tests.test_active_learning_loop import label, make_db  # noqa: E402


@pytest.fixture
def screen(tmp_path):
    """One plate, six wells, 120 crops with a separable latent class."""
    db = tmp_path / "measurements" / "measurements.db"
    rng = np.random.default_rng(0)
    rows, features = [], []
    i = 0
    for row_id in ("r1", "r2"):
        for column in ("c1", "c2", "c3"):
            for _ in range(20):
                path = f"/crops/cell_{i:04d}.png"
                latent = 1 if i % 2 == 0 else 2
                rows.append({
                    "png_path": path, "plateID": "p1", "rowID": row_id,
                    "columnID": column, "fieldID": "f1",
                    "prcfo": f"p1_{row_id}_{column}_f1_o{i}",
                    "cell_id": f"o{i}", "pred": 0.5, "annotate": None})
                features.append({
                    "png_path": path,
                    "signal": float(latent) + rng.normal(0, 0.25),
                    "noise": rng.normal(0, 1.0)})
                i += 1
    make_db(db, rows)
    return {"db": str(db),
            "paths": [r["png_path"] for r in rows],
            "features": pd.DataFrame(features).set_index("png_path")}


def _notes(result) -> str:
    return "\n".join(result.notes)


def _lopsided(paths):
    """40 of class 1 and 8 of class 2, SPREAD ACROSS FOUR WELLS.

    The wells are 20 crops each. Putting the minority class inside one of
    them makes `grouped_split` refuse before any of this is reached -- "a
    leakage-safe well-grouped split cannot put every class in both train
    and test" -- which is the split doing its job and has nothing to say
    about balancing. So both classes appear in wells r1/c1, r1/c2, r1/c3
    and r2/c1.
    """
    labels = {}
    for well_start in (0, 20, 40, 60):
        for offset in range(0, 20, 2):          # ten of class 1 per well
            labels[paths[well_start + offset]] = 1
        for offset in (1, 3):                   # two of class 2 per well
            labels[paths[well_start + offset]] = 2
    return labels


# ---------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------

def test_downsampling_evens_the_classes_and_says_it_did(screen):
    """40 of one class and 8 of the other must fit on 8 and 8.

    ``n_labels`` is the count the model was FITTED on, so a round that
    dropped 32 rows and still reported 48 would be describing a model
    nobody trained.
    """
    label(screen["db"], _lopsided(screen["paths"]))

    result = al.retrain_round(
        screen["db"], "annotate", features=screen["features"],
        seed=0, save_model=False, balance="downsample")

    assert result.n_labels == 16, (
        f"fitted on {result.n_labels}, not on 8 + 8")
    assert "balance=downsample" in _notes(result)
    assert "32 rows of the larger class were dropped" in _notes(result)


def test_the_default_leaves_the_imbalance_to_the_estimator_and_says_so(screen):
    """"balanced" means two different things and the round names which.

    The default estimators already pass ``class_weight='balanced'``, so
    "we balanced it" is ambiguous between reweighting and dropping. Both
    branches write a note for that reason -- silence on the default would
    make the downsampling note read as the only balancing that exists.
    """
    label(screen["db"], _lopsided(screen["paths"]))

    result = al.retrain_round(
        screen["db"], "annotate", features=screen["features"],
        seed=0, save_model=False)

    assert result.n_labels == 48, "the default must not drop rows"
    assert "balance=none" in _notes(result)
    assert "reweights rather than drops" in _notes(result)


def test_downsampling_an_already_even_set_drops_nothing(screen):
    """And says that, rather than claiming a reduction it did not make."""
    labels = {screen["paths"][i]: (1 if i % 2 == 0 else 2) for i in range(40)}
    label(screen["db"], labels)

    result = al.retrain_round(
        screen["db"], "annotate", features=screen["features"],
        seed=0, save_model=False, balance="downsample")

    assert result.n_labels == 40
    assert "already even" in _notes(result)


def test_downsampling_is_deterministic(screen):
    """Same seed, same rows: a round has to be reproducible to be citable."""
    label(screen["db"], _lopsided(screen["paths"]))

    kwargs = dict(features=screen["features"], seed=7, save_model=False,
                  balance="downsample", write_scores=False)
    first = al.retrain_round(screen["db"], "annotate", **kwargs)
    second = al.retrain_round(screen["db"], "annotate", **kwargs)

    assert first.report["accuracy"] == second.report["accuracy"]


# ---------------------------------------------------------------------------
# synthetic negatives
# ---------------------------------------------------------------------------

def test_one_class_still_refuses_when_nothing_is_asked_for(screen):
    """The default is unchanged: a single class is an error with a reason."""
    label(screen["db"], {screen["paths"][i]: 1 for i in range(30)})

    with pytest.raises(ValueError, match="needs at least two classes"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], seed=0,
                         save_model=False)


def test_synthetic_negatives_let_a_one_class_round_fit(screen):
    """The maintainer's rule: draw as many as are annotated, as the other class.

    The fit is the easy half. The assertion that matters is the note:
    these negatives were INVENTED, and anything downstream that reports
    this round has to be able to say so.
    """
    label(screen["db"], {screen["paths"][i]: 1 for i in range(30)})

    result = al.retrain_round(
        screen["db"], "annotate", features=screen["features"], seed=0,
        save_model=False, synthetic_negatives=30)

    assert result.n_labels == 60, "30 annotated plus 30 invented"
    notes = _notes(result)
    assert "INVENTED" in notes
    assert "fitted as class 2" in notes
    assert "RANKING and not a verdict" in notes, (
        "the caveat that stops a bulk accept must be in words, not implied")


def test_the_invented_negatives_are_never_written_to_the_database(screen):
    """They exist for one fit and must not become labels.

    A synthetic negative written into the annotation column would be
    indistinguishable from a human's answer on the next round -- the same
    class of defect as fitting on suggestions, one layer further in.
    """
    label(screen["db"], {screen["paths"][i]: 1 for i in range(30)})
    before = _annotations(screen["db"])

    al.retrain_round(screen["db"], "annotate", features=screen["features"],
                     seed=0, save_model=False, synthetic_negatives=30)

    assert _annotations(screen["db"]) == before


def test_a_non_binary_class_refuses_rather_than_guessing(screen):
    """There is no defensible "other class" for class 3."""
    label(screen["db"], {screen["paths"][i]: 3 for i in range(30)})

    with pytest.raises(ValueError, match="binary classes 1 and 2 only"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], seed=0,
                         save_model=False, synthetic_negatives=30)


def test_asking_for_more_negatives_than_exist_says_so(screen):
    """A refusal that names both numbers, rather than an IndexError."""
    label(screen["db"], {screen["paths"][i]: 1 for i in range(30)})

    with pytest.raises(ValueError, match="only 90 unannotated crops"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], seed=0,
                         save_model=False, synthetic_negatives=500)


def _annotations(db: str) -> dict:
    """Every annotation value in the crop table, keyed by path."""
    con = sqlite3.connect(db)
    rows = con.execute("SELECT png_path, annotate FROM png_list").fetchall()
    con.close()
    return dict(rows)
