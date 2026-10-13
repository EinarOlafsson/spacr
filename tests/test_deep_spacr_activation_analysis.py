"""``deep_spacr.analyze_activation_maps`` — the attribution panel, end to end.

The function had no tests at all: it is the entry point that runs several
attribution methods over a batch of crops, scores each one for faithfulness,
checks whether the methods agree, and optionally runs the model-randomisation
sanity check. Everything below runs on a hand-built CNN on the CPU, with no
download and no training, the same way ``tests/test_attribution.py`` does — a
synthetic model is what makes the assertions have a right answer.

The one property worth stating up front: a method that *fails* on a given
architecture must come back as a row marked ``failed`` with NaN scores, never
as a silently missing row and never as an exception that loses the methods that
did work.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from spacr.attribution import NOT_AN_EXPLANATION


IMG = 12


class TinyCNN(nn.Module):
    """Three conv layers, global pool, linear head of ``n_out`` logits."""

    def __init__(self, n_out=2, in_ch=3):
        super().__init__()
        torch.manual_seed(4321)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(6, n_out)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


@pytest.fixture
def model():
    m = TinyCNN()
    m.eval()
    return m


@pytest.fixture
def image():
    torch.manual_seed(7)
    return torch.rand(3, IMG, IMG)


FAST = ["saliency", "occlusion"]


def test_one_image_and_one_row_per_method(model, image):
    """A bare (C, H, W) tensor is accepted as a batch of one."""
    from spacr.deep_spacr import analyze_activation_maps

    out = analyze_activation_maps(model, image, methods=FAST,
                                  n_steps=3, sanity_check=False)

    table = out["table"]
    assert set(table["method"]) == set(FAST)
    assert set(table["image"]) == {0}
    assert len(out["attributions"]) == 1
    assert set(out.keys()) == {"table", "attributions", "agreement",
                               "sanity", "notes"}
    assert out["sanity"] == {}
    assert NOT_AN_EXPLANATION in out["notes"]
    # every method scored on every faithfulness axis
    for column in ("deletion_auc", "insertion_auc"):
        assert table[column].notna().all()


def test_several_images_are_indexed_and_sorted(model):
    """The rows carry the image they came from and sort by deletion AUC."""
    from spacr.deep_spacr import analyze_activation_maps

    torch.manual_seed(11)
    images = [torch.rand(3, IMG, IMG) for _ in range(3)]

    out = analyze_activation_maps(model, images, methods=FAST,
                                  n_steps=3, sanity_check=False)

    table = out["table"]
    assert len(table) == len(images) * len(FAST)
    assert sorted(table["image"].unique()) == [0, 1, 2]
    assert len(out["attributions"]) == 3
    for i in sorted(table["image"].unique()):
        block = table[table["image"] == i]
        aucs = block["deletion_auc"].tolist()
        assert aucs == sorted(aucs), "rows are ordered by deletion AUC"


def test_no_images_is_refused_with_a_reason(model):
    """An empty batch produces nothing to attribute; say so rather than
    returning an empty table that looks like a finished analysis."""
    from spacr.deep_spacr import analyze_activation_maps

    with pytest.raises(ValueError, match="at least one image"):
        analyze_activation_maps(model, [], methods=FAST)


def test_a_method_that_cannot_run_is_a_failed_row_not_a_lost_one(model, image):
    """An unknown method comes back marked, with NaN scores, and the methods
    that did work are still in the table."""
    from spacr.deep_spacr import analyze_activation_maps

    out = analyze_activation_maps(model, image,
                                  methods=["saliency", "not_a_method"],
                                  n_steps=3, sanity_check=False)

    table = out["table"]
    assert set(table["method"]) == {"saliency", "not_a_method"}
    bad = table[table["method"] == "not_a_method"].iloc[0]
    assert bad["failed"] is True or bad["failed"] == True  # noqa: E712
    assert np.isnan(bad["deletion_auc"]) and np.isnan(bad["insertion_auc"])
    assert bad["pointing_game"] is None
    good = table[table["method"] == "saliency"].iloc[0]
    assert not good["failed"]
    assert not np.isnan(good["deletion_auc"])


def test_masks_score_the_pointing_game_and_drop_the_caveat(model, image):
    """With object masks the pointing game is scored, and the note that says
    it was not goes away."""
    from spacr.deep_spacr import analyze_activation_maps

    mask = np.zeros((IMG, IMG), dtype=bool)
    mask[:IMG // 2, :IMG // 2] = True

    with_mask = analyze_activation_maps(model, image, methods=FAST,
                                        masks=[mask], n_steps=3,
                                        sanity_check=False)
    without = analyze_activation_maps(model, image, methods=FAST,
                                      n_steps=3, sanity_check=False)

    assert with_mask["table"]["pointing_game"].notna().all()
    assert any("pointing game was not scored" in n for n in without["notes"])
    assert not any("pointing game was not scored" in n
                   for n in with_mask["notes"])
    assert without["table"]["pointing_game"].isna().all()


def test_the_sanity_check_runs_per_method_and_is_reported(model, image, capsys):
    """sanity_check=True fills one entry per method and drops the caveat."""
    from spacr.deep_spacr import analyze_activation_maps

    out = analyze_activation_maps(model, image, methods=["saliency"],
                                  n_steps=3, sanity_check=True, verbose=True)

    assert set(out["sanity"]) == {"saliency"}
    assert not any("sanity check was skipped" in n for n in out["notes"])
    printed = capsys.readouterr().out
    assert NOT_AN_EXPLANATION.split("\n")[0][:20] in printed


def test_a_sanity_check_that_raises_is_recorded_not_propagated(model, image,
                                                               monkeypatch):
    """One method blowing up must not lose the whole analysis."""
    import spacr.attribution as attribution
    from spacr.deep_spacr import analyze_activation_maps

    def boom(*args, **kwargs):
        raise RuntimeError("no randomised model available")

    monkeypatch.setattr(attribution, "randomization_sanity_check", boom)

    out = analyze_activation_maps(model, image, methods=["saliency"],
                                  n_steps=3, sanity_check=True, verbose=True)

    assert out["sanity"]["saliency"].startswith("RuntimeError: ")
    assert "no randomised model available" in out["sanity"]["saliency"]
    assert len(out["table"]) == 1


def test_agreement_needs_two_non_flat_maps(model, image):
    """One method cannot agree with itself, so agreement stays None."""
    from spacr.deep_spacr import analyze_activation_maps

    one = analyze_activation_maps(model, image, methods=["saliency"],
                                  n_steps=3, sanity_check=False)
    assert one["agreement"] is None

    two = analyze_activation_maps(model, image, methods=FAST,
                                  n_steps=3, sanity_check=False,
                                  verbose=True)
    assert two["agreement"] is not None
    assert hasattr(two["agreement"], "verdict")


def test_the_default_method_list_is_used_when_none_is_given(model, image):
    """The default panel is one representative per family plus Grad-CAM."""
    from spacr.deep_spacr import analyze_activation_maps

    out = analyze_activation_maps(model, image, n_steps=2, sanity_check=False)

    assert set(out["table"]["method"]) == {
        "gradcam", "saliency", "integrated_gradients", "occlusion"}
