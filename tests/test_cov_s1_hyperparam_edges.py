"""What a search refuses before it takes the machine.

Every search here can run for hours, so each one checks its own arguments
first and says which one is wrong. The alternative -- a step of ``"fast"``
reaching the fit function -- fails somewhere inside UMAP, after the features
have been loaded, with a message about the wrong thing entirely.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.hyperparam import (SearchSpace, WalkAxis, local_direction_search,
                              umap_search, umap_walk_axes, walk_search)


def _never_called(_params):
    raise AssertionError("validation must happen before the first fit")


_START = {"n_neighbors": 15, "min_dist": 0.1}


# ---------------------------------------------------------------------------
# One axis of a Walk
# ---------------------------------------------------------------------------

def test_an_axis_with_no_parameter_name_is_refused():
    """An axis is a direction in the fit function's own parameter space. One
    without a name has nothing to vary."""
    with pytest.raises(ValueError) as excinfo:
        WalkAxis(name="   ", step=1.0)

    assert "A Walk axis needs a parameter name." in str(excinfo.value)


def test_an_axis_resolution_that_is_not_a_whole_number_is_refused():
    """Resolution is how many values the axis contributes to one round, so a
    fraction of a value is not a thing -- and the message has to name the
    axis, because a walk is configured several at a time."""
    with pytest.raises(ValueError) as excinfo:
        WalkAxis(name="n_neighbors", step=1.0, resolution="fine")

    assert "Walk axis 'n_neighbors' needs a whole-number resolution." in str(
        excinfo.value)


def test_a_numeric_axis_with_a_step_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError) as excinfo:
        WalkAxis(name="min_dist", step="a bit")

    assert "Walk axis 'min_dist' needs a numeric step." in str(excinfo.value)


def test_a_step_a_caller_chose_reaches_the_axis_it_names():
    """The panel lets a user set a step per parameter. A step that did not
    reach the axis would leave the walk moving at the shipped default while
    the dialog said otherwise."""
    axes = umap_walk_axes(dict(_START), parameters=("n_neighbors",),
                          steps={"n_neighbors": 4})

    assert [axis.name for axis in axes] == ["n_neighbors"]
    assert axes[0].step == 4


# ---------------------------------------------------------------------------
# The Walk's own limits
# ---------------------------------------------------------------------------

def test_a_round_limit_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError) as excinfo:
        walk_search(_never_called, _START,
                    [WalkAxis(name="n_neighbors", step=1.0)],
                    n_trials="as many as it takes")

    assert "A Walk requires a numeric round limit" in str(excinfo.value)


def test_a_walk_of_no_rounds_is_refused():
    """Zero rounds would return the starting point as the answer without
    ever having scored it -- the walk never fits the centre."""
    with pytest.raises(ValueError) as excinfo:
        walk_search(_never_called, _START,
                    [WalkAxis(name="n_neighbors", step=1.0)], n_trials=0)

    assert "A Walk needs a maximum of at least 1 round." in str(excinfo.value)


def test_a_negative_stopping_threshold_is_refused():
    """A negative threshold means "continue when the score gets worse", which
    is a walk that never stops."""
    with pytest.raises(ValueError) as excinfo:
        walk_search(_never_called, _START,
                    [WalkAxis(name="n_neighbors", step=1.0)],
                    min_improvement=-0.5)

    assert "The minimum improvement must be zero or greater." in str(
        excinfo.value)


# ---------------------------------------------------------------------------
# The two-axis UMAP walk
# ---------------------------------------------------------------------------

def test_a_umap_walk_needs_numeric_steps_and_a_numeric_start():
    for kwargs in ({"n_neighbors_step": "one"},
                   {"min_dist_step": "small"},
                   {"n_neighbors_max": "lots"}):
        with pytest.raises(ValueError) as excinfo:
            local_direction_search(_never_called, _START, **kwargs)
        assert "requires numeric n_neighbors, min_dist, and step sizes" in str(
            excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called,
                               {"n_neighbors": "fifteen", "min_dist": 0.1})
    assert "requires numeric n_neighbors, min_dist, and step sizes" in str(
        excinfo.value)


def test_a_umap_walk_step_of_zero_would_never_move():
    for kwargs in ({"n_neighbors_step": 0}, {"min_dist_step": 0.0}):
        with pytest.raises(ValueError) as excinfo:
            local_direction_search(_never_called, _START, **kwargs)
        assert "steps must be positive" in str(excinfo.value)


def test_a_umap_walk_with_a_negative_threshold_is_refused():
    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called, _START, min_improvement=-1.0)

    assert "minimum improvement must be zero or greater" in str(excinfo.value)


def test_a_threshold_that_is_not_a_number_is_refused_the_same_way():
    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called, _START,
                               min_improvement="a little")

    assert "requires numeric n_neighbors, min_dist, and step sizes" in str(
        excinfo.value)


def test_a_neighbour_ceiling_below_two_is_refused():
    """UMAP itself needs at least two neighbours, so a ceiling of one bounds
    the walk into a region where every fit would fail."""
    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called, _START, n_neighbors_max=1)

    assert "n_neighbors_max must be at least 2." in str(excinfo.value)


def test_a_umap_walk_round_limit_must_be_a_number_and_at_least_one():
    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called, _START, n_trials="many")
    assert "requires numeric n_neighbors, min_dist, and step sizes" in str(
        excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        local_direction_search(_never_called, _START, n_trials=0)
    assert "needs n_trials/max rounds of at least 1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The UMAP sweep's own arguments
# ---------------------------------------------------------------------------

@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 4))


@pytest.fixture
def space():
    return SearchSpace({"n_neighbors": [5], "min_dist": [0.1]})


def test_a_display_dimensionality_that_is_not_two_or_three_is_refused(
        features, space):
    """The embedding is drawn. Four dimensions is a valid UMAP and an
    undrawable panel, and the sweep is the panel."""
    for bad in ("many", 4, 1):
        with pytest.raises(ValueError) as excinfo:
            umap_search(features, space, n_components=bad)
        assert "UMAP n_components must be 2 or 3." in str(excinfo.value)


def test_a_stability_repeat_count_that_is_not_a_number_is_refused(features,
                                                                  space):
    with pytest.raises(ValueError) as excinfo:
        umap_search(features, space, stability_repeats="a few")

    assert "stability_repeats must be a whole number." in str(excinfo.value)


def test_features_that_are_not_a_table_are_refused_by_name(space):
    """``features.shape`` is tried first and ``len()`` second, so a list of
    rows still works; something with neither is not a feature matrix."""
    with pytest.raises(ValueError) as excinfo:
        umap_search(object(), space)

    assert "UMAP features must be a 2-D array-like object." in str(
        excinfo.value)


def test_a_sweep_of_two_rows_says_how_many_are_left(space):
    """Filtering upstream can leave almost nothing, and "3" is the floor a
    neighbourhood statistic needs at all."""
    with pytest.raises(ValueError) as excinfo:
        umap_search([[1.0, 2.0], [3.0, 4.0]], space)

    assert "needs at least 3 rows after filtering" in str(excinfo.value)
    assert "only 2 remain" in str(excinfo.value)


def test_a_neighbour_count_typed_as_a_word_names_the_value(features):
    """A search space comes from text fields. The message quotes the value so
    the user can find which field it came from."""
    with pytest.raises(ValueError) as excinfo:
        umap_search(features, SearchSpace({"n_neighbors": ["fifteen"]}))

    assert "UMAP n_neighbors values must be whole numbers" in str(
        excinfo.value)
    assert "'fifteen'" in str(excinfo.value)


def test_a_walk_over_a_grid_is_refused_because_it_has_no_single_start(
        features):
    """A Walk is a path from one point. Given several values per parameter
    there is no "here" to walk from, and silently picking one would answer a
    question the user did not ask."""
    grid = SearchSpace({"n_neighbors": [5, 10], "min_dist": [0.1]})

    with pytest.raises(ValueError) as excinfo:
        umap_search(features, grid, adaptive=True)

    assert "A Walk needs exactly one starting value" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A sweep that runs, with its optional second analysis failing
# ---------------------------------------------------------------------------

#: Every configuration ``_fake_embedding`` was actually asked to embed.
_EMBED_CALLS = []


def _fake_embedding(feats, params):
    """A deterministic stand-in for UMAP, so the sweep runs without it."""
    _EMBED_CALLS.append(dict(params))
    values = np.asarray(feats, dtype=float)
    shift = float(params.get("min_dist", 0.1))
    return np.column_stack([values[:, 0] + shift, values[:, 1] - shift])


def test_a_clustering_failure_stays_on_its_row_and_keeps_the_embedding(
        features, monkeypatch):
    """Clustering is a second analysis of a map that is already valid.

    Letting its failure kill the row would throw away an embedding that cost
    minutes to compute, and hide from the user which half went wrong.
    """
    import spacr.umap_search as umap_search_module

    def _refuses(*_args, **_kwargs):
        raise RuntimeError("hdbscan is not installed")

    monkeypatch.setattr(umap_search_module, "walk_clusters", _refuses)

    result = umap_search(features, SearchSpace({"n_neighbors": [5],
                                                "min_dist": [0.1]}),
                         embed_fn=_fake_embedding,
                         cluster_during_search=True, neighbourhood_k=5)

    assert result.trials
    row = result.trials[0]
    assert row.score is not None
    assert row.extra_metrics["cluster_error"] == (
        "RuntimeError: hdbscan is not installed")
    assert "embedding" in row.extra_metrics


def test_a_walk_over_the_parameters_the_user_named_uses_those_axes(features):
    """The panel lets a user choose which UMAP parameters the Walk moves
    along. A Walk that ignored the choice would search the two it always
    searched and report progress on parameters nobody selected."""
    result = umap_search(features,
                         SearchSpace({"n_neighbors": [8], "min_dist": [0.2]}),
                         adaptive=True, walk_parameters=("min_dist",),
                         walk_steps={"min_dist": 0.05},
                         n_trials=2, embed_fn=_fake_embedding,
                         neighbourhood_k=5)

    assert result.trials
    varied = {trial.params["min_dist"] for trial in result.trials}
    assert len(varied) > 1, "the walk moved along min_dist"
    assert {trial.params["n_neighbors"] for trial in result.trials} == {8}, \
        "a parameter the user did not name is held fixed"


# ---------------------------------------------------------------------------
# Reusing work across runs
# ---------------------------------------------------------------------------

def test_a_resumed_sweep_replays_its_finished_trials_to_the_progress_bar(
        features, tmp_path):
    """A resumed grid must account for what it is not refitting.

    Reporting only the new trials makes a resume look as though it lost the
    hours it is in fact saving.
    """
    space = SearchSpace({"n_neighbors": [5, 8], "min_dist": [0.1]})
    path = str(tmp_path / "sweep.json")
    umap_search(features, space, embed_fn=_fake_embedding,
                neighbourhood_k=5, checkpoint_path=path)

    _EMBED_CALLS.clear()
    reported = []
    result = umap_search(
        features, space, embed_fn=_fake_embedding,
        neighbourhood_k=5, checkpoint_path=path, resume=True,
        on_trial=lambda trial, done, total: reported.append((done, total)))

    assert _EMBED_CALLS == [], "every configuration was already scored"
    assert len(result.trials) == 2
    assert reported == [(1, 2), (2, 2)]


def test_a_sampler_that_cannot_find_new_points_falls_back_to_the_grid(
        monkeypatch):
    """Rejection sampling is bounded so a pathological space cannot spin for
    ever, and the remainder is filled from the grid in a fixed order -- so a
    seeded search is still reproducible when the sampler gives up."""
    from spacr.hyperparam import random_search

    space = SearchSpace({"n_neighbors": [5, 8, 11, 14]})
    monkeypatch.setattr(SearchSpace, "sample",
                        lambda self, rng: {"n_neighbors": 5})

    result = random_search(lambda params: float(params["n_neighbors"]),
                           space, n_trials=3, seed=0)

    assert [trial.params["n_neighbors"] for trial in result.trials] == \
        [5, 8, 11]


# ---------------------------------------------------------------------------
# What an Activation sweep can be pointed at
# ---------------------------------------------------------------------------

def test_merged_arrays_without_a_mask_plane_fall_back_to_the_crop_tar(
        tmp_path):
    """A merged array from a run that segmented nothing carries no mask plane.

    The pointing game needs one, so those files are skipped -- and the note
    has to say the tar was used instead, because the criterion the user
    selected is quietly unavailable on that source.
    """
    torch = pytest.importorskip("torch")
    import tarfile

    model_path = tmp_path / "model.pth"
    torch.save(torch.nn.Sequential(torch.nn.Flatten(),
                                   torch.nn.Linear(4, 2)), model_path)
    merged = tmp_path / "src" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "field1.npy", np.zeros((8, 8, 2), dtype=np.uint16))
    empty_tar = tmp_path / "crops.tar"
    with tarfile.open(empty_tar, "w"):
        pass

    with pytest.raises(ValueError) as excinfo:
        from spacr.hyperparam import load_activation_data
        load_activation_data({
            "model_path": str(model_path), "src": str(tmp_path / "src"),
            "dataset": str(empty_tar), "channels": [0, 1],
            "image_size": 8, "input_statistics": "symmetric"})

    assert "yielded no images" in str(excinfo.value)


def test_crop_tar_loading_can_explicitly_skip_normalization(tmp_path):
    """``input_statistics='none'`` keeps ToTensor's raw [0, 1] values."""
    import tarfile
    from io import BytesIO

    torch = pytest.importorskip("torch")
    from PIL import Image

    model_path = tmp_path / "model.pth"
    torch.save(torch.nn.Sequential(torch.nn.Flatten(),
                                   torch.nn.Linear(3 * 8 * 8, 2)), model_path)

    pixels = np.empty((8, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 64
    pixels[..., 1] = 128
    pixels[..., 2] = 255
    payload = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(payload, format="PNG")
    image_bytes = payload.getvalue()

    dataset = tmp_path / "crops.tar"
    with tarfile.open(dataset, "w") as archive:
        member = tarfile.TarInfo("plate_A1_1.png")
        member.size = len(image_bytes)
        archive.addfile(member, BytesIO(image_bytes))

    from spacr.hyperparam import load_activation_data
    data = load_activation_data({
        "model_path": str(model_path), "dataset": str(dataset),
        "src": str(tmp_path / "no-merged"), "channels": [1, 2, 3],
        "image_size": 8, "normalize_input": True,
        "input_statistics": "none",
    }, n_images=1)

    observed = data.images[0][:, 0, 0].detach().cpu().numpy()
    assert observed == pytest.approx(np.array([64, 128, 255]) / 255.0)
