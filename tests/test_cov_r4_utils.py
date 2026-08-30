"""Edges of :mod:`spacr.utils`: the branch each helper takes when told less.

Every test here drives one path through a utility that the ordinary pipeline
reaches only with a particular shape of input -- a field with nothing in it, a
stack with no pathogen channel, a checkpoint that never recorded an accuracy,
a torchvision backbone with no classification head, a folder that was already
made. Those are the states real runs arrive in, and each of them used to be
the difference between a number and a lost field.

The file is organised in the order the module is: masks, the measurement
database, the models, the plots, the mask mergers, and the folders.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import utils


# ===========================================================================
# splitting, merging and filtering one field
# ===========================================================================

def _in_memory(mask, **overrides):
    """Call ``_process_single_fov_in_memory`` with the pipeline's defaults."""
    settings = dict(
        intensity_img=None, intensity_channel=None, do_split=True,
        do_perimeter_merge=False, do_intensity_merge=False,
        perimeter_fraction=0.5, area_multiplier=2.0, min_distance=10,
        min_object_area=100, intensity_threshold_method='mean',
        intensity_percentile=75, min_area=0, max_area=0,
        remove_border_objects=False, min_intensity_percentile=0,
        max_intensity_percentile=100)
    settings.update(overrides)
    return utils._process_single_fov_in_memory(
        mask, settings['intensity_img'], settings['intensity_channel'],
        settings['do_split'], settings['do_perimeter_merge'],
        settings['do_intensity_merge'], settings['perimeter_fraction'],
        settings['area_multiplier'], settings['min_distance'],
        settings['min_object_area'], settings['intensity_threshold_method'],
        settings['intensity_percentile'], settings['min_area'],
        settings['max_area'], settings['remove_border_objects'],
        settings['min_intensity_percentile'],
        settings['max_intensity_percentile'])


def _dumbbell():
    """Two overlapping discs the watershed can separate."""
    mask = np.zeros((40, 60), np.uint16)
    yy, xx = np.ogrid[:40, :60]
    mask[((yy - 20) ** 2 + (xx - 18) ** 2) <= 100] = 1
    mask[((yy - 20) ** 2 + (xx - 40) ** 2) <= 100] = 1
    return mask


def test_a_split_that_changed_nothing_is_not_announced(capsys):
    """The line is a report of work done, so no work means no line.

    A split pass runs over every field of a run. Printing "1 -> 1 objects" for
    each of them buries the fields where the split really did something, which
    is the only reason the line exists.
    """
    small = np.zeros((30, 30), np.uint16)
    small[2:5, 2:5] = 1
    small[20:23, 20:23] = 2

    kept = _in_memory(small)

    assert sorted(np.unique(kept).tolist()) == [0, 1, 2]
    assert "split:" not in capsys.readouterr().out

    split = _in_memory(_dumbbell(), min_object_area=10, area_multiplier=0.5,
                       min_distance=5)

    assert sorted(np.unique(split).tolist()) == [0, 1, 2]
    assert "split: 1 → 2 objects" in capsys.readouterr().out


def test_a_field_with_no_objects_is_written_back_unchanged(tmp_path):
    """An empty mask has nothing to merge, and must still be a mask.

    ``_process_single_fov`` has no early return for an empty field -- it is
    the on-disk half of the pipeline and its job is to leave a file behind.
    Skipping the merge for a field with no labels is what keeps ``parent``
    out of it; the file itself is still rewritten so the run has one mask per
    field.
    """
    empty = tmp_path / "empty.npy"
    np.save(empty, np.zeros((16, 16), np.uint16))
    occupied = tmp_path / "occupied.npy"
    labelled = np.zeros((16, 16), np.uint16)
    labelled[2:6, 2:6] = 1
    labelled[10:14, 10:14] = 2
    np.save(occupied, labelled)
    seen = []

    for path in (empty, occupied):
        utils._process_single_fov(
            str(path), None, None, False, True, False, 0.5, 2.0, 10, 100,
            'mean', 75, 0, 0, False, 0, 100,
            lambda *args: seen.append(args), 0, 2, 'merge')

    assert np.array_equal(np.load(empty), np.zeros((16, 16), np.uint16))
    assert sorted(np.unique(np.load(occupied)).tolist()) == [0, 1, 2]
    assert len(seen) == 2


# ===========================================================================
# the crop index and the measurement database
# ===========================================================================

def test_a_crop_mode_with_no_object_id_column_is_refused(tmp_path):
    """An unregistered crop mode cannot be written into ``png_list``.

    ``_map_wells_png`` always returns an object id, so the column list has to
    carry one. When a role was missing from ``PNG_OBJECT_ID_COLUMNS`` the
    frame and the column list disagreed by exactly one column -- and the PNGs
    were already on disk while nothing registered them.
    """
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    paths = [os.path.join(root, "data", "cell_png", f"plate1_A01_1_{i}.png")
             for i in (1, 2)]

    utils.filepaths_to_database(paths, {"timelapse": False}, root, "cell")

    with sqlite3.connect(os.path.join(root, "measurements",
                                      "measurements.db")) as conn:
        registered = pd.read_sql("SELECT * FROM png_list", conn)
    assert list(registered["cell_id"]) == ["o1", "o2"]

    assert "unregistered" not in utils.PNG_OBJECT_ID_COLUMNS
    with pytest.raises(ValueError, match="same length"):
        utils.filepaths_to_database(paths, {"timelapse": False}, root,
                                    "unregistered")


def test_a_widening_that_adds_no_column_stops_instead_of_spinning(tmp_path,
                                                                  monkeypatch):
    """The repair loop is bounded, and the original error is what propagates.

    ``_append_frame`` widens the table when the insert names a column the
    table lacks. If the widening reports nothing added -- another worker won
    the race, or the column is not the one the message named -- retrying for
    ever would hang the field. It retries a fixed number of times and then
    raises what SQLite said.
    """
    attempts = []

    def refuse(conn, table, frame):
        attempts.append(table)
        raise sqlite3.OperationalError(
            f"table {table} has no column named cell_id")

    monkeypatch.setattr(utils, "_insert_frame", refuse)
    monkeypatch.setattr(utils, "_widen_table_for",
                        lambda conn, table, frame: [])
    conn = sqlite3.connect(str(tmp_path / "measurements.db"))
    try:
        with pytest.raises(sqlite3.OperationalError,
                           match="has no column named cell_id"):
            utils._append_frame(conn, "cell", pd.DataFrame({"a": [1]}))
    finally:
        conn.close()

    assert len(attempts) == utils.DB_APPEND_REPAIRS


# ===========================================================================
# the torchvision wrappers
# ===========================================================================

def _headless_backbone():
    """A backbone with no ``fc``/``classifier``/``head`` of any spelling."""
    import torch.nn as nn

    class Headless(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1),
                                          nn.AdaptiveAvgPool2d(1))

        def forward(self, x):
            return self.features(x)

    return Headless()


def test_a_backbone_with_no_head_keeps_its_own_output(monkeypatch):
    """Head removal normalises what it recognises and leaves the rest alone.

    A backbone whose head has none of the five names is not an error: the
    feature dimension is inferred from a real forward pass, and the spaCR
    classifier is attached to whatever came out.
    """
    import torch
    from torchvision import models

    monkeypatch.setattr(models, "headless_net",
                        lambda **kwargs: _headless_backbone(), raising=False)

    model = utils.TorchModel(model_name="headless_net", pretrained=False,
                             num_classes=3)

    assert model.num_ftrs == 4
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 3)


def test_the_v2_wrapper_also_survives_a_backbone_with_no_head(monkeypatch):
    """The same, through the second wrapper, which strips fewer names."""
    import torch
    from torchvision import models

    monkeypatch.setattr(models, "headless_net",
                        lambda **kwargs: _headless_backbone(), raising=False)

    model = utils.TorchModel_v2(model_name="headless_net", pretrained=False,
                                num_classes=2)

    assert model.num_ftrs == 4
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_an_empty_maxvit_classifier_is_left_empty(monkeypatch):
    """MaxViT keeps all but its last linear block -- if it has one.

    The slice ``seq[:-1]`` on an empty classifier would silently produce an
    empty Sequential either way; taking it only when there is something to
    drop is what keeps the special case honest for a stripped backbone.
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    class EmptyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1),
                                          nn.AdaptiveAvgPool2d(1))
            self.classifier = nn.Sequential()

        def forward(self, x):
            return self.classifier(self.features(x))

    monkeypatch.setattr(models, "maxvit_t", lambda **kwargs: EmptyClassifier())

    model = utils.TorchModel(model_name="maxvit_t", pretrained=False,
                             num_classes=2)

    assert list(model.base_model.classifier.children()) == []
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 2)


def test_a_weights_enum_without_default_falls_back_to_the_legacy_flag(
        monkeypatch):
    """Older torchvision releases have no ``DEFAULT`` on the enum.

    The search over ``dir(models)`` matches on name, so an attribute that
    merely looks like a weights enum must not be handed to the constructor as
    one; the legacy ``pretrained=`` call is the fallback.
    """
    from torchvision import models

    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return _headless_backbone()

    class Antique:
        """A weights enum from before ``DEFAULT`` existed."""

    monkeypatch.setattr(models, "Antique_Weights", Antique, raising=False)
    monkeypatch.setattr(models, "antique", factory, raising=False)

    model = utils.TorchModel(model_name="antique", pretrained=True,
                             num_classes=2)

    assert model._get_weight_choice() is None
    assert seen == {"pretrained": True}


def test_the_best_checkpoint_is_ranked_by_role_then_accuracy(tmp_path):
    """A checkpoint with no accuracy is ranked, not crashed on.

    ``metrics['accuracy']`` is absent from a run stopped before its first
    validation pass, and ``artifact_role`` carries values other than the two
    that promote. Neither may make the whole directory unreadable -- the
    selection still has to name a file.
    """
    import torch

    milestone = tmp_path / "model_epoch_3.pth"
    torch.save({"artifact_role": "milestone",
                "metrics": {"accuracy": 0.91},
                "training_state": {"epoch": 3}}, milestone)
    unscored = tmp_path / "model_epoch_7.pth"
    torch.save({"artifact_role": "checkpoint",
                "metrics": {"accuracy": None},
                "training_state": {"epoch": 7}}, unscored)

    assert utils.pick_best_model(str(tmp_path)) == str(milestone)


def test_augmenting_into_a_folder_that_exists_writes_into_it(tmp_path,
                                                             monkeypatch):
    """The destination is usually the folder of an earlier augmentation run."""
    import cv2

    monkeypatch.setattr(utils, "cpu_count", lambda: 2)
    source = tmp_path / "src"
    source.mkdir()
    destination = tmp_path / "dst"
    destination.mkdir()
    (destination / "already_here.txt").write_text("kept", encoding="utf-8")
    image = np.zeros((8, 8, 3), np.uint8)
    image[2:6, 2:4] = 255
    cv2.imwrite(str(source / "cell.png"), image)

    utils.augment_images([str(source / "cell.png")], str(destination))

    written = sorted(p.name for p in destination.iterdir())
    assert "already_here.txt" in written
    assert [name for name in written if name.startswith("cell_")] == [
        "cell_flip_hor.png", "cell_flip_ver.png", "cell_original.png",
        "cell_rot_180.png", "cell_rot_270.png", "cell_rot_90.png"]


# ===========================================================================
# reduction, clustering and the figures
# ===========================================================================

def test_a_gpu_reduction_without_cuml_refuses_rather_than_falling_back():
    """A GPU run that quietly used the CPU would misreport what it measured.

    ``prefer_gpu`` also decides what is forwarded to the constructor: cuML
    releases disagree about ``n_jobs``, so it is a CPU-only argument.
    """
    pytest.importorskip("sklearn")
    if importlib.util.find_spec("cuml") is not None:
        pytest.skip("cuML is installed; the CPU refusal cannot be reached")
    values = np.random.default_rng(0).random((20, 4))

    with pytest.raises(RuntimeError, match="GPU was requested"):
        utils.reduction_and_clustering(
            values, n_neighbors=5, min_dist=0.1, metric='euclidean', eps=0.5,
            min_samples=3, clustering='kmeans', reduction_method='tsne',
            prefer_gpu=True)


def test_a_spectral_embedding_with_a_dense_affinity_takes_no_neighbours():
    """``n_neighbors`` is meaningless for an RBF affinity and is not sent.

    Forwarding it anyway is not inert: scikit-learn builds the neighbourhood
    graph from it, so the option would silently change a fit that does not
    use a graph at all.
    """
    pytest.importorskip("sklearn")
    values = np.random.default_rng(1).random((24, 3))

    embedding, labels, reducer = utils.reduction_and_clustering(
        values, n_neighbors=5, min_dist=0.1, metric='euclidean', eps=0.5,
        min_samples=2, clustering='kmeans', reduction_method='spectral',
        reducer_options={'affinity': 'rbf'})

    assert embedding.shape == (24, 2)
    assert reducer.affinity == 'rbf'
    assert reducer.get_params().get('n_neighbors') is None
    assert set(np.unique(labels)) == {0, 1}


def test_outlines_can_be_computed_and_not_drawn():
    """``plot_outlines`` is what draws, ``smooth_lines`` is only the shape.

    Points-only figures are the common case for a crowded embedding; the hull
    is still computed there, and computing it is what may fail, so the two
    settings are separate.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    embedding = np.vstack([rng.normal(0, 1, (30, 2)),
                           rng.normal(8, 1, (30, 2))])
    labels = np.array([0] * 30 + [1] * 30)
    colors = [(0.2, 0.4, 0.6), (0.6, 0.2, 0.4)]
    centers = [(0.0, 0.0), (8.0, 8.0)]

    figure, axes = plt.subplots(1, 2)
    try:
        utils.plot_clusters(axes[0], embedding, labels, colors, centers,
                            plot_outlines=False, plot_points=True,
                            smooth_lines=True)
        utils.plot_clusters(axes[1], embedding, labels, colors, centers,
                            plot_outlines=True, plot_points=True,
                            smooth_lines=True)

        assert list(axes[0].lines) == []
        assert len(axes[1].lines) == 2
    finally:
        plt.close(figure)


def test_a_string_cluster_label_only_prints_when_asked(capsys):
    """The index is a debugging aid, and a grid has one line per cluster."""
    import matplotlib.pyplot as plt

    images = {"alpha": [np.zeros((4, 4), np.uint8)],
              "beta": [np.ones((4, 4), np.uint8)]}
    colors = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]

    figure = utils.plot_grid(images, colors, 4, False, verbose=False)
    plt.close(figure)
    assert "Lable:" not in capsys.readouterr().out

    figure = utils.plot_grid(images, colors, 4, False, verbose=True)
    plt.close(figure)
    printed = capsys.readouterr().out
    assert "Lable: alpha index: 0" in printed
    assert "Lable: beta index: 1" in printed


def test_a_frame_that_is_neither_2d_nor_3d_gets_no_shape_of_its_own():
    """The target shape is chosen per frame, and only for 2-D and 3-D frames.

    A 4-D frame falls through both arms with whatever shape the previous
    frame set, and scikit-image refuses it. That refusal is the point: the
    alternative is a resized stack whose frames are not the same size as each
    other, which nothing downstream checks.
    """
    plane = np.zeros((8, 8), np.uint8)
    volume = np.zeros((8, 8, 2, 2), np.uint8)
    label = np.zeros((8, 8), np.uint16)

    images, labels = utils.resize_images_and_labels(
        [plane], [label], 4, 4, show_example=False)
    assert images[0].shape == (4, 4)
    assert labels[0].shape == (4, 4)

    with pytest.raises(ValueError):
        utils.resize_images_and_labels([plane, volume], [label, label], 4, 4,
                                       show_example=False)
    with pytest.raises(ValueError):
        utils.resize_images_and_labels([plane, volume], None, 4, 4,
                                       show_example=False)


def test_asking_for_an_example_of_nothing_draws_nothing(monkeypatch):
    """With neither images nor labels there is no pair to show.

    ``plot_resize`` is called with four arrays; reaching it with two empty
    lists drew an empty figure per field of a run that had asked for nothing.
    """
    from spacr import plot

    drawn = []
    monkeypatch.setattr(plot, "plot_resize",
                        lambda *args, **kwargs: drawn.append(args))

    images, labels = utils.resize_images_and_labels(None, None, 4, 4,
                                                    show_example=True)

    assert (images, labels) == ([], [])
    assert drawn == []

    utils.resize_images_and_labels([np.zeros((8, 8), np.uint8)], None, 4, 4,
                                   show_example=True)
    assert len(drawn) == 1


# ===========================================================================
# filtering objects and reading the database
# ===========================================================================

def test_an_intensity_bound_that_is_not_a_whole_number_is_not_a_bound():
    """The bounds are read as integers, so a float bound filters nothing.

    This is the settings contract, not an accident of ``isinstance``: the
    intensity range comes from a spin box that yields whole numbers, and a
    ``None`` half means "no bound on this side".
    """
    frame = pd.DataFrame({
        "cell_area": [10, 20, 30],
        "cell_channel_1_mean_intensity": [5, 50, 500],
    })

    unfiltered = utils._object_filter(frame, "cell", None, [None, None],
                                      [0, 1], 1)
    assert len(unfiltered) == 3

    filtered = utils._object_filter(frame, "cell", None, [10, 100], [0, 1], 1)
    assert list(filtered["cell_channel_1_mean_intensity"]) == [50]


def test_a_metadata_filter_that_is_not_a_string_or_a_list_selects_nothing(
        tmp_path):
    """The two accepted spellings are a string and a list of strings.

    A tuple reads like a list and is not one: no statement is executed, so
    the reader returns an empty index rather than every row. Silently
    returning everything would build an image list from the whole plate for a
    filter the caller believed had narrowed it.
    """
    db_path = str(tmp_path / "measurements.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE png_list (png_path TEXT)")
        conn.executemany("INSERT INTO png_list VALUES (?)",
                         [("/data/plate1_A01_1_1.png",),
                          ("/data/plate1_B02_1_1.png",)])

    assert utils.generate_path_list_from_db(db_path, ["A01"]) == [
        "/data/plate1_A01_1_1.png"]
    assert utils.generate_path_list_from_db(db_path, ("A01",)) == []


def test_correcting_paths_accepts_a_frame_or_a_list_and_nothing_else():
    """A caller that passes neither gets an error naming the missing name."""
    frame = pd.DataFrame({"png_path": ["/old/data/cell_png/a.png"]})

    corrected, _ = utils.correct_paths(frame, "/new")
    assert list(corrected["png_path"]) == ["/new/data/cell_png/a.png"]

    with pytest.raises(UnboundLocalError, match="image_paths"):
        utils.correct_paths({"png_path": ["/old/data/cell_png/a.png"]}, "/new")


def test_morphology_is_offered_only_when_a_morphology_column_is_there():
    """The filter list is built from the columns, not from a fixed menu."""
    intensity_only = utils._available_feature_filters(
        ["cell_channel_1_mean_intensity", "cell_channel_2_mean_intensity"])
    assert intensity_only == ["channel_1", "channel_2"]

    with_shape = utils._available_feature_filters(
        ["cell_channel_1_mean_intensity", "cell_perimeter"])
    assert with_shape == ["channel_1", "morphology"]


# ===========================================================================
# the mask mergers
# ===========================================================================

def test_a_cell_holding_two_objects_is_cleared_with_its_nucleus():
    """A stack without a pathogen channel loses no other channel.

    ``pathogen_dim`` is ``None`` for every run segmented without a pathogen
    mask, and the pathogen resolution inside the loop is skipped for it -- an
    unconditional lookup there indexed the whole stack, because a ``None``
    dimension is ``np.newaxis`` and widens the view instead of raising.
    """
    stack = np.zeros((12, 12, 3), np.uint16)
    stack[2:8, 2:8, 0] = 1              # one cell
    stack[9:11, 9:11, 0] = 2            # a second, with one object
    stack[3:5, 3:5, 1] = 1              # two objects inside the first cell
    stack[6:7, 6:7, 1] = 2
    stack[9:10, 9:10, 1] = 3            # one object inside the second
    stack[3:6, 3:6, 2] = 7              # the first cell's nucleus
    stack[9:11, 9:11, 2] = 8            # the second cell's nucleus

    cleared = utils._remove_multiobject_cells(
        stack.copy(), mask_dim=0, cell_dim=None, nucleus_dim=2,
        pathogen_dim=None, object_dim=1)

    assert sorted(np.unique(cleared[:, :, 0]).tolist()) == [0, 2]
    assert sorted(np.unique(cleared[:, :, 2]).tolist()) == [0, 8]
    assert sorted(np.unique(cleared[:, :, 1]).tolist()) == [0, 1, 2, 3]


def test_a_nucleus_free_cell_joins_a_neighbour_that_has_one():
    """Three states in one field: merged, left alone, and not a nucleus.

    A nucleus lying entirely on background belongs to no cell and must not
    mark one; a cell with no nucleus-bearing neighbour has nowhere to go and
    keeps its own identity; and a cell touching the image border is walked
    like any other, without stepping outside the array.
    """
    cells = np.zeros((12, 12), np.uint16)
    cells[0:4, 0:4] = 1                 # touches the top-left border
    cells[0:4, 4:8] = 2                 # adjacent to cell 1, no nucleus
    cells[8:12, 8:12] = 3               # isolated, no nucleus
    nuclei = np.zeros((12, 12), np.uint16)
    nuclei[1:3, 1:3] = 1                # inside cell 1
    nuclei[5:7, 0:2] = 2                # on background, inside no cell

    merged = utils._merge_cells_without_nucleus(cells, nuclei)

    assert merged[0, 5] == 1, "the nucleus-free neighbour joined cell 1"
    assert sorted(np.unique(merged).tolist()) == [0, 1, 3]
    assert merged[10, 10] == 3


def test_a_parasite_in_one_cell_merges_nothing():
    """Merging is for a parasite the segmentation split across two cells.

    The overlap threshold is a percentage of the parasite, so a cell holding
    a sliver of one is not evidence that the two cells are one cell.
    """
    cells = np.zeros((12, 20), np.uint16)
    cells[2:10, 1:9] = 1
    cells[2:10, 9:17] = 2               # touching, so a merge would show
    parasites = np.zeros((12, 20), np.uint16)
    parasites[4:6, 3:7] = 1             # wholly inside the left cell
    nuclei = np.zeros((12, 20), np.uint16)
    nuclei[3:5, 2:4] = 1
    nuclei[3:5, 12:14] = 2

    kept = utils._merge_cells_based_on_parasite_overlap(
        parasites, cells.copy(), nuclei, None)

    assert len(np.unique(kept)) - 1 == 2, "two cells, not merged"


def test_a_parasite_across_two_cells_merges_them_once():
    """The cell holding 5% of the parasite does not trigger a second merge.

    Every cell the parasite touches is tested, and the merge is done from the
    first label; a second pass over an already-merged pair would relabel
    cells that are no longer there.
    """
    cells = np.zeros((12, 20), np.uint16)
    cells[2:10, 1:9] = 1
    cells[2:10, 9:17] = 2
    parasites = np.zeros((12, 20), np.uint16)
    parasites[5, 1:9] = 1               # 19 of the parasite's 20 pixels are
    parasites[6, 1:9] = 1               # in the left cell, and exactly one
    parasites[7, 1:4] = 1               # -- 5%, which is not above 5% --
    parasites[5, 9] = 1                 # is in the right one
    nuclei = np.zeros((12, 20), np.uint16)
    nuclei[3:5, 2:4] = 1
    nuclei[3:5, 12:14] = 2
    cell_mask = cells.copy()

    merged = utils._merge_cells_based_on_parasite_overlap(
        parasites, cell_mask, nuclei, None, overlap_threshold=5)

    assert len(np.unique(merged)) - 1 == 1, "the split parasite joined them"


def test_adjusting_cell_masks_with_a_matching_organelle_folder(tmp_path,
                                                               capsys):
    """A matching organelle count is not a thing to warn about.

    The warning exists for a folder that is missing files, and printing it on
    every ordinary run would teach the user to ignore it.
    """
    folders = {}
    for name in ("parasite", "cell", "nuclei", "organelle"):
        folder = tmp_path / name
        folder.mkdir()
        folders[name] = str(folder)
    for stem in ("f1.npy", "f2.npy"):
        cells = np.zeros((12, 20), np.uint16)
        cells[2:10, 1:9] = 1
        cells[2:10, 11:19] = 2
        np.save(os.path.join(folders["cell"], stem), cells)
        parasites = np.zeros((12, 20), np.uint16)
        parasites[4:6, 3:7] = 1
        np.save(os.path.join(folders["parasite"], stem), parasites)
        nuclei = np.zeros((12, 20), np.uint16)
        nuclei[3:5, 2:4] = 1
        nuclei[3:5, 12:14] = 2
        np.save(os.path.join(folders["nuclei"], stem), nuclei)
        np.save(os.path.join(folders["organelle"], stem),
                np.zeros((12, 20), np.uint16))

    utils.adjust_cell_masks(folders["parasite"], folders["cell"],
                            folders["nuclei"], folders["organelle"], n_jobs=1)

    assert "does not match other masks" not in capsys.readouterr().out
    assert len(np.unique(np.load(
        os.path.join(folders["cell"], "f1.npy")))) - 1 == 2


# ===========================================================================
# the folders a run leaves behind
# ===========================================================================

def test_models_already_beside_the_package_are_downloaded_once(tmp_path,
                                                               monkeypatch):
    """An existing but empty folder is not a completed download.

    The folder is created by the first attempt, so treating "it exists" as
    "the models are there" left an empty folder behind and every later call
    returned it without downloading anything.
    """
    package = tmp_path / "spacr"
    local = package / "resources" / "models"
    local.mkdir(parents=True)
    monkeypatch.setattr(utils, "spacr_path", str(package / "__init__.py"))
    listed = []

    def list_files(repo_id, repo_type=None):
        listed.append(repo_id)
        return ["cellpose.pth"]

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"weights"

    monkeypatch.setattr(utils, "list_repo_files", list_files)
    monkeypatch.setattr(utils.requests, "get",
                        lambda url, stream=False: Response())

    first = utils.download_models()

    assert first == str(local)
    assert (local / "cellpose.pth").read_bytes() == b"weights"
    assert listed == ["einarolafsson/models"]

    assert utils.download_models() == str(local)
    assert listed == ["einarolafsson/models"], "a full folder is not re-fetched"


def test_cleanup_removes_the_folders_that_are_there(tmp_path):
    """``stack/`` may already be gone, and ``masks/`` may never have existed.

    Cleanup runs after a merged run, sometimes twice, and a missing folder is
    the ordinary state of the second run -- not a reason to stop before the
    numbered channel folders.
    """
    root = tmp_path / "run"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "field1.npy", np.zeros((4, 4), np.uint16))
    (root / "masks").mkdir()
    np.save(root / "masks" / "field1.npy", np.zeros((4, 4), np.uint16))
    (root / "1").mkdir()

    deleted = utils.cleanup_pipeline_folders(str(root))

    assert str(root / "masks") in deleted
    assert str(root / "stack") not in deleted
    assert str(root / "1") in deleted
    assert (root / "merged" / "field1.npy").exists()


# ===========================================================================
# grad-CAM
# ===========================================================================

def test_a_target_layer_with_no_gradient_cannot_produce_a_map():
    """``retain_grad`` is only meaningful where a gradient will arrive.

    PyTorch populates ``.grad`` on leaf tensors only, so the hook asks the
    target layer's output to keep its own. A frozen target layer produces an
    output with no gradient at all, and the map that would come out of it
    would be an array of zeros presented as an explanation -- so it fails
    loudly instead.
    """
    import torch
    import torch.nn as nn

    def build():
        return nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(2, 2, 3, padding=1),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())

    x = torch.randn(1, 3, 16, 16)

    cam = utils.GradCAM(build(), target_layers=["0"], use_cuda=False)(x)
    assert cam.shape == (16, 16)
    assert 0.0 <= float(cam.min()) and float(cam.max()) == pytest.approx(1.0)

    frozen = build()
    for parameter in frozen[0].parameters():
        parameter.requires_grad_(False)

    with pytest.raises(AttributeError, match="NoneType"):
        utils.GradCAM(frozen, target_layers=["0"], use_cuda=False)(x)


# ===========================================================================
# the regression
# ===========================================================================

def test_the_reported_interaction_is_the_strongest_one_for_the_gene():
    """One gene has one effect, and it is its largest, not its last.

    The per-gene maximum is taken over every gRNA interaction term, so a gene
    whose second guide has a smaller effect keeps the first guide's number --
    and the p-value that belongs to it, not to whichever term came last.
    """
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    rows = []
    for gene in ("g1", "g2"):
        for grna in ("s1", "s2", "s3"):
            for replicate in range(6):
                boost = 0.8 if (gene == "g2" and grna == "s2") else 0.0
                boost += 0.2 if (gene == "g2" and grna == "s3") else 0.0
                rows.append({
                    "gene": gene, "grna": grna,
                    "plate": f"p{replicate % 2 + 1}",
                    "row": f"r{replicate % 3 + 1}",
                    "column": f"c{replicate % 2 + 1}",
                    "pred": rng.normal(0.5 if gene == "g1" else 1.0, 0.1)
                            + boost})

    max_effects, max_pvalues, model, table = utils.MLR(pd.DataFrame(rows),
                                                       refine_model=False)

    interactions = {key: value for key, value in model.params.items()
                    if "gene[T." in key and ":grna[T." in key}
    assert len(interactions) == 2, "two guides interact with the same gene"
    strongest = max(interactions, key=lambda key: abs(interactions[key]))
    assert set(max_effects) == {"g2"}
    assert max_effects["g2"] == pytest.approx(interactions[strongest])
    assert max_pvalues["g2"] == pytest.approx(model.pvalues[strongest])
    assert table.loc["g2", "effect"] == pytest.approx(interactions[strongest])


# ===========================================================================
# naming a feature selection
# ===========================================================================

def test_a_channel_and_a_filter_are_named_in_the_order_given():
    """A mixed selection joins one name per member, ints spelled as channels."""
    assert utils.feature_folder_name([1, "mean_intensity"]) == (
        "channel_1_mean_intensity")
    assert utils.feature_folder_name([1, 2]) == "channels_1_2"
    assert utils.feature_folder_name("mean intensity!") == "mean_intensity"
