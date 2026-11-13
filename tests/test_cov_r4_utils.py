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

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import utils


# ===========================================================================
# merging and filtering one field
# ===========================================================================

# ===========================================================================
# the crop index and the measurement database
# ===========================================================================

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


# ===========================================================================
# reduction, clustering and the figures
# ===========================================================================

# ===========================================================================
# filtering objects and reading the database
# ===========================================================================

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


# ===========================================================================
# the folders a run leaves behind
# ===========================================================================

# ===========================================================================
# grad-CAM
# ===========================================================================

# ===========================================================================
# the regression
# ===========================================================================

# ===========================================================================
# naming a feature selection
# ===========================================================================

def test_a_channel_and_a_filter_are_named_in_the_order_given():
    """A mixed selection joins one name per member, ints spelled as channels."""
    assert utils.feature_folder_name([1, "mean_intensity"]) == (
        "channel_1_mean_intensity")
    assert utils.feature_folder_name([1, 2]) == "channels_1_2"
    assert utils.feature_folder_name("mean intensity!") == "mean_intensity"
