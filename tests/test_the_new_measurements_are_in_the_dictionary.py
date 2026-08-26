"""The opt-in measurement families are described, and their knob is declared.

Two families are written only when a setting asks for them: the spatial
context block (``spatial_measurements``) and the corrected Manders
coefficients (``corrected_manders``). Both reach the database as ordinary
per-object columns, so both are subject to the same two rules every other
measurement obeys.

A COLUMN WITH NO ``feature_dict`` ENTRY reports as family ``unknown`` with a
null description and no unit, which is how a real measurement comes to look
like a stray column -- ``spacr.measure`` names this as one of the four ways a
new column is silently lost.

A SETTING THE PIPELINE READS BUT NOTHING DECLARES cannot be set. It has no
default, no type, no tooltip and no category, so the GUI never draws it and
``--set`` is refused by ``check_settings`` -- the knob exists in the code and
nowhere a user can reach.
"""
import pytest

from spacr import feature_dict
from spacr.measure import spatial_column_names


SPATIAL_RADIUS = 50
MANDERS_COLUMNS = (
    "cell_channel_0_channel_1_manders_m1",
    "cell_channel_0_channel_1_manders_m2",
    "cell_channel_0_channel_1_manders_overlap_coefficient",
)


def _described(column):
    entry, = feature_dict.describe_columns([column])
    return entry


@pytest.mark.parametrize("stat", spatial_column_names(SPATIAL_RADIUS))
def test_every_spatial_column_is_described(stat):
    entry = _described(f"cell_{stat}")
    assert entry.family != "unknown", f"{stat} is not in the dictionary"
    assert entry.description, f"{stat} has no description"
    assert entry.unit, f"{stat} has no unit"
    assert entry.object_type == "cell"


@pytest.mark.parametrize("column", MANDERS_COLUMNS)
def test_every_corrected_manders_column_is_described(column):
    entry = _described(column)
    assert entry.family == "correlation"
    assert entry.description
    assert entry.unit
    assert (entry.channel, entry.channel_2) == (0, 1)


def test_the_spatial_family_names_the_setting_that_writes_it():
    entry = _described(f"cell_{spatial_column_names(SPATIAL_RADIUS)[0]}")
    assert "spatial_measurements" in (entry.written_when or "")


def test_the_neighbour_radius_is_a_declared_setting():
    """The radius the column name carries has to be settable."""
    from spacr import settings as settings_module

    assert "spatial_neighbor_radius" in settings_module.expected_types
    assert settings_module.tooltips.get("spatial_neighbor_radius")
    defaults = settings_module.get_measure_crop_settings({})
    assert defaults["spatial_neighbor_radius"] == 50
    categorised = {key
                   for keys in settings_module.categories.values()
                   for key in keys}
    assert "spatial_neighbor_radius" in categorised
