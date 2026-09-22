"""Every organelle slot is preprocessed and measured the way the first one is.

Items 364 and 76. The maintainer, asked on 2026-09-19 whether io.py's
per-channel preprocessing should loop over every organelle slot instead of
handling the first one only, answered "Yes, all slots".

Two things were measured on origin/nightly (92ea1ba00) before this change:

* ``spacr.io._normalize_img_batch`` gave a nucleus, cell, pathogen or
  first-organelle channel that object's own background floor and
  signal-to-noise anchor. It gave every other organelle slot's channel the
  generic ``background`` / ``Signal_to_noise`` pair and ignored
  ``organelleb_background`` and ``organelleb_signal_to_noise``, which the
  settings declare for every slot.
* A second slot already reached the database: raw TIFFs with two organelle
  channels, through the Mask pipeline and Measure, wrote an ``organelleb``
  table whose ``cell_id`` named the containing cell. What Measure received was
  wrong in another way. ``spacr.crops.reconcile_merged_mask_dims`` set
  ``<slot>_mask_dim`` for all 702 slots, ``None`` for the 700 the run never
  had. The settings factory then counted all 702 as declared, and the saved
  settings carried 2,100 rows for the 700 slots the run never had: 2,170
  settings rows where the same run now saves 70.

The end-to-end test stubs only the Cellpose forward pass for the cell and
nucleus. Preprocessing, the organelle segmenter for both slots, the merge
and Measure are the real ones.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _normalise(settings, stack):
    from spacr.io import _normalize_img_batch

    base = {"lower_percentile": 2}
    base.update(settings)
    return _normalize_img_batch(
        stack=stack.copy(), channels=range(stack.shape[-1]),
        save_dtype=np.float32, settings=base)


def _ramp_stack():
    """One field, five channels; channel 4 holds the values 1..4096."""
    ramp = np.arange(1, 4097, dtype=np.float32).reshape(64, 64)
    stack = np.zeros((1, 64, 64, 5), dtype=np.float32)
    for channel in range(5):
        stack[0, :, :, channel] = ramp
    return stack


class TestASecondSlotIsNormalisedWithItsOwnValues:
    """The slot's background floor, anchor and switch reach its channel."""

    OWN = {"background": 300, "signal_to_noise": 2, "remove": True}

    def _slot(self, role):
        return {
            f"{role}_channel": 4,
            f"{role}_background": self.OWN["background"],
            f"{role}_signal_to_noise": self.OWN["signal_to_noise"],
            f"remove_background_{role}": self.OWN["remove"],
        }

    def test_the_second_slot_is_treated_exactly_as_the_first(self):
        stack = _ramp_stack()
        as_first = _normalise(self._slot("organelle"), stack)
        as_second = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb")}, stack)
        np.testing.assert_array_equal(as_second[..., 4], as_first[..., 4])

    def test_its_own_values_change_the_result(self):
        stack = _ramp_stack()
        generic = _normalise({"organelle_channel": 3}, stack)
        as_second = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb")}, stack)
        assert not np.array_equal(as_second[..., 4], generic[..., 4])
        below_floor = stack[0, :, :, 4] < self.OWN["background"]
        assert np.all(as_second[0][below_floor, 4] == 0)
        assert np.any(generic[0][below_floor, 4] > 0)

    def test_a_slot_the_run_does_not_enable_changes_nothing(self):
        stack = _ramp_stack()
        generic = _normalise({"organelle_channel": 3}, stack)
        disabled = _normalise(
            {"organelle_channel": 3, **self._slot("organelleb"),
             "organelleb_channel": None}, stack)
        np.testing.assert_array_equal(disabled, generic)

    def test_the_twentieth_slot_is_reached_too(self):
        from spacr.object_roles import ORGANELLE_ROLES

        role = ORGANELLE_ROLES[19]
        stack = _ramp_stack()
        as_first = _normalise(self._slot("organelle"), stack)
        as_twentieth = _normalise(
            {"organelle_channel": 3, **self._slot(role)}, stack)
        np.testing.assert_array_equal(as_twentieth[..., 4], as_first[..., 4])


class TestEverySlotHasItsOwnBackgroundSwitch:
    """``remove_background_<slot>`` is a real setting for slots two onward.

    Item 364, 2026-09-21. Slots past the first had a background floor and a
    signal-to-noise anchor and no switch to apply the floor with, so a noisy
    second organelle stain could not be clipped without clipping the first.
    The maintainer chose to extend the existing ``remove_background_<object>``
    pattern.
    """

    def test_it_is_declared_typed_and_explained_for_every_slot(self):
        from spacr.organelle_types import ALL_ORGANELLE_ROLES
        from spacr.settings import expected_types, tooltips

        for role in ALL_ORGANELLE_ROLES:
            key = f"remove_background_{role}"
            assert expected_types[key] is bool, key
            assert f"{role}_background" in tooltips[key], key

    def test_the_factory_gives_each_enabled_slot_its_own_switch_off(self):
        from spacr.settings import (
            set_default_settings_preprocess_generate_masks as factory)

        settings = factory({"organelle_channel": 3, "organelleb_channel": 4,
                            "number_of_organelles": 2})
        assert settings["remove_background_organelle"] is False
        assert settings["remove_background_organelleb"] is False
        assert "remove_background_organellec" not in settings

    def test_a_value_the_user_set_survives_the_factory(self):
        from spacr.settings import (
            set_default_settings_preprocess_generate_masks as factory)

        settings = factory({"organelle_channel": 3, "organelleb_channel": 4,
                            "number_of_organelles": 2,
                            "remove_background_organelleb": True})
        assert settings["remove_background_organelleb"] is True

    def test_turning_on_slot_two_clips_slot_two_and_not_slot_one(self):
        stack = _ramp_stack()
        both = {"organelle_channel": 3, "organelleb_channel": 4,
                "organelle_background": 300, "organelleb_background": 300}
        off = _normalise(both, stack)
        on = _normalise({**both, "remove_background_organelleb": True}, stack)
        below = stack[0, :, :, 4] < 300
        assert np.all(on[0][below, 4] == 0)
        assert np.any(off[0][below, 4] > 0)
        np.testing.assert_array_equal(on[..., 3], off[..., 3])

    def test_the_switch_does_not_make_a_file_look_like_it_uses_a_slot(self):
        """A switch alone is not a slot in use, as its name does not start
        with the slot's prefix; counting it would conjure an organelle."""
        from spacr.organelle_types import organelle_count

        assert organelle_count({"remove_background_organellec": True}) == 0

    def test_the_panel_hides_it_with_its_slot(self):
        from spacr.qt.screens.settings_model import object_of_setting

        assert object_of_setting("remove_background_organelleb") == (
            "organelleb")
        assert object_of_setting("remove_background") is None


class TestWhatASlotTakesWhenItsOwnValueIsMissingOrShared:
    """Which of the three values a slot's channel ends up with.

    Added on review of the first pass, which recorded that "with the shipped
    defaults no channel is normalised differently". That holds only while
    every object has a channel to itself, and the first pass read a present
    but empty slot value as a value.
    """

    def test_a_slot_sharing_the_pathogens_channel_wins_what_it_carries(self):
        """Read last, a slot wins each value it carries, as the first did.

        The floor and the anchor become the slot's. The switch only when the
        slot carries one: this dict names no ``remove_background_organelleb``,
        so the one the pathogen turned on stays on for that channel. Every
        slot has declared its own switch since 2026-09-21, so a settings dict
        that went through the factory carries it and the slot's wins.

        Measured on the worktree: pathogen channel 2 at 200/20 sharing a
        channel with ``organelleb`` at the factory's 100/10 normalises at
        100/1000, where a pathogen holding that channel alone uses 200/4000.
        """
        stack = _ramp_stack()
        slot = {"organelle_channel": 3, "organelleb_channel": 4,
                "organelleb_background": 100,
                "organelleb_signal_to_noise": 10}
        pathogen = {"pathogen_channel": 4, "pathogen_background": 300,
                    "pathogen_signal_to_noise": 2,
                    "remove_background_pathogen": True}
        shared = _normalise({**pathogen, **slot}, stack)
        pathogen_alone = _normalise({**pathogen, "organelle_channel": 3},
                                    stack)
        slots_values_switch_left_on = _normalise(
            {**slot, "remove_background_organelleb": True}, stack)
        np.testing.assert_array_equal(
            shared[..., 4], slots_values_switch_left_on[..., 4])
        assert not np.array_equal(shared[..., 4], pathogen_alone[..., 4])

    def test_a_slot_value_left_empty_reads_as_one_the_slot_does_not_carry(
            self):
        """An empty box in the panel falls back; it does not stop the run.

        The first pass read ``settings.get(key, fallback)``, so a key present
        and empty won, and the run died on ``None * None``.
        """
        stack = _ramp_stack()
        generic = _normalise({"organelle_channel": 3}, stack)
        empty = _normalise(
            {"organelle_channel": 3, "organelleb_channel": 4,
             "organelleb_background": None,
             "organelleb_signal_to_noise": None,
             "remove_background_organelleb": None}, stack)
        np.testing.assert_array_equal(empty[..., 4], generic[..., 4])

    def test_the_first_slots_empty_value_falls_back_too(self):
        """The same hazard the first slot carried on origin/nightly."""
        stack = _ramp_stack()
        generic = _normalise({}, stack)
        empty = _normalise(
            {"organelle_channel": 4, "organelle_background": None}, stack)
        np.testing.assert_array_equal(empty[..., 4], generic[..., 4])

    def test_a_floor_of_zero_is_a_value_and_not_an_absence(self):
        """Zero is a floor a user can mean, so it must not read as empty."""
        stack = _ramp_stack()
        below = stack[0, :, :, 4] < 100
        zero_floor = _normalise(
            {"organelle_channel": 4, "organelle_background": 0,
             "remove_background_organelle": True}, stack)
        generic_floor = _normalise(
            {"organelle_channel": 4, "remove_background_organelle": True},
            stack)
        assert np.any(zero_floor[0][below, 4] > 0)
        assert np.all(generic_floor[0][below, 4] == 0)


def test_a_crop_cut_on_demand_takes_a_second_slots_plane_from_the_run(
        tmp_path):
    """``open_crop_source`` forwards every slot's mask plane, not the first's.

    On origin/nightly both lists that forward a run's crop-shaping settings
    -- ``spacr.io.CROP_SHAPE_KEYS`` and the one inside
    ``spacr.crops.resolve_crop_source`` -- named ``organelle_mask_dim`` and
    no other slot, so a run could not say where Organelle 2's plane is.
    """
    from spacr.io import open_crop_source

    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "plate1_A01_1.npy", np.zeros((8, 8, 7), np.uint16))
    source = open_crop_source(
        {"src": str(tmp_path / "plate1"), "crop_source": "merged",
         "cell_mask_dim": 2, "organelle_mask_dim": 5,
         "organelleb_mask_dim": 6},
        object_type="organelleb", verbose=False)
    assert source is not None and source.kind == "merged"
    assert source.spec.mask_dims["organelleb"] == 6
    assert source.spec.mask_dims["organelle"] == 5


SIZE = 96
LEFT_CELL = (slice(8, 88), slice(4, 44))
RIGHT_CELL = (slice(8, 88), slice(52, 92))
NUCLEI = ((slice(40, 56), slice(16, 32)), (slice(40, 56), slice(64, 80)))
SPOTS = {
    "organelle": [((14, 10), 5), ((70, 12), 5), ((74, 34), 5),
                  ((16, 60), 5)],
    "organelleb": [((24, 20), 6), ((70, 60), 6), ((20, 80), 6)],
}


def _channels():
    """Nucleus, cell, organelle 1 and organelle 2 images of one field."""
    nucleus = np.full((SIZE, SIZE), 50, np.uint16)
    cell = np.full((SIZE, SIZE), 50, np.uint16)
    for box in (LEFT_CELL, RIGHT_CELL):
        cell[box] = 800
    for box in NUCLEI:
        nucleus[box] = 900
    organelles = []
    for role in ("organelle", "organelleb"):
        image = np.full((SIZE, SIZE), 50, np.uint16)
        for (row, col), side in SPOTS[role]:
            image[row:row + side, col:col + side] = 3000
        organelles.append(image)
    return [nucleus, cell, *organelles]


def _label_the_painted_squares(src, settings, object_type):
    """Stand-in for the Cellpose pass: label the painted cells or nuclei."""
    from scipy import ndimage

    from spacr.io import _listdir_visible

    out = os.path.join(src, f"{object_type}_mask_stack")
    os.makedirs(out, exist_ok=True)
    position = settings[f"cellpose_{object_type}_channel"]
    for name in sorted(_listdir_visible(src)):
        if not name.endswith(".npz"):
            continue
        with np.load(os.path.join(src, name), allow_pickle=True) as archive:
            data, files = archive["data"], archive["filenames"]
        for index, filename in enumerate(files):
            labels, _count = ndimage.label(data[index, :, :, position] > 0.5)
            np.save(os.path.join(out, str(filename)), labels.astype(np.uint16))


@pytest.fixture(scope="module")
def two_organelle_plate(tmp_path_factory):
    """A plate run once through Mask and Measure with two organelle slots."""
    plate = tmp_path_factory.mktemp("two_organelles") / "plate1"
    plate.mkdir()
    for well in ("A01", "A02"):
        for channel, image in enumerate(_channels(), start=1):
            tifffile.imwrite(
                plate / f"plate1_{well}_T0001F001L01A01Z01C0{channel}.tif",
                image)
    with pytest.MonkeyPatch.context() as patch:
        _run_mask_and_measure(plate, patch)
    return plate


def _run_mask_and_measure(plate, patch):
    import spacr.measure as measure
    import spacr.object as spacr_object
    from spacr.core import preprocess_generate_masks
    from spacr.measure import measure_crop

    patch.setattr(spacr_object, "generate_cellpose_masks_sam",
                  _label_the_painted_squares)
    patch.setattr(
        measure, "_load_zernike_moments",
        lambda: (_ for _ in ()).throw(ImportError("not needed here")))
    patch.setattr(measure, "_ZERNIKE_AVAILABLE", None)

    preprocess_generate_masks({
        "src": str(plate), "metadata_type": "cellvoyager",
        "custom_regex": None, "channels": [0, 1, 2, 3],
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": None,
        "organelle_channel": 2, "organelleb_channel": 3,
        "number_of_organelles": 2,
        "organelle_method": "otsu", "organelleb_method": "otsu",
        "preprocess": True, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "save": True,
        "batch_size": 2, "randomize": False, "normalize": True,
    })
    merged = plate / "merged"
    measure_crop({
        "src": str(merged), "timelapse": False, "channels": [0, 1, 2, 3],
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "save_png": False, "save_arrays": False, "plot": False,
        "save_measurements": True, "n_jobs": 1, "verbose": False,
        "radial_dist": False, "homogeneity": False,
        "calculate_correlation": False, "experiment": "two-organelles",
    })


def _prcf(merged_name):
    well = merged_name.split("_")[1]
    return f"plate1_r1_c{int(well[1:])}_f1"


def _expected_parents(plate, role):
    """``{(prcf, object label): containing cell label}`` from the geometry.

    The cell a spot belongs to is read off the painted layout -- left or
    right half -- and the cell label at that cell's centre, not from any
    parent mapping spaCR computed.
    """
    import json

    merged = plate / "merged"
    layout = json.loads((merged / ".spacr_plane_layout.json").read_text())
    dims = layout["mask_dims"]
    expected = {}
    for name in sorted(os.listdir(merged)):
        if not name.endswith(".npy"):
            continue
        field = np.load(merged / name)
        cells = field[..., dims["cell"]].astype(int)
        left, right = cells[48, 24], cells[48, 72]
        assert left and right and left != right
        plane = field[..., dims[role]].astype(int)
        for label in np.unique(plane[plane > 0]):
            _rows, cols = np.nonzero(plane == label)
            expected[(_prcf(name), int(label))] = int(
                left if cols.mean() < SIZE / 2 else right)
    return expected


@pytest.mark.parametrize("role", ["organelle", "organelleb"])
def test_each_slot_reaches_the_database_with_its_containing_cell(
        two_organelle_plate, role):
    expected = _expected_parents(two_organelle_plate, role)
    assert len(expected) == 2 * len(SPOTS[role]), (
        f"{role}: the segmenter found {len(expected)} objects in two fields, "
        f"{len(SPOTS[role])} were painted in each")
    db = two_organelle_plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            f'SELECT prcf, object_label, cell_id FROM "{role}"').fetchall()
    measured = {(prcf, int(label)): int(cell) for prcf, label, cell in rows}
    assert measured == expected


def test_the_relationships_frame_links_both_slots_to_their_cells(
        two_organelle_plate):
    """Both slots' objects resolve to a parent cell after Measure.

    This builds the frame from the finished database. The stored
    ``relationships`` table is not checked because it is not written: Mask
    calls ``write_relationships`` before Measure has made the object tables,
    and prints a warning. That is older than these slots and true of every
    object type; item 76 records it.
    """
    from spacr.filters import build_relationships_frame

    db = two_organelle_plate / "measurements" / "measurements.db"
    frame = build_relationships_frame(str(db))
    for role in ("organelle", "organelleb"):
        rows = frame[frame["object_type"] == role]
        assert len(rows) == 2 * len(SPOTS[role])
        assert set(rows["parent_type"]) == {"cell"}
        assert rows["parent_label"].notna().all()


def test_measure_is_given_the_two_slots_the_run_has_and_no_others(
        two_organelle_plate):
    db = two_organelle_plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        keys = [row[0] for row in connection.execute(
            "SELECT setting_key FROM settings")]
    slot_dims = sorted(key for key in keys
                       if key.startswith("organelle")
                       and key.endswith("_mask_dim"))
    assert slot_dims == ["organelle_mask_dim", "organelleb_mask_dim"]
    assert not [key for key in keys
                if key.startswith("organellec")], (
        "the saved settings carry a third slot the run never had")
