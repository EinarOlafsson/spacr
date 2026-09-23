"""Secondary growth keeps primary identities and reports losses explicitly."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import mask_engine as engine


def _paired_objects(dtype=np.uint16, ids=(7, 900)):
    """Two separated circular cells with smaller primary footprints."""
    yy, xx = np.ogrid[:80, :120]
    primary = np.zeros((80, 120), dtype=dtype)
    cells = np.zeros_like(primary)
    for label, cx in zip(ids, (30, 90)):
        distance = (yy - 40) ** 2 + (xx - cx) ** 2
        cells[distance <= 18 ** 2] = label
        primary[distance <= 4 ** 2] = label
    image = np.where(cells, 100.0, 0.0).astype(np.float32)
    return image, primary, cells


@pytest.mark.parametrize('growth', ['intensity', 'distance'])
@pytest.mark.parametrize("stop,value", [
    ("seed_fraction", .5), ("absolute", 50),
    ("percentile", 90), ("threshold", 0),
])
def test_every_stop_rule_preserves_the_primary_ids(stop, value, growth):
    image, primary, expected = _paired_objects()
    before_image, before_primary = image.copy(), primary.copy()
    found = engine.secondary_object_instances(
        image, primary, sigma=0, stop=stop, stop_value=value, fill_holes=False, growth=growth)
    np.testing.assert_array_equal(found.labels, expected)
    assert found.labels.dtype == primary.dtype
    assert found.relationships.primary_ids == (7, 900)
    assert found.relationships.matched_ids == (7, 900)
    assert found.relationships.missing_secondary_ids == ()
    assert found.relationships.orphan_secondary_ids == ()
    assert found.relationships.incomplete_primary_ids == ()
    assert found.relationships.unexpanded_primary_ids == ()
    assert (found.level is None) is (stop == "seed_fraction")
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(primary, before_primary)


def test_dark_nuclear_pixels_stay_inside_their_secondary():
    image, primary, expected = _paired_objects()
    image[primary > 0] = 0
    found = engine.secondary_object_instances(
        image, primary, sigma=0, stop="absolute", stop_value=50, fill_holes=False)
    np.testing.assert_array_equal(found.labels, expected)
    assert found.relationships.incomplete_primary_ids == ()


def test_distance_growth_cannot_jump_an_excluded_barrier_to_an_unseeded_island():
    image = np.ones((15, 25), np.float32)
    image[:, 12] = 0
    primary = np.zeros(image.shape, np.uint16)
    primary[7, 3] = 900
    result = engine.secondary_object_instances(image, primary, sigma=0,
                                               growth='distance', stop='absolute',
                                               stop_value=.5, fill_holes=False)
    assert np.all(result.labels[:, :12] == 900)
    assert not result.labels[:, 12:].any()
    assert result.relationships.matched_ids == (900,)
    with pytest.raises(ValueError, match='growth'):
        engine.secondary_object_instances(image, primary, growth='unrecognized')


def test_sparse_uint64_ids_never_become_array_sizes_or_float_identifiers():
    ids = (2**63 + 7, 2**63 + 8)
    image, primary, expected = _paired_objects(np.uint64, ids)
    found = engine.secondary_object_instances(
        image, primary, sigma=0, stop="absolute", stop_value=50)
    np.testing.assert_array_equal(found.labels, expected)
    assert found.relationships.matched_ids == ids
    assert found.labels.dtype == np.uint64


def test_removing_a_small_secondary_does_not_renumber_the_survivor():
    primary = np.zeros((20, 20), dtype=np.uint16)
    primary[2, 2] = 7
    primary[10:14, 10:14] = 900
    image = np.where(primary, 100, 0)
    found = engine.secondary_object_instances(
        image, primary, sigma=0, stop="absolute", stop_value=50, min_area=2)
    assert set(np.unique(found.labels)) == {0, 900}
    assert found.relationships.missing_secondary_ids == (7,)
    assert found.relationships.matched_ids == (900,)
    assert found.relationships.unexpanded_primary_ids == (900,)


def test_a_crop_with_one_primary_keeps_that_primarys_number():
    image, primary, expected = _paired_objects()
    found = engine.secondary_object_instances(
        image[:, 60:], primary[:, 60:], sigma=0, stop="absolute", stop_value=50)
    np.testing.assert_array_equal(found.labels, expected[:, 60:])
    assert found.relationships.matched_ids == (900,)


def test_hole_filling_does_not_overwrite_an_enclosed_primary():
    primary = np.zeros((15, 15), dtype=np.uint16)
    primary[2:13, 2:13] = 7
    primary[3:12, 3:12] = 0
    primary[7, 7] = 900
    found = engine.secondary_object_instances(
        np.zeros(primary.shape), primary, sigma=0, stop="absolute",
        stop_value=1, fill_holes=True)
    assert found.labels[7, 7] == 900
    assert found.labels[5, 5] == 7
    assert found.relationships.incomplete_primary_ids == ()


def test_a_threshold_that_stops_all_growth_reports_unexpanded_primaries():
    image, primary, _ = _paired_objects()
    found = engine.secondary_object_instances(
        image, primary, sigma=0, stop="absolute", stop_value=101, fill_holes=False)
    np.testing.assert_array_equal(found.labels, primary)
    assert found.relationships.unexpanded_primary_ids == (7, 900)
    assert found.level == 101


def test_an_empty_primary_mask_cannot_invent_a_secondary():
    image, primary, _ = _paired_objects()
    primary[:] = 0
    found = engine.secondary_object_instances(image, primary)
    assert np.count_nonzero(found.labels) == 0
    assert found.relationships.primary_ids == ()
    assert found.relationships.secondary_ids == ()
    assert found.level is None


def test_relationship_report_detects_deleted_orphaned_and_incomplete_objects():
    _, primary, secondary = _paired_objects()
    secondary[secondary == 7] = 0
    secondary[40, 90] = 0
    secondary[0, 0] = 1000
    report = engine.primary_secondary_report(primary, secondary)
    assert report.matched_ids == (900,)
    assert report.missing_secondary_ids == (7,)
    assert report.orphan_secondary_ids == (1000,)
    assert report.incomplete_primary_ids == (900,)
    assert report.unexpanded_primary_ids == ()


def test_relationship_ids_are_exact_across_signed_and_unsigned_dtypes():
    primary = np.array([[2**63 - 1, 0]], dtype=np.int64)
    secondary = np.array([[2**63, 0]], dtype=np.uint64)
    report = engine.primary_secondary_report(primary, secondary)
    assert report.matched_ids == ()
    assert report.missing_secondary_ids == (2**63 - 1,)
    assert report.orphan_secondary_ids == (2**63,)


@pytest.mark.parametrize("settings", [
    {"sigma": -1}, {"sigma": np.nan}, {"sigma": np.inf},
    {"stop": "unknown"}, {"stop": "absolute", "stop_value": np.nan},
    {"stop": "percentile", "stop_value": 101},
    {"stop": "percentile", "stop_value": -1},
    {"stop": "seed_fraction", "stop_value": 1.1},
    {"stop": "seed_fraction", "stop_value": -.1}, {"min_area": -1},
    {"min_area": -.1}, {"min_area": np.nan}, {"min_area": np.inf},
])
def test_invalid_growth_settings_are_rejected(settings):
    image, primary, _ = _paired_objects()
    with pytest.raises(ValueError):
        engine.secondary_object_instances(image, primary, **settings)


@pytest.mark.parametrize("primary", [
    np.array([[-1]], dtype=np.int16), np.array([[1.5]]),
    np.array([[True]]), np.zeros((0, 0), dtype=np.uint16),
    np.zeros((2, 2, 2), dtype=np.uint16),
])
def test_invalid_primary_masks_are_rejected(primary):
    with pytest.raises(ValueError):
        engine.secondary_object_instances(np.zeros(primary.shape), primary)


@pytest.mark.parametrize("image", [
    np.ones((2, 3)), np.array([[np.nan, 0], [0, 0]]),
    np.array([[np.inf, 0], [0, 0]]),
])
def test_image_must_be_finite_and_match_the_primary(image):
    with pytest.raises(ValueError, match="finite and match"):
        engine.secondary_object_instances(image, np.ones((2, 2), dtype=np.uint16))


def test_peak_ratios_reject_negative_intensities():
    image, primary, _ = _paired_objects()
    with pytest.raises(ValueError, match="nonnegative intensities"):
        engine.secondary_object_instances(image - 10, primary, stop="seed_fraction")


def test_relationship_report_rejects_different_shapes():
    with pytest.raises(ValueError, match="same shape"):
        engine.primary_secondary_report(np.ones((2, 2), dtype=np.uint16),
                                        np.ones((3, 3), dtype=np.uint16))


@pytest.mark.parametrize('bundle', [False, True])
def test_exact_ids_survive_save_and_reload_for_tiff_and_seg_bundles(tmp_path, bundle):
    from tifffile import imread
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[2:5, 2:5] = 900
    labels[12:15, 12:15] = 900
    filename = 'field_seg.npy' if bundle else 'field.tif'
    if bundle:
        np.save(tmp_path / filename, {'masks': np.zeros_like(labels),
                                     'outlines': np.zeros_like(labels),
                                     'ismanual': np.zeros(900, dtype=bool),
                                     'source_image': 'original.tif'})
    path = engine.save_mask(str(tmp_path), filename, labels, preserve_ids=True)
    if bundle:
        payload = engine.read_seg_bundle(path)
        saved = payload['masks']
        assert payload['source_image'] == 'original.tif'
        assert len(payload['ismanual']) == 900
        assert set(np.unique(payload['outlines'])) == {0, 900}
    else:
        saved = imread(path)
    np.testing.assert_array_equal(saved, labels)
    assert saved.dtype == np.uint16
    assert set(np.unique(engine.canonical_labels(labels))) == {0, 1, 2}


@pytest.mark.parametrize('bundle', [False, True])
def test_unsupported_exact_ids_do_not_overwrite_an_existing_mask(tmp_path, bundle):
    from pathlib import Path
    labels = np.zeros((3, 3), dtype=np.uint64)
    labels[1, 1] = 2**63 + 7
    filename = 'field_seg.npy' if bundle else 'field.tif'
    if bundle:
        np.save(tmp_path / filename, {'masks': np.ones((3, 3), dtype=np.uint16)})
    else:
        engine.save_mask(str(tmp_path), filename, np.ones((3, 3), dtype=np.uint16))
    path = Path(engine.mask_save_path(str(tmp_path), filename))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='uint16'):
        engine.save_mask(str(tmp_path), filename, labels, preserve_ids=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize('labels', [np.array([[-1]]), np.array([[1.2]]),
                                     np.array([[True]]), np.zeros((0, 0), dtype=np.uint16)])
def test_exact_id_saves_reject_invalid_label_images(labels):
    with pytest.raises(ValueError):
        engine.canonical_labels(labels, preserve_ids=True)


def test_exact_id_readout_and_filter_measure_disconnected_pieces_together():
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[2:5, 2:5] = labels[12:15, 12:15] = 900
    image = np.full(labels.shape, 100, dtype=np.uint16)
    lookup = engine.ObjectLookup(labels, image, preserve_ids=True)
    assert lookup.at(3, 3).label == 900
    assert lookup.at(13, 13).label == 900
    assert lookup.measure(900) == (18, 100.0)
    kept, removed = engine.filter_report(labels, image, min_area=10, preserve_ids=True)
    np.testing.assert_array_equal(kept, labels)
    assert removed == []
    discarded, removed = engine.filter_report(labels, image, min_area=19, preserve_ids=True)
    assert not discarded.any()
    assert len(removed) == 1 and removed[0].label == 900 and removed[0].area == 18


@pytest.mark.parametrize('rule', ['clip', 'skip', 'replace'])
def test_exact_id_paste_extends_existing_secondary_without_allocating_ids(rule):
    current = np.zeros((20, 20), dtype=np.uint16)
    current[2:5, 2:5] = 900
    incoming = np.full((5, 5), 900, dtype=np.uint16)
    pasted, ids = engine._paste_region_objects(current, incoming, (2, 2), overlap=rule, preserve_ids=True)
    assert ids == [900]
    assert set(np.unique(pasted)) == {0, 900}
    assert np.count_nonzero(pasted == 900) == 25
    assert np.count_nonzero(current == 900) == 9


def test_exact_id_clip_retains_both_pieces_split_by_another_object():
    current = np.zeros((9, 9), dtype=np.uint16)
    current[:, 4] = 7
    incoming = np.full((5, 5), 900, dtype=np.uint16)
    pasted, ids = engine._paste_region_objects(current, incoming, (2, 2), overlap='clip', preserve_ids=True)
    assert ids == [900]
    assert np.all(pasted[:, 4] == 7)
    assert np.all(pasted[2:7, 2:4] == 900)
    assert np.all(pasted[2:7, 5:7] == 900)
    assert np.count_nonzero(pasted == 900) == 20


def test_exact_id_paste_clips_outside_image_without_changing_incoming_identity():
    current = np.zeros((5, 5), dtype=np.uint16)
    incoming = np.full((4, 4), 900, dtype=np.uint16)
    pasted, ids = engine._paste_region_objects(current, incoming, (-2, -1), preserve_ids=True)
    assert ids == [900]
    assert np.count_nonzero(pasted == 900) == 6
    skipped, ids = engine._paste_region_objects(pasted, np.full((5, 5), 7, dtype=np.uint16),
                                              (0, 0), overlap='skip', preserve_ids=True)
    np.testing.assert_array_equal(skipped, pasted)
    assert ids == []
