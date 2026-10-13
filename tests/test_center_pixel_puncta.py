"""Small, faint centres retain native measurements and immutable parent identity."""
import numpy as np
import pytest

from spacr.qt import cpu_modes


def field():
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[:96, :128]
    image = (40 + rng.normal(0, .15, yy.shape)
             + 12*np.exp(-((yy-35)**2+(xx-35)**2)/4.5)
             + 30*np.exp(-((yy-60)**2+(xx-90)**2)/8)
             + 100*np.exp(-((yy-10)**2+(xx-110)**2)/8)).astype(np.float32)
    parents = np.zeros(image.shape, dtype=np.uint16)
    parents[20:75, 20:55] = 7
    parents[40:80, 75:110] = 19
    return image, parents


def test_faint_small_spots_are_individual_children_with_native_intensity():
    image, parents = field()
    before_image, before_parent = image.copy(), parents.copy()
    labels, rows = cpu_modes.puncta(image, parents, measurements=True)
    accepted = rows[rows.included]
    assert len(accepted) == 2
    assert set(accepted.parent_id) == {7, 19}
    assert labels[35, 35] and labels[60, 90]
    assert labels[35, 35] != labels[60, 90]
    assert labels[10, 110] == 0
    assert np.all(labels[parents == 0] == 0)
    assert np.all(accepted.mask_pixels == 20)
    assert np.all(accepted.center_mean > accepted.local_bg + 3)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(parents, before_parent)


def test_native_threshold_retains_equality_and_rejects_dim_centres():
    image, parents = field()
    _, rows = cpu_modes.puncta(image, parents, measurements=True)
    accepted = rows[rows.included]
    cutoff = float((accepted.center_mean-accepted.local_bg).min())
    at_cutoff = cpu_modes.DEFAULT_PARAMS._replace(puncta_min_corrected=cutoff)
    above = at_cutoff._replace(puncta_min_corrected=cutoff + .001)
    assert np.unique(cpu_modes.puncta(image, parents, at_cutoff)).size == 3
    assert np.unique(cpu_modes.puncta(image, parents, above)).size == 2


@pytest.mark.parametrize('count', [10, 20, 40])
def test_centre_pixel_count_changes_measurement_without_growing_a_watershed(count):
    image, parents = field()
    params = cpu_modes.DEFAULT_PARAMS._replace(puncta_center_pixels=count,
                                               puncta_min_corrected=0)
    labels, rows = cpu_modes.puncta(image, parents, params, measurements=True)
    spot = rows[(rows.y == 60) & (rows.x == 90)].iloc[0]
    assert spot.center_pixels == count
    assert np.count_nonzero(labels == spot.object_label) == count
    assert np.isclose(image[labels == spot.object_label].mean(), spot.center_mean)


def test_overlapping_windows_keep_both_centres_and_report_shared_pixels():
    yy, xx = np.mgrid[:80, :80]
    image = (10 + 40*np.exp(-((yy-40)**2+(xx-35)**2)/3)
             + 35*np.exp(-((yy-40)**2+(xx-39)**2)/3)).astype(np.float32)
    parents = np.zeros(image.shape, dtype=np.uint16)
    parents[10:70, 10:70] = 7
    params = cpu_modes.DEFAULT_PARAMS._replace(puncta_sigmas=(1.5,), puncta_center_pixels=40)
    labels, rows = cpu_modes.puncta(image, parents, params, measurements=True)
    accepted = rows[rows.included]
    assert len(accepted) == 2
    assert labels[40, 35] != labels[40, 39]
    assert np.all(accepted.mask_pixels > 0)
    assert np.any(accepted.mask_pixels < 40)
    assert np.any(accepted.center_mean != image[labels == accepted.iloc[0].object_label].mean())


def test_empty_parents_produce_an_empty_explicit_measurement_table():
    image, parents = field()
    labels, rows = cpu_modes.puncta(image, parents*0, measurements=True)
    assert not labels.any()
    assert rows.empty and {'included', 'parent_id', 'center_mean', 'local_bg'} <= set(rows.columns)


@pytest.mark.parametrize('change', [dict(puncta_center_pixels=82), dict(puncta_k=0),
                                  dict(puncta_sigmas=()), dict(puncta_sigmas=(float('nan'),))])
def test_invalid_scientific_settings_fail_before_detection(change):
    image, parents = field()
    with pytest.raises(ValueError, match='Invalid puncta'):
        cpu_modes.puncta(image, parents, cpu_modes.DEFAULT_PARAMS._replace(**change))


@pytest.mark.parametrize('bad_input', [
    'volume', 'parent_shape', 'image_nan', 'parent_inf',
    'negative_parent', 'fractional_parent',
])
def test_invalid_native_inputs_are_refused_without_changing_them(bad_input):
    image = np.ones((24, 24), dtype=np.float32)
    parents = np.ones(image.shape, dtype=np.float32)
    if bad_input == 'volume':
        image = image[None, ...]
        parents = parents[None, ...]
    elif bad_input == 'parent_shape':
        parents = parents[:-1]
    elif bad_input == 'image_nan':
        image[12, 12] = np.nan
    elif bad_input == 'parent_inf':
        parents[12, 12] = np.inf
    elif bad_input == 'negative_parent':
        parents[12, 12] = -1
    else:
        parents[12, 12] = .5
    prior_image, prior_parents = image.copy(), parents.copy()
    with pytest.raises(ValueError, match='2-D|finite|integer labels'):
        cpu_modes.puncta(image, parents, measurements=True)
    np.testing.assert_array_equal(image, prior_image)
    np.testing.assert_array_equal(parents, prior_parents)


@pytest.mark.parametrize('level', [0, 40])
def test_a_flat_parent_field_has_no_candidate_objects(level):
    image = np.full((32, 32), level, dtype=np.float32)
    parents = np.full(image.shape, 7, dtype=np.uint16)
    labels, rows = cpu_modes.puncta(image, parents, measurements=True)
    assert rows.empty
    assert not labels.any()
    assert {'parent_id', 'center_mean', 'included'} <= set(rows.columns)


def test_a_bright_peak_without_a_complete_centre_window_is_not_measured():
    yy, xx = np.mgrid[:32, :32]
    image = (40 + 60 * np.exp(-((yy - 2)**2 + (xx - 2)**2) / 3)).astype(np.float32)
    parents = np.full(image.shape, 7, dtype=np.uint16)
    params = cpu_modes.DEFAULT_PARAMS._replace(puncta_edge_margin=0)
    prior = image.copy()
    labels, rows = cpu_modes.puncta(image, parents, params, measurements=True)
    assert rows.empty
    assert not labels.any()
    np.testing.assert_array_equal(image, prior)
