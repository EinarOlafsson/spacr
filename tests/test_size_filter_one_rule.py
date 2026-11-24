"""Merge 4: every size filter is one function with one boundary rule.

An object whose size equals the minimum or the maximum is kept (inclusive);
labels keep their ids and touching objects are never merged.
"""
import numpy as np

from spacr.utils import _filter_object, _filter_cp_masks


def _field():
    """Objects of 3, 4, 6 and 9 pixels; ids 2, 5, 7, 11; 2 and 5 touch."""
    mask = np.zeros((10, 12), np.uint16)
    mask[0, 0:3] = 2
    mask[2, 0:2] = 5
    mask[3, 0:2] = 5
    mask[5:7, 0:3] = 7
    mask[0:3, 6:9] = 11
    return mask


def _ids(mask):
    return set(np.unique(mask).tolist()) - {0}


def test_object_exactly_at_each_bound_is_kept():
    for bounds, kept in (((3, None), {2, 5, 7, 11}), ((4, None), {5, 7, 11}),
                         ((3, 9), {2, 5, 7, 11}), ((3, 6), {2, 5, 7}),
                         ((None, 4), {2, 5})):
        assert _ids(_filter_object(_field(), bounds=bounds)) == kept, bounds


def test_legacy_strict_rule_drops_the_object_at_the_bound():
    assert _ids(_filter_object(_field(), bounds=(3, 9), inclusive=False)) == {5, 7}


def test_ids_are_preserved_and_filter_is_in_place():
    original = _field()
    mask = _field()
    out = _filter_object(mask, 4)
    assert out is mask and out.dtype == np.uint16
    assert _ids(out) == {5, 7, 11}
    assert (out[out > 0] == original[out > 0]).all()


def test_positional_arguments_still_work():
    assert _ids(_filter_object(_field(), 4, 6)) == {5, 7}


def test_empty_and_blank_masks():
    assert _filter_object(np.zeros((0, 4), np.uint16), 5).shape == (0, 4)
    assert not _filter_object(np.zeros((4, 4), np.uint16), 5).any()


def _run_cp(minimum, maximum, inclusive=True):
    out = _filter_cp_masks([_field()], [[None]], True, False, minimum, maximum,
                           False, False, [None], False, 5,
                           size_filter_inclusive=inclusive)
    return _ids(np.asarray(out))


def test_make_masks_filter_keeps_objects_exactly_at_the_bounds():
    assert _run_cp(3, 9) == {2, 5, 7, 11}
    assert _run_cp(3, 9, inclusive=False) == {5, 7}


def test_plot_filter_keeps_objects_exactly_at_the_bounds():
    from spacr.plot import _filter_objects_in_plot

    stack = np.stack([_field().astype(np.int32)], axis=2)
    out = _filter_objects_in_plot(stack.copy(), 0, None, None, [0], [[3, 9]], True, True)
    assert _ids(out) == {2, 5, 7, 11}
    out = _filter_objects_in_plot(stack.copy(), 0, None, None, [0], [[3, 9]], True, True,
                                  size_filter_inclusive=False)
    assert _ids(out) == {5, 7}


def test_nucleus_territories_keep_a_nucleus_of_the_minimum_size():
    from spacr.host_pathogen import nucleus_territories

    _, nuclei = nucleus_territories(_field(), expand_px=1, min_nucleus_size=4)
    assert _ids(nuclei) == {5, 7, 11}


def test_backend_drop_small_keeps_the_minimum_size_object():
    from spacr._segmentation_backends import _drop_small

    out = _drop_small(_field(), 4)
    assert int((out > 0).sum()) == 19 and int(out.max()) == 3


def test_gui_remove_small_objects_keeps_ids_and_touching_objects():
    from spacr.qt import mask_engine as me

    out = me.remove_small_objects(_field(), 4)
    assert _ids(out) == {5, 7, 11}
    assert int((out == 5).sum()) == 4


def test_gui_remove_small_objects_binary_mask_is_split_into_components():
    from spacr.qt import mask_engine as me

    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 1
    mask[3:6, 3:6] = 1
    out = me.remove_small_objects(mask, 4)
    assert int((out > 0).sum()) == 9


def test_old_settings_fold_to_the_strict_rule():
    from spacr import settings as S

    assert S.DEFAULT_CHANGES[961] == {'size_filter_inclusive': False}
    assert S.SETTINGS_SCHEMA_VERSION >= 961
    old = {}
    S._fold_older_defaults(old, 959)
    assert old['size_filter_inclusive'] is False
    assert S.set_default_settings_preprocess_generate_masks({})['size_filter_inclusive'] is True
