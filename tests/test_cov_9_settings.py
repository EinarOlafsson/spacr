"""Guards on the settings module's refusals, fallbacks and tuple parsing.

Every case here is a path a user reaches by accident rather than on purpose:
a key that is not an organelle key, a settings file carrying a penalty that
used to be the default, an optional helper module that cannot be imported,
and a list control typed as a tuple. Each one has to behave predictably,
because the alternative is a run that dies inside the settings factory before
any analysis has started.
"""
from __future__ import annotations

import pytest

from spacr import settings as settings_mod


# ---------------------------------------------------------------------------
# _organelle_slot_key
# ---------------------------------------------------------------------------

def test_a_non_organelle_key_is_refused_by_the_slot_translator():
    """Translating a key that is not ``organelle_*`` must raise, not guess.

    The translator builds a secondary slot's key by stripping the
    ``organelle_`` prefix. Handed anything else it would silently produce a
    key that belongs to no slot, and the cloned value would land in a setting
    nothing reads -- so the mistake has to surface at the call.
    """
    with pytest.raises(ValueError) as excinfo:
        settings_mod._organelle_slot_key('cell_diameter', 'pathogen')
    assert 'cell_diameter' in str(excinfo.value)


def test_an_organelle_key_still_translates_to_the_named_slot():
    """The refusal above must not have broken the case it guards."""
    assert settings_mod._organelle_slot_key(
        'organelle_diameter', 'pathogen') == 'pathogen_diameter'


# ---------------------------------------------------------------------------
# get_perform_regression_default_settings: the legacy group-lasso penalty
# ---------------------------------------------------------------------------

def test_a_legacy_group_lasso_penalty_says_so_when_that_family_is_selected(capsys):
    """A saved 0.05 penalty is converted, and the user is told when it matters.

    0.05 was the panel's own default for the whole life of the setting, so a
    settings file carrying it recorded nobody's choice. It is converted to
    cross-validation either way; the message is printed only when
    ``group_lasso`` is the family actually being fitted, because that is the
    only run whose numbers change.
    """
    result = settings_mod.get_perform_regression_default_settings({
        'group_lasso_lambda': settings_mod.LEGACY_GROUP_LASSO_LAMBDA,
        'regression_type': 'group_lasso',
    })
    assert result['group_lasso_lambda'] == 'auto'
    printed = capsys.readouterr().out
    assert 'group_lasso_lambda' in printed
    assert "'auto'" in printed


def test_a_legacy_penalty_under_another_family_is_converted_silently(capsys):
    """Under any other family the conversion changes no fitted number.

    Printing there would be noise on a run the setting is not even read by,
    so the value is corrected without a message.
    """
    result = settings_mod.get_perform_regression_default_settings({
        'group_lasso_lambda': settings_mod.LEGACY_GROUP_LASSO_LAMBDA,
        'regression_type': 'ols',
    })
    assert result['group_lasso_lambda'] == 'auto'
    assert 'group_lasso_lambda' not in capsys.readouterr().out


def test_control_wells_fall_back_to_filter_value_when_the_block_spec_fails(monkeypatch):
    """An unreadable control-block spec must not take the defaults factory down.

    ``control_wells`` is only ever an addition to ``filter_value``; if the
    block spec cannot be computed the run still has to start with the wells
    the user typed, rather than raising inside a settings factory that has
    produced no analysis yet.
    """
    import spacr.well_spec as well_spec

    def _explode(_settings):
        raise RuntimeError('control block spec unreadable')

    monkeypatch.setattr(well_spec, 'control_block_wells', _explode)
    result = settings_mod.get_perform_regression_default_settings({
        'filter_value': ['c1', 'c2'],
    })
    assert result['control_wells'] == ['c1', 'c2']


# ---------------------------------------------------------------------------
# Optional-import fallbacks
# ---------------------------------------------------------------------------

def test_an_unreadable_coordinate_column_yields_no_coordinate_setting(monkeypatch):
    """When the streamer cannot name a coordinate column the answer is None.

    ``coordinate_columns`` is declared a list, and None is what means "use the
    merged masks". Returning a half-derived value here would make the run fail
    its own settings validation instead of falling back to the default source.
    """
    import spacr.stream_dataset as stream_dataset

    def _explode(_object_array):
        raise KeyError('no such object array')

    monkeypatch.setattr(stream_dataset, 'coordinate_column', _explode)
    assert settings_mod._coordinate_columns_for('cell') is None


def test_classes_are_left_alone_when_the_class_folder_cannot_fold(monkeypatch):
    """A failing class fold returns the caller's settings unchanged.

    The fold only fills ``annotation_column`` and ``class_metadata`` from
    ``classes``; when it cannot, the caller's dict must come back intact so
    the values the user did set are still the ones the run uses.
    """
    import spacr.classify_classes as classify_classes

    def _explode(_settings):
        raise ValueError('classes are not foldable')

    monkeypatch.setattr(classify_classes, 'fold_into_classes', _explode)
    original = {'classes': ['a', 'b']}
    assert settings_mod._fold_the_classes(original) is original


def test_outlier_criteria_fall_back_to_the_built_in_four(monkeypatch):
    """The outlier panel keeps its criteria when the filter module is absent.

    The panel and the filtering logic share one list so they cannot disagree;
    if the shared list cannot be imported the panel still has to offer the
    four criteria spaCR has always filtered on rather than an empty control.
    """
    import spacr.outlier_filter as outlier_filter

    monkeypatch.delattr(outlier_filter, 'CRITERIA')
    criteria = settings_mod._outlier_criteria()
    keys = [key for key, _label in criteria]
    assert keys == ['cell_area', 'nucleus_area',
                    'cell_intensity', 'nucleus_intensity']


# ---------------------------------------------------------------------------
# parse_list
# ---------------------------------------------------------------------------

def test_a_typed_tuple_becomes_the_list_it_means():
    """``(1, 2, 3)`` typed into a list control is that list of three."""
    assert settings_mod.parse_list('(1, 2, 3)') == [1, 2, 3]


def test_a_one_element_tuple_keeps_its_single_value():
    """``(3,)`` is python's spelling of one value, not an empty container.

    Flattening it away would silently drop the only entry the user typed.
    """
    assert settings_mod.parse_list('(3,)') == [3]


def test_a_bare_scalar_is_refused_by_a_list_control():
    """A control that means "a list" must not accept a lone number.

    Accepting it would hand the pipeline an int where it indexes a sequence,
    and the failure would surface much later as a type error inside analysis.
    """
    with pytest.raises(ValueError) as excinfo:
        settings_mod.parse_list('5')
    assert 'Invalid format for list' in str(excinfo.value)
