"""A displayed definition must agree with the selected source entry."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_feature_dictionary import verify_detail


@pytest.fixture
def definition():
    doc = SimpleNamespace(key='area', title='Area', description='Actual object size.',
                          unit='Conditional unit.', module='spacr.measure',
                          computed_by='actual_consumer()', object_types=('cell',))
    text = 'Area\nActual object size.\nConditional unit.\nspacr.measure\nactual_consumer()'
    return doc, text


def test_positive_definition_fields_all_present(definition):
    doc, text = definition
    assert verify_detail(doc, text, 'area')['key'] == 'area'


@pytest.mark.parametrize('field', ['description', 'unit', 'module', 'computed_by'])
def test_missing_displayed_source_field_fails(definition, field):
    doc, text = definition
    with pytest.raises(ValueError, match=field):
        verify_detail(doc, text.replace(getattr(doc, field), ''), 'area')


def test_stale_or_absent_feature_fails(definition):
    doc, text = definition
    with pytest.raises(ValueError, match='different feature'):
        verify_detail(doc, text, 'mean_intensity')
    with pytest.raises(ValueError, match='different feature'):
        verify_detail(None, text, 'area')
