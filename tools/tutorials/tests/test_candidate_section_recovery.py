"""Recover the known heading regression without replacing any lesson prose."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_release_candidate import restore_localized_section


def test_english_duplicate_is_recovered_and_every_other_field_stays_identical():
    lesson = {'id': '07_mask', 'section': 'Core', 'scenes': [{'narration': 'Reviewed Spanish'}]}
    before = deepcopy(lesson)
    restore_localized_section(lesson, {'section': 'Core'}, {'section': 'Módulos principales'},
                              {'section': 'Core'})
    assert lesson == {**before, 'section': 'Módulos principales'}


@pytest.mark.parametrize('section,published,published_english', [
    ('Modulos revisados', 'Módulos principales', 'Core'),
    ('Core', 'Core', 'Core'), ('Core', None, 'Core'),
    ('Core', 'New English category', 'New English category'),
])
def test_translated_headings_win_and_english_or_missing_donors_do_not_count(section, published, published_english):
    lesson = {'id': '12_map_barcodes', 'section': section, 'scenes': ['unchanged']}
    before = deepcopy(lesson)
    restore_localized_section(lesson, {'section': 'Core'}, {'section': published},
                              {'section': published_english})
    assert lesson == before
