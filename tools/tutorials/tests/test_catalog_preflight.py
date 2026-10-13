"""A different lesson can invalidate the target's whole caption catalog."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog_preflight import validate_caption_structure


def catalogs():
    english = {'lessons': [
        {'id': 'target', 'scenes': [{'narration': 'One'}]},
        {'id': 'other', 'scenes': [{'narration': 'Two'}, {'narration': 'Three'}]},
    ]}
    translated = copy.deepcopy(english)
    for lesson in translated['lessons']:
        for scene in lesson['scenes']:
            scene['narration'] = 'Übersetzung'
    return english, translated


def test_all_translated_lessons_align_even_in_a_different_order():
    english, translated = catalogs()
    translated['lessons'].reverse()
    validate_caption_structure(english, translated, 'de')


def test_a_different_lesson_has_to_align_too():
    english, translated = catalogs()
    translated['lessons'][1]['scenes'].pop()
    with pytest.raises(ValueError, match='de caption scene count differs for other'):
        validate_caption_structure(english, translated, 'de')


def test_missing_other_lesson_names_the_problem():
    english, translated = catalogs()
    translated['lessons'].pop()
    with pytest.raises(ValueError, match='de captions lack other'):
        validate_caption_structure(english, translated, 'de')


@pytest.mark.parametrize('text', ['', '  ', None, 1])
def test_empty_or_invalid_text_names_the_problem(text):
    english, translated = catalogs()
    translated['lessons'][1]['scenes'][0]['narration'] = text
    with pytest.raises(ValueError, match='caption text is empty or invalid for other'):
        validate_caption_structure(english, translated, 'de')
