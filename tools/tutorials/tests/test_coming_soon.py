"""An unavailable route has no media promises; ready scripts stay identical."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coming_soon import COPY, HELD, OPS, EMBEDDINGS, PLACEHOLDERS, release_catalog


@pytest.fixture
def catalog():
    return {'lessons': [dict(id=identity, number=i, title=identity,
            app_key=identity.split('_', 1)[-1], silent='existing.mp4',
            poster='existing.jpg', example_files=['existing.zip'],
            objectives=['Old promise'], prerequisite='Old input',
            scenes=[{'narration': 'Existing narration'}])
            for i, identity in enumerate(('01_ready', *HELD), 1)]}


@pytest.mark.parametrize('language', COPY)
def test_all_languages_preserve_ready_content_and_explicitly_hold_only_six(catalog, language):
    before = deepcopy(catalog)
    result = release_catalog(catalog, language)
    assert catalog == before
    assert result['lessons'][0] == before['lessons'][0]
    held = [x for x in result['lessons'] if x.get('status') == 'coming_soon']
    assert [x['id'] for x in held] == list(PLACEHOLDERS)
    for lesson in held:
        assert lesson['scenes'] == [] and lesson['objectives'] == []
        assert lesson['availability_title'] == COPY[language][0]
        assert lesson['description'] == COPY[language][1]
        assert not {'silent', 'poster', 'example_files'} & lesson.keys()
    by_id = {lesson['id']: lesson for lesson in held}
    assert by_id[OPS]['app_key'] == 'ops' and by_id[OPS]['host_app_key'] == 'mask'
    assert by_id[EMBEDDINGS]['app_key'] == 'embeddings'
    assert 'host_app_key' not in by_id[EMBEDDINGS]


@pytest.mark.parametrize('change', [
    lambda c: c['lessons'].append(deepcopy(c['lessons'][0])),
    lambda c: c['lessons'].pop(),
    lambda c: c['lessons'].append({'id': OPS}),
    lambda c: c['lessons'].append({'id': EMBEDDINGS}),
])
def test_unexpected_input_cannot_silently_drop_or_duplicate_a_route(catalog, change):
    release_catalog(catalog, 'en')
    change(catalog)
    with pytest.raises(ValueError, match='distinct original lessons'):
        release_catalog(catalog, 'en')
