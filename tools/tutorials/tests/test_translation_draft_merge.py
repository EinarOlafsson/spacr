"""Old draft catalogs must retain new reviewed lesson identities before generation."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translate_refresh import merge_selected
import translate_refresh
from stage_lesson import read, write


def test_old_drafts_keep_their_text_and_new_reviewed_lessons_survive():
    reviewed = {'lessons': [{'id': 'old', 'text': 'Reviewed'},
                            {'id': 'new', 'text': 'New reviewed translation'}]}
    draft = {'lessons': [{'id': 'old', 'text': 'Prior draft'}]}
    before = deepcopy((reviewed, draft))
    result = merge_selected(reviewed, draft, reviewed)
    assert result['lessons'] == [draft['lessons'][0], reviewed['lessons'][1]]
    assert (reviewed, draft) == before


def test_genuinely_missing_translation_still_fails():
    source = {'lessons': [{'id': 'old'}, {'id': 'missing'}]}
    with pytest.raises(ValueError, match='Missing translations'):
        merge_selected({'lessons': [{'id': 'old'}]}, {'lessons': []}, source)


def test_only_explicitly_selected_missing_drafts_can_wait_for_translation():
    source = {'lessons': [{'id': 'old'}, {'id': 'new'}, {'id': 'other'}]}
    reviewed = {'lessons': [{'id': 'old', 'text': 'reviewed'}]}
    drafts = {'lessons': [{'id': 'other', 'text': 'existing draft'}]}
    before = deepcopy((reviewed, drafts))
    result = merge_selected(reviewed, drafts, source, allow_missing={'new'})
    assert result['lessons'] == [reviewed['lessons'][0], drafts['lessons'][0]]
    assert (reviewed, drafts) == before
    # Deferring one selected identity cannot hide a different missing lesson.
    with pytest.raises(ValueError, match='Missing translations'):
        merge_selected(reviewed, {'lessons': []}, source, allow_missing={'new'})
    # The final merge still requires an actual translation of every identity.
    with pytest.raises(ValueError, match='Missing translations'):
        merge_selected(result, {'lessons': []}, source)


def test_cli_seeds_new_reviewed_lessons_before_translation(tmp_path, monkeypatch):
    source = {'lessons': [{'id': 'old', 'scenes': [{'narration': 'Old source.'}]},
                          {'id': 'new', 'scenes': [{'narration': 'New source.'}]}]}
    reviewed = deepcopy(source)
    reviewed['lessons'][1]['scenes'][0]['narration'] = 'Traducción nueva revisada.'
    write(tmp_path / 'catalog/lessons_en.json', source)
    write(tmp_path / 'catalog/lessons_es.json', reviewed)
    write(tmp_path / 'catalog-drafts/lessons_es.json', {'lessons': [source['lessons'][0]]})
    monkeypatch.setenv('SPACR_NLLB_MODEL', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['translate_refresh.py', '--stage', str(tmp_path),
                                    '--lessons', 'old', '--languages', 'es'])

    def translate(partial, *args):
        result = deepcopy(partial)
        result['lessons'][0]['scenes'][0]['narration'] = 'Texto actualizado.'
        return result

    monkeypatch.setattr(translate_refresh.spoken, 'translate_language', translate)
    translate_refresh.main()
    result = read(tmp_path / 'catalog-drafts/lessons_es.json')
    assert result['lessons'][0]['scenes'][0]['narration'] == 'Texto actualizado.'
    assert result['lessons'][1] == reviewed['lessons'][1]
    assert read(tmp_path / 'catalog/lessons_es.json') == reviewed
