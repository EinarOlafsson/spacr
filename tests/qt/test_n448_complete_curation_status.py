"""Curation status names the judged field and respects all navigation outcomes."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr.qt import i18n
from spacr.qt.screens.make_masks import MakeMasksScreen, _StatusLabel


TEMPLATES = {
    ('moved', True): '{judged} kept  —  now on {now}',
    ('moved', False): '{judged} discarded — nothing was deleted  —  now on {now}',
    ('empty', True): '{judged} kept  —  no unjudged fields are left in this queue',
    ('empty', False): '{judged} discarded — nothing was deleted  —  no unjudged fields are left in this queue',
    ('last', True): '{judged} kept  —  that was the last field in this folder',
    ('last', False): '{judged} discarded — nothing was deleted  —  that was the last field in this folder',
    ('other', True): '{judged} kept',
    ('other', False): '{judged} discarded — nothing was deleted',
}
FRENCH = {
    ('moved', True): '{judged} conservé  —  maintenant sur {now}',
    ('moved', False): '{judged} rejeté — aucun élément n’a été supprimé  —  maintenant sur {now}',
    ('empty', True): '{judged} conservé  —  aucun champ non évalué ne reste dans cette file',
    ('empty', False): '{judged} rejeté — aucun élément n’a été supprimé  —  aucun champ non évalué ne reste dans cette file',
    ('last', True): '{judged} conservé  —  c’était le dernier champ de ce dossier',
    ('last', False): '{judged} rejeté — aucun élément n’a été supprimé  —  c’était le dernier champ de ce dossier',
    ('other', True): '{judged} conservé',
    ('other', False): '{judged} rejeté — aucun élément n’a été supprimé',
}


@pytest.mark.parametrize('language', ['en', 'fr'])
@pytest.mark.parametrize('state', ['moved', 'empty', 'last', 'other'])
@pytest.mark.parametrize('keep', [True, False])
def test_actual_verdict_status_is_complete_and_preserves_navigation(
        qapp, monkeypatch, tmp_path, language, state, keep):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    files = ['judged.tif'] if state == 'last' else ['judged.tif', 'following.tif']
    originals = {}
    for name in [*files, 'mask.tif', 'keep_discard.csv']:
        path = tmp_path / name
        path.write_bytes(('private fixture ' + name).encode())
        originals[path] = path.read_bytes()
    status = _StatusLabel()
    calls = []
    subject = SimpleNamespace(
        _current_index=0, _folder=str(tmp_path), _image_files=files,
        _hide_judged=state == 'empty', _status_label=status,
        _blind_label=lambda path: Path(path).name, _visible_indices=lambda: [],
        _sync_queue_tally=lambda: calls.append('tally'))

    def next_field():
        calls.append('next')
        if state == 'moved':
            subject._current_index = 1

    subject._on_next = next_field
    moved = MakeMasksScreen._advance_after_verdict(subject, keep, 'judged.tif')
    expected = (TEMPLATES if language == 'en' else FRENCH)[state, keep].format(
        judged='judged.tif', now='following.tif')

    assert status.text() == expected
    assert moved is (state == 'moved')
    assert subject._current_index == (1 if state == 'moved' else 0)
    assert calls == ['next', 'tally']
    assert all(path.read_bytes() == before for path, before in originals.items())
    status.close()
    status.deleteLater()
