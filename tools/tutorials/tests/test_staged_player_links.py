"""Linkless retained lessons pass, but missing/extra required links never do."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_staged_lesson import check_related_links
import verify_staged_lesson as staged


def test_server_uses_committed_player_not_stale_authoring_copy(tmp_path, monkeypatch):
    workspace, repo = tmp_path/'authoring', tmp_path/'repo'
    production = workspace/'refresh/production'
    live = repo/'docs/source/_extra/tutorials'
    (workspace/'web').mkdir(parents=True); live.mkdir(parents=True)
    (workspace/'web/app_v2.js').write_text('STALE PLAYER')
    (live/'app_v2.js').write_text('CURRENT PLAYER')
    (live/'examples').mkdir(); (live/'examples/example.zip').write_bytes(b'current archive')
    monkeypatch.setattr(staged,'WORKSPACE',workspace)
    monkeypatch.setattr(staged,'REPO',repo)
    monkeypatch.setattr(staged,'DEFAULT_STAGE',workspace/'refresh')
    handler=staged.Handler.__new__(staged.Handler); handler.directory=str(workspace)
    assert Path(handler.translate_path('/web/app_v2.js?v=fresh')).read_text() == 'CURRENT PLAYER'
    assert Path(handler.translate_path('/web/examples/example.zip')).read_bytes() == b'current archive'
    # Media routing remains independent: an existing staged file wins,
    # and untouched lessons retain their original media fallback.
    production.mkdir(parents=True); (production/'new.mp4').write_bytes(b'new')
    assert Path(handler.translate_path('/refresh/production/new.mp4')) == production/'new.mp4'
    assert Path(handler.translate_path('/refresh/production/old.mp4')) == workspace/'production/old.mp4'


@pytest.mark.parametrize('actual,expected', [([], []), (['a'], ['a']),
    (['b', 'a', 'a'], ['a', 'b'])])
def test_exact_authored_link_set_passes(actual, expected):
    check_related_links(actual, expected)


@pytest.mark.parametrize('actual,expected', [([], ['a']), (['a'], []),
    (['a'], ['b']), (['a'], ['a', 'b']), (['a', 'b'], ['a'])])
def test_missing_wrong_and_extra_links_fail(actual, expected):
    with pytest.raises(ValueError, match='Related lesson links differ'):
        check_related_links(actual, expected)
