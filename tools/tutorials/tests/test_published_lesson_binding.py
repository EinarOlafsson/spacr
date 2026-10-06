import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import published_lesson


@pytest.mark.parametrize('mutation', ('publication_revision', 'hosted_narration'))
def test_preserved_lesson_rejects_conflicting_current_publication(tmp_path, monkeypatch, mutation):
    source = published_lesson.CANDIDATE
    for name in ('release-manifest.json', 'checkpoint.json', 'publication-receipt.json',
                 'candidate-browser-checks.json', 'published-media-browser-checks.json',
                 'web/catalog/lessons_en.json'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    monkeypatch.setattr(published_lesson, 'CANDIDATE', tmp_path)
    if mutation == 'publication_revision':
        path = tmp_path / 'publication-receipt.json'
        payload = json.loads(path.read_text())
        payload['commit'] = '0' * 40
    else:
        path = tmp_path / 'published-media-browser-checks.json'
        payload = json.loads(path.read_text())
        case = next(row for row in payload['ready_playback_cases'] if row['lesson'] == '77_embeddings')
        case['audio_sha256'] = '0' * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError):
        published_lesson.check_published_lesson('77_embeddings', 9)
