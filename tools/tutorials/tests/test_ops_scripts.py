"""Every OPS locale must follow the exact authored nine-scene script."""
import hashlib
import json
from pathlib import Path


def test_all_thirteen_reviews_are_bound_to_current_english_and_every_scene():
    root = Path(__file__).resolve().parents[1] / 'lessons'
    english = json.loads((root / '76_ops.json').read_text())
    digest = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    paths = sorted((root / 'reviews').glob('76_ops.*.json'))
    reviews = [json.loads(path.read_text()) for path in paths]
    assert {item['language'] for item in reviews} == {'es', 'fr', 'hi', 'it', 'pt-BR', 'ja', 'zh-CN', 'da', 'de', 'is', 'ko', 'nb', 'sv'}
    assert len(reviews) == 13
    assert english['host_app_key'] == 'mask' and len(english['scenes']) == 9
    for review in reviews:
        assert review['lesson'] == english['id'] and review['english_sha256'] == digest
        assert len(review['scenes']) == len(english['scenes'])
        assert all(isinstance(text, str) and text.strip() for text in review['scenes'])
        assert all(review[key] for key in ('title', 'description', 'objectives', 'prerequisite'))
        assert review['native_speaker_reviewed'] is False
        assert review['listening_reviewed'] is False
