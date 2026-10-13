"""All thirteen language scripts stay pinned to the actual English lesson.

The fifteen-scene ``12_map_barcodes.reviewed.json`` bundle belongs to the
retired 1.5.0.8 recording; the twelve-scene walkthrough is translated one
review file per language, like every other lesson.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {'es', 'fr', 'it', 'pt-BR', 'hi', 'ja', 'zh-CN', 'da', 'de', 'is', 'ko', 'nb', 'sv'}


def test_complete_source_pinned_editorial_matrix():
    english = json.loads((ROOT / 'lessons/12_map_barcodes.json').read_text())
    digest = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    reviews = {path.name.split('.')[1]: json.loads(path.read_text())
               for path in (ROOT / 'lessons/reviews').glob('12_map_barcodes.*.json')
               if path.name != '12_map_barcodes.reviewed.json'}
    assert set(reviews) == LANGUAGES
    assert len(english['scenes']) == 12
    for language, review in reviews.items():
        assert review['language'] == language and review['lesson'] == english['id']
        assert review['english_sha256'] == digest
        assert review['review']['native_speaker_signoff'] is False
        assert review['review']['listening_review'] is False
        assert len(review['scenes']) == len(english['scenes'])
        assert all(isinstance(scene, str) and scene.strip() for scene in review['scenes'])
        assert 'Map Barcodes' in review['title']
        assert 'SRR33531217' in review['prerequisite']
        assert 'primers_3' in review['prerequisite']
        assert len(review['objectives']) == len(english['objectives']) == 3
