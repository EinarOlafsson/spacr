"""Both recorded model lessons retain all languages and the exact source scenes."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'lessons'
LANGUAGES = {'es', 'fr', 'hi', 'it', 'pt-BR', 'ja', 'zh-CN', 'da', 'de', 'is', 'ko', 'nb', 'sv'}


@pytest.mark.parametrize('identity,scenes', [('21_model_compare', 6), ('22_model_zoo', 13)])
def test_all_model_reviews_are_complete_and_bound_to_the_current_source(identity, scenes):
    english = json.loads((ROOT / (identity + '.json')).read_text())
    digest = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    reviews = [json.loads(path.read_text()) for path in (ROOT / 'reviews').glob(identity + '.*.json')]
    assert len(reviews) == 13 and {item['language'] for item in reviews} == LANGUAGES
    assert len(english['scenes']) == scenes and english['host_app_key'] == 'make_masks'
    for review in reviews:
        assert review['lesson'] == identity and review['english_sha256'] == digest
        assert len(review['scenes']) == scenes
        assert all(isinstance(text, str) and text.strip() for text in review['scenes'])
        assert all(review[key] for key in ('title', 'description', 'objectives', 'prerequisite'))
        assert len(review['objectives']) == len(english['objectives'])
        # AI translations stay labelled as such until a person signs them off.
        assert review['review']['native_speaker_signoff'] is False
        assert review['review']['listening_review'] is False
