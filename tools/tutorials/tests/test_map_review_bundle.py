"""All thirteen language scripts stay pinned to the actual English lesson."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apply_translation_review import spoken_form


def test_complete_source_pinned_editorial_matrix():
    english = json.loads((ROOT / 'lessons/12_map_barcodes.json').read_text())
    for scene in english['scenes']:
        scene['speech_text'] = spoken_form(scene['narration'], 'en')
    bundle = json.loads((ROOT / 'lessons/reviews/12_map_barcodes.reviewed.json').read_text())
    assert bundle['english_sha256'] == hashlib.sha256(
        json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert set(bundle['translations']) == {'es', 'fr', 'it', 'pt-BR', 'hi', 'ja', 'zh-CN',
                                            'da', 'de', 'is', 'ko', 'nb', 'sv'}
    assert bundle['review']['native_speaker_signoff'] is False
    assert bundle['review']['human_listening_signoff'] is False
    for translated in bundle['translations'].values():
        assert len(translated['scenes']) == len(english['scenes']) == 13
        assert all(len(scene.strip()) > 40 for scene in translated['scenes'])
        assert 'Map Barcodes' in translated['title']
        assert 'SRR33531217' in translated['prerequisite']
        assert 'primers_3' in translated['prerequisite']
        assert len(translated['objectives']) == 3
        assert translated['section'] and translated['section'] != 'Core'
