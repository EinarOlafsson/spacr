"""Pin the example download and all translated scenes to the recorded lesson."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_all_thirteen_translations_match_the_exact_english_scene_structure():
    english = json.loads((ROOT / 'lessons/77_embeddings.json').read_text())
    sha = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    reviews = [json.loads(p.read_text()) for p in sorted((ROOT / 'lessons/reviews').glob('77_embeddings.*.json'))]
    assert len(reviews) == 13
    assert {r['language'] for r in reviews} == {'da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN'}
    for review in reviews:
        assert review['english_sha256'] == sha
        assert review['lesson'] == english['id']
        assert len(review['scenes']) == len(english['scenes']) == 11
        assert all(text.strip() and text != scene['narration']
                   for text, scene in zip(review['scenes'], english['scenes']))
        assert all(review[key] for key in ('title','description','objectives','prerequisite'))
        assert review['native_speaker_reviewed'] is False
        assert review['listening_reviewed'] is False


def test_download_contains_exact_recorded_source_and_no_private_crops():
    path = ROOT.parents[1] / 'docs/source/_extra/tutorials/examples/Embeddings_API_example.zip'
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'embeddings_example.py', 'README.txt'}
        assert archive.read('embeddings_example.py') == (ROOT / 'embeddings_example.py').read_bytes()
        assert archive.read('README.txt') == (ROOT / 'embeddings_README.txt').read_bytes()
    preflight = json.loads((ROOT / 'evidence/2026-09-12_embeddings_api_preflight.json').read_text())
    assert all(r['helper_sha256'] == hashlib.sha256((ROOT / 'embeddings_example.py').read_bytes()).hexdigest()
               for r in preflight['runs'])
