"""Installed localized CZI resource refusal before any scene expansion."""
import json
from pathlib import Path

import pytest

from tests.test_web_mask_legacy_czi_n698 import legacy as legacy
from tools.web_mask_generator import inputs


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_actual_czi_scene_limit_translation(legacy, monkeypatch, language):
    from spacr import convert

    path, array, _, reads, _ = legacy
    before = array.copy()
    expected_path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language
                     / '2026-10-15-web-czi-scene-limit.json')
    expected = json.loads(expected_path.read_text())['records'][0]['translation']
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    monkeypatch.setattr(convert, '_legacy_czi_series', lambda *args: pytest.fail(
        'Scene expansion ran before the localized limit refusal'))
    with pytest.raises(ValueError) as error:
        inputs.describe_upload(path, max_scenes=1)
    assert str(error.value) == expected
    assert expected != 'The CZI exceeds the configured scene limit.'
    assert reads == []
    assert (array == before).all()
