"""A sequencing read is pronounced reed; scope the correction to its sentence."""
import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'authoring/tools/render_all_voices.py'
tree = ast.parse(SOURCE.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'track_speech_text')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
speech = namespace['track_speech_text']
TARGET = 'This Python verification figure displays read depth from the saved three-barcode GUI run.'


@pytest.mark.parametrize('voice', ['af_heart', 'am_puck', 'bf_emma', 'bm_george'])
def test_the_actual_english_phrase_has_an_explicit_long_e_for_both_dialects(voice):
    assert speech('12_map_barcodes', 'en', voice, TARGET, TARGET) == TARGET.replace(
        'read depth', '[read](/ɹˈid/) depth')


@pytest.mark.parametrize('identity,language,text', [
    ('07_mask', 'en', TARGET), ('12_map_barcodes', 'fr', TARGET),
    ('12_map_barcodes', 'en', 'The read viewer shows one actual read per row.'),
])
def test_other_lessons_languages_and_correct_read_mentions_remain_unchanged(identity, language, text):
    assert speech(identity, language, 'af_heart', text, text) == text


def test_changed_spoken_premise_cannot_silently_skip_the_correction():
    with pytest.raises(ValueError, match='premise changed'):
        speech('12_map_barcodes', 'en', 'af_heart', TARGET, 'unrelated text')


def test_only_the_measured_fable_track_gets_the_extra_mastering_pass():
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'mastering_config')
    constants = {'MASTERING_CONFIG': {'filters': ['normal-encode']},
                 'LOUDNESS_FILTERS': ['normal-encode']}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), constants)
    mastering = constants['mastering_config']
    assert mastering('12_map_barcodes', 'en', 'bm_fable')['filters'] == [
        'normal-encode', 'normal-encode,volume=-2dB']
    for args in [('12_map_barcodes', 'en', 'af_heart'),
                 ('07_mask', 'en', 'bm_fable'), ('12_map_barcodes', 'fr', 'bm_fable')]:
        assert mastering(*args)['filters'] == ['normal-encode']
    assert constants['MASTERING_CONFIG']['filters'] == ['normal-encode']
