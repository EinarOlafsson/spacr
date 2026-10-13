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


@pytest.mark.parametrize('voice', ['af_heart', 'am_puck', 'bf_emma', 'bm_george'])
@pytest.mark.parametrize('text,expected', [
    ('Here the guide is strongly supported as reverse complemented in read one and as stored in read two.',
     'Here the guide is strongly supported as reverse complemented in [read](/ɹˈid/) one and as stored in [read](/ɹˈid/) two.'),
    ('Read one supports the reversed reference, while read two supports the stored orientation.',
     '[Read](/ɹˈid/) one supports the reversed reference, while [read](/ɹˈid/) two supports the stored orientation.'),
    ('Check your own library and read layout in the same way.',
     'Check your own library and [read](/ɹˈid/) layout in the same way.'),
])
def test_recorded_orientation_nouns_use_reed_and_keep_the_correction_scoped(voice, text, expected):
    assert speech('12_map_barcodes', 'en', voice, text, text) == expected
    assert speech('07_mask', 'en', voice, text, text) == text
    assert speech('12_map_barcodes', 'fr', voice, text, text) == text
    with pytest.raises(ValueError, match='premise changed'):
        speech('12_map_barcodes', 'en', voice, text, 'unrelated text')


def test_only_the_measured_fable_tracks_get_the_extra_mastering_pass():
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'mastering_config')
    constants = {'MASTERING_CONFIG': {'filters': ['normal-encode']},
                 'LOUDNESS_FILTERS': ['normal-encode']}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), constants)
    mastering = constants['mastering_config']
    for lesson in ('12_map_barcodes', '07_mask'):
        assert mastering(lesson, 'en', 'bm_fable')['filters'] == [
            'normal-encode', 'normal-encode,volume=-2dB']
    for args in [('12_map_barcodes', 'en', 'af_heart'),
                 ('07_mask', 'en', 'af_heart'), ('08_measure', 'en', 'bm_fable'),
                 ('07_mask', 'fr', 'bm_fable'), ('12_map_barcodes', 'fr', 'bm_fable')]:
        assert mastering(*args)['filters'] == ['normal-encode']
    assert constants['MASTERING_CONFIG']['filters'] == ['normal-encode']
