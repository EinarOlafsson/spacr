"""A one-sentence repair must not silently become a whole-track rewrite."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_platform_heart_cuda import replace_samples, select_sentence


def metadata():
    return {'language':'en', 'voice':'af_heart', 'scenes':[
        {'sentences':[{'text':'First sentence.'}, {'text':'Install CUDA support.'}]},
        {'sentences':[{'text':'Otherwise use CUDA acceleration.'}]}]}


def test_only_first_cuda_sentence_is_selected_without_mutating_the_input():
    source = metadata()
    before = deepcopy(source)
    assert select_sentence(source) == (0, 1, source['scenes'][0]['sentences'][1])
    assert source == before


@pytest.mark.parametrize('key,value', [('voice','af_bella'), ('language','es')])
def test_another_voice_or_language_is_refused(key, value):
    source = metadata()
    select_sentence(source)
    source[key] = value
    with pytest.raises(ValueError, match='reported English Heart'):
        select_sentence(source)


def test_changed_first_occurrence_is_refused_instead_of_repairing_the_next_one():
    source = metadata()
    select_sentence(source)
    source['scenes'][0]['sentences'][1]['text'] = 'CUDA changed.'
    with pytest.raises(ValueError, match='no longer the reported'):
        select_sentence(source)


def sample_contract(function):
    original = np.arange(100, dtype=np.float32)
    before = original.copy()
    replacement = np.full(10, -1, dtype=np.float32)
    result = function(original, replacement, 30, 40)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(result[:30], original[:30])
    np.testing.assert_array_equal(result[30:40], replacement)
    np.testing.assert_array_equal(result[40:], original[40:])


def test_only_authorized_samples_change():
    sample_contract(replace_samples)


def test_contract_detects_a_source_mutation_that_does_no_repair():
    path = Path(__file__).resolve().parents[1] / 'repair_platform_heart_cuda.py'
    source = path.read_text()
    mutant = source.replace('result[start:end] = replacement', 'pass  # deliberately omit repair', 1)
    assert source != mutant
    namespace = {'__name__':'mutant', '__file__':str(path)}
    exec(compile(mutant, str(path), 'exec'), namespace)
    with pytest.raises(AssertionError):
        sample_contract(namespace['replace_samples'])


@pytest.mark.parametrize('start,end,length', [(-1,9,10), (90,110,20), (20,20,0), (30,40,9)])
def test_bounds_and_duration_mismatches_are_refused(start, end, length):
    with pytest.raises(ValueError, match='fit the existing sentence exactly'):
        replace_samples(np.zeros(100), np.zeros(length), start, end)
