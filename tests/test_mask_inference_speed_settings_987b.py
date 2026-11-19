"""mask_eval_batch_size and mask_inference_precision (item 987b): defaults keep today's masks."""
import pytest

from spacr import object as O
from spacr.settings import set_default_settings_preprocess_generate_masks


def test_defaults_keep_the_old_behaviour():
    """A fresh settings dict asks for the old tile batch and the accelerator's number format."""
    settings = set_default_settings_preprocess_generate_masks({'src': '/x'})
    assert settings['mask_eval_batch_size'] == 0
    assert settings['mask_inference_precision'] == 'auto'


@pytest.mark.parametrize('value,fields,expected', [
    (0, 5, 5), (None, 7, 7), ('', 3, 3), (64, 5, 64), ('16', 5, 16)])
def test_the_tile_batch_follows_the_setting(value, fields, expected):
    """0 keeps one tile per field; a positive value is used as given."""
    assert O._eval_tile_batch({'mask_eval_batch_size': value}, fields) == expected
    assert O._eval_tile_batch({}, 4) == 4


@pytest.mark.parametrize('value', [-1, 'many'])
def test_a_bad_tile_batch_is_refused(value):
    """A negative or non-numeric tile batch raises, naming the setting."""
    with pytest.raises(ValueError, match='mask_eval_batch_size'):
        O._eval_tile_batch({'mask_eval_batch_size': value}, 5)


def test_precision_auto_leaves_the_accelerator_choice():
    """auto changes nothing; bf16 and fp32 force the weights' format."""
    base = {'gpu': True, 'use_bfloat16': True}
    assert O._with_inference_precision(base, {}) == base
    assert O._with_inference_precision(base, {'mask_inference_precision': 'auto'}) == base
    assert O._with_inference_precision(base, {'mask_inference_precision': 'fp32'})['use_bfloat16'] is False
    assert O._with_inference_precision({'gpu': True}, {'mask_inference_precision': 'BF16'})['use_bfloat16'] is True
    assert base == {'gpu': True, 'use_bfloat16': True}


def test_a_bad_precision_is_refused():
    """Only auto, bf16 and fp32 are accepted."""
    with pytest.raises(ValueError, match='mask_inference_precision'):
        O._with_inference_precision({}, {'mask_inference_precision': 'fp16'})
