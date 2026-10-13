"""A copied number is not enough when it names the wrong random generator."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_methods import check_recorded_seed


@pytest.mark.parametrize('seed',[0,17,42])
def test_matching_declared_seed_in_digest_and_prose(seed):
    check_recorded_seed(seed, {'run':{'seed':seed}}, f'The random seed was {seed}.')


@pytest.mark.parametrize('value,text',[(42,'The random seed was 42.'),
    (42,'The random seed was 0.'),(0,'The random seed was 42.'),(None,'No seed was recorded.')])
def test_wrong_seed_or_prose_after_positive(value,text):
    check_recorded_seed(0,{'run':{'seed':0}},'The random seed was 0.')
    with pytest.raises(ValueError):
        check_recorded_seed(0,{'run':{'seed':value}},text)
