"""Synthetic evidence fixtures test the disclosure, not a simulation result."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from power_review import INPUTS, review_baselines


def records():
    row = dict.fromkeys(INPUTS, 1)
    row.update(imaging_n_cells_per_well_mu=120, n_wells_per_screen=96)
    return dict(spec=dict(cells_per_well=120, n_plates=1, wells_per_plate=96, n_replicates=2),
        cells_scan=[dict(row, seed_used=s) for s in (1, 2)],
        wells_scan=[dict(row, seed_used=s) for s in (3, 4)],
        answer='Going to 96 wells would take that to 100%.')


def test_same_inputs_different_draws_reject_advice():
    result = review_baselines(records())
    assert result['same_well_count_advice_rejected']
    assert result['independent_seed_groups'] == [[1, 2], [3, 4]]
    assert result['recommendation_approved'] is False


@pytest.mark.parametrize('mutation,message', [
    ('inputs', 'different nominal inputs'), ('seeds', 'distinct recorded'),
    ('headline', 'contradictory headline'), ('missing', 'every baseline')])
def test_changed_premise_is_not_accepted(mutation, message):
    data = deepcopy(records())
    assert review_baselines(data)['same_well_count_advice_rejected']
    if mutation == 'inputs': data['wells_scan'][0]['n_genes_in_library'] = 99
    elif mutation == 'seeds': data['wells_scan'][0]['seed_used'] = 1
    elif mutation == 'headline': data['answer'] = 'Going to 192 wells would take that to 100%.'
    else: data['wells_scan'].pop()
    with pytest.raises(ValueError, match=message): review_baselines(data)
