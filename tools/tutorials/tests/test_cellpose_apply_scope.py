from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose_cellpose_apply_capture import require_scope


def observations():
    return (dict(pipeline=dict(accepted=True),accuracy_validated=False,
        ground_truth_used=False,preview=dict(batch_mask_equal=False),
        filter=dict(raw_unchanged=True,restored=True)),
        dict(accepted=True,preview_default_reference=dict(differs_from_batch=True)))


def test_scope_requires_actual_batch_pass_before_disclosing_preview_difference():
    proof,reference=observations();require_scope(proof,reference)
    for target in ('pipeline','reference'):
        p,r=deepcopy(proof),deepcopy(reference)
        (p['pipeline'] if target=='pipeline' else r)['accepted']=False
        with pytest.raises(ValueError,match='batch and independent'):require_scope(p,r)


def test_preview_mismatch_and_missing_accuracy_cannot_be_relabelled():
    proof,reference=observations();require_scope(proof,reference)
    for key in ('accuracy_validated','ground_truth_used','batch_mask_equal','differs_from_batch'):
        p,r=deepcopy(proof),deepcopy(reference)
        if key in p:p[key]=True
        elif key=='batch_mask_equal':p['preview'][key]=True
        else:r['preview_default_reference'][key]=False
        with pytest.raises(ValueError,match='observed preview mismatch'):require_scope(p,r)


def test_filtered_result_must_restore_after_a_real_positive_case():
    proof,reference=observations();require_scope(proof,reference)
    for key in ('raw_unchanged','restored'):
        broken=deepcopy(proof);broken['filter'][key]=False
        with pytest.raises(ValueError,match='Filtering must'):require_scope(broken,reference)
