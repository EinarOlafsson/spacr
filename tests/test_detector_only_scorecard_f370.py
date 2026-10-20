"""A detector's measured scores remain visible without segmentation metrics."""
from types import SimpleNamespace

import pytest

from spacr.model_zoo import scorecard_html


@pytest.mark.parametrize('metrics,expected', [
    ({'mAP50': 0.0}, ('mAP50', '0.0')),
    ({'mAP50_95': 0.375}, ('mAP50–95', '0.375')),
    ({'mAP50': {'finetuned': 0.75, 'vanilla': 0.5}},
     ('mAP50', '0.7500', '0.5000')),
])
def test_detector_only_measurements_are_visible_without_mutation(metrics, expected):
    import copy

    before = copy.deepcopy(metrics)
    card = scorecard_html(SimpleNamespace(name='detector', metrics=metrics))
    assert '<table' in card
    for text in expected:
        assert text in card
    assert metrics == before


@pytest.mark.parametrize('metrics', [{}, {'note': 'not measured'}, {'training_loss': 0.25}])
def test_unrelated_metadata_does_not_invent_a_scorecard(metrics):
    assert scorecard_html(SimpleNamespace(name='detector', metrics=metrics)) == ''
