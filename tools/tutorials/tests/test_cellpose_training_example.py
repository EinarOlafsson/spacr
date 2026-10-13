"""Check training-evidence guards without running inference or training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1]/'train_cellpose_example.py'
spec = importlib.util.spec_from_file_location('training_example', PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def pairs():
    return {'first': (np.array([[.1,.2],[.3,.4]], dtype=np.float32),
                       np.array([[0,1],[2,2]], dtype=np.uint16)),
            'second': (np.array([[.5,.6],[.7,.8]], dtype=np.float32),
                       np.array([[0,4],[5,5]], dtype=np.uint16))}


def test_every_pair_and_plot_can_be_matched_in_actual_shuffled_order():
    expected = pairs(); order = ['second','first']
    assert example.check_arrays([expected[n][0] for n in order],
                                [expected[n][1] for n in order],expected)==order


def test_wrong_image_refused_after_correct_image_was_accepted():
    expected = pairs(); names = list(expected)
    images = [expected[n][0].copy() for n in names]
    labels = [expected[n][1].copy() for n in names]
    assert example.check_arrays(images,labels,expected)==names
    images[0][1,1] += .01
    with pytest.raises(ValueError,match='pixels'):
        example.check_arrays(images,labels,expected)


def test_wrong_compartment_refused_after_correct_mask_was_accepted():
    expected = pairs(); names = list(expected)
    images = [expected[n][0].copy() for n in names]
    labels = [expected[n][1].copy() for n in names]
    assert example.check_arrays(images,labels,expected)==names
    labels[0][1,1] = 9
    with pytest.raises(ValueError,match='pixels'):
        example.check_arrays(images,labels,expected)


def test_missing_pair_and_duplicate_refused_after_complete_counterpart():
    expected=pairs(); images=[v[0] for v in expected.values()]; labels=[v[1] for v in expected.values()]
    assert len(example.check_arrays(images,labels,expected))==2
    with pytest.raises(ValueError,match='every pair'):
        example.check_arrays(images[:1],labels[:1],expected)
    with pytest.raises(ValueError,match='unique'):
        example.check_arrays([images[0]]*2,[labels[0]]*2,expected)


def test_only_actual_trainable_weight_change_counts_not_dtype_or_diameter():
    initial={'weight':example.tensor_hash(torch.tensor([1.,2.],dtype=torch.bfloat16))}
    positive={'weight':torch.tensor([1.01,2.]),'diam_labels':torch.tensor([20.])}
    assert example.changed_trainable(initial,positive)==['weight']
    unchanged={'weight':torch.tensor([1.,2.],dtype=torch.float32),'diam_labels':torch.tensor([30.])}
    with pytest.raises(ValueError,match='No actual trainable weight changed'):
        example.changed_trainable(initial,unchanged)


def test_missing_and_nonfinite_weights_refused_after_finite_counterpart():
    initial={'weight':example.tensor_hash(torch.tensor([1.,2.]))}
    assert example.changed_trainable(initial,{'weight':torch.tensor([1.,3.])})==['weight']
    with pytest.raises(ValueError,match='cover'):
        example.changed_trainable(initial,{'diam_labels':torch.tensor([30.])})
    with pytest.raises(ValueError,match='nonfinite'):
        example.changed_trainable(initial,{'weight':torch.tensor([1.,float('nan')])})
