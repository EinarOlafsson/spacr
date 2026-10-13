"""The best model must reach disk even when accuracy never crosses a threshold.

Reported as issue #77. `_save_model` used to write a file only every 100th
epoch (or the last), or when accuracy crossed one of the hardcoded thresholds
[0.99, 0.98, 0.95, 0.94]. On a run whose validation accuracy peaks below
0.94 -- ordinary for real biology -- none of those fire, so the ONLY file
written was the final epoch's.

The training loop tracked `best_model_path` separately, but that variable can
only point at a file `_save_model` actually wrote, so it degraded silently to
"the last epoch" and defeated its own purpose.

The reported run peaked at 0.929 validation around epoch 16 and had memorised
the training set by epoch 38; the only file saved was epoch 100, deep into
the overfit region. Every downstream step that applies "the best model" --
top-confidence example images, scoring -- then used the WORST model of the
run.
"""

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from spacr.io import _save_model


BELOW_EVERY_THRESHOLD = 0.929   # the accuracy from the issue


def _model():
    return nn.Linear(4, 2)


def test_a_new_best_is_written_even_below_every_threshold(tmp_path):
    """The whole defect, at the reported number."""
    _save_model(_model(), "cnn", {"accuracy": BELOW_EVERY_THRESHOLD},
                str(tmp_path), epoch=16, epochs=100,
                val_dict={"accuracy": BELOW_EVERY_THRESHOLD},
                is_best=True, best_metric=BELOW_EVERY_THRESHOLD)

    written = os.listdir(tmp_path)
    assert any("_best_" in name for name in written), (
        f"no best checkpoint written at accuracy {BELOW_EVERY_THRESHOLD}, "
        f"which is below every threshold: {written}")


def test_a_worse_epoch_writes_nothing(tmp_path):
    """Otherwise 'best' would just mean 'most recent' again."""
    _save_model(_model(), "cnn", {"accuracy": 0.870}, str(tmp_path),
                epoch=38, epochs=100, val_dict={"accuracy": 0.870},
                is_best=False, best_metric=BELOW_EVERY_THRESHOLD)

    assert not os.listdir(tmp_path), (
        "a non-improving epoch wrote a file, so the best checkpoint can be "
        "overwritten by a worse one")


def test_the_best_file_keeps_one_stable_name(tmp_path):
    """A later best must REPLACE the earlier one, not accumulate.

    A directory of best_epoch_16, best_epoch_23, ... puts the burden of
    knowing which is best back on the reader.
    """
    for epoch, acc in ((16, 0.929), (23, 0.941)):
        _save_model(_model(), "cnn", {"accuracy": acc}, str(tmp_path),
                    epoch=epoch, epochs=100, val_dict={"accuracy": acc},
                    is_best=True, best_metric=acc)

    best = [n for n in os.listdir(tmp_path) if "_best_" in n]
    assert len(best) == 1, f"the best checkpoint accumulated files: {best}"


def test_the_returned_path_prefers_the_best(tmp_path):
    """The caller stores whatever this returns as `best_model_path`."""
    returned = _save_model(
        _model(), "cnn", {"accuracy": BELOW_EVERY_THRESHOLD}, str(tmp_path),
        epoch=100, epochs=100, val_dict={"accuracy": BELOW_EVERY_THRESHOLD},
        is_best=True, best_metric=BELOW_EVERY_THRESHOLD)

    assert returned and "_best_" in os.path.basename(returned), (
        f"the final epoch is also a 'last' save, and the return preferred it "
        f"over the best: {returned}")


def test_the_last_epoch_is_still_written(tmp_path):
    """The best checkpoint must not have cost us the last one."""
    _save_model(_model(), "cnn", {"accuracy": 0.5}, str(tmp_path),
                epoch=100, epochs=100, val_dict={"accuracy": 0.5},
                is_best=False, best_metric=0.9)

    assert any("_last_" in n for n in os.listdir(tmp_path))
