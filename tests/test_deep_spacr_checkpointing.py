"""Regression tests for activation and disk checkpoint correctness."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn


class _CheckpointedClassifier(nn.Module):
    def __init__(self, enabled: bool):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4, 2)
        self.enabled = enabled

    def forward(self, x):
        from spacr.utils import _checkpoint_module

        features = (
            _checkpoint_module(self.backbone, self.backbone, x)
            if self.enabled else self.backbone(x)
        )
        return self.head(features.flatten(1))


@pytest.mark.parametrize("enabled", [False, True])
def test_every_layer_receives_gradients_with_checkpointing_on_or_off(enabled):
    model = _CheckpointedClassifier(enabled).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.functional.cross_entropy(
        model(torch.randn(4, 3, 16, 16)), torch.tensor([0, 1, 0, 1]))
    loss.backward()

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in model.parameters())
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )


def test_checkpoint_recomputation_does_not_double_update_batchnorm():
    model = _CheckpointedClassifier(True).train()
    batchnorm = model.backbone[1]

    output = model(torch.randn(4, 3, 16, 16))
    assert int(batchnorm.num_batches_tracked) == 1
    output.sum().backward()
    assert int(batchnorm.num_batches_tracked) == 1


def test_training_artifact_round_trips_optimizer_scheduler_and_rng(tmp_path):
    from spacr.torch_artifacts import (
        load_model_artifact,
        restore_training_state,
        save_model_artifact,
    )

    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    nn.functional.cross_entropy(
        model(torch.randn(4, 3)), torch.tensor([0, 1, 0, 1])
    ).backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "checkpoint.pth"
    save_model_artifact(
        model, path, optimizer=optimizer, scheduler=scheduler, epoch=7,
        best_metric=0.83, epochs_without_improvement=2,
        metrics={"accuracy": 0.83}, preprocessing={"normalize": True},
        classes=["negative", "positive"], channels=["r", "g", "b"],
        artifact_role="best",
    )

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    restored_model = nn.Linear(3, 2)
    restored, payload = load_model_artifact(
        path, map_location="cpu", model=restored_model)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    state = restore_training_state(
        payload, optimizer=restored_optimizer, scheduler=restored_scheduler)

    assert state == {
        "epoch": 7,
        "best_metric": pytest.approx(0.83),
        "epochs_without_improvement": 2,
    }
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert payload["preprocessing"] == {"normalize": True}
    assert payload["classes"] == ["negative", "positive"]
    assert payload["dependencies"]["torch"]
    # The artifact restores the states captured immediately before the three
    # expected values above were generated.
    assert random.random() == pytest.approx(expected_python)
    assert np.random.random() == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(1), expected_torch)


def test_atomic_save_leaves_existing_checkpoint_on_failure(tmp_path, monkeypatch):
    from spacr import torch_artifacts

    target = tmp_path / "model.pth"
    target.write_bytes(b"previous-good-checkpoint")

    def fail(_payload, temporary):
        with open(temporary, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch_artifacts.torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        torch_artifacts.atomic_torch_save({"new": True}, target)

    assert target.read_bytes() == b"previous-good-checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []
