"""
Tests for the computer-vision training path.

Full training runs (`spacr.deep_spacr.train_test_model`) take minutes even
on a GPU; here we exercise the building blocks and a single-batch forward
pass so regressions in the training stack are caught cheaply:

  1. `choose_model` returns a torchvision-backed classifier with the
     right output shape.
  2. A synthetic PNG dataset laid out as spacr expects (train/<class>/*.png)
     can be turned into a DataLoader.
  3. One forward + backward pass on GPU produces a finite loss and the
     model's parameters actually receive gradients.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu


def _needs_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _needs_torch_cuda(), reason="no CUDA available")


# ---------------------------------------------------------------------------
# Synthetic PNG dataset laid out as spacr's `train/<class>/*.png`
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_png_dataset(tmp_path, rng):
    """Two-class RGB PNG dataset: 6 images/class in train/, 3 in test/,
    all 64x64 to keep the fixture fast."""
    from PIL import Image

    def _emit(root, cls, n):
        out = root / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            # Give the two classes different intensity distributions so
            # gradient descent has a signal to chase.
            if cls == "nc":
                arr = (arr // 3).astype(np.uint8)   # darker
            Image.fromarray(arr).save(out / f"{cls}_{i:03d}.png")

    for cls in ("nc", "pc"):
        _emit(tmp_path / "train", cls, 6)
        _emit(tmp_path / "test", cls, 3)
    return {"src": str(tmp_path), "classes": ["nc", "pc"]}


# ---------------------------------------------------------------------------
# 1. choose_model builds a classifier with the requested head width
# ---------------------------------------------------------------------------

@needs_gpu
@pytest.mark.parametrize("num_classes", [2, 4])
def test_choose_model_builds_classifier_with_correct_head(num_classes):
    import torch
    from spacr.utils import choose_model

    device = torch.device("cuda:0")
    model = choose_model(
        "resnet18", device, init_weights=False,
        dropout_rate=0.0, num_classes=num_classes, verbose=False,
    )
    assert model is not None
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 224, 224, device=device)
        out = model(x)
    # Output shape: (batch, num_classes)
    assert out.shape == (2, num_classes), f"expected (2,{num_classes}), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Synthetic dataset can be turned into a DataLoader by generate_loaders
# ---------------------------------------------------------------------------

@needs_gpu
def test_generate_loaders_produces_dataloader(synth_png_dataset):
    from spacr.io import generate_loaders

    result = generate_loaders(
        src=synth_png_dataset["src"], mode="train",
        image_size=64, batch_size=4,
        classes=synth_png_dataset["classes"],
        channels=["r", "g", "b"], augment=False, verbose=False,
        n_jobs=0,
    )
    # generate_loaders returns a tuple; the first element should be a
    # DataLoader.
    assert result is not None
    if isinstance(result, tuple):
        loader = result[0]
    else:
        loader = result
    from torch.utils.data import DataLoader
    assert isinstance(loader, DataLoader), (
        f"expected DataLoader, got {type(loader).__name__}"
    )


# ---------------------------------------------------------------------------
# 3. One forward+backward pass through resnet18 on synthetic images
#    produces a finite loss and non-zero gradients.
# ---------------------------------------------------------------------------

@needs_gpu
def test_single_training_step_produces_finite_loss(synth_png_dataset):
    """The whole training loop is exercised end-to-end for a single step:
    dataset -> loader -> resnet18 -> loss -> backward -> gradients."""
    import torch
    import torch.nn as nn
    from spacr.io import generate_loaders
    from spacr.utils import choose_model

    device = torch.device("cuda:0")

    result = generate_loaders(
        src=synth_png_dataset["src"], mode="train",
        image_size=64, batch_size=4,
        classes=synth_png_dataset["classes"],
        channels=["r", "g", "b"], augment=False, verbose=False,
        n_jobs=0,
    )
    loader = result[0] if isinstance(result, tuple) else result

    model = choose_model(
        "resnet18", device, init_weights=False,
        dropout_rate=0.0, num_classes=2, verbose=False,
    )
    model = model.to(device).train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Grab one batch and run one optimizer step.
    for batch in loader:
        # DataLoader may yield (images, labels) OR (images, labels, meta).
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            pytest.skip(f"unexpected loader batch shape: {type(batch)}")
        # Move to device, coerce labels to long integer class indices.
        images = images.to(device).float()
        if images.ndim == 3:                       # (C,H,W) -> (1,C,H,W)
            images = images.unsqueeze(0)
        # resnet18 needs 3 input channels; if the loader yielded 1 or 4,
        # take the first 3 (or repeat single channel).
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] > 3:
            images = images[:, :3]

        if isinstance(labels, torch.Tensor):
            labels = labels.to(device).long()
        else:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()

        assert torch.isfinite(loss).item(), f"loss is not finite: {loss.item()}"

        # At least one parameter should have received a non-zero gradient.
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient after backward()"
        assert any(g.abs().sum().item() > 0 for g in grads), (
            "all gradients are zero — backward pass did nothing"
        )

        optimizer.step()
        break
    else:
        pytest.fail("DataLoader produced no batches for the training step")