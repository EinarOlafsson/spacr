"""`use_checkpoint=True` must not turn training into a linear probe.

Reported as issue #76. `torch.utils.checkpoint`'s REENTRANT implementation
decides whether to build a backward graph from whether its INPUT TENSORS
require grad. A batch straight from the dataloader has
``requires_grad=False``, and the backbone's parameters are captured in the
closure where that check cannot see them -- so the checkpointed output
carried no ``grad_fn`` and gradients stopped at the backbone boundary.

Nothing crashed. The classifier sits outside the checkpoint, its own weights
require grad, so the loss had a valid graph and ``backward()`` succeeded.
Gradients reached the head and nothing else, and every run was silently a
linear probe on a frozen pretrained network -- with the reported accuracy of
one.

The fix is `_checkpoint_module`, which passes ``use_reentrant=False``. This
test exists because the failure is INVISIBLE: no error, no warning, a
plausible accuracy, and the only symptom is a model that learns less than it
should.
"""

import pytest

torch = pytest.importorskip("torch")

from spacr.utils import TorchModel


def _grad_counts(use_checkpoint):
    model = TorchModel(model_name="resnet18", pretrained=False,
                       use_checkpoint=use_checkpoint, num_classes=2)
    model.train()
    model(torch.randn(2, 3, 64, 64)).float().sum().backward()

    backbone = [p for n, p in model.named_parameters()
                if n.startswith("base_model")]
    head = [p for n, p in model.named_parameters()
            if not n.startswith("base_model")]

    def learned(params):
        return sum(1 for p in params
                   if p.grad is not None and p.grad.abs().sum() > 0)

    return learned(backbone), len(backbone), learned(head), len(head)


@pytest.mark.parametrize("use_checkpoint", [True, False])
def test_every_backbone_parameter_receives_a_gradient(use_checkpoint):
    """The whole backbone learns, checkpointed or not."""
    got_bb, n_bb, got_head, n_head = _grad_counts(use_checkpoint)

    assert n_bb > 20, "the backbone is too small for this test to mean much"
    assert got_bb == n_bb, (
        f"use_checkpoint={use_checkpoint}: only {got_bb} of {n_bb} backbone "
        f"parameters got a gradient -- the backbone is frozen and this run is "
        f"a linear probe")
    assert got_head == n_head


def test_checkpointing_is_not_silently_a_linear_probe():
    """The two paths must agree about WHICH parameters learn.

    Stated as a comparison because that is the defect: checkpointing was
    supposed to trade compute for memory, not change what trains.
    """
    on = _grad_counts(True)
    off = _grad_counts(False)
    assert on == off, (
        f"checkpointing changed which parameters learn: {on} vs {off}")


def test_the_helper_does_not_use_the_reentrant_implementation():
    """Pinned at the source, because the default flips between torch
    versions and the failure it causes is silent."""
    import inspect

    from spacr import utils

    source = inspect.getsource(utils._checkpoint_module)
    assert "use_reentrant=False" in source, (
        "the reentrant checkpoint is back; it decides on input tensors alone "
        "and a dataloader batch never requires grad")


def test_the_backbone_runner_goes_through_the_helper():
    """A direct `checkpoint(...)` call would reintroduce the bug."""
    import inspect

    from spacr import utils

    source = inspect.getsource(utils.TorchModel._run_backbone_raw)
    assert "_checkpoint_module" in source
