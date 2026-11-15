"""Table-driven tests of reading, writing and refusing spaCR PyTorch checkpoints.

New spaCR artifacts are versioned mappings; older releases wrote whole
``nn.Module`` objects or ``{'model': state_dict}`` mappings, and other tools
write ``{'state_dict': ...}`` files.  ``load_model_artifact`` reads all of them
and refuses every other shape with a sentence naming the file.  The random
generator state that travels with an artifact is restored for every backend
that is present and left alone for every one that is not.  Everything is CPU
only and offline; GPU/Metal backends are stood in for by small doubles.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from spacr import torch_artifacts as TA  # noqa: E402

V = TA.ARTIFACT_VERSION


def _tiny_model(seed=0):
    """A two-parameter module whose weights depend on ``seed``."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 2))
    model.model_name = "tiny"
    model.num_classes = 2
    return model


def _config(**overrides):
    """The configuration ``model_configuration`` records for a plain module."""
    config = {"model_name": "tiny", "num_classes": 2, "dropout_rate": None,
              "use_checkpoint": False, "image_size": 224, "multilabel": False}
    config.update(overrides)
    return config


def _legacy_config(**overrides):
    """What a pre-versioned mapping is read as when it names little."""
    return _config(model_name="maxvit_t", **overrides)


def _same_weights(left, right):
    """Whether two modules hold identical state dictionaries."""
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_load_model_artifact_reads_every_checkpoint_shape(tmp_path, subtests):
    """Current, legacy mapping and whole-module files load; their metadata is exact."""
    saved = _tiny_model(seed=1)
    state = saved.state_dict()
    cases = [
        ("a current artifact loads into the model supplied",
         lambda p: TA.save_model_artifact(saved, p), True, False,
         {"legacy": False, "model_config": _config(), "artifact_role": "model",
          "training_state": {"epoch": 0, "best_metric": None,
                             "epochs_without_improvement": 0}}),
        ("a bare state_dict mapping is a legacy artifact",
         lambda p: TA.atomic_torch_save(
             {"state_dict": state, "num_classes": 3, "image_size": 128}, p),
         True, False,
         {"legacy": True, "training_state": {},
          "model_config": _legacy_config(num_classes=3, image_size=128)}),
        ("a pre-versioned model mapping is a legacy artifact",
         lambda p: TA.atomic_torch_save({"model": state,
                                         "model_name": "tiny"}, p),
         True, False,
         {"legacy": True, "training_state": {}, "model_config": _config()}),
        ("a whole module file comes back as its own module",
         lambda p: TA.atomic_torch_save(saved, p), False, True,
         {"legacy": True, "artifact_role": "legacy_full_module",
          "model_config": _config(), "training_state": {}}),
    ]
    for case_id, write, in_place, own_module, expected in cases:
        with subtests.test(msg=case_id):
            path = str(tmp_path / f"{case_id[:20].replace(' ', '_')}.pth")
            write(path)
            target = _tiny_model(seed=2)
            loaded, payload = TA.load_model_artifact(path, model=target)
            assert (loaded is target) is in_place
            assert (loaded is not target) is own_module
            assert _same_weights(loaded, saved)
            assert {k: payload[k] for k in expected} == expected

    with subtests.test(msg="an artifact naming its architecture rebuilds it"):
        from spacr.utils import TorchModel

        small = TorchModel(model_name="resnet18", pretrained=False,
                           num_classes=3, image_size=64)
        path = TA.save_model_artifact(small, str(tmp_path / "resnet.pth"))
        restored, payload = TA.load_model_artifact(path)
        assert payload["legacy"] is False
        assert restored is not small
        assert (type(restored), restored.model_name, restored.num_classes,
                restored.pretrained) == (TorchModel, "resnet18", 3, False)
        assert _same_weights(restored, small)


def test_load_model_artifact_refuses_what_it_cannot_read(tmp_path, subtests):
    """Each unreadable shape raises ValueError with a sentence naming the file."""
    def artifact(**changes):
        payload = TA.make_model_artifact(_tiny_model())
        payload.update(changes)
        return payload

    cases = [
        ("an artifact from a future spaCR is refused by version",
         artifact(artifact_version=V + 7), True,
         lambda p: f"Unsupported spaCR model artifact version {V + 7}; this "
                   f"installation supports version {V}."),
        ("a mapping with no model state anywhere is refused",
         {"learning_rate": 0.001, "epochs": 20}, True,
         lambda p: f"Unsupported PyTorch checkpoint mapping at '{p}': no "
                   "model state dictionary was found."),
        ("a model state that is not a mapping is refused",
         artifact(model_state_dict=["not", "a", "state", "dict"]), True,
         lambda p: f"Model state in '{p}' is not a state dictionary."),
        ("an empty config needs a model passed in",
         artifact(model_config={}), False,
         lambda p: f"Checkpoint {p!r} does not describe its architecture; "
                   "provide an initialized model explicitly."),
        ("a config naming no architecture refuses to guess",
         artifact(model_config={"num_classes": 3}), False,
         lambda p: f"Checkpoint {p!r} does not describe its architecture; "
                   "provide an initialized model explicitly."),
        ("a tensor file is neither a module nor a mapping",
         torch.zeros(3), False,
         lambda p: f"Unsupported PyTorch artifact at {p!r}: expected an "
                   "nn.Module or checkpoint mapping."),
    ]
    for index, (case_id, payload, with_model, message) in enumerate(cases):
        with subtests.test(msg=case_id):
            path = TA.atomic_torch_save(payload, str(tmp_path / f"bad{index}.pth"))
            model = _tiny_model() if with_model else None
            with pytest.raises(ValueError, match=f"^{re.escape(message(path))}$"):
                TA.load_model_artifact(path, model=model)

    with subtests.test(msg="the nameless checkpoint loads once a model is given"):
        path = str(tmp_path / "bad4.pth")
        target = _tiny_model(seed=3)
        restored, payload = TA.load_model_artifact(path, model=target)
        assert restored is target
        assert payload["model_config"] == {"num_classes": 3}


def test_restore_rng_state_puts_back_every_backend_that_is_present(
        monkeypatch, subtests):
    """CPU, CUDA and Metal generators round-trip; absent ones stay untouched."""
    with subtests.test(msg="no rng state at all is a no-op"):
        before = torch.get_rng_state().clone()
        TA.restore_rng_state(None)
        TA.restore_rng_state({})
        assert torch.equal(torch.get_rng_state(), before)

    with subtests.test(msg="the cpu, python and numpy streams continue exactly"):
        import random

        import numpy as np

        state = TA.capture_rng_state()
        expected = (random.random(), float(np.random.rand()),
                    float(torch.rand(1)))
        TA.restore_rng_state(state)
        assert (random.random(), float(np.random.rand()),
                float(torch.rand(1))) == expected

    with subtests.test(msg="the gpu generators travel with the artifact"):
        restored = []
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            get_rng_state_all=lambda: ["device-0-state"],
            set_rng_state_all=restored.append)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "cuda", fake_cuda, raising=False)
            state = TA.capture_rng_state()
            assert state["torch_cuda"] == ["device-0-state"]
            TA.restore_rng_state(state)
        assert restored == [["device-0-state"]]

    with subtests.test(msg="metal state round-trips; an absent backend is left alone"):
        generator = torch.Generator(device="cpu").manual_seed(179)
        original = generator.get_state().clone()
        calls = []

        def set_state(value):
            calls.append(value.clone())
            generator.set_state(value)

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_available", lambda: False)
            patch.setattr(torch.backends, "mps",
                          SimpleNamespace(is_available=lambda: True))
            patch.setattr(torch, "mps", SimpleNamespace(
                get_rng_state=lambda: generator.get_state(),
                set_rng_state=set_state), raising=False)
            state = TA.capture_rng_state()
            assert torch.equal(state["torch_mps"], original)
            expected = torch.rand(6, generator=generator)
            TA.restore_rng_state({"torch_mps": state["torch_mps"]})
            assert len(calls) == 1 and calls[0].device.type == "cpu"
            assert torch.equal(torch.rand(6, generator=generator), expected)

            patch.setattr(torch.backends.mps, "is_available", lambda: False)
            before = generator.get_state().clone()
            assert "torch_mps" not in TA.capture_rng_state()
            TA.restore_rng_state({"torch_mps": original})
            assert len(calls) == 1
            assert torch.equal(generator.get_state(), before)

            patch.setattr(torch.backends, "mps", None)
            assert "torch_mps" not in TA.capture_rng_state()
            TA.restore_rng_state({"torch_mps": original})
            assert len(calls) == 1
