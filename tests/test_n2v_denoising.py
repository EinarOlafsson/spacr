"""Self-supervised Noise2Void denoising before segmentation.

CAREamics runs in an environment of its own, so these tests stand in for its
worker: they check what spaCR sends it, where the models are kept, that a
resumed run reuses them, that denoising runs before the enhancement chain,
and that the training reaches the run's provenance record.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import psf_pipeline as PP
from spacr import _segmentation_backends as SB


def _stack(folder, count=3, shape=(80, 80, 2)):
    """Write ``count`` raw fields, channel-last, into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        np.save(folder / f"plate1_A0{index + 1}_1.npy",
                rng.poisson(5, shape).astype(np.uint16))
    return folder


class _Trainer:
    """Stands in for ``_n2v_train``: writes a checkpoint, keeps the calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, planes, output, *, epochs):
        self.calls.append((len(planes), output, epochs))
        with open(output, "wb") as stream:
            stream.write(f"model {output} {epochs}".encode())
        return {"checkpoint": output, "epochs": epochs, "patch": 64,
                "train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6],
                "device": "cpu", "seconds": 1.0, "careamics": "0.3.4"}


def _settings(**extra):
    return {"n2v_denoise": True, "n2v_model": "", "n2v_epochs": 3, **extra}


def test_off_means_no_plan_and_no_training(tmp_path):
    trainer = _Trainer()
    assert PP._prepare_n2v({"n2v_denoise": False}, tmp_path, [0],
                           train=trainer) is None
    assert trainer.calls == []
    assert not PP.processing_requested({"n2v_denoise": False})
    assert PP.processing_requested({"n2v_denoise": True})


def test_one_model_per_channel_is_trained_on_the_runs_own_fields(tmp_path):
    _stack(tmp_path / "stack")
    trainer = _Trainer()
    plan = PP._prepare_n2v(_settings(), tmp_path, [1, 0], train=trainer)

    assert [call[0] for call in trainer.calls] == [3, 3]
    assert plan.channels == (1, 0)
    for channel in (0, 1):
        path = tmp_path / "n2v" / f"channel_{channel}.ckpt"
        assert plan.checkpoints[channel] == str(path)
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["channel"] == channel
        assert record["fields"] == ["plate1_A01_1", "plate1_A02_1",
                                    "plate1_A03_1"]
    provenance = plan.provenance()
    assert provenance["method"].startswith("Noise2Void")
    model = provenance["models"]["0"]
    assert model["train_loss"] == [1.0, 0.5] and len(model["sha256"]) == 64
    assert "checkpoint" not in model and "seconds" not in model


def test_a_resumed_run_reuses_its_models_and_new_epochs_retrain(tmp_path):
    _stack(tmp_path / "stack")
    first = _Trainer()
    plan = PP._prepare_n2v(_settings(), tmp_path, [0], train=first)
    again = _Trainer()
    reused = PP._prepare_n2v(_settings(), tmp_path, [0], train=again)
    assert again.calls == []
    assert reused.provenance() == plan.provenance()

    changed = _Trainer()
    PP._prepare_n2v(_settings(n2v_epochs=5), tmp_path, [0], train=changed)
    assert [call[2] for call in changed.calls] == [5]


def test_a_named_model_folder_is_used_as_it_is(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "channel_0.ckpt").write_bytes(b"weights")
    trainer = _Trainer()
    plan = PP._prepare_n2v(_settings(n2v_model=str(folder)), tmp_path, [0],
                           train=trainer)
    assert trainer.calls == []
    assert plan.checkpoints[0] == str(folder / "channel_0.ckpt")

    with pytest.raises(ValueError, match="no model for channel 1"):
        PP._prepare_n2v(_settings(n2v_model=str(folder)), tmp_path, [1],
                        train=trainer)


def test_bad_epochs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="n2v_epochs"):
        PP._prepare_n2v(_settings(n2v_epochs=0), tmp_path, [0])


def test_each_channel_is_denoised_by_its_own_model_on_every_plane():
    plan = PP._N2VPlan(channels=(2, 0), checkpoints={2: "c2", 0: "c0"},
                       records={2: {}, 0: {}})
    seen = []

    def denoise(plane, model):
        seen.append((model, plane.shape))
        return plane + (100.0 if model == "c2" else 10.0)

    image = np.zeros((2, 5, 6, 2), dtype=np.uint16)
    out = plan.apply(image, denoise=denoise)
    assert out.dtype == np.float32 and out.shape == image.shape
    assert np.all(out[..., 0] == 100.0) and np.all(out[..., 1] == 10.0)
    assert sorted(set(seen)) == [("c0", (5, 6)), ("c2", (5, 6))]
    assert len(seen) == 4
    assert image.max() == 0


def test_denoising_runs_before_the_chain_and_joins_the_record(tmp_path,
                                                              monkeypatch):
    order = []

    class _Plan(PP._N2VPlan):
        def apply(self, image, denoise=None):
            order.append("n2v")
            return np.asarray(image, dtype=np.float32) + 1

    def chain(image, _chain, cancel=None):
        order.append("chain")
        return image

    monkeypatch.setattr(PP, "apply_chain", chain)
    plan = _Plan(channels=(0,), checkpoints={0: "c"},
                 records={0: {"sha256": "ab", "epochs": 3}})
    from spacr.qt.detect_chain import Chain

    session = PP._SegmentationPSFSession(None, tmp_path, [0], "v1",
                                         chain=Chain(denoise="gaussian"),
                                         n2v=plan)
    assert session.processes
    out = session.correct(np.zeros((4, 4, 1)))
    assert order == ["n2v", "chain"] and out.max() == 1
    record = json.loads((tmp_path / "psf" /
                         "segmentation_application.json").read_text())
    assert record["configuration"]["n2v"]["models"]["0"]["sha256"] == "ab"


def test_a_record_made_without_denoising_is_unchanged():
    assert "n2v" not in PP._configuration(None, [0], "v1")


class _Worker:
    """Stands in for CAREamics' running worker."""

    def __init__(self):
        self.requests = []

    def request(self, op, **payload):
        self.requests.append((op, payload))
        if op == "n2v_denoise":
            image = np.load(payload["input"])
            np.save(payload["output"], image.astype(np.float32) / 2)
            return {"output": payload["output"]}
        assert all(np.load(path).ndim == 2 for path in payload["inputs"])
        return {"checkpoint": payload["output"], "epochs": payload["epochs"],
                "protocol": 1, "id": 7, "ok": True}


@pytest.fixture
def installed(monkeypatch, tmp_path):
    state = SB._BackendState(name=SB._CAREAMICS, state=SB._INSTALLED,
                             env=str(tmp_path / "env"))
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: state)
    return _Worker()


def test_training_sends_planes_and_patch_sizes_to_the_worker(installed,
                                                             tmp_path):
    planes = [np.ones((70, 90)), np.ones((64, 64))]
    out = tmp_path / "models" / "channel_0.ckpt"
    reply = SB._n2v_train(planes, out, epochs=4,
                          worker_for=lambda name, env: installed)
    op, payload = installed.requests[0]
    assert op == "n2v_train" and reply["epochs"] == 4
    assert not {"protocol", "id", "ok"} & set(reply)
    assert payload["patch"] == SB._N2V_PATCH and payload["batch"] == SB._N2V_BATCH
    assert payload["output"] == str(out) and out.parent.is_dir()


def test_structn2v_asks_the_worker_for_a_structured_mask_only_when_chosen(
        installed, tmp_path):
    planes = [np.ones((64, 64))]
    SB._n2v_train(planes, tmp_path / "a.ckpt",
                  worker_for=lambda name, env: installed)
    SB._n2v_train(planes, tmp_path / "b.ckpt", struct_axes="horizontal",
                  struct_span=3, worker_for=lambda name, env: installed)
    plain, struct = (payload for _op, payload in installed.requests)
    assert "struct_axes" not in plain
    assert struct["struct_axes"] == "horizontal" and struct["struct_span"] == 3


def test_training_refuses_planes_smaller_than_a_patch(installed, tmp_path):
    with pytest.raises(ValueError, match="at least 64"):
        SB._n2v_train([np.ones((32, 128))], tmp_path / "m.ckpt",
                      worker_for=lambda name, env: installed)
    with pytest.raises(ValueError, match="at least one"):
        SB._n2v_train([], tmp_path / "m.ckpt",
                      worker_for=lambda name, env: installed)


def test_denoising_returns_the_workers_plane(installed, tmp_path):
    out = SB._n2v_denoise(np.full((8, 8), 4, np.uint16), tmp_path / "m.ckpt",
                          worker_for=lambda name, env: installed)
    assert out.dtype == np.float32 and np.all(out == 2.0)


def test_without_the_backend_installed_it_says_how_to_install(monkeypatch):
    state = SB._BackendState(name=SB._CAREAMICS, state=SB._INSTALLABLE,
                             reason="not installed")
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: state)
    with pytest.raises(ImportError):
        SB._n2v_denoise(np.ones((8, 8)), "m.ckpt")


def test_the_backend_denoises_and_is_offered_as_alpha():
    spec = SB._SPECS[SB._CAREAMICS]
    assert spec.alpha and not spec.segments
    assert SB._CAREAMICS not in SB._BACKEND_NAMES
    assert SB._n2v_accelerator("cuda:1") == "gpu"
    assert SB._n2v_accelerator("cpu") == "cpu"
