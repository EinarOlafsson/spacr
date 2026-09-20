"""Item 446: the two Cellposes do not take the same settings.

Cellpose-SAM is segmented in process by the mask generator itself and is not
touched here -- that separation is one of the tests. Cellpose 3 runs in an
environment of its own, and what it is asked for has to be what it can
answer.
"""

from __future__ import annotations

import inspect
import sys
import types

import numpy as np
import pytest

import spacr._segmentation_backends as SB


class _FakeCellposeModel:
    """A Cellpose 3 model that records the call and returns one object."""

    def __init__(self, **built):
        self.built = built
        self.calls = []

    def eval(self, image, channels=None, channel_axis=None, diameter=None,
             normalize=True, flow_threshold=0.4, cellprob_threshold=0.0,
             min_size=15, resample=True, batch_size=8):
        self.calls.append(dict(
            channels=channels, channel_axis=channel_axis, diameter=diameter,
            normalize=normalize, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size,
            resample=resample, batch_size=batch_size))
        labels = np.zeros(np.asarray(image).shape[:2], dtype=np.int32)
        labels[0, 0] = 1
        flows = [np.zeros(labels.shape), np.zeros(labels.shape),
                 np.zeros(labels.shape)]
        return labels, flows, None


class _AnythingGoesModel(_FakeCellposeModel):
    """A Cellpose whose ``eval`` takes keywords it was not told about."""

    def eval(self, image, **kwargs):
        self.calls.append(dict(kwargs))
        labels = np.zeros(np.asarray(image).shape[:2], dtype=np.int32)
        return labels, [np.zeros(labels.shape)], None


@pytest.fixture
def cellpose3(monkeypatch):
    """Stand in for the ``cellpose`` package inside a backend environment."""
    made = {}

    def _factory(cls):
        def _build(**kwargs):
            model = cls(**kwargs)
            made["model"] = model
            return model
        return _build

    def _install(cls=_FakeCellposeModel):
        for name in [m for m in sys.modules
                     if m == "cellpose" or m.startswith("cellpose.")]:
            monkeypatch.delitem(sys.modules, name)
        package = types.ModuleType("cellpose")
        package.__path__ = []
        models = types.ModuleType("cellpose.models")
        models.Cellpose = _factory(cls)
        models.CellposeModel = _factory(cls)
        package.models = models
        monkeypatch.setitem(sys.modules, "cellpose", package)
        monkeypatch.setitem(sys.modules, "cellpose.models", models)
        return made

    return _install


def _image():
    return np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)


def test_a_cellpose_3_run_normalizes_the_way_its_weights_were_trained(cellpose3):
    """The defect item 446 was filed for.

    Mask generation scales by the image maximum and then says
    ``normalize=False``, which is right for Cellpose-SAM and wrong for
    weights fitted on Cellpose's percentile normalization.
    """
    made = cellpose3()
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], normalize=False)
    assert made["model"].calls[0]["normalize"] is True
    assert adapter.translated, "the translation has to be reportable"


def test_a_caller_that_wants_normalization_still_gets_it(cellpose3):
    made = cellpose3()
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], normalize=True)
    assert made["model"].calls[0]["normalize"] is True
    assert not adapter.translated


def test_the_batch_size_the_user_set_reaches_cellpose_3(cellpose3):
    made = cellpose3()
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], batch_size=3)
    assert made["model"].calls[0]["batch_size"] == 3
    assert not adapter.ignored


def test_a_setting_this_cellpose_cannot_take_is_named_rather_than_dropped(cellpose3):
    """``**unused`` swallowed these silently before item 446."""
    made = cellpose3()
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], rescale=2.0, niter=17)
    assert adapter.ignored == {"rescale", "niter"}
    assert "rescale" not in made["model"].calls[0]


def test_a_cellpose_that_takes_anything_is_given_everything(cellpose3):
    made = cellpose3(_AnythingGoesModel)
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], rescale=2.0)
    assert adapter.ignored == set()
    assert made["model"].calls[0]["rescale"] == 2.0


def test_a_setting_left_at_none_is_not_reported_as_ignored(cellpose3):
    """A keyword nobody set is not a setting that went missing."""
    cellpose3()
    adapter = SB._Cellpose3Adapter("cyto3", "cpu")
    adapter.eval([_image()], rescale=None)
    assert adapter.ignored == set()


def test_scaling_by_the_maximum_does_not_move_a_percentile():
    """Why turning normalization back on is free: dividing by the maximum is
    a linear scale with no offset, and percentiles scale with it, so
    Cellpose's normalization of x/max(x) is its normalization of x."""
    rng = np.random.default_rng(446)
    x = rng.gamma(2.0, 40.0, size=(128, 128))
    scaled = x / x.max()
    for q in (1.0, 50.0, 99.0):
        assert np.isclose(np.percentile(scaled, q),
                          np.percentile(x, q) / x.max())
    lo, hi = np.percentile(x, [1.0, 99.0])
    slo, shi = np.percentile(scaled, [1.0, 99.0])
    assert np.allclose((x - lo) / (hi - lo), (scaled - slo) / (shi - slo))


class _Worker:
    """A stand-in for the backend's worker process."""

    def __init__(self, reply=None):
        self.requests = []
        self.reply = reply or {}

    def request(self, op, **kwargs):
        self.requests.append((op, kwargs))
        outputs = []
        for index, path in enumerate(kwargs.get("inputs") or ()):
            mask = kwargs["outputs"] + f"/mask_{index}.npy"
            np.save(mask, np.zeros((8, 8), dtype=np.int32))
            outputs.append({"mask": mask, "flows": [None, None, None, None]})
        return {"outputs": outputs, **self.reply}


@pytest.fixture
def remote(monkeypatch, tmp_path):
    """A :class:`_RemoteBackend` whose environment is taken as installed."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))

    def _build(reply=None):
        worker = _Worker(reply)
        state = SB._BackendState(name="cellpose3", state=SB._INSTALLED,
                                 env=str(tmp_path / "env"), in_process=False)
        monkeypatch.setattr(SB, "_backend_state",
                            lambda name, root=None: state)
        backend = SB._RemoteBackend("cellpose3", model="cyto3",
                                    worker_for=lambda name, env: worker)
        return backend, worker

    return _build


def test_the_remote_backend_sends_the_batch_size_it_was_given(remote):
    """It was accepted in the signature and dropped before the worker."""
    backend, worker = remote()
    backend.eval([np.zeros((8, 8), dtype=np.float32)], batch_size=5)
    _op, sent = worker.requests[0]
    assert sent["params"]["batch_size"] == 5


def test_what_the_backend_could_not_honour_is_said_once(remote, capsys):
    backend, _worker = remote({"ignored": ["rescale"],
                               "translated": ["normalize=False became True"]})
    image = [np.zeros((8, 8), dtype=np.float32)]
    backend.eval(image)
    backend.eval(image)
    err = capsys.readouterr().err
    assert err.count("rescale") == 1
    assert err.count("normalize=False became True") == 1
    assert "did not reach the model" in err


def test_a_backend_that_honours_everything_says_nothing(remote, capsys):
    backend, _worker = remote()
    backend.eval([np.zeros((8, 8), dtype=np.float32)])
    assert "did not reach the model" not in capsys.readouterr().err


def test_the_cellpose_sam_path_is_not_this_modules_business():
    """Mask generation builds Cellpose-SAM itself. If that ever changes,
    item 446's promise that SAM is untouched changes with it."""
    with pytest.raises(ValueError, match="built by the mask generator"):
        SB._load_backend("cellpose")


def test_the_mask_generator_still_builds_cellpose_sam_in_process():
    source = inspect.getsource(SB).split("\n")
    del source
    import spacr.object as object_module

    text = inspect.getsource(object_module.generate_cellpose_masks_sam)
    assert "if segmentation_backend == 'cellpose':" in text
    assert "cp_models.CellposeModel(" in text
    assert "normalize=False" in text
