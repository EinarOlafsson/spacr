"""Item 556: SAM2 video propagation as a timelapse mode.

SAM2 runs in an environment of its own, out of process, like every other
backend; nothing here starts it. The worker's half is driven through
``_handle`` with a stand-in ``sam2`` package, the client's half with a
stand-in worker, and the tracker with a stand-in propagation. The
``timelapse_mode`` choice ``sam2`` and the Model Zoo row ``sam2_v1`` are
ALPHA features, registered in ``spacr.settings.ALPHA_FEATURES``.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

import spacr._segmentation_backends as SB  # noqa: E402
from tests.test_cov_object_cellpose_masks import (  # noqa: E402,F401
    _settings, _write_npz, fake_sam_model, fake_timelapse)

pytestmark = pytest.mark.qt


def test_sam2_is_built_in_its_own_environment_without_its_cuda_extension(
        tmp_path):
    spec = SB._SPECS["sam2"]
    assert spec.alpha and not spec.segments and not spec.prefix
    assert spec.models[0] == "sam2.1_hiera_tiny" == spec.default_model
    steps = SB._install_plan(spec, str(tmp_path / "sam2"), ["python"])
    build = [s.argv for s in steps if "sam2==1.1.0" in s.argv]
    assert len(build) == 1 and "--no-build-isolation" in build[0]
    labels = [s.label for s in steps]
    assert labels.index("Install PyTorch") < labels.index(
        "Install SAM2") < labels.index("Build SAM2's extensions")
    env = str(tmp_path / "sam2")
    environ = SB._worker_env("sam2", env)
    assert environ["HF_HOME"] == os.path.join(env, "huggingface")
    assert environ["SAM2_BUILD_CUDA"] == "0"
    assert SB._sam2_hub_id("sam2.1_hiera_base_plus") == (
        "facebook/sam2.1-hiera-base-plus")
    with pytest.raises(ValueError):
        SB._sam2_hub_id("sam2_huge")


class _FakeSam2:
    """A stand-in ``sam2`` whose video predictor moves every seed one pixel
    right per frame forward, and one pixel left per frame backward."""

    def __init__(self):
        self.loaded = []
        self.added = []
        self.passes = []

    def install(self, monkeypatch):
        import torch

        fake = self

        class Predictor:
            memory_attention = types.SimpleNamespace(forward=None)

            @classmethod
            def from_pretrained(cls, hub, device=None):
                fake.loaded.append((hub, device))
                return cls()

            def init_state(self, video_path, offload_video_to_cpu=False):
                names = sorted(os.listdir(video_path))
                assert names[0] == "00000.jpg"
                return {"frames": len(names), "seeds": {}}

            def add_new_mask(self, state, frame_idx, obj_id, mask):
                fake.added.append((frame_idx, obj_id, int(mask.sum())))
                state["seeds"].setdefault(obj_id, (frame_idx, mask))

            def propagate_in_video(self, state, reverse=False):
                fake.passes.append(reverse)
                order = range(state["frames"])
                for frame in (reversed(order) if reverse else order):
                    ids, logits = [], []
                    for obj, (start, mask) in sorted(state["seeds"].items()):
                        shifted = np.roll(mask, frame - start, axis=1)
                        ids.append(obj)
                        logits.append(np.where(shifted, 5.0, -5.0))
                    yield frame, ids, torch.tensor(
                        np.stack(logits)[:, None], dtype=torch.float32)

            def reset_state(self, state):
                state.clear()

        package = types.ModuleType("sam2")
        module = types.ModuleType("sam2.sam2_video_predictor")
        module.SAM2VideoPredictor = Predictor
        package.sam2_video_predictor = module
        monkeypatch.setitem(sys.modules, "sam2", package)
        monkeypatch.setitem(sys.modules, "sam2.sam2_video_predictor", module)


def _ask(adapters, **request):
    return SB._handle("sam2", {"protocol": SB._PROTOCOL, "id": 1,
                               "op": "sam2_propagate", **request}, adapters)


def test_the_worker_follows_each_seed_from_its_own_frame(monkeypatch,
                                                         tmp_path):
    fake = _FakeSam2()
    fake.install(monkeypatch)
    frames = tmp_path / "frames.npy"
    np.save(frames, np.zeros((4, 12, 20), np.uint8))
    seeds = np.zeros((2, 12, 20), np.int32)
    seeds[0, 2:5, 2:5] = 3
    seeds[1, 7:10, 10:13] = 9
    seeded = tmp_path / "seeds.npy"
    np.save(seeded, seeds)
    out = tmp_path / "labels.npy"
    reply = _ask({}, frames=str(frames), seeds=str(seeded),
                 seed_frames=[0, 2], output=str(out), device="cpu",
                 model="sam2.1_hiera_tiny")
    assert reply["ok"], reply
    assert reply["objects"] == 2 and reply["frames"] == 4
    labels = np.load(out)
    assert labels.shape == (4, 12, 20) and labels.dtype == np.int32
    assert [sorted(set(np.unique(f)) - {0}) for f in labels] == [
        [3], [3], [3, 9], [3, 9]]
    assert (labels[3] == 3).sum() == 9 and labels[3, 3, 5] == 3
    assert fake.loaded == [("facebook/sam2.1-hiera-tiny", "cpu")]
    assert fake.passes == [False]
    assert sorted(fake.added) == [(0, 3, 9), (2, 9, 9)]

    back = tmp_path / "back.npy"
    reply = _ask({}, frames=str(frames), seeds=str(seeded),
                 seed_frames=[0, 2], output=str(back), device="cpu",
                 backward=True)
    assert reply["ok"], reply
    assert fake.passes[-2:] == [False, True]
    assert 9 in np.load(back)[0], "backward reaches frames before the seed"


def test_the_worker_refuses_a_seed_outside_the_movie(monkeypatch, tmp_path):
    _FakeSam2().install(monkeypatch)
    frames = tmp_path / "frames.npy"
    np.save(frames, np.zeros((2, 8, 8), np.uint8))
    seeded = tmp_path / "seeds.npy"
    np.save(seeded, np.ones((1, 8, 8), np.int32))
    reply = _ask({}, frames=str(frames), seeds=str(seeded), seed_frames=[5],
                 output=str(tmp_path / "o.npy"), device="cpu")
    assert not reply["ok"] and reply["error"]["type"] == "ValueError"


class _Worker:
    def __init__(self):
        self.requests = []

    def request(self, op, **kw):
        self.requests.append((op, kw))
        frames = np.load(kw["frames"])
        seeds = np.load(kw["seeds"])
        labels = np.zeros(frames.shape[:3], np.int32)
        labels[:] = seeds[0]
        np.save(kw["output"], labels)
        return {"seconds": 1.5, "objects": int(len(np.unique(seeds)) - 1),
                "device": kw["device"], "model": kw["model"]}


def test_the_client_sends_the_movie_and_seeds_and_reads_the_labels(
        monkeypatch, tmp_path):
    ready = SB._BackendState(name="sam2", state=SB._INSTALLED, reason="",
                             env=str(tmp_path / "env"))
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: ready)
    worker = _Worker()
    frames = np.zeros((3, 6, 7), np.uint8)
    seed = np.zeros((6, 7), np.int32)
    seed[1:3, 1:3] = 4
    labels, reply = SB._sam2_propagate(
        frames, {0: seed}, device="cpu",
        worker_for=lambda name, env: worker)
    assert labels.shape == (3, 6, 7) and (labels[2] == 4).sum() == 4
    op, kw = worker.requests[0]
    assert op == "sam2_propagate" and kw["seed_frames"] == [0]
    assert kw["model"] == "sam2.1_hiera_tiny" and kw["backward"] is False
    assert reply["objects"] == 1


def test_the_client_says_where_to_install_sam2(monkeypatch, tmp_path):
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    with pytest.raises(ImportError, match="SAM2"):
        SB._sam2_propagate(np.zeros((2, 4, 4), np.uint8),
                           {0: np.ones((4, 4), np.int32)})


def test_the_movie_is_stretched_once_for_every_frame():
    movie = np.stack([np.full((4, 4), v, np.uint16) for v in (100, 200, 300)])
    movie[0, 0, 0] = 0
    frames = SB._sam2_frames(movie[..., None])
    assert frames.dtype == np.uint8 and frames.shape == (3, 4, 4)
    assert frames[0, 1, 1] < frames[1, 1, 1] < frames[2, 1, 1]


def test_the_tracker_seeds_the_first_frame_and_each_new_object_once(
        tmp_path):
    from spacr.timelapse import _sam2_track_cells

    masks = np.zeros((4, 16, 16), np.int32)
    masks[:, 2:6, 2:6] = 7
    masks[2:, 10:14, 10:14] = 1
    images = np.random.default_rng(0).random((4, 16, 16, 2))
    calls = []

    def propagate(frames, seeds, model=None, device=None):
        calls.append(sorted(seeds))
        out = np.zeros(masks.shape, np.int32)
        for frame, seed in seeds.items():
            for t in range(frame, 4):
                out[t][seed > 0] = seed[seed > 0]
        return out, {"objects": len(calls), "seconds": 0.1, "device": "cpu"}

    src = tmp_path / "run" / "stack"
    src.mkdir(parents=True)
    stack = _sam2_track_cells(str(src), "b1", [f"f{i}" for i in range(4)],
                              "cell", masks, images=images,
                              propagate=propagate)
    assert calls == [[0], [0, 2]]
    stack = np.asarray(stack)
    assert set(np.unique(stack[3])) == {0, 7, 8}
    assert set(np.unique(stack[1])) == {0, 7}
    table = tmp_path / "run" / "tracks" / "sam2_tracks_cell_b1.csv"
    assert table.is_file()


def test_the_tracker_needs_the_images(tmp_path):
    from spacr.timelapse import _sam2_track_cells

    with pytest.raises(ValueError, match="image stack"):
        _sam2_track_cells(str(tmp_path), "b", [], "cell",
                          np.zeros((2, 4, 4), np.int32))


def test_the_sam_generator_routes_to_sam2(tmp_path, monkeypatch,
                                          fake_sam_model, fake_timelapse):
    import spacr.object as O
    import spacr.timelapse as TL

    calls = []

    def track(**kw):
        calls.append(kw)
        return [np.asarray(m, dtype=np.uint16) for m in kw["masks"]]

    monkeypatch.setattr(TL, "_sam2_track_cells", track)
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    settings = _settings(
        src, timelapse=True, timelapse_objects=["cell"],
        timelapse_mode="sam2", timelapse_displacement=None,
        timelapse_memory=3, timelapse_remove_transient=True,
        timelapse_frame_limits=[0, 3],
        cell_min_split_area=0, nucleus_min_split_area=0)
    O.generate_cellpose_masks_sam(str(src), settings, "cell")
    assert len(calls) == 1
    kw = calls[0]
    assert kw["mode"] == "sam2" and kw["object_type"] == "cell"
    assert kw["timelapse_remove_transient"] is True
    assert kw["batch_filenames"] == names and len(kw["masks"]) == 3
    assert np.asarray(kw["images"]).shape == (3, 32, 32, 2)
    assert fake_timelapse["ultrack"] == [] and fake_timelapse["btrack"] == []


def test_the_sam2_choice_and_zoo_row_follow_the_alpha_switch(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from spacr import model_zoo
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.model_zoo import _model_is_alpha_hidden

    choices = ["trackastra", "ultrack", "trackpy", "iou", "btrack",
               "timeflows", "sam2"]
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(choices)
    row = {e.key: e for e in model_zoo.installable_backend_entries()}[
        "sam2_v1"]
    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        AppScreen._gate_alpha_choices("timelapse_mode", combo)
        assert combo.view().isRowHidden(6) is not shown
        assert not combo.view().isRowHidden(0)
        assert _model_is_alpha_hidden(row) is not shown
        combo.setCurrentText("sam2")
        assert combo.currentText() == "sam2", "a saved value still runs"


def test_on_the_cpu_the_bfloat16_memory_enters_attention_as_float32():
    import torch

    seen = []

    class Attention:
        def forward(self, curr=None, memory=None, memory_pos=None, **kw):
            seen.append((memory.dtype, memory_pos.dtype))
            return curr

    predictor = types.SimpleNamespace(memory_attention=Attention())
    SB._sam2_float_memory(predictor)
    out = predictor.memory_attention.forward(
        curr="features", memory=torch.zeros(2, dtype=torch.bfloat16),
        memory_pos=torch.zeros(2))
    assert out == "features"
    assert seen == [(torch.float32, torch.float32)]


def test_click_seeds_propagate_then_a_corrected_frame_repropagates(qtbot):
    from spacr.qt.screens.make_masks import _Sam2SeedDialog

    calls = []

    def fake_propagate(frames, seeds, backward=False):
        calls.append(({t: s.copy() for t, s in seeds.items()}, backward))
        out = np.zeros(frames.shape, np.int32)
        for s in seeds.values():
            out[:] = np.maximum(out, s)
        return out, {"objects": 1}

    frames = np.zeros((3, 20, 20), np.uint8)
    dialog = _Sam2SeedDialog(frames, propagate=fake_propagate, radius=1)
    qtbot.addWidget(dialog)
    dialog.run()
    assert not calls
    dialog.view.clicked.emit(5, 5, False)
    assert dialog.seeds[0][5, 5] == 1
    dialog._new_object()
    assert dialog.object_box.value() == 2
    dialog.view.clicked.emit(15, 15, False)
    dialog.run()
    qtbot.waitUntil(lambda: dialog.labels is not None, timeout=5000)
    assert dialog.labels[2, 15, 15] == 2 and dialog.run_button.isEnabled()
    dialog.slider.setValue(2)
    dialog.object_box.setValue(1)
    dialog.view.clicked.emit(10, 10, False)
    dialog.backward_box.setChecked(True)
    dialog.labels = None
    dialog.run()
    qtbot.waitUntil(lambda: dialog.labels is not None, timeout=5000)
    seeds, backward = calls[-1]
    assert sorted(seeds) == [0, 2] and backward and seeds[2][10, 10] == 1
    dialog.view.clicked.emit(10, 10, True)
    assert 2 not in dialog.seeds


def test_a_failed_propagation_is_reported(qtbot):
    from spacr.qt.screens.make_masks import _Sam2SeedDialog

    def broken(frames, seeds, backward=False):
        raise ImportError("not installed")

    dialog = _Sam2SeedDialog(np.zeros((2, 8, 8), np.uint8), propagate=broken)
    qtbot.addWidget(dialog)
    dialog.view.clicked.emit(3, 3, False)
    dialog.run()
    qtbot.waitUntil(lambda: dialog.run_button.isEnabled(), timeout=5000)
    assert "not installed" in dialog.status.text() and dialog.labels is None


def test_the_sam2_button_is_alpha(qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton, QWidget

    import spacr.qt.preferences as P
    from spacr.qt.screens.make_masks import MakeMasksScreen

    root = QWidget()
    qtbot.addWidget(root)
    button = QPushButton(root)
    button.setObjectName("MakeMasksSam2Button")
    monkeypatch.setattr(P, "_get_show_alpha_features", lambda: False)
    P._apply_alpha_widgets(root)
    assert button.isHidden()
    monkeypatch.setattr(P, "_get_show_alpha_features", lambda: True)
    P._apply_alpha_widgets(root)
    assert not button.isHidden()
    assert hasattr(MakeMasksScreen, "_build_sam2_button")
