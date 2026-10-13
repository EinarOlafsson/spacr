"""Item 555: click- and box-prompted segmentation with micro-SAM in Make Masks.

micro-SAM runs in an environment of its own, out of process, like every
other backend; nothing here starts it. The worker's half is driven through
``_handle`` with a stand-in ``micro_sam`` package, the client's half with a
stand-in worker, and the screen's half with a stand-in client. The category
is an ALPHA feature, registered as ``MakeMasksPromptCategory`` (and the Model
Zoo row as ``microsam_v1``) in ``spacr.settings.ALPHA_FEATURES``.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

import spacr._segmentation_backends as SB  # noqa: E402

pytestmark = pytest.mark.qt

IMG_N = 64


# ===========================================================================
# The environment
# ===========================================================================

def test_micro_sam_installs_into_its_own_environment_without_its_viewer():
    """micro-SAM declares napari, PyQt6 and trackastra (whose solver is the
    proprietary gurobipy); the prompt path imports none of them, so micro-SAM
    goes in with --no-deps after the dependencies that path does import."""
    spec = SB._SPECS["microsam"]
    assert not spec.segments
    assert "microsam" not in SB._BACKEND_NAMES
    steps = SB._install_plan(spec, "/b/microsam", ("/p",))
    labels = [step.label for step in steps]
    assert labels == ["Create the environment", "Update pip",
                      "Install PyTorch", "Install micro-SAM",
                      "Install micro-SAM", "Check it loads"]
    nodeps = steps[4].argv
    assert "--no-deps" in nodeps and nodeps[-1] == "micro-sam==1.8.14"
    assert "--no-deps" not in steps[3].argv
    everything = " ".join(" ".join(step.argv) for step in steps).lower()
    for heavy in ("napari", "pyqt6", "trackastra", "gurobipy",
                  "bioimageio.core"):
        assert heavy not in everything
    assert SB._stale_requirements(
        "microsam", {"requirements": list(spec.requirements)}) == [
            "micro-sam==1.8.14"]
    environ = SB._worker_env("microsam", "/b/microsam")
    assert environ["MICROSAM_CACHEDIR"] == os.path.join("/b/microsam",
                                                        "micro_sam")
    assert "CC-BY-4.0" in spec.licence and "gurobipy" in spec.licence_note


# ===========================================================================
# The worker, with a stand-in micro_sam
# ===========================================================================

class _FakeMicroSam:
    """What the worker imports from micro_sam, recording each call."""

    def __init__(self):
        self.embedded = []
        self.calls = []
        self.loaded = []

    def install(self, monkeypatch):
        package = types.ModuleType("micro_sam")
        util = types.ModuleType("micro_sam.util")
        prompts = types.ModuleType("micro_sam.prompt_based_segmentation")
        util.get_sam_model = self.get_sam_model
        util.precompute_image_embeddings = self.precompute
        prompts.segment_from_points = self.from_points
        prompts.segment_from_box = self.from_box
        prompts.segment_from_box_and_points = self.from_both
        package.util = util
        package.prompt_based_segmentation = prompts
        for name, module in (("micro_sam", package), ("micro_sam.util", util),
                             ("micro_sam.prompt_based_segmentation", prompts)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(SB, "_worker_device", lambda requested=None: "cpu")

    def get_sam_model(self, model_type, device):
        self.loaded.append((model_type, device))
        return ("predictor", model_type)

    def precompute(self, predictor, image, ndim, tile_shape, halo, verbose):
        self.embedded.append((image.shape, tile_shape, halo))
        return {"shape": image.shape[:2]}

    def _mask(self, embeddings, y, x):
        mask = np.zeros((1,) + tuple(embeddings["shape"]), bool)
        mask[0, max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
        return mask, np.array([0.9]), None

    def from_points(self, predictor, points, labels, image_embeddings,
                    return_all):
        self.calls.append(("points", points.tolist(), labels.tolist()))
        return self._mask(image_embeddings, *map(int, points[0]))

    def from_box(self, predictor, box, image_embeddings, return_all):
        self.calls.append(("box", box.tolist()))
        return self._mask(image_embeddings, int(box[0]) + 3, int(box[1]) + 3)

    def from_both(self, predictor, box, points, labels, image_embeddings,
                  return_all):
        self.calls.append(("both", box.tolist(), points.tolist()))
        return self._mask(image_embeddings, *map(int, points[0]))


def _ask(op, adapters, **payload):
    return SB._handle("microsam", dict(payload, protocol=SB._PROTOCOL, id=1,
                                       op=op), adapters)


def test_the_worker_embeds_a_field_once_and_answers_every_prompt_from_it(
        monkeypatch, tmp_path):
    fake = _FakeMicroSam()
    fake.install(monkeypatch)
    adapters = {}
    image = tmp_path / "image.npy"
    np.save(image, np.zeros((40, 50), np.uint8))
    missing = _ask("sam_prompt", adapters, key="f", points=[[5, 5]],
                   labels=[1], box=None, output=str(tmp_path / "m.npy"))
    assert not missing["ok"] and missing["error"]["type"] == "LookupError"
    done = _ask("sam_embed", adapters, image=str(image), key="f",
                model="vit_b_lm", device="cpu")
    assert done["ok"] and done["tiled"] is False and done["shape"] == [40, 50]
    for n, (points, labels, box) in enumerate((
            ([[10, 12]], [1], None),
            ([[10, 12], [20, 30]], [1, 0], None),
            ([], [], [5, 6, 25, 30]),
            ([[10, 12]], [1], [5, 6, 25, 30]))):
        out = tmp_path / f"m{n}.npy"
        reply = _ask("sam_prompt", adapters, key="f", points=points,
                     labels=labels, box=box, output=str(out))
        assert reply["ok"], reply
        mask = np.load(out)
        assert mask.shape == (40, 50) and mask.dtype == bool and mask.any()
        assert reply["pixels"] == int(mask.sum())
    assert len(fake.embedded) == 1 and fake.loaded == [("vit_b_lm", "cpu")]
    assert [call[0] for call in fake.calls] == ["points", "points", "box",
                                                "both"]
    assert fake.calls[1][2] == [1, 0]


def test_a_large_field_is_tiled_and_only_the_newest_fields_are_kept(
        monkeypatch, tmp_path):
    fake = _FakeMicroSam()
    fake.install(monkeypatch)
    adapters = {}
    big = tmp_path / "big.npy"
    np.save(big, np.zeros((SB._MICROSAM_TILE_ABOVE + 1, 64), np.uint8))
    assert _ask("sam_embed", adapters, image=str(big), key="big")["tiled"]
    assert fake.embedded[0][1] == (SB._MICROSAM_TILE, SB._MICROSAM_TILE)
    small = tmp_path / "small.npy"
    np.save(small, np.zeros((32, 32), np.uint8))
    for n in range(SB._MICROSAM_KEEP):
        assert _ask("sam_embed", adapters, image=str(small), key=f"k{n}")["ok"]
    assert "big" not in adapters["sam_embeddings"]
    assert len(adapters["sam_embeddings"]) == SB._MICROSAM_KEEP


# ===========================================================================
# The client, with a stand-in worker
# ===========================================================================

class _FakeWorker:
    """A worker that forgets every field until it is embedded."""

    def __init__(self):
        self.ops = []
        self.embedded = set()

    def request(self, op, *, should_cancel=None, keep_on_cancel=False,
                **payload):
        self.ops.append(op)
        assert keep_on_cancel, "a prompt must not kill the loaded model"
        if op == "sam_embed":
            assert np.load(payload["image"]).shape == (8, 8)
            self.embedded.add(payload["key"])
            return {"seconds": 1.5, "tiled": False, "device": "cpu"}
        if payload["key"] not in self.embedded:
            raise SB._BackendError("micro-SAM raised LookupError: none",
                                   "LookupError")
        mask = np.zeros((8, 8), bool)
        mask[2:4, 2:4] = True
        np.save(payload["output"], mask)
        return {"seconds": 0.01, "score": 0.8, "pixels": 4}


def _installed(monkeypatch, tmp_path):
    state = SB._BackendState("microsam", SB._INSTALLED, "here",
                             str(tmp_path), {"packages": {"micro_sam": "1.8.14"}})
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: state)


def test_the_client_sends_a_field_only_when_the_worker_has_not_embedded_it(
        monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path)
    worker = _FakeWorker()
    client = SB._PromptClient(worker_for=lambda name, env: worker)
    asked = []

    def image():
        asked.append(1)
        return np.zeros((8, 8), np.uint8)

    started = []
    monkeypatch.setattr(SB, "_WORKERS", {})
    first = client.segment("k", image, points=[(2, 2)], labels=[1],
                           on_start=lambda: started.append(1))
    assert started == [1], "a worker about to start must be announced"
    assert worker.ops == ["sam_prompt", "sam_embed", "sam_prompt"]
    assert first["embed_seconds"] == 1.5 and first["mask"].sum() == 4
    assert first["versions"] == {"micro_sam": "1.8.14"}
    again = client.segment("k", image, box=(0, 0, 5, 5))
    assert worker.ops[3:] == ["sam_prompt"] and again["embed_seconds"] is None
    assert asked == [1], "a field already embedded must not be sent again"


def test_the_client_says_where_to_install_micro_sam(monkeypatch, tmp_path):
    state = SB._BackendState("microsam", SB._INSTALLABLE, "not yet",
                             str(tmp_path))
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None: state)
    client = SB._PromptClient(worker_for=lambda *a: pytest.fail("started"))
    ready, reason = client.readiness()
    assert not ready and "Model Zoo" in reason
    with pytest.raises(ImportError, match="micro-SAM is not installed"):
        client.segment("k", np.zeros((4, 4)), points=[(1, 1)], labels=[1])


# ===========================================================================
# The screen, with a stand-in client
# ===========================================================================

class _FakeClient:
    """Answers each prompt with a disk round its first point, or its box."""

    def __init__(self):
        self.prompts = []
        self.embeds = 0

    def readiness(self):
        return True, "stand-in"

    def segment(self, key, image, points=(), labels=(), box=None, *,
                should_cancel=None, on_start=None, on_embed=None):
        self.prompts.append((key, list(points), list(labels), box))
        if len(self.prompts) == 1 and on_start is not None:
            on_start()
        field = image() if callable(image) else image
        if len(self.prompts) == 1 and on_embed is not None:
            self.embeds += 1
            on_embed()
        mask = np.zeros(field.shape[:2], bool)
        if box is not None:
            y0, x0, y1, x1 = box
            mask[y0:y1, x0:x1] = True
        else:
            y, x = points[0]
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            mask[(yy - y) ** 2 + (xx - x) ** 2 <= 16] = True
        return {"mask": mask, "seconds": 0.01, "score": 0.9,
                "embed_seconds": 2.0 if len(self.prompts) == 1 else None,
                "tiled": False, "device": "cpu", "model": "vit_b_lm",
                "versions": {"micro_sam": "1.8.14"}}


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    folder = tmp_path / "field"
    folder.mkdir()
    image = np.zeros((IMG_N, IMG_N), np.uint16)
    image[20:40, 20:40] = 30000
    imageio.imwrite(folder / "img_00.tif", image)
    imageio.imwrite(folder / "img_01.tif", image)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    widget._canvas.resize(600, 400)
    widget._canvas.refresh()
    widget._prompter._client = _FakeClient()
    return widget


def _at(screen, x, y):
    return QPointF(screen._canvas._image_to_canvas(x + 0.5, y + 0.5))


def _click(screen, x, y, button=Qt.LeftButton):
    prompter = screen._prompter
    assert prompter.press(button, _at(screen, x, y))
    assert prompter.release(button, _at(screen, x, y))


def test_the_category_is_hidden_unless_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr import model_zoo
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen
    from spacr.qt.screens.model_zoo import _model_is_alpha_hidden

    row = {e.key: e for e in model_zoo.installable_backend_entries()}[
        "microsam_v1"]
    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        widget = MakeMasksScreen()
        qtbot.addWidget(widget)
        card = widget._prompt_card
        assert card.objectName() == "MakeMasksPromptCategory"
        assert card.isAncestorOf(widget._prompt_section)
        assert widget._prompt_section.objectName() == "SectionCard"
        assert card.isAncestorOf(widget._btn_prompt)
        assert card.isHidden() is not shown
        assert _model_is_alpha_hidden(row) is not shown
        widget._prompter._enabled = True
        assert widget._prompter.enabled is shown, (
            "prompting must not read clicks while its category is hidden")
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(widget)
        assert card.isHidden() is shown
        assert _model_is_alpha_hidden(row) is shown


def test_turning_prompting_on_offers_the_install_when_micro_sam_is_missing(
        screen, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    asked = []
    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: False)
    monkeypatch.setattr(model_zoo_picker, "install_backend",
                        lambda parent, name, **kw: asked.append(name) or False)
    screen._btn_prompt.setChecked(True)
    assert asked == ["microsam"]
    assert not screen._btn_prompt.isChecked()
    assert not screen._prompter.enabled


def test_a_click_gives_an_object_that_is_one_undoable_edit_save_writes(
        screen, qtbot, monkeypatch):
    from spacr.qt import mask_engine as engine

    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: True)
    screen._btn_magnifier.setChecked(True)
    screen._btn_prompt.setChecked(True)
    assert screen._prompter.enabled and not screen._btn_magnifier.isChecked()
    before = np.array(screen._canvas.mask, copy=True)
    _click(screen, 30, 30)
    qtbot.waitUntil(lambda: screen._prompter.pending is not None, timeout=5000)
    assert screen._btn_prompt_accept.isEnabled()
    assert "embedded this field in 2.0 s" in screen._masks_console.text()
    assert np.array_equal(screen._canvas.mask, before), (
        "nothing reaches the mask before it is accepted")

    added = screen._accept_prompt()
    assert len(added) == 1
    new = screen._canvas.mask == added[0]
    assert new[30, 30] and int(new.sum()) > 20
    entry = screen._log.edits[-1]
    assert entry.kind == "prompt" and entry.target == added
    assert entry.detail["tool"] == "micro-SAM"
    assert entry.detail["points"] == [[30, 30, True]]
    assert entry.detail["versions"] == {"micro_sam": "1.8.14"}
    assert screen._prompter.pending is None and not screen._prompter.points

    screen._on_save()
    path = engine.mask_save_path(screen._folder, "img_00.tif",
                                 **screen._layout_kwargs())
    saved = imageio.imread(path)
    assert saved[30, 30] > 0 and int((saved > 0).sum()) == int(new.sum())

    screen._on_undo()
    assert np.array_equal(screen._canvas.mask, before)


def test_points_off_the_object_and_a_box_refine_it_and_escape_discards(
        screen, qtbot, monkeypatch):
    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: True)
    screen._btn_prompt.setChecked(True)
    prompter = screen._prompter
    client = prompter._client
    _click(screen, 30, 30)
    _click(screen, 45, 45, Qt.RightButton)
    qtbot.waitUntil(lambda: prompter.pending is not None
                    and len(prompter.pending["points"]) == 2, timeout=5000)
    assert client.prompts[-1][2] == [1, 0]
    assert len({prompt[0] for prompt in client.prompts}) == 1, (
        "one field must be embedded under one key")

    assert prompter.press(Qt.LeftButton, _at(screen, 20, 22))
    assert prompter.move(_at(screen, 39, 41))
    assert prompter.release(Qt.LeftButton, _at(screen, 39, 41))
    assert prompter.box == (22, 20, 42, 40)
    qtbot.waitUntil(lambda: prompter.pending is not None
                    and prompter.pending["box"] is not None, timeout=5000)

    assert prompter.wants_key(Qt.Key_Backspace)
    assert prompter.key(Qt.Key_Backspace) and len(prompter.points) == 1
    assert prompter.key(Qt.Key_Escape)
    assert prompter.pending is None and not prompter.points
    assert prompter.box is None and not prompter.wants_key(Qt.Key_Escape)


def test_a_point_off_the_object_alone_asks_for_one_on_it(screen,
                                                        monkeypatch):
    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: True)
    screen._btn_prompt.setChecked(True)
    _click(screen, 5, 5, Qt.RightButton)
    assert screen._prompter._client.prompts == []
    assert "before points off it" in screen._masks_console.text()


def test_moving_to_another_field_discards_the_prompt(screen, qtbot,
                                                     monkeypatch):
    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: True)
    screen._btn_prompt.setChecked(True)
    _click(screen, 30, 30)
    qtbot.waitUntil(lambda: screen._prompter.pending is not None,
                    timeout=5000)
    generation = screen._prompter.generation
    screen._on_next()
    assert screen._prompter.pending is None and not screen._prompter.points
    assert screen._prompter.generation > generation
    assert screen._prompter.enabled
