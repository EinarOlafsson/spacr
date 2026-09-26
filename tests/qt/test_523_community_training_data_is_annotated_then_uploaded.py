"""Item 523: community plaque training data, annotated and then uploaded.

    "there should be a button to upload the image or pdf to the spacr (my)
     huggingface ... the user should be prompted to draw boxes where the
     wells are ... and draw plaques as masks ... it should not be possible
     to uploade images without masks, and the user should be warned ..."
    -- the maintainer, 2026-09-25

The upload is faked: the fake receives the contribution folder the real
upload would send, so the test reads exactly what would have reached
Hugging Face. Seeding goes through the same fake detector and segmenter the
other plaque preview tests use.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import model_share  # noqa: E402
from spacr.qt.widgets import model_share_dialog as msd  # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402


class _Box:
    def __init__(self, x0, y0, x1, y1, confidence=0.9):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.confidence = confidence


def _detect(image, weights, confidence=0.25, imgsz=640, min_axis_ratio=0.0):
    return [_Box(10, 10, 40, 40), _Box(60, 10, 90, 40)]


def _square(path=None):
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[40:50, 40:50] = 2
    return labels


def _png(path, size=(60, 60)):
    from PIL import Image

    Image.fromarray(np.full(size + (3,), 90, dtype=np.uint8)).save(path)
    return path


class _Prefs:
    def __init__(self):
        self.store = {}

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


class _Upload:
    def __init__(self):
        self.calls = []

    def __call__(self, folder, kind):
        self.calls.append((folder, kind))
        return f"https://huggingface.co/datasets/{model_share.community_repo(kind)}/discussions/1"


@pytest.fixture
def prefs(monkeypatch):
    store = _Prefs()
    monkeypatch.setattr(msd, "_preferences", lambda: store)
    return store


@pytest.fixture
def panel(qtbot, monkeypatch, prefs):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _consent(dialog, answer=True):
    asked = []

    def ask():
        asked.append(1)
        return answer

    dialog.ask_consent = ask
    return asked


def test_the_button_is_there_in_both_modes(panel):
    for mode in (ppv.PLAQUE_MODE, ppv.FIGURE_MODE):
        panel.set_mode(mode)
        assert not panel._contribute_btn.isHidden()


def test_figure_mode_boxes_every_well_and_writes_yolo_labels(panel, qtbot, tmp_path):
    page = _png(tmp_path / "fig1.png", size=(50, 100))
    (tmp_path / "paper.json").write_text(json.dumps(
        {"key": "doi:10.1/x", "source": "doi", "doi": "10.1/x",
         "title": "Plaques"}))
    panel.set_mode(ppv.FIGURE_MODE)
    panel.load_source_async(str(tmp_path))
    upload = _Upload()
    dialog = panel.contribute_training_data(detect=_detect, upload=upload)
    qtbot.addWidget(dialog)
    asked = _consent(dialog)
    editor = dialog.editor(page)
    assert editor.boxes() == [(10, 10, 40, 40), (60, 10, 90, 40)]
    assert dialog.upload_button.isEnabled()
    assert dialog.paper_edit.text() == "10.1/x"

    editor.set_box(0, 8, 8, 42, 42)
    editor.delete_box(1)
    editor.add_box(55, 12, 95, 45)
    assert dialog.upload()
    assert asked == [1]
    folder, kind = upload.calls[0]
    assert kind == "figures"
    assert sorted(p.name for p in (folder / "images").iterdir()) == ["fig1.png"]
    lines = (folder / "labels" / "fig1.txt").read_text().split()
    assert len(lines) == 10 and lines[0] == "0"
    assert float(lines[1]) == pytest.approx(25 / 100)
    meta = json.loads((folder / "meta" / "fig1.json").read_text())
    assert meta["paper"]["doi"] == "10.1/x"
    assert meta["provenance"] == {"proposed": 2, "kept": [], "moved": [0],
                                  "deleted": [1], "added": 1}
    record = json.loads((folder / "contribution.json").read_text())
    assert record["licence"] == model_share.COMMUNITY_LICENCE
    assert record["repo"] == model_share.COMMUNITY_FIGURES_REPO
    assert record["boxes"] == 2
    assert "review" in dialog.status.text()


def test_plaque_mode_paints_every_plaque_and_writes_a_label_mask(panel, qtbot, tmp_path):
    import tifffile

    image = _png(tmp_path / "well.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(segment=_square)
    upload = _Upload()
    dialog = panel.contribute_training_data(upload=upload)
    qtbot.addWidget(dialog)
    _consent(dialog)
    page = dialog.editor(image)
    assert int(page.labels().max()) == 2
    page.brush.session.erase({"y": 45.0, "x": 45.0}, radius=10.0)
    page.brush.session.paint({"y": 50.0, "x": 10.0}, label=3, radius=4.0)
    page.brush.session.paint({"y": 12.0, "x": 12.0}, label=1, radius=6.0)
    assert dialog.upload()
    folder, kind = upload.calls[0]
    assert kind == "plaques"
    assert (folder / "images" / "well.png").read_bytes() == image.read_bytes()
    mask = tifffile.imread(str(folder / "masks" / "well.tif"))
    assert mask.shape == (60, 60) and mask.dtype == np.uint16
    assert set(np.unique(mask)) == {0, 1, 3}
    meta = json.loads((folder / "meta" / "well.json").read_text())
    assert meta["provenance"] == {"kept": [], "edited": [1], "removed": [2],
                                  "added": 1}
    assert meta["objects"] == 2


def test_an_image_without_annotations_is_refused(panel, qtbot, tmp_path):
    first = _png(tmp_path / "a.png")
    second = _png(tmp_path / "b.png")
    upload = _Upload()
    dialog = panel.contribute_training_data(
        paths=[first, second], segment=_square, upload=upload)
    qtbot.addWidget(dialog)
    asked = _consent(dialog)
    page = dialog.editor(second)
    page.brush.session.erase({"y": 30.0, "x": 30.0}, radius=60.0)
    assert dialog.count(second) == 0
    assert not dialog.upload_button.isEnabled()
    assert not dialog.upload()
    assert upload.calls == [] and asked == []
    assert "b.png" in dialog.status.text()
    with pytest.raises(ValueError):
        model_share.write_contribution(
            "plaques", [{"name": "b.png", "source": str(second),
                         "labels": np.zeros((60, 60), np.int32)}],
            tmp_path / "out", consent={})
    dialog.image_list.setCurrentRow(1)
    dialog.remove_current()
    assert dialog.upload_button.isEnabled()


def test_the_conscience_is_shown_beside_upload_in_both_modes(panel, qtbot, tmp_path):
    _png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    for mode in (ppv.PLAQUE_MODE, ppv.FIGURE_MODE):
        panel.set_mode(mode)
        dialog = panel.contribute_training_data(
            detect=_detect, segment=_square, upload=_Upload())
        qtbot.addWidget(dialog)
        assert dialog.conscience.isVisible()
        text = dialog.conscience.text()
        assert "next" in text and "extra minute" in text
        assert dialog.conscience.parentWidget() is dialog.upload_button.parentWidget()
        dialog.close()


def test_consent_is_asked_once_and_a_refusal_sends_nothing(panel, qtbot, tmp_path, prefs):
    _png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    upload = _Upload()
    dialog = panel.contribute_training_data(segment=_square, upload=upload)
    qtbot.addWidget(dialog)
    refused = _consent(dialog, answer=False)
    assert not dialog.upload()
    assert refused == [1] and upload.calls == []
    assert not msd.community_consented()
    asked = _consent(dialog, answer=True)
    assert dialog.upload()
    assert dialog.upload()
    assert asked == [1] and len(upload.calls) == 2
    assert msd.community_consented()


def test_the_real_upload_opens_a_pull_request_on_the_dataset(monkeypatch, tmp_path):
    sent = {}

    class _Api:
        def __init__(self, token=None):
            sent["token"] = token

        def repo_exists(self, repo_id, repo_type=None):
            sent["exists_checked"] = (repo_id, repo_type)
            return True

        def upload_folder(self, **kwargs):
            sent.update(kwargs)

            class Info:
                pr_url = "https://huggingface.co/datasets/x/discussions/3"
            return Info()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    folder = tmp_path / "20260925-abc"
    folder.mkdir()
    url = model_share.contribute(folder, "plaques", "tok")
    assert url.endswith("/discussions/3")
    assert sent["repo_id"] == model_share.COMMUNITY_PLAQUES_REPO
    assert sent["repo_type"] == "dataset" and sent["create_pr"] is True
    assert sent["path_in_repo"] == "contributions/20260925-abc"
    assert model_share.community_repo("figures") == \
        "einarolafsson/community_toxoplasma_plaque_figures"


def test_boxes_are_drawn_moved_and_deleted_with_the_mouse(qtbot):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    editor = ppv._BoxEditor(np.zeros((50, 100, 3), np.uint8), [(10, 10, 20, 20)])
    qtbot.addWidget(editor)
    editor.setMinimumSize(1, 1)
    editor.resize(400, 200)
    editor.show()
    QTest.mousePress(editor, Qt.LeftButton, Qt.NoModifier, QPoint(200, 80))
    QTest.mouseMove(editor, QPoint(280, 160))
    QTest.mouseRelease(editor, Qt.LeftButton, Qt.NoModifier, QPoint(280, 160))
    assert editor.boxes()[-1] == (50, 20, 70, 40)
    QTest.mousePress(editor, Qt.LeftButton, Qt.NoModifier, QPoint(60, 60))
    QTest.mouseMove(editor, QPoint(100, 60))
    QTest.mouseRelease(editor, Qt.LeftButton, Qt.NoModifier, QPoint(100, 60))
    assert editor.boxes()[0] == (20, 10, 30, 20)
    QTest.mouseClick(editor, Qt.RightButton, Qt.NoModifier, QPoint(240, 120))
    assert editor.boxes() == [(20, 10, 30, 20)]
    assert editor.provenance()["moved"] == [0]


def test_any_named_collection_gets_its_own_community_dataset(monkeypatch, tmp_path):
    """Item 528's route: a user-named collection, image + mask pairs."""
    assert model_share.community_repo("Toxoplasma PV!") == \
        "einarolafsson/community_toxoplasma_pv"
    assert model_share.community_repo("community_nuclei") == \
        "einarolafsson/community_nuclei"
    assert model_share.community_layout("toxoplasma pv") == "masks"
    assert model_share.conscience_for("toxoplasma pv") == model_share.MASK_CONSCIENCE
    image = _png(tmp_path / "cell.png")
    labels = np.zeros((60, 60), np.int32)
    labels[5:15, 5:15] = 4
    folder = model_share.write_contribution(
        "toxoplasma pv", [{"name": "cell.png", "source": str(image),
                           "labels": labels, "extra": {"channel": "GFP"}}],
        tmp_path / "out", consent={"licence": "CC BY 4.0"}, contribution_id="c1")
    record = json.loads((folder / "contribution.json").read_text())
    assert record["repo"] == "einarolafsson/community_toxoplasma_pv"
    assert record["objects"] == 1 and record["layout"] == "masks"
    meta = json.loads((folder / "meta" / "cell.json").read_text())
    assert meta["channel"] == "GFP" and meta["provenance"]["added"] == 1

    made = {}

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_exists(self, repo_id, repo_type=None):
            return False

        def create_repo(self, repo_id, **kwargs):
            made["repo"] = (repo_id, kwargs)

        def upload_file(self, **kwargs):
            made["readme"] = kwargs

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    url = model_share.ensure_community_repo("toxoplasma pv", "tok",
                                            purpose="Trains the PV model.")
    assert url.endswith("/datasets/einarolafsson/community_toxoplasma_pv")
    assert made["repo"][1]["repo_type"] == "dataset"
    assert made["repo"][1]["private"] is False
    readme = made["readme"]["path_or_fileobj"].decode()
    assert "CC BY 4.0" in readme and "Trains the PV model." in readme
    assert "reviewed" in readme


def test_the_consent_helper_asks_once_for_every_screen(prefs):
    asked = []
    assert msd.ask_community_consent(ask=lambda: asked.append(1) or True)
    assert msd.ask_community_consent(ask=lambda: asked.append(2) or True)
    assert asked == [1]


def _widget_point_on(canvas, layer, label):
    from PySide6.QtCore import QPoint

    view = canvas._ensure_canvas()
    for y in range(4, canvas.height() - 4, 3):
        for x in range(4, canvas.width() - 4, 3):
            around = {layer.label_at_world(view.world_at(y + dy, x + dx))
                      for dy in (-4, -1, 0, 3) for dx in (-4, -1, 0, 3)}
            if around == {label}:
                return QPoint(x, y)
    raise AssertionError(f"label {label} is not on screen")


def test_a_whole_plaque_is_deleted_with_one_click_and_recorded(panel, qtbot, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    image = _png(tmp_path / "well.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(segment=_square)
    upload = _Upload()
    dialog = panel.contribute_training_data(upload=upload)
    qtbot.addWidget(dialog)
    _consent(dialog)
    dialog.resize(900, 600)
    dialog.show()
    page = dialog.editor(image)
    canvas = page.canvas
    qtbot.waitUntil(lambda: canvas.width() > 100)
    session = page.brush.session

    page.delete_button.click()
    assert page.delete_button.isChecked()
    assert not page.brush.paint_button.isChecked()
    assert canvas.tool is page._delete_tool

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     _widget_point_on(canvas, page.layer, 0))
    assert set(np.unique(page.labels())) == {0, 1, 2}
    assert [e.kind for e in session.log.edits] == []

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     _widget_point_on(canvas, page.layer, 2))
    assert set(np.unique(page.labels())) == {0, 1}
    assert int((page.labels() == 1).sum()) == 400
    edit = session.log.edits[-1]
    assert (edit.kind, edit.target, edit.n_changed) == ("delete", 2, 100)
    assert page.deleted_objects() == [{"label": 2, "pixels": 100,
                                       "seeded": True}]

    QTest.keyClick(canvas, Qt.Key_Backspace)
    assert set(np.unique(page.labels())) == {0, 1, 2}
    assert page.deleted_objects() == []
    assert session.log.edits[-1].kind == "undo"

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier,
                     _widget_point_on(canvas, page.layer, 2))
    assert set(np.unique(page.labels())) == {0, 1}

    page.brush.paint_button.click()
    assert not page.delete_button.isChecked()
    assert canvas.tool is page.brush.tool

    assert dialog.upload()
    folder, _kind = upload.calls[0]
    meta = json.loads((folder / "meta" / "well.json").read_text())
    assert meta["deleted_objects"] == [{"label": 2, "pixels": 100,
                                        "seeded": True}]
    assert meta["provenance"]["removed"] == [2]
    assert meta["objects"] == 1


def test_the_curation_session_deletes_a_whole_object_as_one_undoable_stroke():
    from spacr.curation import MaskCuration
    from spacr.layers import LayerStack

    data = _square().astype(np.int64)
    layer = LayerStack().add_labels(data.copy(), name="m")
    session = MaskCuration(layer)
    assert session.delete_object(0) is None
    assert session.delete_object(9) is None
    assert len(session.log) == 0
    edit = session.delete_object(1)
    assert (edit.kind, edit.target, edit.n_changed) == ("delete", 1, 400)
    assert not (np.asarray(layer.data) == 1).any()
    assert (np.asarray(layer.data) == 2).sum() == 100
    session.undo()
    assert np.array_equal(np.asarray(layer.data), data)
