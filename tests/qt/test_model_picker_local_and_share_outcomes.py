"""A listed model stays usable when sharing is declined or fails."""
from types import SimpleNamespace

import pytest
from PySide6 import QtCore, QtWidgets

from spacr import model_zoo
from spacr.qt.widgets import model_share, model_share_dialog
from spacr.qt.widgets import model_zoo_picker as module


@pytest.fixture
def picker(qtbot, tmp_path, monkeypatch):
    # Opening a picker in these controller tests must never start a probe.
    monkeypatch.setattr(module.ModelZooPicker, "_warm_the_community_catalogue", lambda self: None)
    monkeypatch.setattr(module.ModelZooPicker, "_probe_backends", lambda self: None)
    monkeypatch.setattr(module, "remembered_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(module, "remembered_sources", lambda: ("spaCR", "cellposeSAM"))
    monkeypatch.setattr(model_zoo, "catalogue", lambda **kwargs: [])
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "")
    monkeypatch.setattr(model_share, "find_token", lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail("an upload was not explicitly faked by this test")
    monkeypatch.setattr(model_share, "share", forbidden)
    monkeypatch.setattr(model_share, "central_upload", forbidden)
    dialog = module.ModelZooPicker(kinds=("cellpose",))
    qtbot.addWidget(dialog)
    return dialog


@pytest.fixture
def messages(monkeypatch):
    calls = []
    box = SimpleNamespace(Yes=1, No=2)
    box.question = lambda *args: calls.append(("question", args)) or box.No
    box.information = lambda *args: calls.append(("information", args))
    box.warning = lambda *args: calls.append(("warning", args))
    monkeypatch.setattr(QtWidgets, "QMessageBox", box)
    monkeypatch.setattr(module, "QMessageBox", box)
    return box, calls


def choose_file(monkeypatch, path):
    monkeypatch.setattr(QtWidgets, "QFileDialog", SimpleNamespace(
        getOpenFileName=lambda *args: (str(path), "")))


def share_dialog(monkeypatch, *, accepted=True):
    opened = []
    fields = {"trained_on": "test images", "licence": "CC0"}
    class Dialog:
        def __init__(self, name, parent):
            opened.append(name)
        def exec(self):
            return int(accepted)
        def values(self):
            return fields.copy()
    monkeypatch.setattr(model_share_dialog, "ShareDialog", Dialog)
    return opened, fields


def test_cancelled_add_does_not_list_or_request_sharing(picker, messages, monkeypatch):
    choose_file(monkeypatch, "")
    before = list(picker._entries)
    picker._add_model()
    assert picker._entries == before
    assert messages[1] == []


def test_unreadable_file_reports_failure_without_listing(picker, messages, monkeypatch, tmp_path):
    path = tmp_path / "not-a-model.pt"
    path.write_bytes(b"not weights")
    choose_file(monkeypatch, path)
    before = list(picker._entries)
    def invalid(path):
        raise ValueError("no supported tensor payload")
    monkeypatch.setattr(model_zoo, "entry_from_file", invalid)
    picker._add_model()
    assert "no supported tensor payload" in picker.status.text()
    assert picker._entries == before and messages[1] == []


@pytest.mark.parametrize("accept_share", [False, True])
def test_local_model_is_usable_independently_of_sharing(picker, messages, monkeypatch, tmp_path, accept_share):
    path = tmp_path / "my_model.pt"
    path.write_bytes(b"local weights")
    entry = model_zoo.ModelEntry(key="my_model", name=path.name, path=str(path))
    choose_file(monkeypatch, path)
    monkeypatch.setattr(model_zoo, "entry_from_file", lambda selected: entry)
    box, calls = messages
    box.question = lambda *args: calls.append(("question", args)) or (box.Yes if accept_share else box.No)
    shared = []
    monkeypatch.setattr(picker, "_share_model", shared.append)
    picker._add_model()
    assert entry in picker._entries
    assert "usable now" in picker.status.text()
    assert shared == ([str(path)] if accept_share else [])
    assert calls[0][1][-1] == box.No
    group = next(i for i, (_, versions) in enumerate(picker._groups)
                 if any(model is entry for _, model in versions))
    picker.table.selectRow(picker._row_of_group(group))
    assert picker.use_button.isEnabled()
    picker._accept_selected()
    assert picker.chosen_path() == str(path)


def test_cancelling_central_scorecard_does_not_read_token_or_upload(picker, messages, monkeypatch):
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://example.invalid/upload")
    opened, fields = share_dialog(monkeypatch, accepted=False)
    def forbidden():
        pytest.fail("a cancelled share must not look for credentials")
    monkeypatch.setattr(model_share, "find_token", forbidden)
    picker._share_model("/local/model.pt")
    assert opened == ["model.pt"] and messages[1] == []


def test_successful_central_share_reports_reply_without_fallback(picker, messages, monkeypatch):
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://example.invalid/upload")
    opened, fields = share_dialog(monkeypatch)
    uploads = []
    monkeypatch.setattr(model_share, "central_upload",
                        lambda path, values: uploads.append((path, values)) or "Held for review")
    def forbidden():
        pytest.fail("central success must not read fallback credentials")
    monkeypatch.setattr(model_share, "find_token", forbidden)
    picker._share_model("/local/model.pt")
    assert uploads == [("/local/model.pt", fields)]
    assert opened == ["model.pt"]
    assert picker.status.text() == "Held for review"
    assert messages[1][0][1][1:] == ("Submitted", "Held for review")


def test_central_failure_without_token_keeps_failure_visible(picker, messages, monkeypatch):
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://example.invalid/upload")
    share_dialog(monkeypatch)
    def fail(*args):
        raise OSError("service unavailable")
    monkeypatch.setattr(model_share, "central_upload", fail)
    picker._share_model("/local/model.pt")
    assert "service unavailable" in picker.status.text()
    assert messages[1][0][1][1] == "Hugging Face login needed"


@pytest.mark.parametrize("central_failed", [False, True])
@pytest.mark.parametrize("upload_failed", [False, True])
def test_fallback_share_uses_one_scorecard_and_reports_the_outcome(
        picker, messages, monkeypatch, central_failed, upload_failed):
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT",
                        "https://example.invalid/upload" if central_failed else "")
    opened, fields = share_dialog(monkeypatch)
    monkeypatch.setattr(model_share, "find_token", lambda: "test-token")
    def central(*args):
        raise OSError("central unavailable")
    monkeypatch.setattr(model_share, "central_upload", central)
    uploads = []
    def upload(path, values, token):
        uploads.append((path, values, token))
        if upload_failed:
            raise OSError("write refused")
        return "https://example.invalid/model"
    monkeypatch.setattr(model_share, "share", upload)
    picker._share_model("/local/model.pt")
    assert opened == ["model.pt"]
    assert uploads == [("/local/model.pt", fields, "test-token")]
    if upload_failed:
        assert picker.status.text() == "Upload failed."
        assert messages[1][-1][0] == "warning"
        assert "write refused" in messages[1][-1][1][-1]
    else:
        assert picker.status.text() == "Shared: https://example.invalid/model"
        assert messages[1][-1][1][1] == "Shared"


def test_no_token_never_opens_a_scorecard(picker, messages, monkeypatch):
    opened, _ = share_dialog(monkeypatch)
    picker._share_model("/local/model.pt")
    assert opened == []
    assert messages[1][0][1][1] == "Hugging Face login needed"


def test_cancelling_fallback_scorecard_does_not_upload(picker, messages, monkeypatch):
    monkeypatch.setattr(model_share, "find_token", lambda: "test-token")
    opened, _ = share_dialog(monkeypatch, accepted=False)
    picker._share_model("/local/model.pt")
    assert opened == ["model.pt"] and messages[1] == []


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("unverified", [False, True])
def test_download_worker_reports_progress_and_exactly_one_ending(monkeypatch, failed, unverified):
    entry = object()
    seen = []
    def fetch(actual, folder, *, require_checksum, progress):
        assert actual is entry and folder == "/local/models"
        assert require_checksum is not unverified
        progress(12, None)
        if failed:
            raise ValueError("checksum mismatch")
        return "/local/models/weights.pt"
    monkeypatch.setattr(model_zoo, "fetch", fetch)
    worker = module._DownloadWorker(entry, "/local/models", unverified=unverified)
    worker.progressed.connect(lambda done, total: seen.append(("progress", done, total)))
    worker.finished.connect(lambda path: seen.append(("finished", path)))
    worker.failed.connect(lambda why: seen.append(("failed", why)))
    worker.run()
    assert seen == [("progress", 12, 0),
                    ("failed", "checksum mismatch") if failed else
                    ("finished", "/local/models/weights.pt")]


def test_failed_preferences_fall_back_without_hiding_the_picker(monkeypatch):
    class BrokenSettings:
        def __init__(self):
            raise OSError("settings store unavailable")
    monkeypatch.setattr(QtCore, "QSettings", BrokenSettings)
    assert module.remembered_model_dir() == module.DEFAULT_MODEL_DIR
    assert module.remembered_sources() == tuple(model_zoo.DEFAULT_ZOO_SOURCES)
    module._remember_model_dir("/local/models")
    module._remember_sources(("spaCR",))
