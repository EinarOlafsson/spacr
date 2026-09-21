"""Model sharing preserves files and metadata, and reports failed transfers."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from spacr.qt.widgets import model_share as sharing


@pytest.fixture(autouse=True)
def no_live_upload(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a sharing test attempted an unconfigured external request")
    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_training_archive_preserves_relative_names_and_bytes(tmp_path):
    source = tmp_path / "training"
    (source / "masks").mkdir(parents=True)
    (source / "image.tif").write_bytes(b"image\x00\xff")
    (source / "masks" / "image.tif").write_bytes(b"mask\x00\xfe")
    archive = sharing.pack_training_data(str(source), str(tmp_path))
    with tarfile.open(archive) as packed:
        files = {entry.name: packed.extractfile(entry).read()
                 for entry in packed.getmembers() if entry.isfile()}
    assert files == {"training/image.tif": b"image\x00\xff",
                     "training/masks/image.tif": b"mask\x00\xfe"}


def test_training_archive_refuses_missing_or_oversized_input_before_creating_tar(
        tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="not a folder"):
        sharing.pack_training_data(str(tmp_path / "absent"), str(tmp_path))
    source = tmp_path / "training"
    source.mkdir()
    (source / "image.tif").write_bytes(b"12345")
    monkeypatch.setattr(sharing, "MAX_TRAIN_BYTES", 4)
    with pytest.raises(RuntimeError, match="limit"):
        sharing.pack_training_data(str(source), str(tmp_path))
    assert list(tmp_path.glob("*.tar")) == []


def _endpoint(monkeypatch, replies):
    requests = []
    responses = iter(replies)
    def respond(request, timeout):
        requests.append((request, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(response)
    monkeypatch.setattr("urllib.request.urlopen", respond)
    monkeypatch.setattr(sharing, "CENTRAL_ENDPOINT", "https://upload.invalid/")
    return requests


@pytest.mark.parametrize("with_training", [False, True])
def test_central_upload_sends_checkpoint_scorecard_and_optional_real_archive(
        tmp_path, monkeypatch, with_training):
    model = tmp_path / "weights.bin"
    model.write_bytes(b"checkpoint\x00\xff")
    fields = {"display_name": "Cells", "kind": "classifier", "trained_on": "plate A",
              "contact": "scientist@example.invalid", "f1": "0.8"}
    replies = [b'["/server/weights.bin"]']
    if with_training:
        source = tmp_path / "training"
        source.mkdir()
        (source / "image.tif").write_bytes(b"image pixels")
        fields["train_data_dir"] = str(source)
        replies.append(b'["/server/training.tar"]')
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    replies += [b'{"event_id":"event-1"}',
                b'event: complete\ndata: ["https://models.invalid/result"]\n']
    requests = _endpoint(monkeypatch, replies)
    assert sharing.central_upload(str(model), fields) == "https://models.invalid/result"
    upload, timeout = requests[0]
    assert upload.full_url == "https://upload.invalid/gradio_api/upload"
    assert timeout == 600
    assert b'filename="weights.bin"' in upload.data
    assert b"checkpoint\x00\xff" in upload.data
    assert "multipart/form-data; boundary=" in upload.get_header("Content-type")
    call, timeout = requests[-2]
    assert call.full_url == "https://upload.invalid/gradio_api/call/upload"
    assert timeout == 120
    data = json.loads(call.data)["data"]
    assert data[:4] == [{"path": "/server/weights.bin", "meta": {"_type": "gradio.FileData"}},
                        "Cells", "classifier", "plate A"]
    assert json.loads(data[4]) == fields
    assert data[5] == fields["contact"]
    assert requests[-1] == ("https://upload.invalid/gradio_api/call/upload/event-1", 120)
    if with_training:
        archive_request, timeout = requests[1]
        assert timeout == 7200
        header, multipart = archive_request.data.split(b"\r\n\r\n", 1)
        assert b'.tar"' in header
        # Parse the tar bytes independently of the production packer.
        with tarfile.open(fileobj=io.BytesIO(multipart), mode="r:") as archive:
            assert archive.extractfile("training/image.tif").read() == b"image pixels"
        assert data[6] == {"path": "/server/training.tar", "meta": {"_type": "gradio.FileData"}}
        assert not list(tmp_path.glob("*.tar"))
    else:
        assert data[6] is None


def test_upload_refuses_unconfigured_endpoint_before_reading_file(monkeypatch):
    monkeypatch.setattr(sharing, "CENTRAL_ENDPOINT", "")
    with pytest.raises(RuntimeError, match="no central upload endpoint"):
        sharing.central_upload("absent.bin", {})


@pytest.mark.parametrize("reply,error", [
    (b'event: complete\ndata: ["error: rejected checkpoint"]', "rejected checkpoint"),
    (OSError("connection lost"), "connection lost"),
])
def test_upload_reports_server_and_transport_errors(tmp_path, monkeypatch, reply, error):
    model = tmp_path / "model"
    model.write_bytes(b"checkpoint")
    _endpoint(monkeypatch, [b'["/server/model"]', b'{"event_id":"one"}', reply])
    with pytest.raises((RuntimeError, OSError), match=error):
        sharing.central_upload(str(model), {})


def test_pending_upload_is_polled_then_times_out(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.write_bytes(b"checkpoint")
    requests = _endpoint(monkeypatch, [b'["/server/model"]', b'{"event_id":"one"}',
                                      b'event: generating\ndata: null'])
    times = iter([100, 101, 701])
    monkeypatch.setattr("time.time", lambda: next(times))
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    with pytest.raises(RuntimeError, match="did not answer in time"):
        sharing.central_upload(str(model), {})
    sleep.assert_called_once_with(3)
    assert len(requests) == 3


@pytest.mark.parametrize("name,expected", [(" Cells / A_2 ", "cells-a-2"), ("///", "model")])
def test_folder_name_is_repository_safe(name, expected):
    assert sharing.slugify(name) == expected


@pytest.mark.parametrize("primary,alternate,expected", [
    (" test-primary ", "test-alternate", "test-primary"),
    (None, " test-alternate ", "test-alternate"),
])
def test_explicit_token_precedes_cached_login(monkeypatch, primary, alternate, expected):
    if primary:
        monkeypatch.setenv("HF_TOKEN", primary)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", alternate)
    assert sharing.find_token() == expected


@pytest.mark.parametrize("raises", [False, True])
def test_cached_login_or_unavailable_login(monkeypatch, raises):
    def token():
        if raises:
            raise OSError("login unavailable")
        return "test-cached-token"
    monkeypatch.setattr(huggingface_hub, "HfFolder", SimpleNamespace(get_token=token), raising=False)
    assert sharing.find_token() == (None if raises else "test-cached-token")


@pytest.mark.parametrize("identity", [
    {"name": "einarolafsson"},
    {"name": "contributor", "orgs": [{"name": "unrelated"}, {"name": "einarolafsson"}]},
    {"name": "contributor", "orgs": None},
])
def test_reachable_shared_repository_is_the_first_upload_target(monkeypatch, identity):
    api = Mock()
    api.whoami.return_value = identity
    factory = Mock(return_value=api)
    monkeypatch.setattr(huggingface_hub, "HfApi", factory)
    assert sharing.target_repo("test-token") == (sharing.SHARE_REPO, True)
    factory.assert_called_once_with(token="test-token")
    api.repo_info.assert_called_once_with(sharing.SHARE_REPO, repo_type="model")


@pytest.mark.parametrize("identified", [False, True])
def test_inaccessible_shared_repo_falls_back_only_to_identified_user(monkeypatch, identified):
    api = Mock()
    api.repo_info.side_effect = OSError("unavailable")
    if identified:
        api.whoami.return_value = {"name": "contributor"}
    else:
        api.whoami.side_effect = OSError("invalid token")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: api)
    assert sharing.target_repo("test-token") == (
        ("contributor/spacr-models", False) if identified else ("", False))


def test_sharing_uploads_exact_checkpoint_and_hashed_card_to_staging(tmp_path, monkeypatch):
    model = tmp_path / "weights.bin"
    model.write_bytes(b"checkpoint bytes")
    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: api)
    monkeypatch.setattr(sharing, "target_repo", lambda token: ("contributor/spacr-models", False))
    fields = {"display_name": "Plate A", "train_loss": "0.25", "val_loss": "0.5"}
    assert sharing.share(str(model), fields, "test-token") == (
        "https://huggingface.co/contributor/spacr-models/tree/main/staging/plate-a")
    api.create_repo.assert_called_once_with("contributor/spacr-models", repo_type="model", exist_ok=True)
    checkpoint, readme = [call.kwargs for call in api.upload_file.call_args_list]
    assert checkpoint == dict(path_or_fileobj=str(model), path_in_repo="staging/plate-a/weights.bin",
                              repo_id="contributor/spacr-models", repo_type="model")
    assert readme["path_in_repo"] == "staging/plate-a/README.md"
    text = readme["path_or_fileobj"].decode()
    assert hashlib.sha256(model.read_bytes()).hexdigest() in text
    assert "| +0.2500 |" in text
    assert "not validated by" in text
    assert "not recorded" in text
    assert "Not stated by the uploader." in text


def test_unknown_upload_destination_is_refused_before_transfer(monkeypatch):
    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: api)
    monkeypatch.setattr(sharing, "target_repo", lambda token: ("", False))
    with pytest.raises(RuntimeError, match="check the token"):
        sharing.share("absent.bin", {}, "test-token")
    api.create_repo.assert_not_called()
    api.upload_file.assert_not_called()


def test_missing_losses_stay_missing_and_reported_values_survive_in_card():
    text = sharing.card({"trained_on": "plate A", "notes": "one cell type only", "f1": "0.81"},
                        "weights.bin", "digest", "owner/models", "staging/weights")
    assert text.startswith("# weights.bin\n")
    assert "**0.81**" in text
    assert "| — | not recorded |" in text
    assert "plate A" in text and "one cell type only" in text
    assert "`staging/weights/weights.bin`" in text
