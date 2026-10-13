"""Item 523/528: community training data without a Hugging Face login.

The upload Space (tools/model_upload_space/) gains a ``contribute``
endpoint. Its handler, ``contribution.receive``, is plain Python with the
Hugging Face client passed in, so it runs here against a fake HfApi: a
folder made by ``write_contribution`` becomes a pull request on the right
dataset, and unknown targets, unpaired images, disallowed files, disguised
files, unsafe archives and oversized contributions are refused before any
call to Hugging Face.

On spaCR's side, ``contribute`` and ``upload_with_own_login`` prefer the
contributor's own login and fall back to the Space without one. The Space's
HTTP is faked by a stand-in ``urlopen`` that runs the real handler, so the
round trip (tar, upload, call, reply) is exercised with nothing leaving the
machine.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import model_share  # noqa: E402
from spacr.qt.widgets import model_share_dialog as msd  # noqa: E402

SPACE = Path(__file__).resolve().parents[2] / "tools" / "model_upload_space"


def _load_space_handler():
    spec = importlib.util.spec_from_file_location(
        "spacr_space_contribution", SPACE / "contribution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


space = _load_space_handler()


class FakeHfApi:
    """Records every call; keeps a copy of each folder sent as a PR."""

    def __init__(self, exists=True, fail=None):
        self.exists = exists
        self.fail = fail
        self.calls = []
        self.sent = {}

    def repo_exists(self, repo_id, repo_type=None):
        self.calls.append(("repo_exists", repo_id, repo_type))
        return self.exists

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs["path_in_repo"],
                           kwargs.get("create_pr", False)))

    def upload_folder(self, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append(("upload_folder", kwargs))
        root = Path(kwargs["folder_path"])
        self.sent = {p.relative_to(root).as_posix(): p.read_bytes()
                     for p in root.rglob("*") if p.is_file()}

        class Info:
            pr_url = (f"https://huggingface.co/datasets/{kwargs['repo_id']}"
                      "/discussions/7")
        return Info()

    def writes(self):
        return [c for c in self.calls if c[0] != "repo_exists"]


CONSENT = {"rights": True, "licence": "CC BY 4.0"}


def _masks_contribution(tmp_path, target="community_toxo_vacuoles", n=2):
    items = []
    for i in range(n):
        image = np.full((40, 50), 30 * (i + 1), np.uint8)
        source = tmp_path / f"cell{i}.png"
        imageio.imwrite(source, image)
        labels = np.zeros((40, 50), np.uint16)
        labels[5:15, 5:15] = 1
        labels[20:30, 20:35] = 2
        items.append({"name": source.name, "source": str(source),
                      "labels": labels})
    return model_share.write_contribution(
        target, items, tmp_path / "out", consent=CONSENT,
        contribution_id="20260926-test-ab12cd34")


def _figures_contribution(tmp_path):
    page = np.zeros((60, 80, 3), np.uint8)
    return model_share.write_contribution(
        "figures", [{"name": "fig1.png", "image": page,
                     "boxes": [(5, 5, 30, 30), (40, 10, 70, 50)]}],
        tmp_path / "out", consent=CONSENT,
        contribution_id="20260926-figs-00000001")


def _tar(folder, tmp_path, name=None):
    path = tmp_path / (name or f"{Path(folder).name}.tar")
    with tarfile.open(path, "w") as tar:
        tar.add(str(folder), arcname=Path(folder).name)
    return path


def test_a_masks_contribution_becomes_a_pull_request_on_its_dataset(tmp_path):
    folder = _masks_contribution(tmp_path)
    api = FakeHfApi()
    reply = space.receive(_tar(folder, tmp_path), "community_toxo_vacuoles",
                          api, who="10.0.0.1", work_dir=str(tmp_path))
    assert reply.startswith("ok: https://huggingface.co/datasets/"
                            "einarolafsson/community_toxo_vacuoles/discussions/7")
    [call] = [c for c in api.calls if c[0] == "upload_folder"]
    kwargs = call[1]
    assert kwargs["create_pr"] is True
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["repo_id"] == "einarolafsson/community_toxo_vacuoles"
    assert kwargs["path_in_repo"] == "contributions/20260926-test-ab12cd34"
    assert "10.0.0.1" not in kwargs["commit_description"]
    assert api.writes() == [call], "a PR and nothing committed directly"
    original = {p.relative_to(folder).as_posix(): p.read_bytes()
                for p in folder.rglob("*") if p.is_file()}
    assert api.sent == original
    assert not [p for p in tmp_path.iterdir()
                if p.name.startswith("contribution-")], "scratch removed"


def test_figures_and_plaques_go_to_plaque_assays_two_datasets(tmp_path):
    figures = _figures_contribution(tmp_path)
    api = FakeHfApi()
    reply = space.receive(_tar(figures, tmp_path), "figures", api)
    assert reply.startswith("ok:"), reply
    assert api.calls[-1][1]["repo_id"] == space.FIGURES_REPO
    assert set(api.sent) >= {"labels/fig1.txt", "images/fig1.png",
                             "meta/fig1.json", "contribution.json"}

    (tmp_path / "p").mkdir()
    plaques = _masks_contribution(tmp_path / "p", target="plaques")
    api = FakeHfApi()
    reply = space.receive(_tar(plaques, tmp_path, "p.tar"),
                          space.PLAQUES_REPO, api)
    assert reply.startswith("ok:"), reply
    assert api.calls[-1][1]["repo_id"] == space.PLAQUES_REPO


@pytest.mark.parametrize("target", [
    "einarolafsson/user-models", "someone/community_x", "community_../x",
    "", "community_", "Community_Upper", "einarolafsson/spacr-model-upload",
    "community_x/../../y", "datasets/einarolafsson/community_x"])
def test_unknown_targets_are_refused_before_anything_is_read(tmp_path,
                                                             target):
    api = FakeHfApi()
    reply = space.receive(tmp_path / "does-not-exist.tar", target, api)
    assert reply.startswith("error: unknown target"), reply
    assert api.calls == []


def test_a_masks_contribution_sent_as_figures_is_refused(tmp_path):
    folder = _masks_contribution(tmp_path)
    api = FakeHfApi()
    reply = space.receive(_tar(folder, tmp_path), "figures", api)
    assert reply.startswith("error:") and "masks/" in reply
    assert api.calls == []


def _broken(tmp_path, change):
    folder = _masks_contribution(tmp_path)
    change(folder)
    api = FakeHfApi()
    reply = space.receive(_tar(folder, tmp_path), "community_toxo_vacuoles",
                          api)
    assert reply.startswith("error:"), reply
    assert api.calls == [], "nothing reaches Hugging Face"
    return reply


def test_an_image_without_its_mask_is_refused_and_named(tmp_path):
    reply = _broken(tmp_path, lambda f: (f / "masks" / "cell1.tif").unlink())
    assert "no masks/ file for: cell1" in reply


def test_a_mask_without_its_image_is_refused_and_named(tmp_path):
    reply = _broken(tmp_path, lambda f: shutil.copy(
        f / "masks" / "cell0.tif", f / "masks" / "stray.tif"))
    assert "no image for masks/: stray" in reply


def test_a_missing_meta_file_is_refused(tmp_path):
    reply = _broken(tmp_path, lambda f: (f / "meta" / "cell0.json").unlink())
    assert "no meta/ file for: cell0" in reply


@pytest.mark.parametrize("extra", ["images/run.exe", "masks/cell0.png",
                                   "script.py", "images/cell9.svg",
                                   "extra/notes.txt"])
def test_files_outside_the_allow_list_are_refused(tmp_path, extra):
    def add(folder):
        target = folder / extra
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    reply = _broken(tmp_path, add)
    assert extra.split("/")[-1] in reply or "not part of" in reply


def test_a_file_that_is_not_what_its_name_says_is_refused(tmp_path):
    reply = _broken(tmp_path, lambda f: (f / "images" / "cell0.png")
                    .write_bytes(b"#!/bin/sh\necho hi\n"))
    assert "images/cell0.png is not a .png file" in reply


def test_a_contribution_without_consent_or_licence_is_refused(tmp_path):
    def strip(folder):
        path = folder / "contribution.json"
        record = json.loads(path.read_text())
        record.pop("consent")
        path.write_text(json.dumps(record))
    assert "consent" in _broken(tmp_path, strip)


def _tar_with(tmp_path, member, data=b"x"):
    path = tmp_path / "evil.tar"
    with tarfile.open(path, "w") as tar:
        tar.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return path


@pytest.mark.parametrize("make", [
    lambda: tarfile.TarInfo("../escape.json"),
    lambda: tarfile.TarInfo("/abs/contribution.json"),
    lambda: tarfile.TarInfo("id/../../x.json"),
])
def test_unsafe_archive_paths_are_refused(tmp_path, make):
    member = make()
    member.size = 1
    api = FakeHfApi()
    reply = space.receive(_tar_with(tmp_path, member),
                          "community_toxo_vacuoles", api)
    assert reply.startswith("error: unsafe path"), reply
    assert api.calls == []


def test_links_in_the_archive_are_refused(tmp_path):
    folder = _masks_contribution(tmp_path)
    path = _tar(folder, tmp_path)
    with tarfile.open(path, "a") as tar:
        link = tarfile.TarInfo(f"{folder.name}/images/passwd.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    api = FakeHfApi()
    reply = space.receive(path, "community_toxo_vacuoles", api)
    assert "not a regular file" in reply and api.calls == []


def test_the_size_caps_refuse_before_anything_is_sent(tmp_path, monkeypatch):
    folder = _masks_contribution(tmp_path)
    archive = _tar(folder, tmp_path)
    monkeypatch.setattr(space, "MAX_TOTAL_BYTES", 100)
    api = FakeHfApi()
    assert "is over" in space.receive(archive, "community_toxo_vacuoles", api)
    monkeypatch.setattr(space, "MAX_TOTAL_BYTES", 10 ** 9)
    monkeypatch.setattr(space, "MAX_ARCHIVE_BYTES", 100)
    assert "over the 100 limit" in space.receive(
        archive, "community_toxo_vacuoles", api)
    monkeypatch.setattr(space, "MAX_ARCHIVE_BYTES", 10 ** 9)
    monkeypatch.setattr(space, "MAX_FILES", 3)
    assert "over 3 files" in space.receive(
        archive, "community_toxo_vacuoles", api)
    assert api.calls == []


def test_a_new_community_name_gets_its_dataset_only_when_allowed(tmp_path):
    folder = _masks_contribution(tmp_path)
    archive = _tar(folder, tmp_path)
    api = FakeHfApi(exists=False)
    reply = space.receive(archive, "community_toxo_vacuoles", api)
    assert reply.startswith("ok:"), reply
    kinds = [c[0] for c in api.writes()]
    assert kinds == ["create_repo", "upload_file", "upload_folder"]
    assert api.writes()[1][1:] == ("README.md", False)

    api = FakeHfApi(exists=False)
    reply = space.receive(archive, "community_toxo_vacuoles", api,
                          create_new=False)
    assert "does not exist" in reply and api.writes() == []

    plaques = tmp_path / "pl"
    plaques.mkdir()
    api = FakeHfApi(exists=False)
    reply = space.receive(_tar(_masks_contribution(plaques, "plaques"),
                               plaques), "plaques", api)
    assert "does not exist" in reply and api.writes() == []


def test_a_hugging_face_failure_is_reported_not_raised(tmp_path):
    folder = _masks_contribution(tmp_path)
    api = FakeHfApi(fail=RuntimeError("403 Forbidden"))
    reply = space.receive(_tar(folder, tmp_path), "community_toxo_vacuoles",
                          api)
    assert reply == "error: RuntimeError: 403 Forbidden"


class FakeSpace:
    """A stand-in for urllib's urlopen that serves the Space's two routes
    by running the real handler against a fake HfApi."""

    def __init__(self, tmp_path, api=None):
        self.tmp_path = tmp_path
        self.api = api or FakeHfApi()
        self.files = {}
        self.urls = []
        self.reply = None

    def __call__(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.urls.append(url)
        if url.endswith("/gradio_api/upload"):
            body = request.data
            name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
            start = body.index(b"\r\n\r\n") + 4
            end = body.rindex(b"\r\n--")
            saved = self.tmp_path / "space_upload" / name
            saved.parent.mkdir(exist_ok=True)
            saved.write_bytes(body[start:end])
            return io.BytesIO(json.dumps([str(saved)]).encode())
        if url.endswith("/gradio_api/call/contribute"):
            handle, target = json.loads(request.data)["data"]
            assert handle["meta"]["_type"] == "gradio.FileData"
            self.reply = space.receive(handle["path"], target, self.api)
            return io.BytesIO(b'{"event_id": "e1"}')
        if url.endswith("/gradio_api/call/contribute/e1"):
            text = f"event: complete\ndata: {json.dumps([self.reply])}\n"
            return io.BytesIO(text.encode())
        raise AssertionError(f"unexpected request {url}")


def test_without_a_login_contribute_goes_through_the_space(tmp_path,
                                                          monkeypatch):
    import urllib.request

    folder = _masks_contribution(tmp_path)
    fake = FakeSpace(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(model_share, "find_token", lambda: None)
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://space.test")

    url = model_share.contribute(folder, "community_toxo_vacuoles")
    assert url == ("https://huggingface.co/datasets/einarolafsson/"
                   "community_toxo_vacuoles/discussions/7")
    assert fake.urls[0] == "https://space.test/gradio_api/upload"
    call = [c for c in fake.api.calls if c[0] == "upload_folder"][0][1]
    assert call["create_pr"] is True
    assert fake.api.sent["contribution.json"] == \
        (folder / "contribution.json").read_bytes()


def test_upload_with_own_login_falls_back_to_the_space(tmp_path,
                                                       monkeypatch):
    import urllib.request

    folder = _figures_contribution(tmp_path)
    fake = FakeSpace(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(model_share, "find_token", lambda: None)
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://space.test")
    url = msd.upload_with_own_login(folder, "figures")
    assert "community_toxoplasma_plaque_figures/discussions/7" in url


def test_a_space_refusal_reaches_the_user_with_a_way_round_it(tmp_path,
                                                             monkeypatch):
    import urllib.request

    folder = _masks_contribution(tmp_path)
    (folder / "masks" / "cell1.tif").unlink()
    fake = FakeSpace(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(model_share, "find_token", lambda: None)
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://space.test")
    with pytest.raises(RuntimeError) as caught:
        msd.upload_with_own_login(folder, "community_toxo_vacuoles")
    text = str(caught.value)
    assert "no masks/ file for: cell1" in text
    assert "huggingface-cli login" in text
    assert fake.api.calls == []


def test_a_login_is_preferred_over_the_space(tmp_path, monkeypatch):
    import huggingface_hub

    folder = _masks_contribution(tmp_path)
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)
    monkeypatch.setattr(model_share, "find_token", lambda: "user-token")

    def no_space(*_a, **_k):
        raise AssertionError("the Space must not be used with a login")

    monkeypatch.setattr(model_share, "central_contribute", no_space)
    url = msd.upload_with_own_login(folder, "community_toxo_vacuoles")
    assert url.endswith("/discussions/7")
    assert api.calls[-1][1]["create_pr"] is True


def test_a_login_that_cannot_create_a_new_dataset_uses_the_space(
        tmp_path, monkeypatch):
    folder = _masks_contribution(tmp_path)
    used = []

    def cannot_create(target, token, **_k):
        raise RuntimeError("this account cannot create it")

    monkeypatch.setattr(model_share, "ensure_community_repo", cannot_create)
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://space.test")
    monkeypatch.setattr(model_share, "central_contribute",
                        lambda f, t: used.append((f, t)) or "pr-url")
    assert model_share.contribute(folder, "community_new_thing",
                                  "user-token") == "pr-url"
    assert used == [(folder, "community_new_thing")]


def test_the_client_refuses_an_oversized_folder_before_sending(tmp_path,
                                                               monkeypatch):
    import urllib.request

    folder = _masks_contribution(tmp_path)
    monkeypatch.setattr(model_share, "MAX_CONTRIBUTION_BYTES", 10)
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "https://space.test")

    def no_network(*_a, **_k):
        raise AssertionError("nothing may be sent")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(RuntimeError, match="upload service takes about"):
        model_share.central_contribute(folder, "community_toxo_vacuoles")


def test_the_space_app_exposes_the_contribute_endpoint():
    source = (SPACE / "app.py").read_text()
    assert 'api_name="contribute"' in source
    assert 'api_name="upload"' in source
    assert "import contribution" in source
    assert "create_pr=True" in (SPACE / "contribution.py").read_text()
    assert os.path.basename(space.__file__) == "contribution.py"
