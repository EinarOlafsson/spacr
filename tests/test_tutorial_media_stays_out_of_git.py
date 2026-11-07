"""Tutorial media stays out of git, behind the pin, and its absence breaks nothing.

Posters, narration, example downloads and the release-candidate media live in
the Hugging Face dataset that ``docs/media_revision.json`` pins. These tests
keep them from being committed again, check the pin and the fetch tool's
offline behaviour, and check that the app and the docs staging cope with the
media not being downloaded.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The largest file a tutorial tree may commit. The biggest text file there
#: (a lesson catalog) is under 2 MB; a video, a narration set or an example
#: archive is far over.
TUTORIAL_FILE_CEILING = 3 * 1024 * 1024


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def media():
    return _load("fetch_docs_media_guard", ROOT / "tools" / "fetch_docs_media.py")


@pytest.fixture(scope="module")
def tracked():
    out = subprocess.run(["git", "-c", "gc.auto=0", "-C", str(ROOT), "ls-files", "-z"],
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [path for path in out.split("\0") if path]


def _pinned(media) -> dict:
    groups = media.load_pin()["groups"]
    return {**groups["tutorials"]["files"], **groups["tutorial_archive"]["files"]}


def test_no_tutorial_media_is_committed_unless_it_is_the_pinned_bytes(media, tracked):
    """A new or changed poster, narration or example archive fails here."""
    pinned = _pinned(media)
    offenders = []
    for path in tracked:
        if not media.is_tutorial_media(path):
            continue
        record = pinned.get(path)
        local = ROOT / path
        if record is None or not local.is_file() or media.sha256(local) != record["sha256"]:
            offenders.append(path)
    assert not offenders, (
        "tutorial media belongs in the staging folder and the pinned dataset, not in git: "
        + ", ".join(offenders[:20]))


def test_no_tutorial_sized_file_is_committed(media, tracked):
    """The size rule catches media of a kind the path rule does not name."""
    pinned = _pinned(media)
    large = [path for path in tracked
             if path.startswith(media.TUTORIAL_TREES) and path not in pinned
             and (ROOT / path).is_file()
             and (ROOT / path).stat().st_size > TUTORIAL_FILE_CEILING]
    assert not large, f"tutorial-sized files committed: {large}"


@pytest.mark.parametrize("path, expected", [
    ("docs/source/_extra/tutorials/production/07_mask/poster.jpg", True),
    ("docs/source/_extra/tutorials/examples/notes.csv", True),
    ("tools/tutorials/release_candidate/web/examples/a.zip", True),
    ("tools/tutorials/authoring/new_lesson/clip.mp4", True),
    ("./tools/tutorials/audio_repairs/x/af_heart.m4a", True),
    ("docs/source/_extra/tutorials/app_v2.js", False),
    ("docs/source/_extra/tutorials/catalog/lessons_en.json", False),
    ("tools/tutorials/release_candidate/web/lesson_catalog.js", False),
    ("tools/tutorials/capture_home.py", False),
    ("docs/source/_static/deck/slides/slide_01.jpg", False),
    ("spacr/resources/icons/logo.png", False),
])
def test_the_media_rule(media, path, expected):
    assert media.is_tutorial_media(path) is expected


def test_gitignore_keeps_the_media_folders_out(media):
    ignored = subprocess.run(
        ["git", "-c", "gc.auto=0", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(folder + "x.jpg" for folder in media.TUTORIAL_MEDIA_FOLDERS),
        capture_output=True, text=True).stdout.split()
    assert sorted(ignored) == sorted(folder + "x.jpg" for folder in media.TUTORIAL_MEDIA_FOLDERS)


def test_the_tutorial_groups_pin_every_file_under_its_tree_path(media):
    groups = media.load_pin()["groups"]
    for name in ("tutorials", "tutorial_archive"):
        spec = groups[name]
        assert spec["repo"] == "einarolafsson/spacr-tutorials"
        assert spec["repo_type"] == "dataset"
        assert spec["files"]
        for path, record in spec["files"].items():
            assert media.is_tutorial_media(path), path
            assert record["path_in_repo"] == media.TUTORIAL_PREFIX + path
            assert len(record["sha256"]) == 64 and record["size"] > 0
    site = groups["tutorials"]["files"]
    assert all(path.startswith(("docs/source/_extra/tutorials/production/",
                                "docs/source/_extra/tutorials/examples/"))
               and "/audio/" not in path for path in site)
    assert "docs/source/_extra/tutorials/examples/Cellpose_training_images_masks.zip" in site


def test_docs_build_fetches_the_tutorial_media_before_sphinx():
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    fetch = workflow.index("tools/fetch_docs_media.py fetch --group tutorials")
    assert fetch < workflow.index("tools/fetch_docs_media.py check --group tutorials")
    assert fetch < workflow.index("sphinx-build")


def _tiny_pin(tmp_path: Path, revision: str = "") -> Path:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    payload = b"poster bytes"
    import hashlib
    pin = {"groups": {"tutorials": {
        "repo": "einarolafsson/spacr-tutorials", "repo_type": "dataset", "revision": revision,
        "files": {"docs/source/_extra/tutorials/production/07_mask/poster.jpg": {
            "path_in_repo": "tree/docs/source/_extra/tutorials/production/07_mask/poster.jpg",
            "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}}}}}
    (root / "docs" / "media_revision.json").write_text(json.dumps(pin), encoding="utf-8")
    return root


def test_unpublished_media_is_reported_not_downloaded_without_a_request(media, tmp_path, monkeypatch):
    root = _tiny_pin(tmp_path)
    tree_path = "docs/source/_extra/tutorials/production/07_mask/poster.jpg"
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="has not been published"):
        media.media_path(tree_path, "tutorials", root / "docs" / "media_revision.json")
    assert media.available(tree_path, "tutorials", root) is None
    assert media.available("not/pinned.jpg", "tutorials", root) is None
    assert media.status("tutorials", root) == {"pinned": 1, "present": 0, "missing": 1, "different": 0}
    poster = root / tree_path
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"poster bytes")
    assert media.available(tree_path, "tutorials", root) == poster
    assert media.status("tutorials", root)["present"] == 1


def test_offline_fetch_failure_reads_as_not_downloaded(media, tmp_path, monkeypatch):
    root = _tiny_pin(tmp_path, revision="a" * 40)

    class Offline(Exception):
        pass

    def refuse(*_args, **_kwargs):
        raise Offline("no network")

    import types
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=refuse))
    tree_path = "docs/source/_extra/tutorials/production/07_mask/poster.jpg"
    assert media.available(tree_path, "tutorials", root) is None
    with pytest.raises(RuntimeError, match="could not fetch"):
        media.media_path(tree_path, "tutorials", root / "docs" / "media_revision.json")


def test_first_publish_uploads_every_file_from_the_staging_folder(media, tmp_path, monkeypatch):
    root = _tiny_pin(tmp_path)
    staging = tmp_path / "staging"
    tree_path = "docs/source/_extra/tutorials/production/07_mask/poster.jpg"
    (staging / tree_path).parent.mkdir(parents=True)
    (staging / tree_path).write_bytes(b"poster bytes")
    sent = []

    class Api:
        def create_commit(self, repo, repo_type, operations, commit_message):
            sent.extend((op.path_in_repo, op.path_or_fileobj) for op in operations)
            return type("Info", (), {"oid": "b" * 40})()

    import types
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=Api, CommitOperationAdd=lambda path_in_repo, path_or_fileobj: types.SimpleNamespace(
            path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)))
    assert media.publish("tutorials", root, source=staging) == "b" * 40
    assert sent == [("tree/" + tree_path, str(staging / tree_path))]
    assert json.loads((root / "docs" / "media_revision.json").read_text())[
        "groups"]["tutorials"]["revision"] == "b" * 40
    assert media.publish("tutorials", root, source=staging) is None


def test_tutorial_tools_write_media_to_staging_and_text_to_the_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_TUTORIALS_STAGING", str(tmp_path / "staging"))
    helper = _load("tutorial_media_guard", ROOT / "tools" / "tutorials" / "tutorial_media.py")
    zip_path = ROOT / "docs/source/_extra/tutorials/examples/New_example.zip"
    target = helper.media_target(zip_path)
    assert target == tmp_path / "staging/hf/tree/docs/source/_extra/tutorials/examples/New_example.zip"
    assert target.parent.is_dir()
    catalog = ROOT / "docs/source/_extra/tutorials/catalog/lessons_en.json"
    assert helper.media_target(catalog) == catalog
    assert helper.media_target("tools/tutorials/lessons/x.json") == ROOT / "tools/tutorials/lessons/x.json"
    target.write_bytes(b"new")
    assert helper.media_source(zip_path) == target
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        helper.media_source(ROOT / "docs/source/_extra/tutorials/examples/Never_written.zip")


def test_staged_player_says_when_the_media_is_not_downloaded(tmp_path):
    budget = _load("docs_media_budget_guard", ROOT / "tools" / "docs_media_budget.py")
    extra = tmp_path / "_extra"
    (extra / "tutorials").mkdir(parents=True)
    assert set(budget.missing_tutorial_media(extra)) == set(
        path for path in json.loads((ROOT / "docs/media_revision.json").read_text(encoding="utf-8"))
        ["groups"]["tutorials"]["files"])
    source = tmp_path / "source.html"
    source.write_text('<!doctype html><html><body class="x"><main></main></body></html>', encoding="utf-8")
    staged = tmp_path / "staged" / "tutorials" / "index.html"
    staged.parent.mkdir(parents=True)
    staged.hardlink_to(source)
    assert budget.mark_media_missing(tmp_path / "staged", 93)
    text = staged.read_text(encoding="utf-8")
    assert text.index('<body class="x">') < text.index("Tutorial media not downloaded.") < text.index("<main>")
    assert "93 lesson posters" in text
    assert "not downloaded" not in source.read_text(encoding="utf-8")
    assert not budget.mark_media_missing(tmp_path / "staged", 93)


def test_the_app_reads_no_tutorial_media(media):
    """Help opens the online library; nothing in the package names the media."""
    offenders = []
    for path in (ROOT / "spacr").rglob("*.py"):
        if "i18n_catalogs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for folder in media.TUTORIAL_MEDIA_FOLDERS:
            if folder.rstrip("/") in text:
                offenders.append(f"{path.relative_to(ROOT)}: {folder}")
    assert not offenders


def test_help_tutorial_links_work_with_the_media_absent(monkeypatch):
    """The lesson lookups read only the bundled index, never the media folders."""
    import builtins

    from spacr.qt import tutorials

    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        name = str(file).replace("\\", "/")
        assert "_extra/tutorials/production" not in name and "_extra/tutorials/examples" not in name
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(tutorials, "_INDEX", None)
    monkeypatch.setattr(tutorials, "_LESSONS", None)
    monkeypatch.setattr(tutorials, "_COMING_SOON", None)
    url = tutorials.tutorial_url("mask")
    assert url.startswith(tutorials.TUTORIALS_URL + "#lesson=")
    assert tutorials.lessons()
