"""The info deck's PDF and animations come from a pinned dataset revision (710).

They left git in item 710's phase 3. The docs build fetches them at the
revision docs/media_revision.json names, checks every hash, and a release
bump that restamped the title picture in git still reaches the PDF.
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = "docs/source/_static/deck"


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pin_names_every_hosted_file_and_git_holds_none_of_them():
    pin = json.loads((ROOT / "docs/media_revision.json").read_text(encoding="utf-8"))
    deck = pin["groups"]["deck"]
    assert deck["repo"] == "einarolafsson/spacr-tutorials"
    assert deck["repo_type"] == "dataset"
    assert len(deck["revision"]) == 40
    assert set(deck["files"]) == {f"{DECK}/spacr_deck.pdf",
                                  f"{DECK}/anim/slide_33_image-33-2.gif",
                                  f"{DECK}/anim/slide_33_image-33-3.gif"}
    for tree_path, record in deck["files"].items():
        assert record["path_in_repo"] == "deck/" + tree_path[len(DECK) + 1:]
        assert len(record["sha256"]) == 64 and record["size"] > 0
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", *deck["files"]],
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked == []
    manifest = json.loads((ROOT / DECK / "slides.json").read_text(encoding="utf-8"))
    assert all((ROOT / DECK / slide["image"]).is_file() for slide in manifest["slides"])


def test_the_docs_build_fetches_and_checks_before_sphinx():
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    fetch = workflow.index("tools/fetch_docs_media.py fetch")
    assert fetch < workflow.index("tools/fetch_docs_media.py check") < workflow.index("sphinx-build")


def _fake_root(tmp_path, version, shown):
    from PIL import Image
    from pypdf import PdfWriter

    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/build_readme_deck.py", root / "tools/build_readme_deck.py")
    deck = root / DECK
    (deck / "slides").mkdir(parents=True)
    Image.new("RGB", (320, 180), "red").save(deck / "slides/slide_01.jpg")
    (deck / "slides.json").write_text(json.dumps({
        "version": shown, "slides": [{"image": "slides/slide_01.jpg", "thumb": "thumbs/slide_01.jpg"}]}))
    hosted = tmp_path / "hosted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(320, 180)
    writer.add_blank_page(320, 180)
    with open(hosted, "wb") as handle:
        writer.write(handle)
    media = _tool("fetch_docs_media")
    (root / "docs/media_revision.json").write_text(json.dumps({"groups": {"deck": {
        "repo": "owner/data", "repo_type": "dataset", "revision": "0" * 40, "version": version,
        "files": {f"{DECK}/spacr_deck.pdf": {"path_in_repo": "deck/spacr_deck.pdf",
                                             "sha256": media.sha256(hosted),
                                             "size": hosted.stat().st_size}}}}}))
    return root, hosted, media


def test_fetch_copies_the_pinned_bytes_when_the_version_matches(tmp_path, monkeypatch):
    root, hosted, media = _fake_root(tmp_path, "1.0", "1.0")
    monkeypatch.setattr(media, "media_path", lambda *args, **kwargs: hosted)
    written = media.fetch("deck", root)
    assert written[f"{DECK}/spacr_deck.pdf"].read_bytes() == hosted.read_bytes()
    assert list(media.check("deck", root)) == []


def test_fetch_restamps_the_title_page_after_a_version_bump(tmp_path, monkeypatch):
    from pypdf import PdfReader

    root, hosted, media = _fake_root(tmp_path, "1.0", "1.1")
    monkeypatch.setattr(media, "media_path", lambda *args, **kwargs: hosted)
    media.fetch("deck", root)
    pdf = root / DECK / "spacr_deck.pdf"
    assert pdf.read_bytes() != hosted.read_bytes()
    after = PdfReader(pdf)
    assert len(after.pages) == 2 and after.pages[0].images
    before = pdf.read_bytes()
    assert list(media.check("deck", root)) == []
    assert pdf.read_bytes() == before


def test_projected_pdf_bytes_are_deterministic_despite_pillow_timestamps(tmp_path):
    import os
    import sys

    root, hosted, _media = _fake_root(tmp_path, "1.0", "1.1")
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    shutil.copyfile(hosted, first)
    shutil.copyfile(hosted, second)
    script = """
import importlib.util
import sys
import time
from pathlib import Path
import PIL.PdfImagePlugin
clock = time.gmtime
PIL.PdfImagePlugin.time.gmtime = lambda *args: clock(int(sys.argv[4]))
spec = importlib.util.spec_from_file_location('projection_media', sys.argv[1])
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
assert media._restamp_pdf(Path(sys.argv[2]), Path(sys.argv[3]))
"""
    for target, timestamp, seed in ((first, "0", "1"), (second, "1700000000", "2")):
        result = subprocess.run([
            sys.executable, "-c", script, str(ROOT / "tools/fetch_docs_media.py"),
            str(root), str(target), timestamp], capture_output=True, text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed), timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("damage", ["raw", "title", "missing", "extra",
                                    "remaining", "bytes", "source-title"])
def test_projected_check_rejects_damage_and_stale_pages_without_repairing_them(
        tmp_path, monkeypatch, damage):
    import io
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    root, hosted, media = _fake_root(tmp_path, "1.0", "1.1")
    original = hosted.read_bytes()
    pin = root / "docs/media_revision.json"
    original_pin = pin.read_bytes()
    monkeypatch.setattr(media, "media_path", lambda *args, **kwargs: hosted)
    media.fetch("deck", root)
    pdf = root / DECK / "spacr_deck.pdf"
    if damage == "raw":
        pdf.write_bytes(original)
    elif damage == "title":
        assert _tool("build_readme_deck").replace_first_page(
            pdf, Image.new("RGB", (320, 180), "blue"))
    elif damage == "bytes":
        pdf.write_bytes(pdf.read_bytes() + b"\nunexpected trailing content")
    elif damage == "source-title":
        Image.new("RGB", (320, 180), "blue").save(root / DECK / "slides/slide_01.jpg")
    else:
        reader = PdfReader(io.BytesIO(pdf.read_bytes()))
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        if damage == "remaining":
            writer.add_blank_page(319, 180)
        elif damage == "extra":
            writer.add_page(reader.pages[1])
            writer.add_blank_page(320, 180)
        with pdf.open("wb") as handle:
            writer.write(handle)
    damaged = pdf.read_bytes()
    assert list(media.check("deck", root)) == [
        f"{DECK}/spacr_deck.pdf: differs from the verified 1.1 deck projection"]
    assert pdf.read_bytes() == damaged
    assert hosted.read_bytes() == original
    assert pin.read_bytes() == original_pin


def test_projected_check_still_verifies_the_raw_cached_source(tmp_path, monkeypatch):
    import huggingface_hub

    root, hosted, media = _fake_root(tmp_path, "1.0", "1.1")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *args, **kwargs: str(hosted))
    media.fetch("deck", root)
    hosted.write_bytes(hosted.read_bytes() + b"tampered raw cache")
    with pytest.raises(RuntimeError, match="does not match the pin"):
        list(media.check("deck", root))


def test_projection_verification_cannot_fall_back_when_the_renderer_is_unavailable(
        tmp_path, monkeypatch):
    root, hosted, media = _fake_root(tmp_path, "1.0", "1.1")
    monkeypatch.setattr(media, "media_path", lambda *args, **kwargs: hosted)
    media.fetch("deck", root)
    monkeypatch.setattr(media, "_restamp_pdf", lambda *args: False)
    assert list(media.check("deck", root)) == [
        f"{DECK}/spacr_deck.pdf: could not verify the release-title projection"]


def test_other_media_groups_keep_exact_raw_pin_semantics(tmp_path, monkeypatch):
    root, hosted, media = _fake_root(tmp_path, "1.0", "1.1")
    pin = root / "docs/media_revision.json"
    data = json.loads(pin.read_text())
    data["groups"]["other"] = data["groups"].pop("deck")
    pin.write_text(json.dumps(data))
    monkeypatch.setattr(media, "media_path", lambda *args, **kwargs: hosted)
    media.fetch("other", root)
    pdf = root / DECK / "spacr_deck.pdf"
    assert pdf.read_bytes() == hosted.read_bytes()
    assert list(media.check("other", root)) == []
    assert media._restamp_pdf(root, pdf)
    assert list(media.check("other", root)) == [f"{DECK}/spacr_deck.pdf: differs from the pin"]


def test_check_names_a_missing_file(tmp_path):
    root, _hosted, media = _fake_root(tmp_path, "1.0", "1.0")
    assert list(media.check("deck", root)) == [f"{DECK}/spacr_deck.pdf: not fetched"]


def test_an_unfetchable_pin_fails_rather_than_skips(tmp_path, monkeypatch):
    import huggingface_hub

    root, _hosted, media = _fake_root(tmp_path, "1.0", "1.0")

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="could not fetch deck/spacr_deck.pdf"):
        media.media_path(f"{DECK}/spacr_deck.pdf", pin=root / "docs/media_revision.json")


def test_the_stamp_knows_which_deck_files_are_pinned():
    deck = _tool("build_readme_deck")
    assert deck.pinned_media(ROOT / DECK / "spacr_deck.pdf")
    assert not deck.pinned_media(ROOT / DECK / "slides/slide_01.jpg")
