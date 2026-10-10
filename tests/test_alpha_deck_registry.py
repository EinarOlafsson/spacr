"""Publishing the alpha page keeps other pages and refuses registry drift."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("refresh_alpha_deck", ROOT / "tools/refresh_alpha_deck.py")
deck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deck)


def test_the_published_alpha_page_covers_the_current_registry():
    from pypdf import PdfReader

    # 634, 2026-10-03: with 633, the five alpha organism pages make 60.
    # 470 (registered in ALPHA_FEATURES, cross-channel label judging) makes 61.
    # 2026-10-03: 634 moved to ALPHA_SPECIES and is drawn as the seven-species
    # strip, so the table lists 60 features beside seven species.
    # 404 (2026-10-04): DINOCell stays an alpha backend, which makes 61.
    # 685 (2026-10-10): the Make Masks vvvv export makes 62.
    assert deck._check_registry(ROOT / "spacr/settings.py") == 62
    pdf = PdfReader(ROOT / "docs/source/_static/deck/spacr_deck.pdf")
    text = " ".join(unicodedata.normalize("NFKC", pdf.pages[3].extract_text()).split())
    for _, rows in deck.GROUPS:
        for _, caption in rows:
            assert unicodedata.normalize("NFKC", caption) in text
    for _, name in deck.SPECIES:
        assert name in text
    assert "62 alpha features and 7 alpha species" in text


def test_unknown_registry_entry_refuses_before_touching_assets(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text("ALPHA_FEATURES = {999999: {}}")
    target = tmp_path / "deck"
    target.mkdir()
    sentinel = target / "spacr_deck.pdf"
    sentinel.write_bytes(b"existing publication")
    with pytest.raises(ValueError, match="999999"):
        deck.refresh(target, source)
    assert sentinel.read_bytes() == b"existing publication"
    assert list(target.iterdir()) == [sentinel]


def test_refresh_preserves_other_pages_and_renders_searchable_text(tmp_path):
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    if not shutil.which("pdftoppm"):
        pytest.skip("poppler is a deck publication dependency")
    pytest.importorskip("PySide6")
    writer = PdfWriter()
    for width in (100, 960, 200):
        writer.add_blank_page(width=width, height=540)
    writer.add_metadata({"/Title": "Keep this title"})
    writer.write(tmp_path / "spacr_deck.pdf")
    slides = [{"title": title, "image": f"{i}.jpg", "thumb": f"{i}-thumb.jpg"}
              for i, title in enumerate(("before", deck.TITLE, "after"))]
    (tmp_path / "slides.json").write_text(json.dumps({"count": 3, "slides": slides}))
    for slide in slides:
        for field in ("image", "thumb"):
            Image.new("RGB", (8, 8), "red").save(tmp_path / slide[field])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    subprocess.run([sys.executable, str(ROOT / "tools/refresh_alpha_deck.py"),
                    "--out", str(tmp_path)], check=True, timeout=60,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    after = PdfReader(tmp_path / "spacr_deck.pdf")
    assert len(after.pages) == 3
    assert [float(page.mediabox.width) for page in after.pages] == [100, 960, 200]
    assert after.metadata.title == "Keep this title"
    assert "62 alpha features" in " ".join(after.pages[1].extract_text().split())
    assert after.pages[0].get_contents() is None
    assert after.pages[2].get_contents() is None
    assert Image.open(tmp_path / "1.jpg").size == (3200, 1800)
    assert Image.open(tmp_path / "1-thumb.jpg").size == (320, 180)
    for name, content in before.items():
        if name not in {"spacr_deck.pdf", "1.jpg", "1-thumb.jpg"}:
            assert (tmp_path / name).read_bytes() == content


def test_render_failure_preserves_the_published_deck(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=960, height=540)
    writer.write(tmp_path / "spacr_deck.pdf")
    (tmp_path / "slides.json").write_text(json.dumps({
        "count": 1, "slides": [{"title": deck.TITLE, "image": "a.jpg", "thumb": "b.jpg"}]}))
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"published image")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def fail(*args):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(deck, "_draw_pdf", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        deck.refresh(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
