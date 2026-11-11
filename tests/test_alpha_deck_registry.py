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
    # 691 (2026-10-11): MitoNet for mitochondria in TEM makes 63.
    # 372 (2026-10-25): the OPS page in Mask Generation makes 64.
    # 791 (2026-11-03): feature QC at the end of Measure makes 65.
    # 770 (2026-11-04): TVN as a batch_correction choice makes 66.
    # 752 (2026-11-05): the OPS phenotype path makes 67.
    # 803 (2026-11-06): the GPU memory budget settings make 68.
    # 2026-11-07: 769 (per-cell export), 800 (size floors from pixel size),
    # 790 (label image import) and 772 (fixed crop scaling, shuffled-label
    # control) make 72.
    # 801 and 802 (2026-11-07): several mask workers per GPU and the
    # segmentation flat field kept for a later masks-only run make 74.
    # 771 (2026-11-07): division-aligned analysis of tracked cells makes 75.
    # 864 and 867 (2026-11-08): division results in minutes and um, and the
    # movie bootstrap of the division effect sizes, and 750 (2026-11-08): the
    # public dataset fetcher, make 78.
    # 751 (2026-11-09): the comparison with a published per-gene table makes 79.
    # 865 and 866 (2026-11-09): budding-yeast size control and the
    # "divides within N frames" predictor make 81 with 751.
    # 795 (2026-11-09): the leave-one-compound-out MoA benchmark makes 82.
    # 873 (2026-11-09): the OPS decode rule (a) + S2 makes 83.
    # 876 and 877 (2026-11-09): the false-call model and the genotype
    # confidence make 85, and 878 (the per-cell comparison) 86.
    # 2026-11-09, BMC paper items: the OPS gene-profile options 883, 887,
    # 890, 891 and 902-907 make 96; the OPS phenotype run 871, 872, 874 and
    # 875 make 100; the profile path and its metrics 879-882 make 104; and
    # 898-901 and 909-911 (titration, classifier evaluation, plaque figure
    # scoring and sizes, division onsets and prediction arms, effector
    # translocation) make 111.
    # 2026-11-10, second integration: the gene graph (884), the
    # protein-knowledge blocks (885) and screen x knowledge fusion (886)
    # make 114.
    # The OPS module atlas (888), the MitoCarta and external-module
    # benchmark (908), gene-class training (889) and the channel-adaptive
    # architectures (897) make 118.
    # Screen plates from one settings file (893), the MoA evaluation with
    # the CPA classifiers (894) and crop shards, embeddings and the well
    # UMAP (896) make 121.
    # 2026-11-10, third integration: the OPS DNA flat-field map from nuclei
    # (912), shared copairs / EFAAR scoring (913), cell-state profiles and
    # the mitotic index test (914), the overlap-zone report (915), the
    # Leiden resolution sweep (916) and the E-test power curve (917) make
    # 127.
    # 2026-11-11, fourth integration: the cropped-site reader (919), the
    # spacer-prefix decode (920), mask-overlap matching to a reference
    # pipeline (922), CORUM r / STRING AUPRC / zero-TPM hit calls (923),
    # the OPS gene map source of Image UMAP (918) and tiled segmentation of
    # large cropped sites (921) make 133.
    assert deck._check_registry(ROOT / "spacr/settings.py") == 133
    # 710: the published PDF is fetched from the pinned dataset revision.
    media_spec = importlib.util.spec_from_file_location("fetch_docs_media", ROOT / "tools/fetch_docs_media.py")
    media = importlib.util.module_from_spec(media_spec)
    media_spec.loader.exec_module(media)
    pdf = PdfReader(media.media_path("docs/source/_static/deck/spacr_deck.pdf"))
    text = " ".join(unicodedata.normalize("NFKC", pdf.pages[3].extract_text()).split())
    for _, rows in deck.GROUPS:
        for _, caption in rows:
            assert unicodedata.normalize("NFKC", caption) in text
    for _, name in deck.SPECIES:
        assert name in text
    assert "129 alpha features and 7 alpha species" in text


def test_unknown_registry_entry_refuses_before_touching_assets(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text("ALPHA_FEATURES = {999999: {}}", encoding="utf-8")
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
    (tmp_path / "slides.json").write_text(json.dumps({"count": 3, "slides": slides}), encoding="utf-8")
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
    assert "129 alpha features" in " ".join(after.pages[1].extract_text().split())
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
        "count": 1, "slides": [{"title": deck.TITLE, "image": "a.jpg", "thumb": "b.jpg"}]}), encoding="utf-8")
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"published image")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def fail(*args):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(deck, "_draw_pdf", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        deck.refresh(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
