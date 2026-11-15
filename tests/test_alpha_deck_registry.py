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
    # 2026-11-13, landing train: nightly's registry had grown to 136 with
    # per-tile affine alignment (930); 930 leaving alpha as the new default
    # and the nuclear-median flat-field (933) joining keep it at 136.
    # 2026-11-13, second landing train: automatic genotyping thresholds
    # (939), per-feature aggregation (937), Align & Stitch on the OPS stitch
    # core (936), the OPS placement seed file (940) and the OPS measurement
    # features in Measure (934) make 141.
    # 2026-11-14, third landing train: the slow measurement families off by
    # default (941) make 142.
    # 2026-11-14, fourth landing train: the large-site mask step under a VRAM
    # cap (945) makes 143.
    # 2026-11-14, fifth landing train: the joint decode auto-tuner (942) and
    # phenotype folders with two acquisitions (947) make 145.
    assert deck._check_registry(ROOT / "spacr/settings.py") == 145
    # 710: the published PDF is fetched from the pinned dataset revision.
    media_spec = importlib.util.spec_from_file_location("fetch_docs_media", ROOT / "tools/fetch_docs_media.py")
    media = importlib.util.module_from_spec(media_spec)
    media_spec.loader.exec_module(media)
    pdf = PdfReader(media.media_path("docs/source/_static/deck/spacr_deck.pdf"))
    # 2026-11-15: the alpha list fills pages 4 and 5 (one page of 145 rows was
    # too small to read). Every registry caption is on exactly one of them.
    pages = [" ".join(unicodedata.normalize("NFKC", pdf.pages[i].extract_text()).split())
             for i in (3, 4)]
    for page, title in zip(pages, deck.TITLES):
        assert title in page
        for _, name in deck.SPECIES:
            assert name in page
        assert "145 alpha features and 7 alpha species" in page
    misplaced = []
    for _, rows in deck.GROUPS:
        for _, caption in rows:
            found = [page.count(unicodedata.normalize("NFKC", caption)) for page in pages]
            if sum(found) != 1:
                misplaced.append((caption, found))
    assert misplaced == [], "each alpha caption appears once, on page 4 or page 5"
    manifest = json.loads((ROOT / "docs/source/_static/deck/slides.json").read_text(encoding="utf-8"))
    assert [slide["title"] for slide in manifest["slides"][3:5]] == list(deck.TITLES)
    assert manifest["count"] == len(pdf.pages) == len(manifest["slides"])


def test_the_two_alpha_pages_share_one_geometry():
    first, second = deck._pages()
    rows = [sum(len(r) for _, r in page) for page in (first, second)]
    assert abs(rows[0] - rows[1]) <= 1
    assert sum(rows) == sum(len(r) for _, r in deck.GROUPS)
    for page in (first, second):
        for column in deck._columns(page):
            for category, _ in column:
                assert category.count("(continued)") <= 1


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


def _fake_deck(folder, widths, titles):
    from PIL import Image
    from pypdf import PdfWriter

    writer = PdfWriter()
    for width in widths:
        writer.add_blank_page(width=width, height=540)
    writer.add_metadata({"/Title": "Keep this title"})
    writer.write(folder / "spacr_deck.pdf")
    slides = [{"title": title, "image": f"slides/slide_{i:02d}.jpg",
               "thumb": f"thumbs/slide_{i:02d}.jpg", "animations": []}
              for i, title in enumerate(titles, 1)]
    (folder / "slides.json").write_text(json.dumps({"count": len(titles), "slides": slides}),
                                        encoding="utf-8")
    for kind in ("slides", "thumbs"):
        (folder / kind).mkdir()
    for i, slide in enumerate(slides):
        for field in ("image", "thumb"):
            Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(folder / slide[field])
    return slides


def _refresh_cli(folder):
    subprocess.run([sys.executable, str(ROOT / "tools/refresh_alpha_deck.py"),
                    "--out", str(folder)], check=True, timeout=60,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})


def test_refresh_splits_the_alpha_slide_and_moves_later_slides_up(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    if not shutil.which("pdftoppm"):
        pytest.skip("poppler is a deck publication dependency")
    pytest.importorskip("PySide6")
    _fake_deck(tmp_path, (100, 960, 200), ("before", deck.TITLE, "after"))
    (tmp_path / "slides" / "slide_03.jpg").write_bytes(b"later slide")
    _refresh_cli(tmp_path)
    after = PdfReader(tmp_path / "spacr_deck.pdf")
    assert len(after.pages) == 4
    assert [float(page.mediabox.width) for page in after.pages] == [100, 960, 960, 200]
    assert after.metadata.title == "Keep this title"
    first, second = (" ".join(after.pages[i].extract_text().split()) for i in (1, 2))
    assert deck.TITLES[0] in first and deck.TITLES[1] in second
    assert "145 alpha features" in first and "145 alpha features" in second
    assert after.pages[0].get_contents() is None
    assert after.pages[3].get_contents() is None
    manifest = json.loads((tmp_path / "slides.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 4
    assert [s["title"] for s in manifest["slides"]] == ["before", *deck.TITLES, "after"]
    assert [s["image"] for s in manifest["slides"]] == [f"slides/slide_{i:02d}.jpg" for i in range(1, 5)]
    assert (tmp_path / "slides" / "slide_04.jpg").read_bytes() == b"later slide"
    for number in (2, 3):
        assert Image.open(tmp_path / f"slides/slide_{number:02d}.jpg").size == (3200, 1800)
        assert Image.open(tmp_path / f"thumbs/slide_{number:02d}.jpg").size == (320, 180)
    # A second refresh replaces the two pages and keeps every other one.
    kept = (tmp_path / "slides" / "slide_04.jpg").read_bytes()
    _refresh_cli(tmp_path)
    again = PdfReader(tmp_path / "spacr_deck.pdf")
    assert [float(page.mediabox.width) for page in again.pages] == [100, 960, 960, 200]
    assert json.loads((tmp_path / "slides.json").read_text(encoding="utf-8")) == manifest
    assert (tmp_path / "slides" / "slide_04.jpg").read_bytes() == kept


def test_render_failure_preserves_the_published_deck(tmp_path, monkeypatch):
    _fake_deck(tmp_path, (960, 200), (deck.TITLE, "after"))

    def snapshot():
        return {p.relative_to(tmp_path).as_posix(): p.read_bytes()
                for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()

    def fail(*args):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(deck, "_draw_pdf", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        deck.refresh(tmp_path)
    assert snapshot() == before
