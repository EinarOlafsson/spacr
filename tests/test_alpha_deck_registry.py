"""Publishing the alpha slide keeps other pages and refuses registry drift."""
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
    # 2026-11-15: OPS runs on chosen sequencing sites (951) make 146.
    # 2026-11-15: split and merge counts in the large-site mask agreement
    # (956) make 147.
    # 2026-11-15, landing train 6 (item 946): the rule (a) + S2 decode (873)
    # and automatic genotyping thresholds (939) leave alpha as the new OPS
    # decode defaults, which makes 145 (the published deck shows the old
    # count until the maintainer publishes it).
    # 2026-11-15: published comparators keep their control rows (953) make 146.
    # 2026-11-15: the barcode read mode chosen per well (952) makes 147.
    # 2026-11-16: near-miss decoys and the shuffle calibration of the decoy
    # FDR scale (949) make 148.
    # 2026-11-16: the matched-FDR sweep of prefix decodes (955) makes 149.
    # 2026-11-16: Make Masks point mode (957) makes 150.
    # 2026-11-16: the gene-pairs consensus mode (948) makes 151.
    # 2026-11-16: Annotate's classifier choice and class thresholds (963)
    # make 152.
    # 2026-11-17: OPS objects windows resume (958) make 153.
    # 2026-11-17: the OPS merge routes with an automatic fallback (962)
    # make 154.
    # 2026-11-17: R2 nuclei on the large-site path (965) make 155.
    # 2026-11-17: the sweep rows' control-row background (959) makes 156.
    # 2026-11-17: Voyager3 cluster annotation (973) makes 157.
    # 2026-11-17: the OPS Live preview, one tab per step (971), makes 158.
    # 2026-11-17: the spacer-prefix minority-read rule (967) makes 159.
    # 2026-11-18: pathogen and organelle objects in the OPS phenotype path
    # (970) make 160.
    # 2026-11-18: the OPS parameter grid search on tiles (960) makes 161.
    # 2026-11-19: the OPS power simulation (968) makes 162.
    # 2026-11-19: distance-aware read weights and the neighbour-bleed
    # discount in the guide call (988) make 163.
    # 2026-11-19: in situ sequencing modes of the prefix decode (961) make 164.
    count = deck._check_registry(ROOT / "spacr/settings.py")
    assert count == 164
    # 710: the published PDF is fetched from the pinned dataset revision.
    media_spec = importlib.util.spec_from_file_location("fetch_docs_media", ROOT / "tools/fetch_docs_media.py")
    media = importlib.util.module_from_spec(media_spec)
    media_spec.loader.exec_module(media)
    pdf = PdfReader(media.media_path("docs/source/_static/deck/spacr_deck.pdf"))
    manifest = json.loads((ROOT / "docs/source/_static/deck/slides.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(pdf.pages) == len(manifest["slides"])
    titles = [slide["title"] for slide in manifest["slides"]]
    # 2026-11-15: the alpha items are grouped into lines on one alpha slide.
    assert titles.count(deck.PUBLISHED_TITLE) == 1
    assert not any(title.startswith("Alpha features (") for title in titles)
    index = titles.index(deck.PUBLISHED_TITLE)
    page = " ".join(unicodedata.normalize("NFKC", pdf.pages[index].extract_text()).split())
    assert deck.PUBLISHED_TITLE in page
    for _, name in deck.SPECIES:
        assert name in page
    assert unicodedata.normalize("NFKC", deck._footer(count)).split("→")[0].strip() in page
    missing = [text for _, lines in deck.GROUPS for text, _ in lines
               if " ".join(unicodedata.normalize("NFKC", text).split()) not in page]
    assert missing == [], "every alpha line is on the published slide"


def test_every_registry_item_is_on_exactly_one_line():
    import ast

    registered = set()
    for node in ast.parse((ROOT / "spacr/settings.py").read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "ALPHA_FEATURES"):
            registered = {ast.literal_eval(key) for key in node.value.keys}
    listed = deck._listed_items()
    assert sorted(n for n in set(listed) if listed.count(n) > 1) == [], "duplicated items"
    assert sorted(registered - set(listed)) == [], "registry items missing from the slide"
    assert sorted(set(listed) - registered) == [], "slide items not in the registry"
    texts = [text for _, lines in deck.GROUPS for text, _ in lines]
    assert len(texts) == len(set(texts))
    assert all(numbers for _, lines in deck.GROUPS for _, numbers in lines)


def test_the_columns_hold_whole_categories_and_balance():
    left, right = deck._columns()
    assert left + right == deck.GROUPS
    sizes = [sum(len(lines) for _, lines in column) for column in (left, right)]
    largest = max(len(lines) for _, lines in deck.GROUPS)
    assert abs(sizes[0] - sizes[1]) <= largest
    # One slide stays readable: captions at 27 px need rows of at least 31 px.
    assert (1545 - 375) // max(sizes) >= 31


def test_duplicated_or_unknown_items_refuse(tmp_path, monkeypatch):
    source = tmp_path / "settings.py"
    numbers = sorted(set(deck._listed_items()))
    source.write_text(f"ALPHA_FEATURES = {{{', '.join(f'{n}: {{}}' for n in numbers)}}}",
                      encoding="utf-8")
    assert deck._check_registry(source) == len(numbers)
    first_category, first_lines = deck.GROUPS[0]
    doubled = ((first_category, first_lines + (("again", (numbers[0],)),)),) + deck.GROUPS[1:]
    monkeypatch.setattr(deck, "GROUPS", doubled)
    with pytest.raises(ValueError, match=rf"duplicated \[{numbers[0]}\]"):
        deck._check_registry(source)


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


def test_refresh_replaces_the_alpha_slide_and_is_idempotent(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    if not shutil.which("pdftoppm"):
        pytest.skip("poppler is a deck publication dependency")
    pytest.importorskip("PySide6")
    _fake_deck(tmp_path, (100, 960, 200), ("before", deck.TITLE, "after"))
    (tmp_path / "slides" / "slide_03.jpg").write_bytes(b"later slide")
    _refresh_cli(tmp_path)
    after = PdfReader(tmp_path / "spacr_deck.pdf")
    assert [float(page.mediabox.width) for page in after.pages] == [100, 960, 200]
    assert after.metadata.title == "Keep this title"
    text = " ".join(after.pages[1].extract_text().split())
    count = deck._check_registry(ROOT / "spacr/settings.py")
    assert deck.PUBLISHED_TITLE in text and f"{count} alpha features" in text
    assert after.pages[0].get_contents() is None
    assert after.pages[2].get_contents() is None
    manifest = json.loads((tmp_path / "slides.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 3
    assert [s["title"] for s in manifest["slides"]] == ["before", deck.PUBLISHED_TITLE, "after"]
    assert [s["image"] for s in manifest["slides"]] == [f"slides/slide_{i:02d}.jpg" for i in range(1, 4)]
    assert (tmp_path / "slides" / "slide_03.jpg").read_bytes() == b"later slide"
    assert Image.open(tmp_path / "slides/slide_02.jpg").size == (3200, 1800)
    assert Image.open(tmp_path / "thumbs/slide_02.jpg").size == (320, 180)
    # A second refresh replaces the alpha page in place and keeps every other one.
    _refresh_cli(tmp_path)
    again = PdfReader(tmp_path / "spacr_deck.pdf")
    assert [float(page.mediabox.width) for page in again.pages] == [100, 960, 200]
    assert json.loads((tmp_path / "slides.json").read_text(encoding="utf-8")) == manifest
    assert (tmp_path / "slides" / "slide_03.jpg").read_bytes() == b"later slide"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["slides", "slides.json", "spacr_deck.pdf", "thumbs"]


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
