"""Guards for the thirty Home-screen candidates under
``spacr/resources/home/versions``.

These are *review artefacts*, not shipped UI — nothing under
``spacr/qt/`` imports them. What the tests protect is that the
artefacts stay honest:

* every categorisation a variant proposes covers all 29 real apps
  exactly once, so a reviewer is never shown a screen that quietly
  drops an app;
* every rendered PNG exists, is exactly 1440x900, and is not a
  solid-colour "the widget never laid out" render;
* the generator still builds a real page out of real widgets with no
  elided or clipped text — text clipping is the defect the home-screen
  rework was raised to fix.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VERSIONS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions")
GENERATORS = os.path.join(VERSIONS, "_generators")

CANVAS = (1440, 900)
N_VARIANTS = 30


def _load(name: str, module_name: str):
    """Import one generator module under an explicit module name."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_common():
    """The generator's ``common`` module, loaded under a private name.

    ``common.py`` has no sibling imports, so it can be loaded without
    claiming the generic top-level names the other two need.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    return _load("common", "_spacr_home_variants_common")


# ---------------------------------------------------------------------------
# The categorisations
# ---------------------------------------------------------------------------

def test_every_categorisation_covers_every_app(gen_common):
    """No proposed grouping may drop, duplicate or invent an app key."""
    gen_common.bootstrap()
    for name in ("CATS_BROAD3", "CATS_STAGE5", "CATS_NARROW8",
                 "CATS_QUESTIONS", "CATS_INTENT4"):
        gen_common.check_coverage(getattr(gen_common, name))
    gen_common.check_coverage(gen_common.cats_current())


def test_orderings_are_permutations_of_the_real_registry(gen_common):
    """Frequency / alphabetical / pinned-first are re-orderings, not edits."""
    gen_common.bootstrap()
    keys = set(gen_common.all_keys())
    assert len(keys) == 29
    for order in (gen_common.by_frequency(), gen_common.alphabetical(),
                  gen_common.pinned_first()):
        assert len(order) == len(keys)
        assert set(order) == keys


def test_use_counts_cover_every_app(gen_common):
    """The frequency-ordered variants need a count for every app."""
    assert set(gen_common.USE_COUNTS) == set(gen_common.all_keys())


# ---------------------------------------------------------------------------
# The rendered artefacts
# ---------------------------------------------------------------------------

def _variant_dirs():
    if not os.path.isdir(VERSIONS):
        return []
    return sorted(
        os.path.join(VERSIONS, n) for n in os.listdir(VERSIONS)
        if n.startswith("v") and os.path.isdir(os.path.join(VERSIONS, n)))


def test_thirty_variant_folders_exist():
    dirs = _variant_dirs()
    if not dirs:
        pytest.skip("variants have not been rendered")
    assert len(dirs) == N_VARIANTS
    numbers = sorted(int(os.path.basename(d)[1:3]) for d in dirs)
    assert numbers == list(range(1, N_VARIANTS + 1))


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_png_is_the_right_size_and_not_blank(theme):
    """A solid-colour render means the widget never laid out."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    dirs = _variant_dirs()
    if not dirs:
        pytest.skip("variants have not been rendered")
    for folder in dirs:
        path = os.path.join(folder, f"{theme}.png")
        assert os.path.isfile(path), f"missing {path}"
        with Image.open(path) as im:
            assert im.size == CANVAS, f"{path} is {im.size}, want {CANVAS}"
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        assert float(arr.std()) > 3.0, f"{path} is near-uniform"
        assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 64, \
            f"{path} has almost no distinct colours"


def test_contact_sheet_and_markdown_exist():
    if not _variant_dirs():
        pytest.skip("variants have not been rendered")
    assert os.path.isfile(os.path.join(VERSIONS, "_sheet.png"))
    md = os.path.join(VERSIONS, "VARIANTS.md")
    assert os.path.isfile(md)
    text = open(md, encoding="utf-8").read()
    for n in range(1, N_VARIANTS + 1):
        assert f"### {n:02d} ·" in text, f"variant {n} missing from VARIANTS.md"


# ---------------------------------------------------------------------------
# The generator still runs
# ---------------------------------------------------------------------------

def test_a_variant_builds_with_no_clipped_or_elided_text():
    """Build two variants for real and audit the laid-out widget tree.

    ``parts`` and ``variants`` import each other by plain name, so they
    have to occupy those entries in ``sys.modules`` while they load;
    the originals are restored afterwards.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    saved = {name: sys.modules.get(name)
             for name in ("common", "parts", "variants")}
    try:
        _load("common", "common")
        _load("parts", "parts")
        variants = _load("variants", "variants")
        render = _load("render", "_spacr_home_variants_render")

        app = sys.modules["common"].bootstrap()
        ctx = sys.modules["common"].Ctx(app, "dark")
        ctx.apply_theme()
        assert len(variants.VARIANTS) == N_VARIANTS
        slugs = [v["slug"] for v in variants.VARIANTS]
        assert len(set(slugs)) == N_VARIANTS

        # 13 (dense two-column) and 23 (illustrated tiles) between them
        # exercise every text widget the set uses.
        for number in (13, 23):
            spec = variants.VARIANTS[number - 1]
            page = spec["build"](ctx)
            page.resize(*CANVAS)
            page.show()
            for _ in range(4):
                app.processEvents()
            report = render.audit(page)
            page.hide()
            page.setParent(None)
            page.deleteLater()
            app.processEvents()
            assert not report["elided"], report["elided"]
            assert not report["clipped"], report["clipped"]
            assert not report["scrollbars"], report["scrollbars"]
            assert not report["overflow"], report["overflow"]
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
