"""The API homepage shows the structure the rest of spaCR shows.

Instruction 374. The homepage and the README already SHARED the tile grid
-- one generated block, included by both -- and what the homepage was
missing was everything around it: nothing said which six modules are the
pipeline, nothing said that Timelapse opens from Mask, and the section that
did have prose listed 215 Python modules alphabetically above a sentence
admitting nobody should read it that way.

Four claims are pinned here, and each one is a thing that was measured
wrong on 2026-09-02:

* the grid is grouped, and by Home's OWN sections rather than a copy of
  them, so a restructure of Home fails this file instead of quietly
  publishing a stale band;
* every folded module is named on the homepage under the host that opens
  it;
* the grid is still one block shared with the README -- one builder, one
  order, one set of destinations, two asset roots;
* no tile's destination moved. Instruction 366 part 3 gave six tiles that
  share three module pages an anchor apiece, and a regrouping is exactly
  the kind of change that would drop them.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
AUTOAPI_INDEX = ROOT / "docs" / "source" / "_autoapi_templates" / "index.rst"
DOCS_INDEX = ROOT / "docs" / "source" / "index.rst"
DOCS_GRID = ROOT / "docs" / "source" / "_generated" / "workflow_grid.rst"
DOCS_FOLDS = ROOT / "docs" / "source" / "_generated" / "folded_modules.rst"

#: The six destinations instruction 366 part 3 gave an anchor to, pinned as
#: literals ON PURPOSE.
#:
#: Every other check in this repository asks whether the grid agrees with
#: `api_docs_url`, which is the right question for drift and the wrong one
#: for this: a change to the resolver moves the tile and the expectation
#: together, and the test still passes. These six are the ones that would
#: silently collapse back onto three shared module pages, which is the
#: state 366 part 3 was filed to end, so they are written out.
ANCHORED_DESTINATIONS = {
    "mask": ("https://einarolafsson.github.io/spacr/api/spacr/core/"
             "index.html#spacr.core.preprocess_generate_masks"),
    "umap": ("https://einarolafsson.github.io/spacr/api/spacr/core/"
             "index.html#spacr.core.generate_image_umap"),
    "analyze_plaques": (
        "https://einarolafsson.github.io/spacr/api/spacr/submodules/"
        "index.html#spacr.submodules.analyze_plaques"),
    "recruitment": (
        "https://einarolafsson.github.io/spacr/api/spacr/submodules/"
        "index.html#spacr.submodules.analyze_recruitment"),
    "invasion": (
        "https://einarolafsson.github.io/spacr/api/spacr/submodules/"
        "index.html#spacr.submodules.analyze_invasion"),
    "replication": (
        "https://einarolafsson.github.io/spacr/api/spacr/submodules/"
        "index.html#spacr.submodules.analyze_replication"),
}


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "spacr_readme_visuals",
        ROOT / "packaging" / "generate_readme_visuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _readme_block(generator) -> str:
    text = _read(README)
    block = text.partition(generator.WORKFLOW_BEGIN)[2]
    return block.partition(generator.WORKFLOW_END)[0]


def _bands(text: str, underline: str, prefix: str) -> "list[tuple[str, list]]":
    """``[(heading, [tile keys under it])]`` as one rendered grid reads.

    Parsed out of the markup rather than asked of the generator, because
    what is being checked is what the committed page SAYS. A generator
    that groups correctly and a page written before it did are exactly
    the disagreement this file exists to catch.
    """
    pattern = re.compile(
        rf"(?m)^(?P<title>[^\n]+)\n{re.escape(underline)}{{3,}}$")
    bands = []
    for match in pattern.finditer(text):
        following = pattern.search(text, match.end())
        body = text[match.end():following.start() if following else len(text)]
        # ROWS ONLY. The substitution definitions that follow the last band
        # name every tile again, and counting those would put the whole
        # grid under whichever heading came last.
        rows = "\n".join(line for line in body.splitlines()
                         if line.startswith("| |"))
        bands.append((match.group("title"),
                      re.findall(rf"\|{prefix}_([a-z0-9_]+)\|", rows)))
    return bands


def _targets(text: str, prefix: str) -> "dict[str, str]":
    """``{tile key: destination}`` from a grid's substitution definitions."""
    pattern = re.compile(
        rf"(?m)^\.\. \|{prefix}_(?P<key>[a-z0-9_]+)\| image::[^\n]*\n"
        rf"(?:   :[^\n]*\n)*?   :target: (?P<target>[^\n]+)$")
    return {m.group("key"): m.group("target").strip()
            for m in pattern.finditer(text)}


def test_the_homepage_groups_its_tiles_by_the_sections_home_has(generator):
    """Same sections, same order, same members -- read from Home.

    `_grouped_apps` already raised when the documented order and the
    registry disagreed; what it never did was EMIT the grouping. This
    asserts the emitted page against `spacr.qt.app` directly, so a section
    renamed, reordered or emptied in the GUI fails here rather than
    leaving the API homepage advertising a category the product dropped.
    """
    from spacr.qt.app import SECTION_ORDER, SECTION_TILE_ORDER, tiled_apps

    import spacr.qt
    spacr.qt.register_self_registering_modules()
    tiled = {key for key, _l, _d, _s in tiled_apps()}
    expected = [
        (section, [key for key in SECTION_TILE_ORDER[section]
                   if key in tiled])
        for section in SECTION_ORDER
    ]
    expected = [(section, keys) for section, keys in expected if keys]

    bands = _bands(_read(DOCS_GRID), generator.SECTION_HEADING_CHAR,
                   "DocModule")
    assert bands == expected, (
        "the API homepage's grid and the Home registry disagree about the "
        "sections or their members. Re-run "
        "packaging/generate_readme_visuals.py")
    # Every tile is under a heading: a band that emitted no heading would
    # pass the comparison above by simply not appearing.
    assert sum(len(keys) for _s, keys in bands) == len(tiled)


def test_the_grid_is_still_one_block_shared_with_the_readme(generator):
    """One builder, one order, one set of destinations, two asset roots.

    The README and the homepage are allowed to differ in exactly two
    things -- the substitution namespace and where the PNGs are served
    from -- because they are rendered by different toolchains. Everything
    else is the shared block, and this fails if the two ever start being
    written separately again.
    """
    assert "../_generated/workflow_grid.rst" in _read(AUTOAPI_INDEX)
    assert "_generated/workflow_grid.rst" in _read(DOCS_INDEX)

    underline = generator.SECTION_HEADING_CHAR
    docs = _read(DOCS_GRID)
    readme = _readme_block(generator)
    assert (_bands(docs, underline, "DocModule")
            == _bands(readme, underline, "Module")), (
        "the README grid and the API homepage grid have stopped agreeing "
        "about their sections or tile order")
    assert _targets(docs, "DocModule") == _targets(readme, "Module"), (
        "the two grids send the same tile to different pages")

    # ONE FUNCTION BUILDS BOTH. The drift the shared block prevents comes
    # back the moment there are two places to add a heading to.
    assert generator._readme_workflow.__code__.co_names.count("_grid_markup")
    assert generator._documentation_workflow.__code__.co_names.count(
        "_grid_markup")
    assert generator._module_grid() == [
        tile for _s, _n, tiles in generator._grid_sections() for tile in tiles]


def test_no_tile_destination_changed(generator):
    """366 part 3's anchors survive the regrouping."""
    from spacr.qt.screens.settings_model import api_docs_url

    for text, prefix in ((_read(DOCS_GRID), "DocModule"),
                         (_readme_block(generator), "Module")):
        targets = _targets(text, prefix)
        assert len(targets) == len(generator._module_grid())
        for key, target in targets.items():
            assert target == api_docs_url(key), (
                f"the {key} tile no longer points where the running "
                "application's Help button points")
        for key, pinned in ANCHORED_DESTINATIONS.items():
            assert targets[key] == pinned, (
                f"the {key} tile lost the entry-point anchor instruction "
                "366 part 3 gave it and fell back to a shared module page")


def test_every_folded_module_is_reachable_under_its_host(generator):
    """The homepage names the other twenty-three, under what opens them.

    The generated fold reference existed and was tested before instruction
    374 and was included by NOTHING, so the twenty-three modules without a
    tile were unreachable from the API homepage that is supposed to
    document them.
    """
    assert "../_generated/folded_modules.rst" in _read(AUTOAPI_INDEX), (
        "the API homepage does not include the fold reference, so a module "
        "with no tile cannot be reached from it")

    from spacr.qt.app import APPS
    from spacr.qt.screens.map_barcodes import fold_description

    names = {key: label for key, label, _desc, _section in APPS}
    text = _read(DOCS_FOLDS)
    hosts = generator._fold_hosts()
    assert hosts, "nothing is folded; the fold seam moved"

    bullets = {}
    for line in text.splitlines():
        match = re.match(r"\* \*\*(?P<host>[^*]+)\*\* opens (?P<rest>.+)$",
                         line)
        if match:
            bullets[match.group("host")] = match.group("rest")

    for key, host in sorted(hosts.items()):
        host_name = names.get(host, host)
        assert host_name in bullets, (
            f"the fold reference never says what {host_name} opens")
        label = names.get(key) or fold_description(key)[0] or key
        assert label in bullets[host_name], (
            f"{label} is folded onto {host_name} but is not listed under it")


def test_the_folded_modules_link_to_their_api_page_where_there_is_one(
        generator):
    """Named under the host is the structure; linked is the reachability.

    Fourteen of the folds came out of this page as bare text because
    `_api_urls` is keyed on the registry and a module folded hard enough
    has no registry row. Asking `api_docs_url` -- the resolver the
    application's own Help button uses -- finds the page for all but the
    five that genuinely do not have one.
    """
    from spacr.qt.screens.settings_model import api_docs_url

    text = _read(DOCS_FOLDS)
    generator._registry()
    index = api_docs_url("no_such_module_key")
    linked = set(re.findall(r"<(https://[^>]+)>`_", text))
    unlinkable = []
    for key in generator._fold_hosts():
        target = api_docs_url(key)
        if target == index:
            unlinkable.append(key)
            continue
        assert target in linked, (
            f"{key} has an API page at {target} and the fold reference "
            "does not link it. Re-run "
            "packaging/generate_readme_visuals.py")
    assert index not in linked, (
        "a fold links to the API index, which is the page the reader is "
        "already on")
    # Pinned so that giving one of these four a page, or losing a page
    # somewhere else, has to be noticed here.
    assert sorted(unlinkable) == [
        "hit_list", "import_images", "methods_export", "napari_bridge",
        "regression_diagnostics",
    ]


def test_the_docs_landing_page_names_the_categories_the_gui_has():
    """The prose above the grid and the bands inside it must agree.

    `docs/source/index.rst` included the same grid and introduced it with
    "six categories: Core, Data, Results & QC, Explore, Assays, Design".
    Home has had four since the 2026-08-31 restructure, and the paragraph
    went unnoticed for as long as the grid it sits above was flat -- a
    reader had nothing to compare it against. Grouping the grid put the
    contradiction on one screen, so the paragraph is now checked.

    The names only: the sentence around them is editorial and a test that
    pinned it would fail on every rewording.
    """
    from spacr.qt.app import SECTION_ORDER, SECTION_NOTES, tiled_apps

    import spacr.qt
    spacr.qt.register_self_registering_modules()
    tiled_apps()  # populates SECTION_NOTES through _refresh_sections
    live = [section for section in SECTION_ORDER if section in SECTION_NOTES]
    text = _read(DOCS_INDEX)
    intro = text.partition("Not every screen is a tile")[0]
    counted = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
               6: "six", 7: "seven", 8: "eight"}[len(live)]
    assert f"into {counted} categories" in intro, (
        f"the landing page does not say the GUI has {counted} categories, "
        f"and it has {live}")
    for section in live:
        assert f"*{section}*" in intro, (
            f"the landing page never names the {section!r} category")
    for retired in ("Results & QC", "Explore", "Design",
                    "Segmentation models"):
        if retired not in live:
            assert f"*{retired}*" not in intro, (
                f"the landing page still advertises the retired {retired!r} "
                "category")


def test_the_complete_reference_is_demoted_but_whole():
    """A collapsed section, not a deletion.

    215 alphabetical module links were the only prose-bearing section on
    the homepage and they answer a contributor's question, not a user's.
    They stay, behind a dropdown, generated by the same loop over the same
    pages -- so every link still resolves -- and the hidden toctree that
    puts those pages in the document tree stays with them, because
    dropping it turns 215 resolving links into 215 Sphinx warnings and the
    docs build runs with -W.
    """
    template = _read(AUTOAPI_INDEX)
    body = template.partition("Complete module reference")[2]
    assert body, "the complete reference is gone, not demoted"

    assert ".. toctree::\n   :hidden:" in body
    assert body.count('pages|selectattr("is_top_level_object")') == 2, (
        "the visible list and the hidden toctree must walk the same pages")
    assert ".. dropdown::" in body
    assert body.index(".. dropdown::") < body.index("* :doc:`"), (
        "the module list is not inside the dropdown")
    assert "* :doc:`{{ page.id }} <{{ page.include_path }}>`" in body, (
        "the links no longer carry an explicit include path, so they "
        "resolve only by luck")

    # The grid is what a user should meet first, so it has to come first.
    assert template.index("workflow_grid.rst") < template.index(
        "Complete module reference")
    assert template.index("folded_modules.rst") < template.index(
        "Complete module reference")
