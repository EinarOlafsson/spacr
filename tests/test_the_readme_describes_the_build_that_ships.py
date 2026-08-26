"""The README and the docs grid name the modules the build actually has.

Twenty modules folded into a host and lost their registry row. The tile grids
are written from that registry, so a folded module left in one advertises a
tool the application no longer offers, and a newly registered module missing
from one is a tool nobody can find. Both went unnoticed for a fortnight
because nothing compared the two.

These tests compare them. They read the live registry rather than a list
written down here, so folding the next module makes them fail until the grids
follow.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
DOCS_GRID = ROOT / "docs" / "source" / "_generated" / "workflow_grid.rst"
FEATURES = ROOT / "docs" / "source" / "features.rst"
LOCALIZED = sorted((ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"))


def _registry():
    import spacr.qt
    from spacr.qt.app import APPS, SECTIONS

    spacr.qt.register_self_registering_modules()
    return list(APPS), list(SECTIONS)


def _folded_keys() -> set:
    """Every module reached from a host's masthead instead of a tile."""
    import importlib
    import pkgutil

    import spacr.qt.screens as screens

    keys = set()
    for module in (m.name for m in pkgutil.iter_modules(screens.__path__)):
        try:
            loaded = importlib.import_module(f"spacr.qt.screens.{module}")
        except Exception:  # a screen that will not import hosts nothing
            continue
        keys.update(getattr(loaded, "FOLDED_APPS", ()) or ())
    from spacr.qt.screens.make_masks import FOLD_ORDER
    keys.update(FOLD_ORDER)
    # `cellpose_all` is the fold's key; `cellpose_masks` is what its tile was
    # called. Both spellings must stay out of the grids.
    if "cellpose_all" in keys:
        keys.add("cellpose_masks")
    return keys


def tiles(text: str, prefix: str = "App") -> set:
    """The app keys a grid draws a tile for."""
    return set(re.findall(rf"\|{prefix}_([a-z0-9_]+)\|", text))


@pytest.fixture(scope="module")
def registry():
    return _registry()


@pytest.fixture(scope="module")
def folded():
    keys = _folded_keys()
    assert keys, "no module is folded; the fold seam moved"
    return keys


class TestTheReadmeGrid:
    def test_the_registry_count_is_deliberately_pinned(self, registry):
        apps, sections = registry
        assert len(apps) == 44
        assert sections == [
            "Core", "Data", "Results & QC", "Explore", "Assays", "Design"
        ]

    def test_no_folded_module_is_offered_as_a_separate_tool(self, folded):
        drawn = tiles(README.read_text(encoding="utf-8"))
        offered = sorted(drawn & folded)
        assert not offered, (
            "the README offers these as separate tools, but each is reached "
            f"from a host's masthead and has no registry row: {offered}")

    def test_every_module_with_a_row_has_a_tile(self, registry):
        apps, _ = registry
        expected = {key for key, _n, _d, section in apps if section != "Core"}
        drawn = tiles(README.read_text(encoding="utf-8"))
        assert expected <= drawn, (
            "these modules are in the registry but have no README tile: "
            f"{sorted(expected - drawn)}")

    def test_it_draws_nothing_the_registry_does_not_have(self, registry):
        apps, _ = registry
        known = {key for key, _n, _d, _s in apps}
        drawn = tiles(README.read_text(encoding="utf-8"))
        assert drawn <= known, (
            f"the README draws tiles for unknown keys: {sorted(drawn - known)}")

    def test_the_core_row_is_the_core_section(self, registry):
        apps, _ = registry
        core = {key for key, _n, _d, section in apps if section == "Core"}
        drawn = tiles(README.read_text(encoding="utf-8"), prefix="Workflow")
        assert drawn - {"arrow"} == core


class TestTheGeneratedDocsGrid:
    def test_it_agrees_with_the_readme(self, registry):
        apps, _ = registry
        expected = {key for key, _n, _d, section in apps if section != "Core"}
        drawn = tiles(DOCS_GRID.read_text(encoding="utf-8"), prefix="DocApp")
        assert drawn == expected, (
            "the docs grid and the registry disagree: "
            f"extra={sorted(drawn - expected)} "
            f"missing={sorted(expected - drawn)}")

    def test_no_folded_module_survives_in_it(self, folded):
        drawn = tiles(DOCS_GRID.read_text(encoding="utf-8"), prefix="DocApp")
        assert not drawn & folded, sorted(drawn & folded)

    def test_it_has_no_section_the_gui_does_not(self, registry):
        _apps, sections = registry
        text = DOCS_GRID.read_text(encoding="utf-8")
        headings = set(re.findall(r"(?m)^([A-Z][^\n]*)\n\^{3,}$", text))
        assert headings <= set(sections), (
            f"the docs grid has sections the GUI does not: "
            f"{sorted(headings - set(sections))}")


class TestTheLocalizedReadmes:
    @pytest.mark.parametrize("path", LOCALIZED, ids=lambda p: p.name)
    def test_it_draws_the_same_tiles_as_the_english_one(self, path):
        assert tiles(path.read_text(encoding="utf-8")) == tiles(
            README.read_text(encoding="utf-8")), (
            f"{path.name} and README.rst disagree about which modules exist")

    @pytest.mark.parametrize("path", LOCALIZED, ids=lambda p: p.name)
    def test_every_tile_it_draws_is_defined(self, path):
        text = path.read_text(encoding="utf-8")
        used = set(re.findall(r"\|((?:App|Workflow)_[a-z0-9_]+)\|(?!\s*image)",
                              text))
        defined = set(re.findall(r"(?m)^\.\. \|([A-Za-z0-9_]+)\| image::",
                                 text))
        assert used <= defined, (
            f"{path.name} uses substitutions it never defines -- the image "
            f"would not render: {sorted(used - defined)}")

    @pytest.mark.parametrize("path", LOCALIZED, ids=lambda p: p.name)
    def test_workflow_accessibility_text_stays_localized(self, path):
        text = path.read_text(encoding="utf-8")
        assert ":alt: Open the " not in text

    @pytest.mark.parametrize("path", LOCALIZED, ids=lambda p: p.name)
    def test_installer_and_dataset_definitions_survive_regeneration(self, path):
        text = path.read_text(encoding="utf-8")
        installer = text.partition(
            ".. spacr-installer-links-begin"
        )[2].partition(".. spacr-installer-links-end")[0]
        for name in (
            "InstallerWindows", "InstallerMacOS", "InstallerLinux",
            "InstallerLegacy",
        ):
            definition = f".. |{name}| image::"
            assert text.count(definition) == 1
            assert definition in installer
        for name in (
            "DataBioStudies", "DataHuggingFace", "DataNCBI",
            "DataSpaCRPower", "DataBioRxiv",
        ):
            definition = f".. |{name}| image::"
            assert text.count(definition) == 1
            assert definition not in installer


class TestTheProseMatchesTheScreens:
    def test_the_readme_names_every_make_masks_tool(self):
        from spacr.qt.screens.make_masks import tool_row_entries

        text = README.read_text(encoding="utf-8")
        for _mode, label, _icon in tool_row_entries():
            assert label in text, (
                f"Make Masks offers a {label!r} tool that the README does "
                "not name")

    def test_the_feature_guide_names_every_fold_host(self):
        text = FEATURES.read_text(encoding="utf-8")
        for host in ("Mask", "Measure", "Annotate", "Classify",
                     "Map Barcodes", "Regression", "Image UMAP",
                     "Make Masks"):
            assert host in text, f"{host} hosts folds but is not documented"

    def test_the_feature_guide_names_the_gui_categories(self, registry):
        _apps, sections = registry
        text = FEATURES.read_text(encoding="utf-8")
        for section in sections:
            assert section in text, (
                f"the GUI has a {section!r} category the feature guide does "
                "not mention")

    def test_nothing_advertises_the_deleted_tk_interface(self):
        for path in [README, FEATURES, DOCS_GRID, *LOCALIZED]:
            text = path.read_text(encoding="utf-8").lower()
            for gone in ("spacr-legacy", "tkinter"):
                assert gone not in text, (
                    f"{path.name} advertises {gone!r}; the Tk interface was "
                    "deleted")
