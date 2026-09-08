"""The API link goes to a page that is ABOUT the setting, not merely a page.

Instruction 383, item 4. `test_tooltip_api_links_resolve.py` asks whether
every link RESOLVES -- that the target is a canonical public module, that
nothing falls through to the generated index -- and all 801 app/setting
pairs passed that while `src` in Mask pointed at the annotation-dataset
generator. "Resolves" and "is the right page for this reader" are
different claims and only the first was being made.

TWO PROPERTIES, and the second is the one the report was about.

* the anchored function READS the setting. Checked against
  ``docs/settings_flow.json``, which is produced by
  ``tools/settings_flow.py`` -- a different AST pass from the
  ``tools/build_setting_consumer_map.py`` one that chose the link. Two
  independent analyses agreeing is worth more than either alone, and a
  link at a function that never touches the value fails here.

* when the ASKING APP's own module reads the setting, the link goes
  there. This is the reported defect stated as a rule: `src` is shown in
  41 panels, Mask's own module reads it, and Mask's link went to another
  module's reader. Before 2026-09-08 this test would have failed on 109
  pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "docs" / "settings_flow.json"


@pytest.fixture(scope="module")
def flow():
    if not FLOW.is_file():
        pytest.skip("settings_flow.json is not generated in this checkout")
    return json.loads(FLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pairs():
    """Every (app, setting) a panel draws, with the link it offers."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import (api_docs_url,
                                                 resolve_default_settings)
    import spacr.qt

    spacr.qt.register_self_registering_modules()
    out = []
    for entry in APPS:
        app = entry[0]
        for key in resolve_default_settings(app):
            out.append((app, key, api_docs_url(app, key)))
    assert out, "no app/setting pairs were collected"
    return out


def own_display_only(module_path: str) -> bool:
    """Whether an app's API module is one the consumer map excludes."""
    dotted = "spacr." + str(module_path).replace("/", ".")
    return dotted.startswith(("spacr.qt.", "spacr.settings"))


def _dynamic_reads() -> set:
    """``(setting, function)`` pairs the map found by a DYNAMIC read.

    Read from the consumer map's own record rather than guessed: a hit
    whose form ends in ``-dynamic`` came from an f-string key or a walk
    over ``settings.items()``, which the flow analysis does not record as
    a named read. Keyed by the FUNCTION as well as the setting, because a
    setting can be read statically in one place and dynamically in
    another, and only the anchored one is being judged.
    """
    consumers = ROOT / "docs" / "setting_consumers.json"
    if not consumers.is_file():
        return set()
    data = json.loads(consumers.read_text(encoding="utf-8"))["consumers"]
    return {(key, f"{h['module']}.{h['qualname']}")
            for key, hits in data.items() for h in hits
            if str(h.get("form", "")).endswith("-dynamic")}


def _anchor(url: str) -> str:
    return url.rsplit("#", 1)[1] if "#" in url else ""


def test_an_anchored_link_points_at_a_function_that_names_the_setting():
    """A link at a function that never mentions the setting is a wrong page.

    READS THE SOURCE, not either analysis. Comparing the two AST passes
    was tried and is not a fair test: they differ in COVERAGE, not in
    correctness. `settings_flow` does not record `self.settings.get('src')`
    inside a method, so it called `spacr.batch.Job.default_label` a
    stranger to `src` when its second line is exactly that read; and the
    consumer map recognises a walk over `settings.items()` behind an
    f-string prefix, which `settings_flow` does not, so eleven correct
    organelle links looked wrong too. A check whose failures are mostly
    its own blind spots is one nobody reads twice.

    What both agree on, and what a reader can verify in a second, is that
    the function they were sent to says the word. That is a floor rather
    than a proof -- it would not have caught the reported `src` defect,
    which property two is for -- but it catches the case with no
    connection at all.

    THE CROSS-TOOL COMPARISON DID ITS JOB BEFORE IT WAS REMOVED. It found
    that `f(x=...)` was recorded as a read of the setting `x` whatever the
    value: 40% of all hits, including `diameter=30`, a hardcoded literal,
    which is why the Plaque assay's `diameter` link pointed at
    `submodules.test_cellpose_model`. That is fixed in the generator.
    """
    import importlib
    import inspect

    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import (api_docs_url,
                                                 resolve_default_settings)
    import spacr.qt

    spacr.qt.register_self_registering_modules()
    dynamic = _dynamic_reads()
    sources: dict = {}
    silent = []
    for entry in APPS:
        app = entry[0]
        for key in resolve_default_settings(app):
            anchor = _anchor(api_docs_url(app, key))
            if not anchor or not anchor.startswith("spacr."):
                continue
            # A DYNAMIC READ NEVER SPELLS THE KEY. `filter_selection` builds
            # `f"{obj}_min_size"`, so `cell_min_size` is nowhere in its
            # module and the link is still right.
            #
            # Matched as a PREFIX because a closure has no importable name:
            # the reader is anchored at `merge_split_filter_masks` while the
            # read is recorded in `merge_split_filter_masks._run_one`.
            if any(fn == anchor or fn.startswith(anchor + ".")
                   for k, fn in dynamic if k == key):
                continue
            module_name, _, qual = anchor.rpartition(".")
            if module_name not in sources:
                try:
                    sources[module_name] = inspect.getsource(
                        importlib.import_module(module_name))
                except Exception:                            # noqa: BLE001
                    sources[module_name] = ""
            text = sources[module_name]
            if text and key not in text:
                silent.append(f"{app}/{key} -> {anchor}")
    assert not silent, (
        "these links send a reader to a module that never mentions the "
        "setting:\n  " + "\n  ".join(sorted(set(silent))[:20]))


def test_the_link_prefers_the_asking_apps_own_module(pairs, flow):
    """THE REPORTED DEFECT, as a rule.

    `_mapped_api_target` was keyed on the setting alone, so a setting
    shown in 41 panels had one destination for all of them. Where the
    app's own module reads the setting, that is the module the reader
    should land in.
    """
    from spacr.qt.screens.settings_model import (_APP_API_MODULE,
                                                 _BATCH_PREFIX_STRANGERS,
                                                 _EVALUATION_DOC_KEYS,
                                                 _UMAP_SEARCH_DOC_KEYS,
                                                 _mapped_api_target)

    # THE HAND-WRITTEN EXCEPTIONS ARE REASONED, and this rule does not
    # overrule them. A batch-correction setting lands on the module that
    # implements the correction rather than on whichever app happens to
    # display it, and the same for the classifier-evaluation keys and
    # UMAP's search keys. Each was checked by a person; the per-app table
    # is mechanical and yields to them.
    #
    # `_BATCH_PREFIX_STRANGERS` is NOT exempt: `batch_fields` and
    # `batch_size` only share a prefix with that family, and sending them
    # to batch_correction is the defect this list records.
    deliberate = set(_EVALUATION_DOC_KEYS) | set(_UMAP_SEARCH_DOC_KEYS)
    readers = {key: {hit["function"] for hit in hits}
               for key, hits in flow["reads"].items()}
    misdirected = []
    for app, key, url in pairs:
        own = _APP_API_MODULE.get(app)
        if not own or key not in readers:
            continue
        if key in deliberate:
            continue
        if key.startswith("batch_") and key not in _BATCH_PREFIX_STRANGERS:
            continue
        # A Qt SCREEN IS NOT WHERE DOCUMENTATION POINTS. The consumer map
        # excludes `spacr.qt.*` and `spacr.settings` by design -- they
        # DISPLAY a setting rather than act on it -- so an app whose own
        # module is one of those has no row to prefer and correctly falls
        # through. `queue` is `qt/plate_queue`.
        if own_display_only(own):
            continue
        own_module = "spacr." + str(own).replace("/", ".")
        # Does the app's OWN module read this setting at all?
        if not any(fn.rsplit(".", 1)[0] == own_module
                   for fn in readers[key]):
            continue
        # THE MODULE CHOICE, NOT THE FINAL URL. A setting with no
        # publishable anchor now presents as a link to the settings-flow
        # page instead of the top of a 4,000-line module -- see
        # `test_an_anchorless_link_goes_to_the_page_that_names_the
        # _setting`. That changes how the answer is shown, not which
        # module was chosen, and this rule is about the choice. Asserting
        # on the URL made a presentation change look like a regression.
        # `_mapped_api_target` names the module the way the URL does --
        # "core", not "spacr.core" -- so compare in that form.
        landed = str(_mapped_api_target(key, app)[0]).replace("/", ".")
        if landed != own_module[len("spacr."):]:
            misdirected.append(f"{app}/{key} -> {landed or 'nothing'}")
    assert not misdirected, (
        "the asking app's own module reads these settings, and the link "
        "goes somewhere else:\n  " + "\n  ".join(sorted(misdirected)[:20]))


def test_the_check_would_have_caught_the_reported_defect(flow):
    """A checker that cannot fail is a checker nobody should trust.

    Asserted against the SHAPE of the report rather than the live map,
    so it keeps testing the rule after the map is fixed -- which it now
    is. `src` is read by `spacr.core`, Mask's own module, and the old
    behaviour sent Mask to `spacr.annotation_dataset`.
    """
    readers = {hit["function"] for hit in flow["reads"].get("src", [])}
    assert any(fn.startswith("spacr.core.") for fn in readers), (
        "`src` is no longer read in spacr.core, so the example this rule "
        "was written from has moved and the rule needs re-grounding")
    assert any(fn.startswith("spacr.annotation_dataset.") for fn in readers), (
        "`src` is no longer read in spacr.annotation_dataset, so the "
        "wrong answer it used to give is no longer reachable")


def test_an_anchorless_link_goes_to_the_page_that_names_the_setting():
    """The `magnification` report, and 185 links like it.

    "i tested the API link for Magnefication and got a page with no
    mention of magnefication" was not a wrong module -- `utils` IS where
    it is read -- but the only consumer is private, so AutoAPI publishes
    no anchor and the reader lands at the top of 4,000 lines. 245 of 796
    links were in that position.

    The settings-flow page names the setting, carries its help text and
    lists every function that reads it, so it answers what the reader
    pressed API to ask. This asserts the fallback fires for the reported
    setting and NOT for one that has a real API anchor, because a
    fallback that fires everywhere would bury the precise links under
    the general page.
    """
    from spacr.qt.screens.settings_model import api_docs_url

    landed = api_docs_url("mask", "magnification")
    assert landed.endswith("settings_flow.html#setting-flow-magnification"), (
        landed)
    # `src` in Mask has a real consumer and must keep pointing at it.
    src = api_docs_url("mask", "src")
    assert "settings_flow.html" not in src, src
    assert src.endswith("#spacr.core.preprocess_generate_masks"), src


def test_the_flow_index_agrees_with_the_page_it_was_written_from():
    """An anchor the page does not carry is worse than no anchor.

    The browser ignores an unknown fragment in silence and the reader
    believes they are looking at the right place -- which is exactly the
    defect item 3 of this instruction closed for the API links. The
    index and the page are written by one program in one run, and this
    is what holds them to it.
    """
    import re
    from pathlib import Path

    from spacr.qt.screens.settings_flow_index import (
        SETTINGS_WITH_A_FLOW_SECTION)

    page = (Path(__file__).resolve().parents[1] / "docs" / "source" /
            "_generated" / "settings_flow.rst")
    if not page.exists():                     # docs artefact, not shipped
        pytest.skip("settings_flow.rst not generated in this checkout")
    drawn = set(re.findall(r"^\.\. _setting-flow-(.+):$",
                           page.read_text(encoding="utf-8"), re.M))
    assert SETTINGS_WITH_A_FLOW_SECTION == drawn, (
        "the generated index and the generated page disagree; re-run "
        "tools/settings_flow.py --rst, which writes both")


def test_the_fallback_is_silent_when_the_index_is_missing():
    """A checkout that never ran the generator still draws its panel.

    The index is a generated file. Importing it eagerly would make a
    missing artefact a crash in the settings panel, which trades a
    better link for a worse failure.
    """
    import builtins

    from spacr.qt.screens import settings_model

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if "settings_flow_index" in name:
            raise ImportError("generated file absent")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = refuse
    try:
        assert settings_model._has_a_flow_section("magnification") is False
        assert "settings_flow.html" not in settings_model.api_docs_url(
            "mask", "magnification")
    finally:
        builtins.__import__ = real_import
