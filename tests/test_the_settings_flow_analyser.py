"""``tools/settings_flow.py``: where a setting goes, and what it will not guess.

Instruction 368 asks for a clickable tree of the functions below a setting's
entry point that actually use it. These tests pin the analysis the tree is
drawn from -- that it finds the real readers, prunes the branches that reach
none, and marks what static analysis cannot follow instead of dropping it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def flow():
    spec = importlib.util.spec_from_file_location(
        "settings_flow", ROOT / "tools" / "settings_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(flow):
    return flow.analyse()


def test_the_propagation_subgraph_is_small_enough_to_draw(data):
    """The finding that makes the feature possible.

    The package has ~156,000 call sites and a naive name match resolves 18%
    of them; a tree over that is neither buildable nor readable. The tree
    needs only the edges settings TRAVEL along, which is a few hundred. If
    this number ever approaches the whole call graph, the design assumption
    has broken and the tree will not be usable.
    """
    assert len(data["edges"]) < 5000, len(data["edges"])
    assert len(data["receivers"]) < 2000, len(data["receivers"])
    resolved = [e for e in data["edges"] if e["confidence"] == "RESOLVED"]
    assert len(resolved) > len(data["edges"]) // 2, (
        "fewer than half the settings-carrying edges resolve; the tree would "
        "be mostly holes")


def test_a_per_object_setting_is_found_through_its_f_string(data):
    """`cell_channel` is never written literally in most of its readers.

    It is built as ``f'{object_type}_channel'``, and a literal-only matcher
    reports it as read by nothing at all. Expanding the object roles is what
    turns most of the dynamic reads into concrete keys.
    """
    readers = {hit["function"] for hit in data["reads"].get("cell_channel", [])}
    assert readers, "cell_channel appears to be read nowhere"
    assert any(hit["form"].endswith("-dynamic")
               for hit in data["reads"]["cell_channel"]), (
        "no dynamic read was resolved; the f-string expansion is not working")
    for role in ("nucleus", "pathogen"):
        assert data["reads"].get(f"{role}_channel"), (
            f"{role}_channel was not expanded alongside cell_channel")


def test_the_tree_names_the_module_a_user_would_expect(data, flow):
    """The worked example from the instruction.

    Clicking `cell_channel` in Mask should lead to the mask entry point and
    the branches below it that use the setting -- not to a module page, and
    not to a function that merely mentions the word.
    """
    tree = flow.tree_for(data, "cell_channel")
    assert "spacr.core.preprocess_generate_masks" in tree
    assert "<-- reads it" in tree, "no reader was marked in the tree"
    # Pruned: a branch that reaches no reader is not drawn.
    for line in tree.splitlines()[1:]:
        assert line.strip(), "the tree has blank rows"


def test_a_setting_nothing_reads_says_so(data, flow):
    """Silence is not an answer. An unread key must be reported as unread."""
    tree = flow.tree_for(data, "a_setting_that_does_not_exist")
    assert "read nowhere" in tree


def test_an_unfollowable_call_is_marked_not_dropped(data):
    """getattr, dispatch dicts, Qt signals and callbacks cannot be followed.

    They must be recorded as UNRESOLVED. A tree that silently omits what it
    could not resolve is worse than one that admits the gap, because it looks
    complete -- and the reader has no way to tell which they are looking at.
    """
    unresolved = [e for e in data["edges"] if e["confidence"] == "UNRESOLVED"]
    assert unresolved, "every edge resolved, which is not credible"
    assert all(e["callee"] is None for e in unresolved)
    assert all(e["raw"] for e in unresolved), (
        "an unresolved edge does not even say what it tried to call")


def test_the_analysis_needs_no_import_of_spacr(flow):
    """Pure ast, so a broken import cannot empty the trees.

    It also means no torch, no Qt and no GPU are needed to build the docs.
    """
    import ast

    source = (ROOT / "tools" / "settings_flow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "spacr" not in imported, sorted(imported)
    for heavy in ("torch", "PySide6", "matplotlib", "cellpose"):
        assert heavy not in imported


# ---------------------------------------------------------------------------
# Phase 3: the page the reader actually clicks.
# ---------------------------------------------------------------------------

def test_every_branch_in_the_page_is_clickable(data, flow):
    """"All branches should be clickable" is the request, so the page must
    not render the tree as preformatted text.

    A ``::`` literal block would be simpler and would show the same shape,
    and nothing in it would be a link. The page uses a line block instead,
    which keeps one source line per output line AND interprets roles.
    """
    page = flow.rst_for(data, keys=["cell_channel"])
    assert "::" not in page.split("cell_channel")[-1].split("Read by")[0], (
        "the tree is in a literal block, where nothing is clickable")
    assert ":py:func:`~spacr.core.preprocess_generate_masks`" in page
    assert page.count(":py:func:") >= 4, "hardly anything is a link"


def test_the_shape_survives_the_line_block(data, flow):
    """Indentation carries the meaning, so it must not be collapsed.

    RST discards ordinary leading whitespace inside a line block, so the
    depth is drawn with non-breaking spaces. Without them every node would
    appear at the same level and the tree would say nothing.
    """
    page = flow.rst_for(data, keys=["cell_channel"])
    rows = [r for r in page.splitlines() if r.startswith("| ")]
    assert rows, "no line-block rows at all"
    depths = {len(r) - len(r.replace(" ", "")) for r in rows}
    assert len(depths) > 1, f"every row is at one depth: {depths}"


def test_a_private_reader_is_named_but_not_linked(data, flow):
    """AutoAPI publishes no page for a private function.

    Emitting a link to one produces a Sphinx warning and a broken link for
    the reader, so those are shown as plain names -- which still tells the
    reader where the value is used.
    """
    page = flow.rst_for(data, keys=["cell_channel"])
    assert "``_normalize_img_batch``" in page
    assert ":py:func:`~spacr.io._normalize_img_batch`" not in page


def test_each_setting_has_an_anchor_to_arrive_at(data, flow):
    """A reader coming from a tooltip lands on that setting, not the top."""
    page = flow.rst_for(data, keys=["cell_channel", "nucleus_channel"])
    assert ".. _setting-flow-cell_channel:" in page
    assert ".. _setting-flow-nucleus_channel:" in page


def test_the_page_says_what_it_could_not_follow(data, flow):
    """The unresolved marker has to survive into the rendered page."""
    page = flow.rst_for(data)
    assert "[UNRESOLVED]" in page
    assert "cannot follow" in page, "the page never explains the marker"


def test_the_defaults_setter_is_not_drawn_as_a_reader(flow):
    """A function that supplies the default is not a place the value GOES.

    Every setting has a defaults-setter by construction -- that is what
    makes it a setting -- so drawing it as a reader adds a row to every
    section and answers a question nobody asked. It was 30% of the
    published page: `set_default_settings_preprocess_generate_masks`
    appeared 803 times across 1057 sections, 636 of those as a leaf,
    burying the function that actually acts on the value.
    """
    assert flow._supplies_the_default(
        "spacr.settings.set_default_settings_preprocess_generate_masks")
    assert flow._supplies_the_default("spacr.settings.deep_spacr_defaults")
    assert flow._supplies_the_default(
        "spacr.settings.get_perform_regression_default_settings")
    assert flow._supplies_the_default("spacr.settings.set_default_classify")
    # The functions that DO act on a value must survive, including ones
    # whose names merely contain the word.
    assert not flow._supplies_the_default("spacr.core.preprocess_generate_masks")
    assert not flow._supplies_the_default("spacr.object.generate_cellpose_masks")
    assert not flow._supplies_the_default("spacr.io._read_and_merge_data")


def test_the_page_stays_inside_the_docs_job_timeout(flow, data):
    """A ratchet on cross-references, because the build clock is the cost.

    The docs job has 30 minutes. Every ``:py:func:`` on this page is an
    xref Sphinx resolves against the whole AutoAPI inventory, and the
    first published version carried 9,427 of them -- which took the build
    from 30 minutes to over 80 and could not go green at any seed.

    The number, not the line count, is what to watch: dropping the
    defaults-setters cut lines by 15% and xrefs by 35%. If this ceiling
    is raised, raise it against a MEASURED build rather than a guess,
    because nothing else in the suite renders a page.

    RAISED TO 7,500 ON 2026-09-09, AGAINST THREE MEASURED BUILDS. The
    rule above was followed rather than the number nudged: instruction
    383 taught both settings analysers two more binding forms and a
    constant fold, which took the page from 6,301 cross-references to
    7,000, and each state was built with `sphinx -E -b html` on the same
    machine, back to back, nothing else changed:

        6,301 xrefs   4:15.20 wall   2,397,992 KB peak   0 warnings
        6,541 xrefs   4:16.34 wall   2,413,244 KB peak   0 warnings
        7,000 xrefs   4:16.19 wall   2,420,284 KB peak   0 warnings

    SEVEN HUNDRED MORE CROSS-REFERENCES COST ABOUT ONE SECOND. Whatever
    took the first published version from 30 minutes to over 80, it does
    not scale from these numbers, so the ceiling is raised to 7,500 --
    500 of headroom, the same margin the 6,500 carried -- rather than
    removed. It is still the only thing standing between this page and a
    docs job that cannot finish, and the next raise wants its own three
    lines above.

    The builds also came back with ZERO warnings, which is the second
    thing this page can break: the docs job runs `-W`, so an xref the
    inventory cannot resolve fails it as surely as a timeout.
    """
    rst = flow.rst_for(data)
    xrefs = rst.count(":py:func:")
    assert xrefs <= 7500, (
        f"{xrefs} cross-references on the settings-flow page, up from the "
        "7,000 measured against a real sphinx-build on 2026-09-09. This is "
        "the build-time ratchet; confirm a real sphinx-build before raising "
        "it, and record the wall clock in the docstring above")


def test_a_setting_read_only_by_its_defaults_setter_is_dropped(flow, data):
    """Nothing to draw is a reason not to draw a section.

    44 settings are read only by the function that defines their default.
    With that row gone they have no flow at all, and an empty section
    with a heading reads as "we looked and there is nothing here" when
    the truth is that the question does not apply.
    """
    rst = flow.rst_for(data)
    drawn = rst.count(".. _setting-flow-")
    assert drawn < len(data["reads"]), (
        "every setting still gets a section, so the defaults-only ones "
        "are being drawn empty rather than dropped")
    assert f"{drawn} settings, of {len(data['reads'])} read anywhere." in rst
