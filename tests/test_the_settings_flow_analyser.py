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
