"""API revisions landing L5 changes in place; stdlib only.

ph_segment streams its staged fields (ws-offload-ws-phsegment-memory), so
spacr.ops_run_infra.FieldStager.load gained the ``mmap`` keyword and its
documented parameter. The reviewed body in
884_911_bmc_paper_items_2_api_arrivals_2026-11-10.json stays as recorded;
this projection advances exactly that one method to its current source
docstring, bound to the current signature (read with ``ast``, nothing
imported from spacr).
"""
import ast
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "spacr.ops_run_infra.FieldStager.load"
BEFORE = ("One staged field's stack.\n\n:param site: the site.\n"
          ":returns: ``(stack or None, unreadable)``.")
SIGNATURE = "self, site: int, mmap: bool=False"
RETURNS = "Tuple[Optional[np.ndarray], list]"


def _load_source(source=None):
    """The current docstring of FieldStager.load, checked against its admitted signature.

    :param source: the text of spacr/ops_run_infra.py; None reads the file.
    :returns: the docstring.
    """
    if source is None:
        source = (ROOT / "spacr/ops_run_infra.py").read_text(encoding="utf-8")
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "FieldStager")
    node = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "load")
    assert ast.unparse(node.args) == SIGNATURE
    assert ast.unparse(node.returns) == RETURNS
    doc = ast.get_docstring(node)
    assert doc.startswith(BEFORE.split(":returns:")[0]) and ":param mmap:" in doc
    return doc


def _current_l5_api(entries):
    """Advance only FieldStager.load; every other body is kept as given.

    :param entries: ``symbol -> docstring`` of a reviewed contract.
    :returns: a copy with the one revised body.
    """
    result = dict(entries)
    if SYMBOL in result:
        current = _load_source()
        assert result[SYMBOL] in (BEFORE, current)
        result[SYMBOL] = current
    return result



# The public names the graph-style rollout (ws-offload-codex-home-graphstyle-g2a-i18n)
# adds beyond graphstyle_g1_api_arrivals_2026-11-23.json: eleven
# spacr.figures.style helpers for the Measure and quality-control views, and
# TileLayoutWidget.resizeEvent, which reflows the tile-layout key; and Home's G5
# fix (ws-offload-claude-home-g5-dnaphase-fix): spacr.measure.mitotic_holdout,
# the one mitotic flag held out of the DNA-phase fit, and
# spacr.ops_tvn.dna_posterior_column, the one name list of the DNA-fit
# posteriors. They arrive
# through tests/data/release_contracts/l5_api_arrivals_2026-11-24.json, each
# body bound to its source docstring.
BIRTH = ROOT / "tests/data/release_contracts/l5_api_arrivals_2026-11-24.json"
ARRIVALS = frozenset({
    "spacr.figures.style." + name for name in (
        "apply_colourbar_theme", "apply_embedding_style",
        "ensure_identity_contrast", "ensure_reference_contrast",
        "external_legend", "image_report_css", "native_histogram_geometry",
        "native_plot_font", "paint_native_histogram", "resolve_plot_theme",
        "styled_reference_line")} | {
    "spacr.qt.screens.align.TileLayoutWidget.resizeEvent",
    "spacr.measure.mitotic_holdout", "spacr.ops_tvn.dna_posterior_column"})
#: The arrivals that are callables of the frozen inventory.
INVENTORY = frozenset({"spacr.qt.screens.align.TileLayoutWidget.resizeEvent",
                       "spacr.measure.mitotic_holdout",
                       "spacr.ops_tvn.dna_posterior_column"})


def _source_docstring(symbol):
    """The cleaned docstring of a function or method, read from its source file.

    :param symbol: the dotted name, ``spacr.<module path>.<name>[.<method>]``.
    :returns: the docstring.
    """
    parts = symbol.split(".")
    for split in range(len(parts), 1, -1):
        path = ROOT.joinpath(*parts[:split]).with_suffix(".py")
        if path.is_file():
            break
    node = ast.parse(path.read_text(encoding="utf-8"))
    for name in parts[split:]:
        node = next(item for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                    and item.name == name)
    return inspect.cleandoc(ast.get_docstring(node, clean=False))


def _current_l5_arrivals(existing=None):
    """The new public names bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    :returns: ``symbol -> docstring``.
    """
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_l5_arrivals(callables):
    """Remove only the exact new public callables from the frozen boundary.

    :param callables: the inventory of public callables.
    :returns: the inventory without the arrivals.
    """
    entries = _current_l5_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == INVENTORY, sorted(found)
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
