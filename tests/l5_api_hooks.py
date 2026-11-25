"""API revisions landing L5 changes in place; stdlib only.

ph_segment streams its staged fields (ws-offload-ws-phsegment-memory), so
spacr.ops_run_infra.FieldStager.load gained the ``mmap`` keyword and its
documented parameter. The reviewed body in
884_911_bmc_paper_items_2_api_arrivals_2026-11-10.json stays as recorded;
this projection advances exactly that one method to its current source
docstring, bound to the current signature (read with ``ast``, nothing
imported from spacr).

The decode audit (ws-offload-ws-audit-decode) documents the pixel-fallback
counts that spacr.ops_decode.field_calls now returns; its reviewed body in
871_911_bmc_paper_items_api_arrivals_2026-11-09.json advances the same way,
with the signature unchanged.
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


FIELD_CALLS = "spacr.ops_decode.field_calls"
FIELD_CALLS_BEFORE = 'Every call of one field under the rule, decoys included.\n\n:param evidence: from :func:`field_evidence`, or a field-cache ``.npz``\n    of the decode-yield study opened with :func:`numpy.load`.\n:param rule: from :func:`decode_rule`.\n:param library: the guide barcodes.\n:param matcher: from :func:`matcher_for`.\n:param decoys: the decoy set from :func:`matcher_for`.\n:param threshold: the read threshold on the spot strength.\n:param footprint: how far beyond a boundary a read may lie, in pixels.\n:param max_distance: the mismatches a read may carry; the runner-up\n    rule raises 0 to 1.\n:param min_fraction: the share of the votes the guide must hold.\n:param single_read_quality: the quality a one-read guide needs.\n:param crosstalk: the unmixing method.\n:param evidence_rows: also return ``evidence``, one row per call --\n    decoys included -- with the read and pixel support behind it\n    (:data:`EVIDENCE_COLUMNS`), which the genotype confidence reads.\n:returns: ``calls`` -- ``{object: {...}}`` with ``guide``, ``decoy``,\n    ``source`` (``primary`` or ``pixel_fallback``), ``n_reads``,\n    ``n_agreeing``, ``fraction``, ``quality``, ``barcode`` and\n    ``n_pixels`` -- for owned objects only; ``nuclei``, the objects this\n    field owns; ``assigned_real`` and ``assigned_decoy``; and read\n    counts for the well report.'
FIELD_CALLS_SIGNATURE = ("evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any], "
                         "library: Sequence[str], matcher, decoys, *, threshold: float, "
                         "footprint: float, max_distance: int, min_fraction: float, "
                         "single_read_quality: float, crosstalk: str='argmax_median', "
                         "evidence_rows: bool=False")


def _field_calls_source():
    """The current docstring of field_calls, checked against its unchanged signature.

    :returns: the docstring.
    """
    source = (ROOT / "spacr/ops_decode.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "field_calls")
    assert ast.unparse(node.args) == FIELD_CALLS_SIGNATURE
    assert ast.unparse(node.returns) == "Dict[str, Any]"
    doc = ast.get_docstring(node)
    assert "assigned_pixel_real" in doc
    return doc


def _current_l5_api(entries):
    """Advance only FieldStager.load and field_calls; every other body is kept as given.

    :param entries: ``symbol -> docstring`` of a reviewed contract.
    :returns: a copy with the revised bodies.
    """
    result = dict(entries)
    for symbol, before, current in ((SYMBOL, BEFORE, _load_source),
                                    (FIELD_CALLS, FIELD_CALLS_BEFORE, _field_calls_source)):
        if symbol in result:
            body = current()
            assert result[symbol] in (before, body), symbol
            result[symbol] = body
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
