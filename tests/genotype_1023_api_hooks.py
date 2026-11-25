"""Public API changes of item 1023, stricter genotype calls; stdlib only.

Two public callables arrive in spacr.ops_decode: split_read_concordance, the
per-well split-read QC metric, and decode_cached_well, a decode from stored
field caches. They arrive through
tests/data/release_contracts/1023_genotype_calls_api_arrivals_2026-11-25.json,
each body bound to its source docstring. One reviewed body advances in place:
spacr.ops_decode.field_calls of
871_911_bmc_paper_items_api_arrivals_2026-11-09.json now names the
pixel-fallback counts (schema 964) and the ``split`` reads of the field (read with
``ast``, nothing imported from spacr).
"""
import ast
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTH = ROOT / "tests/data/release_contracts/1023_genotype_calls_api_arrivals_2026-11-25.json"
ARRIVALS = frozenset({"spacr.ops_decode.split_read_concordance",
                      "spacr.ops_decode.decode_cached_well"})
REVISED = "spacr.ops_decode.field_calls"
REVISED_SIGNATURE = ("evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any], "
                     "library: Sequence[str], matcher, decoys, *, threshold: float, "
                     "footprint: float, max_distance: int, min_fraction: float, "
                     "single_read_quality: float, crosstalk: str='argmax_median', "
                     "evidence_rows: bool=False")


def _node(symbol):
    """The function node of a dotted ``spacr.ops_decode`` name, read from source."""
    tree = ast.parse((ROOT / "spacr/ops_decode.py").read_text(encoding="utf-8"))
    name = symbol.rsplit(".", 1)[-1]
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _source_docstring(symbol):
    """The cleaned docstring of a spacr.ops_decode function, read from its source."""
    return inspect.cleandoc(ast.get_docstring(_node(symbol), clean=False))


def _current_1023_api(entries):
    """Advance only field_calls; every other body is kept as given.

    :param entries: ``symbol -> docstring`` of a reviewed contract.
    :returns: a copy with the one revised body.
    """
    result = dict(entries)
    if REVISED in result:
        node = _node(REVISED)
        assert ast.unparse(node.args) == REVISED_SIGNATURE
        current = _source_docstring(REVISED)
        before = result[REVISED]
        assert before == current or before.split(":returns:")[0] == current.split(":returns:")[0]
        assert "``assigned_pixel_real``" in current and "``split``" in current
        result[REVISED] = current
    return result


def _current_1023_arrivals(existing=None):
    """The new public callables bound to their source docstrings.

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


def _validated_1023_arrivals(callables):
    """Remove only the exact new public callables from the frozen boundary.

    :param callables: the inventory of public callables.
    :returns: the inventory without the arrivals.
    """
    entries = _current_1023_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == ARRIVALS, sorted(found)
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
