"""Arrival contract of the public callables landing L3 adds; stdlib only.

merge1 (ops_setting, the OPS default table), merge5 (build_cellpose_model,
one Cellpose builder), merge9b (batch_mean_sd, one per-well centre-scale) and
cc-gaps with nuisance-reg (dna_phases, read_table_columns, the stage bundle
and the nuisance regression) arrive through
tests/data/release_contracts/l3_api_arrivals_2026-11-24.json, each body bound
to its source docstring (read with ``ast``, nothing imported from spacr).
"""
import ast
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTH = ROOT / "tests/data/release_contracts/l3_api_arrivals_2026-11-24.json"
PUBLIC = frozenset({
    "spacr.ops_settings.ops_setting",
    "spacr.utils.build_cellpose_model",
    "spacr.ops_tvn.batch_mean_sd",
    "spacr.measure.dna_phases",
    "spacr.tabular.read_table_columns",
    "spacr.ops_cell_cycle.stage_bundle",
    "spacr.ops_cell_cycle.import_stage_bundle",
    "spacr.ops_tvn.nuisance_kinds",
    "spacr.ops_tvn.nuisance_source_columns",
    "spacr.ops_tvn.nuisance_covariates",
    "spacr.ops_tvn.regress_nuisance",
})
#: Nested helpers documented on the API pages but not public callables.
NESTED = frozenset({"spacr.utils.build_cellpose_model.construct"})
#: Documented module constants (merge9: the one MAD-to-SD constant).
CONSTANTS = frozenset({"spacr.batch_correction.MAD_SCALE_ROUNDED"})
ARRIVALS = PUBLIC | NESTED | CONSTANTS


def _source_docstring(symbol):
    """The cleaned docstring of ``spacr.<module>.<function>[.<nested function>]``.

    A documented constant's docstring is the string literal after its
    assignment.
    """
    _, module, *path = symbol.split(".")
    node = ast.parse((ROOT / "spacr" / (module + ".py")).read_text(encoding="utf-8"))
    if symbol in CONSTANTS:
        body = node.body
        index = next(i for i, item in enumerate(body)
                     if isinstance(item, ast.Assign)
                     and any(getattr(t, "id", None) == path[0] for t in item.targets))
        return inspect.cleandoc(body[index + 1].value.value)
    for name in path:
        scope = node.body if isinstance(node, ast.Module) else list(ast.walk(node))
        node = next(item for item in scope
                    if isinstance(item, ast.FunctionDef) and item.name == name)
    return inspect.cleandoc(ast.get_docstring(node, clean=False))


def _current_l3_api(existing=None):
    """The new public symbols bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_l3_arrivals(callables):
    """Remove only the exact new public callables from the frozen boundary."""
    entries = _current_l3_api()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == PUBLIC
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
