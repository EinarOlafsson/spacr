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

