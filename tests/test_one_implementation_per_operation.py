"""AGENTS.md rule 43's ratchet: one operation, one implementation.

The maintainer (2026-11-21, restated 2026-11-23): "modules that do the same
thing should not use their own code to do that thing and should instead use
the same code." The corollaries are one file structure, one reader for spaCR
data and one writer for saving it.

``one_implementation_registry.json`` names, for each shared operation, the
owner module(s) and the AST signatures that implement it: library calls
(import aliases resolved), method names, function names, literals or a named
detector. This test counts every signature OUTSIDE the owners, per file, and
compares with ``one_implementation_baseline.json``:

* a file whose count rises, or a file that is not in the baseline, FAILS:
  call the owner instead, or justify the site in the baseline in the same
  commit (with a reason in the commit message);
* a file whose count falls also FAILS until the baseline is lowered, so the
  ceiling only goes down and a converted site cannot silently come back.

Regenerate after converting sites (only lowers numbers in a clean tree):

    python tests/test_one_implementation_per_operation.py --write-baseline

The registry is the place to add a new shared operation. Before writing a
new helper, look it up there (rule 43: check for an existing implementation
first).
"""
from __future__ import annotations

import ast
import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("SPACR_REPO_ROOT", HERE.parent)).resolve()
REGISTRY = pathlib.Path(os.environ.get(
    "ONE_IMPL_REGISTRY", HERE / "one_implementation_registry.json"))
BASELINE = pathlib.Path(os.environ.get(
    "ONE_IMPL_BASELINE", HERE / "one_implementation_baseline.json"))

#: Generated or translated text, not code that performs operations.
SKIP_PARTS = ("spacr/qt/i18n_catalogs/", "spacr/qt/help_api_index.py")


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_zero(node):
    return isinstance(node, ast.Constant) and node.value == 0 and not isinstance(node.value, bool)


def _nonzero_test(node):
    """``x > 0``, ``x != 0`` or ``(a > 0) | (b > 0)``."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and _is_zero(node.comparators[0]):
        return isinstance(node.ops[0], (ast.Gt, ast.NotEq))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _nonzero_test(node.left) or _nonzero_test(node.right)
    return False


def _detect_cytoplasm_where(call, resolved):
    """``np.where(<nucleus> > 0, 0, <cell>)``: cell labels with nucleus pixels zeroed."""
    return (resolved in ("numpy.where",) and len(call.args) == 3
            and _is_zero(call.args[1]) and _nonzero_test(call.args[0])
            and not _is_zero(call.args[2]))


DETECTORS = {"cytoplasm_where": _detect_cytoplasm_where}


class _Scan(ast.NodeVisitor):
    def __init__(self, ops):
        self.ops = ops
        self.alias = {}
        self.hits = collections.Counter()
        self.where = collections.defaultdict(list)

    # imports ------------------------------------------------------------
    def visit_Import(self, node):
        for a in node.names:
            if a.asname:
                self.alias[a.asname] = a.name
            else:
                head = a.name.split(".")[0]
                self.alias[head] = head

    def visit_ImportFrom(self, node):
        if node.module and node.level == 0:
            for a in node.names:
                self.alias[a.asname or a.name] = f"{node.module}.{a.name}"

    def _resolve(self, dotted):
        head, _, rest = dotted.partition(".")
        base = self.alias.get(head, head)
        full = base + ("." + rest if rest else "")
        if full.startswith("np."):
            full = "numpy." + full[3:]
        return full

    def _hit(self, op, node):
        self.hits[op] += 1
        self.where[op].append(node.lineno)

    # signatures ---------------------------------------------------------
    def visit_Call(self, node):
        dotted = _dotted(node.func)
        resolved = self._resolve(dotted) if dotted else None
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None)
        for op, spec in self.ops.items():
            if resolved and resolved in spec["_calls"]:
                self._hit(op, node)
            elif isinstance(node.func, ast.Attribute) and attr in spec["_method"]:
                self._hit(op, node)
            elif attr and attr in spec["_name_calls"]:
                self._hit(op, node)
            elif spec["_detector"] and resolved and spec["_detector"](node, resolved):
                self._hit(op, node)
        self.generic_visit(node)

    def visit_Constant(self, node):
        value = node.value
        for op, spec in self.ops.items():
            if isinstance(value, float) and value in spec["_numbers"]:
                self._hit(op, node)
            elif isinstance(value, str) and value in spec["_strings"]:
                self._hit(op, node)


def load_registry(path=REGISTRY):
    ops = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["operations"]
    prepared = {}
    for name, spec in ops.items():
        if spec.get("delegated_to") and not any(
                spec.get(k) for k in ("calls", "method", "name_calls", "numbers", "strings", "detector")):
            continue
        prepared[name] = {
            "owners": tuple(spec.get("owners", ())),
            "_calls": frozenset(spec.get("calls", ())),
            "_method": frozenset(spec.get("method", ())),
            "_name_calls": frozenset(spec.get("name_calls", ())),
            "_numbers": frozenset(float(n) for n in spec.get("numbers", ())),
            "_strings": frozenset(spec.get("strings", ())),
            "_detector": DETECTORS[spec["detector"]] if spec.get("detector") else None,
        }
    return prepared


def _owned(rel, owners):
    return any(rel == o or (o.endswith("/") and rel.startswith(o)) for o in owners)


def count_sites(root=ROOT, registry=REGISTRY, with_lines=False):
    """``{operation: {file: count}}`` of signatures outside each operation's owners."""
    ops = load_registry(registry)
    root = pathlib.Path(root)
    counts = {op: {} for op in ops}
    lines = {op: {} for op in ops}
    for path in sorted((root / "spacr").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(part in rel for part in SKIP_PARTS):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        scan = _Scan(ops)
        scan.visit(tree)
        for op, n in scan.hits.items():
            if not _owned(rel, ops[op]["owners"]):
                counts[op][rel] = n
                lines[op][rel] = scan.where[op]
    return (counts, lines) if with_lines else counts


def compare(counts, baseline):
    """Messages for every file above, absent from, or below the baseline."""
    problems = []
    for op in sorted(set(counts) | set(baseline)):
        now = counts.get(op, {})
        then = baseline.get(op, {})
        for rel in sorted(set(now) | set(then)):
            n, b = now.get(rel, 0), then.get(rel, 0)
            if n > b:
                problems.append(
                    f"[{op}] {rel}: {n} site(s), baseline {b}. Rule 43: call the owner "
                    f"named in one_implementation_registry.json instead of implementing "
                    f"'{op}' again.")
            elif n < b:
                problems.append(
                    f"[{op}] {rel}: {n} site(s), baseline {b}. Good - lower the baseline "
                    f"in this commit (--write-baseline) so it cannot creep back.")
    return problems


def test_one_implementation_per_operation():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["counts"]
    problems = compare(count_sites(), baseline)
    assert not problems, "\n".join(problems)


def test_registry_owners_exist():
    for op, spec in load_registry().items():
        for owner in spec["owners"]:
            assert (ROOT / owner).exists(), f"[{op}] owner {owner} does not exist"


def test_ratchet_detects_a_new_duplicate(tmp_path):
    """A new module that measures with regionprops itself, writes its own
    TIFFs and derives its own cytoplasm is caught on every count."""
    pkg = tmp_path / "spacr"
    pkg.mkdir()
    (pkg / "measure.py").write_text(
        "from skimage.measure import regionprops_table\n"
        "def owner(l):\n    return regionprops_table(l)\n")
    (pkg / "new_module.py").write_text(
        "import numpy as np\n"
        "import tifffile as tf\n"
        "from skimage import measure as skm\n"
        "def my_measure(cells, nuclei):\n"
        "    cyto = np.where(nuclei > 0, 0, cells)\n"
        "    tf.imwrite('x.tif', cyto)\n"
        "    return skm.regionprops_table(cells), 1.4826\n")
    counts = count_sites(tmp_path)
    assert counts["object_regionprops_table"] == {"spacr/new_module.py": 1}
    assert counts["tiff_write"] == {"spacr/new_module.py": 1}
    assert counts["cytoplasm_derivation"] == {"spacr/new_module.py": 1}
    assert counts["robust_scale_constant"] == {"spacr/new_module.py": 1}
    assert compare(counts, {}), "a site absent from the baseline must fail"
    assert not compare(counts, counts)


def _write_baseline():
    counts = count_sites()
    old = json.loads(BASELINE.read_text(encoding="utf-8"))["counts"] if BASELINE.exists() else None
    if old is not None:
        rises = [p for p in compare(counts, old) if "Rule 43" in p]
        if rises:
            sys.exit("refusing to raise the baseline:\n" + "\n".join(rises))
    total = {op: sum(v.values()) for op, v in counts.items()}
    BASELINE.write_text(json.dumps({
        "_comment": "Generated by --write-baseline. Counts may only go down.",
        "totals": total, "counts": counts}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(total, indent=1, sort_keys=True))


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        _write_baseline()
