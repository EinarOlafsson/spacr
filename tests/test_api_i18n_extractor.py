"""Focused contracts for the source-only AutoAPI localization extractor."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

builder = importlib.import_module("build_documentation_i18n")


# Review-set fingerprints make the coverage contract independent of the
# investigator's /tmp report while still proving that all exact audited ids
# remain represented.  A source/API change requires regenerating and reviewing
# that report before deliberately updating either digest.
_NEW_VISIBLE_DIGEST = (
    "dfb61ead393074e0d5250178e992630f7979ac835a9cae0d88e438fe93cfa554"
)
_ALIASES_DIGEST = (
    "5167459a662cc68d3de274d216297020ba159155bad4e9e8af8e751e69cdba66"
)


def _sha256_lines(lines) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _source_nodes():
    """Yield public source nodes using the same package-name convention."""
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        if any(
            part in {"tests", "__pycache__", "backup_icons", "i18n_catalogs"}
            for part in path.parts
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = builder._module_name(path)
        yield path, module, tree


def _visible_special_members() -> set[str]:
    keys: set[str] = set()
    for path, module, tree in _source_nodes():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    builder._is_visible_function_name(
                        node.name, module_is_package=path.name == "__init__.py",
                    )
                    and builder._clean_doc(node)
                    and node.name.startswith("__")
                ):
                    keys.add(f"{module}.{node.name}")
                continue
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for child in node.body:
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue
                if (
                    child.name.startswith("__")
                    and builder._is_visible_function_name(
                        child.name, module_is_package=False,
                    )
                    and builder._clean_doc(child)
                ):
                    keys.add(f"{module}.{node.name}.{child.name}")
    return keys


def _visible_assignment_docs() -> set[str]:
    keys: set[str] = set()
    for _path, module, tree in _source_nodes():
        keys.update(builder._additional_assignment_docs(tree.body, module))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                owner = f"{module}.{node.name}"
                keys.update(builder._additional_assignment_docs(node.body, owner))
    return keys


def test_public_docstrings_matches_reviewed_visible_coverage():
    docs = builder.public_docstrings()
    dunders = _visible_special_members()
    assignments = _visible_assignment_docs()

    # The 80 omissions in the audited pages are precisely the documented
    # special members and PEP-258/value attributes discovered from source.
    assert len(dunders) == 64
    assert len(assignments) == 16
    assert _sha256_lines(
        [*(f"new_dunder\0{key}" for key in dunders),
         *(f"new_constant_attribute\0{key}" for key in assignments)]
    ) == _NEW_VISIBLE_DIGEST
    assert dunders | assignments <= docs.keys()

    # 6234 canonical bodies plus 119 exact materialized aliases. The 29 shared
    # bodies are the shared live-view contract — `spacr.qt.widgets.
    # preview_contract` and its members (14), and the run/cancel/blocked-reason
    # API the four preview panels gained when they adopted it (15). The final
    # body is Measure's now-public settings-seeding contract; the latest one
    # documents the class-folder naming helper. Catalog regeneration must
    # include all 6,353 records.
    assert len(docs) - len(builder.API_DOC_ALIASES) == 6234
    assert len(docs) == 6353
    assert set(builder.API_DOC_ALIASES) <= docs.keys()

    # These are the only substantive audit bodies intentionally unresolved:
    # one external stdlib inheritance and two source-less Sphinx markers.
    assert "spacr.logging_util.LevelSetFilter.filter" not in docs
    assert "spacr.qt.widgets.gate_spec.Gate.columns" not in docs
    assert "spacr.qt.widgets.gate_spec.Gate.kind" not in docs


def test_documented_dunders_exclude_init_private_and_package_forwarders():
    docs = builder.public_docstrings()

    assert "spacr.version.__getattr__" in docs
    assert "spacr.qt.theme.__getattr__" in docs
    assert "spacr.active_learning.StoppingVerdict.__bool__" in docs
    assert not any(key.endswith(".__init__") for key in docs)
    assert "spacr.illumination._source_folders" not in docs
    # Package-level lazy forwarding hooks are not emitted in AutoAPI pages.
    assert "spacr.__getattr__" not in docs
    assert "spacr.qt.widgets.__getattr__" not in docs


def test_assignment_docs_are_ast_source_text_without_show_value_artifact():
    docs = builder.public_docstrings()
    assignment_keys = _visible_assignment_docs()

    assert len(assignment_keys) == 16
    assert assignment_keys <= docs.keys()
    assert all("Show Value" not in docs[key] for key in assignment_keys)
    assert docs["spacr.batch_correction.METHODS"] == (
        "Supported correction methods."
    )
    assert docs["spacr.anndata_export.ANNDATA_MISSING_MESSAGE"].startswith(
        "Exporting to AnnData (.h5ad) needs the optional `anndata` extra"
    )
    assert docs[
        "spacr.qt.widgets.graph_builder.GraphCanvas.RESCALE_ON_FILTER"
    ].startswith("The chart itself: a spec in, a faceted figure out")


def test_exact_alias_map_and_manifest_records_are_identical():
    docs = builder.public_docstrings()
    aliases = builder.API_DOC_ALIASES

    assert len(aliases) == 119
    assert _sha256_lines(
        f"{alias}\0{canonical}" for alias, canonical in aliases.items()
    ) == _ALIASES_DIGEST
    assert not (set(aliases) & set(aliases.values()))
    assert all(docs[alias] == docs[canonical]
               for alias, canonical in aliases.items())
    assert "spacr.logging_util.LevelSetFilter.filter" not in aliases

    english = builder._english_manifest(docs)["symbols"]
    for alias, canonical in aliases.items():
        assert english[alias]["alias_of"] == canonical
        assert {
            key: value for key, value in english[alias].items()
            if key != "alias_of"
        } == english[canonical]


def test_localized_manifest_materializes_alias_translation(tmp_path, monkeypatch):
    docs = builder.public_docstrings()
    canonical = "spacr.layers.Layer.ndim"
    alias = "spacr.layers.ImageLayer.ndim"
    translations = {key: f"localized:{index}" for index, key in enumerate(docs)}
    # Alias model output must never win: its record references the one
    # canonical translation and all identical freshness hashes.
    translations[canonical] = "localized canonical body"
    translations[alias] = "incorrect duplicate decode"
    monkeypatch.setattr(builder, "API_DIR", tmp_path)

    builder.write_language(docs, "de", translations)
    payload = json.loads((tmp_path / "de.json").read_text(encoding="utf-8"))
    symbols = payload["symbols"]
    assert symbols[alias]["alias_of"] == canonical
    assert symbols[alias]["text"] == "localized canonical body"
    assert {
        key: value for key, value in symbols[alias].items()
        if key != "alias_of"
    } == symbols[canonical]
