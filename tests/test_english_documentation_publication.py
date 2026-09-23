"""English publication still rejects source drift and leaves strict audits intact."""

import importlib
import json
import os
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_docs_version_follows_checkout_even_when_installed_metadata_is_old(tmp_path, monkeypatch):
    import importlib.metadata
    from docs_version import source_version

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.5.0.8")
    (tmp_path / "spacr").mkdir()
    (tmp_path / "setup.py").write_text('VERSION = "1.5.1.0"\nraise RuntimeError("must not execute setup")\n')
    (tmp_path / "spacr/_version.py").write_text('__version__ = "1.5.1.0"\n')
    assert source_version(tmp_path) == "1.5.1.0"
    (tmp_path / "spacr/_version.py").write_text('__version__ = "1.5.0.9"\n')
    with pytest.raises(ValueError, match="versions disagree"):
        source_version(tmp_path)


@pytest.mark.parametrize("module,extractor", [
    ("build_documentation_i18n", "public_docstrings"),
    ("build_i18n_catalogs", "canonical_sources"),
])
@pytest.mark.parametrize("flag,english", [("--audit", False), ("--audit-english", True)])
def test_cli_selects_audit_scope_and_propagates_failure(monkeypatch, module, extractor, flag, english):
    builder = importlib.import_module(module)
    sources = {"source": "Current English"}
    monkeypatch.setattr(builder, extractor, lambda: sources)
    seen = []

    def audit(actual, languages):
        seen.append((actual, tuple(languages)))
        return 1

    monkeypatch.setattr(builder, "audit", audit)
    monkeypatch.setattr(sys, "argv", [module, flag])
    assert builder.main() == 1
    assert seen == [(sources, () if english else tuple(builder.MODEL_SPECS))]


def test_english_api_audit_rejects_changed_source_even_without_locales(tmp_path, monkeypatch):
    import build_documentation_i18n as builder

    source = "Return the measured intensity."
    sources = {"spacr.example": source}
    readme = tmp_path / "README.rst"
    readme.write_text("English readme.\n")
    monkeypatch.setattr(builder, "README_SOURCE", readme)
    monkeypatch.setattr(builder, "API_DIR", tmp_path)
    record = {
        "text": source,
        "source_sha256": builder._source_hash(source),
        "source_blocks_sha256": builder._source_block_hashes(source),
    }
    path = tmp_path / "en.json"
    path.write_text(json.dumps({"schema": 2, "symbols": {"spacr.example": record}}))
    assert builder.audit(sources, ()) == 0
    assert builder.audit({"spacr.example": "Return a different measurement."}, ()) == 1
    assert builder.audit({**sources, "spacr.new_function": "New function."}, ()) == 1


def test_english_runtime_audit_rejects_changed_source_even_without_locales(tmp_path, monkeypatch):
    import build_i18n_catalogs as builder

    sources = {
        "setting_labels": {}, "setting_tooltips": {}, "categories": {},
        "ui": {"Current caption": "Current caption"}, "module_summaries": {},
    }
    monkeypatch.setattr(builder, "CATALOG_DIR", tmp_path)
    values = {
        "SETTING_LABELS": sources["setting_labels"],
        "SETTING_TOOLTIPS": sources["setting_tooltips"],
        "CATEGORY_SOURCES": frozenset(sources["categories"]),
        "UI_SOURCES": frozenset(sources["ui"]),
        "MODULE_SUMMARIES": sources["module_summaries"],
        "SOURCE_HASHES": builder._source_hashes(sources),
    }
    (tmp_path / "en.py").write_text("\n".join(f"{name} = {value!r}" for name, value in values.items()))
    assert builder.audit(sources, ()) == 0
    sources["ui"] = {"Changed caption": "Changed caption"}
    assert builder.audit(sources, ()) == 1


@pytest.mark.skipif(
    not os.environ.get("SPACR_DOCS_BUILT") or
    os.environ.get("SPACR_DOCS_API_LANGUAGE") != "english",
    reason="requires the current English publication build",
)
def test_built_english_publication_declares_its_mode_and_current_api():
    from html.parser import HTMLParser
    import build_documentation_i18n as builder

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.api = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "script" and "api_i18n.js" in attrs.get("src", ""):
                self.api.append(attrs)

    root = Path(os.environ.get("SPACR_DOCS_BUILD_DIR", TOOLS.parent / "docs/_build/html"))
    from docs_version import source_version
    assert f'spaCR {source_version(TOOLS.parent)} documentation' in (root / "index.html").read_text()
    for module in ("image_quality", "host_pathogen", "object", "timeflows_model"):
        page = root / "api/spacr" / module / "index.html"
        parser = Scripts()
        parser.feed(page.read_text())
        assert len(parser.api) == 1
        assert parser.api[0]["data-api-language"] == "english"
        assert parser.api[0]["data-api-catalog-version"]
    assert (root / "_static/api_i18n.js").read_bytes() == (
        TOOLS.parent / "docs/source/_static/api_i18n.js"
    ).read_bytes()
    assert (root / "_static/i18n/api/en.json").read_bytes() == (builder.API_DIR / "en.json").read_bytes()
