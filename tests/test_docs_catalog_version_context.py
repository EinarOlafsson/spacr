"""Catalog-only changes must not rewrite source pages with no API consumer."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "docs/source/conf.py"


@dataclass(frozen=True)
class Script:
    filename: str
    priority: int = 500
    attributes: dict = field(default_factory=dict)

    def __init__(self, filename, priority=500, **attributes):
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "attributes", attributes)


def configured_callbacks():
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    names = {"setup", "_omit_unused_viewcode_catalog_version"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "_qualify_helper_annotations": lambda *args: None,
        "_skip_implementation_data": lambda *args: None,
        "_nested_helper_docs": SimpleNamespace(register_sphinx_directive=lambda app: None),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(CONF), "exec"), namespace)
    callbacks = []
    app = SimpleNamespace(
        config=SimpleNamespace(spacr_source_api_projection=frozenset()),
        add_config_value=lambda *args: None,
        connect=lambda event, callback: callbacks.append(callback) if event == "html-page-context" else None,
    )
    namespace["setup"](app)
    return app, callbacks


def apply_context(context, *, page="_modules/spacr/_gc", template="page.html", doctree=None):
    app, callbacks = configured_callbacks()
    for callback in callbacks:
        callback(app, page, template, context, doctree)
    return context


def context(version="aaaaaaaaaaaaaaaa", body="<h1>Source code</h1><pre>pass</pre>"):
    return {"body": body, "script_files": [
        Script("_static/documentation_options.js"),
        Script("_static/api_i18n.js", priority=500, **{
            "data-api-catalog-version": version,
            "data-api-language": "all",
            "data-guide-languages": "sv de es pt fr is zh_CN ko hi",
        }),
    ]}


def test_catalog_only_change_does_not_change_source_page_scripts():
    first, second = context("aaaaaaaaaaaaaaaa"), context("bbbbbbbbbbbbbbbb")
    apply_context(first)
    apply_context(second)
    assert first == second
    assert "data-api-catalog-version" not in first["script_files"][1].attributes
    assert first["script_files"][1].attributes["data-guide-languages"] == "sv de es pt fr is zh_CN ko hi"


def test_shared_script_objects_do_not_lose_api_page_freshness():
    source = context()
    shared = source["script_files"]
    api = {"body": '<dl class="py function"><dt id="spacr.demo"></dt></dl>', "script_files": shared}
    apply_context(source)
    apply_context(api, page="api/spacr/index", doctree=object())
    assert source["script_files"] is not shared
    assert api["script_files"] is shared
    assert api["script_files"][1].attributes["data-api-catalog-version"] == "aaaaaaaaaaaaaaaa"
    assert source["script_files"][0] is shared[0]


@pytest.mark.parametrize("page,template,body,doctree", [
    ("api/spacr/index", "page.html", "<h1>API</h1>", None),
    ("_modules/spacr/api/example", "page.html", "<pre>pass</pre>", None),
    ("features", "page.html", "<p>Guide</p>", object()),
    ("_modules/custom", "page.html", '<dl class="py function"><dt id="demo"></dt></dl>', None),
    ("_modules/custom", "page.html", '<DL\nclass="py function"><dt id="demo"></dt></DL>', None),
    ("_modules/custom", "custom.html", "<pre>pass</pre>", None),
    ("_modules/custom", "page.html", "<pre>pass</pre>", object()),
])
def test_actual_or_uncertain_api_consumers_keep_version(page, template, body, doctree):
    first, second = context("aaaaaaaaaaaaaaaa", body), context("bbbbbbbbbbbbbbbb", body)
    apply_context(first, page=page, template=template, doctree=doctree)
    apply_context(second, page=page, template=template, doctree=doctree)
    assert first != second
    assert first["script_files"][1].attributes["data-api-catalog-version"] == "aaaaaaaaaaaaaaaa"


def test_normal_sphinx_script_objects_are_copied_without_mutation():
    html = pytest.importorskip("sphinx.builders.html")
    script = html._JavaScript("_static/api_i18n.js", **{
        "data-api-catalog-version": "aaaaaaaaaaaaaaaa", "data-guide-languages": "sv de",
    })
    value = {"body": "<pre>pass</pre>", "script_files": [script]}
    apply_context(value)
    assert value["script_files"][0] is not script
    assert value["script_files"][0].attributes == {"data-guide-languages": "sv de"}
    assert script.attributes["data-api-catalog-version"] == "aaaaaaaaaaaaaaaa"
