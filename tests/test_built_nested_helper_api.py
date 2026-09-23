"""Feature 411: extract and render identical anchors through actual Sphinx."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
helpers = importlib.import_module("nested_helper_docs")
builder = importlib.import_module("build_documentation_i18n")
TEMPLATES = ROOT / "docs/source/_autoapi_templates"
MODULES = {"spacr.example", "spacr._v1_v2_bridge"}
SOURCE = '''"""Example API module."""
def outer(flag):
    """Choose a local policy."""
    if flag:
        def choose(value):
            """Return the first value."""
            return value
    else:
        def choose(value, fallback=None):
            """Return the fallback when no value is supplied."""
            return fallback if value is None else value
    def _hidden(value, marker=None, /, sample='a,b', *, match=value is not None,
                scores={name: n for n, name in enumerate(('x', 'y'))},
                callback=lambda left, right: left + right, **options):
        """Return the unchanged value.

        :param value: The value to return.
        :returns: The input unchanged.
        """
        return value
def _private_parent():
    """This private top-level function must not become API."""
    def leaf():
        """Return the private parent's value."""
class Public:
    """A public class."""
    def method(self):
        """Create a local helper."""
        def leaf():
            """Return the method's value."""
'''
HIDDEN_SOURCE = '''"""This private module's original prose must stay hidden."""
def public_top():
    """This top-level API was hidden with its private module."""
    def _leaf():
        """Return the compatibility value."""
'''


@pytest.fixture(scope="module")
def fixture_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("feature-411-source-")
    package = root / "spacr"
    package.mkdir()
    (package / "__init__.py").write_text('"""spaCR fixture."""\n', encoding="utf-8")
    (package / "example.py").write_text(SOURCE, encoding="utf-8")
    (package / "_v1_v2_bridge.py").write_text(HIDDEN_SOURCE, encoding="utf-8")
    # The real spaCR module template retains these curated workflow links.
    for name in ("api", "core", "measure", "deep_spacr", "sequencing", "ml", "artifacts", "settings"):
        (package / f"{name}.py").write_text(f'"""{name} fixture."""\n', encoding="utf-8")
    return root


def _extract(monkeypatch, root, enabled):
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder, "API_DOC_ALIASES", {})
    monkeypatch.setattr(helpers, "ENABLED_MODULES", frozenset(enabled))
    return builder.public_docstrings()


def test_extractor_adds_only_the_shared_inventory_and_preserves_existing_docs(
    monkeypatch, fixture_source,
):
    before = _extract(monkeypatch, fixture_source, ())
    after = _extract(monkeypatch, fixture_source, MODULES)
    canonical = helpers.entries(helpers.inventory(fixture_source, ignore_patterns=builder.AUTOAPI_IGNORE))
    expected = {entry.qualified_key: entry.docstring for entry in canonical}
    assert len(expected) == 5
    assert set(after) - set(before) == set(expected)
    assert {key: after[key] for key in before} == before
    assert {key: after[key] for key in expected} == expected
    assert "spacr.example._private_parent" not in after
    assert "spacr._v1_v2_bridge.public_top" not in after


def test_empty_rollout_does_not_scan_or_touch_any_catalog(monkeypatch):
    paths = sorted((ROOT / "docs/source/_static/i18n/api").glob("*.json"))
    assert len(paths) == 10
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    previously_enabled = {entry.qualified_key for entry in helpers.active_entries(
        ROOT, ignore_patterns=builder.AUTOAPI_IGNORE,
    )}
    monkeypatch.setattr(helpers, "ENABLED_MODULES", frozenset())

    def forbidden(*args, **kwargs):
        raise AssertionError("An empty rollout must not scan nested sources")

    monkeypatch.setattr(helpers, "inventory", forbidden)
    measured = builder.public_docstrings()
    assert measured
    manifest = json.loads((ROOT / "docs/source/_static/i18n/api/en.json").read_text())
    assert set(measured) == set(manifest["symbols"]) - previously_enabled
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_sphinx_and_extractor_import_the_same_switch(monkeypatch, fixture_source):
    pytest.importorskip("jinja2")
    from jinja2 import Environment
    import build_module_workflows

    def fixture_workflows(env, root):
        assert root == fixture_source
        env.globals["spacr_workflow_includes"] = {}

    monkeypatch.setattr(build_module_workflows, "prepare_jinja", fixture_workflows)

    tree = ast.parse((ROOT / "docs/source/conf.py").read_text())
    hook = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "autoapi_prepare_jinja_env")
    namespace = {"_nested_helper_docs": helpers, "_SOURCE_ROOT": fixture_source,
                 "autoapi_ignore": builder.AUTOAPI_IGNORE}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "real-conf-hook", "exec"), namespace)
    environment = Environment()
    _extract(monkeypatch, fixture_source, MODULES)
    namespace["autoapi_prepare_jinja_env"](environment)
    configured = {
        entry.qualified_key for entries in environment.globals["spacr_nested_helpers"].values()
        for entry in entries
    }
    assert configured == {entry.qualified_key for entry in helpers.active_entries(
        fixture_source, ignore_patterns=builder.AUTOAPI_IGNORE,
    )}
    assert len(configured) == 5


@pytest.fixture(scope="module")
def built_site(fixture_source):
    return _build_site(fixture_source, MODULES, "enabled")


def _build_site(fixture_source, enabled, label, *, inventory_root=None):
    pytest.importorskip("sphinx")
    pytest.importorskip("autoapi")
    source = fixture_source / f"docs-{label}"
    source.mkdir()
    templates = source / "templates"
    shutil.copytree(TEMPLATES, templates)
    # The top-level repository template includes the complete app registry.
    # Only that navigation wrapper is replaced; module/package/helper templates
    # and the actual conf.py hook bodies are used verbatim.
    (templates / "index.rst").write_text(
        "API\n===\n\n.. toctree::\n\n   spacr/index\n", encoding="utf-8",
    )
    conf_tree = ast.parse((ROOT / "docs/source/conf.py").read_text())
    hooks = "\n\n".join(ast.unparse(node) for node in conf_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "autoapi_prepare_jinja_env", "_skip_implementation_data", "setup",
        })
    (source / "conf.py").write_text(
        f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'tools')!r})\n"
        "import nested_helper_docs as _nested_helper_docs\n"
        "import api_visibility as _api_visibility\n"
        # This synthetic package models nested functions, not GUI routes.
        # The real workflow hook has its own source/map/build tests.
        "import build_module_workflows\n"
        "build_module_workflows.prepare_jinja = lambda env, root: env.globals.update(spacr_workflow_includes={})\n"
        f"_nested_helper_docs.ENABLED_MODULES = frozenset({sorted(enabled)!r})\n"
        "spacr_nested_helper_modules = tuple(sorted(_nested_helper_docs.ENABLED_MODULES))\n"
        "spacr_explicit_api_modules = tuple(sorted(_api_visibility.EXPLICIT_MODULES))\n"
        f"_SOURCE_ROOT = Path({str(inventory_root or fixture_source)!r})\n"
        "project = 'Feature 411 fixture'\nextensions = ['sphinx.ext.napoleon', 'autoapi.extension', 'sphinx_design']\n"
        f"autoapi_dirs = [{str(fixture_source / 'spacr')!r}]\n"
        f"autoapi_ignore = {list(builder.AUTOAPI_IGNORE)!r}\n"
        "autoapi_root = 'api'\nautoapi_template_dir = 'templates'\n"
        "autoapi_options = ['members', 'show-inheritance', 'show-module-summary']\n"
        "autoapi_python_class_content = 'both'\nautoapi_keep_files = True\n"
        "autoapi_member_order = 'groupwise'\nhtml_theme = 'furo'\n"
        "napoleon_use_rtype = False\n"
        "exclude_patterns = ['templates/**']\n" + hooks + "\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Fixture\n=======\n\n.. toctree::\n\n   api/index\n", encoding="utf-8",
    )
    output = fixture_source / f"html-{label}"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-E", "-b", "html", "--keep-going",
         str(source), str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    return output


def test_empty_rollout_builds_without_helpers_and_keeps_private_api_hidden(fixture_source):
    output = _build_site(fixture_source, (), "empty")
    page = (output / "api/spacr/example/index.html").read_text()
    assert 'id="spacr.example.outer"' in page
    assert 'id="spacr.example.Public.method"' in page
    assert 'id="spacr.example.outer.choose"' not in page
    assert 'id="spacr.example._private_parent"' not in page
    assert not (output / "api/spacr/_v1_v2_bridge/index.html").exists()


def test_changing_only_the_rollout_invalidates_sphinx_incremental_state(fixture_source):
    output = _build_site(fixture_source, (), "incremental")
    source = fixture_source / "docs-incremental"
    config = source / "conf.py"
    before_sources = {path: path.read_bytes() for path in (fixture_source / "spacr").glob("*.py")}
    text = config.read_text()
    old = "_nested_helper_docs.ENABLED_MODULES = frozenset([])"
    assert text.count(old) == 1
    config.write_text(text.replace(old,
        f"_nested_helper_docs.ENABLED_MODULES = frozenset({sorted(MODULES)!r})"),
        encoding="utf-8")
    # No -E: the registered config value, not a clean build or source change,
    # must invalidate AutoAPI's cached empty selection.
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(source), str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    assert before_sources == {path: path.read_bytes() for path in before_sources}
    page = (output / "api/spacr/example/index.html").read_text()
    assert 'id="spacr.example.outer.choose"' in page
    hidden = (output / "api/spacr/_v1_v2_bridge/index.html").read_text()
    assert 'id="spacr._v1_v2_bridge.public_top._leaf"' in hidden


def test_actual_sphinx_html_and_objects_inventory_have_exactly_the_helper_keys(
    monkeypatch, fixture_source, built_site,
):
    before = _extract(monkeypatch, fixture_source, ())
    after = _extract(monkeypatch, fixture_source, MODULES)
    keys = set(after) - set(before)
    assert len(keys) == 5
    inventory = (built_site / "objects.inv").read_bytes().split(b"\n", 4)[4]
    inventory_lines = zlib.decompress(inventory).decode().splitlines()
    for key in keys:
        module = next(name for name in MODULES if key.startswith(name + "."))
        page = built_site / "api" / Path(*module.split(".")) / "index.html"
        text = page.read_text()
        assert text.count(f'id="{key}"') == 1
        matches = [line for line in inventory_lines if line.startswith(key + " ")]
        assert len(matches) == 1, matches
        assert matches[0].split()[1] == "py:function"
    hidden = (built_site / "api/spacr/_v1_v2_bridge/index.html").read_text()
    assert 'id="spacr._v1_v2_bridge.public_top._leaf"' in hidden
    assert 'id="spacr._v1_v2_bridge.public_top"' not in hidden
    assert "original prose must stay hidden" not in hidden
    page = (built_site / "api/spacr/example/index.html").read_text()
    assert 'id="spacr.example._private_parent.leaf"' in page
    assert 'id="spacr.example._private_parent"' not in page
    assert "Return the first value." in page
    assert "Return the fallback when no value is supplied." in page


def test_real_google_style_helpers_render_both_parameter_contracts(tmp_path):
    pytest.importorskip("bs4")
    from bs4 import BeautifulSoup

    root = tmp_path / "source"
    package = root / "spacr"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""spaCR fixture."""\n', encoding="utf-8")
    for name in ("api", "core", "measure", "deep_spacr", "sequencing", "ml", "artifacts", "settings"):
        (package / f"{name}.py").write_text(f'"""{name} fixture."""\n', encoding="utf-8")
    tree = ast.parse((ROOT / "spacr/io.py").read_text(encoding="utf-8"))
    parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_check_masks")
    module = ast.Module(body=[
        ast.Expr(value=ast.Constant(value="Real Google-style source fixture.")), parent,
    ], type_ignores=[])
    (package / "io.py").write_text(ast.unparse(module), encoding="utf-8")
    output = _build_site(root, {"spacr.io"}, "google")
    page = BeautifulSoup((output / "api/spacr/io/index.html").read_text(), "html.parser")
    signature = page.find(id="spacr.io._check_masks.needs_processing")
    assert signature is not None
    body = signature.parent.find("dd", recursive=False)
    parameters = body.select("dl.field-list")
    assert parameters, "Google Args must render as a real parameter field list"
    text = " ".join(body.get_text(" ", strip=True).split())
    assert "filename" in text
    assert "validated by its header and length" in text
    assert "generated again" in text


def test_source_default_expressions_survive_the_sphinx_signature_parser(built_site):
    pytest.importorskip("bs4")
    from bs4 import BeautifulSoup

    page = BeautifulSoup((built_site / "api/spacr/example/index.html").read_text(), "html.parser")
    signature = page.find(id="spacr.example.outer._hidden")
    assert signature is not None
    defaults = [node.get_text() for node in signature.select(".default_value")]
    expected = [
        "None", "'a,b'", "value is not None",
        "{name: n for n, name in enumerate(('x', 'y'))}",
        "lambda left, right: left + right",
    ]
    # Python 3.10 unparses the tuple target as ``for (n, name)``; 3.12
    # omits those optional parentheses. Compare the active interpreter's
    # canonical source text, while still requiring every displayed value.
    assert defaults == [ast.unparse(ast.parse(value, mode="eval").body) for value in expected]
    assert "/" in signature.get_text() and "**options" in signature.get_text()
    assert "spacr-helper-default" not in str(signature)


@pytest.mark.skipif(os.environ.get("SPACR_NESTED_HELPER_CORPUS") != "1",
                    reason="Opt-in preflight of not-yet-selected real helper documents")
def test_real_helper_corpus_builds_in_an_isolated_english_fixture(tmp_path):
    """Check future helper rendering without activating or translating a slice.

    AutoAPI parses inert module stubs; the shared inventory supplies the actual
    source docstrings and signatures. The switch is changed only in the child
    Sphinx process.
    """
    pytest.importorskip("bs4")
    from bs4 import BeautifulSoup

    definitions = helpers.inventory(ROOT, ignore_patterns=builder.AUTOAPI_IGNORE)
    modules = {definition.module for definition in definitions
               if not definition.ignored_by}
    entries = helpers.entries(definitions, modules=modules)
    assert entries and modules
    original_switch = helpers.ENABLED_MODULES
    root = tmp_path / "corpus"
    package = root / "spacr"
    package.mkdir(parents=True)
    for name in ("api", "core", "measure", "deep_spacr", "sequencing", "ml", "artifacts", "settings"):
        (package / f"{name}.py").write_text(f'"""{name} fixture."""\n', encoding="utf-8")
    paths = {definition.path for entry in entries for definition in entry.definitions}
    for relative in sorted(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Source-only corpus fixture."""\n', encoding="utf-8")
    for directory in [package, *(path for path in package.rglob("*") if path.is_dir())]:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text('"""Source-only package fixture."""\n', encoding="utf-8")

    output = _build_site(root, modules, "corpus", inventory_root=ROOT)
    payload = (output / "objects.inv").read_bytes().split(b"\n", 4)[4]
    inventory_keys = {line.split()[0] for line in zlib.decompress(payload).decode().splitlines()
                      if line.split()[1] == "py:function"}
    assert inventory_keys == {entry.qualified_key for entry in entries}
    pages = {module: BeautifulSoup(
        (output / "api" / Path(*module.split(".")) / "index.html").read_text(), "html.parser",
    ) for module in modules}
    signature_count = default_count = 0
    for entry in entries:
        anchors = pages[entry.module].find_all(id=entry.qualified_key)
        assert len(anchors) == 1, entry.qualified_key
        signatures = anchors[0].parent.find_all("dt", recursive=False)
        assert len(signatures) == len(entry.signatures), entry.qualified_key
        for original, displayed in zip(entry.signatures, signatures):
            # Derive expected values independently from the original signature,
            # not from the rendering adapter whose output this verifies.
            arguments = ast.parse(f"def {original}: pass").body[0].args
            defaults = [ast.unparse(value) for value in
                        [*arguments.defaults, *arguments.kw_defaults] if value is not None]
            rendered = [node.get_text() for node in displayed.select(".default_value")]
            assert rendered == defaults, (entry.qualified_key, original, rendered, defaults)
            signature_count += 1
            default_count += len(defaults)
    assert helpers.ENABLED_MODULES == original_switch
    assert default_count > 0
    print(f"Isolated English preflight: {len(entries)} anchors in {len(modules)} modules, "
          f"{signature_count} signatures and {default_count} exact source defaults; "
          "no production rollout or translation claim.")


def test_language_switch_targets_the_built_helper_not_its_parent(
    monkeypatch, fixture_source, built_site,
):
    spec = importlib.util.spec_from_file_location(
        "_feature_411_frontend_fixture", ROOT / "tests/test_api_i18n_frontend.py",
    )
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)

    docs = _extract(monkeypatch, fixture_source, MODULES)
    keys = ("spacr.example.outer", "spacr.example.outer.choose")
    translated = (
        "Choisit une politique locale.",
        "Renvoie la première valeur.\n\nRenvoie la valeur de repli si aucune valeur n'est fournie.",
    )
    english = {"schema": 2, "language": "en", "symbols": {}}
    french = {"schema": 2, "language": "fr", "symbols": {}}
    for key, target in zip(keys, translated):
        blocks, _ = builder.translatable_blocks(docs[key])
        record = {
            "text": docs[key], "source_sha256": builder._source_hash(docs[key]),
            "source_blocks_sha256": [builder._source_hash(block) for block in blocks],
        }
        english["symbols"][key] = record
        french["symbols"][key] = {
            **record, "text": target,
            "translation_source_blocks_sha256": [
                builder._source_hash(builder._api_translation_source(block)) for block in blocks
            ],
        }
    page = (built_site / "api/spacr/example/index.html").read_text()
    harness = '''
<script src="/api/api_i18n.js" data-api-catalog-version="feature-411-fixture"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    const parent = document.getElementById('spacr.example.outer').parentElement
      .querySelector(':scope > dd > .spacr-api-translation');
    const helper = document.getElementById('spacr.example.outer.choose').parentElement
      .querySelector(':scope > dd > .spacr-api-translation');
    const ok = parent && helper && parent !== helper &&
      parent.textContent.includes('politique locale') &&
      !parent.textContent.includes('première valeur') &&
      helper.textContent.includes('première valeur') &&
      helper.textContent.includes('valeur de repli') &&
      !helper.textContent.includes('politique locale');
    document.body.dataset.nestedTranslation = ok ? 'pass' : 'fail';
  }, 900);
});
</script>
'''
    page = page.replace("</head>", harness + "</head>")
    files = {
        "/api/page.html": page.encode(),
        "/api/api_i18n.js": frontend.SCRIPT.read_bytes(),
        "/api/i18n/api/en.json": json.dumps(english).encode(),
        "/api/i18n/api/fr.json": json.dumps(french).encode(),
    }
    with frontend._server(files) as (base, _requests):
        dom = frontend._dump_dom(f"{base}/api/page.html?lang=fr", budget=2400)
    assert 'data-nested-translation="pass"' in dom

    # Make the shipped browser source look a helper up under its parent key.
    # The same rendered-page contract must now fail, not silently bless it.
    original = files["/api/api_i18n.js"]
    old = b"const record = own(symbols, signature.id) ? symbols[signature.id] : null;"
    assert original.count(old) == 1
    files["/api/api_i18n.js"] = original.replace(
        old,
        b"const lookup = signature.id.endsWith('.choose') ? "
        b"signature.id.slice(0, -7) : signature.id; "
        b"const record = own(symbols, lookup) ? symbols[lookup] : null;",
    )
    with frontend._server(files) as (base, _requests):
        mutated_dom = frontend._dump_dom(f"{base}/api/page.html?lang=fr", budget=2400)
    assert 'data-nested-translation="fail"' in mutated_dom


@pytest.mark.skipif(not os.environ.get("SPACR_DOCS_BUILT"), reason="requires a fresh full docs build")
def test_enabled_helpers_exist_in_the_full_built_site():
    active = helpers.active_entries(ROOT, ignore_patterns=builder.AUTOAPI_IGNORE)
    if not active:
        assert not helpers.ENABLED_MODULES, "A nonempty rollout must produce helper entries"
        return
    output = ROOT / "docs/_build/html"
    for entry in active:
        page = output / "api" / Path(*entry.module.split(".")) / "index.html"
        assert page.is_file(), page
        assert page.read_text().count(f'id="{entry.qualified_key}"') == 1


@pytest.mark.skipif(not os.environ.get("SPACR_DOCS_BUILT"), reason="requires a fresh full docs build")
@pytest.mark.parametrize("language", sorted(builder.MODEL_SPECS))
@pytest.mark.parametrize("catalog_scope", ["full_catalog", "helper_slice"])
@pytest.mark.parametrize("module, count, hidden", [
    ("spacr.object", 3, ("_cellpose_z_segment_fn",)),
    ("spacr.timeflows_model", 7, ("_ctc_track_masks", "_training_window")),
])
def test_enabled_helpers_use_their_own_real_catalog_entries_in_the_browser(
    language, catalog_scope, module, count, hidden,
):
    """Check helper rendering independently and retain whole-catalog acceptance."""
    spec = importlib.util.spec_from_file_location(
        "_feature_411_real_frontend", ROOT / "tests/test_api_i18n_frontend.py",
    )
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    active = helpers.active_entries(ROOT, ignore_patterns=builder.AUTOAPI_IGNORE)
    keys = [entry.qualified_key for entry in active if entry.module == module]
    assert len(keys) == count
    page = (ROOT / "docs/_build/html/api" / Path(*module.split('.')) / "index.html").read_text()
    for name in hidden:
        assert f'id="{module}.{name}"' not in page
    english = json.loads((builder.API_DIR / "en.json").read_text())
    catalog = json.loads((builder.API_DIR / f"{language}.json").read_text())
    expected = {key: catalog["symbols"][key]["text"] for key in keys}
    for key in keys:
        assert catalog["symbols"][key]["source_sha256"] == english["symbols"][key]["source_sha256"]
    if catalog_scope == "helper_slice":
        # Exact shipped records in an explicitly bounded catalog fixture.
        # The full_catalog cases above still reject unrelated stale entries.
        english = {**english, "symbols": {key: english["symbols"][key] for key in keys}}
        catalog = {**catalog, "symbols": {key: catalog["symbols"][key] for key in keys}}
    harness = r'''
<script src="/api/api_i18n.js" data-api-catalog-version="feature-411-real"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    const expected = EXPECTED;
    const normalize = value => value.replace(/\s+/g, '');
    const nodes = Object.entries(expected).map(([key, target]) => {
      const anchor = document.getElementById(key);
      const node = anchor && anchor.parentElement.querySelector(':scope > dd > .spacr-api-translation');
      if (!node) return null;
      const prose = node.cloneNode(true);
      const fields = [...target.matchAll(/^:param\s+(\w+):/gm)].map(match => match[1]);
      const renderedFields = [...prose.querySelectorAll('dt')].map(field => field.textContent);
      if (!fields.every(field => renderedFields.some(label => label.includes(field)))) return null;
      prose.querySelectorAll('.spacr-api-translation__label, dt').forEach(element => element.remove());
      const visibleTarget = target.replace(/``([^`]+)``/g, '$1')
        .replace(/^:param\s+\w+:\s*/gm, '').replace(/^:returns:\s*/gm, '');
      return normalize(prose.textContent) === normalize(visibleTarget) ? node : null;
    });
    document.body.dataset.realHelpers = nodes.every(Boolean) && new Set(nodes).size === Object.keys(expected).length ? 'pass' : 'fail';
  }, 900);
});
</script>
'''.replace("EXPECTED", json.dumps(expected, ensure_ascii=False))
    page = page.replace("</head>", harness + "</head>")
    files = {
        "/api/page.html": page.encode(),
        "/api/api_i18n.js": frontend.SCRIPT.read_bytes(),
        "/api/i18n/api/en.json": json.dumps(english).encode(),
        f"/api/i18n/api/{language}.json": json.dumps(catalog).encode(),
    }
    with frontend._server(files) as (base, _requests):
        dom = frontend._dump_dom(f"{base}/api/page.html?lang={language}", budget=2400)
    assert 'data-real-helpers="pass"' in dom, (catalog_scope, module, language)
