"""Focused browser/static contracts for the external API translation UI."""

from __future__ import annotations

import ast
import contextlib
import fnmatch
import json
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "source" / "_static" / "api_i18n.js"
CONF = ROOT / "docs" / "source" / "conf.py"
ENGLISH_CATALOG = (
    ROOT / "docs" / "source" / "_static" / "i18n" / "api" / "en.json"
)
CHROME = shutil.which("google-chrome") or shutil.which("chromium")
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _conf_assignment(name: str):
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Sphinx configuration assignment not found: {name}")


def _record(text: str, source_hash: str, block_hash: str, *, english=False):
    record = {
        "text": text,
        "source_sha256": source_hash,
        "source_blocks_sha256": [block_hash],
    }
    if not english:
        record["translation_source_blocks_sha256"] = [HEX_C]
    return record


def _catalog(language: str, module_text: str, member_text: str, *, stale=False):
    localized = language != "en"
    return {
        "schema": 2,
        "language": language,
        "symbols": {
            "spacr.demo": _record(
                module_text,
                HEX_C if stale else HEX_A,
                HEX_B,
                english=not localized,
            ),
            "spacr.demo.run": _record(
                member_text,
                HEX_B,
                HEX_A,
                english=not localized,
            ),
        },
    }


ENGLISH = _catalog("en", "English module.", "English member.")


def _page(harness: str, *, before_script: str = "") -> bytes:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>API frontend fixture</title>
<script>{before_script}</script>
<script src="/api/api_i18n.js" data-api-catalog-version="unit-content-v1"></script>
<script>
window.addEventListener("DOMContentLoaded", () => {{ {harness} }});
</script></head><body>
<main><article role="main" id="furo-main-content">
<section id="module-spacr.demo"><span id="spacr-demo"></span>
<h1>spacr.demo<a class="headerlink">¶</a></h1><p>English module body</p>
<section id="module-contents"><h2>Module Contents</h2>
<dl class="py function"><dt id="spacr.demo.run">run()</dt><dd>English body</dd></dl>
</section></section>
</article></main></body></html>""".encode()


@contextlib.contextmanager
def _server(files, *, delays=None, statuses=None):
    requests = []
    delays = delays or {}
    statuses = statuses or {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlsplit(self.path)
            requests.append((parsed.path, parsed.query))
            if delays.get(parsed.path):
                time.sleep(delays[parsed.path])
            status = statuses.get(parsed.path, 200 if parsed.path in files else 404)
            body = files.get(parsed.path, b"not found")
            if isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json" if parsed.path.endswith(".json")
                else "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _browser_files(page: bytes, **catalogs):
    files = {
        "/api/page.html": page,
        "/api/api_i18n.js": SCRIPT.read_bytes(),
        "/api/i18n/api/en.json": json.dumps(ENGLISH).encode(),
    }
    for language, catalog in catalogs.items():
        files[f"/api/i18n/api/{language}.json"] = json.dumps(catalog).encode()
    return files


def _dump_dom(url: str, *, budget=1800) -> str:
    if not CHROME:
        pytest.skip("Chrome/Chromium is required for the focused browser test")
    with tempfile.TemporaryDirectory(prefix="spacr-api-i18n-chrome-") as profile:
        completed = subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-sync",
                "--metrics-recording-only",
                "--no-first-run",
                "--renderer-process-limit=1",
                f"--user-data-dir={profile}",
                f"--virtual-time-budget={budget}",
                "--dump-dom",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
            # Six fast-suite Python cells can launch Chrome concurrently on
            # hosted runners.  Startup alone exceeded 20 seconds once under
            # that contention even though the page's virtual-time budget is
            # only 1.8 seconds.  Keep every DOM assertion and allow the
            # external process enough wall time to start deterministically.
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout


def test_frontend_source_has_failure_and_freshness_guards():
    script = SCRIPT.read_text(encoding="utf-8")
    conf = CONF.read_text(encoding="utf-8")

    assert 'document.querySelector(\'article[role="main"]\')' in script
    assert "apiArticle.prepend(wrapper)" in script
    assert "navigator.languages" in script
    assert 'if (base === "zh") return null' in script
    assert "safeStorageGet" in script and "safeStorageSet" in script
    assert "requestSerial" in script and "AbortController" in script
    assert "validateAgainstEnglish" in script
    assert "payload.schema !== 2" in script
    assert "translation_source_blocks_sha256" in script
    assert 'window.addEventListener("popstate"' in script
    assert "failureHistoryMode" in script
    assert "void selectLanguage" in script
    assert "data-api-catalog-version" in conf
    assert "_api_catalog_hasher" in conf
    assert ':scope > section[id^="module-spacr"] > h1' in script
    assert "localizedFieldName" in script and "RST_TERMS" in script


def test_autoapi_emits_only_canonical_module_members():
    options = _conf_assignment("autoapi_options")
    ignored = _conf_assignment("autoapi_ignore")
    assert "members" in options and "undoc-members" in options
    # Imported members are duplicate aliases. Their generated ids can be
    # noncanonical and therefore cannot be joined exactly to `spacr.*` catalog
    # keys without ambiguous suffix matching.
    assert "imported-members" not in options
    # Resource generator scripts otherwise become non-package top-level API
    # modules (`render`, `parts`, etc.) that cannot match the `spacr.*` catalog.
    assert "*/resources/*/_generators/*" in ignored
    generators = list((ROOT / "spacr" / "resources").glob("**/_generators/*.py"))
    assert generators
    assert all(any(fnmatch.fnmatch(str(path), pattern) for pattern in ignored)
               for path in generators)


def test_english_catalog_keys_are_exactly_joinable():
    payload = json.loads(ENGLISH_CATALOG.read_text(encoding="utf-8"))
    symbols = payload["symbols"]

    assert payload["schema"] == 2 and payload["language"] == "en"
    assert symbols
    assert all(key == "spacr" or key.startswith("spacr.") for key in symbols)


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_furo_placement_catalog_version_safe_rst_and_persistence():
    translated = (
        "Resumen :func:`spacr.demo.run`.\n\n"
        ".. warning::\n\n   <img src=x onerror=alert(1)> Nunca.\n\n"
        "* Uno\n* Dos\n\n:param value: Valor\n:returns: Resultado\n\n"
        "Ejemplo::\n\n   >>> print(\"<unsafe>\")"
    )
    es = _catalog("es", translated, "Miembro traducido.")
    harness = """
setTimeout(() => {
  const article = document.querySelector('article[role="main"]');
  const panels = article.querySelectorAll('.spacr-api-translation');
  const moduleHeading = article.querySelector(
    ':scope > section[id^="module-spacr"] > h1');
  const modulePanel = moduleHeading.nextElementSibling;
  const ok = article.firstElementChild.classList.contains('spacr-api-language') &&
    article.children[1].id === 'module-spacr.demo' && panels.length === 2 &&
    modulePanel.classList.contains('spacr-api-translation') &&
    modulePanel.querySelector('.admonition.warning') &&
    modulePanel.querySelector('.admonition-title').textContent === 'Advertencia' &&
    modulePanel.querySelector('code.rst-role-func') &&
    modulePanel.querySelector('ul li') &&
    modulePanel.querySelector('dl.field-list') &&
    modulePanel.querySelector('dl.field-list dt').textContent === 'Parámetro value' &&
    [...modulePanel.querySelectorAll('dl.field-list dt')]
      .some((node) => node.textContent === 'Devuelve') &&
    modulePanel.querySelector('pre code') && !modulePanel.querySelector('img') &&
    modulePanel.textContent.includes('<img src=x') &&
    document.querySelector('.spacr-api-language select').value === 'es' &&
    localStorage.getItem('spacr-doc-language') === 'es' &&
    window.unhandled === 0;
  document.body.dataset.result = ok ? 'pass' : 'fail';
}, 600);
"""
    before = """
window.unhandled = 0;
window.addEventListener('unhandledrejection', (event) => {
  window.unhandled += 1; event.preventDefault();
});
"""
    files = _browser_files(_page(harness, before_script=before), es=es)
    with _server(files) as (base, requests):
        dom = _dump_dom(f"{base}/api/page.html?lang=es")

    assert 'data-result="pass"' in dom
    catalog_requests = [item for item in requests if item[0].endswith(".json")]
    assert {path for path, _query in catalog_requests} == {
        "/api/i18n/api/en.json",
        "/api/i18n/api/es.json",
    }
    assert all("v=unit-content-v1" in query for _path, query in catalog_requests)


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_chinese_renderer_labels_are_localized():
    translated = (
        "摘要。\n\n.. note::\n\n   请仔细检查。\n\n"
        ":param value: 输入值\n:returns: 结果"
    )
    zh = _catalog("zh_CN", translated, "已翻译的成员。")
    harness = """
setTimeout(() => {
  const heading = document.querySelector(
    'article[role="main"] > section[id^="module-spacr"] > h1');
  const panel = heading.nextElementSibling;
  const terms = [...panel.querySelectorAll('dl.field-list dt')]
    .map((node) => node.textContent);
  const ok = panel.classList.contains('spacr-api-translation') &&
    panel.lang === 'zh-CN' &&
    panel.querySelector('.admonition-title').textContent === '备注' &&
    terms.includes('参数 value') && terms.includes('返回') &&
    document.querySelector('.spacr-api-language__label').textContent === 'API 语言';
  document.body.dataset.result = ok ? 'pass' : 'fail';
}, 500);
"""
    files = _browser_files(_page(harness), zh_CN=zh)
    with _server(files) as (base, _requests):
        dom = _dump_dom(f"{base}/api/page.html?lang=zh_CN")

    assert 'data-result="pass"' in dom


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_latest_request_wins_and_only_successful_locale_updates_state():
    de = _catalog("de", "Deutsches Modul.", "Deutsches Mitglied.")
    es = _catalog("es", "Módulo español.", "Miembro español.")
    harness = """
const select = document.querySelector('.spacr-api-language select');
select.value = 'de'; select.dispatchEvent(new Event('change'));
setTimeout(() => {
  select.value = 'es'; select.dispatchEvent(new Event('change'));
}, 20);
setTimeout(() => {
  const text = document.querySelector('article[role="main"]').textContent;
  const ok = select.value === 'es' && text.includes('Módulo español') &&
    !text.includes('Deutsches Modul') &&
    new URL(location.href).searchParams.get('lang') === 'es' &&
    localStorage.getItem('spacr-doc-language') === 'es' &&
    window.unhandled === 0;
  document.body.dataset.result = ok ? 'pass' : 'fail';
}, 700);
"""
    before = """
window.unhandled = 0;
window.addEventListener('unhandledrejection', (event) => {
  window.unhandled += 1; event.preventDefault();
});
"""
    files = _browser_files(_page(harness, before_script=before), de=de, es=es)
    with _server(
        files, delays={"/api/i18n/api/de.json": 0.3},
    ) as (base, _requests):
        dom = _dump_dom(f"{base}/api/page.html")

    assert 'data-result="pass"' in dom


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_popstate_restores_prior_successful_language():
    de = _catalog("de", "Deutsches Modul.", "Deutsches Mitglied.")
    es = _catalog("es", "Módulo español.", "Miembro español.")
    harness = """
const select = document.querySelector('.spacr-api-language select');
const observations = [];
select.value = 'de'; select.dispatchEvent(new Event('change'));
setTimeout(() => {
  observations.push(select.value === 'de' && location.search.includes('lang=de'));
  select.value = 'es'; select.dispatchEvent(new Event('change'));
}, 220);
setTimeout(() => {
  observations.push(select.value === 'es' && location.search.includes('lang=es'));
  history.back();
}, 430);
setTimeout(() => {
  observations.push(select.value === 'de' && location.search.includes('lang=de'));
  history.back();
}, 700);
setTimeout(() => {
  observations.push(select.value === 'en' && !location.search);
  document.body.dataset.result = observations.every(Boolean) ? 'pass' : 'fail';
}, 900);
"""
    files = _browser_files(_page(harness), de=de, es=es)
    with _server(files) as (base, _requests):
        dom = _dump_dom(f"{base}/api/page.html", budget=2200)

    assert 'data-result="pass"' in dom


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_malformed_and_http_failures_roll_back_without_persisting_locale():
    stale_fr = _catalog("fr", "Module français.", "Membre français.", stale=True)
    harness = """
const select = document.querySelector('.spacr-api-language select');
const checks = [];
select.value = 'fr'; select.dispatchEvent(new Event('change'));
setTimeout(() => {
  checks.push(select.value === 'en' &&
    !document.querySelector('.spacr-api-translation') &&
    localStorage.getItem('spacr-doc-language') === 'en' &&
    new URL(location.href).searchParams.get('lang') === 'en');
  select.value = 'hi'; select.dispatchEvent(new Event('change'));
}, 260);
setTimeout(() => {
  checks.push(select.value === 'en' &&
    !document.querySelector('.spacr-api-translation') &&
    localStorage.getItem('spacr-doc-language') === 'en' &&
    new URL(location.href).searchParams.get('lang') === 'en' &&
    window.unhandled === 0);
  document.body.dataset.result = checks.every(Boolean) ? 'pass' : 'fail';
}, 600);
"""
    before = """
window.unhandled = 0;
window.addEventListener('unhandledrejection', (event) => {
  window.unhandled += 1; event.preventDefault();
});
"""
    files = _browser_files(_page(harness, before_script=before), fr=stale_fr)
    files["/api/i18n/api/hi.json"] = b"service unavailable"
    with _server(
        files, statuses={"/api/i18n/api/hi.json": 503},
    ) as (base, _requests):
        dom = _dump_dom(f"{base}/api/page.html")

    assert 'data-result="pass"' in dom


@pytest.mark.skipif(not CHROME, reason="Chrome/Chromium not installed")
def test_storage_exceptions_and_zh_tw_do_not_block_next_browser_locale():
    de = _catalog("de", "Deutsches Modul.", "Deutsches Mitglied.")
    before = """
Object.defineProperty(navigator, 'languages', {value: ['zh-TW', 'de-DE']});
Object.defineProperty(navigator, 'language', {value: 'zh-TW'});
Storage.prototype.getItem = function () { throw new Error('storage denied'); };
Storage.prototype.setItem = function () { throw new Error('storage denied'); };
window.unhandled = 0;
window.addEventListener('unhandledrejection', (event) => {
  window.unhandled += 1; event.preventDefault();
});
"""
    harness = """
setTimeout(() => {
  const select = document.querySelector('.spacr-api-language select');
  const text = document.querySelector('article[role="main"]').textContent;
  document.body.dataset.result = select.value === 'de' &&
    text.includes('Deutsches Modul') && window.unhandled === 0 ? 'pass' : 'fail';
}, 500);
"""
    files = _browser_files(_page(harness, before_script=before), de=de)
    with _server(files) as (base, requests):
        dom = _dump_dom(f"{base}/api/page.html")

    assert 'data-result="pass"' in dom
    assert not any(path.endswith("zh_CN.json") for path, _query in requests)
