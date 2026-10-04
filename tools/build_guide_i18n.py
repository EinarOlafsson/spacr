"""Translated user guides: gettext catalogs, Sphinx hooks and audit.

The Sphinx user guides (``docs/source/*.rst``) are translated with Sphinx's
own gettext machinery. English stays the source of truth:

* ``extract`` runs the ``gettext`` builder in guides-only mode and writes the
  English message templates (``.pot``) to a build directory.
* ``update`` merges those templates into ``docs/i18n/guides/<lang>/LC_MESSAGES
  /<doc>.po``. A changed English paragraph gets a new msgid; its old
  translation is kept only as a *fuzzy* hint, which Sphinx never renders, so
  the paragraph falls back to English until it is translated again.
* ``export``/``import`` move pending messages through a JSON worklist so a
  translator never edits ``.po`` quoting by hand. ``import`` refuses a
  translation that changes a literal, a role target, a URL or a UI name.
* ``audit`` reports coverage, stale (fuzzy) and missing messages per page and
  checks each catalog's review label and UI-name glossary.
* ``build`` renders one language into its own ``/<lang>/`` subtree next to the
  English site, with ``-W``.

Imported by ``docs/source/conf.py`` as a Sphinx extension in guides-only mode
(``SPACR_DOCS_GUIDES_ONLY=1``): the translated trees reuse the English API
reference and tutorial player through relative links instead of rebuilding
or copying them.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "source"
LOCALE_DIR = ROOT / "docs" / "i18n" / "guides"
GLOSSARY_DIR = LOCALE_DIR / "glossary"
LANGUAGES = ("sv", "de", "es", "pt", "fr", "is", "zh_CN", "ko", "hi")

REVIEW_KIND = "AI technical review (Claude Opus 5.5), no native-speaker signoff"
TRANSLATOR = "Claude Opus 5.5 (direct AI translation)"
_REVIEWERS = {"claude-opus-5.5": "Claude Opus 5.5", "codex": "Codex"}
_REVIEW_AUTHORS = {
    REVIEW_KIND: ("Claude Opus 5.5",),
    "AI technical review (Codex), no native-speaker signoff": ("Codex",),
    "AI technical review (Claude Opus 5.5 / Codex), no native-speaker signoff":
        ("Claude Opus 5.5", "Codex"),
}
SUPPORTED_REVIEW_KINDS = frozenset(_REVIEW_AUTHORS)

# Pages served only in English. ``settings_flow`` is 23,000 lines of
# generated call-graph listings (3.6 MB of HTML) regenerated from the source
# tree; translated trees link to the English page instead of copying it.
ENGLISH_ONLY_PAGES = frozenset({"settings_flow"})


def excluded_message(domain: str, msgid: str = "") -> bool:
    """True for a message that is never translated (English-only page)."""
    return domain in ENGLISH_ONLY_PAGES


# Banner text on every translated page. ``label`` is the page-level
# equivalent of the API panels' "Translated API documentation" label.
BANNERS: Mapping[str, Mapping[str, str]] = {
    "sv": {
        "label": "Översatt användarhandledning",
        "review": "AI-baserad teknisk granskning (Claude Opus 5.5), "
                  "ingen granskning av modersmålstalare.",
        "fallback": "Stycken som ännu inte är översatta, eller vars engelska "
                    "original har ändrats, visas på engelska.",
        "original": "Engelskt original",
        "select": "Språk",
    },
    "de": {
        "label": "Übersetztes Benutzerhandbuch",
        "review": "KI-gestützte technische Prüfung (Claude Opus 5.5), "
                  "keine Freigabe durch Muttersprachler.",
        "fallback": "Absätze, die noch nicht übersetzt sind oder deren "
                    "englisches Original sich geändert hat, erscheinen auf "
                    "Englisch.",
        "original": "Englisches Original",
        "select": "Sprache",
    },
    "es": {
        "label": "Guía de usuario traducida",
        "review": "Revisión técnica por IA (Claude Opus 5.5), sin "
                  "aprobación de hablantes nativos.",
        "fallback": "Los párrafos aún no traducidos, o cuyo original en "
                    "inglés ha cambiado, se muestran en inglés.",
        "original": "Original en inglés",
        "select": "Idioma",
    },
    "pt": {
        "label": "Guia do usuário traduzido",
        "review": "Revisão técnica por IA (Claude Opus 5.5), sem aprovação "
                  "de falantes nativos.",
        "fallback": "Parágrafos ainda não traduzidos, ou cujo original em "
                    "inglês mudou, aparecem em inglês.",
        "original": "Original em inglês",
        "select": "Idioma",
    },
    "fr": {
        "label": "Guide de l’utilisateur traduit",
        "review": "Relecture technique par IA (Claude Opus 5.5), sans "
                  "validation par un locuteur natif.",
        "fallback": "Les paragraphes pas encore traduits, ou dont l’original "
                    "anglais a changé, s’affichent en anglais.",
        "original": "Original anglais",
        "select": "Langue",
    },
    "is": {
        "label": "Þýddar notendaleiðbeiningar",
        "review": "Tæknileg yfirferð gervigreindar (Claude Opus 5.5), "
                  "enginn móðurmálshafi hefur samþykkt þýðinguna.",
        "fallback": "Efnisgreinar sem hafa ekki enn verið þýddar, eða þar "
                    "sem enski frumtextinn hefur breyst, birtast á ensku.",
        "original": "Enskur frumtexti",
        "select": "Tungumál",
    },
    "zh_CN": {
        "label": "已翻译的用户指南",
        "review": "AI 技术审校（Claude Opus 5.5），未经母语人士审定。",
        "fallback": "尚未翻译或英文原文已更改的段落以英文显示。",
        "original": "英文原文",
        "select": "语言",
    },
    "ko": {
        "label": "번역된 사용자 가이드",
        "review": "AI 기술 검토(Claude Opus 5.5), 원어민 검수 없음.",
        "fallback": "아직 번역되지 않았거나 영어 원문이 바뀐 단락은 영어로 "
                    "표시됩니다.",
        "original": "영어 원문",
        "select": "언어",
    },
    "hi": {
        "label": "अनूदित उपयोगकर्ता मार्गदर्शिका",
        "review": "AI तकनीकी समीक्षा (Claude Opus 5.5), किसी मूल भाषी "
                  "द्वारा अनुमोदित नहीं।",
        "fallback": "जिन अनुच्छेदों का अभी अनुवाद नहीं हुआ है, या जिनका "
                    "अंग्रेज़ी मूल बदल गया है, वे अंग्रेज़ी में दिखते हैं।",
        "original": "अंग्रेज़ी मूल",
        "select": "भाषा",
    },
}


# --------------------------------------------------------------------------
# Which languages have catalogs
# --------------------------------------------------------------------------

def catalog_languages(locale_dir: Path = LOCALE_DIR) -> tuple[str, ...]:
    """Languages with at least one committed ``.po`` catalog, in site order."""
    return tuple(
        language for language in LANGUAGES
        if any((locale_dir / language / "LC_MESSAGES").glob("*.po"))
    )


# --------------------------------------------------------------------------
# Sphinx extension (guides-only mode)
# --------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _is_tree_relative(uri: str) -> bool:
    return bool(uri) and not _SCHEME_RE.match(uri) and not uri.startswith(("#", "/"))


def _source_read(app, docname, source):
    """Drop toctree entries served only by the English site."""
    if docname != app.config.root_doc:
        return
    for page in ("api/index", *sorted(ENGLISH_ONLY_PAGES)):
        source[0] = re.sub(rf"(?m)^[ \t]+{re.escape(page)}[ \t]*\n", "", source[0])


def _doctree_read(app, doctree):
    """Point relative links (tutorial player, downloads) at the English root."""
    from docutils import nodes

    if app.config.language in (None, "en"):
        return
    depth = app.env.docname.count("/") + 1
    prefix = "../" * depth
    for node in doctree.findall(nodes.reference):
        uri = node.get("refuri", "")
        if node.get("internal") or not _is_tree_relative(uri):
            continue
        node["refuri"] = prefix + uri


def _page_context(app, pagename, templatename, context, doctree):
    language = app.config.language
    if language in (None, "en") or doctree is None:
        return
    text = BANNERS.get(language)
    if not text:
        return
    from html import escape

    depth = pagename.count("/") + 1
    original = "../" * depth + pagename + ".html"
    catalog_path = LOCALE_DIR / language / "LC_MESSAGES" / f"{pagename}.po"
    review_kind = (catalog_review_kind(catalog_path)
                   if catalog_path.is_file() else None) or REVIEW_KIND
    authors = " / ".join(_REVIEW_AUTHORS.get(review_kind, ("Claude Opus 5.5",)))
    review_text = text["review"].replace("Claude Opus 5.5", authors)
    banner = (
        f'<aside class="spacr-guide-translation" lang="{language.replace("_", "-")}"'
        f' data-review-kind="{escape(review_kind)}">'
        f'<p class="spacr-guide-translation__label">{escape(text["label"])}</p>'
        f'<p>{escape(review_text)} <span lang="en">({escape(review_kind)})</span> '
        f'<span class="spacr-guide-translation__swatch"></span>'
        f'{escape(text["fallback"])} '
        f'<a href="{escape(original)}">{escape(text["original"])}</a></p>'
        f"</aside>"
    )
    context["body"] = banner + context.get("body", "")


def _missing_reference(app, env, node, contnode):
    """Resolve API cross-references to the English API tree."""
    if app.config.language in (None, "en"):
        return None
    inventory = getattr(app, "_spacr_english_inventory", None)
    if inventory is None:
        return None
    from docutils import nodes

    domain = node.get("refdomain", "")
    reftype = node.get("reftype", "")
    target = node.get("reftarget", "")
    if domain == "std" and reftype == "doc":
        docname = target.lstrip("/")
        if docname in app.env.found_docs:
            return None
        uri = inventory.get(("std:doc", docname))
    elif domain == "std" and reftype == "ref":
        uri = inventory.get(("std:label", target.lower()))
    elif domain == "py":
        uri = None
        candidates = [target]
        module = node.get("py:module")
        if module:
            candidates.insert(0, f"{module}.{target}")
        for candidate in candidates:
            for kind in ("function", "class", "method", "module", "attribute",
                         "data", "exception", "property"):
                uri = inventory.get((f"py:{kind}", candidate))
                if uri:
                    break
            if uri:
                break
    else:
        return None
    if not uri:
        return None
    depth = node.get("refdoc", "").count("/") + 1
    reference = nodes.reference("", "", internal=False,
                                refuri="../" * depth + uri)
    reference.append(contnode)
    return reference


def read_inventory(path: Path) -> dict[tuple[str, str], str]:
    """``{(domain:role, name): uri}`` from a Sphinx ``objects.inv`` (v2)."""
    import zlib

    data = Path(path).read_bytes()
    lines = data.split(b"\n", 4)
    if not lines[0].startswith(b"# Sphinx inventory version 2"):
        raise ValueError(f"{path}: not a version 2 Sphinx inventory")
    body = zlib.decompress(lines[4]).decode("utf-8")
    pattern = re.compile(r"(.+?)\s+(\S+)\s+(-?\d+)\s+?(\S*)\s+(.*)")
    mapping = {}
    for line in body.splitlines():
        match = pattern.match(line.rstrip())
        if not match:
            continue
        name, kind, _priority, uri, _display = match.groups()
        if uri.endswith("$"):
            uri = uri[:-1] + name
        mapping[(kind, name)] = uri
    return mapping


def _load_english_inventory(app):
    if app.config.language in (None, "en") or app.builder.format != "html":
        return
    path = os.environ.get("SPACR_DOCS_ENGLISH_INVENTORY", "").strip() or str(
        ROOT / "docs" / "_build" / "html" / "objects.inv")
    app._spacr_english_inventory = read_inventory(Path(path))


def _share_english_images(app, exception):
    """Reuse byte-identical images from the English tree next to this one."""
    if exception is not None or app.config.language in (None, "en") \
            or app.builder.format != "html":
        return
    inventory = os.environ.get("SPACR_DOCS_ENGLISH_INVENTORY", "").strip()
    english = Path(inventory).parent if inventory else ROOT / "docs/_build/html"
    outdir = Path(app.outdir)
    images = outdir / "_images"
    if not images.is_dir() or not (english / "_images").is_dir():
        return
    shared = []
    for path in sorted(images.iterdir()):
        twin = english / "_images" / path.name
        if twin.is_file() and twin.read_bytes() == path.read_bytes():
            shared.append(path.name)
    if not shared:
        return
    names = "|".join(re.escape(name) for name in shared)
    pattern = re.compile(rf'((?:src|href)=")((?:\.\./)*)(_images/(?:{names})")')
    for page in outdir.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        updated = pattern.sub(lambda m: m[1] + "../" + m[2] + m[3], text)
        if updated != text:
            page.write_text(updated, encoding="utf-8")
    for name in shared:
        (images / name).unlink()


def setup(app):
    # conf.py's own setup() connects AutoAPI's hook; guides-only mode does not
    # load AutoAPI, so register the event name to keep that hook inert.
    try:
        app.add_event("autoapi-skip-member")
    except Exception:  # already registered
        pass
    app.connect("build-finished", _share_english_images)
    app.connect("builder-inited", _load_english_inventory)
    app.connect("source-read", _source_read)
    app.connect("doctree-read", _doctree_read)
    app.connect("missing-reference", _missing_reference)
    app.connect("html-page-context", _page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


# --------------------------------------------------------------------------
# Message validation
# --------------------------------------------------------------------------

_LITERAL_RE = re.compile(r"``(.+?)``")
_ROLE_RE = re.compile(r":([a-z]+(?::[a-z]+)?):`([^`]*)`")
_ROLE_TARGET_RE = re.compile(r"<([^<>]+)>\s*$")
_URL_RE = re.compile(r"https?://[^\s<>`]+")
_LINK_TARGET_RE = re.compile(r"`[^`]*<([^<>`]+)>`_{1,2}")
_SUBST_RE = re.compile(r"\|[\w-]+\|")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODEBLOCK_RE = re.compile(r"^\s*\.\.\s")


def _role_targets(text: str) -> list[str]:
    targets = []
    for role, body in _ROLE_RE.findall(text):
        match = _ROLE_TARGET_RE.search(body)
        target = match.group(1) if match else body
        if role in ("ref", "doc") or match or role.startswith(("py", "func",
                                                               "class", "meth",
                                                               "mod", "attr")):
            targets.append(f"{role}:{target.strip()}")
    return sorted(targets)


def invariants(text: str) -> dict[str, list[str]]:
    """Tokens a translation must carry over unchanged."""
    without_roles = _ROLE_RE.sub(" ", text)
    return {
        "literals": sorted(_LITERAL_RE.findall(text)),
        "roles": _role_targets(text),
        "urls": sorted(set(_URL_RE.findall(without_roles))
                       | set(_LINK_TARGET_RE.findall(without_roles))),
        "substitutions": sorted(_SUBST_RE.findall(_LITERAL_RE.sub(" ", text))),
    }


def has_prose(text: str) -> bool:
    """False when a message holds only literals, references, URLs, numbers."""
    rest = _LITERAL_RE.sub(" ", text)
    # A role or link with an explicit title keeps its title: that is prose.
    rest = re.sub(r":[a-z]+(?::[a-z]+)?:`([^`<]*)<[^`>]*>`", r" \1 ", rest)
    rest = re.sub(r"`([^`<]*)<[^`>]*>`_{1,2}", r" \1 ", rest)
    rest = _ROLE_RE.sub(" ", rest)
    rest = _URL_RE.sub(" ", _SUBST_RE.sub(" ", rest))
    return bool(re.search(r"[^\W\d_]{2,}", rest))


def ui_names(text: str) -> list[str]:
    """Bold UI names (menu paths split at arrows) mentioned in a message."""
    names = []
    for span in _BOLD_RE.findall(text):
        for part in re.split(r"\s*→\s*", span):
            part = part.strip()
            if part:
                names.append(part)
    return names


def message_problems(msgid: str, msgstr: str,
                     glossary: Mapping[str, str] | None = None) -> list[str]:
    """Reasons a translation is unsafe to publish (empty list: acceptable)."""
    problems = []
    if not msgstr.strip():
        return ["empty translation"]
    source, target = invariants(msgid), invariants(msgstr)
    for key in source:
        if source[key] != target[key]:
            problems.append(
                f"{key} differ: {source[key]!r} != {target[key]!r}")
    if glossary:
        shown = ui_names(msgstr)
        for name in ui_names(msgid):
            expected = glossary.get(name)
            if expected and expected not in shown:
                problems.append(f"UI name {name!r} must read **{expected}**")
    if msgid.count("**") != msgstr.count("**"):
        problems.append("bold markup count differs")
    return problems


# --------------------------------------------------------------------------
# Catalog files
# --------------------------------------------------------------------------

def _babel():
    from babel.messages import pofile
    from babel.messages.catalog import Catalog
    return pofile, Catalog


def read_catalog(path: Path):
    pofile, _Catalog = _babel()
    with path.open("rb") as stream:
        catalog = pofile.read_po(stream, ignore_obsolete=True)
    catalog._spacr_review_kind = catalog_review_kind(path) or REVIEW_KIND
    return catalog


def write_catalog(path: Path, catalog) -> None:
    pofile, _Catalog = _babel()
    review_kind = getattr(catalog, "_spacr_review_kind", REVIEW_KIND)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.header_comment = (
        f"# spaCR user guide translation ({catalog.locale_identifier or ''}).\n"
        f"# {review_kind}.\n"
        "# English source: docs/source (authoritative). Built by\n"
        "# tools/build_guide_i18n.py; a changed English message is kept only\n"
        "# as a fuzzy hint and renders in English until re-translated."
    )
    catalog.mime_headers  # normalize
    tmp = path.with_suffix(".po.tmp")
    with tmp.open("wb") as stream:
        pofile.write_po(stream, catalog, width=0, no_location=True,
                        omit_header=False, sort_output=False,
                        include_previous=False, ignore_obsolete=True)
    text = tmp.read_text(encoding="utf-8")
    # Babel drops unknown headers; add the review label after Language.
    header = f'"X-Review-Kind: {review_kind}\\n"\n'
    if "X-Review-Kind:" not in text:
        text = text.replace('"MIME-Version:', header + '"MIME-Version:', 1)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def catalog_review_kind(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines()[:40]:
        match = re.match(r'^"X-Review-Kind: (.*)\\n"$', line)
        if match:
            return match.group(1)
    return None


def load_templates(pot_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """``{domain: [(msgid, context-free)]}`` from a gettext build."""
    pofile, _Catalog = _babel()
    templates = {}
    for path in sorted(pot_dir.rglob("*.pot")):
        domain = path.relative_to(pot_dir).with_suffix("").as_posix()
        if domain in ENGLISH_ONLY_PAGES:
            continue
        with path.open("rb") as stream:
            catalog = pofile.read_po(stream)
        templates[domain] = [message.id for message in catalog
                             if message.id and isinstance(message.id, str)]
    return templates


def update_language(language: str, pot_dir: Path,
                    locale_dir: Path = LOCALE_DIR) -> dict[str, dict[str, int]]:
    """Merge English templates into the language catalogs.

    Translations keep their msgid; an English change creates a new msgid.
    Babel's fuzzy matching carries the previous translation as a hint that
    Sphinx does not render.
    """
    pofile, Catalog = _babel()
    summary = {}
    for path in sorted(pot_dir.rglob("*.pot")):
        domain = path.relative_to(pot_dir).with_suffix("").as_posix()
        if domain in ENGLISH_ONLY_PAGES:
            continue
        with path.open("rb") as stream:
            template = pofile.read_po(stream)
        target = locale_dir / language / "LC_MESSAGES" / f"{domain}.po"
        if target.exists():
            catalog = read_catalog(target)
        else:
            catalog = Catalog(locale=language, project="spaCR user guide",
                              fuzzy=False)
        catalog.update(template, no_fuzzy_matching=False,
                       update_header_comment=False)
        catalog.language_team = "spaCR AI translation <noreply@spacr>"
        if not catalog.last_translator:
            catalog.last_translator = TRANSLATOR
        catalog.fuzzy = False
        for message in list(catalog):
            message.locations = []
            if message.id and not message.string and not has_prose(message.id):
                message.string = message.id     # literals, numbers, names only
        write_catalog(target, catalog)
        summary[domain] = _counts(catalog)
    return summary


def _counts(catalog) -> dict[str, int]:
    total = translated = fuzzy = 0
    for message in catalog:
        if not message.id:
            continue
        total += 1
        if message.string and message.fuzzy:
            fuzzy += 1
        elif message.string:
            translated += 1
    return {"total": total, "translated": translated, "fuzzy": fuzzy,
            "missing": total - translated - fuzzy}


def load_glossary(language: str) -> dict[str, str]:
    path = GLOSSARY_DIR / f"{language}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["terms"]


# A runtime row that still carries these English words next to translated
# ones is a runtime-catalog defect; it is reported, not copied into guides.
_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "and", "with", "your", "this", "that", "from", "choose", "route",
    "into", "only", "when", "which", "input", "output",
})


# Runtime rows found wrong while translating the guides (wrong sense, not
# just style) live in ``glossary/<lang>.defects.json``. The guides use a
# correct term and the row is reported for a runtime-catalog fix instead of
# being copied into the glossary.
def runtime_defects(language: str) -> dict[str, str]:
    path = GLOSSARY_DIR / f"{language}.defects.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["defects"]


def runtime_ui_name(name: str, language: str):
    """Return an exact UI row or setting label used by the running app.

    ``Checked images`` uses the prefix of its exact counted button caption;
    only the known trailing count in parentheses is omitted for the guide.
    """
    sys.path.insert(0, str(ROOT))
    from spacr.qt.i18n import _exact_translation
    from spacr.qt.i18n_catalogs import en as _english_catalog, setting_label

    if name == "Checked images":
        # The guide omits the changing count from this actual button caption.
        template = _exact_translation("Checked images ({count})", language)
        if not template or template.count("{count}") != 1:
            return None
        suffix = re.search(r"\s*(?:\(\{count\}\)|（\{count\}）)\s*$", template)
        if suffix is None:
            return None
        return template[:suffix.start()].strip() or None
    exact = _exact_translation(name, language)
    if exact:
        return exact
    if re.search(r"\bN\b", name):
        # Guides write a counted button as "Use N workers (recommended)"; the
        # app row is "Use {count} workers (recommended)". Show it with N.
        template = _exact_translation(re.sub(r"\bN\b", "{count}", name, count=1),
                                      language)
        if template and template.count("{count}") == 1:
            return template.replace("{count}", "N")
    for key, label in getattr(_english_catalog, "SETTING_LABELS", {}).items():
        if "." not in key and label == name:
            return setting_label(key, name, language)
    return None


def defect_snapshot(language: str) -> dict[str, dict[str, str]]:
    """``{English: {"runtime": wrong app value, "guide": term the guides use}}``.

    A defect is open while the app still shows exactly ``runtime``. Once the
    runtime catalog is corrected the glossary adopts the app's new name and
    :func:`retarget_fixed_defects` switches the guides over to it.
    """
    path = GLOSSARY_DIR / f"{language}.defects.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("snapshot", {})


def _bold_alignment(msgid: str, msgstr: str):
    """Pairs of (English part, translated part) for aligned bold spans."""
    source, target = _BOLD_RE.findall(msgid), _BOLD_RE.findall(msgstr)
    if len(source) != len(target):
        return []
    pairs = []
    for left, right in zip(source, target):
        left_parts = re.split(r"\s*→\s*", left)
        right_parts = re.split(r"\s*→\s*", right)
        if len(left_parts) == len(right_parts):
            pairs.extend(zip((x.strip() for x in left_parts),
                             (y.strip() for y in right_parts)))
    return pairs


def snapshot_defects(language: str, locale_dir: Path = LOCALE_DIR) -> dict:
    """Record each open defect's wrong runtime value and the guides' term."""
    path = GLOSSARY_DIR / f"{language}.defects.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    used: dict[str, dict[str, int]] = {}
    for po in sorted((locale_dir / language / "LC_MESSAGES").glob("*.po")):
        for message in read_catalog(po):
            if message.id and message.string and not message.fuzzy:
                pairs = _bold_alignment(message.id, message.string)
                if message.id in data["defects"]:   # a heading or table cell
                    pairs.append((message.id, message.string))
                for english, term in pairs:
                    if english in data["defects"]:
                        used.setdefault(english, {}).setdefault(term, 0)
                        used[english][term] += 1
    snapshot = data.get("snapshot", {})
    for english in data["defects"]:
        runtime = runtime_ui_name(english, language)
        if english in used:
            guide = max(used[english], key=used[english].get)
        else:
            guide = snapshot.get(english, {}).get("guide", "")
        snapshot[english] = {"runtime": runtime or "", "guide": guide}
    data["snapshot"] = snapshot
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    return snapshot


def retarget_fixed_defects(language: str, locale_dir: Path = LOCALE_DIR) -> dict:
    """Switch the guides to corrected runtime names and close those defects.

    A defect is fixed when the app's row differs from the recorded wrong
    value. Every bold occurrence of the guides' interim term for that UI name
    becomes the app's new name; the result must still pass
    :func:`message_problems`, and the defect leaves ``defects.json``.
    """
    path = GLOSSARY_DIR / f"{language}.defects.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = {}
    for english, record in data.get("snapshot", {}).items():
        current = runtime_ui_name(english, language) or ""
        if english in data["defects"] and current and current != record["runtime"]:
            fixed[english] = (record["guide"], current)
    if not fixed:
        return {}
    changed = 0
    for po in sorted((locale_dir / language / "LC_MESSAGES").glob("*.po")):
        catalog = read_catalog(po)
        touched = False
        for message in catalog:
            if not message.id or not message.string or message.fuzzy:
                continue
            text = message.string
            for english, (old, new) in fixed.items():
                if not old or english not in message.id:
                    continue
                if english in ui_names(message.id):
                    text = re.sub(r"\*\*([^*]+)\*\*",
                                  lambda m: "**" + re.sub(
                                      r"(^|(?<=→ ))" + re.escape(old) + r"(?=$| →)",
                                      new, m.group(1)) + "**", text)
                else:
                    # Headings, table cells and image alt text name the
                    # control in plain text: replace the interim term there.
                    text = text.replace(old, new)
            if text != message.string:
                message.string = text
                touched = True
                changed += 1
        if touched:
            write_catalog(po, catalog)
    for english in fixed:
        data["defects"].pop(english, None)
        data["snapshot"].pop(english, None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    manual = sorted(english for english, (old, _new) in fixed.items() if not old)
    return {"fixed": sorted(fixed), "messages": changed,
            "check_by_hand": manual}


def build_glossary(language: str, pot_dir: Path) -> dict[str, str]:
    """English UI names in the guides -> the running app's translation."""
    names = set()
    for domain, messages in load_templates(pot_dir).items():
        for msgid in messages:
            if not excluded_message(domain, msgid):
                names.update(ui_names(msgid))
    terms, suspect = {}, {}
    defects = runtime_defects(language)
    snapshots = defect_snapshot(language)
    for name in sorted(names):
        if len(name) > 60 or "``" in name or not re.search(r"[A-Za-z]", name):
            continue
        # Exact catalog rows only: the composed/term fallbacks can splice
        # English and translated words, which is not a name the app shows.
        translated = runtime_ui_name(name, language)
        if not translated or translated == name:
            continue
        record = snapshots.get(name)
        if name in defects and (record is None or record["runtime"] == translated):
            suspect[name] = translated
            continue
        words = set(re.findall(r"[a-z]+", translated.lower()))
        if (words & _ENGLISH_FUNCTION_WORDS & set(re.findall(r"[a-z]+", name.lower()))
                or translated.count("(") + translated.count("（") != name.count("(")):
            suspect[name] = translated
            continue
        terms[name] = translated
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    (GLOSSARY_DIR / f"{language}.json").write_text(json.dumps({
        "schema": 1,
        "language": language,
        "source": "spacr.qt.i18n.tr (runtime UI catalogs)",
        "note": "Bold UI names in the guides must use the name the app shows "
                "in this language.",
        "terms": terms,
        "runtime_suspect": suspect,
    }, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return terms


def export_worklist(language: str, output: Path, domains: Iterable[str] | None = None,
                    include_fuzzy: bool = True,
                    locale_dir: Path = LOCALE_DIR) -> int:
    """Write pending messages as JSON ``[{domain, msgid, hint, msgstr}]``."""
    selected = set(domains or ())
    glossary = load_glossary(language)
    rows = []
    for path in sorted((locale_dir / language / "LC_MESSAGES").rglob("*.po")):
        domain = path.relative_to(locale_dir / language / "LC_MESSAGES") \
            .with_suffix("").as_posix()
        if selected and domain not in selected:
            continue
        for message in read_catalog(path):
            if not message.id or excluded_message(domain, message.id):
                continue
            if message.string and not message.fuzzy:
                continue
            if message.fuzzy and not include_fuzzy:
                continue
            ui = {name: glossary[name] for name in ui_names(message.id)
                  if name in glossary}
            rows.append({"domain": domain, "msgid": message.id,
                         "hint": message.string if message.fuzzy else "",
                         "ui": ui, "msgstr": ""})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    return len(rows)


def import_worklist(language: str, worklist: Path, strings: Path | None = None,
                    locale_dir: Path = LOCALE_DIR,
                    reviewer: str | None = None) -> tuple[int, list[str]]:
    """Apply a filled worklist; rejects rows that fail validation.

    ``strings`` optionally holds the translations as a JSON list aligned with
    the worklist rows (or ``{"<row index>": translation}``), so a translator
    does not have to copy the English back out.
    """
    if reviewer is not None and reviewer not in _REVIEWERS:
        raise ValueError(f"Unsupported guide reviewer: {reviewer}")
    rows = json.loads(worklist.read_text(encoding="utf-8"))
    if strings is not None:
        filled = json.loads(strings.read_text(encoding="utf-8"))
        if isinstance(filled, list):
            if len(filled) != len(rows):
                raise ValueError(f"{strings}: {len(filled)} strings for {len(rows)} rows")
            filled = dict(enumerate(filled))
        for index, value in filled.items():
            rows[int(index)]["msgstr"] = value or ""
    glossary = load_glossary(language)
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    applied, rejected = 0, []
    for domain, domain_rows in by_domain.items():
        path = locale_dir / language / "LC_MESSAGES" / f"{domain}.po"
        catalog = read_catalog(path)
        existing_reviewers = set()
        if reviewer is not None:
            kind = getattr(catalog, "_spacr_review_kind", REVIEW_KIND)
            if kind not in SUPPORTED_REVIEW_KINDS:
                raise ValueError(f"Unsupported guide review label in {path}: {kind}")
            if any(message.id and message.string and not message.fuzzy
                   and has_prose(message.id) for message in catalog):
                existing_reviewers.update(_REVIEW_AUTHORS[kind])
        domain_applied = 0
        for row in domain_rows:
            msgstr = row.get("msgstr", "")
            if not msgstr:
                continue
            message = catalog.get(row["msgid"])
            if message is None:
                rejected.append(f"{domain}: stale msgid {row['msgid'][:60]!r}")
                continue
            problems = message_problems(row["msgid"], msgstr, glossary)
            if problems:
                rejected.append(f"{domain}: {row['msgid'][:60]!r}: {'; '.join(problems)}")
                continue
            message.string = msgstr
            message.flags.discard("fuzzy")
            if reviewer is not None:
                comment = (f"AI technical review ({_REVIEWERS[reviewer]}), "
                           "no native-speaker signoff.")
                if comment not in message.user_comments:
                    message.user_comments.append(comment)
            applied += 1
            domain_applied += 1
        if reviewer is not None and domain_applied:
            existing_reviewers.add(_REVIEWERS[reviewer])
            catalog._spacr_review_kind = next(
                kind for kind, authors in _REVIEW_AUTHORS.items()
                if set(authors) == existing_reviewers)
            catalog.last_translator = f"{_REVIEWERS[reviewer]} (AI technical translation)"
        write_catalog(path, catalog)
    return applied, rejected


_PREFIX_RE = re.compile(r"^(:ref:`[^`]+`|\*\*[^*]+\*\*)(: | — )(.+)$", re.S)


def prefill_from_runtime(language: str, locale_dir: Path = LOCALE_DIR,
                         domains: Iterable[str] | None = None) -> dict[str, int]:
    """Reuse the running app's own translation where a guide message is a UI row.

    Much of ``workflows`` is the module-workflow text the app shows in its
    workflow panel. A pending message that is exactly a runtime catalog row,
    or ``<ref or **name**>: <row>``, takes the app's translation, so the guide
    and the app say the same thing. Every reused string still has to pass
    :func:`message_problems`.
    """
    sys.path.insert(0, str(ROOT))
    from spacr.qt.i18n import _exact_translation

    glossary = load_glossary(language)
    selected = set(domains or ())
    counts: dict[str, int] = {}
    for path in sorted((locale_dir / language / "LC_MESSAGES").glob("*.po")):
        domain = path.stem
        if selected and domain not in selected:
            continue
        catalog = read_catalog(path)
        changed = 0
        for message in catalog:
            if not message.id or (message.string and not message.fuzzy):
                continue
            candidate = _exact_translation(message.id, language)
            if not candidate:
                match = _PREFIX_RE.match(message.id)
                if match:
                    head, separator, rest = match.groups()
                    body = _exact_translation(rest, language)
                    if head.startswith("**"):
                        name = head[2:-2]
                        shown = glossary.get(name) or _exact_translation(name, language)
                        head = f"**{shown}**" if shown else head
                    candidate = f"{head}{separator}{body}" if body else None
            if not candidate or candidate == message.id:
                continue
            if message_problems(message.id, candidate, glossary):
                continue
            message.string = candidate
            message.flags.discard("fuzzy")
            changed += 1
        if changed:
            write_catalog(path, catalog)
            counts[domain] = changed
    return counts


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit(pot_dir: Path, languages: Iterable[str],
          locale_dir: Path = LOCALE_DIR) -> dict:
    """Coverage and staleness per language and page.

    ``stale`` counts messages whose English changed (fuzzy) or disappeared
    since the catalog was last merged; they render in English. ``invalid``
    lists published translations that fail :func:`message_problems`.
    """
    templates = load_templates(pot_dir)
    report = {"schema": 1,
              "review_kind": "AI technical review; recorded per page, no native-speaker signoff",
              "languages": {}}
    for language in languages:
        glossary = load_glossary(language)
        pages = {}
        invalid = []
        label_missing = []
        review_kinds = {}
        for domain, msgids in templates.items():
            path = locale_dir / language / "LC_MESSAGES" / f"{domain}.po"
            current = {msgid for msgid in msgids
                       if not excluded_message(domain, msgid)}
            if not path.exists():
                pages[domain] = {"total": len(current), "translated": 0,
                                 "stale": 0, "missing": len(current)}
                continue
            review_kinds[domain] = catalog_review_kind(path)
            if review_kinds[domain] not in SUPPORTED_REVIEW_KINDS:
                label_missing.append(domain)
            catalog = read_catalog(path)
            translated = stale = 0
            for message in catalog:
                if not message.id or excluded_message(domain, message.id):
                    continue
                if message.id not in current:
                    if message.string:
                        stale += 1
                    continue
                if message.string and not message.fuzzy:
                    translated += 1
                    problems = message_problems(message.id, message.string, glossary)
                    if problems:
                        invalid.append({"page": domain, "msgid": message.id[:80],
                                        "problems": problems})
                elif message.fuzzy and message.string:
                    stale += 1
            pages[domain] = {"total": len(current), "translated": translated,
                             "stale": stale,
                             "missing": len(current) - translated}
        total = sum(page["total"] for page in pages.values())
        done = sum(page["translated"] for page in pages.values())
        report["languages"][language] = {
            "total": total, "translated": done,
            "coverage": round(done / total, 4) if total else 0.0,
            "stale": sum(page["stale"] for page in pages.values()),
            "invalid": invalid, "label_missing": label_missing,
            "pages": pages,
            "review_kinds": review_kinds,
        }
    return report


# --------------------------------------------------------------------------
# Builds
# --------------------------------------------------------------------------

def _sphinx_python() -> str:
    return os.environ.get("SPACR_SPHINX_PYTHON", sys.executable)


def _env(language: str | None = None, inventory: Path | None = None) -> dict:
    env = dict(os.environ)
    env["SPACR_DOCS_GUIDES_ONLY"] = "1"
    if inventory is not None:
        env["SPACR_DOCS_ENGLISH_INVENTORY"] = str(inventory)
    return env


def extract(output: Path, doctrees: Path | None = None) -> int:
    doctrees = doctrees or output.parent / (output.name + "-doctrees")
    command = [_sphinx_python(), "-m", "sphinx", "-q", "-E", "-b", "gettext",
               "-d", str(doctrees), str(SOURCE_DIR), str(output)]
    return subprocess.call(command, env=_env(), cwd=ROOT)


def build(language: str, output: Path, inventory: Path,
          doctrees: Path | None = None, warnings_are_errors: bool = True) -> int:
    doctrees = doctrees or output.parent / f".doctrees-guide-{language}"
    command = [_sphinx_python(), "-m", "sphinx", "-q", "-E", "-b", "html",
               "-D", f"language={language}", "-d", str(doctrees)]
    if warnings_are_errors:
        command += ["-W", "--keep-going"]
    command += [str(SOURCE_DIR), str(output)]
    return subprocess.call(command, env=_env(language, inventory), cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="write English .pot templates")
    p.add_argument("--output", type=Path, default=ROOT / "docs/_build/guide-gettext")
    p = sub.add_parser("update", help="merge templates into .po catalogs")
    p.add_argument("--pot", type=Path, default=ROOT / "docs/_build/guide-gettext")
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p = sub.add_parser("glossary", help="UI names from the runtime catalogs")
    p.add_argument("--pot", type=Path, default=ROOT / "docs/_build/guide-gettext")
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p = sub.add_parser("export", help="write pending messages as JSON")
    p.add_argument("--language", required=True, choices=LANGUAGES)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--page", action="append")
    p = sub.add_parser("defects", help="snapshot open runtime defects, or "
                       "switch the guides to runtime names that were fixed")
    p.add_argument("action", choices=("snapshot", "retarget"))
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p = sub.add_parser("prefill", help="reuse exact runtime UI translations")
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p.add_argument("--page", action="append")
    p = sub.add_parser("import", help="apply a filled JSON worklist")
    p.add_argument("--language", required=True, choices=LANGUAGES)
    p.add_argument("worklist", type=Path)
    p.add_argument("--strings", type=Path,
                   help="translations as a JSON list aligned with the worklist")
    p.add_argument("--reviewer", choices=tuple(_REVIEWERS),
                   help="record the actual AI reviewer for imported messages")
    p = sub.add_parser("audit", help="coverage, staleness and label report")
    p.add_argument("--pot", type=Path, default=ROOT / "docs/_build/guide-gettext")
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p.add_argument("--output", type=Path)
    p.add_argument("--strict", action="store_true",
                   help="fail on invalid translations or a missing label")
    p = sub.add_parser("build", help="render translated guides with -W")
    p.add_argument("--language", action="append", choices=LANGUAGES)
    p.add_argument("--html", type=Path, default=ROOT / "docs/_build/html",
                   help="English site; each language goes to <html>/<lang>")
    p.add_argument("--doctrees", type=Path)
    args = parser.parse_args(argv)

    languages = tuple(getattr(args, "language", None) or catalog_languages())
    if args.command == "extract":
        return extract(args.output)
    if args.command == "update":
        for language in languages:
            summary = update_language(language, args.pot)
            done = sum(row["translated"] for row in summary.values())
            total = sum(row["total"] for row in summary.values())
            print(f"{language}: {done}/{total} translated")
        return 0
    if args.command == "glossary":
        for language in languages:
            print(f"{language}: {len(build_glossary(language, args.pot))} UI names")
        return 0
    if args.command == "defects":
        for language in languages:
            if args.action == "snapshot":
                print(f"{language}: {len(snapshot_defects(language))} open defects recorded")
            else:
                print(f"{language}: {retarget_fixed_defects(language) or 'no fixed defects'}")
        return 0
    if args.command == "prefill":
        for language in languages:
            counts = prefill_from_runtime(language, domains=args.page)
            print(f"{language}: {sum(counts.values())} messages from the app catalogs "
                  f"{counts}")
        return 0
    if args.command == "export":
        count = export_worklist(args.language, args.output, args.page)
        print(f"{count} pending messages -> {args.output}")
        return 0
    if args.command == "import":
        applied, rejected = import_worklist(args.language, args.worklist, args.strings,
                                           reviewer=args.reviewer)
        print(f"applied {applied}; rejected {len(rejected)}")
        for line in rejected:
            print("  " + line)
        return 1 if rejected else 0
    if args.command == "audit":
        report = audit(args.pot, languages)
        report["generated"] = _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds")
        if args.output:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
        failing = False
        for language, row in report["languages"].items():
            print(f"{language}: {row['translated']}/{row['total']} "
                  f"({row['coverage']:.1%}), stale {row['stale']}, "
                  f"invalid {len(row['invalid'])}, "
                  f"unlabelled {len(row['label_missing'])}")
            failing |= bool(row["invalid"] or row["label_missing"])
        return 1 if (failing and args.strict) else 0
    if args.command == "build":
        inventory = args.html / "objects.inv"
        status = 0
        for language in languages:
            code = build(language, args.html / language, inventory,
                         args.doctrees and args.doctrees / language)
            print(f"{language}: sphinx exit {code}")
            status = status or code
        return status
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
