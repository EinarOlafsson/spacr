"""Generate the data the Help search field matches against.

Instruction 422. The search box beside the Help menu has to find a module, a
setting, an API entry or a preference by name -- and the one thing that would
make it worse than useless is a hand-written list, because the first rename
sends a user somewhere the thing is not, and a search result is BELIEVED.

So three of the four indexes are generated here from artefacts that already
exist, and the fourth (modules and settings) is read from the live registries
at runtime because those are already data rather than prose.

What this writes into ``spacr/qt/help_api_index.py``:

``API_ENTRIES``
    ``(dotted symbol, summary)`` for every PUBLIC symbol in
    ``docs/source/_static/i18n/api/en.json`` -- the manifest
    ``tools/build_documentation_i18n.py`` writes, which is the same list the
    published API pages are built from. The summary is the first paragraph of
    the English docstring, squeezed onto one line and truncated, so the field
    can match on what a symbol DOES and not only on what it is called.

``SETTING_CONSUMERS``
    ``key -> ((module, qualname), ...)`` from ``docs/setting_consumers.json``.
    ``spacr/qt/screens/setting_api_targets.py`` already names ONE consumer per
    setting, which is what a settings row links to; the search field promises
    "the API entry of EACH function that takes that setting", so it needs all
    of them. Closures are dropped: Sphinx cannot address one, so a link to it
    would be a dead link with extra steps.

``PREFERENCE_ENTRIES``
    ``(label, tab title, tab object name, tooltip)`` read off the REAL
    preferences dialog, built once in a child process.

    AN AST WALK WAS TRIED FIRST AND MEASURED SHORT. The dialog has no schema
    -- it is seven thousand lines of procedure -- and parsing the procedure
    found 60 of the 121 rows the dialog actually renders. It missed every row
    whose caption is a variable (``Starfield direction``), every row a loop
    builds from a table (the four Performance budget buttons), and every row
    inside a nested panel that has a form of its own (``Font family`` and the
    rest of the figure style block, 40 rows on the Figures tab alone). It
    also produced one row captioned ``First graph — {shape}``, which is a
    template and not a caption a user will ever see.

    Building the dialog answers all of that by construction, so that is what
    this does. The child gets its own ``HOME``, ``XDG_CONFIG_HOME`` and
    ``XDG_DATA_HOME`` under a temporary directory: a tool that constructs
    ``QSettings`` against the real store is how a maintainer's ``qt.conf``
    got six entries wiped on 2026-09-19, and this one cannot reach it.

Run::

    python tools/build_help_search_index.py           # rewrite the module
    python tools/build_help_search_index.py --check   # fail if it is stale

:returns: 0 when the generated module is what this script would write.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
API_MANIFEST = ROOT / "docs" / "source" / "_static" / "i18n" / "api" / "en.json"
CONSUMER_MAP = ROOT / "docs" / "setting_consumers.json"
OUTPUT = ROOT / "spacr" / "qt" / "help_api_index.py"

#: How much of a docstring's first paragraph is worth carrying. Long enough
#: that the sentence still reads as a sentence in a result row, short enough
#: that ten thousand of them stay a file rather than a database.
SUMMARY_LIMIT = 140


def _summary(text: str) -> str:
    """The first paragraph of ``text`` as one truncated line.

    :param text: a docstring, or ``""``.
    :returns: the summary, possibly ending in an ellipsis.
    """
    first = str(text or "").strip().split("\n\n")[0]
    squeezed = " ".join(first.split())
    if len(squeezed) <= SUMMARY_LIMIT:
        return squeezed
    return squeezed[: SUMMARY_LIMIT - 1].rstrip() + "…"


def _is_public(symbol: str) -> bool:
    """Whether every component of a dotted name is public.

    ``spacr.qt.app.MainWindow._build_menu_bar`` has no API page, so offering
    it as a result is offering a link to nothing.

    :param symbol: a dotted symbol name.
    :returns: True when no component starts with an underscore.
    """
    return all(
        part == "__main__" or not part.startswith("_")
        for part in symbol.split(".")
    )


def api_entries() -> List[Tuple[str, str]]:
    """Every public API symbol with a one-line summary.

    :returns: ``(symbol, summary)`` sorted by symbol.
    :raises SystemExit: when the manifest has not been built.
    """
    if not API_MANIFEST.is_file():
        raise SystemExit(
            f"{API_MANIFEST} is missing -- run "
            "tools/build_documentation_i18n.py --sources-only first")
    payload = json.loads(API_MANIFEST.read_text(encoding="utf-8"))
    symbols = payload.get("symbols") or {}
    out: List[Tuple[str, str]] = []
    for symbol in sorted(symbols):
        if not _is_public(symbol):
            continue
        record = symbols[symbol]
        text = record.get("text", "") if isinstance(record, dict) else ""
        out.append((symbol, _summary(text)))
    return out


def setting_consumers() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """Every addressable function that reads each setting.

    :returns: ``key -> ((module, qualname), ...)``, sorted and deduplicated.
    :raises SystemExit: when the consumer map has not been built.
    """
    if not CONSUMER_MAP.is_file():
        raise SystemExit(
            f"{CONSUMER_MAP} is missing -- run "
            "tools/build_setting_consumer_map.py first")
    payload = json.loads(CONSUMER_MAP.read_text(encoding="utf-8"))
    consumers = payload.get("consumers") or {}
    out: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    for key in sorted(consumers):
        seen: List[Tuple[str, str]] = []
        for record in consumers[key]:
            if not isinstance(record, dict) or record.get("nested"):
                continue
            module = str(record.get("module") or "")
            qualname = str(record.get("qualname") or "")
            if not module or not qualname or "." in qualname:
                continue
            if not _is_public(f"{module}.{qualname}"):
                continue
            pair = (module, qualname)
            if pair not in seen:
                seen.append(pair)
        if seen:
            out[key] = tuple(sorted(seen))
    return out


#: Printed by the child process immediately before the JSON payload, so a Qt
#: warning on stdout cannot be mistaken for the answer.
PREFERENCE_MARKER = "---PREFERENCE-ROWS---"


def _catch_the_tooltips_before_they_move() -> Dict[str, str]:
    """Snapshot each row's tooltip before the dialog sweeps them into the strip.

    A FINISHED PREFERENCES DIALOG HAS NO TOOLTIPS. ``explain_every_row``
    moves a row's sentence from its field onto its label, and
    ``_everything_explains_itself_in_the_strip`` then moves every remaining
    one into the hint bar -- deliberately, so a control explains itself once
    rather than twice. Reading the built dialog therefore found the
    description of exactly none of its 121 rows.

    So the sweep is wrapped rather than fought: the original still runs and
    the dialog is still built exactly as a user gets it, and this keeps a
    copy of what each row said on the way past.

    :returns: the dict the wrapper fills, caption -> sentence. It is empty
        until the dialog is built.
    """
    from PySide6.QtWidgets import QFormLayout, QLabel

    from spacr.qt import preferences

    caught: Dict[str, str] = {}
    original = preferences.explain_every_row

    def explain_every_row(dialog):
        """Record every row's sentence, then explain the rows as usual."""
        for form in dialog.findChildren(QFormLayout):
            for index in range(form.rowCount()):
                label_item = form.itemAt(index, QFormLayout.LabelRole)
                field_item = form.itemAt(index, QFormLayout.FieldRole)
                label = label_item.widget() if label_item else None
                field = field_item.widget() if field_item else None
                if not isinstance(label, QLabel):
                    continue
                caption = label.text().replace("&", "").strip()
                if not caption or caught.get(caption):
                    continue
                tip = _tooltip_under(field, label)
                if tip:
                    caught[caption] = tip
        return original(dialog)

    preferences.explain_every_row = explain_every_row
    return caught


def _tooltip_under(*widgets) -> str:
    """The first tooltip on a row, looking inside its wrappers.

    A preferences row is often ``addRow(label, _hbox_wrap(column))``, so the
    widget in the form has no tooltip of its own and the sentence that says
    what the preference DOES is two levels in, on the slider.

    :param widgets: widgets to search, outermost first.
    :returns: the first non-empty tooltip found, or ``""``.
    """
    from PySide6.QtWidgets import QWidget

    for widget in widgets:
        if widget is None:
            continue
        if widget.toolTip():
            return str(widget.toolTip())
        for child in widget.findChildren(QWidget):
            if child.toolTip():
                return str(child.toolTip())
    return ""


def _description_of(label: str, caught: Dict[str, str], *widgets) -> str:
    """What a preference row says it does.

    Three sources, in the order they are trustworthy:
    ``spacr.qt.preferences.PREFERENCE_TIPS`` (the registry, keyed by
    caption), then the tooltip caught on its way into the hint strip, then
    whatever is still on the widgets.

    :param label: the row's caption.
    :param caught: what :func:`_catch_the_tooltips_before_they_move` kept.
    :param widgets: the row's field widget, then its caption widget.
    :returns: the description, or ``""``.
    """
    from spacr.qt.preferences import PREFERENCE_TIPS

    return str(PREFERENCE_TIPS.get(label, "") or caught.get(label, "")
               or _tooltip_under(*widgets))


def _collect_preference_rows() -> List[Tuple[str, str, str, str]]:
    """Build the real preferences dialog and read its rows off it.

    Runs in the CHILD process. The fractal backdrop is switched on first --
    process-locally, through :func:`spacr.qt.theme.enable_spaceout`, which
    saves nothing -- because the Fractal tab is built only when it is on and
    those thirteen preferences are as searchable as any other while it is on.

    :returns: ``(label, tab title, tab object name, tooltip)`` per row.
    """
    from PySide6.QtWidgets import (
        QApplication, QFormLayout, QLabel, QTabWidget, QWidget,
    )

    app = QApplication.instance() or QApplication([])
    from spacr.qt.theme import enable_spaceout
    from spacr.qt.preferences import PreferencesDialog

    caught = _catch_the_tooltips_before_they_move()
    enable_spaceout()
    dialog = PreferencesDialog(None)
    app.processEvents()

    tabs = dialog.findChild(QTabWidget, "PreferencesTabs") \
        or dialog.findChild(QTabWidget)
    rows: List[Tuple[str, str, str, str]] = []
    seen: set = set()
    if tabs is None:
        return rows
    for index in range(tabs.count()):
        title = tabs.tabText(index).replace("&", "")
        holder = tabs.widget(index)
        if holder is None:
            continue
        pages = [w for w in holder.findChildren(QWidget)
                 if str(w.objectName()).startswith("PreferencesTab")]
        if str(holder.objectName()).startswith("PreferencesTab"):
            pages.insert(0, holder)
        for page in pages:
            object_name = str(page.objectName())
            for form in page.findChildren(QFormLayout):
                for row in range(form.rowCount()):
                    item = form.itemAt(row, QFormLayout.LabelRole)
                    caption = item.widget() if item is not None else None
                    if not isinstance(caption, QLabel):
                        continue
                    label = caption.text().replace("&", "").strip()
                    if not label or (label, object_name) in seen:
                        continue
                    seen.add((label, object_name))
                    field = form.itemAt(row, QFormLayout.FieldRole)
                    widget = field.widget() if field is not None else None
                    rows.append((label, title, object_name,
                                 _description_of(label, caught, widget,
                                                 caption)))
    dialog.deleteLater()
    app.processEvents()
    rows.sort(key=lambda row: (row[1], row[0]))
    return rows


def preference_entries() -> List[Tuple[str, str, str, str]]:
    """Every labelled row of the preferences dialog, from the dialog itself.

    Spawns this same script with ``--emit-preferences`` in a process whose
    ``HOME`` and XDG directories point at a throwaway tree, so nothing it
    constructs can reach the preferences of whoever is running it.

    :returns: ``(label, tab title, tab object name, tooltip)`` per row.
    :raises SystemExit: when the child could not build the dialog.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="spacr-help-index-") as sandbox:
        environment = dict(os.environ)
        environment.update({
            "HOME": sandbox,
            "XDG_CONFIG_HOME": os.path.join(sandbox, "config"),
            "XDG_DATA_HOME": os.path.join(sandbox, "data"),
            "XDG_CACHE_HOME": os.path.join(sandbox, "cache"),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(ROOT),
        })
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--emit-preferences"],
            capture_output=True, env=environment, cwd=str(ROOT), timeout=600)
    if completed.returncode != 0:
        raise SystemExit(
            "the preferences dialog could not be built:\n"
            + completed.stderr.decode("utf-8", "replace")[-4000:])
    text = completed.stdout.decode("utf-8", "replace")
    if PREFERENCE_MARKER not in text:
        raise SystemExit(
            "the preferences child printed no rows:\n" + text[-4000:])
    payload = text.split(PREFERENCE_MARKER, 1)[1]
    return [tuple(row) for row in json.loads(payload)]


def _render(api: Sequence[Tuple[str, str]],
            consumers: Dict[str, Tuple[Tuple[str, str], ...]],
            preferences: Sequence[Tuple[str, str, str, str]]) -> str:
    """Compose the generated module.

    :param api: rows for ``API_ENTRIES``.
    :param consumers: rows for ``SETTING_CONSUMERS``.
    :param preferences: rows for ``PREFERENCE_ENTRIES``.
    :returns: the full text of ``spacr/qt/help_api_index.py``.
    """
    lines: List[str] = [
        '"""What the Help search field matches on. Generated -- do not edit.',
        "",
        "Written by ``tools/build_help_search_index.py`` from the API manifest,",
        "the setting consumer map and the source of the preferences dialog. See",
        "``docs/notes/spacr/qt/help_api_index.md`` for why none of it is typed",
        "by hand.",
        '"""',
        "",
        "API_ENTRIES = (",
    ]
    for symbol, summary in api:
        lines.append(f"    ({symbol!r}, {summary!r}),")
    lines.append(")")
    lines.append("")
    lines.append("SETTING_CONSUMERS = {")
    for key in sorted(consumers):
        pairs = ", ".join(f"({m!r}, {q!r})" for m, q in consumers[key])
        lines.append(f"    {key!r}: ({pairs},),")
    lines.append("}")
    lines.append("")
    lines.append("PREFERENCE_ENTRIES = (")
    for label, title, object_name, tip in preferences:
        lines.append(
            f"    ({label!r}, {title!r}, {object_name!r}, {tip!r}),")
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def build() -> str:
    """Compose the generated module from every source.

    :returns: the text that belongs in :data:`OUTPUT`.
    """
    return _render(api_entries(), setting_consumers(), preference_entries())


def main(argv: Sequence[str] | None = None) -> int:
    """Write the generated module, or check that it is current.

    :param argv: command-line arguments; ``sys.argv[1:]`` when ``None``.
    :returns: process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 when the committed module is not what this would write")
    parser.add_argument(
        "--emit-preferences", action="store_true",
        help="print the preference rows as JSON; used by the child process "
             "this script spawns, not by hand")
    args = parser.parse_args(argv)
    if args.emit_preferences:
        rows = _collect_preference_rows()
        print(PREFERENCE_MARKER)
        print(json.dumps(rows))
        return 0
    text = build()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(ROOT)} is stale -- rerun "
                  "tools/build_help_search_index.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} is current")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
