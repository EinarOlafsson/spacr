"""Ratcheted coverage for spaCR's two runtime-translation layers.

The compact layer owns captions assembled from registries and first-run data:
there is no literal Qt call for the catalog generator to find.  The external
layer owns the much larger static/widget, setting, category and module-summary
surface and binds every translation to a hash of its English source.  These
tests keep the two contracts complementary instead of copying thousands of
generated records into ``i18n._ROWS``.

Deliberate exclusions from the compact surface are language names written in
their own language, provider/product identities, user-entered values and the
generated setting/static-Qt prose.  Native names and product identities are
not translated; generated prose is independently guarded below by the
external source-hash registry.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Post-sweep compact surface on 2026-08-29.  The count catches additions; the
# digest also catches a replacement that happens to keep the count unchanged.
#
# 205 on 2026-08-30, +4/-1.  Admitted: "Field browser" and "Quarantine or
# restore this field", the category and label for the Q key that quarantines a
# field -- it was bound in the QC field browser and on no map, so the cheat
# sheet told the reader it did not exist.  Also "the QC field browser" and
# "the Annotate and Make Masks screens and the QC field browser", the scopes
# that say where Q and the arrows work.  Retired: "the Annotate and Make Masks
# screens", which the longer scope replaces -- the arrows drive the field
# browser too, and the shorter wording had stopped being true.
COMPACT_CAPTION_COUNT = 205
COMPACT_CAPTION_SHA256 = (
    "9ada5b49f7286c502267c4a0cc091dde025e366efb518c72c9a37c027a2d93a8"
)

# The complementary source-bound layer is pinned separately.  Keys are
# catalog record identities (table plus key), not translated values: adding a
# static caption, setting, category, or module summary must therefore move
# this reviewed inventory deliberately even when the builder can generate a
# source hash automatically.
EXTERNAL_SOURCE_COUNTS = {
    "SETTING_LABELS": 1002,
    "SETTING_TOOLTIPS": 997,
    "CATEGORY_HELP": 192,
    "UI": 2625,
    "MODULE_SUMMARIES": 64,
}
EXTERNAL_SOURCE_KEY_SHA256 = (
    "db46475608d02e6c30ffdedab7f8620d165b448d20a5a7c73b2129870ee2aacd"
)

# Calls whose literal argument is chrome owned by the compact catalog on the
# onboarding surfaces.  Their registry/data-driven captions are added by
# ``compact_user_facing_captions`` below as well.
_ONBOARDING_LITERAL_CALLS = {
    "QCheckBox",
    "QLabel",
    "QPushButton",
    "Toggle",
    "_say",
    "addButton",
    "setText",
    "setWindowTitle",
    "tr",
}
_ONBOARDING_PATHS = (
    ROOT / "spacr" / "qt" / "widgets" / "setup_slides.py",
    ROOT / "spacr" / "qt" / "first_run.py",
    ROOT / "spacr" / "qt" / "install_consent.py",
)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _onboarding_literal_captions() -> set[str]:
    """Find independently authored literal chrome on first-run surfaces."""
    found: set[str] = set()
    for path in _ONBOARDING_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _ONBOARDING_LITERAL_CALLS:
                continue
            # ``addButton(caption, role)`` has one caption.  All other calls
            # in this focused set likewise expose their caption first.
            if not node.args:
                continue
            try:
                value = ast.literal_eval(node.args[0])
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
    return found


_CUSTOM_WIDGET_ARGUMENTS = {
    # positional indices, keyword names
    "AiToggleLabel": ((1, 2), {"text", "tooltip"}),
    "Card": ((0, 1), {"title", "subtitle"}),
    "FlatButton": ((0, 2), {"text", "tooltip"}),
    "FlatComboBox": ((1,), {"tooltip"}),
    "FlatSpinBox": ((1,), {"tooltip"}),
    "Toggle": ((0,), {"text"}),
}

# Visible technical identities and formatting shells deliberately excluded
# from both translation layers.  Their exact spelling carries a dimensional,
# backend, field-name or interpolation contract; the separate assertion below
# prevents a term fallback from rewriting them.
_RUNTIME_IDENTITY_CAPTIONS = {
    "%d px",
    "3D",
    '<a href="api">API</a>',
    "Cellpose-SAM",
    "GPU",
    "MIP",
    "RdBu_r",
    "image_path",
    "metadata_column_map.json",
    "png_list",
    "png_path",
    "spaCR",
    "{report}",
    "■ {note}",
}


def _static_string(node: ast.AST, constants: dict[str, ast.AST]) -> str | None:
    """Resolve one literal custom-widget argument without using the builder."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return _static_string(constants[node.id], constants)
    if (
        isinstance(node, ast.Call)
        and _call_name(node) == "tr"
        and node.args
    ):
        return _static_string(node.args[0], constants)
    return None


def _custom_widget_literal_captions() -> set[str]:
    """Independently find literal text carried by spaCR widget constructors."""
    found: set[str] = set()
    for path in sorted((ROOT / "spacr" / "qt").rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants: dict[str, ast.AST] = {}
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                constants[statement.targets[0].id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                constants[statement.target.id] = statement.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            contract = _CUSTOM_WIDGET_ARGUMENTS.get(_call_name(node))
            if contract is None:
                continue
            positions, keywords = contract
            candidates = [
                node.args[index] for index in positions if index < len(node.args)
            ]
            candidates.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in keywords
            )
            for candidate in candidates:
                value = _static_string(candidate, constants)
                if value is not None and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", value):
                    found.add(value.strip())
    return found


def _indirect_registry_captions() -> set[str]:
    """Independently enumerate captions passed through data registries."""
    from spacr.qt.preferences import (
        MODE_LABELS,
        MODE_NOTES,
        MODE_WARNINGS,
        PREFERENCE_TIPS,
    )
    from spacr.qt.preview_registry import PREVIEWS
    from spacr.qt.screens.app_screen import DIMENSION_TOGGLES
    from spacr.qt.screens.batch import ON_ERROR_LABELS
    from spacr.qt.screens.hyperparam import TOGGLE_TEXT, TOGGLE_TOOLTIP
    from spacr.qt.screens.parameter_sweep import (
        SWEEP_TOGGLE_TEXT,
        SWEEP_TOGGLE_TOOLTIP,
    )
    from spacr.qt.widgets.ambient import (
        ANIMATION_CHOICES,
        DRIFT_DIRECTIONS,
        PALETTE_SETS,
        animation_label,
        animation_note,
        drift_direction_label,
        drift_direction_note,
    )
    from spacr.qt.widgets.preview_contract import (
        PREVIEW_BUSY_MESSAGE,
        PREVIEW_CANCEL_TEXT,
        PREVIEW_CANCELLED_MESSAGE,
        PREVIEW_RUN_TEXT,
        PREVIEW_RUNNING_MESSAGE,
        PRIMARY_NOTES,
    )
    from spacr.qt.widgets.preview_controls import (
        ALL_CHANNELS,
        MAX_SETS_TOOLTIP,
    )

    found = set(PREFERENCE_TIPS) | set(PREFERENCE_TIPS.values())
    found.update(MODE_LABELS.values())
    found.update(MODE_NOTES.values())
    found.update(value for value in MODE_WARNINGS.values() if value)
    found.update(label for label, _value in ON_ERROR_LABELS)
    for _dimension, label, tooltip in DIMENSION_TOGGLES:
        found.update((label, tooltip))
    found.update((
        TOGGLE_TEXT,
        TOGGLE_TOOLTIP,
        SWEEP_TOGGLE_TEXT,
        SWEEP_TOGGLE_TOOLTIP,
        ALL_CHANNELS,
        MAX_SETS_TOOLTIP,
        PREVIEW_RUN_TEXT,
        PREVIEW_CANCEL_TEXT,
        PREVIEW_BUSY_MESSAGE,
        PREVIEW_CANCELLED_MESSAGE,
        PREVIEW_RUNNING_MESSAGE,
        "Preview failed: {error}",
        "Channels drawn in {mode} primaries.",
    ))
    found.update(PRIMARY_NOTES.values())
    for spec in PREVIEWS.values():
        found.add(spec.title)
        found.add(
            spec.tooltip
            or "Show a preview of what these settings produce."
        )
    for name in ANIMATION_CHOICES:
        found.update((animation_label(name), animation_note(name)))
    for spec in PALETTE_SETS.values():
        found.update((spec.label, spec.note))
    for name in DRIFT_DIRECTIONS:
        found.update((
            drift_direction_label(name),
            drift_direction_note(name),
        ))
    return {str(value).strip() for value in found if str(value).strip()}


def _shortcut_caption_fields() -> set[str]:
    """Return shortcut copy while deliberately excluding key identifiers."""
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    return {
        value.strip()
        for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
        for value in (spec.label, spec.category, spec.scope)
        if value.strip()
    }


def compact_user_facing_captions() -> frozenset[str]:
    """Discover the complete caption surface that exact ``_ROWS`` owns.

    This predicate intentionally follows semantic UI registries rather than
    reading the catalog it audits: Home app/section names, fold buttons,
    first-run slides/questions/choices, tour text, terms chrome and literal
    onboarding dialog/button captions and shortcut labels, categories and
    scopes.  A caption added to any of those sources therefore enters this
    set before it has a translation row.  Shortcut key identifiers are not
    copy and remain in their platform-native spelling.

    Language choices remain in their own native scripts and installed AI
    provider names remain product identities.  The generic provider fallback
    is prose, so it is included.  Setting help, category help, module
    summaries and other static Qt prose are generated-catalog candidates and
    are covered by :func:`test_every_generated_catalog_candidate_has_a_source_hash`.
    """
    import spacr.qt
    from spacr.qt.app import APPS
    from spacr.qt.first_run import DEFAULT_TOUR
    from spacr.qt.setup_screen import questions
    from spacr.qt.terms import TRANSLATIONS, register_translations
    from spacr.qt.widgets.ambient import (
        ANIMATION_CHOICES,
        animation_label,
    )
    from spacr.qt.widgets.fold_strip import folded_modules
    from spacr.qt.widgets.setup_slides import ANIMATION_LABEL, SLIDES

    # Terms and late app registrations use the same supported registration
    # seam as plugins.  Registering is idempotent and makes the relationship
    # with the runtime ``_ROWS`` object deterministic in any import order.
    # Calling the complete self-registration seam here is also an import-order
    # regression test: a compact fold row that disagrees with its host app's
    # registered metadata raises instead of being hidden in the startup log.
    spacr.qt.register_self_registering_modules()
    register_translations()

    found = _onboarding_literal_captions()
    found.update(name for _key, name, _desc, _section in APPS)
    found.update(section for _key, _name, _desc, section in APPS)
    found.update(entry[0] for entry in folded_modules().values())
    found.update(
        text
        for title, blurb, _keys in SLIDES
        for text in (title, blurb)
    )
    found.add(ANIMATION_LABEL)
    found.update(question[1] for question in questions())
    for key, _label, _getter, _setter, choices in questions():
        if key == "language":
            # Native names are identity labels for readers who cannot yet
            # read the selected UI language.
            continue
        if key == "ai_provider":
            # Product names/logos stay exact.  The empty-value fallback is a
            # sentence fragment and is therefore translated.
            found.update(
                label for value, label in (choices or ()) if not value
            )
            continue
        found.update(label for _value, label in (choices or ()))
    found.update(animation_label(name) for name in ANIMATION_CHOICES)
    found.update(
        text
        for step in DEFAULT_TOUR
        for text in (step.title, step.body)
    )
    found.update(_shortcut_caption_fields())
    found.update(source for source, _values in TRANSLATIONS)
    return frozenset(found)


def test_compact_user_facing_caption_surface_has_exact_rows_and_is_pinned():
    """Every compact caption has one exact ``_ROWS`` row, never a fallback.

    The independent semantic discovery runs before either catalog is read.
    Consequently, adding one of these captions to a generated registry cannot
    make it escape the literal-row requirement: it remains in ``discovered``
    and fails here until an exact row is reviewed.
    """
    from spacr.qt.i18n import _ROWS

    discovered = compact_user_facing_captions()
    missing = sorted(discovered - set(_ROWS))
    assert not missing, (
        "new compact user-facing captions need exact i18n._ROWS entries:\n  "
        + "\n  ".join(repr(value) for value in missing)
    )
    assert len(discovered) == COMPACT_CAPTION_COUNT, (
        f"compact caption surface changed from {COMPACT_CAPTION_COUNT} to "
        f"{len(discovered)}; add/review exact rows, then move the ratchet"
    )
    digest = hashlib.sha256(
        "\0".join(sorted(discovered)).encode("utf-8")
    ).hexdigest()
    assert digest == COMPACT_CAPTION_SHA256, (
        "compact caption set changed without moving its reviewed fingerprint"
    )


def test_compact_and_generated_caption_owners_are_disjoint():
    """A caption belongs to the reviewed compact or generated layer, not both."""
    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import en

    duplicated = sorted(set(_ROWS) & set(en.UI_SOURCES))
    assert not duplicated, (
        "compact captions must not acquire a second generated owner:\n  "
        + "\n  ".join(repr(value) for value in duplicated)
    )


def test_shortcut_copy_enters_the_compact_ratchet_but_keys_do_not():
    """Shortcut labels/categories/scopes are copy; bindings are identities."""
    from spacr.qt.i18n import _ROWS, VALID_LANGUAGE_CODES, tr
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    discovered = compact_user_facing_captions()
    assert _shortcut_caption_fields() <= discovered

    key_identifiers = {
        spec.keys for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
    }
    assert not (key_identifiers & set(_ROWS)), (
        "shortcut bindings must retain QKeySequence/native platform spelling"
    )
    for language in VALID_LANGUAGE_CODES[1:]:
        annotate = tr("Annotate", language)
        make_masks = tr("Make Masks", language)
        joint_scope = tr("the Annotate and Make Masks screens", language)
        assert annotate in joint_scope
        assert make_masks in joint_scope
        assert annotate in tr("the Annotate screen", language)
        assert make_masks in tr("the Make Masks screen", language)


def test_shortcut_overlay_renders_localized_copy_and_native_keys(
    monkeypatch,
    qtbot,
):
    """A newly opened map follows the language without translating bindings."""
    from PySide6.QtWidgets import QLabel, QWidget

    from spacr.qt.shortcuts import ShortcutOverlay, native

    expected = {
        "sv": ("Pensel  —  skärmen Skapa masker", "SKAPA MASKER"),
        "ko": ("브러시  —  마스크 만들기 화면", "마스크 만들기"),
    }
    for language, (brush_text, category_text) in expected.items():
        monkeypatch.setenv("SPACR_LANGUAGE", language)
        window = QWidget()
        window.resize(1400, 900)
        qtbot.addWidget(window)
        overlay = ShortcutOverlay(window)

        labels = overlay.findChildren(QLabel)
        copy = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayLabel"
        }
        categories = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayCategory"
        }
        keys = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayKeys"
        }

        assert brush_text in copy
        assert category_text in categories
        assert native("B") in keys


def test_compact_rows_are_unique_and_dynamic_registries_are_unambiguous():
    """Reject silent dict duplicates and two translations for one caption."""
    path = ROOT / "spacr" / "qt" / "i18n.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "_ROWS"
            and isinstance(statement.value, ast.Dict)
        ):
            keys = [ast.literal_eval(key) for key in statement.value.keys]
            break
    duplicates = sorted(
        key for key, count in Counter(keys).items() if count > 1
    )
    assert keys, "could not locate the literal _ROWS catalog"
    assert not duplicates, f"duplicate _ROWS keys: {duplicates}"

    from spacr.qt.app import APPS, registered_metadata
    from spacr.qt.i18n import _ROWS, _TERM_ROWS
    from spacr.qt.terms import TRANSLATIONS

    names = {key: name for key, name, _desc, _section in APPS}
    dynamic = [
        (names[key], tuple(values))
        for key, values in registered_metadata("translations").items()
    ]
    dynamic.extend((source, tuple(values)) for source, values in TRANSLATIONS)
    conflicts = [
        source
        for source, values in dynamic
        if source not in _ROWS or _ROWS[source] != values
    ]
    assert not conflicts, f"ambiguous/missing registered rows: {conflicts}"
    overlap_conflicts = sorted(
        source
        for source in set(_ROWS) & set(_TERM_ROWS)
        if _ROWS[source] != _TERM_ROWS[source]
    )
    assert not overlap_conflicts, (
        f"exact and term catalogs disagree for: {overlap_conflicts}"
    )


def test_spanish_compact_rows_use_consistent_formal_register():
    """User instructions must not mix informal Spanish into formal chrome."""
    from spacr.qt.i18n import _ROWS

    spanish = {source: values[2] for source, values in _ROWS.items()}
    informal_pronoun = re.compile(
        r"\b(?:tú|tu|tus|te|ti|contigo|puedes|estés|hayas|veas)\b",
        re.IGNORECASE,
    )
    rejected_phrases = (
        "Abre una incidencia",
        "Activa o desactiva el UMAP",
        "Borra el cuadro",
        "Carga con un clic",
        "Ejecuta spacr-doctor",
        "Genera un conjunto",
        "Haz clic",
        "Pasa el cursor",
        "Pulsa Esc",
        "Suelta una carpeta",
        "También puedes",
        "Usa Demostraciones",
        "cuando pulsas Enviar",
        "ejecútalo en una terminal",
        "elige un conjunto",
        "en tu navegador",
        "introduce {code}",
        "para que veas",
        "si cambias de opinión",
        "selecciona ⓘ",
        "te muestra",
        "Úsalos",
    )
    offenders = {
        source: target
        for source, target in spanish.items()
        if informal_pronoun.search(target)
        or any(phrase.casefold() in target.casefold()
               for phrase in rejected_phrases)
    }
    assert not offenders, f"informal Spanish compact captions: {offenders}"


def test_every_generated_catalog_candidate_has_a_source_hash():
    """The non-compact UI surface is complete in the external registry."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n_catalogs import en

    candidates = set(builder.extract_static_ui_sources())
    missing_sources = sorted(candidates - set(en.UI_SOURCES))
    assert not missing_sources, (
        "generated UI candidates missing from en.UI_SOURCES:\n  "
        + "\n  ".join(repr(value) for value in missing_sources)
    )
    stale = sorted(
        source
        for source in candidates
        if en.SOURCE_HASHES.get(("UI", source))
        != hashlib.sha256(source.encode("utf-8")).hexdigest()
    )
    assert not stale, (
        "generated UI candidates missing current source hashes:\n  "
        + "\n  ".join(repr(value) for value in stale)
    )


def test_external_caption_layer_is_complete_exclusive_and_pinned():
    """Every non-compact caption has one reviewed source-bound record.

    This is the generated layer's counterpart to the literal ``_ROWS``
    ratchet.  A newly discovered caption cannot pass merely because catalog
    generation notices it: its table/key changes this count or digest and
    requires an explicit review and ratchet update.
    """
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import en

    canonical = builder.canonical_sources()
    external = {
        "SETTING_LABELS": canonical["setting_labels"],
        "SETTING_TOOLTIPS": canonical["setting_tooltips"],
        "CATEGORY_HELP": canonical["categories"],
        "UI": canonical["ui"],
        "MODULE_SUMMARIES": canonical["module_summaries"],
    }
    counts = {table: len(records) for table, records in external.items()}
    assert counts == EXTERNAL_SOURCE_COUNTS, (
        "external caption inventory changed; review every new/removed record "
        f"before moving the ratchet: {counts}"
    )

    identities = sorted(
        (table, str(key))
        for table, records in external.items()
        for key in records
    )
    digest = hashlib.sha256(
        "\0".join(
            f"{table}\0{key}" for table, key in identities
        ).encode("utf-8")
    ).hexdigest()
    assert digest == EXTERNAL_SOURCE_KEY_SHA256, (
        "external caption identities changed without moving their reviewed "
        "fingerprint"
    )

    english = {
        "SETTING_LABELS": en.SETTING_LABELS,
        "SETTING_TOOLTIPS": en.SETTING_TOOLTIPS,
        "CATEGORY_HELP": en.CATEGORY_SOURCES,
        "UI": en.UI_SOURCES,
        "MODULE_SUMMARIES": en.MODULE_SUMMARIES,
    }
    assert {
        table: set(records) for table, records in external.items()
    } == {
        table: set(records) for table, records in english.items()
    }
    assert not (set(_ROWS) & set(external["UI"])), (
        "compact and source-bound caption layers must be disjoint"
    )


def test_custom_widgets_and_indirect_registries_enter_one_i18n_layer():
    """Dynamic/custom captions have exactly one explicit reviewed owner."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES, en

    discovered = set(builder.extract_static_ui_sources())
    independently_expected = (
        _custom_widget_literal_captions() | _indirect_registry_captions()
    )
    missing_ownership = sorted(
        independently_expected
        - _RUNTIME_IDENTITY_CAPTIONS
        - set(_ROWS)
        - discovered
    )
    assert not missing_ownership, (
        "custom/indirect runtime captions bypass both i18n layers:\n  "
        + "\n  ".join(repr(value) for value in missing_ownership)
    )

    ambiguous_ownership = sorted(
        source
        for source in independently_expected - _RUNTIME_IDENTITY_CAPTIONS
        if int(source in _ROWS) + int(source in en.UI_SOURCES) != 1
    )
    assert not ambiguous_ownership, (
        "custom/indirect captions need exactly one compact or generated "
        "owner:\n  "
        + "\n  ".join(repr(value) for value in ambiguous_ownership)
    )

    external = (
        independently_expected - _RUNTIME_IDENTITY_CAPTIONS - set(_ROWS)
    )
    assert external <= set(en.UI_SOURCES)
    for language in CATALOG_LANGUAGES:
        module = import_module(f"spacr.qt.i18n_catalogs.{language}")
        missing = sorted(external - set(module.UI))
        assert not missing, f"{language} lacks indirect UI rows: {missing}"
        stale = sorted(
            source
            for source in external
            if module.SOURCE_HASHES.get(("UI", source))
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        assert not stale, f"{language} has stale indirect UI hashes: {stale}"
        blank = sorted(source for source in external if not module.UI[source].strip())
        assert not blank, f"{language} has blank indirect UI rows: {blank}"


def test_runtime_identity_captions_remain_exact_in_every_language():
    """Technical names, field names and formatting shells stay exact."""
    from spacr.qt.i18n import VALID_LANGUAGE_CODES, tr

    for language in VALID_LANGUAGE_CODES:
        assert {
            source: tr(source, language)
            for source in _RUNTIME_IDENTITY_CAPTIONS
        } == {source: source for source in _RUNTIME_IDENTITY_CAPTIONS}


def test_dynamic_preview_messages_are_rendered_from_localized_templates(
    monkeypatch,
):
    """Runtime-formatted preview status text must use catalog templates."""
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES
    from spacr.qt.widgets.preview_contract import (
        PREVIEW_BUSY_MESSAGE,
        PRIMARY_NOTES,
        LivePreviewContract,
        preview_failure_message,
    )

    class Label:
        def __init__(self):
            self.value = ""

        def setText(self, value):  # noqa: N802 - minimal Qt label contract
            self.value = str(value)

    class Panel(LivePreviewContract):
        def __init__(self):
            self._status = Label()

        def display_primaries(self):
            return "rgb"

    for language in CATALOG_LANGUAGES:
        monkeypatch.setattr(
            i18n, "current_language", lambda code=language: code
        )
        failure = preview_failure_message("E42")
        expected_failure = i18n.tr(
            "Preview failed: {error}", language, error="E42"
        )
        assert failure == expected_failure
        assert failure != "Preview failed: E42"

        panel = Panel()
        panel.set_preview_status(PREVIEW_BUSY_MESSAGE)
        assert panel._status.value == i18n.tr(
            PREVIEW_BUSY_MESSAGE, language
        )
        assert panel._status.value != PREVIEW_BUSY_MESSAGE

        panel.display_primaries = lambda: "tritanope"
        note = panel.display_primaries_note()
        assert note == i18n.tr(PRIMARY_NOTES["tritanope"], language)
        assert note != PRIMARY_NOTES["tritanope"]
