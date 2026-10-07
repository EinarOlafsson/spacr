"""Shared appearance and visible-path checks for the current tutorial pass."""
from __future__ import annotations

import re
import sys
from pathlib import Path

CAPTURE_THEME = "dark"
CAPTURE_BACKDROP = "blobs"
_PRIVATE_PATH = re.compile(r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s]+|/mnt/[^\s<>\"']+|/(?:run/)?media/[^\s<>\"']+|/nas_mnt(?:/[^\s<>\"']*)?)")


#: Preferences -> Show alpha features, as QSettings stores it
#: (``spacr.qt.preferences._KEY_SHOW_ALPHA_FEATURES``). Everything registered
#: in ``spacr.settings.ALPHA_FEATURES`` is hidden while it is off. The
#: maintainer's rule: no alpha feature gets a tutorial, so every recording
#: runs with it off. Only a Preferences-lesson scene that shows the toggle
#: itself may opt in (``allow_alpha_toggle_scene``).
ALPHA_FEATURES_KEY = "prefs/show_alpha_features"
#: Preferences -> Show alpha species (e86e25966): the alpha organism pages
#: (Plasmodium, Candida, Trypanosoma, Leishmania, Giardia, Virus, Mammalian).
#: Every recording runs with it off too; only Toxoplasma may appear.
ALPHA_SPECIES_KEY = "prefs/show_alpha_species"

#: The maintainer's exception (2026-10-04): exactly these lessons may show
#: alpha content, each only the kind named here ("species": the organism
#: pages behind Show alpha species; "features": Show alpha features). Every
#: other lesson stays alpha OFF. test_no_alpha_feature_tutorials reads this
#: same table, so the allow-list lives in one place.
ALPHA_LESSONS = {
    "86_alpha_organism_modules": "species",
    "87_alpha_features": "features",
}


#: Session restore (647/648): an ordinary start reopens the last module, its
#: settings and its folder (Preferences -> Session, default on), and unsaved
#: settings autosave as drafts that are offered back after a crash. A
#: recording must show exactly what a viewer gets on a fresh start, so every
#: recording starts as ``spacr --fresh`` with no remembered session and no
#: crash drafts to offer. The switch itself stays at its default (on), so
#: Preferences scenes show the default truthfully.
RESTORE_SESSION_KEY = "prefs/restore_last_session"
SESSION_KEY = "session/last"
DRAFTS_KEY = "session/drafts"
FRESH_FLAG = "--fresh"

#: 651/652: a launch running a version newer than the one last seen shows a
#: "What's new" dialog. Every capture profile marks the running version as
#: seen, so the dialog never appears in a recording, and capture refuses a
#: frame while it is open.
WHATS_NEW_SEEN_KEY = "updates/last_seen_version"
WHATS_NEW_PREVIOUS_KEY = "updates/previous_version"
WHATS_NEW_DIALOG = "WhatsNewDialog"


def running_version():
    """The installed spaCR version the way the app reads it, or ``"unknown"``.

    Reads package metadata only (``spacr.updater._installed_version``), so
    shell launchers need not import spaCR.
    """
    from importlib.metadata import PackageNotFoundError, version

    for name in ("spacr", "spacr-nightly"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


def _mark_whats_new_seen(store, version=None):
    """Store ``version`` (default: the running one) as the last seen."""
    version = version or running_version()
    if version and version != "unknown":
        store.setValue(WHATS_NEW_SEEN_KEY, version)
    store.remove(WHATS_NEW_PREVIOUS_KEY)
    return version


def fresh_argv(argv=()):
    """``argv`` for a spaCR launch that starts fresh (``--fresh`` first, once)."""
    return [FRESH_FLAG, *[arg for arg in argv if arg != FRESH_FLAG]]


def launcher_accepts_fresh(app_source):
    """Whether the spaCR whose ``spacr/qt/app.py`` is ``app_source`` knows --fresh.

    Older released packages (install recordings) predate the flag and would
    read it as a module name; their brand-new profiles have nothing to reopen.
    """
    try:
        return "_OPEN_FRESH" in Path(app_source).read_text(errors="replace")
    except OSError:
        return False


def force_fresh_start():
    """Make the in-process app start as ``spacr --fresh`` with no session to reopen.

    In-process recorders build ``MainWindow`` without ``main()``, so the
    command-line flag is set the way ``main()`` sets it. Also removes the
    remembered session and any crash drafts, so nothing is reopened and no
    recovery offer appears, and marks the running version as seen so no
    "What's new" dialog opens. Preferences -> Session is left at its default
    (on) so Preferences scenes show it truthfully. Call before the main
    window is built. Returns True when the app has session restore at all.
    """
    from spacr.qt import app as gui
    from spacr.qt import preferences as prefs

    flag = getattr(gui, "_OPEN_FRESH", None)
    if flag is None:
        return False
    flag[0] = True
    store = prefs._settings()
    store.remove(RESTORE_SESSION_KEY)            # back to the default (on)
    store.remove(SESSION_KEY)
    store.remove(DRAFTS_KEY)
    _mark_whats_new_seen(store)                  # no "What's new" dialog
    store.sync()
    return True


def force_fresh_start_in_profiles(config_homes, version=None):
    """Drop the remembered session and crash drafts from each profile.

    Preferences -> Session is put back to its default (on): the fresh start
    comes from ``--fresh`` and the empty session state, not from the switch.
    The running spaCR version (or ``version``) is marked as seen, so no
    "What's new" dialog opens.

    For launchers that prepare a profile before the app starts. Each
    ``config_home`` is an ``XDG_CONFIG_HOME`` (store ``spacr/qt.conf``).

    :returns: the store files written.
    """
    from PySide6.QtCore import QSettings

    written = []
    for config_home in config_homes:
        path = Path(config_home) / "spacr" / "qt.conf"
        path.parent.mkdir(parents=True, exist_ok=True)
        store = QSettings(str(path), QSettings.IniFormat)
        store.remove(RESTORE_SESSION_KEY)        # the default (on)
        store.remove(SESSION_KEY)
        store.remove(DRAFTS_KEY)
        _mark_whats_new_seen(store, version)     # no "What's new" dialog
        store.sync()
        if store.status() != QSettings.NoError:
            raise RuntimeError(f"Cannot clear the remembered session in {path}")
        written.append(path)
    return written


def verify_profile_starts_fresh(config_home):
    """Refuse a profile whose next spaCR start would reopen a session.

    For recordings that launch an installed spaCR the way a user types it
    (``spacr``, without ``--fresh``) in a brand-new home: the profile must
    hold no remembered module and no crash drafts. Reads the store as text,
    so the recorder needs no Qt. Returns the store path checked.
    """
    path = Path(config_home) / "spacr" / "qt.conf"
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return path
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key = line.split("=", 1)[0].strip()
        if section == "session" and key == "drafts":
            raise RuntimeError(
                f"Capture refused: {path} holds crash drafts spaCR would offer")
        if section == "session" and key == "last" and re.search(
                r'\\?"module\\?"\s*:\s*\\?"[^"\\]', line):
            raise RuntimeError(
                f"Capture refused: {path} would reopen the last session")
    return path

def verify_no_whats_new(window=None):
    """Refuse while a "What's new" dialog is open (651/652)."""
    candidates = []
    if window is not None:
        candidates.append(getattr(window, "_whats_new_dialog", None))
    app_module = sys.modules.get("PySide6.QtWidgets")
    app = app_module.QApplication.instance() if app_module is not None else None
    if app is not None:
        candidates.extend(app.topLevelWidgets())
    for widget in candidates:
        try:
            shown = (widget is not None and widget.objectName() == WHATS_NEW_DIALOG
                     and widget.isVisible())
        except RuntimeError:                      # already deleted
            shown = False
        if shown:
            raise RuntimeError(
                "Capture refused: the \"What's new\" dialog is open; capture "
                "profiles mark the running version as seen")

def refuse_whats_new_titles(titles):
    """Refuse when an X window title shows the "What's new" dialog.

    For recordings of an installed spaCR in another process, where only the
    window list is visible. A brand-new profile's first launch only
    remembers its version, so the dialog should never be there.
    """
    for title in titles:
        if str(title).startswith("What's new"):
            raise RuntimeError(
                f"Capture refused: the \"What's new\" dialog is open ({title!r})")

def verify_fresh_start(window=None):
    """Refuse to record when spaCR started in restore mode.

    Also refused while a "What's new" dialog is open. Refused when the app
    was not started fresh (``--fresh`` or
    :func:`force_fresh_start`), when the window reopened a remembered
    session, or when crash drafts are offered. Preferences -> Session may be
    on (its default): ``--fresh`` skips it.

    :returns: True when checked; False for an app without session restore.
    """
    verify_no_whats_new(window)
    gui = sys.modules.get("spacr.qt.app")
    flag = getattr(gui, "_OPEN_FRESH", None) if gui is not None else None
    if flag is None:
        return False
    if not flag[0]:
        raise RuntimeError(
            "Capture refused: spaCR started in restore mode. Recordings start "
            "with --fresh (force_fresh_start) and no remembered session")
    if window is not None:
        restored = getattr(window, "_restored_session_settings", ("", {}))
        if restored and restored[0]:
            raise RuntimeError(
                f"Capture refused: spaCR reopened the last session ({restored[0]})")
        box = getattr(window, "_drafts_box", None)
        try:
            offered = box is not None and box.isVisible()
        except RuntimeError:                      # the dialog was deleted
            offered = False
        if offered or getattr(window, "_pending_drafts", None):
            raise RuntimeError(
                "Capture refused: spaCR is offering unsaved settings from a crash")
    return True

def alpha_lesson_kind(lesson_id):
    """The alpha kind ``lesson_id`` may show, or None for every other lesson."""
    return ALPHA_LESSONS.get(lesson_id)


def _alpha_features_shown():
    """Whether the running app would show alpha features right now."""
    from spacr.qt import preferences as prefs

    getter = getattr(prefs, "_get_show_alpha_features", None)
    if getter is not None:
        return bool(getter())
    # An app that predates the gate has no alpha features to show.
    return False


def force_alpha_features_off():
    """Turn Show alpha features off in the isolated Qt store.

    Call before the main window is built: screens read the gate as they are
    constructed. Returns True when the app has the preference at all.
    """
    from spacr.qt import preferences as prefs

    species = getattr(prefs, "_set_show_alpha_species", None)
    if species is not None:
        species(False)
    setter = getattr(prefs, "_set_show_alpha_features", None)
    if setter is None:
        return False
    setter(False)
    return True


def force_alpha_features_off_in_profiles(config_homes):
    """Write Show alpha features = off into each private profile's store.

    For launchers that prepare a profile before the app starts (the neutral
    capture wrapper). Each ``config_home`` is an ``XDG_CONFIG_HOME``; the
    spaCR store under it is ``spacr/qt.conf`` (QSettings organisation
    ``spacr``, application ``qt``). Other keys are kept.

    :returns: the store files written.
    """
    from PySide6.QtCore import QSettings

    written = []
    for config_home in config_homes:
        path = Path(config_home) / "spacr" / "qt.conf"
        path.parent.mkdir(parents=True, exist_ok=True)
        store = QSettings(str(path), QSettings.IniFormat)
        store.setValue(ALPHA_FEATURES_KEY, False)
        store.setValue(ALPHA_SPECIES_KEY, False)
        store.sync()
        if store.status() != QSettings.NoError:
            raise RuntimeError(f"Cannot turn alpha features off in {path}")
        written.append(path)
    return written


def verify_alpha_features_off(*, allow_alpha_toggle_scene=False, alpha_lesson=None):
    """Refuse a frame while the running app shows alpha features.

    :param allow_alpha_toggle_scene: the explicit opt-in for a Preferences
        lesson scene that shows the Show alpha features toggle itself.
    :param alpha_lesson: the lesson being recorded; only a lesson listed in
        :data:`ALPHA_LESSONS` as ``"species"`` may record with Show alpha
        species on.
    :returns: whether alpha features are shown (only ever True with the
        opt-in).
    """
    from spacr.qt import preferences as prefs

    species = getattr(prefs, "_get_show_alpha_species", None)
    if species is not None and species() and alpha_lesson_kind(alpha_lesson) != "species":
        raise RuntimeError(
            "Capture refused: Preferences -> Show alpha species is on. Only "
            "Toxoplasma may appear in a tutorial; record with it off")
    shown = _alpha_features_shown()
    if shown and not allow_alpha_toggle_scene and alpha_lesson_kind(alpha_lesson) != "features":
        raise RuntimeError(
            "Capture refused: Preferences -> Show alpha features is on. Alpha "
            "features get no tutorials; record with it off (only a Preferences "
            "scene showing the toggle itself may opt in)")
    return shown


def configure_appearance(theme=CAPTURE_THEME, backdrop=CAPTURE_BACKDROP, *, home_preferences=False):
    """Set the requested recording appearance in the isolated Qt store.

    Also turns Show alpha features off: no recording shows alpha features.
    And starts the app fresh (``--fresh``, no remembered session, no crash
    drafts; Preferences -> Session stays at its default): no recording
    reopens a remembered session.
    """
    expected = "data_art_impulse_lens" if home_preferences else CAPTURE_BACKDROP
    if (theme, backdrop) != (CAPTURE_THEME, expected):
        raise ValueError("Tutorial captures require dark mode and the Blobs backdrop")
    from spacr.qt import preferences as prefs

    prefs.set_theme(theme)
    prefs.set_ambient_animation(backdrop)
    force_alpha_features_off()
    force_fresh_start()


def verify_appearance(window, *, allow_alpha_toggle_scene=False, alpha_lesson=None,
                      home_preferences=False):
    """Check the effective palette and the backdrop widgets being painted.

    Also refuses the frame while Show alpha features is on, unless
    ``allow_alpha_toggle_scene`` is the Preferences toggle scene's opt-in.
    """
    from PySide6.QtGui import QPalette

    from spacr.qt import preferences as prefs
    from spacr.qt.widgets.ambient import AmbientWidget
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    alpha_shown = verify_alpha_features_off(
        allow_alpha_toggle_scene=allow_alpha_toggle_scene, alpha_lesson=alpha_lesson)
    verify_fresh_start(window)
    if prefs.resolve_effective_theme() != CAPTURE_THEME:
        raise RuntimeError("Capture refused: the effective theme is not dark")
    expected = "data_art_impulse_lens" if home_preferences else CAPTURE_BACKDROP
    if not prefs.get_ambient_enabled() or prefs.get_ambient_animation() != expected:
        raise RuntimeError("Capture refused: Blobs is disabled or another backdrop is selected")
    if window.palette().color(QPalette.Window).lightness() >= 128:
        raise RuntimeError("Capture refused: the window is painting a light palette")
    visible = [widget for widget in window.findChildren(AmbientWidget) if widget.isVisible()]
    if not visible or any(widget.theme() != expected for widget in visible):
        raise RuntimeError("Capture refused: visible backdrop widgets are not Blobs")
    if not any(widget.frames_painted > 0 for widget in visible):
        raise RuntimeError("Capture refused: the Blobs backdrop has not painted a frame")
    if any(widget.isVisible() for widget in window.findChildren(DnaRainWidget)):
        raise RuntimeError("Capture refused: DNA rain covers the requested Blobs backdrop")
    receipt = {"theme": CAPTURE_THEME, "backdrop": expected,
               "painted_frames": sum(widget.frames_painted for widget in visible)}
    if alpha_shown:
        receipt["alpha_features_shown_for_toggle_scene"] = True
    return receipt


def exclude_special_backdrops(window):
    """Expose the shared Blobs backdrop in the recording namespace.

    Map Barcodes normally draws its own DNA rain over the window backdrop.
    Hiding that decorative layer is a capture presentation choice, like
    excluding News. No application code, controls or analysis results change.
    """
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    hidden = []
    for widget in window.findChildren(DnaRainWidget):
        if widget.isVisible():
            widget.hide()
            hidden.append(type(widget).__name__)
    return hidden


def exclude_release_history(window):
    """Omit the historical News aside from current-version recordings.

    This is a capture presentation choice, like omitting recent-run history.
    Application sources, release notes and workflow results are unchanged.
    """
    from spacr.qt.widgets.home import NewsPanel

    panels = window.findChildren(NewsPanel)
    for panel in panels:
        panel.hide()
    return len(panels)


def verify_visible_paths(windows, prepared_root):
    """Refuse visible personal/mounted paths before saving a tutorial frame.

    This inspects Qt text surfaces, including visible table cells. Images and
    external desktop windows still require the sampled-frame visual review.
    """
    from PySide6.QtCore import Qt
    from spacr import __version__
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QLabel,
        QLineEdit,
        QListView,
        QPlainTextEdit,
        QTextEdit,
        QWidget,
    )

    allowed = str(Path(prepared_root).absolute())
    if _PRIVATE_PATH.search(allowed):
        raise ValueError("Use a neutral prepared capture directory, for example /tmp/spacr-tutorials")
    for window in windows:
        for widget in [window, *window.findChildren(QWidget)]:
            if not widget.isVisible():
                continue
            texts = [widget.windowTitle()]
            if isinstance(widget, (QLabel, QLineEdit)):
                texts.append(widget.text())
            elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
                texts.append(widget.toPlainText())
            elif isinstance(widget, QComboBox):
                texts.append(widget.currentText())
            if isinstance(widget, QAbstractItemView) and widget.model() is not None:
                model = widget.model()
                index = widget.indexAt(widget.viewport().rect().topLeft())
                first_row = max(0, index.row())
                for row in range(first_row, model.rowCount(widget.rootIndex())):
                    row_visible = False
                    # QListWidget's internal model exposes columnCount as a
                    # private C++ method in PySide; a list always has one.
                    columns = 1 if isinstance(widget, QListView) else model.columnCount(widget.rootIndex())
                    for column in range(columns):
                        cell = model.index(row, column, widget.rootIndex())
                        if widget.visualRect(cell).intersects(widget.viewport().rect()):
                            row_visible = True
                            texts.append(str(model.data(cell, Qt.DisplayRole) or ""))
                    if not row_visible and row > first_row:
                        break
            for text in texts:
                plain_text = re.sub(r"<[^>]+>", "", text)
                releases = re.findall(
                    r"\bspaCR[\s-]+(\d+\.\d+\.\d+(?:\.\d+)?)", plain_text, re.I)
                if any(version != __version__ for version in releases):
                    raise RuntimeError(
                        "Capture refused: visible text references another spaCR release")
                for match in _PRIVATE_PATH.finditer(text):
                    path = match.group(0)
                    if path != allowed and not path.startswith(allowed + "/"):
                        raise RuntimeError(
                            "Capture refused: a visible text surface contains a personal "
                            "or mounted path outside the prepared capture directory: "
                            f"{type(widget).__name__} {widget.objectName()!r}: {path}")


def main(argv=None):
    """``capture_policy.py --force-alpha-off CONFIG_HOME...`` for shell launchers (alpha off, no remembered session)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--force-alpha-off", nargs="+", type=Path, required=True,
                        metavar="CONFIG_HOME")
    args = parser.parse_args(argv)
    for path in force_alpha_features_off_in_profiles(args.force_alpha_off):
        print(f"alpha features off: {path}")
    # The same profiles also start fresh: no remembered session, no drafts.
    for path in force_fresh_start_in_profiles(args.force_alpha_off):
        print(f"no remembered session or drafts: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
