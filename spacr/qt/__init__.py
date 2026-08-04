"""
Modern PySide6 (Qt 6) GUI for spacr.

Runs alongside the classic Tk GUI (spacr.gui) — nothing here touches the
Tk stack. To launch:

    spacr-qt            # CLI shortcut (see setup.py entry_points)
    python -m spacr.qt  # equivalent

The old GUI keeps working via:
    spacr               # Tk (classic)
    python -m spacr

The Qt code lives in three layers:
    theme.py              — palette + QSS stylesheet
    widgets/              — reusable custom widgets (tiles, sections, ...)
    screens/              — one Qt widget per app screen
                            (startup, mask, measure, ...)
    app.py                — main window + QApplication bootstrap
"""
from __future__ import annotations

import os
import re
import sys

__all__ = ["run"]

_VERSION_FLAGS = frozenset({"-v", "-version", "--version"})

#: Launch noise the user cannot act on, matched against Qt's own log lines.
#:
#: These do NOT come through Python's warning system, so
#: `warnings.filterwarnings` never sees them — they are written by Qt's
#: categorised logging, which is why they survived every filter in
#: `spacr/__init__.py`.
#:
#: * the OpenType line fires once per screen that lays out text in a script
#:   "Open Sans" has no table for. Qt falls back to a font that does and the
#:   text renders correctly; the message is a note, not a failure.
_QT_NOISE = re.compile(
    r"OpenType support missing for|"
    r"This plugin does not support (propagateSizeHints|raise)"
)


def _install_quiet_qt_logging() -> None:
    """Drop known-harmless Qt log lines, pass everything else through.

    Deliberately a filter rather than a blanket mute: a Qt warning about a
    real problem — a missing plugin, a failed shader, an invalid pixmap — is
    often the only clue there is, and swallowing the category wholesale is how
    that clue gets lost.
    """
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    def handler(mode, context, message):
        if _QT_NOISE.search(message or ""):
            return
        stream = sys.stderr
        label = {
            QtMsgType.QtDebugMsg: "Qt debug",
            QtMsgType.QtInfoMsg: "Qt info",
            QtMsgType.QtWarningMsg: "Qt warning",
            QtMsgType.QtCriticalMsg: "Qt critical",
            QtMsgType.QtFatalMsg: "Qt fatal",
        }.get(mode, "Qt")
        print(f"{label}: {message}", file=stream)

    qInstallMessageHandler(handler)


#: Third-party warnings a spaCR user cannot act on, as
#: ``(message regex, module regex)``. Unlike :data:`_QT_NOISE` these DO come
#: through Python's warning system, so a filter is the right tool — but a
#: filter written carelessly is how a real warning gets lost, so both halves
#: of each entry are deliberate:
#:
#: * the message pattern is NOT anchored. ``warnings.filterwarnings``
#:   matches ``message`` with ``re.match``, so a filter written against the
#:   sentence a user quoted out of a traceback misses the same notice from a
#:   build that prefixes it. Every pattern here begins with ``.*`` on
#:   purpose.
#: * the module pattern IS present, and it is the raising frame's DOTTED
#:   ``__name__`` — ``warnings.warn`` reads ``globals()["__name__"]``, so a
#:   pattern written against ``cellpose/dynamics.py`` (the path a traceback
#:   shows) matches nothing at all. It is also matched with ``re.match``,
#:   hence the explicit ``(\.|$)`` rather than a bare prefix that would also
#:   catch a ``cellpose_something`` package. Scoping to the library keeps
#:   "ignore this sentence" from also swallowing the same sentence raised by
#:   spaCR's own code.
#:
#: The single entry: Cellpose 4 builds a sparse COO tensor in ``dynamics.py``
#: for every mask it makes, and torch notes that invariant checking is off.
#: It names a torch internal, it fires on the first mask of every run, and
#: there is nothing a spaCR user can do about it.
_LIBRARY_NOISE: tuple[tuple[str, str], ...] = (
    (r".*[Ss]parse invariant checks are implicitly disabled",
     r"cellpose(\.|$)"),
)


def _quiet_library_warnings() -> None:
    """Ignore :data:`_LIBRARY_NOISE`, and nothing else, for this process.

    ``spacr/__init__.py`` installs the same rule at import, which is what a
    headless ``spacr-run`` or a spawned worker gets, so on a clean launch
    this finds the filter already in place and does nothing. That is the
    intended steady state. What it is for is the case where it is *not*
    already in place: ``warnings.filters`` is process-global mutable state
    and anything the launcher imports may reset it, and a screen that calls
    ``warnings.resetwarnings()`` instead of ``catch_warnings()`` would
    otherwise leave the rest of the session noisy. Re-asserting at launch
    costs nothing and is the same shape as the two quieters beside it.

    Idempotent, and it has to be: ``warnings.filterwarnings`` prepends
    unconditionally, so a function called on every ``run()`` — which the
    test suite drives repeatedly in one process — would otherwise grow
    ``warnings.filters`` without bound.
    """
    import warnings

    for message, module in _LIBRARY_NOISE:
        # A filters entry is (action, message_re, category, module_re, lineno)
        # with the two patterns compiled, so `.pattern` recovers what was asked
        # for and the comparison is against the request rather than the object.
        if any(action == "ignore" and category is UserWarning
               and getattr(msg_re, "pattern", None) == message
               and getattr(mod_re, "pattern", None) == module
               for action, msg_re, category, mod_re, _lineno
               in warnings.filters):
            continue
        warnings.filterwarnings(
            "ignore", message=message, category=UserWarning, module=module)


def _quiet_gtk_accessibility() -> None:
    """Stop GTK printing "Not loading module atk-bridge" on every window.

    Qt's GTK platform theme pulls in GTK, which then reports that the AT-SPI
    bridge is built in and need not be loaded as a module. It is written to
    stderr by GTK itself in C, so neither Python's warning filters nor a Qt
    message handler can reach it — the only lever is the environment variable
    GTK reads before it decides, and it has to be set before GTK loads.

    Only set when absent, so a user who deliberately wants the bridge (a
    screen-reader setup) is not overridden.
    """
    os.environ.setdefault("NO_AT_BRIDGE", "1")

#: Distributions that only `pip install "spacr[qt]"` brings in. PySide6 is
#: declared in the `qt` extra (setup.py), *not* in core, so a plain
#: `pip install spacr` followed by `spacr` used to die on an unhandled
#: `ImportError: No module named 'PySide6'` raised six frames deep inside
#: `spacr/qt/app.py`. shiboken6 is PySide6's binding runtime and fails the
#: same way when a wheel is half-installed.
_QT_EXTRA_MODULES = frozenset({"PySide6", "shiboken6", "qtawesome"})

_QT_MISSING_MESSAGE = """\
spaCR's graphical interface needs the optional Qt extra, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[qt]"

Then run `spacr` again.

No display available? The pipelines run headless without Qt:

    spacr-run --list\
"""


def _missing_qt_extra(exc: ImportError) -> str | None:
    """Identify the Qt-extra distribution whose absence raised ``exc``.

    Only failures that name a module from :data:`_QT_EXTRA_MODULES` count.
    Anything else is a genuine bug inside the GUI package and must keep its
    traceback rather than be reported as a missing install.

    Args:
        exc: The ``ImportError`` raised while importing :mod:`spacr.qt.app`.

    Returns:
        The top-level module name to name in the install hint, or ``None``
        when ``exc`` is unrelated to the Qt extra.
    """
    # ModuleNotFoundError sets `.name` to the module that could not be found;
    # a failed `from PySide6.QtCore import ...` sets it to `PySide6.QtCore`.
    root = (getattr(exc, "name", None) or "").split(".", 1)[0]
    if root in _QT_EXTRA_MODULES:
        return root
    # Import hooks and hand-raised ImportErrors may leave `.name` unset, so
    # fall back to the message text before giving up on the friendly path.
    text = str(exc)
    for module in sorted(_QT_EXTRA_MODULES):
        if module in text:
            return module
    return None


def run(argv: list[str] | None = None) -> int:
    """Launch the Qt GUI. Public entry point used by both `spacr-qt` and
    `python -m spacr.qt`.

    Args:
        argv: Optional CLI arguments. The first positional element,
              if present, opens directly into that app screen key
              (e.g. `spacr-qt mask`).

    Returns:
        The exit code returned by `QApplication.exec()`, or ``1`` when the
        optional Qt extra is not installed.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Before anything imports Qt, GTK or torch: the AT-SPI variable is only
    # read while GTK loads, the Qt handler has to be in place before the
    # first widget lays out text, and the warning filter has to be in place
    # before the pipeline preloader reaches cellpose.
    _quiet_gtk_accessibility()
    _install_quiet_qt_logging()
    _quiet_library_warnings()

    if len(argv) == 1 and argv[0] in _VERSION_FLAGS:
        from spacr.version import get_version

        print(get_version())
        return 0

    try:
        from .app import launch
    except ImportError as exc:
        module = _missing_qt_extra(exc)
        if module is None:
            raise
        print(_QT_MISSING_MESSAGE.format(module=module), file=sys.stderr)
        return 1

    register_self_registering_modules()

    # Deliberately outside the `try`: an ImportError raised *during* a run —
    # a screen lazily importing an optional reader, say — is a real failure
    # and must not be reported as "Qt is not installed".
    return launch(argv)


#: Modules that own an app and register it through
#: :func:`spacr.qt.app.register_app` from their own file, rather than through
#: a row written into ``app.py``. Each exposes a zero-argument, idempotent
#: ``register()``.
#:
#: They are imported by :func:`register_self_registering_modules`, which
#: :func:`run` calls between ``from .app import launch`` and the call to it —
#: and that position is the whole point. ``app.py`` is fully executed by then,
#: so ``register_app`` exists to be imported; and ``MainWindow.__init__`` has
#: not run yet, so the menu bar, the sidebar and Home have not yet read the
#: registry. A module imported any earlier — from ``widgets/__init__.py``,
#: say, which ``app.py`` itself imports on its 39th line — finds
#: ``spacr.qt.app`` half-initialised and can register nothing.
SELF_REGISTERING_MODULES = (
    "spacr.qt.widgets.feature_dictionary",
    # Not an app of its own: it registers a screen FACTORY for every module
    # that declares ports, so the generic AppScreen gains the auto-chaining /
    # staleness / next-step strip without a line inside the shared screen.
    "spacr.qt.chaining",
    # Also not an app: it decorates the Measure screen with the segmentation
    # verdict seg_qc already computed and the Mask screen with the diameter
    # estimator, by wrapping whatever factory is registered for those two
    # keys. Listed AFTER chaining so the normal launch order composes onto
    # chaining's screen rather than the other way round; both orders work.
    "spacr.qt.prerun",
    "spacr.qt.screens.run_compare",
    # Three Explore screens built on the Graph Builder's spec engine. Each
    # owns a tested, idempotent register() that fans its name, intro, CLI
    # note, api_module and nine translations out of one register_app call.
    # All that was ever missing was the row that runs it: the agent that
    # wrote them could not add it while this file was being edited.
    "spacr.qt.screens.trellis",
    "spacr.qt.screens.gate_editor",
    "spacr.qt.screens.feature_explorer",
    # Flags an object — or a whole well, which is the more common failure —
    # as extreme by a robust rule, and writes a COLUMN rather than dropping a
    # row. Safe to have on by default for exactly that reason: nothing it
    # decides is destructive until the user acts on it.
    "spacr.qt.screens.outliers",
    # Four-parameter logistic with a confidence interval on EC50 -- and a
    # refusal wherever an EC50 would be a guess: an incomplete curve reports a
    # one-sided bound instead of a number, and non-monotone data is not fitted
    # at all. Filed under Design, with Power: an EC50 is fitted to choose the
    # concentration the next experiment will use.
    "spacr.qt.screens.dose_response",
    # A control's measured value plate by plate across a campaign, with limits
    # estimated from a STATED baseline and applied forward, so a drift is
    # visible before it has ruined the screen rather than after.
    "spacr.qt.screens.control_chart",
    # Every project on disk in one table -- stage, size, last run, what is
    # stale -- built entirely on `spacr.projects`, which is built on ports,
    # artifacts, data_manager and chaining. A project the registry has never
    # seen is listed too; that is the case it exists for.
    "spacr.qt.screens.project_browser",
    # Not an app either: it corrects the maturity label on the modules whose
    # evidence no longer matches "alpha". Listed LAST, after every module
    # that registers an app of its own, because it can only reassess apps
    # that are in the registry by the time it runs — a module registered
    # after it would keep whatever stage it declared.
    "spacr.qt.maturity",
)


def register_self_registering_modules() -> tuple[str, ...]:
    """Import each of :data:`SELF_REGISTERING_MODULES` and let it register.

    Idempotent — every ``register()`` is written to be safe to call twice, so
    a second launch in one process (the test suite does this) does not raise
    on a duplicate app key.

    One module's failure costs that module's app and nothing else: an
    optional panel must never stop the GUI from starting.

    :returns: the module names that registered without raising.
    """
    import importlib
    import logging

    registered: list[str] = []
    for name in SELF_REGISTERING_MODULES:
        try:
            importlib.import_module(name).register()
        except Exception:
            logging.getLogger("spacr.qt").exception(
                "Could not register the app owned by %s", name)
        else:
            registered.append(name)
    return tuple(registered)
