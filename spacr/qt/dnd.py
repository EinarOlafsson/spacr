"""
Drag-and-drop system for AppScreens.

Design:

* :class:`DropHandler` — per-module policy: what folders/files this
  screen accepts, how to fix a "close-but-not-quite" drop, and what
  to do once a drop is accepted.
* :func:`install_dropzone` — attaches Qt drop event handlers to any
  widget (usually the AppScreen itself) and wires them to a
  :class:`DropHandler`.
* :func:`suggest_alternatives_dialog` — the "did you mean X?"
  chooser shown when the dropped folder can't be used as-is but a
  sibling / child folder can.

Behaviour common to every module:

* Dropping a ``*.csv`` file → treat as a settings CSV and call the
  screen's ``apply_settings_dict`` (imports settings, doesn't
  overwrite the source folder).
* Dropping a folder → hand off to the module's ``DropHandler``.
  If it's a good fit, the handler calls ``screen._set_src`` (or
  equivalent). If it's a near-miss the user gets the "did you mean"
  dialog.
* Dropping multiple folders → the handler is called once per folder
  in the order the OS delivers them. Modules that don't handle
  multi-drop degrade to first-only.

Where the work happens
----------------------

A drop is delivered by Qt on the GUI thread and every path in it is a path
the USER chose -- which may sit on a sleeping ``autofs`` share that takes
more than twenty seconds to answer one stat (see :mod:`spacr.qt.path_probe`).
So the drop is split in two, and the seam is the rule for anything added
here:

* :func:`_classify_drop` asks the disk everything the drop needs to know,
  on the screen's drop scanner. No Qt, no widgets, plain data out.
* :func:`_deliver_drop` acts on that data on the GUI thread: the settings
  import, ``handler.apply``, the rejection report, the "did you mean" dialog.

:func:`_route_drop` joins them, and keeps concurrent drops on one screen in
the order the user made them.

Per-module policies live in :mod:`spacr.qt.dnd_handlers`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence
from weakref import WeakKeyDictionary

from PySide6.QtCore import QEvent, QMimeData, QObject, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.dnd")

# File extensions that count as images for "does this folder have
# images?" checks. Keep in sync with spacr.io's readers.
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi",
              ".nd2", ".lif")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DropHandler(ABC):
    """Per-module drop policy.

    Subclasses implement:
        can_accept(path)          — is this path good to go?
        apply(path, screen)       — wire it into the screen.
    And optionally override:
        suggest_alternatives(p)   — return nearby folders that DO fit.
        error_message(p)          — return the "why not?" string.
        accepts_multiple()        — True if multi-folder drops make sense.
    """

    # -- public API subclasses implement -----------------------------------
    @abstractmethod
    def can_accept(self, path: Path) -> bool:
        """Return True if ``path`` (folder OR file) is usable as-is."""

    @abstractmethod
    def apply(self, path: Path, screen) -> None:
        """Wire ``path`` into ``screen`` (set src, populate settings, etc.)."""

    def suggest_alternatives(self, path: Path) -> List[Path]:
        """When ``can_accept`` returns False, return sibling/child folders
        that WOULD be accepted so the UI can prompt "did you mean…".

        Default: no suggestions.
        """
        return []

    def error_message(self, path: Path) -> str:
        """Human-friendly explanation for why ``path`` can't be used."""
        return f"This module can't use {path.name!r}."

    def accepts_multiple(self) -> bool:
        """Return True to be called per-folder on multi-item drops."""
        return False


def install_dropzone(target: QWidget, handler: DropHandler,
                       screen: QWidget) -> None:
    """Wire ``target`` to accept drops routed through ``handler``.

    Typically called from ``AppScreen.__init__``: ``target`` is
    ``self`` and ``screen`` is also ``self``. Splitting them lets
    non-AppScreen widgets install a dropzone that acts on a
    different owner (e.g. a specific input row).

    :param target: the QWidget that receives drag/drop events.
    :param handler: the module's DropHandler policy.
    :param screen: the widget passed to ``handler.apply`` — usually
        the AppScreen.
    """
    target.setAcceptDrops(True)

    # Store the handler + owning-screen on the widget itself so the
    # event filter can look them up without capturing them in a
    # closure that would keep the target alive after destruction.
    target._dnd_handler = handler
    target._dnd_screen = screen
    # Filter is parented to target — Qt cleans it up when target dies.
    f = _DropzoneFilter(target)
    target.installEventFilter(f)


def install_for(target: QWidget, app_key: str, screen: QWidget = None) -> bool:
    """Attach ``app_key``'s drop policy to ``target``. Never raises.

    The one line a screen adds to accept drops. Which policy that is comes
    from :func:`spacr.qt.dnd_handlers.get_handler`, so a screen never names a
    handler class and a screen with no declared policy still gets the
    source-folder fallback.

    Failure is a missing convenience, not a broken screen — a Qt build with no
    drag-and-drop, or a handler whose import fails, must not stop the screen
    being constructed. It is logged and the screen goes up without a dropzone.

    :param target: the widget that receives the drag/drop events.
    :param app_key: the registered app key, e.g. ``"graph_builder"``.
    :param screen: the object handed to ``handler.apply``; ``target`` when
        omitted.
    :returns: whether the dropzone was installed.
    """
    try:
        from .dnd_handlers import get_handler
        install_dropzone(target, get_handler(app_key), screen or target)
        return True
    except Exception:
        LOG.debug("no dropzone installed for %s", app_key, exc_info=True)
        return False


class _DropzoneFilter(QObject):
    """Event filter that routes drag/drop events on ``target`` into
    the :class:`DropHandler` attached to it.

    :param target: the widget whose drag and drop events are routed. Also
        the QObject parent -- and assigned BEFORE ``super().__init__`` for
        the reason in the constructor: parenting can deliver a ChildAdded
        synchronously, and this object is an event filter, so it has to be
        able to answer for itself already.
    """

    def __init__(self, target: QWidget):
        # QObject parenting can synchronously deliver a ChildAdded event to
        # the target.  Set this first so eventFilter is fully initialized even
        # during super().__init__ (standalone tool screens exposed this race).
        """Install on ``target`` and route its drops to that widget's handler."""
        self._target = target
        super().__init__(target)   # parent → auto-cleanup

    def eventFilter(self, obj, event):    # noqa: N802  (Qt naming)
        # `getattr`, not `self._target`, and the reason is not defensiveness
        # for its own sake. Qt goes on delivering events to a filter after the
        # target's C++ half is gone, and PySide6 clears the Python wrapper's
        # __dict__ when that happens -- so `self._target` raises AttributeError
        # from INSIDE the Qt event loop, which prints
        #
        #     Error calling Python override of QObject::eventFilter()
        #     AttributeError: '_DropzoneFilter' object has no attribute '_target'
        #
        # once per delivered event, and cannot be caught by any caller because
        # there is no Python caller. A filter whose target is gone has nothing
        # to filter, so declining the event is both correct and quiet.
        #
        # The same shape as `RunHandle.is_running` swallowing "Internal C++
        # object already deleted": the destroyed wrapper IS the answer, not an
        # error condition.
        target = getattr(self, "_target", None)
        if target is None or obj is not target:
            return False
        et = event.type()
        if et == QEvent.DragEnter:
            self._on_drag_enter(event)
            return True
        if et == QEvent.DragMove:
            event.acceptProposedAction()
            return True
        if et == QEvent.Drop:
            self._on_drop(event)
            return True
        return False

    # -- handlers ----------------------------------------------------------
    def _on_drag_enter(self, event: QDragEnterEvent) -> None:
        """Accept a drag that carries local paths, and no other."""
        mime = event.mimeData()
        if _mime_has_local_paths(mime):
            event.acceptProposedAction()

    def _on_drop(self, event: QDropEvent) -> None:
        """Route a drop: settings CSVs to the importer, the rest to the screen.

        THE DROP IS ACCEPTED AS SOON AS THERE IS SOMETHING TO DO WITH IT, not at
        the end. Accepting only after the routing meant a settings-CSV-only drop
        -- which IS handled -- was reported back to the operating system as
        rejected, so the drag animation snapped back while the import ran.

        AND NOTHING HERE TOUCHES THE FILESYSTEM. Every path in a drop is a path
        the USER chose, which on one such workstation includes ``/nas_mnt``
        shares behind an ``autofs`` mount, measured: a single stat on
        a sleeping one had not returned after TWENTY SECONDS. This method runs
        inside Qt's delivery of ``QEvent.Drop``, i.e. on the GUI thread, so the
        old body -- ``p.is_file()`` to split the CSVs out, then ``can_accept``
        and ``suggest_alternatives`` walking the folder -- froze the whole
        application for as long as the mount took to wake. The freeze had no
        traceback and was reported as "spacr crashes"; see
        :mod:`spacr.qt.path_probe` for the rest of that story.

        So the split, the classification and the CSV read all happen on a
        worker (:func:`_route_drop`) and only the widget work comes back here.
        The acceptance above is unaffected: it happens before the scan starts,
        so the OS drag animation still lands the moment the user lets go.
        """
        paths = _mime_local_paths(event.mimeData())
        if not paths:
            return
        # Tell the drag source the drop landed as soon as we know we have
        # something to do with it. Doing this only at the very end meant a
        # settings-CSV-only drop (which IS handled below) was reported back
        # to the OS as rejected.
        event.acceptProposedAction()
        handler: DropHandler = self._target._dnd_handler
        screen = self._target._dnd_screen
        _route_drop(paths, handler, screen)


# ---------------------------------------------------------------------------
# Drop routing: the scan half runs on a worker, the widget half on the GUI
# ---------------------------------------------------------------------------

def _classify_drop(paths: Sequence[Path], handler: DropHandler,
                   takes_csv: bool, multiple: bool) -> List[dict]:
    """Ask the disk everything the drop needs to know. WORKER THREAD ONLY.

    No Qt, no widgets, plain data out -- the same contract as the scans in
    :mod:`spacr.qt.dnd_handlers`. ``handler.can_accept``,
    ``suggest_alternatives`` and ``error_message`` are pure policy over the
    filesystem and are safe here; ``handler.apply`` is NOT, and stays on the
    GUI thread in :func:`_deliver_drop`.

    :param paths: what the user dropped, in the order the OS delivered it.
    :param takes_csv: whether the screen exposes ``apply_settings_dict``. A
        CSV is a universal settings import only on screens that have the
        importer; special-purpose screens (Plate Queue, Batch Runner, Import
        Project) give CSVs their own meaning and must receive them through
        their handler instead of losing them to a no-op.
    :param multiple: ``handler.accepts_multiple()``, read on the GUI thread
        because it is pure Python and reading it here would be one more thing
        the worker has to be trusted with.
    :returns: one entry per path the drop will act on, in drop order.

    NEVER RAISES, and that is load-bearing rather than tidy. A ``JobRunner``
    calls ``on_done`` only for a job that SUCCEEDED, so an exception escaping
    here would take the whole delivery with it: no import, no rejection
    report, no dialog, no status line -- a drop that silently did nothing,
    where the same exception on the GUI thread at least printed a traceback.
    Every path is therefore classified inside its own guard and a failure is
    carried back as a rejection for :func:`_deliver_drop` to report.
    """
    report: List[dict] = []
    others = 0
    for path in paths:
        entry: dict = {"path": path, "csv": None}
        try:
            if takes_csv and path.suffix.lower() == ".csv" and path.is_file():
                entry["csv"] = _read_settings_csv(path)
                report.append(entry)
                continue
            others += 1
            # Modules that do not handle multi-drop degrade to first-only,
            # and the ones beyond the first are dropped HERE rather than
            # scanned and then discarded -- a second sleeping mount is a
            # second freeze.
            if not multiple and others > 1:
                continue
            entry["accepted"] = bool(handler.can_accept(path))
            if not entry["accepted"]:
                entry["message"] = handler.error_message(path)
                entry["alternatives"] = list(
                    handler.suggest_alternatives(path))
        except Exception as exc:                                 # noqa: BLE001
            # A policy that raises used to raise inside Qt's event delivery,
            # where there is no Python caller to catch it. Off the GUI thread
            # it would be swallowed by the runner instead, and the user would
            # be told nothing at all -- so it is carried back as a rejection.
            LOG.debug("drop classification failed for %s", path,
                      exc_info=True)
            entry["csv"] = None
            entry["accepted"] = False
            entry["error"] = str(exc)
        report.append(entry)
    return report


def _deliver_drop(report: Sequence[dict], handler: DropHandler,
                  screen) -> None:
    """Act on a finished :func:`_classify_drop`. GUI THREAD ONLY.

    Everything that needs a widget lives here: the settings import, the
    handler's ``apply``, the rejection report and the "did you mean" dialog.
    This function asks the disk nothing itself -- every answer it acts on
    arrived in ``report``.

    ``handler.apply`` IS still called from here, and apply is allowed to
    touch the filesystem: it wires the path into widgets, so it cannot move
    to a worker wholesale. Most of them do touch it -- ``is_file`` to decide
    whether to take the parent, ``resolve``, a probe for a sibling database
    -- and of the nineteen that do, only ``ModelZooDropHandler`` submits a
    scan of its own. WHAT MAKES THAT SAFE IS NOT THAT THEY DEFER, IT IS THAT
    THEY ARE SECOND: the path apply stats is the path :func:`_classify_drop`
    just walked on the worker, so by the time apply runs the ``autofs`` mount
    is awake and the kernel's dentry cache is warm, and the stat that took
    twenty seconds cold takes microseconds. The expensive half is the WAKE,
    and the accept decision that used to trigger it on the GUI thread is what
    moved off.

    So the rule for a new handler is a narrow one, not a blanket permission:
    apply may stat what the classification already touched. A handler that
    reaches somewhere else -- a different share, a tree the scan never
    visited, a large file read whole -- is back on a cold mount on the GUI
    thread and belongs on ``_scan_then``, the way the model zoo's recursive
    walk does.

    :param report: what the scan found, in drop order.
    :param handler: the module's policy, for ``apply`` only.
    :param screen: the widget the handler wires the drop into.
    """
    # CSVs first, as they always were: a settings CSV and a folder in one
    # drop means the folder wins, because it is applied last.
    for entry in report:
        if entry.get("csv") is not None:
            _apply_settings_csv(entry["path"], screen, scan=entry["csv"])

    for entry in report:
        if entry.get("csv") is not None:
            continue
        path = entry["path"]
        if entry.get("accepted"):
            try:
                handler.apply(path, screen)
            except Exception as e:
                _report_drop_problem(
                    screen, path, f"The drop handler failed: {e}",
                    "Check that the path is readable and that its contents "
                    "match this module, then try again.",
                )
            continue
        if entry.get("error"):
            _report_drop_problem(
                screen, path,
                f"The drop handler failed: {entry['error']}",
                "Check that the path is readable and that its contents "
                "match this module, then try again.",
            )
            continue
        alternatives = entry.get("alternatives") or []
        why = entry.get("message", "")
        suggestion = (
            "Choose one of the compatible nearby paths shown in the "
            "dialog."
            if alternatives else
            "Open this module's source setting and choose a file or "
            "folder matching the required layout."
        )
        _report_drop_problem(
            screen, path, why, suggestion, alternatives=alternatives,
        )
        if alternatives:
            pick = suggest_alternatives_dialog(
                screen, path, alternatives, why=why,
            )
            if pick is not None:
                try:
                    handler.apply(pick, screen)
                except Exception as e:
                    _report_drop_problem(
                        screen, pick, f"The drop handler failed: {e}",
                        "Check that the path is readable and try again.",
                    )
        else:
            QMessageBox.information(
                screen, "Nothing to drop into",
                f"{why}\n\nSuggestion: {suggestion}",
            )


class _PendingDrop:
    """One drop whose classification is still out. GUI THREAD ONLY.

    :param deliver: what to run once this drop's turn comes, given its
        report.
    """

    __slots__ = ("deliver", "report", "answered")

    def __init__(self, deliver: Callable[[object], None]) -> None:
        """Hold a drop that has been made but not yet classified.

        `answered` and `report` start empty because the classification runs
        on a worker: the queue needs an entry the moment the drop is made, so
        that a LATER drop cannot be delivered ahead of this one, and the
        entry has nothing in it until the scanner comes back.

        :param deliver: what to run once this drop's turn comes, given its
            report.
        """
        self.deliver = deliver
        self.report = None
        self.answered = False


#: Drops still being classified, per screen, oldest first.
#:
#: WHY THERE IS A QUEUE AT ALL. Classifications run on the screen's drop
#: scanner, whose ``JobRunner`` starts a thread per job and does not serialise
#: them -- so two drops finish in whatever order the FILESYSTEM answers, not
#: the order the user made them. Drop a folder from a sleeping share, watch
#: nothing happen, drop a local folder instead, and twenty seconds later the
#: first one lands last and overwrites the source the user actually chose.
#: Inline, that reordering was impossible: each drop finished before the next
#: could be delivered.
#:
#: The answer is not to discard the late one -- the user is owed its console
#: report either way -- but to hold each answer until every drop made BEFORE
#: it has been delivered. Same paths applied, same warnings printed, same
#: winner, just later.
#:
#: A queue that HOLDS deliveries has to be able to let one go, and that is
#: the other half: a classification cancelled with its screen (closing one
#: shuts its scanner down) never calls back, so its slot would sit unanswered
#: at the head of the queue and silently swallow every later drop on a screen
#: spaCR keeps and shows again. :func:`_forget_abandoned` writes those off
#: when the next drop arrives.
#:
#: Weak keys because a screen that is closed mid-scan must not be held alive
#: by a queue nobody will ever drain; every access is on the GUI thread, so
#: no lock.
_pending_drops: "WeakKeyDictionary" = WeakKeyDictionary()


def _scan_in_flight(screen) -> bool:
    """Whether ``screen``'s drop scanner still owes an answer.

    The import is deferred and guarded for the reason :func:`_route_drop`
    defers its own: :mod:`spacr.qt.dnd_handlers` imports this module at the
    top. A screen with no scanner has nothing in flight, which is also the
    truthful answer when the module cannot be imported at all.
    """
    try:
        from .dnd_handlers import scan_is_busy
        return bool(scan_is_busy(screen))
    except Exception:                                            # noqa: BLE001
        return False


def _forget_abandoned(screen, queue: list) -> None:
    """Let go of the slots on ``screen`` that no scan will ever answer.

    A slot is answered by its classification's completion handler, and a
    ``JobRunner`` runs that handler ONLY for a job that succeeded and was not
    cancelled. LEAVING A SCREEN MID-SCAN CANCELS IT -- ``_DropScanner``
    shuts its runner down on the screen's Close event -- and a spaCR screen
    is CACHED, not destroyed: the user comes back to it with an unanswered
    slot at the head of its queue, and every later drop on that screen is
    held behind a slot that is never filled in. Drag a folder on, nothing
    happens; drag another, nothing happens; for the life of the window, with
    no message anywhere. The queue exists to delay deliveries, not to lose
    them.

    The repair is taken when a NEW drop arrives, because that is the only
    moment a wedged queue costs anything -- and it is safe there: with
    nothing in flight for this screen no answer is coming, so a slot
    unanswered now is unanswered for good. The scanner is shared with the
    folder reads a handler submits from ``apply``, so the test can only ever
    be conservative: one of those in flight DELAYS writing a dead slot off,
    it never writes off a slot whose answer is still on its way.

    Answered slots are kept, and kept in order: they are owed a delivery, and
    :func:`_drain` gives it to them as soon as the head of the queue is ready.
    """
    if not queue or _scan_in_flight(screen):
        return
    abandoned = sum(1 for slot in queue if not slot.answered)
    if not abandoned:
        return
    LOG.debug("letting go of %d drop(s) whose classification never came back",
              abandoned)
    queue[:] = [slot for slot in queue if slot.answered]


def _run_delivery(slot) -> None:
    """Hand one filled slot to its delivery, whatever that delivery does.

    Every delivery goes through here so that one that raises costs only
    itself. The drops still queued behind it are owed their turn -- and a
    delivery reached through :func:`_scan_then`'s INLINE path would otherwise
    throw back into :func:`_route_drop`, whose fallback would answer by
    classifying the whole drop a second time.
    """
    try:
        slot.deliver(slot.report)
    except Exception:                                            # noqa: BLE001
        LOG.exception("delivering a drop failed")


#: Queues whose :func:`_drain` is running right now, held by identity.
#:
#: A DELIVERY OPENS MODAL DIALOGS, AND A MODAL DIALOG RUNS A NESTED QT EVENT
#: LOOP. :func:`_deliver_drop` reaches ``QMessageBox.information`` for a
#: rejected drop, ``suggest_alternatives_dialog`` for a near-miss and
#: ``QMessageBox.warning`` for a settings CSV that would not load; each of
#: them spins Qt's event loop inside the delivery, and Qt goes on dispatching
#: queued signals there -- including the ``_on_settled`` of a LATER drop's
#: scan. That re-enters :func:`_answer_drop` and, unguarded, delivered the
#: later drop in the MIDDLE of the earlier one: the newer folder was applied
#: first and the older one's ``handler.apply`` overwrote it on the way out.
#: The wrong source wins, which is precisely what :data:`_pending_drops`
#: exists to prevent -- reached through the one door the queue did not watch.
#:
#: Identity, not membership: a ``list`` is unhashable, so this cannot be a
#: set, and two screens' queues can compare equal while being different
#: queues. GUI thread only, so no lock.
_draining: List[list] = []


def _drain(queue: list) -> None:
    """Deliver every drop on ``queue`` that is ready, oldest first.

    Re-entrant-safe: a nested call (see :data:`_draining`) returns at once
    and leaves the work to the loop already running, which re-reads the
    queue after every delivery and so picks up anything answered during one.
    """
    if any(active is queue for active in _draining):
        return
    _draining.append(queue)
    try:
        # POP BEFORE DELIVERING, AND RE-READ AFTER. A slot left on the queue
        # while its own delivery is running would be delivered a second time
        # by a scan that lands from inside it, and one that raised would
        # block every later drop on this screen for the life of the window.
        # Taking the whole ready run off in one batch would fix that too,
        # but it would then miss the slots answered DURING the run -- and a
        # delivery that opens a modal dialog is exactly when a slow scan
        # lands. `_run_delivery` swallows what a delivery raises, so one bad
        # drop cannot break the loop.
        while queue and queue[0].answered:
            _run_delivery(queue.pop(0))
    finally:
        for index, active in enumerate(_draining):
            if active is queue:
                del _draining[index]
                break


def _queue_drop(screen, deliver: Callable[[object], None]):
    """Take this drop's place in ``screen``'s delivery order.

    :returns: the slot to answer when the scan lands, or ``None`` for a
        screen that cannot be tracked -- an object with no weak reference has
        nowhere to keep a queue, and an unordered delivery beats no delivery.
    """
    try:
        queue = _pending_drops.get(screen)
        if queue is None:
            queue = []
            _pending_drops[screen] = queue
    except TypeError:               # not weak-referenceable / not hashable
        return None
    # A new drop is the moment to notice that an older one can never be
    # delivered, and to stop holding this one behind it.
    _forget_abandoned(screen, queue)
    slot = _PendingDrop(deliver)
    queue.append(slot)
    # Anything that was waiting behind a slot just written off is ready now.
    # The new slot is at the BACK and unanswered, so it holds nothing up and
    # nothing here delivers it.
    _drain(queue)
    return slot


def _answer_drop(screen, slot, report) -> None:
    """Fill ``slot`` in, then deliver every drop now ready, oldest first."""
    if slot is None:                # untracked screen: nothing to order
        return
    slot.report = report
    slot.answered = True
    queue = _pending_drops.get(screen)
    if queue is None or slot not in queue:
        # This slot lost its place in line: the screen's queue is gone, or a
        # later drop wrote this one off as abandoned (:func:`_forget_abandoned`)
        # and the scan answered after all. Deliver it where it stands -- the
        # ordering guarantee went with its place, and a drop delivered out of
        # order still beats the silent no-op of one never delivered.
        _run_delivery(slot)
        return
    _drain(queue)


def _route_drop(paths: Sequence[Path], handler: DropHandler, screen) -> None:
    """Classify ``paths`` off the GUI thread, then act on the result on it.

    The scan goes through :mod:`spacr.qt.dnd_handlers`' per-screen drop
    scanner rather than a JobRunner of our own, because that is the same
    facility a mask drop already uses for the same reason, it parents its
    thread to the screen (Qt aborts the process when a running QThread is
    destroyed), it guards against the screen having gone by the time the
    answer lands, and it runs the scan inline when there is nowhere to keep a
    thread -- better a stall than a drop that reports nothing.

    The import is deferred because :mod:`spacr.qt.dnd_handlers` imports this
    module at the top; a private name, because that module is not ours to
    edit and the scanner it owns is exactly the thing needed here.

    Concurrent drops on one screen are delivered in the order they were
    made, whatever order their scans finish in -- see :data:`_pending_drops`.
    """
    takes_csv = hasattr(screen, "apply_settings_dict")
    multiple = bool(handler.accepts_multiple())

    def scan():
        return _classify_drop(paths, handler, takes_csv, multiple)

    slot = _queue_drop(screen, lambda report: _deliver_drop(
        report, handler, screen))

    def deliver(report):
        if slot is None:
            # Untracked screen: no queue to order it against, and no
            # `_answer_drop` to keep a delivery that raises to itself. It is
            # kept here instead -- see :func:`_run_delivery` for why an
            # exception must not leave this callback.
            untracked = _PendingDrop(
                lambda answer: _deliver_drop(answer, handler, screen))
            untracked.report = report
            _run_delivery(untracked)
            return
        _answer_drop(screen, slot, report)

    try:
        from .dnd_handlers import _scan_then
    except Exception:                                            # noqa: BLE001
        LOG.debug("no drop scanner available; classifying inline",
                  exc_info=True)
        deliver(scan())
        return
    try:
        dispatched = bool(_scan_then(screen, scan, deliver))
    except Exception:                                            # noqa: BLE001
        # The scanner refused outright. Answer the slot anyway: an
        # unanswered slot is a screen whose every later drop is silently
        # queued behind it forever.
        LOG.debug("the drop scanner refused; classifying inline",
                  exc_info=True)
        deliver(scan())
        return
    if not dispatched and slot is not None and not slot.answered:
        # ``_scan_then`` returns False both for a scan it ran INLINE and for
        # one that RAISED there -- and in the second case it never calls
        # back at all. Left alone, this drop's slot would stay unanswered at
        # the head of the screen's queue and hold every later drop behind it
        # for the life of the window. :func:`_classify_drop` is written never
        # to raise, so this is the guard for the day something beneath it
        # does; the drop is reported rather than silently forgotten.
        LOG.warning("a dropped path was never classified: %s",
                    ", ".join(str(item) for item in paths))
        deliver(_unclassified(paths))


def _unclassified(paths: Sequence[Path]) -> List[dict]:
    """A report that says only "this drop could never be looked at".

    Delivered when a classification did not come back at all. EVERY path is
    reported as rejected rather than left out: a drop that produces no
    console line, no status line and no dialog is exactly the silent no-op
    the delivery queue exists to prevent, and the user has already been shown
    the drag landing.
    """
    return [{"path": path, "csv": None, "accepted": False,
             "error": "the drop could not be classified"}
            for path in paths]


def _find_console(screen):
    """Return the nearest spaCR console, including the host app's console."""
    console = getattr(screen, "_console", None)
    if console is not None:
        return console
    try:
        window = screen.window()
    except Exception:
        return None
    console = getattr(window, "_console", None)
    if console is not None:
        return console
    # Standalone tool screens are hosted alongside AppScreens. Prefer the
    # most recently visited screen so rejected drops never disappear merely
    # because the tool itself has no embedded console.
    screens = getattr(window, "_screens", {}) or {}
    visit_order = list(getattr(window, "_visit_order", []) or [])
    for key in reversed(visit_order + list(screens)):
        candidate = screens.get(key)
        console = getattr(candidate, "_console", None)
        if console is not None:
            return console
    try:
        from spacr.qt.widgets.console_panel import ConsolePanel
        consoles = window.findChildren(ConsolePanel)
        if consoles:
            return consoles[-1]
    except Exception:
        pass
    return None


def _report_drop_problem(screen, path: Path, reason: str, suggestion: str,
                         alternatives: Sequence[Path] = ()) -> str:
    """Print an actionable rejected-drop report and optionally ask the AI."""
    lines = [
        f"[drop rejected] {path}",
        f"Reason: {reason}",
        f"Suggestion: {suggestion}",
    ]
    if alternatives:
        lines.append(
            "Compatible nearby paths: " +
            ", ".join(str(item) for item in alternatives)
        )
    message = "\n".join(lines) + "\n"
    LOG.warning(message.rstrip())
    console = _find_console(screen)
    displayed_inline = False
    if console is not None:
        append = getattr(console, "append_error", None) or getattr(
            console, "append_stdout", None)
        if append is not None:
            append(message)
            displayed_inline = True
        try:
            from spacr.qt.ai.settings import get_route_errors_through_ai
            provider = console._current_provider()
            ai_active = getattr(console, "_ai_active", False)
            if callable(ai_active):
                ai_active = ai_active()
            if (get_route_errors_through_ai()
                    and bool(ai_active)
                    and provider is not None):
                console.open_error_flow(
                    message,
                    active_app=getattr(screen, "app_key", None),
                    show_raw=False,
                )
        except Exception:
            LOG.debug("Could not route rejected drop through AI",
                      exc_info=True)
    # Standalone tools use a read-only summary/log pane instead of an
    # AppScreen ConsolePanel. Put the same actionable text there as well.
    if not displayed_inline:
        for attr in ("_summary", "_log", "_console_text"):
            widget = getattr(screen, attr, None)
            append = getattr(widget, "appendPlainText", None)
            if callable(append):
                append(message.rstrip())
                displayed_inline = True
                break
    status = getattr(screen, "_set_status", None)
    if callable(status):
        try:
            status(f"Drop rejected: {reason} Suggestion: {suggestion}")
        except Exception:
            pass
    return message


# ---------------------------------------------------------------------------
# Mime helpers
# ---------------------------------------------------------------------------

def _mime_has_local_paths(mime: QMimeData) -> bool:
    if not mime.hasUrls():
        return False
    return any(u.isLocalFile() for u in mime.urls())


def _mime_local_paths(mime: QMimeData) -> List[Path]:
    return [Path(u.toLocalFile()) for u in mime.urls()
            if u.isLocalFile()]


# ---------------------------------------------------------------------------
# Universal CSV → settings importer
# ---------------------------------------------------------------------------

#: Header shapes that identify a CSV as a spaCR SETTINGS export rather than
#: data. Everything else dropped on a screen is data for one of its inputs.
_SETTINGS_HEADER_PAIRS = (("key", "value"), ("setting_key", "setting_value"))


def _csv_header(path: Path) -> list:
    """The first row's column names, read without loading the file.

    WORKER THREAD ONLY -- it opens the file, and the file is one the user
    dropped. See :meth:`_DropzoneFilter._on_drop`.

    Empty for anything whose first row cannot be read, which is the honest
    answer and not only the convenient one: the callers all ask "does this
    header say settings?", and a file with no readable header does not.
    ``OSError`` alone was too narrow -- ``csv`` raises its own ``Error`` for
    a first field longer than ``csv.field_size_limit()`` (128 KiB), which is
    every binary anyone ever renamed to ``.csv``, and that exception used to
    travel all the way out through Qt's event delivery.
    """
    try:
        import csv as _csv
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            row = next(_csv.reader(fh), [])
        return [str(name).strip().lower() for name in row]
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not read a CSV header from %s", path, exc_info=True)
        return []


def _looks_like_settings_csv(path: Path) -> bool:
    """Whether ``path``'s header says settings. WORKER THREAD ONLY: reads."""
    return _header_is_settings(_csv_header(path))


def _header_is_settings(header: Sequence[str]) -> bool:
    """Whether a header ALREADY READ says settings. Free; no disk."""
    names = set(header or ())
    return any(set(pair) <= names for pair in _SETTINGS_HEADER_PAIRS)


def _read_settings_csv(path: Path) -> dict:
    """Read a dropped CSV far enough to route it. WORKER THREAD ONLY.

    Everything :func:`_apply_settings_csv` needs off the disk, taken once on
    a thread: the header both routers key on, and -- when the header says
    settings -- the parsed settings themselves. ``load_settings`` reads the
    whole file, which is exactly as unsafe on the GUI thread as the stat that
    started all this.

    :returns: ``{"header": [...], "settings": dict|None, "error": str|None}``.
        ``settings`` stays None for a data CSV, which is not an error: it
        goes to one of the screen's file inputs instead.
    """
    scan = {"header": _csv_header(path), "settings": None, "error": None}
    if not _header_is_settings(scan["header"]):
        return scan
    try:
        from spacr.utils import load_settings
        # spaCR's own save_settings writes Key/Value columns; other tools
        # (and older spaCR CSVs) use setting_key/setting_value. load_settings
        # RAISES on a column mismatch rather than returning something
        # non-dict, so the second form has to be tried in its own except —
        # otherwise the fallback was unreachable and every
        # setting_key/setting_value CSV was reported as a failed import.
        try:
            loaded = load_settings(str(path),
                                     setting_key="Key",
                                     setting_value="Value")
        except Exception:
            loaded = None
        if not isinstance(loaded, dict):
            loaded = load_settings(str(path))
        if isinstance(loaded, dict):
            scan["settings"] = loaded
    except Exception as exc:                                     # noqa: BLE001
        scan["error"] = str(exc)
    return scan


def _route_data_csv_to_inputs(path: Path, screen, header=None):
    """Offer a data CSV to the screen's file inputs. Returns the key it took.

    Dropping ``plate1_dv.csv`` on the regression screen used to go to the
    settings importer, which reported "CSV file must contain setting_key and
    setting_value columns" -- an accurate statement about a file that never
    claimed to be settings, and a dead end for the very gesture the input
    widgets exist to support.

    Score and count tables are told apart by their header: a count export
    carries a gRNA name and a count, a score export carries neither.

    :param header: the header :func:`_read_settings_csv` already took off the
        disk on a worker thread. Omitting it reads the file HERE, which is
        why the drop path always passes one.
    """
    model = getattr(screen, "_settings_model", None)
    widgets = getattr(model, "_widgets", {}) if model is not None else {}
    if not widgets:
        return None
    try:
        from .widgets.file_list import FilePathListWidget
    except Exception:  # pragma: no cover - Qt import guard
        return None

    header = set(_csv_header(path) if header is None else header)
    is_count = bool({"grna", "grna_name"} & header) and "count" in header

    # An ANNOTATION table is neither side of the pairing.
    #
    # Classifying only count-vs-score meant everything that was not a count
    # became a score, so a gRNA barcode export (name, sequence) landed in the
    # score column of the pairing table. It has no plate, no well and no
    # response; it annotates results after the fit. Recognised by carrying an
    # identifier and no per-well coordinates.
    identifiers = {"name", "gene id", "gene_id", "geneid", "gene", "grna",
                   "grna_name"}
    coordinates = {"row", "rowid", "row_name", "col", "column", "columnid",
                   "column_name", "prc", "well", "plate", "plateid"}
    is_metadata = (not is_count
                   and bool(identifiers & header)
                   and not (coordinates & header))
    if is_metadata:
        widget = widgets.get("metadata_files")
        if isinstance(widget, FilePathListWidget):
            widget.add_paths([str(path)])
            return "metadata_files"

    # THE PAIRED TABLE IS TRIED FIRST, and that ordering is the whole fix.
    #
    # The regression panel replaced its separate score_data / count_data
    # lists with one paired_data table. This router looked for those two keys
    # as FilePathListWidgets, found neither, and fell through to
    # metadata_files -- the only FilePathListWidget left on the screen. So
    # every CSV dropped on the regression panel went to metadata: score
    # tables, count tables, all of it.
    paired = widgets.get("paired_data")
    adder = getattr(paired, "add_paths_for_side", None)
    if callable(adder):
        adder([str(path)], "count" if is_count else "score")
        return "paired_data (count)" if is_count else "paired_data (score)"

    # Most specific first: a count table must not land in the score slot just
    # because that widget happens to come first in the panel.
    preferred = ("count_data", "score_data") if is_count else \
        ("score_data", "count_data")
    for key in (*preferred, "metadata_files"):
        widget = widgets.get(key)
        if isinstance(widget, FilePathListWidget):
            widget.add_paths([str(path)])
            return key
    return None


def _apply_settings_csv(path: Path, screen,
                        scan: Optional[dict] = None) -> None:
    """Import a settings CSV, or route a data CSV to the screen's inputs.

    GUI-thread half: it touches widgets, so it may not touch the disk. What
    the disk had to say arrives in ``scan``.

    Silent no-op if the screen doesn't have that method (AnnotateScreen,
    MakeMasksScreen — they don't use the SettingsWidgets model).

    :param scan: a finished :func:`_read_settings_csv`. ``None`` reads the
        file here, which is correct for a caller that is already off the GUI
        thread (or in a test) and wrong for the drop path -- see
        :meth:`_DropzoneFilter._on_drop`.
    """
    if not hasattr(screen, "apply_settings_dict"):
        return
    if scan is None:
        scan = _read_settings_csv(path)
    header = scan.get("header") or []
    # A dropped file is DATA unless its header says it is settings. Deciding
    # by header rather than by extension is what lets the regression screen
    # accept four score CSVs and four count CSVs by drag and drop.
    if not _header_is_settings(header):
        taken = _route_data_csv_to_inputs(path, screen, header=header)
        if taken:
            if hasattr(screen, "_console"):
                screen._console.append_stdout(
                    f"[drop] added {path.name} to {taken}\n")
            return
        _report_drop_problem(
            screen, path,
            f"{path.name} is not a settings CSV, and this screen has no file "
            f"input that accepts it.",
            "Settings CSVs have Key/Value or setting_key/setting_value "
            "columns. Data CSVs can be dropped on a screen that has a score, "
            "count or metadata input.")
        return
    failure = scan.get("error")
    if failure is None:
        try:
            loaded = scan.get("settings")
            if isinstance(loaded, dict):
                n = screen.apply_settings_dict(loaded)
                if hasattr(screen, "_console"):
                    screen._console.append_stdout(
                        f"[drop] imported {n} settings from {path.name}\n"
                    )
            return
        except Exception as e:
            # The read succeeded and the screen refused what it read. Same
            # report as a failed read, because to the user it is the same
            # sentence: this CSV did not become settings.
            failure = str(e)
    _report_drop_problem(
        screen, path, f"Settings CSV import failed: {failure}",
        "Export a settings CSV from spaCR, or verify that the file has "
        "Key/Value or setting_key/setting_value columns.",
    )
    QMessageBox.warning(screen, "CSV import failed", str(failure))


# ---------------------------------------------------------------------------
# "Did you mean X?" dialog
# ---------------------------------------------------------------------------

def suggest_alternatives_dialog(
    parent, original: Path, alternatives: Sequence[Path], why: str = "",
) -> Optional[Path]:
    """Modal that lets the user pick from ``alternatives``.

    :returns: the chosen Path, or None if cancelled.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Did you mean…")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)

    header = QLabel(
        f"<b>{original.name}</b> can't be used as-is."
        + (f"<br><span style='color:gray;'>{why}</span>" if why else "")
        + "<br><br>Nearby folders that WOULD work:"
    )
    header.setTextFormat(Qt.RichText)
    header.setWordWrap(True)
    layout.addWidget(header)

    lst = QListWidget()
    for alt in alternatives:
        item = QListWidgetItem(str(alt))
        lst.addItem(item)
    lst.setCurrentRow(0)
    layout.addWidget(lst, 1)

    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None
    row = lst.currentRow()
    if row < 0:
        return None
    return alternatives[row]


def choose_one_dialog(parent, headline: str, question: str,
                      options: Sequence[str]) -> Optional[str]:
    """Ask which of ``options`` was meant. ``None`` when nobody answered.

    Distinct from :func:`suggest_alternatives_dialog`, which says the drop
    *cannot be used*. This one is asked when the drop resolved perfectly and
    landed on more than one right answer — two tables in the database, two
    masks in ``masks/`` — where "did you mean…" would be telling the user
    they made a mistake they did not make.

    :param parent: the widget to centre the dialog on.
    :param headline: what was found, e.g. "plate1.db holds 4 tables."
    :param question: what is being asked, e.g. "Which one should be loaded?"
    :param options: the candidates, in the order to offer them.
    :returns: the chosen option, or None when cancelled.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Which one?")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)

    header = QLabel(f"<b>{headline}</b><br>{question}")
    header.setTextFormat(Qt.RichText)
    header.setWordWrap(True)
    layout.addWidget(header)

    listing = QListWidget()
    for option in options:
        listing.addItem(QListWidgetItem(str(option)))
    listing.setCurrentRow(0)
    listing.itemDoubleClicked.connect(lambda *_: dlg.accept())
    layout.addWidget(listing, 1)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None
    row = listing.currentRow()
    return None if row < 0 else str(options[row])


# ---------------------------------------------------------------------------
# Filesystem helpers reused by handlers
#
# WORKER-THREAD ONLY, ALL THREE. They list directories, and the directory is
# always one the user dropped -- which on the maintainer's machine reaches
# ``/nas_mnt`` shares behind an ``autofs`` mount that took more than twenty
# seconds to answer a single stat on 2026-09-04. Called from a handler's
# ``can_accept`` or ``suggest_alternatives``, they are already on the drop
# scanner's thread (see :func:`_route_drop`); called from anywhere that draws,
# they are the freeze. Same contract as the scans in
# :mod:`spacr.qt.dnd_handlers`: no Qt, no widgets, data out.
# ---------------------------------------------------------------------------

def has_images_in(path: Path, min_count: int = 1,
                    exts: Sequence[str] = IMAGE_EXTS) -> bool:
    """Return True if ``path`` contains at least ``min_count`` image
    files at its top level (does not recurse). Worker thread only."""
    if not path.is_dir():
        return False
    count = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in exts:
            count += 1
            if count >= min_count:
                return True
    return False


def find_image_folders_nearby(path: Path, max_depth: int = 1,
                                min_count: int = 1) -> List[Path]:
    """Search parent + immediate children of ``path`` for folders that
    contain images. Excludes ``path`` itself if it already qualifies.

    Handy for the "did you mean X?" prompt when the user drops the
    wrong sibling of a plate folder. Worker thread only: it lists two levels
    of a folder the user chose.
    """
    hits: List[Path] = []
    # One level up: check siblings
    if path.parent and path.parent.is_dir():
        for sib in path.parent.iterdir():
            if sib.is_dir() and sib != path and has_images_in(sib, min_count):
                hits.append(sib)
    # One level down: check immediate children
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and has_images_in(child, min_count):
                hits.append(child)
    return hits


def sample_image_names(path: Path, n: int = 8,
                         exts: Sequence[str] = IMAGE_EXTS) -> List[Path]:
    """Return up to ``n`` image paths from ``path`` — used by the
    filename-regex preview in the mask handler. Worker thread only."""
    if not path.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in exts:
            out.append(child)
            if len(out) >= n:
                break
    return out
