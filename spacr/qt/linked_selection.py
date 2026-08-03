"""Process-wide linked selection and filter, so the views stop being islands.

:mod:`spacr.selection` holds the logic and knows nothing about Qt. This module
is the thin part that makes it *shared*: one object per process that every open
view subscribes to, so lassoing a cluster in the UMAP highlights the same cells
on the plate heatmap, in the measurement table and in the crop grid.

Why a singleton rather than passing a model around
--------------------------------------------------

The views are constructed independently by ``AppScreen`` as the user opens
tabs; none of them owns another, and there is no common parent below the main
window to hang a shared model off. A process-wide accessor is the same shape
:func:`spacr.qt.bridge.registry` already uses for run state, for the same
reason, and it means a view added later joins the conversation by importing one
function.

The subscription rule
---------------------

**Views must disconnect in ``closeEvent``.** This object outlives every screen,
and holds plain references to whatever connected to it. A lambda would keep a
destroyed page alive as a receiver — the exact leak
:class:`spacr.qt.widgets.home.HomePage` documents for the run registry — so
connect bound methods and drop them on close.

:class:`LinkedView` is that rule written down once. A view joins with three
lines rather than re-deriving the echo-suppression and disconnect dance::

    class UmapView(LinkedView, QWidget):          # 1. mix it in, FIRST
        def __init__(self, parent=None):
            super().__init__(parent)
            self.link_selection("umap")           # 2. subscribe + name yourself

        def closeEvent(self, event):
            self.unlink_selection()               # 3. and let go
            super().closeEvent(event)

        # then override only what it cares about
        def on_linked_selection_changed(self, selection):
            self._repaint_highlight(selection)

and publishes with ``self.publish_selection(frame_or_keys)``, which stamps the
view's own name so the view does not repaint for the echo of its own lasso.

Opening objects somewhere else
------------------------------

Linking answers "we are all looking at the same population". It does not
answer "show me *these twelve* crops, in this order, because a model got them
wrong" — that is a jump to another view, not a change of shared state, so it
travels as a one-shot :class:`~spacr.selection.ObjectRequest` through
:func:`open_objects` instead.

The point of routing it is that neither end imports the other. A scatter plot
that wants a crop shown, and a confusion matrix that wants a cell's errors
shown, both call :func:`open_objects`; the Annotate screen calls
:func:`register_object_opener` once. Neither plot has to know Annotate exists,
and Annotate does not grow a method per caller.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ..selection import DataFilter, ObjectRequest, Selection, as_key_index

__all__ = [
    "LinkedSelection",
    "LinkedView",
    "NoObjectOpener",
    "ObjectOpener",
    "DEFAULT_OPEN_KIND",
    "linked_selection",
    "open_objects",
    "open_request",
    "register_object_opener",
    "unregister_object_opener",
    "has_object_opener",
    "object_opener_kinds",
]

#: The destination :func:`open_objects` routes to when the caller does not say
#: otherwise. "Show me these objects" almost always means "show me the crops",
#: and the crop grid is Annotate.
DEFAULT_OPEN_KIND = "annotate"

#: What :func:`register_object_opener` takes: one call, one request, any
#: return value (the screen that opened, ``True``, or ``None`` — it is passed
#: back to the caller unchanged).
ObjectOpener = Callable[[ObjectRequest], Any]


class NoObjectOpener(LookupError):
    """Nothing is registered to open objects of the requested kind.

    Raised rather than returning quietly. A silent no-op here is a button that
    does nothing on click, which is indistinguishable from a slow one and gets
    reported as "the app froze". A caller that legitimately might have no
    destination — a context-menu entry it wants to grey out rather than
    offer — should ask :func:`has_object_opener` first.
    """


class LinkedSelection(QObject):
    """The shared filter and selection every linked view reads.

    Signals:
        filter_changed()      — the population narrowed or widened
        selection_changed()   — the highlighted subset moved
        objects_opened(request) — an :class:`~spacr.selection.ObjectRequest`
            was routed somewhere. Emitted after the opener returned, so a view
            following along never chases a jump that failed.

    The first two are separate signals because they cost different amounts to
    honour. A filter change means a view has to re-query and re-lay-out; a
    selection change usually means it only has to repaint. Collapsing them into
    one ``changed`` would make every lasso trigger a full reload of a
    million-row table.

    The opener registry lives on the instance rather than in a module global
    so that a test — or a second window — gets its own routing table by
    constructing its own :class:`LinkedSelection`.
    """

    filter_changed = Signal()
    selection_changed = Signal()
    objects_opened = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter = DataFilter()
        self._selection = Selection.none()
        self._openers: Dict[str, ObjectOpener] = {}

    # -- filter --------------------------------------------------------
    @property
    def filter(self) -> DataFilter:
        """The active filter. Mutate through :meth:`set_filter`, not in place.

        Returned rather than copied because a copy per read would be wasteful
        on a hot path, but mutating it directly will not emit — which is the
        one way to get views showing different populations.
        """
        return self._filter

    def set_filter(self, data_filter: DataFilter) -> None:
        """Replace the filter and tell every view, even if it looks the same.

        No equality short-circuit on purpose. ``DataFilter`` holds a list of
        dataclasses, and a caller that mutated one in place then handed the
        same object back would compare equal to itself and emit nothing,
        leaving the views showing a population that no longer matches the
        controls.
        """
        self._filter = data_filter
        self.filter_changed.emit()

    def clear_filter(self) -> None:
        self.set_filter(DataFilter())

    # -- selection -----------------------------------------------------
    @property
    def selection(self) -> Selection:
        return self._selection

    def set_selection(self, selection: Selection) -> None:
        """Publish a new highlighted subset.

        ``selection.source`` names the view that made it, so a view can ignore
        the echo of its own selection rather than re-applying what it just
        drew — which otherwise costs a repaint per view per lasso, and can
        loop if a view normalises what it publishes.
        """
        self._selection = selection
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        """Return to the resting state.

        Distinct from selecting nothing: :class:`spacr.selection.Selection`
        keeps "no selection" and "an empty selection" apart so views can draw
        the resting state differently from a lasso that caught nothing.
        """
        self.set_selection(Selection.none())

    def select_frame(self, frame: pd.DataFrame, source: str = "",
                     *, timelapse: bool = False) -> None:
        """Convenience: publish the rows of ``frame`` as the selection."""
        self.set_selection(
            Selection.from_frame(frame, source=source, timelapse=timelapse))

    # -- convenience for views -----------------------------------------
    def visible(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` narrowed by the active filter.

        The one call a view needs to honour the filter. Selection is deliberately
        NOT applied — a selection highlights, it does not hide, and a view that
        dropped unselected rows would make the lasso destructive.
        """
        return self._filter.apply(frame)

    # -- routing objects to whatever can show them ----------------------
    def register_object_opener(self, kind: str,
                               fn: ObjectOpener) -> Optional[ObjectOpener]:
        """Offer to open objects of ``kind``, and return whoever had it before.

        Registering a kind that is already taken REPLACES it, because that is
        what re-opening a screen looks like: the second Annotate is the live
        one and the first is on its way out. The displaced opener is returned
        rather than dropped so a caller that wants to chain or restore it can.

        The matching :meth:`unregister_object_opener` is identity-checked for
        the same reason — a screen closing must not take the registration of
        the screen that replaced it.

        :raises ValueError: on a blank kind.
        :raises TypeError: if ``fn`` is not callable — caught here, where the
            registration is, rather than at the click that would have used it.
        """
        key = str(kind).strip()
        if not key:
            raise ValueError("an object opener needs a non-blank kind")
        if not callable(fn):
            raise TypeError(
                f"object opener for {key!r} is not callable "
                f"({type(fn).__name__})")
        previous = self._openers.get(key)
        self._openers[key] = fn
        return previous

    def unregister_object_opener(self, kind: str,
                                 fn: Optional[ObjectOpener] = None) -> bool:
        """Withdraw ``kind``; ``True`` if this call is what removed it.

        With ``fn`` given, removes it only if that is the opener currently
        registered. A closing screen should always pass its own opener: two
        Annotate screens opened in a session means the first one's
        ``closeEvent`` runs *after* the second has registered, and an
        unconditional withdrawal would leave the live screen unreachable.
        """
        key = str(kind).strip()
        current = self._openers.get(key)
        if current is None or (fn is not None and current is not fn):
            return False
        del self._openers[key]
        return True

    def has_object_opener(self, kind: str) -> bool:
        """Whether anything is registered for ``kind``.

        For greying out a menu entry rather than offering one that raises.
        """
        return str(kind).strip() in self._openers

    def object_opener_kinds(self) -> Tuple[str, ...]:
        """Every registered kind, sorted — for diagnostics and menus."""
        return tuple(sorted(self._openers))

    def open_objects(self, keys: Any, *, reason: str,
                     kind: str = DEFAULT_OPEN_KIND, source: str = "",
                     timelapse: bool = False,
                     context: Optional[Mapping[str, Any]] = None) -> Any:
        """Show exactly these objects, wherever ``kind`` is registered.

        The instance-level form of the module function of the same name; see
        :func:`open_objects` for the argument contract.
        """
        return self.open_request(ObjectRequest(
            keys=keys, reason=reason, source=source, kind=kind,
            timelapse=timelapse, context=context or {}))

    def open_request(self, request: ObjectRequest) -> Any:
        """Route an already-built request, and return what the opener returned.

        Split from :meth:`open_objects` so a request can be built where the
        data is — in a worker, off the event loop — and routed later on the
        GUI thread, without that hop having to re-list every argument.

        The request is NOT published as the shared selection. Opening a subset
        somewhere and highlighting it everywhere are separate acts, and doing
        both here would wipe the lasso the user opened it from. A receiver
        that wants both publishes :meth:`~spacr.selection.ObjectRequest.as_selection`.

        :raises NoObjectOpener: if nothing is registered for the kind.
        """
        kind = request.kind or DEFAULT_OPEN_KIND
        opener = self._openers.get(kind)
        if opener is None:
            known = ", ".join(self.object_opener_kinds()) or "nothing"
            raise NoObjectOpener(
                f"nothing is registered to open {kind!r} objects "
                f"(registered: {known}). The screen that opens them may not "
                f"have been created yet — ask has_object_opener({kind!r}) "
                f"before offering the action.")
        if request.kind != kind:
            request = replace(request, kind=kind)
        result = opener(request)
        self.objects_opened.emit(request)
        return result


class LinkedView:
    """What a view mixes in to join the linked selection.

    A mixin rather than a base class: the views are already
    ``QWidget`` subclasses, and this holds no Qt state of its own — no
    ``__init__``, only class-level defaults — so it composes with any of them.
    **List it first** (``class UmapView(LinkedView, QWidget)``), so its methods
    win over anything Qt happens to name the same.

    The three lines of the contract are in the module docstring. What the
    mixin buys over hand-wiring, beyond the typing:

    * **Echo suppression.** :meth:`publish_selection` stamps the view's own
      name, and the subscriber drops selections carrying it. Without that,
      every lasso costs the drawing view a repaint of what it already drew,
      and a view that normalises what it publishes oscillates.
    * **One disconnect.** :meth:`unlink_selection` is idempotent and
      flag-guarded: Qt does not raise on a disconnect that finds nothing, it
      prints ``libpyside: Failed to disconnect`` to stderr where no ``except``
      can reach it, and a screen closed twice — which Qt does on teardown —
      would print one every time.
    * **Bound methods only.** The process-wide link outlives every screen; the
      slots it holds are bound methods of the view, so Qt severs them when the
      widget is destroyed even if a ``closeEvent`` is missed.
    """

    #: The name this view publishes under, and the name it ignores the echo
    #: of. Set by :meth:`link_selection`.
    link_source: str = ""
    _link: Optional[LinkedSelection] = None
    _link_connected: bool = False
    _link_echo: bool = False

    # -- opting in ------------------------------------------------------
    def link_selection(self, source: str, *,
                       link: Optional[LinkedSelection] = None,
                       echo: bool = False) -> LinkedSelection:
        """Subscribe to the shared filter and selection as ``source``.

        :param source: this view's name, stamped onto everything it publishes.
        :param link: the :class:`LinkedSelection` to join. Defaults to the
            process-wide one; a test (or a second window) passes its own.
        :param echo: hear your own published selections too. Off by default —
            a view that just drew a lasso does not need to be told about it.
        :returns: the link, so a caller can keep the reference.

        Calling this twice re-subscribes rather than double-subscribing: a
        screen that re-links on reload would otherwise get two callbacks per
        change, and repaint twice for every lasso for the rest of the session.
        """
        if self._link_connected:
            self.unlink_selection()
        self.link_source = str(source)
        self._link_echo = bool(echo)
        self._link = link if link is not None else linked_selection()
        self._link.filter_changed.connect(self._linked_filter_changed)
        self._link.selection_changed.connect(self._linked_selection_changed)
        self._link_connected = True
        return self._link

    def unlink_selection(self) -> None:
        """Stop listening. Safe to call on a view that never linked."""
        if not self._link_connected:
            return
        self._link_connected = False
        self._link.filter_changed.disconnect(self._linked_filter_changed)
        self._link.selection_changed.disconnect(self._linked_selection_changed)

    @property
    def is_linked(self) -> bool:
        """Whether this view is currently subscribed."""
        return self._link_connected

    @property
    def link(self) -> LinkedSelection:
        """The link this view is on — the process-wide one until it joins one.

        Non-``None`` before :meth:`link_selection`, deliberately: publishing
        without subscribing is a legitimate half of the contract (a view that
        drives the others but redraws itself), and it should not need to
        subscribe just to reach ``visible()``.
        """
        return self._link if self._link is not None else linked_selection()

    # -- what the view overrides ----------------------------------------
    def on_linked_filter_changed(self, data_filter: DataFilter) -> None:
        """The shared population moved: re-query and re-lay-out.

        Default: nothing, so a view can subscribe for selections alone.
        """

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """The highlighted subset moved: repaint.

        Not called for this view's own selections unless it linked with
        ``echo=True``. Remember that ``selection.keys is None`` is the resting
        state — draw it differently from a lasso that caught nothing.

        Default: nothing, so a view can subscribe for the filter alone.
        """

    # -- publishing ------------------------------------------------------
    def publish_selection(self, keys: Any, *,
                          timelapse: bool = False) -> Selection:
        """Publish ``keys`` as the shared highlight, stamped with this view.

        ``keys`` is anything :func:`~spacr.selection.as_key_index` takes — the
        lassoed rows as a frame, or bare keys.
        """
        selection = Selection(keys=as_key_index(keys, timelapse=timelapse),
                              source=self.link_source)
        self.link.set_selection(selection)
        return selection

    def clear_linked_selection(self) -> None:
        """Return everyone to the resting state (not to an empty selection)."""
        self.link.clear_selection()

    def publish_filter(self, data_filter: DataFilter) -> None:
        """Narrow the shared population from this view."""
        self.link.set_filter(data_filter)

    def linked_visible(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` narrowed by the shared filter. A selection never hides."""
        return self.link.visible(frame)

    def open_objects(self, keys: Any, *, reason: str,
                     kind: str = DEFAULT_OPEN_KIND,
                     timelapse: bool = False,
                     context: Optional[Mapping[str, Any]] = None) -> Any:
        """Ask for these objects to be shown, as this view.

        The same call as :func:`open_objects` with ``source`` filled in.
        """
        return self.link.open_objects(
            keys, reason=reason, kind=kind, source=self.link_source,
            timelapse=timelapse, context=context)

    # -- internals -------------------------------------------------------
    def _linked_filter_changed(self) -> None:
        self.on_linked_filter_changed(self.link.filter)

    def _linked_selection_changed(self) -> None:
        selection = self.link.selection
        if (not self._link_echo and self.link_source
                and selection.source == self.link_source):
            return
        self.on_linked_selection_changed(selection)


_LINKED: Optional[LinkedSelection] = None


def linked_selection() -> LinkedSelection:
    """The process-wide :class:`LinkedSelection` (created on first use)."""
    global _LINKED
    if _LINKED is None:
        _LINKED = LinkedSelection()
    return _LINKED


def register_object_opener(kind: str, fn: ObjectOpener) -> Optional[ObjectOpener]:
    """Offer to open objects of ``kind`` process-wide.

    The half of the routing contract a *destination* implements. Annotate, in
    its constructor::

        register_object_opener("annotate", self.open_object_request)

    and in ``closeEvent``::

        unregister_object_opener("annotate", self.open_object_request)

    where ``open_object_request(request: ObjectRequest) -> Any`` is the one
    method it has to grow. Everything the caller wanted to say is on the
    request: ``request.keys`` (an Index of
    :data:`~spacr.selection.OBJECT_KEY_COLUMNS` keys, in the caller's order,
    de-duplicated), ``request.reason`` (put it in the header),
    ``request.source``, ``request.timelapse`` and ``request.context``. The
    return value is handed back to the caller unchanged.

    :returns: the opener this one displaced, or ``None``.
    """
    return linked_selection().register_object_opener(kind, fn)


def unregister_object_opener(kind: str,
                             fn: Optional[ObjectOpener] = None) -> bool:
    """Withdraw ``kind`` process-wide; ``True`` if this call removed it.

    Pass the opener you registered: with two screens of the same kind open,
    the one closing must not withdraw the one that replaced it.
    """
    return linked_selection().unregister_object_opener(kind, fn)


def has_object_opener(kind: str) -> bool:
    """Whether anything process-wide can open ``kind``.

    Ask before offering the action, so an unavailable destination is a greyed
    menu entry rather than a :class:`NoObjectOpener` on click.
    """
    return linked_selection().has_object_opener(kind)


def object_opener_kinds() -> Tuple[str, ...]:
    """Every kind registered process-wide, sorted."""
    return linked_selection().object_opener_kinds()


def open_objects(keys: Any, *, reason: str, kind: str = DEFAULT_OPEN_KIND,
                 source: str = "", timelapse: bool = False,
                 context: Optional[Mapping[str, Any]] = None) -> Any:
    """Show exactly these objects, wherever ``kind`` is registered.

    The half of the routing contract a *caller* uses. A scatter plot::

        open_objects(row, reason="clicked in the UMAP", source="umap")

    A confusion-matrix cell, worst errors first::

        open_objects(errors_sorted_by_confidence,
                     reason="predicted infected · annotated uninfected",
                     source="classifier_evaluation",
                     context={"scores": scores_by_key})

    Neither imports Annotate, and Annotate grows no method per caller.

    :param keys: what to open — a :class:`pandas.DataFrame` carrying
        :data:`~spacr.selection.OBJECT_KEY_COLUMNS`, a
        :class:`~spacr.selection.Selection`, one key string, or an iterable of
        key strings. Order is kept and duplicates dropped, so "worst first"
        survives the trip. See :func:`~spacr.selection.as_key_index`.
    :param reason: why these objects, in the words the destination will show.
        Required and non-blank.
    :param kind: the destination; defaults to :data:`DEFAULT_OPEN_KIND`.
    :param source: the view asking, for the destination's header.
    :param timelapse: the keys carry a timepoint.
    :param context: extras for the destination (per-key scores, a column to
        annotate into). Copied; the destination cannot mutate the caller's.
    :returns: whatever the opener returned, unchanged.
    :raises NoObjectOpener: nothing is registered for ``kind``.
    :raises ValueError: blank ``reason``, or a resting ``Selection``.
    :raises TypeError: ``keys`` is not something that names objects.

    An empty ``keys`` is not an error: a confusion-matrix cell with no errors
    in it is a real answer, and the destination showing "0 objects · <reason>"
    beats an exception every caller has to guard.
    """
    return linked_selection().open_objects(
        keys, reason=reason, kind=kind, source=source, timelapse=timelapse,
        context=context)


def open_request(request: ObjectRequest) -> Any:
    """Route an already-built :class:`~spacr.selection.ObjectRequest`.

    For building the request where the data is — off the event loop — and
    routing it on the GUI thread.
    """
    return linked_selection().open_request(request)
