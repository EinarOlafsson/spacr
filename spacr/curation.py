"""``B12`` ``C7`` — correcting a mask and a track by hand, on the record.

Two jobs that had no answer at all, and one rule that binds them.

**The masks are wrong in specific, obvious places.** Cellpose merges two
touching cells, or clips a lobe, and until now the only remedies were to
re-run segmentation with different parameters and hope, or to throw the field
away. Painting the fix takes four seconds and there was no brush.

**btrack output is never perfect.** A track breaks when a cell divides or
briefly leaves focus, and two tracks get swapped when cells touch. Timelapse
analysis downstream is only as good as the tracks, and there was no way to
join, split or delete one. That, more than the brush, is what has made
timelapse unusable: a velocity computed over a track that is really two cells
is not a noisy number, it is a wrong one.

**A corrected dataset must be distinguishable from a raw one.** This is the
rule, and it is why this module exists rather than a couple of mutating
helpers. A hand-edited mask that looks exactly like a segmented one is a
reproducibility hole: six months later nobody can say which fields were
touched, by whom, or what they looked like before — and a reviewer asking "did
you edit the data?" gets an answer based on memory. So every correction here
goes through :class:`CurationLog`: an append-only ledger, written beside the
artefact it describes, recording what changed, when, and to what. Nothing can
be edited without leaving one, because the edit methods are the only way in
and they all append.

What the ledger is, and is not
------------------------------

It is a *provenance* record, not an undo file. :meth:`MaskCuration.undo` works
off an in-memory history of exact pixel values and is bounded; the ledger
keeps one small JSON entry per action forever. Making one serve both would
mean either a ledger big enough to hold every painted voxel or an undo stack
that silently forgot.

It is deliberately a sidecar rather than a table in ``measurements.db``. Mask
and track curation happen on a folder of TIFFs long before (and often
without) a measurement database, and a provenance record that only exists once
you have measured is a record that misses the edits most worth having.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CurationError",
    "CurationEdit",
    "CurationLog",
    "LabelEdit",
    "MaskCuration",
    "TrackCuration",
    "LOG_SUFFIX",
    "TRACK_COLUMNS",
    "log_path_for",
    "is_curated",
]


class CurationError(ValueError):
    """A correction that cannot mean what it was asked to mean.

    Raised rather than silently doing nothing. A "join" button that quietly
    declines when the two tracks overlap in time leaves the user believing
    the join happened, and the ledger — which is the whole point — would then
    disagree with the data.
    """


#: What a curation ledger is called: the artefact's own name plus this. A
#: sidecar rather than a hidden file, because a provenance record nobody can
#: see when they copy the folder is a provenance record that gets lost.
LOG_SUFFIX = ".curation.json"

#: The canonical track table, from :data:`spacr.zstack.BASE_TRACK_COLUMNS`.
#: Imported lazily in :func:`_track_columns` so this module stays importable
#: without the z-stack machinery; named here so the contract is visible.
TRACK_COLUMNS: Tuple[str, ...] = ("frame", "track_id", "original_label",
                                  "x", "y")


def _track_columns() -> Tuple[str, ...]:
    """The track columns, from the one definition in :mod:`spacr.zstack`."""
    try:
        from .zstack import BASE_TRACK_COLUMNS

        return tuple(BASE_TRACK_COLUMNS)
    except Exception:
        return TRACK_COLUMNS


def _now() -> str:
    """UTC, to the second, in ISO-8601 with an explicit offset.

    UTC and not local time: a ledger is read on a different machine from the
    one that wrote it, and a naive timestamp from a laptop in another zone has
    reordered a sequence of edits before.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log_path_for(artifact: Any) -> str:
    """Where the ledger for ``artifact`` lives: ``<artifact>.curation.json``.

    Keyed on the full name including its extension, so ``masks.tif`` and
    ``masks.npy`` in one folder get their own ledgers instead of sharing one
    and interleaving two histories.

    :param artifact: the artefact's own path — anything :func:`os.fspath`
        accepts, so a :class:`pathlib.Path` as readily as a string. Nothing
        is opened or checked and the file need not exist, which is what lets
        a ledger be opened for a mask that is about to be written.
    """
    return f"{os.fspath(artifact)}{LOG_SUFFIX}"


def is_curated(artifact: Any) -> bool:
    """Whether ``artifact`` has been edited by hand — the question the rule
    exists to answer.

    ``True`` only when a ledger exists *and* holds at least one edit. An empty
    ledger left by a session that opened the brush and painted nothing is not
    a curated dataset, and reporting it as one would make the flag useless by
    making it always true.

    :param artifact: the artefact itself — the mask or tracks file, not its
        ledger; the ``.curation.json`` suffix is appended here. Only the
        sidecar is opened, so the artefact may be absent or unreadable
        without changing the answer. A sidecar that exists but will not parse
        answers ``True``: a damaged provenance record is a reason to be
        suspicious, not grounds for certifying the data as raw.
    """
    path = log_path_for(artifact)
    if not os.path.isfile(path):
        return False
    try:
        return bool(CurationLog.read(path).edits)
    except Exception:
        # An unreadable ledger is a reason to be suspicious, not a reason to
        # certify the data as raw.
        return True


@dataclass(frozen=True)
class CurationEdit:
    """One recorded correction.

    :ivar kind: what was done — ``"paint"``, ``"join"``, ``"split"``,
        ``"delete"``.
    :ivar target: what it was done to, in the terms of that kind: a label, a
        track id, a pair of track ids.
    :ivar when: UTC ISO-8601, from :func:`_now`.
    :ivar who: the operating-system user, so a shared dataset says which
        person made a call. Not identity in any security sense — it is the
        name to ask, not a signature.
    :ivar n_changed: how much moved (voxels painted, rows re-assigned). The
        number that makes a stroke that did nothing distinguishable from one
        that repainted a third of the field.
    :ivar detail: anything else worth keeping — the brush radius, the frame a
        split happened at, the labels that were overwritten.
    """

    kind: str
    target: Any
    when: str = field(default_factory=_now)
    who: str = ""
    n_changed: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "when": self.when,
                "who": self.who, "n_changed": int(self.n_changed),
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurationEdit":
        """Rebuild one edit from its :meth:`to_dict` form.

        :param data: a single entry out of a ledger's ``edits`` list. Every
            key is optional, so an entry written by an older version — or one
            hand-trimmed in the JSON — loads instead of taking the whole
            ledger down with it. A missing key becomes an empty value
            whatever the field declares: ``""`` for ``kind``, ``when`` and
            ``who``, ``None`` for ``target``, ``0`` for ``n_changed``, ``{}``
            for ``detail``. A missing ``when`` is therefore ``""`` and not
            the current time, so a reconstructed edit never claims a
            timestamp it does not have. Keys this class does not know are
            dropped; put anything you want kept under ``detail``.
        """
        return cls(kind=str(data.get("kind", "")),
                   target=data.get("target"),
                   when=str(data.get("when", "")),
                   who=str(data.get("who", "")),
                   n_changed=int(data.get("n_changed", 0) or 0),
                   detail=dict(data.get("detail") or {}))

    def describe(self) -> str:
        """One line, for a panel and for the ledger's own summary."""
        who = f" by {self.who}" if self.who else ""
        return (f"{self.when}{who}: {self.kind} {self.target} "
                f"({self.n_changed} changed)")


class CurationLog:
    """An append-only ledger of corrections to one artefact.

    :param artifact: what the edits were made to — a mask file, a tracks CSV.
        Recorded in the ledger so a file that gets renamed still says what it
        was when it was edited.
    :param source: what made the edits ("spacr-qt curation").

    Append-only in the API as well as in spirit: there is no ``remove`` and no
    way to rewrite an entry. An undone paint appends an ``undo`` edit rather
    than deleting the ``paint`` — the fact that something was painted and then
    taken back is itself part of what happened, and a ledger you can quietly
    tidy is not evidence of anything.
    """

    def __init__(self, artifact: Any = "", *, source: str = "spacr"):
        self.artifact = str(artifact or "")
        self.source = str(source)
        self._edits: List[CurationEdit] = []

    # -- the ledger ---------------------------------------------------------
    @property
    def edits(self) -> Tuple[CurationEdit, ...]:
        """Everything recorded, oldest first."""
        return tuple(self._edits)

    def append(self, kind: str, target: Any, *, n_changed: int = 0,
               **detail: Any) -> CurationEdit:
        """Record one correction and return it.

        :param kind: the verb. The curation classes write ``"paint"``,
            ``"undo"``, ``"join"``, ``"split"`` and ``"delete"``; nothing
            here restricts it, but it is the key :meth:`counts` groups by and
            the word :meth:`describe` prints, so a second spelling of an
            existing action reads as a second kind of edit.
        :param target: what was corrected, in that kind's own terms — a label
            for a paint, a track id for a split, a pair of ids for a join.
            Stored as handed over, and the writer falls back to ``str`` for
            anything :mod:`json` will not take: a numpy integer comes back
            out of the ledger as the string ``"7"`` and no longer matches the
            id it came from, which is why the track operations pass ids
            through :func:`_plain` first.
        :param n_changed: how much actually moved — voxels painted, rows
            re-assigned. Left at its default of 0 the entry cannot be told
            from an action that did nothing, which is most of what this
            number is for.
        :param detail: any further keys worth keeping with the entry: the
            brush radius, the frame a split happened at, the labels that were
            overwritten. They land in :attr:`CurationEdit.detail` verbatim
            and are serialised with the rest of the ledger, so the same
            ``str`` fallback applies to their values.
        """
        edit = CurationEdit(kind=str(kind), target=target, who=_user(),
                            n_changed=int(n_changed), detail=dict(detail))
        self._edits.append(edit)
        return edit

    def __len__(self) -> int:
        return len(self._edits)

    def counts(self) -> Dict[str, int]:
        """How many of each kind of edit."""
        out: Dict[str, int] = {}
        for edit in self._edits:
            out[edit.kind] = out.get(edit.kind, 0) + 1
        return out

    def describe(self) -> str:
        """The ledger in words — what a panel shows and a report quotes."""
        if not self._edits:
            return "No corrections: this data is as the pipeline produced it."
        parts = ", ".join(f"{n} {kind}" for kind, n
                          in sorted(self.counts().items()))
        people = sorted({e.who for e in self._edits if e.who})
        by = f" by {', '.join(people)}" if people else ""
        return (f"{len(self._edits)} correction(s){by} — {parts}. "
                f"First {self._edits[0].when}, last {self._edits[-1].when}. "
                f"This data has been curated by hand.")

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": 1, "artifact": self.artifact,
                "source": self.source,
                "edits": [edit.to_dict() for edit in self._edits]}

    def write(self, path: Any) -> str:
        """Write the ledger to ``path``, atomically. Returns the path.

        Atomic because the ledger is written after every action, including
        while a long session is running: a half-written JSON file left by a
        crash would make the whole history unreadable, and the history is the
        one thing that cannot be reconstructed from the data.

        :param path: the ledger file to write, extension and all — no suffix
            is added, so pass the full ``<artifact>.curation.json`` name or
            use :meth:`write_beside` to build it. Missing parent directories
            are created. The bytes go to a hidden ``.<name>.tmp`` sibling and
            are then renamed over the target, so the destination directory
            must allow creating a file and not merely overwriting one, and
            any ledger already at ``path`` is replaced whole rather than
            appended to.
        """
        target = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = os.path.join(parent, f".{os.path.basename(target)}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True,
                      default=str)
        os.replace(temporary, target)
        return target

    def write_beside(self, artifact: Any) -> str:
        """Write to ``<artifact>.curation.json``.

        :param artifact: the artefact to sit beside. It need not be the one
            this ledger names: :attr:`artifact` is set when the log is
            created and is *not* updated here, so a ledger written next to a
            copy still records which file the edits were actually made to.
        """
        return self.write(log_path_for(artifact))

    @classmethod
    def read(cls, path: Any) -> "CurationLog":
        """Read a ledger back. A missing file is an empty ledger.

        :param path: the ledger itself, not the artefact — use
            :meth:`read_beside` when you have the artefact's name. A path
            that does not exist gives an empty ledger, which is how a first
            session starts and why this is safe to call unguarded; a file
            that exists but is not JSON lets the decode error out rather than
            reporting the data as never edited.
        """
        target = os.fspath(path)
        if not os.path.isfile(target):
            return cls()
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        log = cls(data.get("artifact", ""),
                  source=data.get("source", "spacr"))
        log._edits = [CurationEdit.from_dict(entry)
                      for entry in data.get("edits") or []]
        return log

    @classmethod
    def read_beside(cls, artifact: Any) -> "CurationLog":
        """Read ``<artifact>.curation.json``.

        :param artifact: the artefact whose sidecar to open; the suffix is
            appended here. No sidecar means an empty ledger whose
            :attr:`artifact` is ``""`` rather than this name, so a session
            that intends to write should construct its own log with the
            artefact rather than editing the one this returns.
        """
        return cls.read(log_path_for(artifact))


def _user() -> str:
    """The operating-system user, or ``""``. Never raises."""
    try:
        return os.environ.get("USER") or os.environ.get("USERNAME") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Painting a mask
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelEdit:
    """Exactly which elements a stroke changed, and what they were.

    The undo record. Holding the *previous values* rather than a whole copy
    of the array is what makes an unbounded-looking history affordable: a
    brush stroke touches a few thousand voxels of a field that is tens of
    millions, so a hundred strokes cost less than one copy of the mask.

    :ivar index: one integer array per axis — what
        :meth:`spacr.layers.LabelsLayer.brush_index` returned, restricted to
        the elements that actually changed.
    :ivar before: the label each of those elements held.
    :ivar after: the label they were set to.
    :ivar radius: the brush radius this dab was laid with, in world units.
        Carried on the dab rather than read off the session when the stroke
        closes, because the session's radius is a mutable default and the
        ledger has to say what *happened*, not what the controls read
        afterwards.
    """

    index: Tuple[np.ndarray, ...]
    before: np.ndarray
    after: int
    radius: float = 0.0

    def __len__(self) -> int:
        return int(len(self.before))

    def revert(self, layer) -> int:
        """Put the previous labels back. Returns how many elements moved.

        Element by element rather than one assignment, because a stroke that
        crossed three objects has three previous labels and restoring "the"
        previous label would flatten them into one — which is a *new* editing
        mistake introduced by the undo.

        :param layer: the labels layer to write back into — the same layer
            the dab was taken from, still the same shape. :attr:`index` holds
            raw element indices, not world coordinates, so reverting against
            a re-loaded, re-cropped or differently oriented array silently
            restores the old labels in the wrong places instead of failing.
            Only ``set_labels_at`` is used, so the layer's subscribers hear
            one notification per distinct label restored, not one per
            element.
        """
        if not len(self.before):
            return 0
        moved = 0
        for value in np.unique(self.before):
            mask = self.before == value
            part = tuple(axis[mask] for axis in self.index)
            moved += layer.set_labels_at(part, int(value))
        return moved


class MaskCuration:
    """A brush over a :class:`spacr.layers.LabelsLayer`, with an undo history
    and a ledger.

    :param layer: the labels layer to edit.
    :param artifact: what the mask is stored as, for the ledger. Defaults to
        the layer's name.
    :param history: how many strokes :meth:`undo` can walk back. Bounded, so a
        long session cannot grow without limit; the *ledger* is unbounded and
        is what a reviewer reads.

    Strokes, not points. A drag is dozens of :meth:`paint` calls and one thing
    the user did, so :meth:`begin_stroke` / :meth:`end_stroke` group them and
    undo takes back the whole stroke. Painting without opening a stroke is
    still legal — one dab is one stroke — because a click is a legitimate
    edit and should not need ceremony.
    """

    def __init__(self, layer, *, artifact: Any = "", history: int = 64,
                 log: Optional[CurationLog] = None):
        self.layer = layer
        self.artifact = str(artifact or getattr(layer, "name", "") or "mask")
        self.history = max(1, int(history))
        self.log = log if log is not None else CurationLog(
            self.artifact, source="spacr-qt curation")
        self._strokes: List[List[LabelEdit]] = []
        self._open: Optional[List[LabelEdit]] = None
        # Views that want to redraw when the LEDGER moves. The layer's own
        # subscribers fire per dab, which is mid-stroke -- a panel listening
        # only to those redraws before end_stroke has recorded anything, and
        # so never shows the entry it is there to show. Bound methods only:
        # a session outlives nothing here, but a lambda would keep a closed
        # panel alive as a receiver.
        self._listeners: List[Any] = []
        #: The label the brush paints. 0 erases, which is what "delete this
        #: bit of the mask" means to a labels layer.
        self.label = 1
        #: Brush radius in WORLD units — µm on a calibrated stack, pixels on
        #: an uncalibrated one. World, not elements, so the same brush covers
        #: the same physical distance on an anisotropic stack.
        self.radius = 3.0

    # -- who is watching ------------------------------------------------------
    def subscribe(self, fn) -> None:
        """Call ``fn(edit)`` whenever a correction is recorded.

        The seam a panel refreshes off. Distinct from subscribing to the
        *layer*, which fires once per dab: a stroke is many dabs and one
        ledger entry, and a view that wants to show the entry has to hear
        about the entry.

        :param fn: called as ``fn(edit)`` with the :class:`CurationEdit` just
            appended, synchronously, on whichever thread made the edit. Pass
            a bound method: registration is de-duplicated by equality, and a
            bound method looked up fresh compares equal to the one already
            held, so ``subscribe`` twice is a no-op and :meth:`unsubscribe`
            works — where each fresh lambda is a new object that stacks up
            and cannot be removed, besides keeping a closed panel alive as a
            receiver. An exception raised inside ``fn`` is swallowed: the
            correction has already happened to the data, and one view's
            failed redraw must not be reported as a failed edit.
        """
        if fn not in self._listeners:
            self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        """Stop listening. Safe for something that never subscribed.

        :param fn: matched by equality against what :meth:`subscribe` was
            given, so the usual teardown — handing back the same bound method
            — removes it, while a lambda can only be removed by passing the
            very object that was subscribed. Anything not currently
            registered is ignored rather than raising, so a panel's close
            handler need not know whether it ever connected.
        """
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _record(self, kind: str, target: Any, *, n_changed: int = 0,
                **detail: Any) -> CurationEdit:
        """Append to the ledger and tell whoever is watching."""
        edit = self.log.append(kind, target, n_changed=n_changed, **detail)
        for listener in list(self._listeners):
            try:
                listener(edit)
            except Exception:
                # One view's redraw must not take the correction with it --
                # the edit has already happened to the data.
                pass
        return edit

    # -- strokes ------------------------------------------------------------
    def begin_stroke(self) -> None:
        """Start grouping paints into one undoable action."""
        if self._open is None:
            self._open = []

    @staticmethod
    def _summarise(edits: Sequence[LabelEdit]) -> Tuple[Any, Any]:
        """``(what was painted, at what radius)`` — read off the dabs.

        Derived, never typed. Recording ``self.label`` here instead said
        whatever the controls happened to hold when the stroke closed, which
        for a *provenance* record is the worst kind of wrong: it is a
        confident, plausible, false statement about what was done. A scalar
        when the stroke was uniform, a sorted list when it was not.
        """
        labels = sorted({int(edit.after) for edit in edits})
        radii = sorted({float(edit.radius) for edit in edits})
        return (labels[0] if len(labels) == 1 else labels,
                radii[0] if len(radii) == 1 else radii)

    def end_stroke(self) -> Optional[CurationEdit]:
        """Close the stroke and record it. ``None`` if nothing changed.

        A stroke that changed nothing — the user pressed and released without
        moving, over pixels that already held the brush label — is not
        recorded. A ledger padded with no-op entries is one nobody reads.
        """
        edits = self._open or []
        self._open = None
        if not edits:
            return None
        changed = sum(len(edit) for edit in edits)
        if not changed:
            return None
        self._strokes.append(edits)
        while len(self._strokes) > self.history:
            self._strokes.pop(0)
        painted, radius = self._summarise(edits)
        return self._record(
            "paint", painted, n_changed=changed,
            radius=radius, dabs=len(edits),
            replaced=sorted({int(v) for edit in edits
                             for v in np.unique(edit.before)}))

    # -- painting -----------------------------------------------------------
    def paint(self, world: Mapping[str, float],
              label: Optional[int] = None,
              radius: Optional[float] = None) -> int:
        """Paint one dab and remember exactly what it changed.

        :param world: the brush centre as ``{axis: coordinate}`` in WORLD
            units, keyed by the layer's axis names. Axes the mapping leaves
            out are taken as 0, which is what a 2-D click on a 3-D stack
            means once the viewer has filled in the slice it is showing. A
            centre that puts the whole ball off the grid paints nothing and
            returns 0 rather than raising.
        :param label: the value to write; ``None`` means :attr:`label`. 0 is
            background, so painting 0 is an erase — see :meth:`erase`.
            Elements that already hold this value are not counted, not
            recorded, and cannot be undone, because nothing happened to them.
        :param radius: brush radius in world units — µm on a calibrated
            stack, pixels on an uncalibrated one; ``None`` means
            :attr:`radius`. The brush is a ball in world space, so on an
            anisotropic stack it reaches fewer z-slices than y-rows. 0 is not
            a one-element brush: it covers only an element whose centre the
            point lands on exactly, so a click at a fractional coordinate
            changes nothing.
        :returns: how many elements changed.
        """
        label = int(self.label if label is None else label)
        radius = float(self.radius if radius is None else radius)
        index = self.layer.brush_index(world, radius=radius)
        if not index or not len(index[0]):
            return 0
        before = np.asarray(self.layer.data[index]).copy()
        moved = before != label
        if not moved.any():
            return 0
        index = tuple(axis[moved] for axis in index)
        before = before[moved]
        changed = self.layer.set_labels_at(index, label)
        edit = LabelEdit(index=index, before=before, after=label,
                         radius=radius)
        if self._open is not None:
            self._open.append(edit)
        else:
            # A bare dab is its own stroke, so it is undoable and recorded
            # like any other.
            self._open = [edit]
            self.end_stroke()
        return changed

    def erase(self, world: Mapping[str, float],
              radius: Optional[float] = None) -> int:
        """Paint background. The same act; named for what it is.

        :param world: the brush centre in world units, exactly as for
            :meth:`paint`.
        :param radius: world-space radius; ``None`` means :attr:`radius`, the
            same default the brush paints with — the eraser has no size of
            its own, so widening the brush widens this too.
        :returns: how many elements changed.
        """
        return self.paint(world, label=0, radius=radius)

    # -- undo ---------------------------------------------------------------
    def undo(self) -> Optional[CurationEdit]:
        """Take back the last stroke. ``None`` when there is nothing to undo.

        Appends an ``undo`` entry rather than removing the ``paint`` one. That
        something was painted and then taken back is part of what happened,
        and a ledger that can be quietly tidied is not evidence of anything.
        """
        if not self._strokes:
            return None
        stroke = self._strokes.pop()
        moved = 0
        # Newest dab first: two dabs that overlapped must be reverted in the
        # reverse of the order they were laid down, or the older one's
        # "before" values overwrite the newer one's.
        for edit in reversed(stroke):
            moved += edit.revert(self.layer)
        painted, _radius = self._summarise(stroke)
        return self._record(
            "undo", painted, n_changed=moved, dabs=len(stroke),
            restored=sorted({int(v) for edit in stroke
                             for v in np.unique(edit.before)}))

    @property
    def can_undo(self) -> bool:
        return bool(self._strokes)

    def __len__(self) -> int:
        """How many strokes are in the undo history."""
        return len(self._strokes)

    # -- persistence --------------------------------------------------------
    def save_log(self, artifact: Optional[Any] = None) -> str:
        """Write the ledger beside the artefact. Returns the path.

        :param artifact: what to write beside; the ledger goes to
            ``<artifact>.curation.json``. Anything falsy — including the
            default ``None`` — means :attr:`artifact`, which when the session
            was built without one is only the layer's *name*, so the ledger
            lands in the process's working directory rather than next to the
            image. Pass the mask's real path here, or at construction, if
            that is not what you want. Writing to a different place does not
            change the artefact name recorded inside the ledger.
        """
        return self.log.write_beside(artifact or self.artifact)


# ---------------------------------------------------------------------------
# Curating tracks
# ---------------------------------------------------------------------------

class TrackCuration:
    """Join, split and delete tracks by hand, on the record.

    :param tracks: a track table — :data:`spacr.zstack.BASE_TRACK_COLUMNS`,
        i.e. ``frame``, ``track_id``, ``original_label`` and the centroid.
        Copied, so the caller's frame is never edited underneath them.
    :param artifact: the tracks CSV, for the ledger.

    Every operation leaves the table *consistent*, and consistency here has a
    definition worth stating because it is what the checks enforce:

    * one row per ``(track_id, frame)`` — a track is one object's path, so a
      track that is in two places at one time is not a track;
    * every track's frames are the frames it actually has, and a join may not
      produce a track that overlaps itself in time.

    :meth:`check` returns the violations rather than raising, so a table that
    arrived broken can be *shown* to be broken instead of making every
    operation on it fail with the same message.
    """

    def __init__(self, tracks: pd.DataFrame, *, artifact: Any = "",
                 log: Optional[CurationLog] = None):
        columns = _track_columns()
        missing = [c for c in ("frame", "track_id") if c not in tracks.columns]
        if missing:
            raise CurationError(
                f"a track table needs {missing}; this one has "
                f"{sorted(tracks.columns)[:8]}... The canonical columns are "
                f"{list(columns)}.")
        self.tracks = tracks.copy().reset_index(drop=True)
        self.artifact = str(artifact or "tracks")
        self.log = log if log is not None else CurationLog(
            self.artifact, source="spacr-qt curation")

    # -- reading ------------------------------------------------------------
    @property
    def track_ids(self) -> List[Any]:
        """Every track id present, sorted."""
        return sorted(self.tracks["track_id"].unique().tolist())

    def frames_of(self, track_id: Any) -> List[Any]:
        """The frames ``track_id`` appears in, sorted.

        :param track_id: matched with ``==`` against the ``track_id`` column,
            so it has to be the same kind of value the table holds — 3 finds
            nothing in a table of strings. An id that is not there gives an
            empty list rather than raising: this is a reader, and the
            operations do their own existence check. The result is sorted on
            the frame values themselves, so a frame column of strings sorts
            lexicographically and ``"10"`` lands before ``"2"``.
        """
        rows = self.tracks[self.tracks["track_id"] == track_id]
        return sorted(rows["frame"].unique().tolist())

    def span(self, track_id: Any) -> Optional[Tuple[Any, Any]]:
        """``(first frame, last frame)`` of a track, or ``None`` if absent.

        :param track_id: the track to measure, matched as in
            :meth:`frames_of`. Unknown gives ``None``, which is how to ask
            "is this track here at all" in one call. Both ends are inclusive,
            so a track living in one frame answers with that frame twice
            rather than an empty or half-open range, and a track with gaps
            answers with its outer bounds — the span is not the frame count.
        """
        frames = self.frames_of(track_id)
        return (frames[0], frames[-1]) if frames else None

    def to_frame(self) -> pd.DataFrame:
        """The curated table, sorted by track then frame."""
        return self.tracks.sort_values(
            ["track_id", "frame"], kind="stable").reset_index(drop=True)

    def _next_id(self) -> int:
        """A track id nothing is using. Integer, and above every existing one.

        Above rather than in a gap: a split whose new track reused the id of a
        track deleted ten minutes ago makes the ledger ambiguous, and the
        ledger is the point.
        """
        numeric = pd.to_numeric(self.tracks["track_id"], errors="coerce")
        top = numeric.max()
        return int(top) + 1 if pd.notna(top) else 1

    # -- consistency ---------------------------------------------------------
    def check(self) -> List[str]:
        """Everything wrong with the table right now, as sentences.

        Empty for a consistent table. Returned rather than raised so a broken
        table can be shown to the user; the *operations* raise, because an
        operation that would create a violation must not happen.
        """
        problems: List[str] = []
        duplicated = self.tracks.duplicated(["track_id", "frame"], keep=False)
        if duplicated.any():
            offenders = self.tracks.loc[duplicated, ["track_id", "frame"]]
            pairs = sorted({(str(t), str(f)) for t, f
                            in zip(offenders["track_id"], offenders["frame"])})
            problems.append(
                f"{len(pairs)} (track, frame) pair(s) appear more than once — "
                f"a track cannot be in two places at one time: "
                f"{pairs[:5]}")
        return problems

    def _require_track(self, track_id: Any) -> None:
        if track_id not in set(self.tracks["track_id"]):
            raise CurationError(
                f"no track {track_id!r} in this table; have "
                f"{self.track_ids[:8]}...")

    # -- operations ----------------------------------------------------------
    def join(self, first: Any, second: Any) -> CurationEdit:
        """Make ``second`` a continuation of ``first``.

        The commonest correction there is: a track breaks when a cell briefly
        leaves focus, and the same cell comes back with a new id, so one cell
        becomes two half-length tracks and every velocity computed from them
        is wrong at the join.

        Refused when the two overlap in time. Two tracks present in the same
        frame are two objects, and joining them would put one track in two
        places at once — which is exactly the state :meth:`check` exists to
        forbid, and producing it silently would corrupt the table on the way
        to fixing it.

        :param first: the track that survives. Its id is what every joined
            row ends up carrying and what downstream analysis will see, so
            pass the one you want to keep — usually the earlier half, though
            nothing here requires it.
        :param second: the track absorbed. Its rows are re-labelled in place,
            never moved or re-timed, and its id then no longer exists in the
            table. Nothing checks that the two halves are adjacent or even
            near each other in time — only that they do not overlap — so this
            will happily join tracks fifty frames apart if you ask it to.
        :returns: the ledger entry.
        :raises CurationError: on an unknown track, joining a track to
            itself, or a time overlap.
        """
        self._require_track(first)
        self._require_track(second)
        if first == second:
            raise CurationError(
                f"cannot join track {first!r} to itself")
        shared = set(self.frames_of(first)) & set(self.frames_of(second))
        if shared:
            raise CurationError(
                f"tracks {first!r} and {second!r} are both present in "
                f"frame(s) {sorted(shared)[:5]}, so they are two objects at "
                f"that time and cannot be one track. Split one of them first "
                f"if the overlap is itself the error.")
        mask = self.tracks["track_id"] == second
        moved = int(mask.sum())
        self.tracks.loc[mask, "track_id"] = first
        return self.log.append(
            "join", [_plain(first), _plain(second)], n_changed=moved,
            kept=_plain(first), absorbed=_plain(second),
            frames=[_plain(f) for f in self.frames_of(first)])

    def split(self, track_id: Any, at_frame: Any) -> CurationEdit:
        """Break ``track_id`` in two: ``at_frame`` starts the new track.

        The other half of the commonest pair of errors: two cells touch, the
        tracker swaps them, and one id follows cell A then cell B. Splitting
        at the frame where it changed hands turns one wrong track into two
        right ones.

        :param track_id: the track to break. It keeps the frames before
            ``at_frame`` and keeps its id, so references to the head stay
            valid; the tail is what gets renamed.
        :param at_frame: the first frame of the NEW track — rows at this
            frame and after are re-assigned, rows before it are left alone.
            Compared with ``<`` and ``>=``, so it need not be a frame the
            track actually has; any value with rows on both sides of it
            works, and one that would leave a side empty is refused rather
            than quietly doing nothing.
        :returns: the ledger entry, whose ``detail['new_track']`` is the id
            the tail was given.
        :raises CurationError: for an unknown track, or a frame that would
            leave one side empty — a "split" that moves nothing is a no-op the
            user will read as having worked.
        """
        self._require_track(track_id)
        frames = self.frames_of(track_id)
        head = [f for f in frames if f < at_frame]
        tail = [f for f in frames if f >= at_frame]
        if not head or not tail:
            raise CurationError(
                f"splitting track {track_id!r} at frame {at_frame!r} would "
                f"leave one side empty (its frames are {frames[:8]}...). "
                f"Split at a frame inside the track.")
        new_id = self._next_id()
        mask = ((self.tracks["track_id"] == track_id)
                & (self.tracks["frame"] >= at_frame))
        moved = int(mask.sum())
        self.tracks.loc[mask, "track_id"] = new_id
        return self.log.append(
            "split", _plain(track_id), n_changed=moved, at_frame=_plain(at_frame),
            new_track=new_id, head_frames=[_plain(f) for f in head],
            tail_frames=[_plain(f) for f in tail])

    def delete(self, track_id: Any) -> CurationEdit:
        """Remove a track entirely — debris, or a tracker artefact.

        The rows go, but the ledger keeps what went: how many rows and which
        frames, so a count that changed between two analyses can be explained
        rather than argued about.

        :param track_id: the track to drop. Every row carrying it goes, in
            every frame — there is no partial delete, so :meth:`split` the
            track first if only one end of it is debris. An id that is not in
            the table raises :class:`CurationError` instead of removing
            nothing and recording a delete that did not happen.
        """
        self._require_track(track_id)
        frames = self.frames_of(track_id)
        mask = self.tracks["track_id"] == track_id
        removed = int(mask.sum())
        self.tracks = self.tracks.loc[~mask].reset_index(drop=True)
        return self.log.append(
            "delete", _plain(track_id), n_changed=removed,
            frames=[_plain(f) for f in frames])

    # -- persistence ---------------------------------------------------------
    def save(self, path: Any) -> str:
        """Write the curated table AND its ledger. Returns the CSV path.

        One call, deliberately. A curated table written without its ledger is
        exactly the reproducibility hole this module exists to close, and
        leaving the second write to the caller is how that happens.

        :param path: where the CSV goes; missing parent directories are
            created. Two files are written, not one — the ledger lands at
            ``<path>.curation.json`` beside it, so copying the CSV onward
            without that sidecar drops the record of every correction. The
            artefact name inside :attr:`log` is repointed at this path first,
            so the saved ledger names the file it was saved with rather than
            whatever the session was opened on. The table is written sorted
            by track then frame, and any file already at ``path`` is
            overwritten.
        """
        target = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.to_frame().to_csv(target, index=False)
        self.log.artifact = target
        self.log.write_beside(target)
        return target

    def describe(self) -> str:
        """The table and its history, in words."""
        problems = self.check()
        state = ("consistent" if not problems
                 else f"INCONSISTENT — {problems[0]}")
        return (f"{len(self.track_ids)} track(s) over {len(self.tracks)} "
                f"row(s); {state}. {self.log.describe()}")


def _plain(value: Any) -> Any:
    """A numpy scalar as something ``json.dump`` will accept."""
    if isinstance(value, np.generic):
        return value.item()
    return value
