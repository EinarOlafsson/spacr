"""One contract for every live view, written down once.

Four modules ship a live view — Mask
(:mod:`~spacr.qt.widgets.live_preview`), Timelapse
(:mod:`~spacr.qt.widgets.timelapse_preview`), Measure
(:mod:`~spacr.qt.widgets.measure_preview`) and Motility
(:mod:`~spacr.qt.widgets.motility_preview`) — and each of them grew its own
answer to the same handful of questions. The inventory taken before this
module was written, one column per live view:

============================  ===========  ===========  ===========  ===========
question                      Mask         Timelapse    Measure      Motility
============================  ===========  ===========  ===========  ===========
what the run button says      Run preview  Run preview  Refresh      Run preview
                                                        crops
public entry point            run_preview  run_preview  refresh      run_preview
can the user cancel it        method only  no           no           no
does it say why it cannot     yes          yes          NO — silent  yes
what "busy" reads as          "Preview     "Preview     n/a          "Preview
                              already      already                   already
                              running."    running."                 running."
worker freed on ``finished``  yes          no —         n/a          no —
                                           deleteLater               deleteLater
announces its result          preview_     preview_     NO — none    preview_
                              ready        ready                     ready
work runs off the GUI thread  QThread      QThread      JobRunner    QThread
what a knob change costs      nothing —    a re-link,   a re-crop,   a rescore
                              explicit     cached       superseded   from the
                              run          masks        by token     cache
where the picture goes        twin zoom    twin zoom    crop grid    matplotlib
                              canvases     canvases                  plot
============================  ===========  ===========  ===========  ===========

The last three rows are differences of substance — a crop grid is not a
pair of zoom canvases, and a preview that re-links from a cache should not
be made to demand a button press for the sake of symmetry. They stay. The
rows above them are the ones that had no reason to differ, and every one of
them is now a single implementation.

Three of those rows were bugs rather than differences of taste. Measure
returned from its re-crop without a word when no array was loaded, so the
button did nothing and said nothing. Timelapse and Motility wired
``QThread.finished`` to ``worker.deleteLater`` and dropped their own
reference in the result slot — the pattern
:func:`spacr.qt.bridge.make_thread` documents as a race, and the one the
Mask panel had already been fixed away from — which leaves a running
``QThread`` owned by nobody when the user closes the screen mid-pass. And
Measure never told anyone its pass had landed, so no screen could react to
a crop preview the way it reacts to the other three.

:class:`LivePreviewContract` is what the four panels now inherit. It owns
the vocabulary (:data:`PREVIEW_RUN_TEXT` and the message constants), the
run guard, the cancellation token and the busy/idle button state, so a
change to any of them is one change rather than four.

The words
---------
A **live view** is a panel that re-renders the module's own output from the
current settings, on real input, before a run. That is Mask, Timelapse,
Measure and Motility, and it is what the **Live** toggle opens. The Image
UMAP explorer is not one: it makes an already-computed embedding
clickable, and no setting changes what it draws. Its toggle therefore says
**Interactive**, not Live — the same word had been on both.

**Cancel** and **Stop** are also two words on purpose. A live view's
Cancel throws the pass away: nothing it produced is kept, because a half
-segmented field is not a result. A hyperparameter search's Stop
(:class:`spacr.qt.screens.hyperparam.HyperparamPanel`) finishes the trial
in flight and keeps every trial already scored, marking the result
partial. Same button position, deliberately different verbs, because a
user who presses one expecting the other loses either nothing or hours.

The search panel's progress is already reported the way this contract
wants a live view's to be — ``on_trial`` is called with ``idx + 1`` *after*
a trial has been scored, so "12 of 40 configurations evaluated" counts
work done rather than work started.

Cellpose
--------
:func:`preview_cellpose_model` is the other half of the sharing.
``model_type=`` is accepted and IGNORED by Cellpose 4, so a preview that
passed it silently segmented with ``cpsam`` — including for a user who had
just trained their own checkpoint in spaCR's Train Cellpose module. That
defect was written twice, in two preview files, and fixed twice. One
constructor here means the next Cellpose API change is one fix.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "PREVIEW_RUN_TEXT",
    "PREVIEW_CANCEL_TEXT",
    "PREVIEW_BUSY_MESSAGE",
    "PREVIEW_CANCELLED_MESSAGE",
    "PREVIEW_RUNNING_MESSAGE",
    "LivePreviewContract",
    "PRIMARY_NOTES",
    "preview_cellpose_model",
    "preview_failure_message",
]

#: What each non-RGB display mode does to the channels, in one sentence.
#:
#: Shown on the panel whenever a mode is active. An image recoloured in
#: silence is one a reader takes for the raw colours -- and, if they export
#: it, publishes as them. Stating the mapping costs a clause; not stating it
#: costs a figure legend that is wrong.
PRIMARY_NOTES = {
    "cmy": "Channels drawn as cyan / magenta / yellow (publication style, "
           "not a colour-blind mode).",
    "deuteranope": "Red channel drawn as yellow, for deuteranopia.",
    "protanope": "Red channel drawn as yellow, for protanopia.",
    "tritanope": "Blue channel drawn as magenta, for tritanopia.",
}

#: The primary action on every live view. One noun for one thing.
PREVIEW_RUN_TEXT = "Run preview"

#: The control that abandons the pass in flight, beside the run button.
PREVIEW_CANCEL_TEXT = "Cancel"

#: Refusal when a pass is already in flight. Every panel says this.
PREVIEW_BUSY_MESSAGE = "Preview already running."

#: Acknowledgement of :meth:`LivePreviewContract.cancel_preview`.
PREVIEW_CANCELLED_MESSAGE = "Preview cancelled."

#: What a panel says between pressing run and the first result.
PREVIEW_RUNNING_MESSAGE = "Running preview…"


def preview_failure_message(error: Any) -> str:
    """Phrase one failed preview pass the same way in every module.

    :param error: the exception, or the error string a worker emitted.
    :returns: the sentence for the panel's status line.
    """
    return f"Preview failed: {error}"


def preview_cellpose_model(model_name: Any, gpu: Optional[bool] = None):
    """Build the Cellpose model a live view segments with.

    ``model_type=`` is accepted-and-IGNORED by Cellpose 4 — it logs "not
    used in v4.0.1+" and drops it, leaving ``pretrained_model`` at its
    ``cpsam`` default. Passing the user's choice there therefore segmented
    with ``cpsam`` whatever they picked, including a checkpoint they had
    just trained in spaCR's own Train Cellpose module.
    :func:`spacr.utils._resolve_cellpose_pretrained` is what the pipeline
    uses: it maps the legacy pre-SAM names onto ``cpsam`` and says so once,
    and returns a path unchanged when the name is a fine-tuned checkpoint.

    Cellpose and torch are imported inside the call, so importing a preview
    module cold — as the test suite does — needs no CUDA-capable stack.

    :param model_name: the model name or checkpoint path the user picked.
    :param gpu: force the device choice; ``None`` asks torch.
    :returns: a ``cellpose.models.CellposeModel``.
    """
    from cellpose import models as cp_models
    from spacr.utils import _resolve_cellpose_pretrained

    if gpu is None:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    return cp_models.CellposeModel(
        gpu=bool(gpu),
        pretrained_model=_resolve_cellpose_pretrained(str(model_name)),
        device=None)


class LivePreviewContract:
    """What every live-view panel promises, implemented once.

    A panel mixes this in beside ``QWidget`` and supplies three things: a
    ``_status`` label, a ``_run_btn`` button, and
    :meth:`_preview_blocked_reason`. It gets the run guard, the cancel
    token, the busy/idle button state and the shared wording for free.

    The token is the cancellation mechanism. Neither Cellpose nor a numpy
    read exposes an interrupt, so a cancelled pass is left to run itself
    out and its answer is dropped on arrival: :meth:`preview_token` is
    captured when a pass starts and compared with :meth:`preview_stale`
    when it lands.
    """

    #: Sentence used when nothing is loaded. Panels override it.
    PREVIEW_SOURCE_HINT = "Load an image first."

    # -- what the panel supplies ------------------------------------------

    def _preview_blocked_reason(self) -> str:
        """Hook: why a preview cannot run right now, or ``""``.

        The default answers for a panel that has not overridden it, which
        is never the right answer for long — a live view that cannot say
        why it is idle looks broken.
        """
        return ""

    # -- what every panel gets --------------------------------------------

    def preview_blocked_reason(self) -> str:
        """Say why the preview cannot run, or ``""`` when it can.

        Never raises: a panel that fails while explaining itself would
        take the status line down with it.
        """
        try:
            return str(self._preview_blocked_reason() or "")
        except Exception:
            return ""

    def can_preview(self) -> bool:
        """True when pressing run would start a pass."""
        return not self.preview_blocked_reason()

    def preview_running(self) -> bool:
        """True while a preview pass is in flight."""
        worker = getattr(self, "_worker", None)
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            # The C++ side is already gone; nothing is in flight.
            return False

    def preview_status(self) -> str:
        """The panel's current status line."""
        label = getattr(self, "_status", None)
        return "" if label is None else str(label.text())

    def display_primaries(self) -> str:
        """Which primaries this view draws channels in.

        Read from the GLOBAL preference, never from a control on one panel.
        A user who needs the substitution needs it in every view and every
        session; a per-screen toggle is one they have to re-find, and the
        screen they forget to set is the one that misleads them.

        A view MAY override this -- a figure being prepared for publication
        wants ``cmy`` whatever the author's vision -- but every view starts
        here.

        :returns: one of :data:`spacr.crops.DISPLAY_PRIMARIES`.
        """
        try:
            from ..preferences import image_display_primaries
            return image_display_primaries()
        except Exception:
            # No QSettings, no Qt: the untransformed image is the honest
            # answer, and never worse than failing to draw one.
            return "rgb"

    def display_primaries_note(self) -> str:
        """One sentence naming the mapping, or ``""`` for plain RGB."""
        mode = self.display_primaries()
        if mode == "rgb":
            return ""
        return PRIMARY_NOTES.get(mode, f"Channels drawn in {mode} primaries.")

    def set_preview_status(self, text: Any) -> None:
        """Put one sentence on the panel's status line.

        The display-primaries note is appended here rather than at each
        call site, so no panel and no code path can recolour an image
        without saying so. When the preference is off -- which it is for
        almost everybody -- this changes nothing at all.
        """
        label = getattr(self, "_status", None)
        if label is None:
            return
        sentence = str(text)
        note = self.display_primaries_note()
        if note and note not in sentence:
            sentence = f"{sentence}  ·  {note}" if sentence else note
        label.setText(sentence)

    def preview_token(self) -> int:
        """The generation of the pass now current."""
        return int(getattr(self, "_run_token", 0))

    def preview_stale(self, token: Any) -> bool:
        """True when a result carrying ``token`` has been superseded."""
        if token is None:
            return False
        try:
            return int(token) != self.preview_token()
        except (TypeError, ValueError):
            return False

    def cancel_preview(self) -> bool:
        """Abandon the pass in flight, if there is one.

        The work is not killed — neither Cellpose nor a numpy read can be
        interrupted — the token is bumped so its answer lands as a no-op,
        and the panel is returned to the idle state at once.

        :returns: True when a running pass was abandoned.
        """
        self._run_token = self.preview_token() + 1
        running = self.preview_running() or self._extra_work_in_flight()
        self._cancel_extra_work()
        if running:
            self.set_preview_status(PREVIEW_CANCELLED_MESSAGE)
        self.set_preview_busy(False)
        return bool(running)

    def set_preview_busy(self, busy: bool) -> None:
        """Enable exactly one of run / cancel."""
        run = getattr(self, "_run_btn", None)
        if run is not None:
            run.setEnabled(not busy)
        cancel = getattr(self, "_cancel_btn", None)
        if cancel is not None:
            cancel.setEnabled(bool(busy))

    def begin_preview(self) -> bool:
        """The shared guard at the top of every ``run_preview``.

        Refuses out loud — a live view that declines in silence is the
        defect this contract exists to remove — and otherwise marks the
        panel busy and hands back a fresh token via :meth:`preview_token`.

        :returns: True when the caller should start a pass.
        """
        reason = self.preview_blocked_reason()
        if reason:
            self.set_preview_status(reason)
            return False
        if self.preview_running():
            self.set_preview_status(PREVIEW_BUSY_MESSAGE)
            return False
        self.set_preview_busy(True)
        return True

    # -- optional: panels whose work is not a QThread ----------------------

    def _extra_work_in_flight(self) -> bool:
        """Hook for a panel whose pass runs somewhere other than ``_worker``."""
        return False

    def _cancel_extra_work(self) -> None:
        """Hook: drop whatever :meth:`_extra_work_in_flight` reported."""
        return None
