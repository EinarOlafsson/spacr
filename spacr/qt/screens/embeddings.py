"""Embeddings — a label-free vector for every object, beside the measured panel.

A SCREEN, NOT A SCRIPT. Cell-DINO, OpenPhenom and SubCell are all Python or
command line. No graphical platform exposes any of them.

The engine serves this lab. The screen offers that embedding without any
Python.

All the arithmetic is in :mod:`spacr.embeddings` and none is here. This module
is the surface, and three of the engine's decisions shape it:

**THE CHANNEL POLICY IS THE FIRST CONTROL, NOT A BURIED OPTION.** 386 calls
the channel problem "the design problem" and warns against letting it be
decided implicitly by whatever the backbone wants. Per-channel encoding keeps
the channel in every column name and costs N forward passes; projection mixes
the channels into three and costs one. Those produce columns of identical
width and dtype that mean entirely different things, so the screen makes the
choice explicit and says what each one costs.

**A DIMENSION IS NOT A PHENOTYPE, and the screen says so where a user will
read it.** The table shows the first few dimensions because a user needs to
see that something happened; the caption says they are meaningless
individually. Presenting `emb_c2_017 = 0.43` without that sentence invites
exactly the reading the family cannot survive.

**THE ENCODER IS A MODEL, so it is named with its checksum.** The run summary
carries the model-zoo entry from :func:`spacr.embeddings.encoder_entry` --
backbone, policy and the digest of the weights on this machine -- because two
runs' dimension 17 are the same number and not the same thing unless all
three match.

The embedding runs off the GUI thread through
:class:`spacr.qt.job_runner.JobRunner`, like every other compute in the Qt
layer: a plate of crops through a backbone is minutes, and ``threaded=False``
runs the identical code inline so a test drives the shipped path.

:func:`register` is **not** called at import; read its docstring.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QTableWidget,
    QVBoxLayout, QWidget,
)

from ..app_catalog import declared_app
from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.sortable_table import install_sorting, table_item
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.embeddings")

__all__ = ["EmbeddingsScreen", "make_embeddings_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "PREVIEW_DIMENSIONS",
           "DIMENSION_CAVEAT"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "embeddings"

#: How many dimensions the preview table shows. Enough to see that the run
#: produced numbers, few enough that nobody mistakes the table for the result
#: -- the result is the whole vector and it is not readable.
PREVIEW_DIMENSIONS = 8

#: Shown under the table, every time. 386's own warning, in the one place a
#: user reading a dimension will actually be looking.
DIMENSION_CAVEAT = (
    "A single dimension is not a phenotype and means nothing on its own. "
    "These are only interpretable as a vector — through a reduction, a "
    "retrieval, or an attribution."
)


class EmbeddingsScreen(QWidget):
    """Pick a crop source and a channel policy, and embed every object.

    :param parent: the usual Qt parent.
    :param threaded: ``False`` runs the embedding inline, emitting the same
        signals in the same order, so a test drives the screen synchronously
        without the behaviour diverging.
    """

    def __init__(self, parent=None, *, threaded: bool = True):
        """Build the screen: the controls above, the preview table below."""
        super().__init__(parent)
        self.setObjectName("EmbeddingsScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._result = None
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setSpacing(SPACING["sm"])
        self._header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Load crops, choose how the channels are encoded, "
                        "then embed.",
        )
        head.addWidget(self._header)
        self._source = QLabel("no crops loaded", self)
        self._source.setObjectName("EmbeddingsSourceLabel")
        head.addWidget(self._source, 1)
        outer.addLayout(head)

        controls = QHBoxLayout()
        controls.setSpacing(SPACING["sm"])

        controls.addWidget(QLabel("Channels:", self))
        self._policy = QComboBox(self)
        self._policy.setObjectName("EmbeddingsPolicyPicker")
        # THE COST IS IN THE CAPTION, not only the tooltip. 386 asks for this
        # choice to be explicit; a user who cannot see what it costs will
        # pick whichever is first and never revisit it.
        # SHORT ENOUGH TO TRANSLATE. These are runtime catalog rows, and the
        # machine translator returns long clause-heavy English unchanged --
        # the earlier five-line tooltip failed the zh_CN gate outright. One
        # idea per sentence, which a tooltip wants anyway.
        # The PASS COUNT stays in the caption: 386 asks for the cost to be
        # visible where the choice is made, and a test pins it.
        self._policy.addItem("Per channel (one pass per stain)", "per_channel")
        self._policy.addItem("Project to three (one pass)", "project")
        # VERIFIED AGAINST THE zh_CN MODEL BEFORE BEING WRITTEN. The M2M
        # checkpoint returns a string UNCHANGED -- not an error -- when
        # "channel" and "stain" appear in the same row, and the catalog audit
        # then reports it as "remains exact English". Six variants were run
        # through `_translate_batches` to find that; this wording avoids the
        # pair and comes back as Chinese. See instruction 394.
        self._policy.setToolTip(
            "Encoding runs separately for every stain and names it in the "
            "column. Projection mixes them into three and is faster.")
        controls.addWidget(self._policy)

        controls.addWidget(QLabel("Backbone:", self))
        self._backbone = QComboBox(self)
        self._backbone.setObjectName("EmbeddingsBackbonePicker")
        self._backbone.setEditable(True)
        self._backbone.setToolTip(
            "Any encoder timm can resolve. The model-zoo entry records which "
            "one, with the checksum of its weights: two runs' dimension 17 "
            "are the same number and not the same thing unless the backbone, "
            "the weights and the channel policy all match.")
        controls.addWidget(self._backbone)

        controls.addWidget(QLabel("Batch:", self))
        self._batch = QSpinBox(self)
        self._batch.setObjectName("EmbeddingsBatchSize")
        self._batch.setRange(1, 4096)
        self._batch.setValue(64)
        self._batch.setToolTip(
            "Crops per forward pass. Lower it if the card runs out of "
            "memory; it changes speed and nothing else.")
        controls.addWidget(self._batch)

        controls.addStretch(1)
        self._run = QPushButton("Embed", self)
        self._run.setObjectName("PrimaryButton")
        self._run.setEnabled(False)
        self._run.setToolTip("Load crops first")
        self._run.clicked.connect(self.embed)
        controls.addWidget(self._run)
        outer.addLayout(controls)

        # install_sorting + table_item, like every other view in the app.
        # A preview of eight dimensions is exactly the table someone sorts --
        # "which objects score highest on dimension 3" is the only question
        # a raw embedding column can answer by eye -- and Qt's default sort
        # is lexicographic, so -0.0412 would rank above 0.9031.
        self._table = install_sorting(QTableWidget(0, 0, self))
        self._table.setObjectName("EmbeddingsPreviewTable")
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        outer.addWidget(self._table, 1)

        self._caveat = QLabel(DIMENSION_CAVEAT, self)
        self._caveat.setObjectName("EmbeddingsCaveatLabel")
        self._caveat.setWordWrap(True)
        outer.addWidget(self._caveat)

        self._status = QLabel("", self)
        self._status.setObjectName("EmbeddingsStatusLabel")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._fill_backbones()
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into: a tooltip that only appears over the control
        # is one the user meets after they have already decided what to put
        # in it. One post-pass rather than a convention every hand-built row
        # has to remember -- the same call `live_preview.py` ends with.
        from .settings_model import retarget_field_tooltips

        retarget_field_tooltips(self)

    # -- inputs -----------------------------------------------------------

    def _fill_backbones(self) -> None:
        """Offer the engine's default first, and never an empty list."""
        from ...embeddings import DEFAULT_BACKBONE

        for name in (DEFAULT_BACKBONE, "resnet50", "convnext_tiny",
                     "vit_base_patch16_224"):
            if self._backbone.findText(name) < 0:
                self._backbone.addItem(name)
        self._backbone.setCurrentText(DEFAULT_BACKBONE)

    def set_crops(self, crops: np.ndarray, *, label: str = "") -> None:
        """Hand the screen an ``(objects, height, width, channels)`` stack.

        The channel axis comes last. Getting it backwards does not raise
        here; it reaches the model and fails there with a message about
        tensor shapes.

        That layout is what :mod:`spacr.crops` produces and what
        :func:`spacr.embeddings.embed_array` documents.

        :param crops: the object stack, channels LAST. Not copied: the
            screen keeps this array and hands it to the backbone, so a
            caller that mutates it afterwards changes what gets embedded.
        :raises ValueError: when the stack is not four-dimensional.
        """
        crops = np.asarray(crops)
        if crops.ndim != 4:
            raise ValueError(
                f"crops are (objects, height, width, channels); got shape "
                f"{crops.shape}. spacr.crops produces this stack, and "
                f"spacr.embeddings.embed_array expects channels last.")
        self._crops = crops
        self._source.setText(
            label or f"{crops.shape[0]} objects x {crops.shape[-1]} channels")
        self._run.setEnabled(True)
        self._run.setToolTip("Encode every object")

    # -- the run ----------------------------------------------------------

    def spec(self):
        """The :class:`spacr.embeddings.EmbeddingSpec` the controls describe."""
        from ...embeddings import EmbeddingSpec

        return EmbeddingSpec(
            backbone=str(self._backbone.currentText()).strip(),
            channel_policy=str(self._policy.currentData()),
            batch_size=int(self._batch.value()),
        )

    def embed(self) -> None:
        """Runs the encoder in the background and shows the result."""
        crops = getattr(self, "_crops", None)
        if crops is None:
            self._status.setText("Load crops first.")
            return
        spec = self.spec()
        self._status.setText(f"Embedding {crops.shape[0]} objects…")

        def work():
            """Run the backbone off the GUI thread.

            The import is inside because it pulls torch in: a user who never
            opens this screen should not pay for it, and a user who does
            should pay for it once, here, rather than at launch.
            """
            from ...embeddings import embed_array

            return embed_array(crops, spec)

        self._jobs.submit(work, self._on_embedded)

    def _on_embedded(self, result) -> None:
        """Fill the preview and say which encoder produced it."""
        self._result = result
        # `EmbeddingResult` carries the matrix and the names separately and
        # offers `to_frame(object_ids)`. The screen has no object ids -- it
        # was handed a crop stack, not a table -- so it builds the frame from
        # the two directly rather than inventing ids that would then look
        # like a join key.
        frame = pd.DataFrame(np.asarray(result.values),
                             columns=list(result.columns))
        self._frame = frame
        self._fill_preview(frame)

        from ...embeddings import encoder_entry

        entry = encoder_entry(self.spec())
        digest = entry.sha256[:12] + "…" if entry.sha256 else "no checksum"
        self._status.setText(
            f"{len(frame)} objects x {len(frame.columns)} dimensions. "
            f"Encoder {entry.name}, weights {digest}.")

    def _fill_preview(self, frame: pd.DataFrame) -> None:
        """Show the first few dimensions, and only the first few.

        The whole matrix is the result and it is not readable; a table of
        2,048 columns would invite scrolling through it as though a column
        meant something.
        """
        columns = list(frame.columns)[:PREVIEW_DIMENSIONS]
        rows = min(len(frame), 50)
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(rows)
        for row in range(rows):
            for index, column in enumerate(columns):
                # The displayed text is rounded to four places; the SORT
                # KEY is the float, so two dimensions that both print
                # -0.0000 still order by what they actually are.
                value = float(frame.iloc[row][column])
                item = table_item(f"{value:.4f}", key=value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(row, index, item)

    def _on_job_failed(self, message: str) -> None:
        """A refusal is a sentence on the screen, never a silent empty table."""
        self._status.setText(str(message))
        LOG.warning("embedding failed: %s", message)

    # -- lifecycle --------------------------------------------------------

    def is_busy(self) -> bool:
        """Whether a run is in flight."""
        return bool(self._jobs.active_jobs())

    def closeEvent(self, event):            # noqa: N802 - Qt name
        """Let the job runner stop its thread before the widget goes.

        :param event: Qt's close event, passed to the base class unchanged.
            Never ignored -- a screen that refuses to close because a run is
            in flight would trap the window, so the run is stopped instead.
        """
        try:
            self._jobs.shutdown()
        except Exception:                    # noqa: BLE001
            pass
        super().closeEvent(event)


def make_embeddings_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return EmbeddingsScreen()


# The row is declared in `spacr.qt.app_catalog`, read back here rather than
# restated, so the name, the blurb and the nine translations have one
# spelling and no second copy to drift from.
_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
APP_NAME_TRANSLATIONS = _ROW.translations


def register() -> bool:
    """Put Embeddings in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, not at import, so
    importing this module to reach :class:`EmbeddingsScreen` from a test or a
    notebook does not mutate process-wide state.

    ``SECTION_DATA``, beside the other things that turn a table into numbers.
    An embedding is a feature source, not an analysis: it produces columns
    that the reductions, the regressions and the hit calling then consume
    exactly as they consume the measured panel -- which is the whole point of
    386 step 5.
    """
    from ..app import register_app

    return register_app(APP_KEY, make_embeddings_screen)
