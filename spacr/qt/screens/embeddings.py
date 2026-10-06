"""Embeddings — a label-free vector for every object, beside the measured panel.

A SCREEN, NOT A SCRIPT. Cell-DINO, OpenPhenom and SubCell are all Python or
command line. No graphical platform exposes any of them.

The engine serves this lab. The screen offers that embedding without any
Python.

**What it is for.** Giving every object a vector that no label went into.
The measured panel says how big a cell is and how bright each stain was;
an embedding says what the cell LOOKS like, in a few hundred numbers a
pretrained backbone produces without being told what to look for. That is
what makes it worth having beside the panel rather than instead of it --
a phenotype nobody thought to measure is in the vector, and is in no
column.

**What it needs.** Crops, and a choice about the channels. The crops come
from the row at the top of the screen, which is :mod:`spacr.crop_loader`:
either a ``measurements.db``, narrowed to one object class, one plate or
any condition its crop table supports, or a folder of crop PNGs. Whichever
it is, the pixels are read the way every other screen reads them --
:func:`spacr.crops.resolve_crop_source` decides between the crops already
under ``data/`` and cutting them from ``merged/*.npy``, and says which it
chose. Nothing has to be prepared first: a plate that has been through
Measure with crop output on is ready.

**What it produces.** One row per object and one column per dimension,
named ``emb_c<channel>_<dimension>`` under the per-channel policy and
``emb_<dimension>`` under projection, with the model-zoo entry that made
them -- backbone, policy and the checksum of the weights on this machine.
Two runs' dimension 17 are the same number and not the same thing unless
all three match, which is why the entry is part of the result rather than
a log line. THE MATRIX LIVES IN THE SCREEN AND IS NOT YET WRITTEN
ANYWHERE: the table below shows the first few dimensions of the first
fifty objects, and there is no export button. Reaching the whole matrix
means :func:`spacr.embeddings.embed_array` from Python for now.

**What to do next.** Treat the columns as a feature source, not as a
result. They are consumed exactly as the measured panel is -- a reduction
to see whether the objects separate, a regression against the perturbation,
a retrieval to find the objects most like one you picked. Which is also the
warning the screen repeats under the table: a single dimension is not a
phenotype, and the only honest way to read one of these numbers is through
something that takes the whole vector.

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
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QStackedWidget,
    QTableWidget, QVBoxLayout, QWidget,
)

from ...crop_loader import (CROP_SOURCE_DATABASE, CROP_SOURCES,
                            DEFAULT_CROP_LIMIT, DEFAULT_PAGE_SIZE)
from ..app_catalog import declared_app
from ..i18n import tr
from ..widgets.measurements_example import install_test_data_button
from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.collapsible_splitter import FoldSection
from ..widgets.sortable_table import install_sorting, table_item
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.embeddings")

__all__ = ["EmbeddingsScreen", "make_embeddings_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "PREVIEW_DIMENSIONS",
           "DIMENSION_CAVEAT", "NOTHING_LOADED", "NOTHING_LOADED_HINT",
           "MAX_CROP_LIMIT"]

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

#: The screen's opening state, before anything has been asked for. It is a
#: heading rather than an empty table, because an empty table reads as a run
#: that produced nothing.
NOTHING_LOADED = "No crops loaded"

#: And what to do about it. Every other state of this panel replaces both
#: lines, so the panel is never showing a stale instruction beside a result.
NOTHING_LOADED_HINT = (
    "Choose a measurements database or a folder of crop PNGs above, then "
    "press Load crops. A database can be narrowed to one object class, one "
    "plate, or any condition its columns support."
)

#: The most crops the panel will let anybody ask for in one load. Not a
#: limit of the loader, which has none: it is the point past which a stack
#: of 96x96x3 crops stops fitting beside a backbone on an ordinary card, and
#: a spin box that offers a number nothing can hold is a trap.
MAX_CROP_LIMIT = 200_000


class _CropState(QWidget):
    """The panel shown where the preview goes until there is a preview.

    NOT :class:`spacr.qt.widgets.empty_state.EmptyState`, and the reason is
    the only reason: that widget takes its title and subtitle at
    construction and offers no way to change them, while every sentence this
    panel shows is different -- what to do first, what was loaded, and the
    refusal a particular query earned. It carries the same object names, so
    it is the same two type styles on the page.
    """

    def __init__(self, parent=None):
        """Build the two labels and open on the nothing-loaded state."""
        super().__init__(parent)
        self.setObjectName("EmbeddingsCropState")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["xl"], SPACING["xl"],
                                 SPACING["xl"], SPACING["xl"])
        outer.setSpacing(SPACING["sm"])
        outer.addStretch(1)
        self._title = QLabel(NOTHING_LOADED, self)
        self._title.setObjectName("TitleHeading")
        self._title.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._title)
        self._detail = QLabel(NOTHING_LOADED_HINT, self)
        self._detail.setObjectName("SubtitleSmall")
        self._detail.setAlignment(Qt.AlignCenter)
        self._detail.setWordWrap(True)
        outer.addWidget(self._detail)
        outer.addStretch(1)

    def say(self, title: str, detail: str) -> None:
        """Replace both lines at once.

        Both, always, so a heading can never be left standing over the
        explanation of a different state -- which is how "No crops" ends up
        under a sentence telling the reader to press Embed.

        :param title: the heading.
        :param detail: the sentence under it.
        """
        self._title.setText(str(title))
        self._detail.setText(str(detail))

    def title(self) -> str:
        """The heading currently shown."""
        return self._title.text()

    def detail(self) -> str:
        """The sentence currently shown under it."""
        return self._detail.text()


class EmbeddingsScreen(QWidget):
    """Pick a crop source and a channel policy, and embed every object.

    The crop loader above the encoder controls is the screen's entry point:
    :mod:`spacr.crop_loader` plans the selection, this screen pages through
    it on a worker thread, and :meth:`set_crops` -- which used to be
    reachable only from Python -- is what each finished load calls.

    :param parent: the usual Qt parent.
    :param threaded: ``False`` runs the embedding and the crop load inline,
        emitting the same signals in the same order, so a test drives the
        screen synchronously without the behaviour diverging.
    """

    #: A page of crops has been read: ``(done, total)``. Emitted from the
    #: loading thread, so it is queued onto the GUI thread by Qt; nothing
    #: connected to it may assume it runs where it was emitted.
    crops_progress = Signal(int, int)

    #: A load finished, carrying how many crops it put on the screen.
    crops_loaded = Signal(int)

    def __init__(self, parent=None, *, threaded: bool = True):
        """Build the screen: the controls above, the preview table below."""
        super().__init__(parent)
        self.setObjectName("EmbeddingsScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._result = None
        self._scale_record: dict = {}
        self._crop_record: Dict[str, Any] = {}
        self._plan = None
        self._stop = threading.Event()
        self._loading = False
        self._read_path = ""
        self._subcell_channels: Optional[tuple[int, int, int, int]] = None
        self._policy_before_subcell: Optional[str] = None
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self.crops_progress.connect(self._on_crops_progress)

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

        picker = QHBoxLayout()
        picker.setSpacing(SPACING["sm"])
        picker.addWidget(QLabel("Crops from:", self))
        self._where = QComboBox(self)
        self._where.setObjectName("EmbeddingsCropSourcePicker")
        for value, label in CROP_SOURCES:
            self._where.addItem(label, value)
        self._where.setToolTip(
            "A measurements database lets you choose which objects to embed. "
            "A crop folder takes every PNG in it.")
        self._where.currentIndexChanged.connect(self._on_source_changed)
        picker.addWidget(self._where)

        self._path = QLineEdit(self)
        self._path.setObjectName("EmbeddingsCropPath")
        self._path.setPlaceholderText(
            "path to measurements.db, or to a *_png crop folder")
        self._path.editingFinished.connect(self._on_path_changed)
        picker.addWidget(self._path, 1)

        self._browse = QPushButton("Browse…", self)
        self._browse.setObjectName("EmbeddingsBrowseButton")
        self._browse.setToolTip("Find the database or the crop folder")
        self._browse.clicked.connect(self.choose_path)
        picker.addWidget(self._browse)
        example = install_test_data_button(
            self, picker, lambda _folder, db: self._use_example_database(db),
            say=lambda text: self._status.setText(text))
        example.setObjectName("EmbeddingsTestDataButton")
        outer.addLayout(picker)

        selection = QHBoxLayout()
        selection.setSpacing(SPACING["sm"])
        self._class_label = QLabel("Object:", self)
        selection.addWidget(self._class_label)
        self._object = QComboBox(self)
        self._object.setObjectName("EmbeddingsObjectClassPicker")
        self._object.setToolTip(
            "Which object class to embed. It picks the label column in "
            "png_list and the mask plane a streamed crop is cut by, so "
            "asking for nuclei out of a cell crop table returns nothing "
            "rather than returning cells.")
        selection.addWidget(self._object)

        self._plate_label = QLabel("Plate:", self)
        selection.addWidget(self._plate_label)
        self._plate = QComboBox(self)
        self._plate.setObjectName("EmbeddingsPlatePicker")
        self._plate.addItem("every plate", "")
        self._plate.setToolTip(
            "One plate, or all of them. A scale is estimated per run, so a "
            "run over two plates scales both by one number.")
        selection.addWidget(self._plate)

        self._filter_label = QLabel("Where:", self)
        selection.addWidget(self._filter_label)
        self._filter = QLineEdit(self)
        self._filter.setObjectName("EmbeddingsCropFilter")
        self._filter.setPlaceholderText("any condition, e.g. cell_area > 500")
        self._filter.setToolTip(
            "A condition over the crop table's own columns, in SQL. The "
            "database is opened read-only, so nothing typed here can change "
            "it.")
        selection.addWidget(self._filter, 1)

        self._limit_label = QLabel("At most:", self)
        selection.addWidget(self._limit_label)
        self._limit = QSpinBox(self)
        self._limit.setObjectName("EmbeddingsCropLimit")
        self._limit.setRange(1, MAX_CROP_LIMIT)
        self._limit.setValue(DEFAULT_CROP_LIMIT)
        self._limit.setToolTip(
            "How many of the matching crops to load. The rest are counted "
            "and named, not silently dropped.")
        selection.addWidget(self._limit)

        self._load = QPushButton("Load crops", self)
        self._load.setObjectName("EmbeddingsLoadButton")
        self._load.setToolTip("Choose a database or a folder first")
        self._load.setEnabled(False)
        self._load.clicked.connect(self.load_crops)
        selection.addWidget(self._load)
        outer.addLayout(selection)

        controls = QHBoxLayout()
        controls.setSpacing(SPACING["sm"])

        controls.addWidget(QLabel("Channels:", self))
        self._policy = QComboBox(self)
        self._policy.setObjectName("EmbeddingsPolicyPicker")
        self._policy.addItem("Per channel (one pass per stain)", "per_channel")
        self._policy.addItem("Project to three (one pass)", "project")
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
        self._add_foundation_picker(controls)

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
        self._add_well_mil_button(controls)
        self._add_dino_button(controls)
        self._add_use_for_pickers(controls)
        outer.addLayout(controls)

        self._table = install_sorting(QTableWidget(0, 0, self))
        self._table.setObjectName("EmbeddingsPreviewTable")
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._state = _CropState(self)
        self._pages = QStackedWidget(self)
        self._pages.setObjectName("EmbeddingsPreviewPages")
        self._pages.addWidget(self._state)
        self._pages.addWidget(self._table)
        self._pages.setCurrentWidget(self._state)
        self._pages_section = FoldSection(
            self._pages, "Crop preview",
            persist_key="embeddings/Crop preview")
        outer.addWidget(self._pages_section, 1)

        self._caveat = QLabel(DIMENSION_CAVEAT, self)
        self._caveat.setObjectName("EmbeddingsCaveatLabel")
        self._caveat.setWordWrap(True)
        outer.addWidget(self._caveat)

        self._status = QLabel("", self)
        self._status.setObjectName("EmbeddingsStatusLabel")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._fill_backbones()
        self._fill_object_classes(())
        self._on_source_changed()
        from .settings_model import retarget_field_tooltips

        retarget_field_tooltips(self)


    def _add_foundation_picker(self, controls) -> None:
        """The Foundation model picker, alpha-gated, beside the backbone.

        A single-cell foundation model chosen here replaces the backbone for
        the run, whether the picker is shown or not; None keeps the backbone.
        """
        from ...embeddings import _FOUNDATION_MODELS
        from ..preferences import _apply_alpha_widgets

        label = QLabel(tr("Foundation model:"), self)
        label.setObjectName("EmbeddingsFoundationLabel")
        controls.addWidget(label)
        self._foundation = QComboBox(self)
        self._foundation.setObjectName("EmbeddingsFoundationPicker")
        self._foundation.addItem(tr("None (use the backbone)"), "")
        for name, info in _FOUNDATION_MODELS.items():
            self._foundation.addItem(tr(info["label"]), name)
        self._foundation.setToolTip(tr(
            "A model trained on microscopy rather than photographs. "
            "OpenPhenom and ChAda-ViT take any number of stains. SubCell's "
            "two-plane model takes DNA then protein; its four-plane model "
            "needs explicit microtubules, ER, DNA and protein mapping. "
            "Weights download once. Cell-DINO needs "
            "an official checkpoint and is not yet supported by this version. "
            "Default None (use the "
            "backbone)."))
        controls.addWidget(self._foundation)
        self._subcell_button = QPushButton(tr("Channels…"), self)
        self._subcell_button.setObjectName("EmbeddingsSubCellChannelsButton")
        self._subcell_button.setToolTip(tr(
            "Map four crop channels to microtubules, ER, DNA and protein "
            "before running SubCell's four-plane model."))
        self._subcell_button.clicked.connect(self._choose_subcell_channels)
        controls.addWidget(self._subcell_button)
        self._foundation.currentIndexChanged.connect(
            self._sync_subcell_controls)
        self._sync_subcell_controls()
        _apply_alpha_widgets(label)
        _apply_alpha_widgets(self._foundation)

    def _sync_subcell_controls(self) -> None:
        """Show the four-plane choice and keep its policy caption truthful."""
        from ..preferences import _get_show_alpha_features

        selected = self._foundation.currentData() == "subcell_rybg"
        project = self._policy.findData("project")
        if selected:
            if self._policy_before_subcell is None:
                self._policy_before_subcell = str(self._policy.currentData())
            self._policy.setCurrentIndex(project)
            self._policy.setItemText(
                project, tr("Four mapped planes (one pass)"))
            self._policy.setEnabled(False)
        else:
            self._policy.setItemText(
                project, tr("Project to three (one pass)"))
            self._policy.setEnabled(True)
            if self._policy_before_subcell is not None:
                previous = self._policy.findData(self._policy_before_subcell)
                self._policy.setCurrentIndex(max(previous, 0))
                self._policy_before_subcell = None
        crops = getattr(self, "_crops", None)
        count = 0 if crops is None else int(crops.shape[-1])
        self._subcell_button.setEnabled(selected and count >= 4)
        if selected and count < 4:
            self._subcell_button.setToolTip(tr(
                "Load crops with at least four channels before mapping "
                "SubCell's microtubules, ER, DNA and protein planes."))
        else:
            self._subcell_button.setToolTip(tr(
                "Map four crop channels to microtubules, ER, DNA and protein "
                "before running SubCell's four-plane model."))
        alpha_on = _get_show_alpha_features()
        self._subcell_button.setVisible(selected and alpha_on)
        self._subcell_button.setProperty(
            "_spacr_alpha_hid", bool(selected and not alpha_on))

    def _subcell_mapping_error(
            self, channels: tuple[object, ...] | None,
            available: int) -> str:
        """Explain an incomplete, repeated or stale four-plane mapping."""
        if not isinstance(channels, tuple) or len(channels) != 4 or any(
                not isinstance(channel, int) or isinstance(channel, bool)
                for channel in channels):
            return tr("Choose a crop channel for each SubCell plane: "
                      "microtubules, ER, DNA and protein.")
        if len(set(channels)) != 4:
            return tr("Choose four different crop channels for SubCell.")
        if any(channel < 0 or channel >= available for channel in channels):
            return tr("A selected SubCell channel is outside the loaded "
                      "crops. Open Channels… and map the four planes again.")
        return ""

    def _subcell_channels_dialog(self) -> QDialog:
        """Build four initially unchosen selectors in official R/Y/B/G order."""
        crops = getattr(self, "_crops", None)
        available = 0 if crops is None else int(crops.shape[-1])
        dialog = QDialog(self)
        dialog.setObjectName("EmbeddingsSubCellChannelsDialog")
        dialog.setWindowTitle(tr("SubCell four-plane channels"))
        layout = QVBoxLayout(dialog)
        guidance = QLabel(tr(
            "Assign four different crop channels in SubCell's official "
            "order. No stain identity is guessed from channel position."),
            dialog)
        guidance.setWordWrap(True)
        layout.addWidget(guidance)
        form = QFormLayout()
        microtubules = QComboBox(dialog)
        microtubules.setObjectName("EmbeddingsSubCellMicrotubulesChannel")
        er = QComboBox(dialog)
        er.setObjectName("EmbeddingsSubCellErChannel")
        dna = QComboBox(dialog)
        dna.setObjectName("EmbeddingsSubCellDnaChannel")
        protein = QComboBox(dialog)
        protein.setObjectName("EmbeddingsSubCellProteinChannel")
        selectors = (microtubules, er, dna, protein)
        for role, (label, selector) in enumerate((
            (tr("Microtubules (R)"), microtubules),
            (tr("ER (Y)"), er),
            (tr("DNA (B)"), dna),
            (tr("Protein (G)"), protein),
        )):
            selector.addItem(tr("Choose channel…"), None)
            for index in range(available):
                selector.addItem(tr("Channel {position} (index {index})").format(
                    position=index + 1, index=index), index)
            if self._subcell_channels is not None:
                matching = selector.findData(self._subcell_channels[role])
                selector.setCurrentIndex(max(matching, 0))
            form.addRow(label, selector)
        layout.addLayout(form)
        problem = QLabel("", dialog)
        problem.setObjectName("EmbeddingsSubCellChannelsProblem")
        problem.setWordWrap(True)
        layout.addWidget(problem)
        actions = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        layout.addWidget(actions)

        def accept_mapping() -> None:
            """Store only a complete four-index choice, then close."""
            chosen = tuple(selector.currentData() for selector in selectors)
            current = getattr(self, "_crops", None)
            count = 0 if current is None else int(current.shape[-1])
            reason = self._subcell_mapping_error(chosen, count)
            if reason:
                problem.setText(reason)
                return
            self._subcell_channels = chosen
            self._status.setText(tr(
                "SubCell channel mapping saved in microtubules, ER, DNA, "
                "protein order."))
            dialog.accept()

        actions.accepted.connect(accept_mapping)
        actions.rejected.connect(dialog.reject)
        return dialog

    def _choose_subcell_channels(self) -> None:
        """Open the mapping form without retaining its Qt wrappers."""
        dialog = self._subcell_channels_dialog()
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _add_use_for_pickers(self, controls) -> None:
        """Alpha controls that score the embedding and feed it onward.

        A label table gives the retrieval scorecard of the chosen backbone;
        the picker then runs an image UMAP or a classifier on the vectors.
        """
        from ..preferences import _apply_alpha_widgets

        self._labels: Dict[str, str] = {}
        self._labels_button = QPushButton(tr("Labels…"), self)
        self._labels_button.setObjectName("EmbeddingsLabelsButton")
        self._labels_button.setToolTip(tr(
            "Choose a table with one row per embedded crop, in crop order, "
            "and a 'label' column (else the first column). The backbone is "
            "scored on it: kNN accuracy, mean average precision and "
            "precision at 10 against chance, and the classifier uses the "
            "same labels. Default no labels."))
        self._labels_button.clicked.connect(lambda: self._choose_labels())
        controls.addWidget(self._labels_button)
        label = QLabel(tr("Use for:"), self)
        label.setObjectName("EmbeddingsUseForLabel")
        controls.addWidget(label)
        self._use_for = QComboBox(self)
        self._use_for.setObjectName("EmbeddingsUseForPicker")
        self._use_for.addItem(tr("Image UMAP"), "umap")
        self._use_for.addItem(tr("Classifier"), "classifier")
        self._use_for.setToolTip(tr(
            "What to do with the embedding of the chosen backbone or "
            "foundation model. Image UMAP maps every crop to two "
            "dimensions; Classifier cross-validates a logistic regression "
            "on the labels. Default Image UMAP."))
        controls.addWidget(self._use_for)
        self._use_for_run = QPushButton(tr("Run"), self)
        self._use_for_run.setObjectName("EmbeddingsUseForRun")
        self._use_for_run.setToolTip(tr(
            "Run the chosen use on the last embedding. Default off until "
            "an embedding exists."))
        self._use_for_run.clicked.connect(lambda: self._use_embeddings())
        controls.addWidget(self._use_for_run)
        for widget in (self._labels_button, label, self._use_for,
                       self._use_for_run):
            _apply_alpha_widgets(widget)

    def _choose_labels(self, path: str = "") -> str:
        """Read a label table and score the last embedding against it.

        :param path: the table; asks for one when empty.
        :returns: the table used, or ``''`` when nothing was chosen or the
            table does not fit the embedding.
        """
        from ...tabular import read_table

        if not path:
            path, _filter = QFileDialog.getOpenFileName(
                self, tr("Choose a label table"), "",
                tr("Tables (*.csv *.tsv *.parquet *.feather *.xlsx)"))
        if not path:
            return ""
        frame = read_table(str(path), report=None)
        column = "label" if "label" in frame.columns else frame.columns[0]
        values = frame[column].tolist()
        embedded = getattr(self, "_frame", None)
        if embedded is not None and len(values) != len(embedded):
            self._status.setText(tr(
                "The label table has {rows} rows but {crops} crops were "
                "embedded.").format(rows=len(values), crops=len(embedded)))
            return ""
        self._labels = {str(i): ("" if pd.isna(v) else str(v))
                        for i, v in enumerate(values)}
        if embedded is not None:
            self._show_scorecard()
        return str(path)

    def _show_scorecard(self) -> None:
        """Measure the encoder's retrieval scorecard on the labels."""
        from ...embeddings import _scored_encoder_entry

        spec = getattr(self._result, "spec", None) or self.spec()
        entry = _scored_encoder_entry(spec, self._frame, self._labels)
        self._entry = entry
        card = entry.metrics
        if not card:
            self._status.setText(tr(
                "The labels cannot be scored: two classes with labelled "
                "crops are needed."))
            return
        self._status.setText(tr(
            "Encoder {name}: kNN accuracy {knn:.2f}, mAP {map:.2f} (chance "
            "{chance:.2f}), precision at 10 {prec:.2f}, over {n} crops in "
            "{classes} classes.").format(
                name=entry.name, knn=card["knn_accuracy"], map=card["map"],
                chance=card["chance_map"], prec=card["precision_at_k"],
                n=int(card["n"]), classes=int(card["classes"])))

    def _use_embeddings(self) -> None:
        """Run the picked use (image UMAP or classifier) on the embedding."""
        frame = getattr(self, "_frame", None)
        if frame is None:
            self._status.setText(tr("Embed the crops first."))
            return
        use = str(self._use_for.currentData())
        labels = dict(self._labels)
        if use == "classifier" and not labels:
            self._status.setText(tr("Choose labels first."))
            return

        def work():
            """Map or classify off the GUI thread."""
            from ... import embeddings as emb

            if use == "umap":
                return use, emb._embedding_umap(frame)
            return use, emb._embedding_classifier_scorecard(frame, labels)

        self._status.setText(tr("Running on the embedding…"))
        self._jobs.submit(work, self._on_used)

    def _on_used(self, result) -> None:
        """Show the UMAP coordinates or the classifier scorecard."""
        use, answer = result
        spec = getattr(self._result, "spec", None) or self.spec()
        if use == "umap":
            self._umap = answer
            self._fill_preview(answer)
            self._status.setText(tr(
                "Image UMAP of {n} crops from {name}.").format(
                    n=len(answer), name=spec.backbone))
            return
        self._status.setText(tr(
            "Classifier on {name}: accuracy {acc:.2f} ± {sd:.2f} over "
            "{folds} folds (chance {chance:.2f}), {n} crops in {classes} "
            "classes.").format(
                name=spec.backbone, acc=answer["accuracy"],
                sd=answer["accuracy_sd"], folds=int(answer["folds"]),
                chance=answer["chance"], n=int(answer["n"]),
                classes=int(answer["classes"])))

    def _add_well_mil_button(self, controls) -> None:
        """The alpha button that learns which cells carry a well label."""
        from ..preferences import _apply_alpha_widgets

        self._well_mil = QPushButton(tr("Learn from well labels…"), self)
        self._well_mil.setObjectName("EmbeddingsWellMilButton")
        self._well_mil.setToolTip(tr(
            "Choose a per-cell table with a 'well' column, a 'well_label' "
            "column (1 for treated or knockout wells, 0 for controls) and "
            "embedding or numeric feature columns. An attention model learns "
            "from the well labels alone which cells carry the phenotype. "
            "Two tables are written beside the input: each cell's attention "
            "and each well's probability. Runs on the CPU. Default 4-fold "
            "cross-validation over wells."))
        self._well_mil.clicked.connect(lambda: self._learn_from_well_labels())
        controls.addWidget(self._well_mil)
        _apply_alpha_widgets(self._well_mil)

    def _mil_column_form(self, columns):
        """A form for choosing the well, label and feature columns.

        :param columns: the table's column names.
        :returns: ``(dialog, well, label, features)``: the dialog, the two
            combo boxes and the feature list; the defaults are the ``wellID``
            or ``well`` column, ``well_label`` and the ``emb_`` columns.
        """
        from PySide6.QtWidgets import (QAbstractItemView, QComboBox,
                                       QDialog, QDialogButtonBox,
                                       QFormLayout, QListWidget)

        columns = [str(c) for c in columns]
        dialog = QDialog(self)
        dialog.setObjectName("EmbeddingsWellMilForm")
        dialog.setWindowTitle(tr("Learn from well labels"))
        form = QFormLayout(dialog)
        well = QComboBox(dialog)
        well.setObjectName("EmbeddingsWellMilWellColumn")
        well.addItems(columns)
        for name in ("wellID", "well"):
            if name in columns:
                well.setCurrentText(name)
                break
        label = QComboBox(dialog)
        label.setObjectName("EmbeddingsWellMilLabelColumn")
        label.addItems(columns)
        if "well_label" in columns:
            label.setCurrentText("well_label")
        features = QListWidget(dialog)
        features.setObjectName("EmbeddingsWellMilFeatureColumns")
        features.setSelectionMode(QAbstractItemView.MultiSelection)
        features.addItems(columns)
        wanted = [c for c in columns if c.startswith("emb_")]
        for row, name in enumerate(columns):
            if name in wanted:
                features.item(row).setSelected(True)
        form.addRow(tr("Well column"), well)
        form.addRow(tr("Label column"), label)
        form.addRow(tr("Feature columns"), features)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        return dialog, well, label, features

    def _ask_mil_columns(self, columns):
        """Ask for the columns; ``None`` when the form is dismissed."""
        from PySide6.QtWidgets import QDialog

        dialog, well, label, features = self._mil_column_form(columns)
        if dialog.exec() != QDialog.Accepted:
            return None
        chosen = [i.text() for i in features.selectedItems()]
        return {"well_column": well.currentText(),
                "label_column": label.currentText(),
                "feature_columns": chosen or None}

    def _learn_from_well_labels(self, path: str = "",
                                columns=None, labels: str = "") -> str:
        """Train the well-label attention model on a per-cell table.

        The table is read, a form asks for the well, label and feature
        columns, the model is cross-validated and then fitted on every well
        in the background, and ``<name>_mil_cells.csv`` and
        ``<name>_mil_wells.csv`` are written beside it. The scorecard and a
        montage of the highest-attention crops are shown on the screen.

        A ``measurements.db`` is read through its stored crop embeddings,
        joined to object ids and wells; the well labels then come from a
        second table with a well column and a ``well_label`` column.

        :param path: the table or database; asks for one when empty.
        :param columns: ``well_column``, ``label_column`` and
            ``feature_columns`` for the model; asks with a form when ``None``.
        :param labels: the well-label table for a database; asks for one
            when empty.
        :returns: the table used, or ``''`` when a dialog was dismissed or
            the database has no stored embeddings.
        """
        from ...tabular import read_table

        if not path:
            path, _filter = QFileDialog.getOpenFileName(
                self, tr("Choose a per-cell table with well labels"), "",
                tr("Tables or measurements.db (*.csv *.tsv *.parquet "
                   "*.feather *.xlsx *.db)"))
        if not path:
            return ""
        path = str(path)
        if path.lower().endswith(".db"):
            from ...embeddings import _mil_frame_from_db

            if not labels:
                labels, _filter = QFileDialog.getOpenFileName(
                    self, tr("Choose the well-label table for the stored "
                             "embeddings"), os.path.dirname(path),
                    tr("Tables (*.csv *.tsv *.parquet *.feather *.xlsx)"))
            if not labels:
                return ""
            try:
                frame = _mil_frame_from_db(path, str(labels))
            except ValueError as exc:
                self._status.setText(str(exc))
                return ""
        else:
            frame = read_table(path, report=None)
        if columns is None:
            columns = self._ask_mil_columns(frame.columns)
            if columns is None:
                return ""
        columns = dict(columns)
        self._status.setText(tr("Learning from well labels…"))

        def work():
            """Score and fit off the GUI thread."""
            from ...embeddings import _mil_from_table
            from ...tabular import write_table

            cells, wells, card = _mil_from_table(frame, **columns)
            stem = os.path.splitext(path)[0]
            write_table(cells, stem + "_mil_cells.csv")
            write_table(wells, stem + "_mil_wells.csv")
            return card, cells

        self._jobs.submit(work, self._on_well_mil_done)
        return path

    def _on_well_mil_done(self, result) -> None:
        """Show the scorecard and the highest-attention crops."""
        card, cells = result
        self._mil_card = dict(card)
        self._status.setText(tr(
            "Well-label model: held-out well AUROC {mil:.2f} (mean-feature "
            "baseline {mean:.2f}) over {wells} wells. Cell attention and "
            "well probabilities were written beside the table.").format(
                mil=card["mil_auroc"], mean=card["mean_auroc"],
                wells=int(card["wells"])))
        self._show_mil_result(card, cells)

    def _show_mil_result(self, card, cells, top: int = 24) -> None:
        """A window with the scorecard and a montage of the top crops.

        The montage holds the ``top`` cells with the highest attention that
        have a ``png_path``; without crop paths only the scorecard shows.
        """
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPainter, QPixmap
        from PySide6.QtWidgets import (QDialog, QLabel, QTableWidget,
                                       QVBoxLayout)

        from ..hidpi import scaled_for

        dialog = QDialog(self)
        dialog.setObjectName("EmbeddingsWellMilResult")
        dialog.setWindowTitle(tr("Well-label model"))
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(card), 2, dialog)
        table.setObjectName("EmbeddingsWellMilScorecard")
        table.setHorizontalHeaderLabels([tr("Measure"), tr("Value")])
        for row, (key, value) in enumerate(card.items()):
            table.setItem(row, 0, table_item(str(key)))
            text = (f"{value:.3f}" if isinstance(value, float)
                    else str(value))
            table.setItem(row, 1, table_item(text))
        install_sorting(table)
        layout.addWidget(table)
        montage = QLabel(dialog)
        montage.setObjectName("EmbeddingsWellMilMontage")
        self._mil_montage_count = 0
        if "png_path" in cells.columns and "mil_attention" in cells.columns:
            ranked = cells.sort_values("mil_attention", ascending=False)
            tiles = []
            for crop in ranked["png_path"].dropna().astype(str):
                pixmap = QPixmap(crop)
                if not pixmap.isNull():
                    tiles.append(scaled_for(pixmap, montage, 64))
                if len(tiles) >= top:
                    break
            if tiles:
                across = min(len(tiles), 8)
                down = -(-len(tiles) // across)
                ratio = self.devicePixelRatioF()
                sheet = QPixmap(int(across * 66 * ratio),
                                int(down * 66 * ratio))
                sheet.setDevicePixelRatio(ratio)
                sheet.fill(Qt.black)
                painter = QPainter(sheet)
                for i, tile in enumerate(tiles):
                    painter.drawPixmap(
                        QPoint((i % across) * 66 + 1, (i // across) * 66 + 1),
                        tile)
                painter.end()
                montage.setPixmap(sheet)
                self._mil_montage_count = len(tiles)
        if not self._mil_montage_count:
            montage.setText(tr(
                "No crop images: add a 'png_path' column to see the cells "
                "with the highest attention."))
        layout.addWidget(montage)
        self._mil_result = dialog
        dialog.show()

    def _add_dino_button(self, controls) -> None:
        """The alpha button that pretrains a backbone on the loaded crops."""
        from ..preferences import _apply_alpha_widgets

        self._dino = QPushButton(tr("Pretrain on these crops…"), self)
        self._dino.setObjectName("EmbeddingsDinoPretrainButton")
        self._dino.setToolTip(tr(
            "Self-supervised (DINO) pretraining of a ResNet-18 on the loaded "
            "crops, no labels needed, under the chosen channel policy. Choose "
            "a checkpoint file; it is saved after every epoch, and choosing "
            "the same file again resumes the run. When it finishes the "
            "checkpoint is offered in the Foundation model picker. Slow "
            "without a GPU. Default 20 epochs from random weights at 64 "
            "px."))
        self._dino.clicked.connect(lambda: self._pretrain_dino())
        controls.addWidget(self._dino)
        _apply_alpha_widgets(self._dino)

    def _pretrain_dino(self, path: str = "", *, epochs: int = 20) -> str:
        """Pretrain a backbone on the loaded crops and offer it as an encoder.

        Training runs in the background and writes a resumable checkpoint to
        ``path``; when it ends the checkpoint is added to the Foundation
        model picker and chosen, so the next Embed uses it.

        :param path: the checkpoint file; asks for one when empty.
        :param epochs: epochs to train up to.
        :returns: the checkpoint path, or ``''`` when nothing was started.
        """
        crops = getattr(self, "_crops", None)
        if crops is None:
            self._status.setText(tr("Load crops first."))
            return ""
        if not path:
            path, _filter = QFileDialog.getSaveFileName(
                self, tr("Save the pretrained backbone as"), "",
                tr("PyTorch checkpoints (*.pt)"))
        if not path:
            return ""
        path = str(path)
        policy = str(self._policy.currentData())
        self._status.setText(tr("Pretraining on {n} crops…").format(
            n=crops.shape[0]))

        def work():
            """Train off the GUI thread."""
            from ...embeddings import _dino_pretrain

            return _dino_pretrain(crops, path, channel_policy=policy,
                                  epochs=epochs, pretrained=False)

        self._jobs.submit(work, self._on_dino_done)
        return path

    def _on_dino_done(self, summary) -> None:
        """Offer the new checkpoint in the picker and choose it."""
        from ...embeddings import _DINO_PREFIX

        self._dino_summary = dict(summary)
        data = _DINO_PREFIX + str(summary["path"])
        index = self._foundation.findData(data)
        if index < 0:
            self._foundation.addItem(
                tr("Own DINO: {name}").format(
                    name=os.path.basename(str(summary["path"]))), data)
            index = self._foundation.count() - 1
        self._foundation.setCurrentIndex(index)
        loss = summary["loss"][-1] if summary["loss"] else float("nan")
        self._status.setText(tr(
            "Pretrained {epochs} epochs (last loss {loss:.3f}). The "
            "checkpoint is chosen in the Foundation model picker; press "
            "Embed to use it.").format(epochs=int(summary["epochs"]),
                                       loss=loss))

    def _fill_backbones(self) -> None:
        """Offer the engine's default first, and never an empty list."""
        from ...embeddings import DEFAULT_BACKBONE

        for name in (DEFAULT_BACKBONE, "resnet50", "convnext_tiny",
                     "vit_base_patch16_224"):
            if self._backbone.findText(name) < 0:
                self._backbone.addItem(name)
        self._backbone.setCurrentText(DEFAULT_BACKBONE)

    def _fill_object_classes(self, classes) -> None:
        """Offer the classes this database has, or every class it could have.

        With nothing chosen yet the full vocabulary is offered, because a
        combo that is empty until a path is typed looks broken. Once a
        database is open the list is what IS in it, so a class that can only
        ever return nothing is not on offer -- and the current choice is kept
        when the new database still has it, so re-pointing at a second plate
        does not silently switch which object is being embedded.
        """
        from ...png_list import PNG_LIST_ID_COLUMNS

        wanted = self._object.currentData() or "cell"
        names = tuple(classes) or tuple(PNG_LIST_ID_COLUMNS)
        self._object.blockSignals(True)
        try:
            self._object.clear()
            for name in names:
                self._object.addItem(name, name)
            index = self._object.findData(wanted)
            self._object.setCurrentIndex(max(0, index))
        finally:
            self._object.blockSignals(False)

    def _fill_plates(self, names) -> None:
        """Offer every plate in the database, with 'every plate' first."""
        wanted = self._plate.currentData() or ""
        self._plate.blockSignals(True)
        try:
            self._plate.clear()
            self._plate.addItem("every plate", "")
            for name in names:
                self._plate.addItem(str(name), str(name))
            index = self._plate.findData(wanted)
            self._plate.setCurrentIndex(max(0, index))
        finally:
            self._plate.blockSignals(False)

    def _use_example_database(self, database) -> None:
        """Point the crop source at the example plate's measurements database.

        Selects the database source, fills the path and refreshes what it
        offers; the crops are loaded when Load crops is pressed.

        :param database: the example plate's ``measurements.db``.
        """
        index = self._where.findData(CROP_SOURCE_DATABASE)
        if index >= 0:
            self._where.setCurrentIndex(index)
        self._path.setText(str(database))
        self._on_path_changed()

    def crop_source(self) -> str:
        """Which kind of place the crops come from."""
        return str(self._where.currentData() or CROP_SOURCE_DATABASE)

    def _on_source_changed(self, *_args) -> None:
        """Grey the controls the chosen source does not read.

        A folder has no classes, no plates and no columns to filter on. The
        house rule from :mod:`spacr.crop_source` is that a control the user
        can edit that changes nothing is worse than one that is not there,
        so these are disabled rather than left to mislead.
        """
        database = self.crop_source() == CROP_SOURCE_DATABASE
        for widget in (self._class_label, self._object, self._plate_label,
                       self._plate, self._filter_label, self._filter):
            widget.setEnabled(database)
        self._path.setPlaceholderText(
            "path to measurements.db" if database
            else "path to a folder of crop PNGs")
        self._on_path_changed()

    def _on_path_changed(self, *_args) -> None:
        """Enable Load once there is a path, and read what the database offers.

        NOTHING HERE TOUCHES THE FILESYSTEM. Whether the path is there is a
        stat, and a stat against a hung or unmounting network share blocks
        for as long as the mount takes to give up -- on this thread, which
        is the GUI one. This method runs on every edit of the path box and
        again at the end of every load, when the user did nothing at all, so
        the freeze would arrive unprompted.

        So Load is offered for any non-empty path and the answer comes from
        pressing it: :func:`spacr.crop_loader.plan_crops` names the path it
        could not find, which is a better sentence than a greyed button, and
        it says it from a worker thread.
        """
        path = str(self._path.text()).strip()
        self._load.setEnabled(bool(path) and not self._loading)
        self._load.setToolTip(
            "Read these crops" if path
            else "Choose a database or a folder first")
        if not path or self.crop_source() != CROP_SOURCE_DATABASE:
            return
        full = os.path.expanduser(path)
        if full == self._read_path:
            return
        self._read_path = full
        self._read_database_choices(full)

    def _read_database_choices(self, path: str) -> None:
        """Fill the class and plate pickers from ``path``, off the GUI thread.

        Run once per database rather than once per edit: this is reached
        from every path change AND from the end of every load, since both
        put the controls back, and re-reading a database on a network mount
        each time a load finishes is a pause nobody asked for.
        """
        def work():
            """Ask the database what it holds. Two cheap indexed reads.

            THE STAT IS HERE, on the worker, because it is the part that
            blocks on a slow mount. A path that is not a file yet is not an
            error -- it is one somebody is halfway through typing -- so it
            answers with nothing to offer rather than raising, which would
            put a refusal in the status line for every keystroke.
            """
            from ...crop_loader import object_classes, plates

            if not os.path.isfile(path):
                return (), ()
            return object_classes(path), plates(path)

        def done(answer) -> None:
            """Put the answer in the two combos."""
            classes, names = answer
            self._fill_object_classes(classes)
            self._fill_plates(names)

        self._jobs.submit(work, done)

    def choose_path(self) -> str:
        """Ask for the database or the folder, and remember the answer.

        :returns: what was chosen, or ``''`` when the dialog was dismissed,
            so a caller can tell a cancel from a choice.
        """
        if self.crop_source() == CROP_SOURCE_DATABASE:
            path, _filter = QFileDialog.getOpenFileName(
                self, "Choose a measurements database", self._path.text(),
                "SQLite databases (*.db *.sqlite);;Every file (*)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Choose a folder of crop PNGs", self._path.text())
        if path:
            self._path.setText(str(path))
            self._on_path_changed()
        return str(path or "")

    def crop_query(self):
        """The :class:`spacr.crop_loader.CropQuery` the controls describe.

        :returns: the query a press of Load would run, which is what makes
            the panel testable without pressing anything.
        """
        from ...crop_loader import CropQuery

        return CropQuery(
            source=self.crop_source(),
            path=os.path.expanduser(str(self._path.text()).strip()),
            object_type=str(self._object.currentData() or "cell"),
            plate=str(self._plate.currentData() or ""),
            where=str(self._filter.text()).strip(),
            limit=int(self._limit.value()),
            page_size=DEFAULT_PAGE_SIZE,
        )

    def is_loading(self) -> bool:
        """Whether a crop load is in flight."""
        return bool(self._loading)

    def load_crops(self) -> None:
        """Plan the selection, then read it a page at a time.

        TWO JOBS, NOT ONE, and the split is the point: planning is a count
        and a select, so it comes back in milliseconds and the screen can
        say what was matched -- or why nothing was -- before committing to
        read a single pixel. Only then does the second job start, and it
        reports after every page.
        """
        if self._loading:
            self.stop_loading()
            return
        query = self.crop_query()
        if not query.path:
            self._refuse("No crops loaded",
                         "Choose a measurements database or a folder of crop "
                         "PNGs first.")
            return
        self._stop.clear()
        self._set_loading(True)
        self._status.setText(f"Looking for {query.describe()}…")

        def work():
            """Ask what matches. No pixels are read here."""
            from ...crop_loader import plan_crops

            return plan_crops(query)

        self._jobs.submit(work, self._on_planned)

    def stop_loading(self) -> None:
        """Stop a load between pages, keeping the pages already read.

        The flag is checked before each page rather than inside one, so a
        stop costs at most one page and never leaves a half-written crop in
        the stack.
        """
        self._stop.set()
        self._status.setText("Stopping after this page…")

    def _set_loading(self, loading: bool) -> None:
        """Turn Load into Stop for the duration, and park the other controls.

        EMBED IS PUT BACK THE WAY IT WAS, not left off. A load that matched
        nothing unloads nothing, so the crops from the previous load are
        still there and still encodable; leaving Embed disabled would make
        the panel's own sentence about them false, and there is no way for a
        user to get it back without loading again.
        """
        self._loading = bool(loading)
        self._load.setText("Stop" if loading else "Load crops")
        self._load.setEnabled(True if loading else bool(
            str(self._path.text()).strip()))
        for widget in (self._where, self._path, self._browse, self._object,
                       self._plate, self._filter, self._limit):
            widget.setEnabled(not loading)
        self._run.setEnabled(
            False if loading else getattr(self, "_crops", None) is not None)
        if not loading:
            self._on_source_changed()

    def _on_planned(self, plan) -> None:
        """Say what matched, and start reading it -- or say why nothing did.

        A STOP PRESSED DURING PLANNING IS HONOURED HERE. The plan job cannot
        be interrupted -- it is one count and one select -- so a user who
        presses Stop while it is in flight is answered when it returns, by
        not starting the read at all. Submitting it anyway would read a page
        and then refuse, which is slower and says the wrong thing.
        """
        self._plan = plan
        if plan.is_empty:
            self._set_loading(False)
            self._refuse("No crops", plan.empty_reason)
            return
        if self._stop.is_set():
            self._stopped(f"Loading {plan.describe()} was stopped before any "
                          f"crop was read.")
            return
        self._status.setText(f"Loading {plan.describe()}…")
        self._state.say(f"Loading {plan.count:,} crops",
                        plan.describe())
        stop = self._stop
        emit = self.crops_progress.emit

        def work():
            """Read the pages on a worker thread.

            ``emit`` is a bound signal and the only thing here that reaches
            the GUI: Qt queues it onto the receiving thread, so the progress
            line moves without this function ever touching a widget.
            """
            from ...crop_loader import load_crops

            record: Dict[str, Any] = {}
            crops = load_crops(plan, progress=emit,
                               cancelled=stop.is_set, record=record)
            return crops, record

        self._jobs.submit(work, self._on_crops_read)

    def _on_crops_progress(self, done: int, total: int) -> None:
        """One page in. Runs on the GUI thread, however it was emitted."""
        self._status.setText(f"Loaded {done:,} of {total:,} crops…")

    def _on_crops_read(self, answer) -> None:
        """Hand the stack to the screen and say what it cost."""
        crops, record = answer
        self._crop_record = dict(record)
        self._set_loading(False)
        self.set_crops(crops, label=self._loaded_label(record))
        self._status.setText(self._loaded_sentence(record))
        self.crops_loaded.emit(int(crops.shape[0]))

    def _loaded_label(self, record) -> str:
        """The short form, for the label beside the module header."""
        shape = record.get("crop_shape") or []
        size = "x".join(str(v) for v in shape[:2]) if shape else "?"
        channels = shape[2] if len(shape) > 2 else "?"
        return (f"{int(record.get('loaded', 0)):,} objects, {size}, "
                f"{channels} channels")

    def _loaded_sentence(self, record) -> str:
        """The long form: what was taken, out of what, and what was changed.

        Every clause here is a thing a reader would otherwise have to guess
        at, and three of them change what the numbers mean: a capped load is
        a subset of the plate, a dropped row is a crop that does not exist to
        be loaded, and a conformed crop has been padded or trimmed.

        THE CAP CLAUSE IS DRIVEN BY ``capped``, NOT BY ``loaded < matched``.
        The two are not the same on the merged route, where rows that cannot
        be cut leave ``loaded`` below ``matched`` with no limit involved, and
        "raise 'At most' to take more" is then an instruction that returns
        the same crops and prints the same sentence again.
        """
        loaded = int(record.get("loaded", 0))
        matched = int(record.get("matched", loaded))
        dropped = int(record.get("dropped", 0))
        parts = [f"{loaded:,} crops loaded"]
        if record.get("stopped"):
            parts.append(f"stopped early; {matched:,} matched")
        elif record.get("capped"):
            parts.append(f"the first of {matched:,} that matched — raise "
                         f"'At most' to take more")
        if dropped:
            parts.append(f"{dropped:,} more matched but could not be cut "
                         f"from merged/*.npy")
        conformed = int(record.get("conformed", 0))
        if conformed:
            shape = record.get("crop_shape") or []
            frame = "x".join(str(v) for v in shape[:2])
            parts.append(f"{conformed:,} padded or trimmed to {frame}")
        parts.append(str(record.get("source", "")))
        return ". ".join(part for part in parts if part) + "."

    def _refuse(self, title: str, detail: str) -> None:
        """Put a refusal where the result would have been, and in the status.

        Both places, because they are read at different moments: the panel is
        where somebody looking for the crops is looking, and the status line
        is what survives a scroll.

        A REFUSAL UNLOADS NOTHING, and the panel has to say so or it
        contradicts the rest of the screen. A query that matched nothing
        leaves the previous stack exactly where it was, with Embed still
        enabled and the header still naming it, and a panel reading "No
        crops" over that is the screen telling a reader two things at once.
        """
        crops = getattr(self, "_crops", None)
        if crops is not None:
            detail = (f"{detail} The {crops.shape[0]:,} crops already loaded "
                      f"are untouched, and Embed still encodes them.")
        self._state.say(title, detail)
        self._pages.setCurrentWidget(self._state)
        self._status.setText(detail)
        LOG.info("%s: %s", title, detail)

    def crop_record(self) -> Dict[str, Any]:
        """What the last load actually read, as the loader recorded it."""
        return dict(self._crop_record)

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
        self._scale_record = {}
        self._source.setText(
            label or f"{crops.shape[0]} objects x {crops.shape[-1]} channels")
        if (self._subcell_channels is not None
                and self._subcell_mapping_error(
                    self._subcell_channels, int(crops.shape[-1]))):
            self._subcell_channels = None
            self._status.setText(tr(
                "The new crops have fewer channels. Reopen Channels… to "
                "map SubCell's four planes."))
        self._sync_subcell_controls()
        self._run.setEnabled(True)
        self._run.setToolTip("Encode every object")
        self._state.say(
            f"{crops.shape[0]:,} crops loaded",
            "Choose how the channels are encoded, then press Embed. The "
            "table fills with the first few dimensions of each object.")
        self._pages.setCurrentWidget(self._state)


    def spec(self):
        """The :class:`spacr.embeddings.EmbeddingSpec` the controls describe."""
        from ...embeddings import CHANNEL_PROJECT, EmbeddingError, EmbeddingSpec

        foundation = str(self._foundation.currentData() or "")
        if foundation == "subcell_rybg":
            if self._subcell_channels is None:
                raise EmbeddingError(tr(
                    "Choose four distinct crop channels before running "
                    "SubCell's four-plane model."))
            return EmbeddingSpec(
                backbone=foundation,
                channel_policy=CHANNEL_PROJECT,
                channels=self._subcell_channels,
                batch_size=int(self._batch.value()),
                normalize=False,
            )
        return EmbeddingSpec(
            backbone=foundation or str(self._backbone.currentText()).strip(),
            channel_policy=str(self._policy.currentData()),
            batch_size=int(self._batch.value()),
        )

    def embed(self) -> None:
        """Runs the encoder in the background and shows the result."""
        crops = getattr(self, "_crops", None)
        if crops is None:
            self._status.setText("Load crops first.")
            return
        if self._foundation.currentData() == "subcell_rybg":
            if crops.shape[-1] < 4:
                self._status.setText(tr(
                    "SubCell's four-plane model needs crops with at least "
                    "four channels. Load a suitable crop source first."))
                return
            reason = self._subcell_mapping_error(
                self._subcell_channels, int(crops.shape[-1]))
            if reason:
                self._status.setText(reason)
                return
        spec = self.spec()
        record = self._scale_record
        self._status.setText(f"Embedding {crops.shape[0]} objects…")

        def work():
            """Run the backbone off the GUI thread.

            The import is inside because it pulls torch in: a user who never
            opens this screen should not pay for it, and a user who does
            should pay for it once, here, rather than at launch.
            """
            from ...embeddings import _embed_plate

            return _embed_plate(crops, spec, record=record)

        self._jobs.submit(work, self._on_embedded)

    def _on_embedded(self, result) -> None:
        """Fill the preview and say which encoder produced it."""
        self._result = result
        frame = pd.DataFrame(np.asarray(result.values),
                             columns=list(result.columns))
        self._frame = frame
        self._fill_preview(frame)

        from ...embeddings import encoder_entry

        entry = encoder_entry(getattr(result, "spec", None) or self.spec())
        digest = entry.sha256[:12] + "…" if entry.sha256 else "no checksum"
        self._status.setText(
            f"{len(frame)} objects x {len(frame.columns)} dimensions. "
            f"Encoder {entry.name}, weights {digest}.")
        if self._labels and len(self._labels) == len(frame):
            self._show_scorecard()

    def _fill_preview(self, frame: pd.DataFrame) -> None:
        """Show the first few dimensions, and only the first few.

        The whole matrix is the result and it is not readable; a table of
        2,048 columns would invite scrolling through it as though a column
        meant something.
        """
        columns = list(frame.columns)[:PREVIEW_DIMENSIONS]
        rows = min(len(frame), 50)
        self._pages.setCurrentWidget(self._table)
        self._table.setSortingEnabled(False)
        try:
            self._table.setRowCount(0)
            self._table.setColumnCount(len(columns))
            self._table.setHorizontalHeaderLabels(columns)
            self._table.setRowCount(rows)
            for row in range(rows):
                for index, column in enumerate(columns):
                    value = float(frame.iloc[row][column])
                    item = table_item(f"{value:.4f}", key=value)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self._table.setItem(row, index, item)
        finally:
            self._table.setSortingEnabled(True)

    def _on_job_failed(self, message: str) -> None:
        """A refusal is a sentence on the screen, never a silent empty table.

        One handler for both jobs, because both fail the same way as far as
        the reader is concerned. A load that failed also has to put the
        controls back: leaving Load reading "Stop" after the job it would
        have stopped has already died is a button that does nothing.

        A STOP IS NOT A FAILURE, and the one case that arrives here is a
        stop taken before the first page finished -- the loader has no
        crops to return and says so by raising. Heading that "Crops could
        not be loaded" tells a user their machine refused the thing they
        themselves just cancelled, so the stop is headed as a stop. It is
        told apart by the flag the user set, not by the message.
        """
        text = str(message)
        if self._loading and self._stop.is_set():
            self._stopped(text)
        elif self._loading:
            self._set_loading(False)
            self._refuse("Crops could not be loaded", text)
        else:
            self._status.setText(text)
        LOG.warning("embeddings job ended: %s", text)

    def _stopped(self, detail: str) -> None:
        """Put the controls back after a stop, and head it as one.

        Through :meth:`_refuse` for the body, because a stop unloads nothing
        either: whatever was on the screen before is still there, still
        encodable, and the panel has to say so.
        """
        self._set_loading(False)
        self._refuse("Loading stopped", detail)


    def is_busy(self) -> bool:
        """Whether a run is in flight.

        A crop load counts. It is the longer of the two jobs by far -- a
        plate of crops off a network mount is minutes -- and a screen that
        reported itself idle through it would let the window close over a
        live read.
        """
        return bool(self._jobs.active_jobs()) or self._loading

    def closeEvent(self, event):            # noqa: N802 - Qt name
        """Let the job runner stop its thread before the widget goes.

        The stop flag is set FIRST. ``shutdown`` waits a bounded time and
        then parks a worker that outlasts it, so a load told to stop between
        pages finishes in one page instead of reading the rest of the plate
        into an array whose screen has gone.

        :param event: Qt's close event, passed to the base class unchanged.
            Never ignored -- a screen that refuses to close because a run is
            in flight would trap the window, so the run is stopped instead.
        """
        self._stop.set()
        try:
            self._jobs.shutdown()
        except Exception:                    # noqa: BLE001
            pass
        super().closeEvent(event)


def make_embeddings_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return EmbeddingsScreen()


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
