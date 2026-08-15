"""Dedicated workbenches for CV explanation and hit investigation."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...hit_attribution import promote_hit_calls, undo_hit_promotion
from ...hit_investigation import investigate_hit
from ...surrogate import (MODEL_FAMILIES, available_backends,
                          explain_classifier, write_surrogate_result)
from ..job_runner import JobRunner
from ..linked_selection import has_object_opener, open_objects
from ..theme import SPACING, mark_surface
from .app_screen import ModuleHeader

APP_KEY = "explain_cv"
APP_NAME = "Explain CV Model"
APP_DESCRIPTION = (
    "Explain CV decisions with measured features, then resolve screen hits "
    "to candidate cells")
APP_INTRO = (
    "Explain what an existing vision classifier responds to using measured "
    "area, intensity and texture, held-out fidelity, permutation importance "
    "and SHAP. Then investigate a regression hit across guide-containing and "
    "matched control wells without treating a well-level read fraction as a "
    "direct single-cell barcode.")
APP_CLI_NOTE = (
    "Model Explanation is an interactive workbench. Headless callers use "
    "spacr.surrogate.explain_classifier and spacr.hit_attribution directly.")


def _read_only_item(value: Any) -> QTableWidgetItem:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = ""
    elif isinstance(value, float):
        text = f"{value:.5g}"
    else:
        text = str(value)
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _fill_table(table: QTableWidget, frame: pd.DataFrame,
                *, limit: int = 1000) -> None:
    shown = frame.head(limit).copy()
    if not isinstance(shown.index, pd.RangeIndex):
        shown = shown.reset_index()
    table.setSortingEnabled(False)
    table.clear()
    table.setColumnCount(len(shown.columns))
    table.setHorizontalHeaderLabels([str(column) for column in shown.columns])
    table.setRowCount(len(shown))
    for row_index, row in enumerate(shown.itertuples(index=False, name=None)):
        for column_index, value in enumerate(row):
            table.setItem(row_index, column_index, _read_only_item(value))
    table.setSortingEnabled(True)
    if shown.shape[1]:
        table.horizontalHeader().setSectionResizeMode(
            shown.shape[1] - 1, QHeaderView.Stretch)


class _PathRow(QWidget):
    def __init__(self, *, folder: bool = False, parent=None):
        super().__init__(parent)
        self.folder = bool(folder)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.button = QPushButton("Browse…", self)
        self.button.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit parity
        self.edit.setText(str(value or ""))

    def _browse(self) -> None:  # pragma: no cover - modal native picker
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, "Choose folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose file", "", "Data (*.csv *.db *.sqlite);;All files (*)")
        if path:
            self.setText(path)


class ExplainCvPanel(QWidget):
    """Run and render one provenance-bearing surrogate explanation."""

    def __init__(self, host=None, parent=None):
        super().__init__(parent)
        self.host = host
        self.result = None
        self._jobs = JobRunner(self, threaded=True, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._failed)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.database = _PathRow()
        self.predictions = _PathRow()
        self.predictions.edit.editingFinished.connect(
            self._refresh_prediction_columns)
        self.output = _PathRow(folder=True)
        self.path_column = QComboBox(); self.path_column.setEditable(False)
        self.path_column.addItems(["path", "png_path", "image_path"])
        self.prediction_column = QComboBox(); self.prediction_column.setEditable(False)
        self.prediction_column.addItems(["pred", "prediction", "class"])
        self.backend = QComboBox(); self.backend.setEditable(False)
        availability = available_backends()
        for key, label in MODEL_FAMILIES.items():
            self.backend.addItem(label, key)
            index = self.backend.count() - 1
            info = availability.get(key, {})
            if not info.get("available", False):
                model_item = self.backend.model().item(index)
                if model_item is not None:
                    model_item.setEnabled(False)
                self.backend.setItemData(index, info.get("reason", "Unavailable"), Qt.ToolTipRole)
        self.split = QComboBox(); self.split.addItems(["well", "plate"])
        form.addRow("Measurements database", self.database)
        form.addRow("Existing CV predictions", self.predictions)
        form.addRow("Crop path column", self.path_column)
        form.addRow("Prediction column", self.prediction_column)
        form.addRow("Surrogate model", self.backend)
        form.addRow("Held-out grouping", self.split)
        form.addRow("Output folder", self.output)
        outer.addLayout(form)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Explain model")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run)
        self.open_objects_button = QPushButton("Open held-out objects")
        self.open_objects_button.clicked.connect(self.open_held_out_objects)
        self.open_objects_button.setEnabled(False)
        self.activation_button = QPushButton("Open Activation Maps")
        self.activation_button.clicked.connect(self.open_activation_maps)
        actions.addWidget(self.run_button)
        actions.addWidget(self.open_objects_button)
        actions.addWidget(self.activation_button)
        actions.addStretch(1)
        outer.addLayout(actions)
        self.status = QLabel(
            "Choose predictions already produced by the CV model. The model is not rerun here.")
        self.status.setWordWrap(True); self.status.setObjectName("Muted")
        outer.addWidget(self.status)

        self.results = QTabWidget()
        self.summary = QPlainTextEdit(); self.summary.setReadOnly(True)
        self.importance = QTableWidget(); self.metrics = QTableWidget()
        self.confusion = QTableWidget(); self.shap = QTableWidget()
        self.correlations = QTableWidget(); self.held_out = QTableWidget()
        self.distributions = QTableWidget()
        self._distribution_frame = pd.DataFrame()
        for table in (self.importance, self.metrics, self.confusion, self.shap,
                      self.correlations, self.held_out, self.distributions):
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setAlternatingRowColors(True); mark_surface(table)
        self.results.addTab(self.summary, "Fidelity")
        self.results.addTab(self.importance, "Importance")
        self.results.addTab(self.metrics, "Class metrics")
        self.results.addTab(self.confusion, "Confusion")
        self.results.addTab(self.shap, "Per-cell SHAP")
        self.results.addTab(self.correlations, "Correlations")
        self.results.addTab(self.distributions, "Feature distributions")
        self.results.addTab(self.held_out, "Held-out cells")
        self.importance.itemSelectionChanged.connect(
            self._show_selected_feature_distribution)
        outer.addWidget(self.results, 1)

    def _refresh_prediction_columns(self) -> None:
        """Populate column dropdowns from the selected prediction artifact."""
        path = self.predictions.text()
        if not path or not os.path.isfile(path):
            return
        try:
            columns = [str(value) for value in pd.read_csv(path, nrows=0).columns]
        except Exception as exc:
            self.status.setText(f"Could not read prediction columns: {exc}")
            return
        previous_path = self.path_column.currentText()
        previous_prediction = self.prediction_column.currentText()
        path_candidates = [column for column in columns if "path" in column.lower()]
        prediction_candidates = [column for column in columns
                                 if column not in path_candidates]
        for combo, candidates, previous in (
            (self.path_column, path_candidates or columns, previous_path),
            (self.prediction_column, prediction_candidates or columns,
             previous_prediction),
        ):
            combo.clear(); combo.addItems(candidates)
            preferred = previous if previous in candidates else next(
                (name for name in ("path", "png_path", "image_path", "pred",
                                   "prediction", "class") if name in candidates),
                candidates[0] if candidates else "")
            combo.setCurrentText(preferred)

    def run_analysis(self, database: str, predictions: str, *,
                     path_column: str = "path", prediction_column: str = "pred",
                     model_family: str = "random_forest", split_by: str = "well",
                     output: str = ""):
        prediction_frame = pd.read_csv(predictions)
        result = explain_classifier(
            database, prediction_frame, path_column=path_column,
            prediction_column=prediction_column, model_family=model_family,
            split_by=split_by, verbose=False)
        paths = write_surrogate_result(result, output) if output else {}
        return result, paths

    def run(self) -> None:
        database, predictions = self.database.text(), self.predictions.text()
        if not database or not predictions:
            self._failed("Choose a measurements database and predictions CSV.")
            return
        self.run_button.setEnabled(False)
        self.status.setText("Joining predictions to measured objects and fitting held-out surrogate…")
        kwargs = {
            "database": database, "predictions": predictions,
            "path_column": self.path_column.currentText(),
            "prediction_column": self.prediction_column.currentText(),
            "model_family": str(self.backend.currentData()),
            "split_by": self.split.currentText(), "output": self.output.text(),
        }
        self._jobs.submit(lambda: self.run_analysis(**kwargs), self._loaded)

    def _loaded(self, payload) -> None:
        self.result, paths = payload
        self.run_button.setEnabled(True)
        self.summary.setPlainText(self.result.summary())
        _fill_table(self.importance,
                    self.result.importance if self.result.is_faithful else
                    self.result.importance.iloc[0:0])
        self.results.setTabText(
            1, "Importance" if self.result.is_faithful else "Importance (withheld)")
        _fill_table(self.metrics, self.result.class_metrics)
        _fill_table(self.confusion, self.result.confusion)
        _fill_table(self.shap, self.result.shap_values)
        _fill_table(self.correlations, self.result.correlated_features)
        self._distribution_frame = self.result.feature_distributions.copy()
        _fill_table(self.distributions, self._distribution_frame)
        _fill_table(self.held_out, self.result.held_out)
        self.open_objects_button.setEnabled(
            has_object_opener("annotate") and not self.result.held_out.empty)
        suffix = f" Saved {len(paths)} artifacts." if paths else ""
        self.status.setText(
            f"Held-out fidelity {self.result.fidelity:.3f}; majority baseline "
            f"{self.result.baseline:.3f}.{suffix}")

    def _show_selected_feature_distribution(self) -> None:
        """Link an importance selection to its held-out class distribution."""
        if self._distribution_frame.empty:
            return
        selected = self.importance.selectedItems()
        if not selected:
            _fill_table(self.distributions, self._distribution_frame)
            return
        header = [
            self.importance.horizontalHeaderItem(index).text()
            for index in range(self.importance.columnCount())]
        try:
            feature_column = header.index("feature")
        except ValueError:
            return
        item = self.importance.item(selected[0].row(), feature_column)
        if item is None:
            return
        feature = item.text()
        _fill_table(
            self.distributions,
            self._distribution_frame.loc[
                self._distribution_frame["feature"].astype(str) == feature])

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(f"Could not explain model: {message}")

    def open_held_out_objects(self) -> None:
        if self.result is None or self.result.held_out.empty:
            return
        try:
            open_objects(self.result.held_out, reason="Held-out surrogate objects",
                         source=APP_KEY)
        except Exception as exc:
            self._failed(str(exc))

    def open_activation_maps(self) -> None:
        if self.host is not None:
            self.host._on_train_requested("activation_maps", {})

    def closeEvent(self, event) -> None:  # noqa: N802
        self._jobs.shutdown()
        super().closeEvent(event)


class InvestigateHitPanel(QWidget):
    """Guide-fraction-aware, cross-fitted candidate-cell investigation."""

    def __init__(self, host=None, parent=None):
        super().__init__(parent)
        self.host = host
        self.result = None
        self.investigation = None
        self.attribution_run_id = ""
        self.promotion_id = ""
        self._jobs = JobRunner(self, threaded=True, app_key="investigate_hit")
        self._jobs.job_failed.connect(self._failed)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.database = _PathRow(); self.predictions = _PathRow(); self.fractions = _PathRow()
        self.predictions.edit.editingFinished.connect(
            self._refresh_prediction_columns)
        self.regression_folder = _PathRow(folder=True)
        self.gene = QLineEdit(); self.guides = QLineEdit()
        self.guides.setPlaceholderText("EAF1_g1, EAF1_g2")
        self.score = QComboBox(); self.score.setEditable(False)
        self.score.addItems(["prediction", "pred", "class"])
        self.features = QLineEdit()
        self.features.setPlaceholderText("blank = safe numeric morphology features")
        self.direction = QComboBox(); self.direction.addItems(["positive", "negative"])
        self.annotation = QLineEdit("hit_like")
        form.addRow("Measurements database", self.database)
        form.addRow("Existing CV predictions", self.predictions)
        form.addRow("Well/guide fractions", self.fractions)
        form.addRow("Regression results folder", self.regression_folder)
        form.addRow("Selected gene", self.gene)
        form.addRow("Target guides", self.guides)
        form.addRow("Phenotype score column", self.score)
        form.addRow("Effect direction", self.direction)
        form.addRow("Independent morphology features", self.features)
        form.addRow("Promotion annotation column", self.annotation)
        outer.addLayout(form)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Investigate hit"); self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run)
        self.open_button = QPushButton("Open candidate crops"); self.open_button.clicked.connect(self.open_candidates)
        self.promote_button = QPushButton("Promote calls to annotation"); self.promote_button.clicked.connect(self.promote)
        self.undo_button = QPushButton("Undo promotion"); self.undo_button.clicked.connect(self.undo)
        self.umap_button = QPushButton("Compare in Image UMAP"); self.umap_button.clicked.connect(self.open_umap)
        for button in (self.open_button, self.promote_button, self.undo_button, self.umap_button):
            button.setEnabled(False)
        for button in (self.run_button, self.open_button, self.promote_button,
                       self.undo_button, self.umap_button):
            actions.addWidget(button)
        actions.addStretch(1); outer.addLayout(actions)
        self.status = QLabel(
            "Candidate probabilities are weakly supervised hit-like morphology, not observed guide identity.")
        self.status.setWordWrap(True); self.status.setObjectName("Muted")
        outer.addWidget(self.status)
        self.tabs = QTabWidget()
        self.summary = QPlainTextEdit(); self.summary.setReadOnly(True)
        self.well_table = QTableWidget(); self.cell_table = QTableWidget(); self.guide_table = QTableWidget()
        self.threshold_table = QTableWidget()
        self.embedding_table = QTableWidget(); self.gallery_table = QTableWidget()
        for table in (self.well_table, self.cell_table, self.guide_table,
                      self.threshold_table, self.embedding_table,
                      self.gallery_table):
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setAlternatingRowColors(True); mark_surface(table)
        self.tabs.addTab(self.summary, "Evidence")
        self.tabs.addTab(self.well_table, "Wells")
        self.tabs.addTab(self.guide_table, "Guides")
        self.tabs.addTab(self.threshold_table, "Threshold sensitivity")
        self.tabs.addTab(self.cell_table, "Candidate cells")
        self.tabs.addTab(self.embedding_table, "Control-fitted embedding")
        self.tabs.addTab(self.gallery_table, "Blinded review gallery")
        outer.addWidget(self.tabs, 1)

    def configure_hit(self, *, folder: str = "", gene: str = "",
                      effect: float = 0.0, guides: Sequence[str] = (),
                      fdr: float = float("nan"), phenotype: str = "",
                      guide_agreement: float = float("nan"),
                      n_guides: int = 0, well_support: int = 0) -> None:
        self.regression_folder.setText(folder)
        self.gene.setText(gene)
        self.guides.setText(", ".join(str(value) for value in guides))
        self.direction.setCurrentText("positive" if float(effect) >= 0 else "negative")
        if phenotype:
            if self.score.findText(phenotype) < 0:
                self.score.addItem(phenotype)
            self.score.setCurrentText(phenotype)
        self.gene.setProperty("source_fdr", fdr)
        self.gene.setProperty("source_effect", effect)
        self.gene.setProperty("source_guide_agreement", guide_agreement)
        self.gene.setProperty("source_n_guides", n_guides)
        self.gene.setProperty("source_well_support", well_support)

    def _refresh_prediction_columns(self) -> None:
        path = self.predictions.text()
        if not path or not os.path.isfile(path):
            return
        try:
            columns = [str(value) for value in pd.read_csv(path, nrows=0).columns]
        except Exception as exc:
            self._failed(f"could not read prediction columns: {exc}")
            return
        previous = self.score.currentText()
        candidates = [column for column in columns
                      if "path" not in column.lower() and column != "prcfo"]
        self.score.clear(); self.score.addItems(candidates)
        preferred = previous if previous in candidates else next(
            (name for name in ("prediction", "pred", "class", "score")
             if name in candidates), candidates[0] if candidates else "")
        self.score.setCurrentText(preferred)

    def run_analysis(self, *, database: str, predictions: str, fractions: str,
                     gene: str, guides: Sequence[str], score: str,
                     direction: str, features: Sequence[str], folder: str):
        source_fdr = self.gene.property("source_fdr")
        source_agreement = self.gene.property("source_guide_agreement")
        return investigate_hit({
            "db_path": database,
            "predictions_file": predictions,
            "guide_fractions_file": fractions,
            "results_folder": folder,
            "target_gene": gene,
            "target_guides": list(guides),
            "score_column": score,
            "hit_phenotype": score,
            "hit_effect": float(self.gene.property("source_effect") or 0),
            "hit_fdr": (float(source_fdr) if source_fdr is not None
                        else float("nan")),
            "hit_guide_agreement": (
                float(source_agreement) if source_agreement is not None
                else float("nan")),
            "hit_n_guides": int(self.gene.property("source_n_guides") or 0),
            "hit_well_support": int(
                self.gene.property("source_well_support") or 0),
            "hit_direction": direction,
            "hit_feature_columns": list(features),
            "hit_store_database": True,
            "verbose": False,
        })

    def run(self) -> None:
        guides = [value.strip() for value in self.guides.text().split(",") if value.strip()]
        features = [value.strip() for value in self.features.text().split(",") if value.strip()]
        required = [self.database.text(), self.predictions.text(), self.fractions.text(),
                    self.regression_folder.text(), self.gene.text().strip(),
                    self.score.currentText().strip()]
        if not all(required) or not guides:
            self._failed(
                "Choose database, prediction/fraction tables, regression "
                "folder, gene, guides and score column.")
            return
        self.run_button.setEnabled(False); self.status.setText("Cross-fitting candidate cells…")
        kwargs = {
            "database": self.database.text(), "predictions": self.predictions.text(),
            "fractions": self.fractions.text(), "gene": self.gene.text().strip(),
            "guides": guides, "score": self.score.currentText().strip(),
            "direction": self.direction.currentText(), "features": features,
            "folder": self.regression_folder.text(),
        }
        self._jobs.submit(lambda: self.run_analysis(**kwargs), self._loaded)

    def _loaded(self, payload) -> None:
        self.result = payload["result"]
        self.investigation = payload
        self.attribution_run_id = payload["attribution_run_id"]
        self.run_button.setEnabled(True)
        self.summary.setPlainText(self.result.summary() + "\n\n" + "\n".join(
            f"{key}: {value}" for key, value in self.result.validation.items()))
        _fill_table(self.well_table, self.result.wells)
        _fill_table(self.guide_table, self.result.guide_evidence)
        _fill_table(self.threshold_table, self.result.threshold_sensitivity)
        _fill_table(self.embedding_table, payload["embedding"])
        _fill_table(self.gallery_table, payload["gallery"])
        ordered = self.result.cells.sort_values("hit_like_probability", ascending=False)
        display = [column for column in (
            *self.result.object_columns, "target_guide_fraction",
            self.result.score_column, "candidate_rank", "hit_like_probability",
            "hit_like_uncertainty", "hit_like_call", "attribution_fold")
            if column in ordered.columns]
        _fill_table(self.cell_table, ordered[display])
        self.open_button.setEnabled(has_object_opener("annotate") and not ordered.empty)
        self.promote_button.setEnabled(True)
        self.umap_button.setEnabled(False)
        self.status.setText(
            f"Stored {len(ordered):,} versioned cell probabilities. "
            "Promotion remains an explicit reversible step.")

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True); self.status.setText(f"Could not investigate hit: {message}")

    def open_candidates(self) -> None:
        if self.result is None:
            return
        selected = self.result.cells.sort_values("hit_like_probability", ascending=False).head(200)
        try:
            open_objects(selected, reason=f"Top {self.result.target_gene}-hit-like candidates",
                         source=APP_KEY,
                         context={"scores": dict(zip(
                             selected.get("prcfo", selected.index).astype(str),
                             selected["hit_like_probability"]))})
        except Exception as exc:
            self._failed(str(exc))

    def promote(self) -> None:
        if not self.attribution_run_id or not self.annotation.text().strip():
            return
        try:
            self.promotion_id = promote_hit_calls(
                self.database.text(), self.result,
                run_id=self.attribution_run_id,
                annotation_column=self.annotation.text().strip())
        except Exception as exc:
            self._failed(str(exc)); return
        self.undo_button.setEnabled(True); self.umap_button.setEnabled(True)
        self.status.setText(
            f"Promoted calls to {self.annotation.text().strip()!r}; the previous values are retained for Undo.")

    def undo(self) -> None:
        if not self.promotion_id:
            return
        try:
            count = undo_hit_promotion(self.database.text(), self.promotion_id)
        except Exception as exc:
            self._failed(str(exc)); return
        self.undo_button.setEnabled(False); self.umap_button.setEnabled(False)
        self.status.setText(f"Restored {count:,} previous annotation values.")

    def open_umap(self) -> None:
        if self.host is None:
            return
        source = os.path.dirname(self.database.text())
        self.host._on_train_requested(
            "umap", {"src": source, "color_by": self.annotation.text().strip()})

    def closeEvent(self, event) -> None:  # noqa: N802
        self._jobs.shutdown(); super().closeEvent(event)


class ModelExplanationScreen(QWidget):
    """The screen that explains a computer-vision model's decisions.

    Wraps :class:`ExplainCvPanel` in the standard module chrome. The header's
    instruction is the order the panel enforces -- fidelity FIRST, then
    importance -- because an importance ranking read off a model that does
    not fit is a ranking of nothing, and it looks identical to a good one.

    :param host: the main window, for screen navigation.
    :param parent: Qt parent.
    """

    def __init__(self, host=None, parent=None):
        super().__init__(parent)
        self.host = host
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.addWidget(ModuleHeader(
            APP_NAME, description=APP_DESCRIPTION,
            instruction="Start with fidelity; only then interpret importance or candidate cells."))
        self.explain = ExplainCvPanel(host=host)
        outer.addWidget(self.explain, 1)
        from ..dnd import install_for
        install_for(self, "explain_cv")


class InvestigateHitScreen(QWidget):
    """Dedicated post-regression screen with explicit promotion and undo."""

    def __init__(self, host=None, parent=None):
        super().__init__(parent)
        self.host = host
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.addWidget(ModuleHeader(
            "Investigate Hit",
            description=(
                "Resolve an exact regression hit to cross-fitted hit-like cells "
                "and well-level quantitative evidence."),
            instruction=(
                "Candidate probabilities are not observed guide identities; "
                "promotion is explicit, fresh-column only, and reversible.")))
        self.investigate = InvestigateHitPanel(host=host)
        outer.addWidget(self.investigate, 1)
        from ..dnd import install_for
        install_for(self, "investigate_hit")

    def configure_hit(self, **request) -> None:
        self.investigate.configure_hit(**request)

    def apply_seed(self, seed: Dict[str, Any]) -> None:
        """Accept the normal MainWindow hand-off from Hit List."""
        self.configure_hit(
            folder=str(seed.get("results_folder", "")),
            gene=str(seed.get("target_gene", "")),
            effect=float(seed.get("hit_effect", 0.0) or 0.0),
            guides=tuple(seed.get("target_guides", ())),
            fdr=float(seed.get("hit_fdr", float("nan"))),
            phenotype=str(seed.get("hit_phenotype", "")),
            guide_agreement=float(seed.get(
                "hit_guide_agreement", float("nan"))),
            n_guides=int(seed.get("hit_n_guides", 0) or 0),
            well_support=int(seed.get("hit_well_support", 0) or 0),
        )


def make_model_explanation_screen(app_key: Optional[str] = None,
                                  host=None) -> QWidget:
    """Build the model-explanation screen, for the app registry.

    :param app_key: accepted and unused -- the registry calls every factory
        with the key it registered, and this screen serves exactly one.
    :param host: the main window, passed through for navigation.
    :returns: a new :class:`ModelExplanationScreen`.
    """
    return ModelExplanationScreen(host=host)


def make_investigate_hit_screen(app_key: Optional[str] = None,
                                host=None) -> QWidget:
    """Build the hit-investigation screen, for the app registry.

    :param app_key: accepted and unused, as above.
    :param host: the main window, passed through for navigation.
    :returns: a new :class:`InvestigateHitScreen`.
    """
    return InvestigateHitScreen(host=host)
