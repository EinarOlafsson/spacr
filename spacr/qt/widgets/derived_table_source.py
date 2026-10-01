"""Shared derived-table actions for plotting and gating screens."""
from PySide6.QtWidgets import QDialog, QPushButton

from ...derived_tables import execute, save_definition, schemas


class DerivedTableSource:
    """Add named merges to an existing table screen without owning its loader."""

    def _install_merge_button(self, layout):
        """Place the merge action immediately after the screen's table picker.

        :param layout: Header layout containing the table picker.
        """
        self._merge_definition = None
        self._merge_button = QPushButton("Merge tables", self)
        self._merge_button.setObjectName("MergeTablesButton")
        self._merge_button.setToolTip("Combine measurements into a named table with a validated preview")
        self._merge_button.setEnabled(False)
        self._merge_button.clicked.connect(self.open_merge_dialog)
        layout.addWidget(self._merge_button)

    def open_merge_dialog(self):
        """Create a validated persistent result and load it through the normal path."""
        from .merge_tables_dialog import MergeTablesDialog
        if not self._path:
            return
        try:
            available = schemas(self._path)
            selected = [t for t in getattr(self, "_tables", [self._table_picker.currentText()])
                        if t in available]
            kwargs = {"selected": selected}
            if self._merge_definition:
                kwargs["initial_definition"] = self._merge_definition
            dialog = MergeTablesDialog(self._path, self, **kwargs)
            if dialog.exec() == QDialog.Accepted:
                self.use_derived_table(dialog.definition)
        except Exception as exc:
            self._source.setText(f"Could not merge tables: {exc}")

    def use_derived_table(self, definition):
        """Validate and persist a definition, then use the screen's ordinary loader.

        :param definition: Named definition bound to the currently loaded database.
        """
        source = self._path
        self._jobs.cancel()
        self._source.setText("Creating merged table…")

        def work():
            """Validate and persist the source-bound definition on a worker."""
            frame, _report = execute(source, definition)
            return save_definition(source, frame.attrs.get("merge_definition", definition))

        self._jobs.submit(work, lambda name: self.load_path(source, table=name))

    def _derived_frame_loaded(self, frame):
        """Retain reproduction metadata and update image-only actions.

        :param frame: Newly loaded source or derived frame.
        """
        self._merge_definition = frame.attrs.get("merge_definition")
        database = bool(self._path and not str(self._path).lower().endswith((".csv", ".tsv", ".txt")))
        self._merge_button.setEnabled(database and len(getattr(self, "_paths", [])) <= 1)
        can_link = not self._merge_definition or frame.attrs.get("image_provenance", False)
        for name in ("_annotate", "_export", "_to_annotate"):
            button = getattr(self, name, None)
            if button is not None:
                enabled = bool(can_link)
                if name == "_to_annotate":
                    enabled = enabled and self.builder.canvas.selected_count() > 0
                button.setEnabled(enabled)
                if not can_link:
                    button.setToolTip("Image linking is unavailable without verified spaCR image/object provenance; tabular plotting and gating remain available.")

    def _has_merge_image_provenance(self):
        """Whether image-specific actions can safely link the current derived rows."""
        return not self._merge_definition or bool(
            self._frame is not None and self._frame.attrs.get("image_provenance", False))
