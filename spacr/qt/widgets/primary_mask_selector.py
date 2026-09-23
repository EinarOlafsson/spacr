"""Choose a primary object class and mask source for secondary segmentation.

Primary masks load on a worker, with one queued replacement at most. A field
or source change invalidates the old snapshot immediately; late results never
become the source for another image. This widget reads masks and never saves.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bridge import drain_thread
from ..i18n import tr
from ..secondary_masks import read_primary_source


class _SourceWorker(QThread):
    """Decode, validate and checksum one primary source away from the GUI."""

    def __init__(self, token, request, parent=None):
        """Store a field/source generation and its immutable read arguments."""
        super().__init__(parent)
        self.token, self.request = token, request
        self.result, self.error, self.count = None, '', 0

    def run(self):
        """Return one owned source snapshot or a readable failure."""
        try:
            self.result = read_primary_source(**self.request)
            self.count = int(np.count_nonzero(np.unique(self.result.labels)))
        except Exception as error:
            self.error = str(error)


class PrimaryMaskSelector(QWidget):
    """Primary/secondary class selectors and an asynchronous primary-mask picker.

    :param parent: owning detection settings group.
    :ivar snapshot: validated PrimaryMaskSource, or None while loading/invalid.
    :ivar changed: emitted when the snapshot is invalidated or replaced.
    """

    changed = Signal()

    def __init__(self, parent=None):
        """Build class, file/folder and reload controls without reading a file."""
        super().__init__(parent)
        self.snapshot = None
        self.error = ''
        self._field = None
        self._bound_image = None
        self._serial = 0
        self._worker = None
        self._pending = None
        self._closed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.primary_class, self.secondary_class = QComboBox(), QComboBox()
        for box in (self.primary_class, self.secondary_class):
            for caption, value in ((tr('Nucleus'), 'nucleus'), (tr('Cell'), 'cell'),
                                   (tr('Pathogen'), 'pathogen')):
                box.addItem(caption, value)
            box.setEditable(True)
        self.secondary_class.setCurrentIndex(1)
        self.primary_class.setToolTip(tr(
            'Object type represented by the primary masks. Choose a class or enter a custom name. '
            'This labels the saved relationship; it does not resegment the source mask.'))
        self.secondary_class.setToolTip(tr(
            'Object type assigned to the grown masks. Choose a class or enter a custom name '
            'different from the primary class. Growth is controlled by the other settings.'))
        form.addRow(tr('Primary object class'), self.primary_class)
        form.addRow(tr('Secondary object class'), self.secondary_class)
        self.path = QLineEdit()
        self.path.setPlaceholderText(tr('Primary-mask file or folder'))
        form.addRow(tr('Primary masks'), self.path)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        for caption, callback in ((tr('File…'), self._choose_file),
                                  (tr('Folder…'), self._choose_folder),
                                  (tr('Reload'), self.reload)):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel(tr('Choose a primary mask from a different file than the editable output mask.'))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.path.editingFinished.connect(self._source_changed)
        self.primary_class.currentTextChanged.connect(self.reload)
        self.secondary_class.currentTextChanged.connect(self.reload)

    @staticmethod
    def _class_name(box):
        """Return the stable role key or an explicitly typed custom class name."""
        return str(box.currentData() if box.currentIndex() >= 0 and
                   box.currentText() == box.itemText(box.currentIndex()) else box.currentText()).strip()

    def bind_field(self, image_path, shape, output_path):
        """Invalidate the previous field and resolve this image's primary mask.

        :param image_path: source image identifying the current queue entry.
        :param shape: image height and width.
        :param output_path: editable mask path, checked against source aliases.
        """
        self._field = (str(image_path), tuple(shape), str(output_path))
        self.reload()

    def clear_field(self):
        """Drop a stale source when the image cannot be loaded or is closed."""
        self._field = None
        self.reload()

    def _source_changed(self):
        """Bind a newly chosen explicit file to the currently displayed image."""
        self._bound_image = self._field[0] if self._field is not None else None
        self.reload()

    def _choose_file(self):
        """Choose a primary mask for this one field; no data is written."""
        path, _ = QFileDialog.getOpenFileName(self, tr('Choose primary mask'), '',
                                             tr('Masks (*.tif *.tiff *.png *.npy)'))
        if path:
            self.path.setText(path)
            self._source_changed()

    def _choose_folder(self):
        """Choose a folder whose mask names match each queue image's stem."""
        path = QFileDialog.getExistingDirectory(self, tr('Choose primary-mask folder'))
        if path:
            self.path.setText(path)
            self._source_changed()

    def reload(self, *_args):
        """Invalidate first, then queue at most one replacement source read."""
        self._serial += 1
        self.snapshot = None
        self.error = ''
        self._pending = None
        self.changed.emit()
        if self._closed or self._field is None or not self.path.text().strip():
            self.status.setText(tr('Choose a primary mask from a different file than the editable output mask.'))
            return
        image, shape, output = self._field
        request = dict(source=self.path.text().strip(), image_path=image,
                       shape=shape, output_path=output,
                       primary_class=self._class_name(self.primary_class),
                       secondary_class=self._class_name(self.secondary_class),
                       bound_image=self._bound_image)
        self.status.setText(tr('Loading primary mask…'))
        self._pending = (self._serial, request)
        if self._worker is None:
            self._start_pending()

    def _start_pending(self):
        """Run the newest pending request without overlapping source workers."""
        if self._pending is None or self._closed:
            return
        token, request = self._pending
        self._pending = None
        self._worker = _SourceWorker(token, request, self)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _finished(self):
        """Publish only the source that still belongs to the selected field."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        if not self._closed and worker.token == self._serial:
            self.snapshot, self.error = worker.result, worker.error
            self.status.setText(tr('Primary mask unavailable: {error}', error=worker.error)
                                if worker.error else tr('{n} primary objects ready.', n=worker.count))
            self.changed.emit()
        worker.deleteLater()
        self._start_pending()

    def restore_source(self, record):
        """Restore a saved primary-source ledger record for the current image.

        :param record: mapping from PrimaryMaskSource.provenance. Its original
            image binding is retained; a wrong-field record is refused on read.
        """
        for box, field in ((self.primary_class, 'primary_class'),
                           (self.secondary_class, 'secondary_class')):
            old = box.blockSignals(True)
            value = str(record.get(field, ''))
            index = box.findData(value)
            if index >= 0:
                box.setCurrentIndex(index)
            else:
                box.setEditText(value)
            box.blockSignals(old)
        self.path.setText(str(record.get('selection', record.get('path', ''))))
        self._bound_image = record.get('image_path')
        self.reload()

    def shutdown(self):
        """Drain the active source worker before the owning screen is destroyed."""
        self._closed = True
        self._pending = None
        self._serial += 1
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.requestInterruption()
            drain_thread(worker, timeout_ms=5000)
