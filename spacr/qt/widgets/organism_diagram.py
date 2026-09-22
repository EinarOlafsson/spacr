"""Offline SwissBioPics cells with selectable UniProt compartment labels.

The Toxoplasma hyperLOPIT-to-SL mapping follows Einar Olafsson's Starplast
``celldiagram.py`` (MIT, revision 53703bafdf0604576a05b589400051a6b09e62ea).
Artwork is by Philippe Le Mercier, SIB, CC BY 4.0; source records accompany
the SVGs. Only the rendered copy loses hidden metadata and its credit logo;
the original asset and visible attribution are retained. No PyQt6 import,
network request or measured localization data is involved.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..i18n import tr
from ..preferences import scaled_px


COMPARTMENT_SL = {
    "rhoptries 1": "SL0233", "rhoptries 2": "SL0233",
    "micronemes": "SL0163", "dense granules": "SL0281",
    "apicoplast": "SL0018", "IMC": "SL0362",
    "mitochondrion - soluble": "SL0173",
    "mitochondrion - membranes": "SL0171",
    "Golgi": "SL0132", "ER": "SL0095", "ER 2": "SL0095",
    "nucleus - chromatin": "SL0191", "nucleus - non-chromatin": "SL0191",
    "nucleolus": "SL0188", "cytosol": "SL0091",
    "PM - integral": "SL0039", "PM - peripheral 1": "SL0039",
    "PM - peripheral 2": "SL0039", "tubulin cytoskeleton": "SL0090",
}
APICOMPLEXAN_LABELS = {
    "Rhoptries": "SL0233", "Micronemes": "SL0163", "Apicoplast": "SL0018",
    "Inner membrane complex": "SL0362", "Mitochondrion": "SL0173",
    "Golgi apparatus": "SL0132", "Endoplasmic reticulum": "SL0095",
    "Nucleus": "SL0191", "Nucleolus": "SL0188", "Cytosol": "SL0091",
    "Cell membrane": "SL0039", "Microtubule cytoskeleton": "SL0090",
}
YEAST_LABELS = {
    "Bud": "SL0027", "Bud neck": "SL0029", "Cell wall": "SL0041",
    "Cell membrane": "SL0039", "Nucleus": "SL0191", "Nucleolus": "SL0188",
    "Mitochondrion": "SL0173", "Golgi apparatus": "SL0132",
    "Endoplasmic reticulum": "SL0095", "Vacuole": "SL0272",
}
_SVG = "{http://www.w3.org/2000/svg}"
_COMPARTMENT_COLOURS = {
    "SL0091": "#eee6d9", "SL0233": "#df96af", "SL0163": "#e2bc70",
    "SL0281": "#b9a2ce", "SL0018": "#8ec7ae", "SL0362": "#9bc8cb",
    "SL0173": "#e8ae87", "SL0171": "#d18e6f", "SL0132": "#87bfcf",
    "SL0095": "#b0c4ce", "SL0191": "#b8b6d9", "SL0188": "#9998c6",
    "SL0039": "#9daec0", "SL0090": "#b7c88e", "SL0027": "#b5d2ba",
    "SL0029": "#99b5a8", "SL0041": "#d6c6a4", "SL0272": "#b4d4d4",
}


def _diagram_svg(source: str, selected: str = "") -> bytes:
    """Prepare bundled SVG markup and outline one selected SL group.

    :param source: original UTF-8 SwissBioPics SVG text.
    :param selected: SL identifier to accent; an empty string shows the cell.
    :returns: XML bytes suitable for QSvgRenderer; the source is unchanged.
    """
    selected = "SL0173" if selected == "SL0171" else selected
    root = ET.fromstring(source)
    canvas_width = float(root.get("viewBox").split()[2])
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.removeprefix(_SVG)
            if (tag in {"text", "a"} or child.get("id") == "sib_copyright"
                    or child.get("id") == "path_1_"
                    or (tag == "rect" and float(child.get("width", "0"))
                        >= canvas_width * 0.95)
                    or (tag == "g" and not len(child) and not child.get("id"))):
                parent.remove(child)
            elif tag == "path" and child.get("d", "").startswith("M0,0"):
                child.set("fill", "none")
                child.set("stroke", "none")
    for node in root.iter():
        colour = _COMPARTMENT_COLOURS.get(node.get("id"))
        if colour:
            for shape in node.iter():
                if (shape.tag.removeprefix(_SVG) in {
                        "path", "rect", "circle", "ellipse", "polygon"}
                        and shape.get("fill") not in {"none", "transparent"}):
                    shape.set("fill", colour)
    for node in root.iter():
        if node.get("id") == selected and selected:
            for shape in node.iter():
                if shape.tag.removeprefix(_SVG) in {
                        "path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}:
                    shape.set("stroke", "#00a896")
                    shape.set("stroke-width", "4")
    return ET.tostring(root, encoding="utf-8")


class _CellArtwork(QWidget):
    """Aspect-preserving SVG canvas with a bounded drawing height."""

    def __init__(self, source: str, parent=None):
        """Keep original markup and render it without hidden annotations."""
        super().__init__(parent)
        self.source = source
        self.renderer = QSvgRenderer(self)
        self.setMinimumHeight(scaled_px(300))
        self.setMaximumHeight(scaled_px(380))
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.select("")

    def select(self, location: str) -> None:
        """Highlight a UniProt SL identifier, or clear with an empty string."""
        self.renderer.load(QByteArray(_diagram_svg(self.source, location)))
        self.update()

    def paintEvent(self, event) -> None:
        """Fit the cell to the available rectangle without stretching it."""
        size = self.renderer.viewBoxF().size()
        scale = min(self.width() / size.width(), self.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        target = QRectF((self.width() - width) / 2, (self.height() - height) / 2,
                        width, height)
        painter = QPainter(self)
        self.renderer.render(painter, target)
        painter.end()


class OrganismDiagram(QWidget):
    """A cell illustration and keyboard-accessible compartment selector.

    :param app_key: organism key; only Toxoplasma uses hyperLOPIT labels.
    :param path: bundled SVG path; no external resource is fetched.
    :param parent: owning Qt widget.
    """

    def __init__(self, app_key: str, path: Path, parent=None):
        """Build the illustration, label selector and shared-shape caption."""
        super().__init__(parent)
        self.labels = (COMPARTMENT_SL if app_key == "toxoplasma" else
                       YEAST_LABELS if app_key == "candida" else APICOMPLEXAN_LABELS)
        self.setObjectName("OrganismDiagram")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.artwork = _CellArtwork(path.read_text(encoding="utf-8"), self)
        self.artwork.setAccessibleName(tr("Cell compartments"))
        layout.addWidget(self.artwork, 1)
        label = QLabel(tr("hyperLOPIT compartment") if app_key == "toxoplasma"
                       else tr("UniProt compartment"))
        self.selector = QComboBox(self)
        self.selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setMinimumContentsLength(12)
        self.selector.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.selector.setAccessibleName(label.text())
        label.setBuddy(self.selector)
        self.selector.addItem(tr("All compartments"), "")
        for name, location in self.labels.items():
            self.selector.addItem(tr(name), location)
        layout.addWidget(label)
        layout.addWidget(self.selector)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.caption)
        self.selector.currentIndexChanged.connect(self._select)
        self._select()

    def _select(self, index: int = 0) -> None:
        """Accent the selected location and disclose labels sharing its shape."""
        location = self.selector.currentData()
        self.artwork.select(location)
        if not location:
            self.caption.setText(tr("Select a label to highlight its compartment."))
            return
        shared = [tr(name) for name, code in self.labels.items() if code == location]
        text = f'{location[:2]}-{location[2:]} · ' + " / ".join(shared)
        if location == "SL0171":
            text += "\n" + tr("The membrane label uses the mitochondrial outline; it has no separate shape.")
        self.caption.setText(text)
