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
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np

from PySide6.QtCore import QByteArray, QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QVBoxLayout, QWidget

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
"""Toxoplasma hyperLOPIT display labels mapped to UniProt SL identifiers.

Several labels share a location. The mitochondrial membrane label retains
``SL0171`` as its identifier and uses the ``SL0173`` outline when drawn.
"""
APICOMPLEXAN_LABELS = {
    "Rhoptries": "SL0233", "Micronemes": "SL0163", "Apicoplast": "SL0018",
    "Inner membrane complex": "SL0362", "Mitochondrion": "SL0173",
    "Golgi apparatus": "SL0132", "Endoplasmic reticulum": "SL0095",
    "Nucleus": "SL0191", "Nucleolus": "SL0188", "Cytosol": "SL0091",
    "Cell membrane": "SL0039", "Microtubule cytoskeleton": "SL0090",
}
"""Shared apicomplexan compartment labels mapped to UniProt SL identifiers."""
YEAST_LABELS = {
    "Bud": "SL0027", "Bud neck": "SL0029", "Cell wall": "SL0041",
    "Cell membrane": "SL0039", "Nucleus": "SL0191", "Nucleolus": "SL0188",
    "Mitochondrion": "SL0173", "Golgi apparatus": "SL0132",
    "Endoplasmic reticulum": "SL0095", "Vacuole": "SL0272",
}
"""Generic budding-yeast compartment labels mapped to UniProt SL identifiers."""
_SVG = "{http://www.w3.org/2000/svg}"
_COMPARTMENT_COLOURS = {
    "SL0091": "#eee6d9", "SL0233": "#df96af", "SL0163": "#e2bc70",
    "SL0281": "#b9a2ce", "SL0018": "#8ec7ae", "SL0362": "#9bc8cb",
    "SL0173": "#e8ae87", "SL0171": "#d18e6f", "SL0132": "#87bfcf",
    "SL0095": "#b0c4ce", "SL0191": "#b8b6d9", "SL0188": "#9998c6",
    "SL0039": "#9daec0", "SL0090": "#b7c88e", "SL0027": "#b5d2ba",
    "SL0029": "#99b5a8", "SL0041": "#d6c6a4", "SL0272": "#b4d4d4",
}


def _clean_tree(source: str):
    """Remove metadata and canvas decoration while retaining compartment geometry."""
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
    return root


def _render_tree(root, *, portrait: bool, solid: bool = False) -> bytes:
    """Build white outlines or solid hit geometry, with the apical end upward."""
    root = deepcopy(root)
    for node in root.iter():
        if node.tag.removeprefix(_SVG) not in {
                "path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}:
            continue
        if node.get("d", "").startswith("M0,0"):
            node.set("style", "fill:none;stroke:none")
            continue
        fill = "#ffffff" if solid and node.get("fill") not in {"none", "transparent"} else "none"
        node.set("style", f"fill:{fill};stroke:#ffffff;stroke-width:1.4;stroke-opacity:1;fill-opacity:1")
        node.set("fill", fill)
        node.set("stroke", "#ffffff")
    if portrait:
        x, y, width, height = map(float, root.get("viewBox").split())
        group = ET.Element(_SVG + "g", {"transform": f"rotate(90) translate({-x:g} {-(y + height):g})"})
        for child in list(root):
            root.remove(child)
            group.append(child)
        root.append(group)
        root.set("viewBox", f"0 0 {height:g} {width:g}")
    return ET.tostring(root, encoding="utf-8")


def _diagram_svg(source: str, *, portrait: bool = True) -> bytes:
    """Return upright white outlines.

    Interactive fills are painted from compartment masks by _CellArtwork,
    so no selection changes or replaces the original bundled artwork.
    """
    return _render_tree(_clean_tree(source), portrait=portrait)


class _CellArtwork(QWidget):
    """White compartment outlines, colored selections and pixel-accurate hover."""

    hovered = Signal(str)
    clicked = Signal(str)

    def __init__(self, source: str, parent=None, *, portrait=True, locations=()):
        """Keep original markup and render it without hidden annotations."""
        super().__init__(parent)
        self.source = source
        self.portrait = portrait
        self.tree = _clean_tree(source)
        self.locations = tuple(dict.fromkeys("SL0173" if code == "SL0171" else code for code in locations))
        self.selected = set()
        self.hover_location = ""
        self._masks = {}
        self._masks_size = None
        self.renderer = QSvgRenderer(self)
        self.renderer.load(QByteArray(_render_tree(self.tree, portrait=portrait)))
        self.setMinimumHeight(scaled_px(360))
        self.setMaximumHeight(scaled_px(560))
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMouseTracking(True)

    def select(self, locations) -> None:
        """Keep any number of SL locations filled until they are deselected."""
        if isinstance(locations, str):
            locations = [locations] if locations else []
        self.selected = {"SL0173" if code == "SL0171" else code for code in locations}
        self.update()

    def _target(self) -> QRectF:
        """Fit the rotated artwork inside the rounded panel with a small margin."""
        size = self.renderer.viewBoxF().size()
        scale = min(max(1, self.width()-24) / size.width(), max(1, self.height()-24) / size.height())
        width, height = size.width() * scale, size.height() * scale
        return QRectF((self.width()-width)/2, (self.height()-height)/2, width, height)

    def masks(self) -> dict:
        """Cache each compartment's real painted pixels, not its bounding box."""
        size = (self.width(), self.height())
        if self._masks_size == size:
            return self._masks
        groups = {node.get("id"): node for node in self.tree.iter()}
        masks = {}
        for code in self.locations:
            if code not in groups:
                continue
            root = ET.Element(self.tree.tag, self.tree.attrib)
            for defs in self.tree.findall(_SVG + "defs"):
                root.append(deepcopy(defs))
            root.append(deepcopy(groups[code]))
            renderer = QSvgRenderer(QByteArray(_render_tree(root, portrait=self.portrait, solid=True)))
            picture = QImage(max(1, size[0]), max(1, size[1]), QImage.Format_RGBA8888)
            picture.fill(Qt.transparent)
            painter = QPainter(picture)
            renderer.render(painter, self._target())
            painter.end()
            pixels = np.frombuffer(picture.bits(), dtype=np.uint8).reshape(picture.height(), picture.bytesPerLine())
            covered = pixels[:, 3:picture.width()*4:4] > 40
            masks[code] = (picture, covered.copy(), int(covered.sum()))
        self._masks, self._masks_size = masks, size
        return masks

    def organelle_at(self, x: float, y: float) -> str:
        """Hit the smallest actual shape under the pointer, including nested organelles."""
        x, y = int(x), int(y)
        if not (0 <= x < self.width() and 0 <= y < self.height()):
            return ""
        hits = [(area, code) for code, (_, covered, area) in self.masks().items() if covered[y, x]]
        return min(hits)[1] if hits else ""

    def set_hover(self, location: str) -> None:
        """Update transient highlighting without changing persistent selections."""
        location = "SL0173" if location == "SL0171" else location
        if self.hover_location != location:
            self.hover_location = location
            self.hovered.emit(location)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        """Link the compartment under the pointer to the legend below."""
        self.set_hover(self.organelle_at(event.position().x(), event.position().y()))

    def leaveEvent(self, event) -> None:
        """Clear transient color while keeping checked compartments filled."""
        self.set_hover("")
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Toggle the clicked anatomical component without clearing other selections."""
        if event.button() == Qt.LeftButton:
            location = self.organelle_at(event.position().x(), event.position().y())
            if location:
                self.clicked.emit(location)

    def paintEvent(self, event) -> None:
        """Paint colored masks under white outlines on a translucent black panel."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.setBrush(QColor(0, 0, 0, 215))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), scaled_px(14), scaled_px(14))
        active = self.selected | ({self.hover_location} if self.hover_location else set())
        masks = self.masks() if active else {}
        for code in sorted(active, key=lambda key: masks.get(key, (None,None,0))[2], reverse=True):
            if code not in masks:
                continue
            colored = masks[code][0].copy()
            tint = QPainter(colored)
            tint.setCompositionMode(QPainter.CompositionMode_SourceIn)
            tint.fillRect(colored.rect(), QColor(_COMPARTMENT_COLOURS.get(code, "#73cfce")))
            tint.end()
            painter.setOpacity(0.75)
            painter.drawImage(0, 0, colored)
        painter.setOpacity(1)
        self.renderer.render(painter, self._target())
        painter.end()


class OrganismDiagram(QWidget):
    """A cell illustration and keyboard-accessible compartment selector.

    :param app_key: organism key; only Toxoplasma uses hyperLOPIT labels.
    :param path: bundled SVG path; no external resource is fetched.
    :param parent: owning Qt widget.
    """

    def __init__(self, app_key: str, path: Path, parent=None):
        """Build upright artwork, a multi-select legend and compartment descriptions."""
        super().__init__(parent)
        self.labels = (COMPARTMENT_SL if app_key == "toxoplasma" else
                       YEAST_LABELS if app_key == "candida" else APICOMPLEXAN_LABELS)
        self.setObjectName("OrganismDiagram")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        source = path.read_text(encoding="utf-8")
        root = ET.fromstring(source)
        self.descriptions = {node.get("id"): " ".join(text.itertext()).strip()
                             for node in root.iter() for text in node.findall(_SVG + "text")
                             if text.get("property") == "description"}
        self.artwork = _CellArtwork(source, self, portrait=app_key != "candida", locations=self.labels.values())
        self.artwork.setAccessibleName(tr("Cell compartments"))
        layout.addWidget(self.artwork, 1)
        label = QLabel(tr("hyperLOPIT compartment") if app_key == "toxoplasma"
                       else tr("UniProt compartment"))
        self.selector = QListWidget(self)
        self.selector.setSelectionMode(QAbstractItemView.NoSelection)
        self.selector.setMouseTracking(True)
        self.selector.viewport().installEventFilter(self)
        self.selector.setFixedHeight(scaled_px(164))
        self.selector.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.selector.setAccessibleName(label.text())
        label.setBuddy(self.selector)
        for name, location in self.labels.items():
            item = QListWidgetItem(tr(name), self.selector)
            item.setData(Qt.UserRole, location)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setToolTip(tr(self.descriptions.get(location, "")))
        legend_heading = QHBoxLayout()
        legend_heading.addWidget(label, 1)
        self.clear_button = QPushButton(tr("Clear components"), self)
        self.clear_button.clicked.connect(self.clear_components)
        legend_heading.addWidget(self.clear_button)
        layout.addLayout(legend_heading)
        layout.addWidget(self.selector)
        self.caption = QLabel()
        self.caption.setTextFormat(Qt.PlainText)
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.caption)
        self.selector.itemChanged.connect(self._select)
        self.selector.itemEntered.connect(lambda item: self.artwork.set_hover(item.data(Qt.UserRole)))
        self.artwork.hovered.connect(self._describe)
        self.artwork.clicked.connect(self._toggle_location)
        self._select()

    def clear_components(self) -> None:
        """Clear all persistent selections and the transient compartment highlight."""
        self.selector.blockSignals(True)
        for index in range(self.selector.count()):
            self.selector.item(index).setCheckState(Qt.Unchecked)
        self.selector.blockSignals(False)
        self.artwork.set_hover("")
        self._select()

    def _select(self, *_args) -> None:
        """Update all checked compartments while retaining independent selections."""
        items = [self.selector.item(i) for i in range(self.selector.count())]
        selected = [item.data(Qt.UserRole) for item in items if item.checkState() == Qt.Checked]
        self.artwork.select(selected)
        location = self.artwork.hover_location or (selected[-1] if selected else "")
        self._describe(location)

    def _toggle_location(self, location: str) -> None:
        """Toggle a shared anatomical shape; individual LOPIT classes remain selectable below."""
        matches = [self.selector.item(i) for i in range(self.selector.count())
                   if ("SL0173" if self.selector.item(i).data(Qt.UserRole) == "SL0171"
                       else self.selector.item(i).data(Qt.UserRole)) == location]
        checked = any(item.checkState() == Qt.Checked for item in matches)
        for item in matches:
            item.setCheckState(Qt.Unchecked if checked else Qt.Checked)

    def _describe(self, location: str) -> None:
        """Color matching legend entries and explain the hovered UniProt/LOPIT location."""
        self.selector.blockSignals(True)
        for i in range(self.selector.count()):
            item = self.selector.item(i)
            code = item.data(Qt.UserRole)
            target = "SL0173" if code == "SL0171" else code
            active = item.checkState() == Qt.Checked or target == location
            item.setForeground(QColor(_COMPARTMENT_COLOURS.get(target, "#73cfce")) if active else self.palette().text())
        self.selector.blockSignals(False)
        if not location:
            self.caption.setText(tr("Hover over the cell to identify a compartment. Check several labels to keep them highlighted."))
            return
        shared = [tr(name) for name, code in self.labels.items() if code == location]
        text = f'{location[:2]}-{location[2:]} · ' + " / ".join(shared)
        description = self.descriptions.get(location, "")
        if description:
            text += "\n" + tr(description)
        if len(shared) > 1:
            text += "\n" + tr("These LOPIT classes share one anatomical outline; the diagram does not distinguish their protein populations.")
        if location == "SL0171":
            text += "\n" + tr("The membrane label uses the mitochondrial outline; it has no separate shape.")
        self.caption.setText(text)

    def eventFilter(self, watched, event):
        """End the legend's transient highlight when the pointer leaves it."""
        if watched is self.selector.viewport() and event.type() == QEvent.Leave:
            self.artwork.set_hover("")
        return super().eventFilter(watched, event)
