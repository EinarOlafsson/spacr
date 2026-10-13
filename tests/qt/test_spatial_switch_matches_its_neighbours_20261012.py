"""A renamed actions-row switch keeps the shared toggle look.

The Spatial switches carry their own object names so the alpha gate can find
them. The toggle's sheet used to select ``QLabel#AiToggleLabel``, so a renamed
switch lost its weight, size and padding and drew in the plain label font
beside 3D, Time, OPS and AI.
"""

from PySide6.QtWidgets import QWidget

from spacr.qt.widgets.ai_toggle_label import AiToggleLabel


def test_a_renamed_switch_draws_like_an_unrenamed_one(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    plain = AiToggleLabel(text="OPS", tooltip="t", parent=host)
    renamed = AiToggleLabel(text="Spatial", tooltip="t", parent=host)
    renamed.setObjectName("MaskSpatialSwitch")
    plain.ensurePolished()
    renamed.ensurePolished()
    assert renamed.objectName() == "MaskSpatialSwitch"
    assert renamed.font().weight() == plain.font().weight()
    assert renamed.font().family() == plain.font().family()
    assert renamed.font().pixelSize() == plain.font().pixelSize()
    assert "#AiToggleLabel" not in renamed.styleSheet()
    assert "font-weight: 600" in renamed.styleSheet()
