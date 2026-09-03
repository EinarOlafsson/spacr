"""A settings field fills its section whatever the platform style says.

Issue 115, reported from macOS: "field and setting do not expand with
container". The cause was measured rather than guessed -- ``Section`` built
its ``QFormLayout`` and never set a field-growth policy, so the answer was
delegated to whatever style was active.

WHY A HOSTILE STYLE AND NOT JUST A WIDTH ASSERTION. With Fusion, the default
IS ``AllNonFixedFieldsGrow``, so a plain Linux test passes whether or not the
policy is set -- it would agree with the mutant and prove nothing. This
installs a QProxyStyle that answers ``SH_FormLayoutFieldGrowthPolicy`` with
``FieldsStayAtSizeHint``: hostile, but a valid Qt answer, and the shape the
reporter's platform style chose. Measured on a 1,178 px section: 1,115 px for
the field with the policy named, 108 px without it.

Delete the ``setFieldGrowthPolicy`` call in ``spacr/qt/widgets/section.py``
and this test goes red. That is the point of it.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFormLayout, QLineEdit, QProxyStyle, QStyle, QStyleFactory, QWidget

from spacr.qt.widgets.section import Section

SECTION_PX = 1178


class _FieldsStayAtSizeHintStyle(QProxyStyle):
    """A style that refuses to grow form fields, which Qt permits."""

    def styleHint(self, hint, option=None, widget=None,          # noqa: N802
                  returnData=None):
        if hint == QStyle.StyleHint.SH_FormLayoutFieldGrowthPolicy:
            return int(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint.value)
        return super().styleHint(hint, option, widget, returnData)


@pytest.fixture()
def hostile_section(qtbot, qapp):
    """An expanded section holding one field, under the hostile style.

    THE STYLE GOES ON THE APPLICATION, not on the widgets. A QFormLayout asks
    the style of the widget that OWNS it, and setting it on the section or the
    field leaves the body's own style untouched -- which is how the first
    version of this test passed against a mutant with the fix deleted. It is
    restored by name afterwards; handing back the previous style object risks
    a double free, because setStyle takes ownership.
    """
    previous = qapp.style().objectName() or "Fusion"
    style = _FieldsStayAtSizeHintStyle()
    qapp.setStyle(style)

    host = QWidget()
    qtbot.addWidget(host)
    section = Section("Settings", parent=host, expanded=True)
    field = QLineEdit()
    section.add_row("A setting", field)

    host.resize(SECTION_PX, 400)
    section.resize(SECTION_PX, 300)
    host.show()
    qtbot.waitExposed(host)
    for _ in range(4):                    # let the layout actually settle
        qtbot.wait(10)
    # HOST AND STYLE HANDED BACK, not just the two widgets under test. Drop
    # the last Python reference to either and the C++ objects go with it --
    # the section's own QFormLayout included, which fails as "already
    # deleted" rather than as anything to do with layout.
    yield section, field, host, style
    qapp.setStyle(QStyleFactory.create(previous) or QStyleFactory.create("Fusion"))


def test_the_field_fills_the_section_under_a_hostile_style(hostile_section):
    section, field, _host, _style = hostile_section
    assert section.width() >= SECTION_PX * 0.9, "the section itself is narrow"
    # Generous: the label column, margins and spacing take their share. The
    # defect this guards against left the field at ~108 px of 1,178.
    assert field.width() > SECTION_PX * 0.5, (
        f"the field is {field.width()} px of a {section.width()} px section; "
        "the form's field-growth policy is being decided by the style again")


def test_the_policy_is_named_rather_than_inherited(hostile_section):
    """The layout must answer for itself, not ask the style."""
    section, _field, _host, _style = hostile_section
    assert (section._form.fieldGrowthPolicy()
            == QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
