"""286: each level says what it does, in the numbers the code uses, in every
language, under the same name everywhere.

"Each level's tooltip states the intended hardware profile, memory/cache
behavior, animation behavior and cleanup trade-off in concrete language. The
same names and meanings appear in first-run setup ... [and] every supported
language."

Three things were short of that:

* the tooltips promised behaviour no code has -- Laptop "kept to one
  worker" and "caches are dropped as soon as a run finishes", Workstation
  "nothing is dropped until you ask" -- and named no number;
* none of them reached a translator: the runtime extractor iterated the two
  dicts and collected their KEYS, and the dialog quoted the hardware notes
  without passing them through ``tr``;
* first-run setup captioned the selector "spaCR mode", the name of the
  control this item removed, while Preferences calls it "Performance".
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import memory_budget as mb
from spacr.qt import preferences as P

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("level", P.PERFORMANCE_LEVELS)
def test_each_tooltip_states_the_budget_its_level_enforces(level):
    idle, cache, _headroom = mb.recommended_for(level)
    note = P.PERFORMANCE_NOTES[level]
    assert f"{idle:g} minutes" in note, (level, note)
    assert f"{cache} MB" in note, (level, note)


def test_no_tooltip_promises_what_no_code_does():
    said = " ".join(P.PERFORMANCE_NOTES.values()).lower()
    for claim in ("one worker", "nothing is dropped until you ask",
                  "as soon as a run finishes"):
        assert claim not in said, f"a tooltip still promises {claim!r}"


def test_the_setup_screen_captions_the_selector_as_preferences_does(qtbot):
    from PySide6.QtWidgets import QComboBox, QFormLayout

    from spacr.qt.setup_screen import questions

    (setup_caption,) = [q[1] for q in questions() if q[0] == "spacr_mode"]

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "PerformanceLevel")
    captions = [form.labelForField(combo) for form in
                dlg.findChildren(QFormLayout)]
    (label,) = [widget for widget in captions if widget is not None]

    assert setup_caption == label.text(), (
        f"setup calls the selector {setup_caption!r} and Preferences calls "
        f"it {label.text()!r}")


def test_the_budget_tooltips_translate_the_hardware_they_name(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QFormLayout, QSpinBox

    from spacr.qt import i18n
    from spacr.qt.widgets.hint_bar import HintBar

    hardware = set(mb.HARDWARE_NOTES.values())
    real = i18n.tr

    def marked(text, language=None, **values):
        if str(text) in hardware:
            return f"[{text}]"
        return real(text, language, **values)

    monkeypatch.setattr(i18n, "tr", marked)
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    spin = dlg.findChild(QSpinBox, "CacheCeilingMb")
    # THE EXPLANATION IS MOVED, TWICE, AND NOT LOST. The dialog first moves
    # every form field's tooltip onto its row LABEL and clears the field,
    # then its HintBar moves the label's tooltip into the hint line and the
    # label's accessible description. So the sentence is read from the row
    # label, in whichever of those three places it ended up.
    labels = [form.labelForField(spin) for form in
              dlg.findChildren(QFormLayout)]
    (label,) = [widget for widget in labels if widget is not None]
    bars = dlg.findChildren(HintBar)
    tip = (label.toolTip() or label.accessibleDescription()
           or next((bar.explains(label) for bar in bars
                    if bar.explains(label)), ""))
    assert tip, "the cache ceiling row has no explanation at all"
    for text in hardware:
        assert f"[{text}]" in tip, (
            f"{text!r} is quoted in the tooltip without being translated")
