"""Item 598: the contribute dialogs show a working dataset link.

The maintainer read the destination as
``huggingface.co/datasets/einarolafssoncommunity_toxoplasma_datasets``. The
text always held the slash, but word wrap broke the address right after
``einarolafsson/``, leaving the slash at a line end, and the label could
not be clicked, selected or copied. The address is now an https link on a
line of its own that does not wrap, in Make Masks' and Plaque Assay's
contribute dialogs and in the thank-you after sending.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QTextDocument  # noqa: E402

from spacr.qt.widgets import model_share_dialog as msd  # noqa: E402

REPO = "einarolafsson/community_toxoplasma_datasets"
URL = f"https://huggingface.co/datasets/{REPO}"
PR = f"{URL}/discussions/1"


class _Prefs:
    def __init__(self):
        self.store = {}

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


@pytest.fixture
def prefs(monkeypatch):
    store = _Prefs()
    monkeypatch.setattr(msd, "_preferences", lambda: store)
    return store


def _is_link_label(label):
    flags = label.textInteractionFlags()
    return (label.openExternalLinks()
            and label.textFormat() == Qt.RichText
            and bool(flags & Qt.LinksAccessibleByMouse)
            and bool(flags & Qt.TextSelectableByMouse))


def _address_lines(html, width):
    doc = QTextDocument()
    doc.setHtml(html)
    doc.setTextWidth(width)
    doc.size()
    lines = []
    block = doc.begin()
    while block.isValid():
        layout, text = block.layout(), block.text()
        for i in range(layout.lineCount()):
            line = layout.lineAt(i)
            lines.append(text[line.textStart():
                              line.textStart() + line.textLength()])
        block = block.next()
    return lines


def test_make_masks_target_is_a_whole_clickable_link(qtbot, prefs):
    dialog = msd.ContributeMasksDialog(threaded=False)
    qtbot.addWidget(dialog)
    dialog.name_edit.setText("toxoplasma datasets")
    label = dialog.target_label
    assert label.objectName() == "ContributeMasksTarget"
    assert _is_link_label(label)
    assert "einarolafsson/community_toxoplasma_datasets" in label.text()
    assert f'href="{URL}"' in label.text()
    lines = _address_lines(label.text(), 300)
    assert URL in lines, lines


def test_make_masks_without_a_name_says_so(qtbot, prefs):
    dialog = msd.ContributeMasksDialog(threaded=False)
    qtbot.addWidget(dialog)
    dialog.name_edit.setText("")
    assert "href" not in dialog.target_label.text()
    assert dialog.target_label.text().startswith("Name the dataset")


def test_make_masks_thank_you_links_the_pull_request(qtbot, prefs):
    dialog = msd.ContributeMasksDialog(threaded=False)
    qtbot.addWidget(dialog)
    dialog._on_uploaded(PR)
    assert _is_link_label(dialog.status)
    assert f'href="{PR}"' in dialog.status.text()
    dialog._on_failed("bad <thing> & more")
    assert "&lt;thing&gt; &amp; more" in dialog.status.text()


def test_plaque_assay_dialog_links_its_dataset_and_pull_request(qtbot, prefs):
    from spacr.qt.widgets.model_share import COMMUNITY_PLAQUES_REPO
    from spacr.qt.widgets.plaque_preview import ContributeDialog

    dialog = ContributeDialog("plaque", [], threaded=False)
    qtbot.addWidget(dialog)
    label = dialog.target_label
    assert _is_link_label(label)
    assert (f'href="https://huggingface.co/datasets/{COMMUNITY_PLAQUES_REPO}"'
            in label.text())
    named = ContributeDialog("plaque", [], threaded=False,
                             target="community_toxoplasma_datasets")
    qtbot.addWidget(named)
    assert "einarolafsson/community_toxoplasma_datasets" in \
        named.target_label.text()
    dialog._on_uploaded(PR)
    assert f'href="{PR}"' in dialog.status.text()
    assert dialog.status.openExternalLinks()
