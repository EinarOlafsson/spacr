"""The public issue pathway always exposes the exact payload first."""
from __future__ import annotations


def test_preview_defaults_to_stricter_path_redaction(qtbot):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    dialog = IssuePreviewDialog({
        "title": "boom",
        "body": 'File "/home/alice/secret/plate7.tif", line 4',
        "fingerprint": "abc123",
    })
    qtbot.addWidget(dialog)
    assert dialog.strip_paths.isChecked()
    assert "plate7.tif" not in dialog.body_edit.toPlainText()
    assert "<PATH>" in dialog.body_edit.toPlainText()


def test_preview_returns_user_edits_exactly(qtbot):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    dialog = IssuePreviewDialog({
        "title": "before", "body": "before body", "fingerprint": "abc123"
    })
    qtbot.addWidget(dialog)
    dialog.title_edit.setText("after")
    dialog.body_edit.setPlainText("safe edited body")
    assert dialog.approved_report() == {
        "title": "after",
        "body": "safe edited body",
        "fingerprint": "abc123",
    }


def test_unchecking_strict_redaction_restores_sanitized_source(qtbot):
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    source = 'File "~/project/image.tif", line 2'
    dialog = IssuePreviewDialog(
        {"title": "t", "body": source, "fingerprint": "f"}
    )
    qtbot.addWidget(dialog)
    dialog.strip_paths.setChecked(False)
    assert dialog.body_edit.toPlainText() == source
