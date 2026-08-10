"""Classifier Evaluation workbench rendering, filtering, and registration."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.classifier_evaluation import (
    LeakageReport,
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens.classifier_evaluation import (
    APP_INTRO,
    APP_KEY,
    APP_NAME,
    APP_SECTION,
    ClassifierEvaluationScreen,
)


@pytest.fixture
def evaluation_root(tmp_path):
    """Write one small, representative evaluation bundle."""
    paths = [
        "plate1_A01_1_1.png",
        "plate1_A02_1_2.png",
        "plate2_B01_1_3.png",
        "plate2_B02_1_4.png",
    ]
    evaluation = evaluate_predictions(
        [0, 1, 0, 1],
        np.asarray([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.7, 0.3],
        ]),
        paths,
        classes=["negative", "positive"],
        fold_ids=[1, 1, 2, 2],
        calibration_method="none",
        calibration_bins=4,
    )
    report = LeakageReport(
        group_by="well",
        train_samples=2,
        validation_samples=2,
        overlap_counts={
            "exact": 0, "augmentation_family": 0, "object": 0,
            "field": 0, "well": 0, "plate": 1,
        },
        examples={
            "exact": [], "augmentation_family": [], "object": [],
            "field": [], "well": [], "plate": ["plate1"],
        },
        split_name="outer_1",
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation",
        evaluation,
        leakage_reports=[report],
    ).parents[1]


@pytest.fixture
def screen(qtbot, qt_theme_applied, evaluation_root):
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    return widget


#: The workbench's own settings, as Classify must offer them. Their GROUP was
#: "Evaluation Workbench" until 2026-08-06/07 (c41a75b6, 30500970), when
#: Classify's nine categories became six: "Validation", "Evaluation Workbench"
#: and "Monitoring & Runtime" were three headings asking one question — how do
#: I know whether this worked — so nobody looking for a cross-validation
#: setting knew which of the three to open. They are now under "Evaluation &
#: Results", and this test pins the placement rather than the old name.
EVALUATION_SETTINGS = [
    "classifier_evaluation",
    "nested_cv_inner_folds",
    "evaluation_calibration",
    "evaluation_bins",
    "evaluation_fail_on_leakage",
    "leakage_audit_train_test",
    "leakage_hash_content",
    "leakage_require_identity",
]

EVALUATION_GROUP = "Evaluation & Results"


def test_registration_metadata_matches_app_registry():
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import (
        api_docs_url,
        categories_for_app,
        get_categories,
    )

    row = next(item for item in APPS if item[0] == APP_KEY)
    assert row[1] == APP_NAME == "Classifier Evaluation"
    assert row[3] == APP_SECTION == "Results & QC"
    assert APP_INTRO
    categories = categories_for_app("classify", get_categories())
    assert EVALUATION_GROUP in categories
    for key in EVALUATION_SETTINGS:
        # Placed EXACTLY once, and in the evaluation group. Naming its home
        # by hand is what broke when the groups merged; asking where every
        # home is catches the failure the rename could actually cause — a
        # setting silently falling through into the "Additional Settings"
        # catch-all, where it is curated by nobody.
        homes = [name for name, keys in categories.items() if key in keys]
        assert homes == [EVALUATION_GROUP], (key, homes)
        assert api_docs_url("classify", key).endswith(
            "/spacr/classifier_evaluation/index.html"
        )


def test_scan_loads_all_evaluation_views(screen):
    assert len(screen.bundles) == 1
    assert screen.bundle is not None
    assert screen._confusion.rowCount() == 2
    assert screen._confusion.columnCount() == 3
    assert screen._per_plate.rowCount() == 2
    assert screen._calibration.rowCount() > 0
    assert screen._predictions.rowCount() == 4
    assert '"accuracy"' in screen._overview.toPlainText()
    assert '"passed": true' in screen._leakage.toPlainText()
    assert "4 held-out prediction" in screen._status.text()


def test_prediction_filter_searches_across_columns(screen):
    screen._prediction_filter.setText("plate2 B01")
    assert screen._predictions.rowCount() == 1
    screen._prediction_filter.clear()
    assert screen._predictions.rowCount() == 4


def test_missing_source_is_inline_and_not_silent(
        qtbot, qt_theme_applied, tmp_path):
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(tmp_path / "does-not-exist"))
    widget.scan()
    assert widget.bundle is None
    assert "FileNotFoundError" in widget.last_error
    assert "Could not scan" in widget._status.text()


def test_copy_path_uses_loaded_manifest(screen, monkeypatch):
    copied = []

    class Clipboard:
        def setText(self, value):
            copied.append(value)

    monkeypatch.setattr(
        "spacr.qt.screens.classifier_evaluation.QApplication.clipboard",
        lambda: Clipboard(),
    )
    screen._copy_current_path()
    assert copied == [str(screen.bundle["path"])]
