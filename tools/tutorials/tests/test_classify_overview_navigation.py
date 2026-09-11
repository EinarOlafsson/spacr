"""Pin genuine tooltip identity and the navigation-only recording boundary."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compose_classify_overview import check_navigation


def valid():
    keys = ['classifier_evaluation', 'explain_cv', 'activation', 'train_compare', 'feature_explorer']
    return dict(model_started=False, family_restored='cv', nested_feature_explorer_opened=True,
                folds=[dict(key=k, tooltip=k, displayed_tooltip=k) for k in keys])


def test_actual_distinct_routes_and_matching_visible_text_are_accepted():
    check_navigation(valid())


def test_stale_tooltip_from_previous_button_is_rejected():
    proof = valid()
    proof['folds'][1]['displayed_tooltip'] = proof['folds'][0]['tooltip']
    with pytest.raises(ValueError):
        check_navigation(proof)


def test_duplicate_route_cannot_replace_missing_route():
    proof = valid()
    proof['folds'][1] = dict(proof['folds'][0])
    with pytest.raises(ValueError):
        check_navigation(proof)


def test_started_model_cannot_be_called_navigation_only():
    proof = valid()
    proof['model_started'] = True
    with pytest.raises(ValueError):
        check_navigation(proof)


def test_unrestored_family_is_rejected():
    proof = valid()
    proof['family_restored'] = 'ml'
    with pytest.raises(ValueError):
        check_navigation(proof)
