"""Register the settings-driven Explain CV Model application."""
from __future__ import annotations

APP_KEY = "explain_cv"
APP_NAME = "Explain CV Model"
APP_DESCRIPTION = (
    "Reproduce CV decisions from measured features, then inspect gain, "
    "held-out permutation importance and SHAP")
APP_INTRO = (
    "Choose an existing per-object prediction file and its measurements.db. "
    "spaCR holds wells (or plates) intact, reports fidelity against the "
    "majority baseline before showing any importance, and excludes scores, "
    "classes, identifiers and annotations that would leak the answer. Random "
    "Forest, histogram gradient boosting and optional XGBoost are distinct "
    "recorded backends; an unavailable XGBoost choice is never substituted.")
APP_TRANSLATIONS = (
    "Förklara CV-modell", "CV-Modell erklären", "Explicar modelo CV",
    "解释 CV 模型", "Explicar modelo de VC", "CV मॉडल समझाएँ",
    "CV 모델 설명", "Skýra CV-líkan", "Expliquer le modèle CV")

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "register"]


def register() -> bool:
    """Add the module through spaCR's single application-registration seam."""
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    # Import registers the settings only when this app itself is registered.
    from ...surrogate import register_explain_cv_settings
    from .model_explanation import make_model_explanation_screen
    register_explain_cv_settings()
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=make_model_explanation_screen,
        stage=STAGE_ALPHA, title=APP_NAME, intro=APP_INTRO,
        api_module="surrogate", entry="spacr.surrogate:run_explain_cv",
        defaults_module="spacr.surrogate", translations=APP_TRANSLATIONS)
    return True


register()
