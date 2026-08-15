"""Register the settings-driven Investigate Hit application."""
from __future__ import annotations

APP_KEY = "investigate_hit"
APP_NAME = "Investigate Hit"
APP_DESCRIPTION = (
    "Return one exact regression hit to cross-fitted candidate cells and "
    "well-level quantitative evidence")
APP_INTRO = (
    "Carry the exact regression run, gene, phenotype direction, FDR and guide "
    "support back to measured cells. The first output is an honest score-based "
    "review ranking. An optional hierarchical mixture then assigns cross-fitted "
    "hit-like probabilities without forcing sequencing fraction to equal cell "
    "prevalence. Comparisons use wells as the independent unit; stored calls "
    "are versioned and never overwrite hand annotations.")
APP_TRANSLATIONS = (
    "Undersök träff", "Treffer untersuchen", "Investigar acierto",
    "调查命中", "Investigar acerto", "हिट की जाँच करें",
    "히트 조사", "Rannsaka niðurstöðu", "Examiner le résultat")

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "register"]


def _make_screen(app_key=None, host=None):
    """Import attribution and dataframe dependencies only when opened."""
    from .model_explanation import make_investigate_hit_screen
    return make_investigate_hit_screen(app_key=app_key, host=host)


def register() -> bool:
    """Add Investigate Hit through the common application registry."""
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=_make_screen,
        stage=STAGE_ALPHA, title=APP_NAME, intro=APP_INTRO,
        api_module="hit_investigation",
        entry="spacr.hit_investigation:investigate_hit",
        defaults_module="spacr.hit_investigation",
        translations=APP_TRANSLATIONS)
    return True


register()
