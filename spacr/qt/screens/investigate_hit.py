"""Register the settings-driven Investigate Hit application."""
from __future__ import annotations

from ..app_catalog import declared_app, register_declared

APP_KEY = "investigate_hit"

# The row this screen puts in the registry is declared in
# `spacr.qt.app_catalog`, which is what lets the app be registered without
# importing this module -- the launch reads the table, not the screen. These
# read the same row back rather than restating it, so the name, the blurb and
# the nine translations have one spelling and no second copy to drift from.
_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_TRANSLATIONS = _ROW.translations

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "register"]


def _make_screen(app_key=None, host=None):
    """Import attribution and dataframe dependencies only when opened."""
    from .model_explanation import make_investigate_hit_screen
    return make_investigate_hit_screen(app_key=app_key, host=host)


def register() -> bool:
    """Add Investigate Hit through the common application registry."""
    return register_declared(__name__) is not None


register()
