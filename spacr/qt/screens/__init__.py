"""Screen widgets — one per spacr app.

Screens that own their registry row (:func:`spacr.qt.app.register_app`) rather
than appearing in the table inside ``app.py`` have to be *imported* for that
row to exist, so they are imported here — this package is on the path of every
screen the window builds.

Each import is wrapped: a new screen with an import-time bug must not take
every other screen down with it. The same posture ``app.py`` takes towards
plugin registrations, for the same reason.
"""
import logging as _logging

for _module in ("data_manager", "pipeline_graph"):
    try:
        __import__(f"{__name__}.{_module}")
    except Exception:  # pragma: no cover - defensive, per-module
        _logging.getLogger(__name__).exception(
            "Could not register the %s screen", _module)
del _module
