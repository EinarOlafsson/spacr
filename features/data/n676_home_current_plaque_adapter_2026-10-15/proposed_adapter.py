"""Proposed recorder protocol for Work review; no capture or inference on import.

The owner supplies real pointer actions and event processing. This proposal
does not restore the retired generic preview's area-filter interface.
"""
from pathlib import Path
from time import monotonic


def owned_panel(screen, click, settle):
    """Reveal the current screen-owned plaque panel using real controls."""
    panel = getattr(screen, "_live_preview", None)
    switch = getattr(screen, "_preview_switch", None)
    if panel is None or switch is None:
        raise ValueError("The current screen-owned Plaque preview is missing")
    if not panel.isVisible():
        click(switch)
        settle(.2)
    if not panel.isVisible():
        raise ValueError("The current Plaque preview did not become visible")
    if panel.mode() != "plaque":
        click(panel._mode_switch._buttons["plaque"])
        settle(.2)
    if panel.mode() != "plaque":
        raise ValueError("The current Plaque mode was not selected")
    return panel


def _context(panel):
    """Bind a pass to its visible input and current inference controls."""
    path = panel.current_path()
    settings = panel.current_settings()
    return (panel.mode(), None if path is None else str(Path(path).resolve()),
            *(settings.get(key) for key in (
                "plaque_model", "diameter", "flow_threshold", "CP_prob",
                "colony_counting")))


def run_owned_preview(panel, click, settle, timeout, *, clock=monotonic):
    """Observe a new accepted result and retirement, never a surviving worker.

    ``click`` must be the owner's physical pointer action. A previously stored
    result, a cancelled pass or an idle panel alone is never accepted. Timeout
    requests cancellation through the real button and disconnects the observer;
    Work's existing process cleanup still owns eventual thread retirement.
    """
    context = _context(panel)
    if context[0] != "plaque" or context[1] is None:
        raise ValueError("Select a current Plaque input before running")
    if panel.preview_running():
        raise ValueError("The previous preview has not retired")
    if not panel._run_btn.isVisible() or not panel._run_btn.isEnabled():
        raise ValueError("The actual Run preview control is unavailable")
    results = []

    def completed(result):
        results.append(result)

    panel.preview_ready.connect(completed)
    try:
        deadline = clock() + timeout
        click(panel._run_btn)
        while True:
            if clock() >= deadline:
                if panel._cancel_btn.isVisible() and panel._cancel_btn.isEnabled():
                    click(panel._cancel_btn)
                raise TimeoutError("The current Plaque preview exceeded its bounded wait")
            if _context(panel) != context:
                raise ValueError("The preview input or inference settings changed")
            if results and not panel.preview_running():
                if len(results) != 1:
                    raise ValueError("More than one preview completion was observed")
                result = results[0]
                if (not isinstance(result, dict) or result.get("error")
                        or result is not panel._plaque_result
                        or not {"path", "labels", "count"} <= result.keys()
                        or str(Path(result["path"]).resolve()) != context[1]):
                    raise ValueError("The completion does not identify the current Plaque pass")
                return result
            if not panel.preview_running() and not results:
                raise ValueError("The preview ended without an accepted result")
            settle(.05)
    finally:
        try:
            panel.preview_ready.disconnect(completed)
        except (RuntimeError, TypeError):
            pass
