from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tutorial_capture_all_modules",
    ROOT / "tools" / "capture_all_modules.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class _Button:
    def click(self) -> None:
        pass

    def isCheckable(self) -> bool:
        return False


class _Strip:
    buttons = ()

    def button_for(self, key: str):
        return _Button() if key == "hit_list" else None


class _Host:
    _fold_pages = None
    _fold_strip = _Strip()
    _folds = None
    _sweep_card = None

    def __init__(self, hits) -> None:
        class Card:
            shown = False

            def show(self) -> None:
                self.shown = True

        class Tabs:
            current = None

            def setCurrentWidget(self, widget) -> None:
                self.current = widget

        self._results_panel = type(
            "Results", (), {"hits": hits, "tabs": Tabs()}
        )()
        self._figures_card = Card()
        self.results_raised = False

    def _raise_the_results_tab(self) -> None:
        self.results_raised = True


class _Window:
    _startup = None

    def __init__(self, host) -> None:
        self._screens = {"regression": host}
        self.selected = None

    def _on_nav_selected(self, key: str) -> None:
        self.selected = key


class _Page:
    parent = "host"

    def setParent(self, parent) -> None:
        self.parent = parent


class _Pages:
    def __init__(self) -> None:
        self.pages = [object(), _Page(), _Page()]
        self.current = 2

    def count(self) -> int:
        return len(self.pages)

    def widget(self, index: int):
        return self.pages[index]

    def removeTab(self, index: int) -> None:
        self.pages.pop(index)

    def setCurrentIndex(self, index: int) -> None:
        self.current = index


def test_hit_list_capture_uses_the_results_panels_live_hits_tab(monkeypatch):
    expected = object()
    host = _Host(expected)
    window = _Window(host)
    monkeypatch.setattr(module, "settle", lambda *_args, **_kwargs: None)

    actual = module.open_tutorial_target(
        window,
        {"app_key": "hit_list", "host_app_key": "regression"},
        object(),
    )

    assert window.selected == "regression"
    assert host.results_raised
    assert host._figures_card.shown
    assert host._results_panel.tabs.current is expected
    assert actual is expected


def test_reset_host_view_removes_previously_opened_fold_pages():
    pages = _Pages()
    folded_pages = list(pages.pages[1:])
    host = type(
        "Host",
        (),
        {
            "_fold_pages": pages,
            "_fold_strip": _Strip(),
            "_folds": None,
            "_sweep_card": None,
        },
    )()

    module._reset_host_view(host)

    assert pages.count() == 1
    assert pages.current == 0
    assert all(page.parent is None for page in folded_pages)


def test_capture_failure_report_survives_failure_and_clears_on_success(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    report = catalog / "capture_failures.json"
    report.write_text("stale\n")
    failures = [{"lesson": "48_hit_list", "error": "failed"}]

    assert module.finish_capture_batch(failures) == report
    assert json.loads(report.read_text()) == failures

    assert module.finish_capture_batch([]) is None
    assert not report.exists()
