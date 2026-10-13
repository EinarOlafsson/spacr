"""317: report prose is translated, identifiers and application counts are not."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import i18n, settings_pack
from spacr.qt.screens import app_screen
from spacr.qt.widgets.console_panel import ConsolePanel

pytestmark = pytest.mark.qt

_HEAD = (
    "[example] {applied} settings applied to the form from {name}; "
    "{accepted} CSV keys accepted."
)
_RENAMED = "[example] Renamed {count} settings: {renames}"
_DROPPED = "[example] Dropped {count} unknown or retired settings: {keys}"
_ELSEWHERE = (
    "[example] Ignored {count} settings available in this build "
    "but not on this form: {keys}"
)
_MALFORMED = "[example] Skipped {count} unreadable CSV row(s)."

# Controlled test translations, not production catalog entries. Reordering
# placeholders proves that values are inserted AFTER the stable prose lookup.
_FRENCH = {
    _HEAD: "[exemple] {name} : {accepted} clés CSV acceptées ; "
           "{applied} paramètres appliqués au formulaire.",
    _RENAMED: "[exemple] Paramètres renommés ({count}) : {renames}",
    _DROPPED: "[exemple] Paramètres inconnus ou retirés, ignorés ({count}) : {keys}",
    _ELSEWHERE: "[exemple] Paramètres disponibles mais absents "
                "de ce formulaire ({count}) : {keys}",
    _MALFORMED: "[exemple] Lignes CSV illisibles ignorées : {count}",
}


class _Console:
    """Use the real notice/localization path without constructing widgets."""

    append_notice = ConsolePanel.append_notice
    _append_notice_on_gui_thread = ConsolePanel._append_notice_on_gui_thread

    def __init__(self):
        self.lines = []

    def _on_gui_thread(self):
        return True

    def append_stdout(self, text):
        self.lines.append(text)


@pytest.fixture
def french_console(monkeypatch):
    """Keep real tr formatting while supplying a tiny translated catalog."""
    seen = []

    def lookup(source, language):
        seen.append((source, language))
        return _FRENCH.get(source)

    monkeypatch.setattr(i18n, "current_language", lambda: "fr")
    monkeypatch.setattr(i18n, "_exact_translation", lookup)
    console = _Console()
    console.lookups = seen
    return console


def _emit(console, report, applied):
    """Fail clearly before the staged private report helper is installed."""
    helper = getattr(app_screen, "_append_example_pack_report", None)
    assert callable(helper), "example reporting has no localized report helper"
    helper(console, report, applied)


def test_notice_translates_before_inserting_identifiers(french_console):
    """Positive control for the real console/tr path, independent of 317."""
    french_console.append_notice(
        _HEAD + "\n", applied=1, accepted=3, name="mask_{literal}.csv")

    assert french_console.lines == [
        "[exemple] mask_{literal}.csv : 3 clés CSV acceptées ; "
        "1 paramètres appliqués au formulaire.\n",
    ]
    assert french_console.lookups == [(_HEAD, "fr")]


def test_every_report_category_is_translated_without_translating_keys(
        french_console):
    """Live-on-other-forms and unknown keys need different translated reasons."""
    report = settings_pack.PackReport(
        applied=["channels", "verbose"],
        renamed=[("old_{raw}", "cell_flow_threshold")],
        dropped=["unknown_{raw}", "timelapse"],
        elsewhere=["timelapse"],
        malformed=2,
        source="mask_{raw}.csv",
    )

    _emit(french_console, report, applied=1)

    assert french_console.lines == [
        "[exemple] mask_{raw}.csv : 3 clés CSV acceptées ; "
        "1 paramètres appliqués au formulaire.\n",
        "[exemple] Paramètres renommés (1) : old_{raw} → cell_flow_threshold\n",
        "[exemple] Paramètres inconnus ou retirés, ignorés (1) : unknown_{raw}\n",
        "[exemple] Paramètres disponibles mais absents "
        "de ce formulaire (1) : timelapse\n",
        "[exemple] Lignes CSV illisibles ignorées : 2\n",
    ]
    assert french_console.lookups == [
        (source, "fr")
        for source in (_HEAD, _RENAMED, _DROPPED, _ELSEWHERE, _MALFORMED)
    ]


@pytest.mark.parametrize("applied", [0, 1])
def test_csv_keys_and_applied_form_settings_are_counted_separately(
        french_console, applied):
    """Two accepted aliases may target one widget, which may reject the value."""
    report = settings_pack.PackReport(
        applied=["cell_flow_threshold"],
        renamed=[("old_flow", "cell_flow_threshold")],
        source="gen_masks_settings.csv",
    )

    _emit(french_console, report, applied)

    assert french_console.lines[0] == (
        "[exemple] gen_masks_settings.csv : 2 clés CSV acceptées ; "
        f"{applied} paramètres appliqués au formulaire.\n"
    )
    assert len(french_console.lines) == 2


def test_a_clean_pack_emits_no_spurious_loss_notice(french_console):
    """The detailed notices must remain conditional on actual report entries."""
    report = settings_pack.PackReport(applied=["verbose"], source="mask.csv")

    _emit(french_console, report, applied=1)

    assert french_console.lines == [
        "[exemple] mask.csv : 1 clés CSV acceptées ; "
        "1 paramètres appliqués au formulaire.\n",
    ]
    assert french_console.lookups == [(_HEAD, "fr")]


@pytest.mark.parametrize("replace_screen", [False, True], ids=["same", "rebuilt"])
@pytest.mark.parametrize("fail_apply", [False, True], ids=["applied", "error"])
def test_real_example_caller_reports_the_actual_form_application_count(
        tmp_path, monkeypatch, french_console, replace_screen, fail_apply):
    """Use the actual count and the visible console, even after form replacement."""
    directory = tmp_path / "settings"
    directory.mkdir()
    (directory / "gen_masks_settings.csv").write_text(
        "Key,Value\nverbose,True\ncell_flow_threshold,0.4\n",
        encoding="utf-8")
    values = {"verbose": True, "cell_flow_threshold": 0.4}
    report = settings_pack.PackReport(
        applied=list(values), source="gen_masks_settings.csv")
    monkeypatch.setattr(settings_pack, "settings_from_pack",
                        lambda *args, **kwargs: (dict(values), report))
    received = []
    window = SimpleNamespace(_screens={})
    owner = [window]
    visible_console = _Console() if replace_screen else french_console

    def apply(loaded):
        received.append(dict(loaded))
        if replace_screen:
            window._screens["mask"] = SimpleNamespace(_console=visible_console)
            # MainWindow registers the replacement, detaches the old widget,
            # and schedules its deletion before bulk application returns.
            owner[0] = screen
        if fail_apply:
            raise ValueError("invalid setting {literal}")
        return 1

    screen = SimpleNamespace(
        app_key="mask",
        _EXAMPLE_SETTINGS_FILES=app_screen.AppScreen._EXAMPLE_SETTINGS_FILES,
        _load_settings_csv=lambda path: dict(values),
        reanchor_example_paths=lambda loaded, destination: loaded,
        apply_settings_dict=apply,
        _console=french_console,
        window=lambda: owner[0],
        _settings_model=SimpleNamespace(
            _defaults=dict(values), collect=lambda: dict(values)),
    )
    window._screens["mask"] = screen

    count = app_screen.AppScreen.apply_settings_that_came_with(screen, tmp_path)

    assert count == (0 if fail_apply else 1)
    assert received == [values]
    expected = [
        "[example] gen_masks_settings.csv could not be applied: "
        "invalid setting {literal}\n",
    ] if fail_apply else [
        "[exemple] gen_masks_settings.csv : 2 clés CSV acceptées ; "
        "1 paramètres appliqués au formulaire.\n",
    ]
    assert visible_console.lines == expected
    if replace_screen:
        assert french_console.lines == [], "the report reached the retired form"
