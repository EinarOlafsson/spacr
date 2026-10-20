"""Installed-release cache review precedes the report notice and tour."""
import json
from types import MethodType, SimpleNamespace

import pytest

from spacr import install_profile
from spacr.qt import app as app_module
from spacr.qt import install_consent


class SettingsStore:
    def __init__(self, path):
        self.path = path
        self.values = {}
        self.syncs = 0

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.syncs += 1
        self.path.write_text(json.dumps(self.values), encoding="utf-8")


class FinishedSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, result):
        for callback in self.callbacks:
            callback(result)


@pytest.fixture
def startup(monkeypatch, tmp_path):
    store = SettingsStore(tmp_path / "isolated-settings.json")
    events = []
    state = {"version": "1.5.1.5", "profile": None, "notice": None}
    window = SimpleNamespace(_closing=False, _cache_cleanup_busy=False)
    window._resolve_version = lambda: state["version"]
    window._review_installation_caches = lambda: events.append("cache-preview")
    window._tour_timer = SimpleNamespace(start=lambda delay: events.append(("tour", delay)))
    for name in ["_maybe_offer_installation_cache_review", "_finish_cache_cleanup_review",
                 "_continue_installer_onboarding", "_resume_the_tour"]:
        setattr(window, name, MethodType(getattr(app_module.MainWindow, name), window))
    monkeypatch.setattr(app_module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(install_profile, "read_profile", lambda: state["profile"])
    monkeypatch.setattr(install_consent, "_settings", lambda: store)

    def notice(parent):
        assert parent is window
        events.append("report-notice")
        return state["notice"]

    monkeypatch.setattr(install_consent, "maybe_show_anonymous_report_notice", notice)
    return window, state, store, events


def test_source_checkout_skips_cache_review_and_continues_notice(startup):
    window, _, store, events = startup
    window._maybe_offer_installation_cache_review()
    assert events == ["report-notice", ("tour", 500)]
    assert store.values == {} and store.syncs == 0 and not store.path.exists()


def test_frozen_first_offer_waits_before_notice_and_does_not_mark_reviewed(startup, monkeypatch):
    window, _, store, events = startup
    monkeypatch.setattr(app_module.sys, "frozen", True)
    window._maybe_offer_installation_cache_review()
    assert events == ["cache-preview"]
    assert window._cache_cleanup_install_token == "1.5.1.5|"
    assert store.values == {} and store.syncs == 0


def test_installer_profile_qualifies_without_frozen_flag(startup):
    window, state, store, events = startup
    state["profile"] = {"created_utc": "2026-10-15T08:00:00Z"}
    window._maybe_offer_installation_cache_review()
    assert events == ["cache-preview"]
    assert window._cache_cleanup_install_token == "1.5.1.5|2026-10-15T08:00:00Z"
    assert store.values == {}


def test_same_installation_token_skips_offer(startup):
    window, state, store, events = startup
    state["profile"] = {"created_utc": "profile-a"}
    store.values["installer/cache_reviewed_install"] = "1.5.1.5|profile-a"
    window._maybe_offer_installation_cache_review()
    assert events == ["report-notice", ("tour", 500)]
    assert store.syncs == 0


@pytest.mark.parametrize("version,profile,expected", [
    ("1.5.1.6", {"created_utc": "profile-a"}, "1.5.1.6|profile-a"),
    ("1.5.1.5", {"created_utc": "profile-b"}, "1.5.1.5|profile-b"),
])
def test_new_release_or_profile_reoffers_review(startup, version, profile, expected):
    window, state, store, events = startup
    state.update(version=version, profile=profile)
    store.values["installer/cache_reviewed_install"] = "1.5.1.5|profile-a"
    window._maybe_offer_installation_cache_review()
    assert events == ["cache-preview"]
    assert window._cache_cleanup_install_token == expected
    assert store.values["installer/cache_reviewed_install"] == "1.5.1.5|profile-a"
    assert store.syncs == 0


@pytest.mark.parametrize("version", ["", "dev"])
def test_development_or_unknown_version_does_not_persist_review(startup, version):
    window, state, store, events = startup
    state.update(version=version, profile={"created_utc": "profile-a"})
    window._maybe_offer_installation_cache_review()
    assert events == ["report-notice", ("tour", 500)]
    assert store.values == {} and store.syncs == 0


def test_failed_preview_has_no_marker_and_releases_state_before_onboarding(startup):
    window, state, store, events = startup
    state["profile"] = {"created_utc": "profile-a"}
    window._maybe_offer_installation_cache_review()
    window._cache_cleanup_busy = True
    window._cache_cleanup_guard = object()
    window._finish_cache_cleanup_review(reviewed=False)
    assert events == ["cache-preview", "report-notice", ("tour", 500)]
    assert store.values == {} and store.syncs == 0 and not store.path.exists()
    assert window._cache_cleanup_busy is False and window._cache_cleanup_guard is None
    assert window._cache_cleanup_install_token is None and window._cache_cleanup_after is None


def test_keep_after_displayed_choices_records_installation_and_then_shows_notice(startup):
    window, state, store, events = startup
    state["profile"] = {"created_utc": "profile-a"}
    window._maybe_offer_installation_cache_review()
    window._finish_cache_cleanup_review(reviewed=True)
    assert events == ["cache-preview", "report-notice", ("tour", 500)]
    assert store.values == {"installer/cache_reviewed_install": "1.5.1.5|profile-a"}
    assert json.loads(store.path.read_text(encoding="utf-8")) == store.values and store.syncs == 1
    assert window._cache_cleanup_install_token is None and window._cache_cleanup_after is None


@pytest.mark.parametrize("reviewed", [False, True])
def test_late_review_finish_during_close_neither_marks_nor_continues(startup, reviewed):
    window, state, store, events = startup
    state["profile"] = {"created_utc": "profile-a"}
    window._maybe_offer_installation_cache_review()
    window._cache_cleanup_busy = True
    window._cache_cleanup_guard = object()
    window._closing = True
    window._finish_cache_cleanup_review(reviewed=reviewed)
    assert events == ["cache-preview"]
    assert store.values == {} and store.syncs == 0 and not store.path.exists()
    assert window._cache_cleanup_busy is False and window._cache_cleanup_guard is None
    assert window._cache_cleanup_install_token is None and window._cache_cleanup_after is None


def test_report_notice_finishes_before_tour_timer_starts(startup):
    window, state, _, events = startup
    finished = FinishedSignal()
    state["notice"] = SimpleNamespace(finished=finished)
    window._continue_installer_onboarding()
    assert events == ["report-notice"] and len(finished.callbacks) == 1
    finished.emit(0)
    assert events == ["report-notice", ("tour", 500)]


def test_closing_window_does_not_start_report_notice_or_tour(startup):
    window, _, store, events = startup
    window._closing = True
    window._continue_installer_onboarding()
    assert events == [] and store.values == {}
