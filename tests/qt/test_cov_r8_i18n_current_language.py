"""`current_language`, and a caching scope nothing ever opens.

The function answers the active language without creating an import
cycle -- `preferences` imports i18n, so i18n reaches back for
`get_language` lazily and falls back to the default if that fails.

Its ContextVar cache WAS dead, and is not any more. `_RESOLVED_LANGUAGE`
was declared and read here and set nowhere in the package, so both arms
that consult it were unreachable and `current_language` re-read the
preference on every call. `i18n.ui_language_resolved_once` now opens it,
and `settings_model.language_resolved_once` opens that in turn -- so one
scope covers both this module's `tr` and settings_model's own dicts.

The numbers this exists for. The Mask screen's build was measured asking
the preference store what language the interface was in 3,516 times for
1,538 settings; the scope in settings_model fixed that side. Building
Preferences was still asking 346 times through 415 QSettings reads,
because `tr` does not go through settings_model at all -- arming this
ContextVar took that build from 55 ms to 30 ms and the reads from 415
to 70.
"""
from __future__ import annotations

import pytest

from spacr.qt import i18n as I


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    monkeypatch.delenv(I.ENV_LANGUAGE, raising=False)
    yield


class TestAnsweringTheActiveLanguage:

    def test_it_answers_a_known_code(self):
        code = I.current_language()
        assert isinstance(code, str) and code
        assert code == I.normalize_language(code), (
            "the answer is not in normalised form")

    def test_the_environment_overrides_the_preference(self, monkeypatch):
        """A headless run sets the language without a preference store."""
        monkeypatch.setenv(I.ENV_LANGUAGE, "de")
        assert I.current_language() == "de"

    def test_an_unknown_environment_value_falls_back(self, monkeypatch):
        monkeypatch.setenv(I.ENV_LANGUAGE, "not-a-language")
        assert I.current_language() == I.DEFAULT_LANGUAGE

    def test_a_preference_store_that_will_not_answer_falls_back(self,
                                                                monkeypatch):
        """THE IMPORT CYCLE THIS FUNCTION EXISTS FOR.

        `preferences` imports i18n, so i18n reaches back for
        `get_language` lazily. If that import or call fails, the default
        language is used -- a GUI that would not start because it could
        not decide on English is worse than one that guesses English.
        """
        import builtins

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "preferences" in name or "get_language" in (fromlist or ()):
                raise ImportError("preferences is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert I.current_language() == I.DEFAULT_LANGUAGE


class TestTheContextVarCacheThatIsNowOpened:
    """`_RESOLVED_LANGUAGE` is read here and set by `ui_language_resolved_once`.

    Both arms that consult it are live. The previous version of this class
    pinned the opposite -- that nothing in the package set it -- and said
    a future caller wiring it up would want a description to check
    against. This is that description.
    """

    def test_exactly_one_place_in_the_package_sets_the_context_var(self):
        """One setter, and it is the context manager.

        Kept as a whole-package sweep rather than a call to the manager,
        because the thing worth catching is a SECOND setter: two scopes
        that both reset the same ContextVar would each drop the other's
        cache, and the symptom would be a silent return to re-reading the
        preference store rather than anything that fails.
        """
        import pathlib
        import re

        root = pathlib.Path(I.__file__).resolve().parent.parent
        setters = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"_RESOLVED_LANGUAGE\s*\.\s*set\b", text):
                setters.append(path.name)
        assert setters == ["i18n.py"], (
            f"_RESOLVED_LANGUAGE is set in {setters}; it should be set only "
            "by ui_language_resolved_once")

    def test_the_scope_is_empty_outside_the_context_manager(self):
        assert I._RESOLVED_LANGUAGE.get() is None

    def test_the_manager_opens_the_scope_and_closes_it(self):
        with I.ui_language_resolved_once():
            assert I._RESOLVED_LANGUAGE.get() is not None
            code = I.current_language()
            assert I._RESOLVED_LANGUAGE.get() == {"code": code}
        assert I._RESOLVED_LANGUAGE.get() is None

    def test_the_language_is_read_once_inside_the_scope(self, monkeypatch):
        """The point of the whole thing, as a count.

        `get_language` is what reaches QSettings. Inside the scope it is
        asked once no matter how many strings are translated; outside it,
        once per string.
        """
        from spacr.qt import preferences as P

        calls = []
        real = P.get_language
        monkeypatch.setattr(P, "get_language",
                            lambda: (calls.append(1), real())[1])

        calls.clear()
        for _ in range(20):
            I.current_language()
        assert len(calls) == 20, "unscoped calls should each read the store"

        calls.clear()
        with I.ui_language_resolved_once():
            for _ in range(20):
                I.current_language()
        assert len(calls) == 1, f"scoped calls read the store {len(calls)}x"

    def test_a_nested_scope_does_not_drop_the_outer_cache(self):
        """Nesting is the normal case, not an edge one.

        A screen wraps its whole panel build and a helper wraps itself.
        The inner scope must leave the ContextVar alone on the way out or
        the outer build starts re-reading the store half way through.
        """
        with I.ui_language_resolved_once():
            outer = I._RESOLVED_LANGUAGE.get()
            I.current_language()
            with I.ui_language_resolved_once():
                assert I._RESOLVED_LANGUAGE.get() is outer
            assert I._RESOLVED_LANGUAGE.get() is outer
        assert I._RESOLVED_LANGUAGE.get() is None

    def test_a_raising_body_still_closes_the_scope(self):
        with pytest.raises(ValueError):
            with I.ui_language_resolved_once():
                raise ValueError("boom")
        assert I._RESOLVED_LANGUAGE.get() is None

    def test_the_environment_still_wins_inside_a_scope(self, monkeypatch):
        """The override is consulted before the scope is filled.

        A headless run sets the language by environment; caching must not
        let a persisted preference outrank it.
        """
        monkeypatch.setenv(I.ENV_LANGUAGE, "de")
        with I.ui_language_resolved_once():
            assert I.current_language() == "de"

    def test_the_settings_model_scope_opens_this_one_too(self):
        """One scope, both caches -- which is why callers only open one."""
        from spacr.qt.screens import settings_model as SM

        with SM.language_resolved_once():
            assert I._RESOLVED_LANGUAGE.get() is not None
            assert SM._LANGUAGE_SCOPE is not None
        assert I._RESOLVED_LANGUAGE.get() is None
        assert SM._LANGUAGE_SCOPE is None

    def test_the_live_language_cache_is_elsewhere(self):
        """`settings_model.language_resolved_once` is the one in use.

        It keeps its own dict, nests (a screen wraps its panel build and
        `build_sections` wraps itself), and is discarded when the
        outermost scope exits so a later build sees a language change.
        """
        from spacr.qt.screens import settings_model as SM

        assert hasattr(SM, "language_resolved_once")
        assert hasattr(SM, "_LANGUAGE_SCOPE")

    def test_supplying_a_scope_by_hand_would_be_honoured(self):
        """The arms are dead, not broken -- shown without pretending the
        program can reach them.

        This sets the ContextVar directly, which nothing in spaCR does.
        It documents what the code WOULD do, so a future caller that
        wires it up has a description to check against.
        """
        token = I._RESOLVED_LANGUAGE.set({"code": "fr"})
        try:
            assert I.current_language() == "fr"
        finally:
            I._RESOLVED_LANGUAGE.reset(token)

    def test_an_empty_scope_is_filled_by_the_first_call(self):
        token = I._RESOLVED_LANGUAGE.set({})
        try:
            code = I.current_language()
            assert I._RESOLVED_LANGUAGE.get() == {"code": code}
        finally:
            I._RESOLVED_LANGUAGE.reset(token)
