"""`current_language`, and a caching scope nothing ever opens.

The function answers the active language without creating an import
cycle -- `preferences` imports i18n, so i18n reaches back for
`get_language` lazily and falls back to the default if that fails.

Its ContextVar cache is dead. `_RESOLVED_LANGUAGE` is declared and READ
here and set nowhere in the package: the scope that actually caches
language lookups during a panel build is
`settings_model.language_resolved_once`, which keeps its own dict. So
both arms that consult the ContextVar are unreachable, and
`current_language` re-reads the preference on every call.

That is worth knowing rather than merely recording. The Mask screen's
build was measured asking the preference store what language the
interface was in 3,516 times for 1,538 settings; the scope in
settings_model is what fixed it, and this one looks like an earlier
attempt that was never wired up.
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


class TestTheContextVarCacheThatIsNeverOpened:
    """`_RESOLVED_LANGUAGE` is read here and set nowhere.

    Both arms that consult it are therefore unreachable. Pinned to the
    fact that makes them so, and recorded because the live cache is a
    DIFFERENT mechanism in a different module.
    """

    def test_nothing_in_the_package_sets_the_context_var(self):
        """If something ever does, the two arms become live."""
        import pathlib
        import re

        root = pathlib.Path(I.__file__).resolve().parent.parent
        setters = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"_RESOLVED_LANGUAGE\s*\.\s*set\b", text):
                setters.append(str(path))
        assert setters == [], (
            f"_RESOLVED_LANGUAGE is now set in {setters}; the caching arms "
            "in current_language are live and want tests")

    def test_the_scope_is_empty_during_an_ordinary_call(self):
        assert I._RESOLVED_LANGUAGE.get() is None

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
