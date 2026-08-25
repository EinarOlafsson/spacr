"""First-run setup — the four places it is allowed to give up quietly.

Every question on this screen has a working default, and the module's rule
is that nothing about the questions may take a launch down. That rule only
has meaning where the fallback is exercised, so each one is reached here by
breaking the thing it guards:

* the version lookup, when ``spacr`` has no ``__version__`` -- which is what
  a partially installed or vendored copy looks like;
* the language list, when the translation table cannot be imported. The
  fallback is English alone, and the screen must still open;
* the provider list, when the AI provider registry raises. The fallback is
  the empty list, and an empty list REMOVES the question rather than asking
  the user to choose between providers none of which exist;
* :func:`current`, when one preference getter raises -- the key it belongs
  to is dropped and the other answers still populate the screen.
"""
from __future__ import annotations


import spacr
from spacr.qt import preferences as prefs
from spacr.qt import setup_screen


# ---------------------------------------------------------------------------
# current_version
# ---------------------------------------------------------------------------

def test_the_running_version_is_reported():
    assert setup_screen.current_version() == str(spacr.__version__)


def test_a_package_with_no_version_is_unknown_not_a_crash(monkeypatch):
    """``from .. import __version__`` raises ImportError with no attribute."""
    monkeypatch.delattr(spacr, "__version__")
    assert setup_screen.current_version() == "unknown"


def test_an_unknown_version_still_decides_whether_to_open(monkeypatch):
    monkeypatch.delattr(spacr, "__version__")
    setup_screen.mark_answered("unknown")
    assert setup_screen.should_open() is False
    setup_screen.mark_answered("0.0.1")
    assert setup_screen.should_open() is True


# ---------------------------------------------------------------------------
# _language_choices
# ---------------------------------------------------------------------------

def test_every_language_is_offered_in_its_own_script():
    choices = setup_screen._language_choices()
    codes = [code for code, _native in choices]
    assert "en" in codes
    assert len(choices) > 1
    assert all(native for _code, native in choices)


def test_a_broken_translation_table_leaves_english(monkeypatch):
    """The screen must open even when the language list cannot be built."""
    from spacr.qt import i18n

    monkeypatch.delattr(i18n, "LANGUAGES")
    assert setup_screen._language_choices() == [("en", "English")]


def test_the_language_question_survives_a_broken_translation_table(
        monkeypatch):
    from spacr.qt import i18n

    monkeypatch.delattr(i18n, "LANGUAGES")
    asked = {row[0]: row for row in setup_screen.questions()}
    assert asked["language"][4] == [("en", "English")]


# ---------------------------------------------------------------------------
# _provider_choices
# ---------------------------------------------------------------------------

def test_a_provider_registry_that_raises_removes_the_question(monkeypatch):
    """No providers is not a question with no true answers -- it is no
    question at all."""
    from spacr.qt.ai import providers

    def explode():
        raise RuntimeError("provider discovery failed")

    monkeypatch.setattr(providers, "list_providers", explode)

    assert setup_screen._provider_choices() == []
    assert "ai_provider" not in [row[0] for row in setup_screen.questions()]


def test_a_registry_of_nameless_providers_also_removes_the_question(
        monkeypatch):
    """A provider whose name is blank cannot be offered or stored."""
    from spacr.qt.ai import providers

    class Nameless:
        name = ""

    monkeypatch.setattr(providers, "list_providers", lambda: [Nameless()])
    assert setup_screen._provider_choices() == []


def test_whatever_is_available_leads_the_provider_list(monkeypatch):
    """A pinned name that has gone away is worse than no preference."""
    from spacr.qt.ai import providers

    class Named:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(providers, "list_providers",
                        lambda: [Named("claude_code"), Named("codex")])
    choices = setup_screen._provider_choices()
    assert choices[0] == ("", "whatever is available")
    assert ("claude_code", "claude code") in choices


# ---------------------------------------------------------------------------
# current / apply
# ---------------------------------------------------------------------------

def test_one_unreadable_setting_does_not_empty_the_screen(monkeypatch):
    """A getter that raises drops its own key and no other."""
    def explode():
        raise RuntimeError("preference store is unreadable")

    monkeypatch.setattr(prefs, "get_hash_inputs", explode)

    answers = setup_screen.current()
    assert "hash_inputs" not in answers
    assert "language" in answers
    assert "theme" in answers


def test_every_question_answers_itself_by_default():
    answers = setup_screen.current()
    keys = [row[0] for row in setup_screen.questions()]
    assert set(answers) == set(keys)


def test_one_refused_answer_does_not_lose_the_others(monkeypatch):
    """``apply`` reports the failure and still writes everything else."""
    def refuse(_value):
        raise ValueError("not a mode")

    monkeypatch.setattr(prefs, "set_spacr_mode", refuse)

    trouble = setup_screen.apply({"spacr_mode": "nonsense",
                                  "hash_inputs": True})
    assert trouble == ["spacr_mode: not a mode"]
    assert prefs.get_hash_inputs() is True
