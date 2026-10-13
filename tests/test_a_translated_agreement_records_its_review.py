"""The EULA's translated presentations, and the one rule they all obey.

`spacr/qt/terms_i18n` is fifteen statements and six of them had never run:
`register()`'s refusal, and what the three accessors answer for a language
that has no presentation. Instruction 352 named it as worth testing
BECAUSE it is small -- six lines of fifteen reads as 52.94 %, which is the
same ratio as 149 lines of 368, and a percentage cannot tell those apart.

WHY THE REFUSAL IS THE INTERESTING LINE. Section 11.4 of the agreement says
that where a translation differs from the English, the English governs. A
presentation is therefore a CONVENIENCE, and the one thing it must never
be is an anonymous one: a reader has to be able to find out whether a
human checked it or a model drafted it. `register()` enforces that by
refusing a blank review, which is the only validation in the module.
"""
from __future__ import annotations

import pytest

from spacr.qt import terms_i18n


@pytest.fixture
def clean_registry(monkeypatch):
    """A registry of its own, so a test cannot alter what the app shows."""
    monkeypatch.setattr(terms_i18n, "_TRANSLATIONS", {})
    return terms_i18n


class TestAPresentationCannotBeAnonymous:

    def test_a_blank_review_is_refused(self, clean_registry):
        with pytest.raises(ValueError, match="review status"):
            clean_registry.register("sv", ("En paragraf.",), "")

    def test_the_refusal_names_the_language(self, clean_registry):
        """A run registering nine locales has to say which one was wrong."""
        with pytest.raises(ValueError, match="^ko:"):
            clean_registry.register("ko", ("한 문단.",), "")

    def test_a_machine_draft_is_allowed_when_it_says_so(self, clean_registry):
        """The rule is that the status is RECORDED, not that it is human.

        Refusing machine drafts outright would leave nine locales with no
        presentation at all; refusing an UNMARKED one is what keeps a
        reader able to judge.
        """
        clean_registry.register("de", ("Ein Absatz.",), "machine draft, unreviewed")
        assert "machine draft" in clean_registry.review_note("de")


class TestALanguageWithNoPresentationFallsBackQuietly:

    def test_paragraphs_answers_None_rather_than_raising(self, clean_registry):
        """None means "use the English", which is always correct here."""
        assert clean_registry.paragraphs("is") is None

    def test_the_review_note_is_empty_rather_than_missing(self, clean_registry):
        """An empty string so a caller can print it without a guard."""
        assert clean_registry.review_note("is") == ""

    def test_available_is_empty_and_sorted_when_nothing_is_registered(
            self, clean_registry):
        assert clean_registry.available() == ()


class TestWhatIsRegisteredComesBack:

    def test_the_paragraphs_round_trip_in_order(self, clean_registry):
        """The English ORDER is the contract -- clause 3 must stay third."""
        clauses = ("One.", "Two.", "Three.")
        clean_registry.register("pt", clauses, "reviewed by a speaker")
        assert clean_registry.paragraphs("pt") == clauses

    def test_available_is_sorted_so_the_picker_is_stable(self, clean_registry):
        for code in ("sv", "de", "fr"):
            clean_registry.register(code, ("x",), "reviewed")
        assert clean_registry.available() == ("de", "fr", "sv")

    def test_a_second_registration_replaces_the_first(self, clean_registry):
        """Re-importing a locale module must not leave two presentations."""
        clean_registry.register("fr", ("Ancien.",), "reviewed")
        clean_registry.register("fr", ("Nouveau.",), "reviewed again")
        assert clean_registry.paragraphs("fr") == ("Nouveau.",)
        assert clean_registry.available() == ("fr",)

    def test_the_code_is_stored_as_text(self, clean_registry):
        """`available()` sorts, and sorting mixed types raises."""
        clean_registry.register(12, ("x",), "reviewed")
        assert clean_registry.available() == ("12",)
