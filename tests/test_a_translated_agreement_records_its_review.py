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


@pytest.fixture
def recorded_terms(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import terms

    store = QSettings(str(tmp_path / "terms.ini"), QSettings.Format.IniFormat)
    store.setValue("onboarding/terms_agreed_version", "4.3")
    store.setValue("onboarding/terms_agreed_at", "2026-01-01T12:00:00+00:00")
    store.sync()
    monkeypatch.setattr(terms, "_settings", lambda: store)
    return terms, store


def _stored_values(store):
    return {key: store.value(key) for key in store.allKeys()}


def test_an_unreadable_profile_language_displays_english_without_accepting(
        recorded_terms, monkeypatch):
    from spacr.qt import i18n

    terms, store = recorded_terms
    before = _stored_values(store)

    def unavailable_language():
        raise OSError("profile settings unavailable")

    monkeypatch.setattr(i18n, "current_language", unavailable_language)

    assert terms.terms_text() == "\n\n".join(terms.TERMS)
    assert _stored_values(store) == before
    assert terms.agreement_record()["version"] == "4.3"
    assert terms.agreement_record()["current_version"] == "4.4"


@pytest.mark.parametrize("unavailable", [False, True], ids=["missing", "failed"])
def test_an_unavailable_translation_displays_english_without_accepting(
        recorded_terms, clean_registry, monkeypatch, unavailable):
    terms, store = recorded_terms
    before = _stored_values(store)
    if unavailable:
        def unavailable_paragraphs(language):
            assert language == "is"
            raise OSError("translation resource unavailable")

        monkeypatch.setattr(clean_registry, "paragraphs", unavailable_paragraphs)

    assert terms.terms_text("is") == "\n\n".join(terms.TERMS)
    assert _stored_values(store) == before
    assert terms.agreement_record()["license"] == "BSD 3-Clause License"
    assert terms.agreement_record()["current_version"] == "4.4"


@pytest.mark.parametrize(
    "accepted_version,expected_license",
    [("", ""), ("4.3", "BSD 3-Clause License"), ("4.4", "")],
    ids=["never-accepted", "historic-bsd", "current-license-unreadable"],
)
def test_an_unreadable_accepted_license_does_not_become_the_current_license(
        recorded_terms, monkeypatch, accepted_version, expected_license):
    terms, store = recorded_terms
    store.setValue("onboarding/terms_agreed_version", accepted_version)
    accepted_at = "2026-01-01T12:00:00+00:00" if accepted_version else ""
    store.setValue("onboarding/terms_agreed_at", accepted_at)
    license_name = "BSD 3-Clause License" if accepted_version == "4.3" else (
        terms.LICENSE_NAME if accepted_version else "")
    store.setValue("onboarding/terms_agreed_license", license_name)
    store.sync()
    before = _stored_values(store)

    class UnreadableLicense:
        def value(self, key, default=""):
            if key == "onboarding/terms_agreed_license":
                raise OSError("accepted license is unreadable")
            return store.value(key, default)

    monkeypatch.setattr(terms, "_settings", UnreadableLicense)

    assert terms.agreement_record() == {
        "version": accepted_version,
        "accepted_at": accepted_at,
        "current_version": "4.4",
        "license": expected_license,
    }
    assert _stored_values(store) == before
