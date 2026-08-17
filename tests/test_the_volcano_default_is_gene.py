"""A run that says nothing about `volcano` draws the GENE volcano.

Asked for on 2026-08-17: "in regression make gene the default of volcano".

The code and its own documentation disagreed: settings.py set 'grna' while
the docstring for the same setting has always said "'gene' (default)". So
the request and the documentation agreed and the code was the odd one out.
"""
from __future__ import annotations


def test_the_default_is_gene():
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({})

    assert settings["volcano"] == "gene"


def test_an_explicit_choice_still_wins():
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({"volcano": "grna"})

    assert settings["volcano"] == "grna"


def test_the_documentation_agrees_with_the_code():
    """They disagreed for as long as the setting has existed, which is how a
    default gets reported as a bug -- the docstring said "'gene' (default)"
    and the code set 'grna'.

    Read from the settings module's own text rather than through a helper:
    `descriptions` is a dict in some builds and a callable in others, and a
    test that silently got "" from the wrong one would assert nothing.
    """
    import pathlib

    import spacr.settings

    text = pathlib.Path(spacr.settings.__file__).read_text()
    entry = text.split("'volcano': \"", 1)[1].split('",', 1)[0]

    assert "'gene' (default)" in entry, entry[:200]
    assert "setdefault('volcano', 'gene')" in text
