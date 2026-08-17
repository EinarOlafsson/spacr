"""The `volcano` setting is gone, and an old settings file still loads.

Asked for 2026-08-17: "remove the Volcano setting in regression, it is now
redundant".

It chose which coefficient table the volcano was drawn from -- 'gene' |
'grna' | 'all' -- BEFORE the run. Instruction 129 A moved that choice onto
the plot: the interactive volcano filters between genes and guides by
right-click, on the same fit, with no re-run. A setting that could answer the
question once was redundant the moment that landed.

It replaces tests/test_the_volcano_default_is_gene.py, which pinned the
default this removes. That default (`gene`) is now simply what the code does.
"""
from __future__ import annotations

import inspect
import pathlib


def _regression_defaults(overrides=None):
    from spacr.settings import get_perform_regression_default_settings

    return get_perform_regression_default_settings(dict(overrides or {}))


# --------------------------------------------------------------------------- #
#  Gone
# --------------------------------------------------------------------------- #

def test_a_fresh_run_has_no_volcano_setting():
    assert "volcano" not in _regression_defaults()


def test_nothing_in_the_regression_reads_it():
    """A key nobody sets but somebody still reads is a KeyError waiting for
    the first old settings file."""
    import spacr.ml

    source = pathlib.Path(spacr.ml.__file__).read_text()
    assert "settings['volcano']" not in source
    assert "settings.get('volcano')" not in source


def test_it_is_out_of_the_documented_settings():
    """A described setting that does not exist is worse than an undocumented
    one: a reader sets it and nothing happens."""
    import spacr.settings

    source = pathlib.Path(spacr.settings.__file__).read_text()
    assert "'volcano': \"(str) - Which coefficient table" not in source


# --------------------------------------------------------------------------- #
#  And an old settings file still works
# --------------------------------------------------------------------------- #

def test_an_old_settings_file_still_loads():
    """Every regression run before 2026-08-17 wrote `volcano` into its
    settings CSV. A saved file that suddenly fails to load is a worse outcome
    than a key nothing reads."""
    settings = _regression_defaults({"volcano": "grna",
                                     "regression_type": "ols"})

    assert "volcano" not in settings
    assert settings["regression_type"] == "ols"


def test_the_rest_of_an_old_file_survives():
    """Dropping the retired key must not drop anything beside it."""
    settings = _regression_defaults({
        "volcano": "all", "regression_type": "rlm", "fdr_alpha": 0.01,
        "min_cell_count": 250})

    assert settings["regression_type"] == "rlm"
    assert settings["fdr_alpha"] == 0.01
    assert settings["min_cell_count"] == 250


# --------------------------------------------------------------------------- #
#  What the removal must NOT have taken with it
# --------------------------------------------------------------------------- #

def test_the_gene_list_is_still_produced():
    """`custom_volcano_plot` was called through that branch and also RETURNS
    the hit list the GT1 phenotype plot and the ME49 transcription heatmap
    are built from. Collapsing the branch to `gene` keeps it; deleting the
    call would have silently removed two reports."""
    import spacr.ml

    source = inspect.getsource(spacr.ml.perform_regression)
    assert "gene_list = custom_volcano_plot(" in source
    assert "gene_merged_df" in source


def test_the_legacy_gate_still_applies():
    """The old volcano stays off by default -- removing its table selector
    must not have removed the boolean that suppresses it."""
    import spacr.ml

    source = inspect.getsource(spacr.ml.perform_regression)
    assert "draw=draw_legacy_volcano" in source


def test_the_interactive_filter_is_what_replaced_it():
    """The justification, asserted: the choice this setting made is offered
    on the plot instead."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    levels = {key for key, _label in RegressionResultsPanel.LEVELS}
    assert levels == {None, "gene", "grna"}
