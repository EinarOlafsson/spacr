"""The Dose-Response screen's report when several analyses speak at once.

Pinned here, through the pickers and ``fit()`` on a synchronous screen:

* with no group column, the plates are pooled as one curve ("all rows"), and
  a pool the engine refused is named with its reason;
* pooled EC50s and a selectivity index are separated by a blank line, and a
  one-sided index carries the engine's note;
* a checkerboard split by a group column is scored per group, and its
  surfaces are set off from the selectivity lines above them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen  # noqa: E402
from spacr.qt.widgets.dose_response import (  # noqa: E402
    STATUS_REFUSED,
    STATUS_UNBOUNDED,
    four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = 27.0 / 3.0 ** np.arange(10)


def _plate(name, rng, *, flat=False, host_ec50=None):
    dose = np.repeat(DOSES, 3)
    signal = four_parameter_logistic(dose, 0.0, 100.0, 0.0, 1.0)
    if flat:
        signal = np.full(dose.size, 50.0)
    columns = {"plate": name, "gene": "geneA", "conc_uM": dose,
               "signal": signal + rng.normal(0.0, 1.0, dose.size)}
    if host_ec50 is not None:
        host = four_parameter_logistic(dose, 0.0, 100.0,
                                       np.log10(host_ec50), -1.0)
        columns["host"] = host + rng.normal(0.0, 1.0, dose.size)
    return pd.DataFrame(columns)


def _screen(qtbot, frame, concentration, response):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="plates")
    widget.concentration_picker.setCurrentText(concentration)
    widget.response_picker.setCurrentText(response)
    return widget


def _lines(widget):
    return widget.report.toPlainText().splitlines()


def test_with_no_group_a_refused_pool_is_named_for_all_rows(qtbot):
    rng = np.random.default_rng(3)
    frame = pd.concat([_plate("P1", rng), _plate("P2", rng, flat=True)],
                      ignore_index=True)
    widget = _screen(qtbot, frame, "conc_uM", "signal")
    assert widget.spec().group is None, "one curve for the whole table"
    widget.plate_picker.setCurrentText("plate")

    widget.fit()

    pooled = widget._pooled[""]
    assert not isinstance(pooled, str)
    assert pooled.status == STATUS_REFUSED
    line = next(line for line in _lines(widget)
                if line.startswith("all rows: not pooled — "))
    assert line.endswith(pooled.note)


def test_pooled_and_selectivity_lines_are_kept_apart(qtbot):
    rng = np.random.default_rng(3)
    frame = pd.concat([_plate("P1", rng, host_ec50=20.0),
                       _plate("P2", rng, host_ec50=20.0)],
                      ignore_index=True)
    widget = _screen(qtbot, frame, "conc_uM", "signal")
    widget.plate_picker.setCurrentText("plate")
    widget.host_picker.setCurrentText("host")

    widget.fit()

    index = widget._selectivity[""]
    assert index.status == STATUS_UNBOUNDED
    lines = _lines(widget)
    pooled_at = next(i for i, line in enumerate(lines)
                     if line.startswith("all rows: pooled EC50 "))
    si_at = next(i for i, line in enumerate(lines)
                 if line.startswith("all rows: selectivity index "))
    assert lines[si_at - 1] == "" and si_at > pooled_at
    assert lines[si_at].endswith(f" — {index.note}")
    assert "host EC50 is not bounded" in lines[si_at]


def _board(pair):
    rows = []
    rng = np.random.default_rng(len(pair))
    doses = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    for a in doses:
        for b in doses:
            ea = four_parameter_logistic(max(a, 1e-12), 0.0, 1.0, 0.0, 1.0)
            eb = four_parameter_logistic(max(b, 1e-12), 0.0, 1.0,
                                         np.log10(3.0), 1.0)
            survival = 1.0 - (ea + eb - ea * eb)
            for _ in range(3):
                noisy = survival + rng.normal(0.0, 0.002)
                rows.append({"pair": pair, "cmpd_a_uM": a, "cmpd_b_uM": b,
                             "survival": noisy, "host": noisy})
    return pd.DataFrame(rows)


def test_a_grouped_checkerboard_is_scored_per_group_below_the_index(qtbot):
    frame = pd.concat([_board("AB"), _board("ABC")], ignore_index=True)
    widget = _screen(qtbot, frame, "cmpd_a_uM", "survival")
    widget.group_picker.setCurrentText("pair")
    widget.second_dose_picker.setCurrentText("cmpd_b_uM")
    widget.host_picker.setCurrentText("host")

    widget.fit()

    assert set(widget._synergy) == {"AB", "ABC"}
    assert all(not isinstance(surface, str)
               for surface in widget._synergy.values())
    lines = _lines(widget)
    first_surface = next(i for i, line in enumerate(lines)
                         if " Bliss excess over " in line)
    assert lines[first_surface - 1] == "", (
        "the surfaces are set off from the selectivity lines above them")
    assert any(line.startswith("AB: selectivity index ")
               or line.startswith("AB: no selectivity index ")
               for line in lines[:first_surface])


def test_a_surface_with_no_scorable_well_says_so_with_its_note(
        qtbot, monkeypatch):
    """A Loewe surface is NaN wherever the observed effect lies outside a
    single agent's plateaus; when that is every well the report says no
    well could be scored, with the engine's note, instead of a max and min
    over nothing. The engine's answer is stood in for, the screen is real."""
    from spacr.qt.screens import dose_response as screen_module
    from spacr.qt.widgets.dose_response import InteractionSurface

    def _nothing_scored(dose_a, dose_b, response, *, fit_a, fit_b):
        shape = (2, 2)
        return InteractionSurface(
            model="bliss", dose_a=np.array([0.0, 1.0]),
            dose_b=np.array([0.0, 1.0]), observed=np.full(shape, np.nan),
            expected=np.full(shape, np.nan), excess=np.full(shape, np.nan),
            n_cells=0, note="every effect lies outside the plateaus")

    monkeypatch.setattr(screen_module, "bliss_surface", _nothing_scored)
    widget = _screen(qtbot, _board("AB"), "cmpd_a_uM", "survival")
    widget.second_dose_picker.setCurrentText("cmpd_b_uM")

    widget.fit()

    assert widget._synergy[""].n_cells == 0
    assert ("all rows: no combination well could be scored against Bliss"
            " — every effect lies outside the plateaus") in _lines(widget)
    assert not any(" excess over " in line for line in _lines(widget))
