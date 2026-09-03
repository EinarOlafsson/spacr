"""A figure bundle records what it could, and refuses to overwrite an
earlier one.

The bundle is the reproducibility half of a saved figure: the picture, the
rows behind it, the test, and the settings. Three edges decide whether it can
be trusted -- a corrected p-value has to appear beside the raw one or the
reader applies the wrong threshold; a renderer that fails must not take the
data and statistics down with it; and a directory that already exists is
never replaced, because a folder replaced silently loses an earlier save.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from spacr.figures import bundle
from spacr.figures.stats import Comparison


def _comparison(**extra) -> Comparison:
    kwargs = dict(test="t-test", statistic=2.5, p_value=0.01,
                  groups=["a", "b"], n=[6, 6], unit="well")
    kwargs.update(extra)
    return Comparison(**kwargs)


def test_a_corrected_p_value_is_written_beside_the_raw_one():
    """Reporting only the raw p would apply the uncorrected threshold."""
    for field in ("unit", "effect_size", "effect_name", "ci", "assumptions",
                  "reason", "correction", "p_adjusted"):
        assert f":ivar {field}:" in (Comparison.__doc__ or "")
    rows = bundle.statistics_rows(
        _comparison(p_adjusted=0.04, correction="Holm"))
    adjusted = [row for row in rows if row[0] == "p_adjusted"]
    assert adjusted == [("p_adjusted", 0.04, "Holm")], rows


def test_no_correction_writes_no_corrected_row():
    """An absent correction must not appear as a number the reader trusts."""
    rows = bundle.statistics_rows(_comparison())
    assert not [row for row in rows if row[0] == "p_adjusted"], rows


def test_a_renderer_that_fails_still_leaves_the_data_and_statistics(tmp_path):
    """The numbers behind a figure outlive the picture that failed to draw."""
    def _explode(_path):
        raise RuntimeError("no backend")

    out = bundle.save(str(tmp_path), "my graph", render=_explode,
                      data=pd.DataFrame({"value": [1, 2, 3]}))
    written = sorted(os.listdir(out))
    assert written == ["data.csv", "settings.json", "statistics.csv"], written
    assert pd.read_csv(Path(out) / "data.csv").shape[0] == 3


def test_a_second_save_sits_beside_the_first(tmp_path):
    """Replacing the folder would take the earlier save's data with it."""
    first = bundle.save(str(tmp_path), "graph", render=lambda p: Path(p).touch())
    second = bundle.save(str(tmp_path), "graph", render=lambda p: Path(p).touch())
    assert first != second
    assert os.path.isdir(first) and os.path.isdir(second)


def test_a_thousand_existing_folders_is_refused_not_silently_reused(tmp_path,
                                                                    monkeypatch):
    """Running out of suffixes must never resolve to overwriting one."""
    monkeypatch.setattr(bundle.os.path, "exists", lambda _p: True)
    with pytest.raises(FileExistsError) as excinfo:
        bundle._unique(str(tmp_path / "graph"))
    assert "graph" in str(excinfo.value)


def test_a_settings_value_json_cannot_hold_is_recorded_as_its_text(tmp_path):
    """An unserialisable setting is still evidence of how the graph was made."""
    class _Opaque:
        def __str__(self):
            return "threshold=0.5"

    out = bundle.save(str(tmp_path), "graph", render=lambda p: Path(p).touch(),
                      settings={"gate": _Opaque(), "keep": {1, 2}})
    payload = json.loads((Path(out) / "settings.json").read_text())
    assert payload["gate"] == "threshold=0.5"
    assert isinstance(payload["keep"], str)
