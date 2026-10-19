"""Measurement helpers accept Windows paths.

Found natively on Windows 11 2026-10-19: count_phenotypes and analyze_recruitment tested
``src.endswith('/measurements/measurements.db')``, which a backslash path never matches, so a selected
database had measurements/measurements.db appended again and was "not found". annotate_predictions took a
file name with ``split('/')[-1]``, which returns a whole Windows path, so plate/well/field parsing broke.
"""
import sqlite3

import pandas as pd

from spacr import submodules, utils


def test_count_phenotypes_accepts_a_backslash_database_path(tmp_path, monkeypatch):
    folder = tmp_path / "plate" / "measurements"
    folder.mkdir(parents=True)
    db = folder / "measurements.db"
    con = sqlite3.connect(db)
    pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["r1", "r1"], "columnID": ["c1", "c1"],
                  "value": [1, 2]}).to_sql("png_list", con, index=False)
    con.close()
    monkeypatch.setattr(submodules, "display", lambda *a, **k: None, raising=False)
    windows_style = str(db).replace("/", "\\")
    submodules.count_phenotypes({"src": windows_style, "annotation_column": "value"})
    assert (folder / "phenotype_counts.csv").is_file()


def test_annotate_predictions_reads_windows_paths(tmp_path):
    csv = tmp_path / "predictions.csv"
    pd.DataFrame({"path": [r"C:\data\plate1\data\1_A01_3_7.png", "/data/plate1/1_B02_1_2.png"],
                  "pred": [0.9, 0.1]}).to_csv(csv, index=False)
    df = utils.annotate_predictions(str(csv))
    assert df["filename"].tolist() == ["1_A01_3_7.png", "1_B02_1_2.png"]
    assert df["well"].tolist() == ["A01", "B02"] and df["object"].tolist() == ["7", "2"]
