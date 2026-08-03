"""Explicit cardinality contracts for object-table merges."""

import sqlite3

import pandas as pd
import pytest

from spacr.io import MergeCardinalityError, _merge_with_cardinality
from spacr.utils import (_merge_and_save_to_database,
                         _update_database_with_merged_info, merge_dataframes)


def test_many_to_one_allows_repeated_object_metadata_keys():
    """Many objects can share a well, but the well-count table stays unique."""
    metadata = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1", "plate1_r1_c2"],
        "object_label": [1, 2, 1],
    })
    counts = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c2"],
        "cells_per_well": [2, 1],
    })

    merged = _merge_with_cardinality(
        metadata,
        counts,
        on="prc",
        validate="many_to_one",
        left_name="object metadata",
        right_name="well counts",
    )

    assert list(merged["cells_per_well"]) == [2, 2, 1]


def test_many_to_one_names_duplicated_right_keys():
    metadata = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1"],
        "object_label": [1, 2],
    })
    counts = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1"],
        "cells_per_well": [2, 99],
    })

    with pytest.raises(MergeCardinalityError) as excinfo:
        _merge_with_cardinality(
            metadata,
            counts,
            on="prc",
            validate="many_to_one",
            left_name="object metadata",
            right_name="well counts",
        )

    message = str(excinfo.value)
    assert "validate='many_to_one'" in message
    assert "well counts has duplicated ['prc']" in message
    assert "plate1_r1_c1" in message


def test_one_to_one_names_duplicated_indexes_on_both_sides():
    left = pd.DataFrame({"area": [10, 11]}, index=["key1", "key1"])
    right = pd.DataFrame({"intensity": [1, 2]}, index=["key1", "key1"])

    with pytest.raises(MergeCardinalityError) as excinfo:
        _merge_with_cardinality(
            left,
            right,
            left_index=True,
            right_index=True,
            validate="one_to_one",
            left_name="grouped cell data",
            right_name="grouped nucleus data",
        )

    message = str(excinfo.value)
    assert "grouped cell data has duplicated index" in message
    assert "grouped nucleus data has duplicated index" in message
    assert "key1" in message


def test_measurement_writer_rejects_duplicate_object_labels(tmp_path):
    """Morphology/intensity fan-out is rejected before rows reach SQLite."""
    morphology = pd.DataFrame({
        "label": [1, 1],
        "cell_area": [10.0, 11.0],
    })
    intensity = pd.DataFrame({
        "label": [1],
        "cell_channel_0_mean_intensity": [3.0],
    })

    with pytest.raises(
            pd.errors.MergeError,
            match="not a one-to-one merge"):
        _merge_and_save_to_database(
            morphology,
            intensity,
            "cell",
            str(tmp_path),
            "plate1_A01_1",
            "experiment",
        )

    assert not (tmp_path / "measurements" / "measurements.db").exists()


def test_database_metadata_update_rejects_duplicate_source_keys(tmp_path):
    """Updating a table must not multiply its existing rows."""
    db_path = tmp_path / "measurements.db"
    existing = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "png_path": ["one.png", "two.png"],
    })
    with sqlite3.connect(db_path) as connection:
        existing.to_sql("png_list", connection, index=False)

    annotations = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1"],
        "condition": ["control", "treated"],
    })
    with pytest.raises(
            pd.errors.MergeError,
            match="not a many-to-one merge"):
        _update_database_with_merged_info(
            str(db_path),
            annotations,
            columns=["condition", "prcfo"],
        )

    with sqlite3.connect(db_path) as connection:
        unchanged = pd.read_sql_query("SELECT * FROM png_list", connection)
    pd.testing.assert_frame_equal(unchanged, existing)


def test_image_feature_merge_rejects_duplicate_feature_keys():
    """One feature vector, at most, may be attached to each crop key."""
    image_paths = pd.DataFrame(
        {"png_path": ["one.png"]},
        index=["plate1_r1_c1_f1_o1"],
    )
    features = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1"],
        "area": [10.0, 11.0],
    })

    with pytest.raises(
            pd.errors.MergeError,
            match="not a many-to-one merge"):
        merge_dataframes(features, image_paths, verbose=False)


# ===========================================================================
# Key contracts on the joins that were still unguarded
#
# Every test below constructs a duplicated join key and asserts the merge
# REFUSES. Without the matching ``validate=`` each one of them passes
# silently and returns more rows than went in -- duplicated cells,
# duplicated tracks, duplicated regression hits.
# ===========================================================================

def _png_frame(prcfos, paths=None):
    return pd.DataFrame({
        "prcfo": list(prcfos),
        "png_path": list(paths) if paths is not None
                    else [f"/src/{p}.png" for p in prcfos],
    })


# --- gui_elements.AnnotateApp.prefilter_paths_annotations -------------------

def _annotate_app(**attrs):
    """An ``AnnotateApp`` with ``__init__`` skipped and the DB touch stubbed."""
    from spacr.gui_elements import AnnotateApp
    app = object.__new__(AnnotateApp)
    app._ensure_annotation_column = lambda: None
    defaults = dict(db_path="/nowhere/measurements.db",
                    annotation_column="annotate", image_type=None,
                    measurement="cell_area", threshold=0,
                    threshold_direction="higher")
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(app, key, value)
    return app


def test_annotate_prefilter_refuses_a_duplicated_crop_key(monkeypatch):
    """Two png_list rows for one object must not double the measurement rows."""
    import spacr.io as sio

    measurements = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "cell_area": [100.0, 200.0],
    })
    # o1 was cropped twice -- a re-run of the crop step appending a second set
    # of rows, or a second crop_mode whose object labels collide.
    png_list = _png_frame(["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1",
                           "plate1_r1_c1_f1_o2"])

    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: measurements.copy())
    monkeypatch.setattr(sio, "_read_db", lambda db, tables: [png_list.copy()])

    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        _annotate_app().prefilter_paths_annotations()


def test_annotate_prefilter_accepts_one_crop_per_object(monkeypatch):
    """The healthy shape still goes through, and nothing is duplicated."""
    import spacr.io as sio

    measurements = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "cell_area": [100.0, 200.0],
    })
    png_list = _png_frame(["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"])

    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: measurements.copy())
    monkeypatch.setattr(sio, "_read_db", lambda db, tables: [png_list.copy()])

    app = _annotate_app()
    app.prefilter_paths_annotations()
    assert len(app.filtered_paths_annotations) == 2


# --- qt.annotate_engine.fetch_filtered_paths --------------------------------

def test_qt_fetch_filtered_paths_refuses_a_duplicated_crop_key(tmp_path,
                                                               monkeypatch):
    """Same join, same contract, in the Qt annotate backend."""
    import spacr.io as sio
    from spacr.qt import annotate_engine

    db_path = tmp_path / "measurements.db"
    db_path.write_bytes(b"")            # only os.path.isfile is consulted

    measurements = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "cell_area": [100.0, 200.0],
    })
    png_list = _png_frame(["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1",
                           "plate1_r1_c1_f1_o2"])

    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: measurements.copy())
    monkeypatch.setattr(sio, "_read_db", lambda db, tables: [png_list.copy()])

    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        annotate_engine.fetch_filtered_paths(
            str(db_path), "annotate", ["cell_area"], [0.0], ["higher"])


# --- plot.jitterplot_by_annotation ------------------------------------------

def test_jitterplot_refuses_a_duplicated_crop_key(monkeypatch):
    """A doubled png_list would draw every cell twice in the jitter plot."""
    import spacr.io as sio

    measurements = pd.DataFrame(
        {"recruitment": [1.0, 2.0]},
        index=pd.Index(["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
                       name="prcfo"),
    )
    png_list = _png_frame(["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o1",
                           "plate1_r1_c1_f1_o2"])

    monkeypatch.setattr(
        sio, "_read_and_merge_data",
        lambda locs, tables, **k: (measurements.copy(), []))
    monkeypatch.setattr(sio, "_read_db", lambda loc, tables: [png_list.copy()])

    from spacr.plot import jitterplot_by_annotation
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        jitterplot_by_annotation("/exp/src", "annotation", "recruitment")


# --- utils.combine_results ---------------------------------------------------

def test_combine_results_refuses_a_repeated_feature():
    """One importance row per feature, one test result per feature."""
    from spacr.utils import combine_results

    rf_df = pd.DataFrame({"Feature": ["area", "perimeter"],
                          "Importance": [0.7, 0.3]})
    # 'area' tested twice -- two runs' results concatenated, say.
    anova_df = pd.DataFrame({"Feature": ["area", "area"],
                             "ANOVA_Statistic": [1.0, 9.0],
                             "ANOVA_pValue": [0.01, 0.9]})
    kruskal_df = pd.DataFrame({"Feature": ["perimeter"],
                               "Kruskal_Statistic": [2.0],
                               "Kruskal_pValue": [0.2]})

    # 'right dataset', not 'left': the FIRST merge has to be the one that
    # refuses. Left unguarded it fans out to three rows and the failure only
    # surfaces on the second merge, which then blames the left frame for a
    # duplicate the ANOVA table introduced.
    with pytest.raises(pd.errors.MergeError,
                       match="not unique in right dataset"):
        combine_results(rf_df, anova_df, kruskal_df)


def test_combine_results_refuses_a_repeated_feature_in_the_kruskal_frame():
    """The second merge is contracted too, not only the first."""
    from spacr.utils import combine_results

    rf_df = pd.DataFrame({"Feature": ["area", "perimeter"],
                          "Importance": [0.7, 0.3]})
    anova_df = pd.DataFrame({"Feature": ["area"],
                             "ANOVA_Statistic": [1.0],
                             "ANOVA_pValue": [0.01]})
    kruskal_df = pd.DataFrame({"Feature": ["perimeter", "perimeter"],
                               "Kruskal_Statistic": [2.0, 3.0],
                               "Kruskal_pValue": [0.2, 0.3]})

    with pytest.raises(pd.errors.MergeError,
                       match="not unique in right dataset"):
        combine_results(rf_df, anova_df, kruskal_df)


def test_combine_results_keeps_one_row_per_feature():
    from spacr.utils import combine_results

    rf_df = pd.DataFrame({"Feature": ["area", "perimeter"],
                          "Importance": [0.7, 0.3]})
    anova_df = pd.DataFrame({"Feature": ["area"],
                             "ANOVA_Statistic": [1.0],
                             "ANOVA_pValue": [0.01]})
    kruskal_df = pd.DataFrame({"Feature": ["perimeter"],
                               "Kruskal_Statistic": [2.0],
                               "Kruskal_pValue": [0.2]})

    combined = combine_results(rf_df, anova_df, kruskal_df)
    assert list(combined["Feature"]) == ["area", "perimeter"]
    assert pd.isna(combined.loc[0, "Kruskal_pValue"])
    assert pd.isna(combined.loc[1, "ANOVA_pValue"])


# --- utils.merge_regression_res_with_metadata --------------------------------

def _isoform_metadata(path):
    """A metadata CSV shaped like the bundled ``toxoplasma_metadata.csv``.

    That file lists 30 Gene IDs two to four times -- one row per transcript,
    each carrying its own protein length and GO terms. Joined without a
    contract, those genes came back multiplied in every regression result.
    """
    pd.DataFrame({
        "Gene ID": ["TGME49_200320", "TGME49_200320", "TGME49_203450"],
        "Protein Length": [230.0, 279.0, 100.0],
        "Description": ["HXGPRT isoform 1", "HXGPRT isoform 2", "BCP1"],
    }).to_csv(path, index=False)
    return path


def test_regression_metadata_merge_keeps_one_row_per_feature(tmp_path, capsys):
    """The isoform rows must not multiply the regression results."""
    from spacr.utils import merge_regression_res_with_metadata

    results = tmp_path / "results.csv"
    pd.DataFrame({
        "feature": ["C(gene)[T.200320_1]", "C(gene)[T.200320_2]",
                    "C(gene)[T.203450_1]", "Intercept"],
        "coefficient": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(results, index=False)
    metadata = _isoform_metadata(tmp_path / "metadata.csv")

    merged = merge_regression_res_with_metadata(str(results), str(metadata))

    # Four regression terms in, four rows out. Before the contract this
    # returned six: both 200320 gRNAs matched both isoform rows.
    assert len(merged) == 4
    assert list(merged["coefficient"]) == [1.0, 2.0, 3.0, 4.0]
    # The first annotation row of the gene is the one that survives.
    assert merged.loc[0, "Description"] == "HXGPRT isoform 1"
    assert merged.loc[1, "Description"] == "HXGPRT isoform 1"
    assert merged.loc[2, "Description"] == "BCP1"
    assert pd.isna(merged.loc[3, "Description"])      # 'Intercept' has no gene

    # The collapse is reported, not hidden.
    out = capsys.readouterr().out
    assert "share 1 gene id(s)" in out
    assert "200320" in out


def test_regression_metadata_merge_refuses_a_duplicated_gene(tmp_path,
                                                             monkeypatch):
    """``validate=`` is the backstop if the de-duplication ever regresses."""
    from spacr.utils import merge_regression_res_with_metadata

    results = tmp_path / "results.csv"
    pd.DataFrame({
        "feature": ["C(gene)[T.200320_1]"], "coefficient": [1.0],
    }).to_csv(results, index=False)
    metadata = _isoform_metadata(tmp_path / "metadata.csv")

    # Neuter the collapse; the merge itself must still refuse to fan out.
    monkeypatch.setattr(pd.DataFrame, "drop_duplicates",
                        lambda self, *a, **k: self)

    with pytest.raises(pd.errors.MergeError, match="not a many-to-one merge"):
        merge_regression_res_with_metadata(str(results), str(metadata))


def test_regression_metadata_merge_allows_many_features_per_gene(tmp_path):
    """Many gRNAs per gene is the normal shape and must keep working."""
    from spacr.utils import merge_regression_res_with_metadata

    results = tmp_path / "results.csv"
    pd.DataFrame({
        "feature": [f"C(gene)[T.200320_{i}]" for i in range(1, 5)],
        "coefficient": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(results, index=False)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"Gene ID": ["TGME49_200320"],
                  "Description": ["HXGPRT"]}).to_csv(metadata, index=False)

    merged = merge_regression_res_with_metadata(str(results), str(metadata))
    assert len(merged) == 4
    assert set(merged["Description"]) == {"HXGPRT"}
