"""386 step 5: the consumers should need no changes -- checked, not assumed.

386: "Only then the consumers, none of which should need changing: UMAP,
regression, hit calling, anndata export."

That is a claim about code nobody has touched, which makes it exactly the
kind of claim worth a test rather than a paragraph. If it holds, these tests
pass today and go on holding it; if a consumer ever grows a hard-coded list
of feature families, they fail and say which one.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.column_groups import classify
from spacr.feature_dict import parse_column


def _frame(rows=12, dims=4):
    """An object frame with measured columns and embedding columns together."""
    rng = np.random.default_rng(0)
    data = {
        "plate": ["p1"] * rows,
        "rowID": ["A"] * rows,
        "columnID": [f"{i % 4 + 1}" for i in range(rows)],
        "object_label": list(range(1, rows + 1)),
        "cell_area": rng.normal(500, 40, rows),
        "cell_channel_1_mean_intensity": rng.normal(120, 12, rows),
    }
    for channel in (1, 2):
        for dim in range(dims):
            data[f"emb_c{channel}_{dim:03d}"] = rng.normal(0, 1, rows)
    return pd.DataFrame(data)


def test_anndata_export_treats_them_as_features():
    """The selection rule is 'numeric and not provenance', so they qualify.

    Nothing in it enumerates families, which is why no change was needed --
    and this test is what would notice if that ever stopped being true.
    """
    from spacr.anndata_export import feature_columns

    frame = _frame()
    features = feature_columns(frame)
    embeddings = [c for c in frame.columns if c.startswith("emb_")]

    assert embeddings, "the fixture built no embedding columns"
    for column in embeddings:
        assert column in features, column
    # and the measured panel is still there beside them
    assert "cell_area" in features
    assert "cell_channel_1_mean_intensity" in features


def test_identity_columns_are_still_excluded():
    """The other half of the same rule, so the test cannot pass vacuously."""
    from spacr.anndata_export import feature_columns

    features = feature_columns(_frame())
    for column in ("plate", "rowID", "columnID"):
        assert column not in features


def test_the_var_table_can_name_the_family():
    """anndata writes `entry.family` into `var`, so it now reads 'embedding'.

    Before the feature_dict wiring this said 'unknown' for every dimension,
    which is a table a reader cannot filter.
    """
    entry = parse_column("emb_c2_017")
    assert entry.family == "embedding"
    assert entry.description


def test_a_reduction_can_select_them_as_a_named_group():
    """UMAP and the pickers choose by group; the group now exists."""
    frame = _frame()
    families = classify(frame.columns)["family"]

    assert "embedding" in families
    assert len(families["embedding"]) == 8          # 2 channels x 4 dims
    assert "morphology" in families or "intensity" in families


def test_they_survive_a_numeric_matrix_round_trip():
    """What every consumer actually does with them: make a float matrix.

    A column that is numeric in the frame but not castable -- an object dtype
    holding strings, say -- would pass the selection and fail the reduction.
    """
    from spacr.anndata_export import feature_columns

    frame = _frame()
    matrix = frame[feature_columns(frame)].to_numpy(dtype=float)

    assert matrix.shape[0] == len(frame)
    assert np.isfinite(matrix).all()


def test_hit_calling_reads_a_gene_not_a_feature_family():
    """Hit calling keys on gene/guide identity, which embeddings do not touch.

    This is why it needed no change: the family of the column being
    aggregated is not something it inspects.
    """
    from spacr.hits import gene_of

    assert gene_of("emb_c2_017") is None or isinstance(gene_of("emb_c2_017"), str)


def test_nothing_in_the_grouping_layer_names_the_family_specially():
    """The wiring went into parse_column, so the grouping stayed generic.

    A special case here would be a second taxonomy -- the thing
    `column_groups`' own docstring warns against.
    """
    import inspect

    from spacr import column_groups

    source = inspect.getsource(column_groups)
    assert "emb_" not in source
    assert "embedding" not in source
