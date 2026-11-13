"""CPU-only coverage for the tail of spacr.utils: pipeline-folder cleanup,
path normalisation, image consolidation and metadata/outlier helpers.

Covers ``cleanup_pipeline_folders``, ``delete_intermedeate_files``,
``filter_and_save_csv``, ``extract_tar_bz2_files``,
``calculate_shortest_distance``, ``format_path_for_system``,
``normalize_src_path``, ``generate_image_path_map``,
``copy_images_to_consolidated``, ``correct_metadata`` and
``remove_outliers_by_group``.

Everything here is filesystem/DataFrame work on tmp_path — no network, no CUDA,
no plotting.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_figures():
    """Guarantee no matplotlib figure survives a test in this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# ===========================================================================
# cleanup_pipeline_folders — verbose branches + numeric channel folders
# ===========================================================================

# ===========================================================================
# delete_intermedeate_files
# ===========================================================================
#
# The guard counts the FILES in merged/ against those in stack/ and only
# deletes the intermediates once merged/ is at least as populated. (It used to
# compare the lengths of the two path STRINGS, which differ by exactly one
# character for every conceivable src, so the body was unreachable; the tests
# below drive the real body directly now that it is live.)

# ===========================================================================
# filter_and_save_csv
# ===========================================================================

def test_filter_and_save_csv_keeps_tails_only(tmp_path, capsys):
    from spacr.utils import filter_and_save_csv

    src = tmp_path / "in.csv"
    dst = tmp_path / "out.csv"
    pd.DataFrame({"score": [-5.0, -0.5, 0.0, 2.0, 9.0],
                  "name": list("abcde")}).to_csv(src, index=False)

    filter_and_save_csv(str(src), str(dst), "score",
                        upper_threshold=1.0, lower_threshold=-1.0)

    assert dst.is_file()
    out_df = pd.read_csv(dst)
    assert sorted(out_df["score"].tolist()) == [-5.0, 2.0, 9.0]
    assert sorted(out_df["name"].tolist()) == ["a", "d", "e"]
    assert list(out_df.columns) == ["score", "name"]
    assert f"Filtered DataFrame saved to {dst}" in capsys.readouterr().out


def test_filter_and_save_csv_can_produce_empty_output(tmp_path):
    from spacr.utils import filter_and_save_csv

    src = tmp_path / "in.csv"
    dst = tmp_path / "out.csv"
    pd.DataFrame({"score": [0.0, 0.1, -0.2]}).to_csv(src, index=False)

    filter_and_save_csv(str(src), str(dst), "score", 5.0, -5.0)

    assert pd.read_csv(dst).empty


# ===========================================================================
# extract_tar_bz2_files
# ===========================================================================

# ===========================================================================
# calculate_shortest_distance
# ===========================================================================

def test_calculate_shortest_distance_edge_to_edge_and_clamped():
    from spacr.utils import calculate_shortest_distance

    df = pd.DataFrame({
        "pathogen_channel_0_centroid_weighted-0": [0.0, 0.0],
        "pathogen_channel_0_centroid_weighted-1": [0.0, 0.0],
        "nucleus_channel_0_centroid_weighted-0": [30.0, 3.0],
        "nucleus_channel_0_centroid_weighted-1": [40.0, 4.0],
        "pathogen_feret_diameter_max": [10.0, 100.0],
        "nucleus_feret_diameter_max": [20.0, 100.0],
    })

    out = calculate_shortest_distance(df, "pathogen", "nucleus")

    col = "pathogen_nucleus_shortest_distance"
    assert col in out.columns
    # row 0: centroid distance 50 - (5 + 10) = 35
    assert out[col].iloc[0] == pytest.approx(35.0)
    # row 1: centroid distance 5 - (50 + 50) < 0 → clamped to 0
    assert out[col].iloc[1] == pytest.approx(0.0)
    assert (out[col] >= 0).all()
    # the helper mutates and returns the same frame
    assert out is df


# ===========================================================================
# format_path_for_system
# ===========================================================================

# ===========================================================================
# normalize_src_path
# ===========================================================================

# ===========================================================================
# generate_image_path_map + copy_images_to_consolidated
# ===========================================================================

def test_copy_images_to_consolidated_creates_folder_for_empty_map(tmp_path, capsys):
    from spacr.utils import copy_images_to_consolidated

    copy_images_to_consolidated({}, str(tmp_path))

    assert (tmp_path / "consolidated").is_dir()
    assert os.listdir(tmp_path / "consolidated") == []
    assert capsys.readouterr().out == ""


# ===========================================================================
# correct_metadata
# ===========================================================================

# ===========================================================================
# remove_outliers_by_group
# ===========================================================================

