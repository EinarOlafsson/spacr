"""Timeflows label QC: every stack given on the command line is audited."""


def test_stacks_that_share_a_folder_name_are_all_audited():
    """2026-09-21, the first run on real movies: the Cell Tracking Challenge
    keeps every movie's labels in a folder called TRA, and keyed on that the
    stacks overwrote one another until only the last was audited."""
    from spacr.timeflows_qc import dataset_names

    paths = ["/d/ctc_hela/01_GT/TRA", "/d/ctc_hela/02_GT/TRA",
             "/d/ctc_u373/01_GT/TRA", "/d/other/stack.tif"]
    assert dataset_names(paths) == ["ctc_hela/01_GT/TRA", "02_GT/TRA",
                                    "ctc_u373/01_GT/TRA", "stack"]
