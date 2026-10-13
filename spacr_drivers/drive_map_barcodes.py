"""Drive map barcodes on the synthetic demo reads, in every mode it offers.

The synthetic pair is 4,800 reads over 288 (row, column, gRNA) combinations,
which is small enough to run in seconds and large enough to cross the chunk
boundary several times -- and crossing it is the point. Both output tables
accumulate per chunk, so a single-chunk run cannot fail either way; this
driver forces five chunks and checks the shape of what comes out:

* ``unique_combinations.csv`` must carry no ``Unnamed`` column. One per chunk
  appeared when the writer round-tripped an index it had not written.
* ``qc.csv`` must be ONE row. It is a total, not a log, and one row per chunk
  is what a per-chunk append looks like.

WHAT EACH MODE IS FOR. ``paired`` reads R1 and R2 together. ``single`` reads
one direction only and does NOT reverse-complement, so ``single``/``R2`` on a
library written in the R1 orientation correctly maps nothing -- that is the
orientation trap this module exists to make visible, and a run that quietly
returns zero counts is the failure it looks like.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (check, dataset_root, preflight, require, run, scratch,
                      stage)

DEFAULT_ROOT = ("/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/synthetic"
                "/map_barcodes")

REQUIRED = ("demo_R1_001.fastq.gz", "demo_R2_001.fastq.gz",
            "barcodes/grna.csv", "barcodes/row.csv", "barcodes/column.csv")

#: The read window: eight bases of column barcode, the constant region, the
#: guide, then eight bases of row barcode.
TARGET = "TGCTGTTTCCAGCATAGCTCTTAAAC"

#: 1,000 reads a chunk over 4,800 reads is five chunks, which is what makes a
#: per-chunk accumulation bug visible.
CHUNK = 1000

MODES = (("paired", "R1"), ("single", "R1"), ("single", "R2"))

#: What each mode has to come out with: (combinations, reads mapped). The
#: demo was written with 288 combinations and 4,800 reads, so these are the
#: fixture's own numbers rather than a tolerance.
#:
#: THE COUNTS ARE THE CHECK, not the file list. A mapper that writes both
#: tables and attributes none of its reads leaves exactly the artefacts a
#: "did it run" test looks for -- which is how a zero-count run reads as a
#: success. ``single``/``R2`` is the one mode whose right answer IS zero:
#: the single-end path does not reverse-complement, so a library written in
#: the R1 orientation maps nothing from R2. Its zero is expected here, and
#: any of the other two coming out at zero is not.
EXPECTED = {("paired", "R1"): (288, 4800),
            ("single", "R1"): (288, 4800),
            ("single", "R2"): (0, 0)}


def settings_for(work, mode, direction):
    """The settings one mode of the demo run needs."""
    return dict(
        src=str(work),
        grna_csv=str(work / "barcodes" / "grna.csv"),
        row_csv=str(work / "barcodes" / "row.csv"),
        column_csv=str(work / "barcodes" / "column.csv"),
        mode=mode, single_direction=direction,
        target_sequence=TARGET, offset_start=-8, expected_end=89,
        chunk_size=CHUNK, n_jobs=2, save_h5=False, test=False, fill_na=False)


def report(work, mode, direction):
    """Summarise one mode's output tables and check both shape and totals.

    :returns: ``(combinations, reads)`` summed over the samples the run
        wrote, so the caller can compare them with what the fixture holds.
    :raises WrongAnswer: on a stray index column, a QC table that is a log
        rather than a total, or no count table at all.
    """
    import pandas as pd

    tables = sorted(work.glob("demo_*/unique_combinations.csv"))
    check(tables, f"{mode}/{direction} wrote no count table under {work}")
    combinations = reads = 0
    for path in tables:
        counts = pd.read_csv(path)
        qc = pd.read_csv(path.with_name("qc.csv"))
        stray = [c for c in counts.columns if str(c).startswith("Unnamed")]
        print(f"  {mode}/{direction}: {len(counts)} combinations, "
              f"{int(counts['count'].sum())} reads, qc rows {len(qc)}")
        check(not stray,
              f"{mode}/{direction}: the count table carries {stray}, which is "
              f"an index column the writer round-tripped once per chunk")
        check(len(qc) == 1,
              f"{mode}/{direction}: qc.csv has {len(qc)} rows. It is a total "
              f"over the run, so one row is the only right answer; one per "
              f"chunk is what a per-chunk append looks like")
        combinations += len(counts)
        reads += int(counts["count"].sum())
    return combinations, reads


def main(argv):
    """Run every mode against a scratch copy of the demo reads."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the synthetic map-barcodes demo")
    print(f"dataset root: {root}")
    print("reads: the SYNTHETIC demo pair, not real sequencing")

    from spacr.sequencing import generate_barecode_mapping

    for mode, direction in MODES:
        work = scratch(f"map_barcodes_{mode}_{direction}")
        stage(root, REQUIRED, work)
        settings = settings_for(work, mode, direction)
        preflight(settings, "map_barcodes")
        generate_barecode_mapping(settings)
        got = report(work, mode, direction)
        want = EXPECTED[(mode, direction)]
        check(got == want,
              f"{mode}/{direction} mapped {got[1]} reads into {got[0]} "
              f"combinations; the demo pair holds {want[1]} reads over "
              f"{want[0]} combinations. A count table that exists is not a "
              f"count table that is right.")

    print("\nevery mode mapped the reads the demo holds, into the "
          "combinations it holds, and both output tables have the shape "
          "they should")


if __name__ == "__main__":
    run(main)
