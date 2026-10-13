"""Drive map barcodes on real sequencing reads, and prove the orientation trap.

The synthetic demo shows that the mapper works. Only the real reads show what
goes wrong on a real library, and what goes wrong is ORIENTATION: the row and
gRNA references have to be given as reverse complements while the column
reference is given forward. Get it wrong and spaCR maps zero reads and reports
nothing wrong -- a silent zero count that looks exactly like a failed library.

So this driver runs the same reads twice, once with the orientation the
recorded settings name and once with the forward files, and prints both. The
second run is not a mistake; it is the reproduction of the failure.

Only the first slice of the lane is read. The full file is over a hundred
million reads, and the identity space is established long before that.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (check, dataset_root, preflight, read_settings, require,
                      run, scratch, settings_file, stage)

DEFAULT_ROOT = "/mnt/wd12tb/sequencing"

#: The lane, and the settings that produced the reference count table.
REQUIRED = ("seq_3/EO1_R1*.fastq.gz", "seq_3/EO1_R2*.fastq.gz",
            "seq_3/settings/sequencing_paired_R1.csv")

SETTINGS_CANDIDATES = ("seq_3/settings/sequencing_paired_R1.csv",)

#: Reads to copy out of the lane. Four hundred thousand is a few seconds of
#: work and enough to fill forty thousand combinations.
READS = 400_000

#: The reference keys are named by the settings; these are the same files in
#: the other orientation, used to reproduce the silent-zero-count failure.
FLIPPED = {"grna_csv": ("_RC", ""), "row_csv": ("_RC", ""),
           "column_csv": ("", "_RC")}


def head_fastq(source, destination, reads):
    """Copy the first ``reads`` records of a gzipped FASTQ.

    Slicing here rather than pointing the run at the whole lane is what keeps
    this driver a few seconds instead of an hour, and it never touches the
    original: the source is only read.
    """
    lines = reads * 4
    with gzip.open(source, "rb") as handle, \
            gzip.open(destination, "wb", compresslevel=1) as out:
        for index, line in enumerate(handle):
            if index >= lines:
                break
            out.write(line)
    return destination


def flip_orientation(settings):
    """The same references in the wrong orientation.

    Each ``*_csv`` path gains or loses its ``_RC`` suffix, which is precisely
    the mistake that maps nothing and says nothing.
    """
    flipped = dict(settings)
    for key, (drop, add) in FLIPPED.items():
        path = Path(str(settings[key]))
        stem = path.stem
        stem = stem[:-len(drop)] if drop and stem.endswith(drop) else stem
        flipped[key] = str(path.with_name(f"{stem}{add}{path.suffix}"))
    return flipped


def mapped_reads(work):
    """How many reads and combinations one run produced."""
    import pandas as pd

    total = combinations = 0
    for path in sorted(Path(work).glob("*/unique_combinations.csv")):
        counts = pd.read_csv(path)
        combinations += len(counts)
        total += int(counts["count"].sum()) if len(counts) else 0
    return total, combinations


def main(argv):
    """Map a slice of the lane in the recorded orientation and in the wrong one."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the sequencing lane")
    print(f"dataset root: {root}")
    recorded = settings_file(root, SETTINGS_CANDIDATES,
                             what="the map-barcodes run")
    print(f"settings:     {recorded}")
    base = read_settings(recorded)

    barcode_dir = Path(str(base["grna_csv"])).parent
    if not barcode_dir.is_dir():
        raise SystemExit(
            f"the settings name barcode references under {barcode_dir}, which "
            f"is not on this machine.")

    from spacr.sequencing import generate_barecode_mapping

    mapped = {}
    for label, settings in (("recorded orientation", base),
                            ("forward references", flip_orientation(base))):
        work = scratch(f"map_barcodes_real_{label.split()[0]}")
        stage(barcode_dir.parent, [f"{barcode_dir.name}/*.csv"], work)
        read_1 = sorted(root.glob(REQUIRED[0]))[0]
        read_2 = sorted(root.glob(REQUIRED[1]))[0]
        head_fastq(read_1, work / read_1.name, READS)
        head_fastq(read_2, work / read_2.name, READS)

        run_settings = dict(settings)
        run_settings["src"] = str(work)
        for key in FLIPPED:
            run_settings[key] = str(work / barcode_dir.name /
                                    Path(str(settings[key])).name)
        run_settings["save_h5"] = False
        preflight(run_settings, "map_barcodes")
        generate_barecode_mapping(run_settings)

        reads, combinations = mapped_reads(work)
        share = 100.0 * reads / READS
        print(f"\n{label}: {reads} of {READS} reads mapped ({share:.0f}%), "
              f"{combinations} combinations")
        mapped[label] = reads

    check(mapped["recorded orientation"] > 0,
          f"the recorded orientation mapped 0 of {READS} reads. These are the "
          f"reads and the references the shipped count table was built from, "
          f"so zero here is the mapper and not the library -- and it is zero "
          f"that no error was raised about.")
    check(mapped["forward references"] < mapped["recorded orientation"],
          f"flipping every _RC reference to its forward file changed the "
          f"count from {mapped['recorded orientation']} to "
          f"{mapped['forward references']}. Orientation is supposed to decide "
          f"whether a read matches at all, so a flip that costs nothing means "
          f"the references are not reaching the match.")
    print(f"\nthe orientation decides the count: "
          f"{mapped['recorded orientation']} reads with the recorded "
          f"references against {mapped['forward references']} with the "
          f"forward ones, on the same {READS} reads")


if __name__ == "__main__":
    run(main)
