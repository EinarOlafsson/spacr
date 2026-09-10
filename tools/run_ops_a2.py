#!/usr/bin/env python3
"""Stitch one well of one cycle with the A2 modules, and print the three numbers.

Instruction 372, PHASE A2. The pipeline is four modules --
`ops_layout` proposes the adjacency, `ops_register` measures each pair,
`ops_solve` places every tile, `ops_stitch` runs the three and reports --
and this is the one command that points them at a folder of tiles.

IT EXISTS SO THE FIRST REAL RUN IS NOT A SCRIPT SOMEBODY WRITES AGAIN.
Every number in 372's PARTS 6 to 11 came from a private driver in a
temporary directory: a hardcoded path, a channel picked by index, and a
correlation of its own. Those numbers are good and the driver is gone.
This one is committed, so the run that checks the modules against them
can be repeated by anybody.

    python tools/run_ops_a2.py --src /path/to/plate --well A1 --cycle 1

WHAT TO COMPARE IT AGAINST, from PART 11-D, well A1 cycle 1 of
`screenA/20200202_6W-LaC024A`:

    624 pairs proposed, 624 accepted, 333 of 333 tiles placed
    canvas 26,855 x 26,865 px
    residual median 0.2 px, p90 0.4, max 0.7
    106 s

THE TOLERANCE IS TWO NUMBERS BECAUSE THE STAGE IS NOT SQUARE, and a
single one was measured wrong before this script shipped: `down` on well
A1 is (1267, 9) and `right` is (-9, 1268), so a true edge is within a
pixel or two ALONG the raster and about nine ACROSS it. `--tolerance 8`
applied to both axes refused 525 of 624 real adjacencies -- which reads
as the method failing and is a parameter one pixel below the geometry.
`--tolerance` is the along-axis allowance and `--skew` the across-axis
one; the skew belongs to the microscope rather than the well, so another
acquisition wants its own measured rather than this default trusted.

A DISAGREEMENT WITH THOSE IS THE POINT OF RUNNING IT, not a nuisance:
the modules and that script are two implementations of one method, and
the only way to find out whether the packaged one carries the same
answer is to ask it the same question.

THE PLANE IS CHOSEN, NOT ASSUMED. Cycle 1 carries Hoechst and cycles
2..11 do not (PART 10), so `--channel` says which plane the geometry is
solved on and defaults to 0. Handing a five-channel stack to the stitch
would be refused by `stitch_well` with that reason, which is the right
error and not this script's job to produce.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The filename pattern spaCR already ships for this acquisition. Kept as
#: the default rather than re-derived: `get_preprocess_ops_settings` uses
#: the same one, and two patterns for one layout is how a rename goes
#: unnoticed on one path and not the other.
DEFAULT_META_REGEX = (
    r'(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2})'
    r'.*?Site[-_](?P<site>\d+)(?:_[0-9]+)?\.(?:tif|tiff)$'
)


def index_tiles(src: Path, well: str, cycle: int, *,
                pattern: str = DEFAULT_META_REGEX,
                recursive: bool = True) -> Dict[int, Path]:
    """``site -> path`` for one well of one cycle.

    :param src: the acquisition folder.
    :param well: the well name, e.g. "A1".
    :param cycle: the sequencing cycle, matched against the `c<N>` field.
    :param pattern: the filename regex.
    :param recursive: search subfolders.
    :returns: the tiles, keyed by site index.
    :raises SystemExit: when nothing matched, with the counts that say
        WHICH half of the pattern failed -- a run that silently stitches
        zero tiles is worse than one that stops.
    """
    expression = re.compile(pattern, re.IGNORECASE)
    every = (src.rglob("*") if recursive else src.glob("*"))
    seen = 0
    matched = 0
    found: Dict[int, Path] = {}
    for path in sorted(every):
        if not path.is_file():
            continue
        seen += 1
        match = expression.search(path.name)
        if not match:
            continue
        matched += 1
        fields = match.groupdict()
        if (fields.get("well") or "").upper() != well.upper():
            continue
        if int(fields.get("chan") or 0) != int(cycle):
            continue
        found[int(fields["site"])] = path
    if not found:
        raise SystemExit(
            f"no tiles for well {well} cycle {cycle} under {src}: "
            f"{seen} files seen, {matched} matched the name pattern. "
            "If the second number is zero the regex is wrong for this "
            "acquisition; if it is large the well or cycle is.")
    return found


def read_plane(path: Path, channel: int = 0):
    """One 2-D plane out of a tile, whatever dimensions the file has.

    :param path: the image.
    :param channel: which plane, when the file holds more than one.
    :returns: a 2-D float array.
    """
    import numpy as np
    import tifffile

    array = np.asarray(tifffile.imread(str(path)))
    while array.ndim > 2:
        index = channel if array.shape[0] > channel else 0
        array = array[index]
        channel = 0
    return array.astype(np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    """Stitch one well and print the count, the residual and the canvas."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True,
                        help="the acquisition folder")
    parser.add_argument("--well", default="A1")
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--channel", type=int, default=0,
                        help="which plane the geometry is solved on")
    parser.add_argument("--overlap", type=int, default=213,
                        help="the raster's overlap in pixels")
    parser.add_argument("--tolerance", type=int, default=4,
                        help="how far ALONG the raster an edge may land "
                             "from the layout's prediction")
    parser.add_argument("--skew", type=int, default=None,
                        help="how far ACROSS it may -- the stage's own "
                             "skew, measured at 9 px on well A1. None "
                             "takes ops_register.SKEW_PX (24)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="keep every backend on the CPU")
    parser.add_argument("--regex", default=DEFAULT_META_REGEX)
    parser.add_argument("--out", type=Path,
                        help="write the transform table here as JSON")
    args = parser.parse_args(argv)

    from spacr.ops_layout import round_well_layout
    from spacr.ops_stitch import stitch_well

    tiles = index_tiles(args.src, args.well, args.cycle, pattern=args.regex)
    sites = sorted(tiles)
    print(f"{len(sites)} tiles for well {args.well} cycle {args.cycle}")
    layout = round_well_layout(len(sites))

    # RE-READ RATHER THAN HELD. 333 tiles of 1480 px is 2.9 GB and the
    # registration needs two at a time; PART 5's whole argument is that
    # the ceiling must not depend on the plate.
    def read(site: int):
        return read_plane(tiles[site], args.channel)

    started = time.perf_counter()
    well = stitch_well(read, layout, overlap=args.overlap,
                       tolerance=args.tolerance, skew=args.skew,
                       gpu=not args.no_gpu, sites=sites)
    elapsed = time.perf_counter() - started

    print(well.summary())
    print(f"elapsed {elapsed:.1f} s")
    backends = {getattr(one, "backend", "?") for one in well.edges.values()}
    print(f"backends used: {', '.join(sorted(backends))}")
    if not well.canvas_agrees():
        print("THE CANVAS DISAGREES WITH THE LAYOUT. A uniform error is "
              "invisible to the residual and the count; this is the check "
              "that sees it. Do not trust the placements.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "well": args.well,
            "cycle": args.cycle,
            "tiles": len(sites),
            "placed": well.placed,
            "edges": {"proposed": well.proposed, "accepted": well.accepted},
            "canvas": well.canvas,
            "expected_canvas": well.expected_canvas,
            "elapsed_s": round(elapsed, 1),
            "placements": {str(site): [round(y, 2), round(x, 2)]
                           for site, (y, x) in sorted(well.placements.items())},
        }, indent=1), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
