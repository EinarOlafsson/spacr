"""Owned once / twice or more / never, for the nuclei of one OPS phenotype field.

usage: python tools/ops_seam_ownership_counts.py RUN WELL SITE [--radius PX ...]

RUN is an OPS run folder after ph_ownership, WELL a well (A1) and SITE a
phenotype field (243). Two counts are printed as JSON, both over the nuclei
with a copy in SITE:

* ``components``: the physical cells of ``ops_cross_field_<WELL>.parquet``
  (column ``component``, from the ph_ownership pass: whole copies paired
  when their nucleus centroids, in stitched coordinates, are mutual nearest
  neighbours within ``ops_cross_field_radius_px``, and each cut copy joined
  to its nearest whole copy). ``before`` counts the copies that do not touch
  their field edge (``touches_field_edge`` False, the edge-only rule);
  ``after`` counts ``is_owner``.
* ``r<PX>px`` for every ``--radius``: an independent grouping that joins
  any two copies of different fields whose stitched nucleus centroids lie
  within PX phenotype pixels (``well_y``/``well_x`` of the table divided by
  the median field scale of ``ops_phenotype``), with the same two counts.
  A cut copy has a centroid moved by its missing part, so small radii
  leave cut copies apart from their whole twin and count them as never
  owned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def _counts(groups, flags) -> dict:
    """Owned once / twice or more / never of each group."""
    owned = flags.astype(int).groupby(groups).sum()
    return {"nuclei": int(len(owned)), "owned_once": int((owned == 1).sum()),
            "owned_twice_or_more": int((owned >= 2).sum()),
            "never_owned": int((owned == 0).sum())}


def main(argv=None) -> int:
    """Print the counts; return 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("well")
    parser.add_argument("site", type=int)
    parser.add_argument("--radius", type=float, action="append", default=[])
    args = parser.parse_args(argv)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from spacr import ops_engine
    from spacr.ops_store import read_table

    path = ops_engine._ph_cross_field_path(args.run, args.well)
    if not os.path.isfile(path):
        parser.error(f"no ownership table at {path}; run ph_ownership first")
    table = pd.read_parquet(path)
    if args.site not in set(table["site"].astype(int)):
        parser.error(f"field {args.site} is not in {path}")
    whole = ~table["touches_field_edge"].astype(bool)
    owner = table["is_owner"].astype(bool)
    mine = table.loc[table["site"] == args.site, "component"].unique()
    inside = table["component"].isin(mine)
    out = {"table": path, "well": args.well, "site": args.site,
           "copies_in_site": int((table["site"] == args.site).sum()),
           "owner_rule": str(table["owner_rule"].iloc[0]) if len(table) else None,
           "components": {
               "before": _counts(table.loc[inside, "component"], whole[inside]),
               "after": _counts(table.loc[inside, "component"], owner[inside])}}
    if args.radius:
        placed = read_table(os.path.join(args.run, "measurements.db"),
                            "ops_phenotype")
        placed = placed[placed["well"].astype(str) == args.well]
        scale = float(np.median(placed["scale"].astype(float)))
        points = table[["well_y", "well_x"]].to_numpy(float) / scale
        sites = table["site"].to_numpy()
        tree = cKDTree(points)
        for radius in args.radius:
            pairs = tree.query_pairs(radius, output_type="ndarray")
            pairs = pairs[sites[pairs[:, 0]] != sites[pairs[:, 1]]]
            _n, group = connected_components(coo_matrix(
                (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                shape=(len(table), len(table))), directed=False)
            group = pd.Series(group, index=table.index)
            chosen = group.isin(group[table["site"] == args.site].unique())
            out[f"r{radius:g}px"] = {
                "before": _counts(group[chosen], whole[chosen]),
                "after": _counts(group[chosen], owner[chosen])}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
