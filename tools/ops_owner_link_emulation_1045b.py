"""1045b: the paper emulation (one_number_A1.py footing) on a WS proxy run, to see where its nearest links come from.

Owner = own_nc (nearest field centre, refined offset/scale frame); physical cell = 2 px root (11_fair_counts).
Step 1 inherit: an unlinked own_nc owner takes the link of any copy of its 2 px root.
Step 2 nearest: an owner still unlinked takes the nearest SBS nucleus within 3 px that is unused, under
two readings of 'unused': (A) not linked to any copy at all; (B) not linked to any own_nc owner copy.
Each step-2 link is then checked against spaCR's 1026 component: is its nucleus held by a copy of the
same component (so spaCR's component inheritance already gives it)?
Item 1045b. Run it on a run folder after a merge with ops_owner_link_inherit 'off' (the frozen
links) to see how many of the paper's nearest links spaCR's component inheritance already gives.
usage: python tools/ops_owner_link_emulation_1045b.py RUN [LINKS.parquet | ops_merge]
(cwd = the spaCR tree; LINKS defaults to the run's ops_merge table, well A1)."""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from spacr.ops_store import read_table
if len(sys.argv) not in (2, 3):
    sys.exit(__doc__)
run = sys.argv[1]; lname = sys.argv[2] if len(sys.argv) == 3 else "ops_merge"; F = 2960
db = os.path.join(run, "measurements.db")
if lname == "ops_merge":
    links = read_table(db, "ops_merge"); links = links[links.well == "A1"][["site", "label", "object_id"]]
else:
    links = pd.read_parquet(os.path.join(run, lname))[["site", "label", "object_id"]]
ph = read_table(db, "ops_phenotype").set_index("site")
g = pd.DataFrame({"oy": ph.refined_offset_y.fillna(ph.offset_y), "ox": ph.refined_offset_x.fillna(ph.offset_x),
                  "sc": ph.refined_scale.fillna(ph.scale)})
obj = read_table(db, "ops_objects"); obj = obj[obj.well == "A1"]
c = pd.read_parquet(os.path.join(run, "ops_phenotype_cells_A1.parquet"), columns=["site", "label", "centroid_row", "centroid_col"])
gg = g.reindex(c.site); c["Y"] = gg.oy.values + gg.sc.values * c.centroid_row.values; c["X"] = gg.ox.values + gg.sc.values * c.centroid_col.values
sites = g.index.values; tree = cKDTree(np.column_stack([g.oy + g.sc * F / 2, g.ox + g.sc * F / 2]))
c["own"] = tree.query(c[["Y", "X"]].values)[1] == np.searchsorted(sites, c.site.values)
site = c.site.values; pr = cKDTree(c[["Y", "X"]].values).query_pairs(2.0, output_type="ndarray"); pr = pr[site[pr[:, 0]] != site[pr[:, 1]]]
par = np.arange(len(c))
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in pr:
    ra, rb = f(a), f(b)
    if ra != rb: par[max(ra, rb)] = min(ra, rb)
c["root"] = [f(i) for i in range(len(c))]
x = pd.read_parquet(os.path.join(run, "ops_cross_field_A1.parquet"))[["site", "label", "component"]]
c = c.merge(x, on=["site", "label"], how="left").merge(links, on=["site", "label"], how="left")
o = c[c.own].copy()
root_link = c.dropna(subset=["object_id"]).groupby("root").object_id.first()
o["inh"] = o.object_id.fillna(o.root.map(root_link))
print("own_nc owners", len(o), "linked as run", int(o.object_id.notna().sum()), "after root inherit", int(o.inh.notna().sum()))
T = obj[["centroid_y", "centroid_x"]].values; ids = obj.object_id.values
for name, used_ids in (("A unused=no copy linked", set(c.object_id.dropna()) | set(o.inh.dropna())),
                       ("B unused=no owner copy linked", set(o.inh.dropna()))):
    u = o[o.inh.isna()]; d, j = cKDTree(T).query(u[["Y", "X"]].values, k=8, distance_upper_bound=3.0)
    got = []
    taken = set(used_ids)
    cand = sorted((float(dd), int(r), int(ids[jj])) for r, (drow, jrow) in enumerate(zip(d, j)) for dd, jj in zip(drow, jrow) if np.isfinite(dd) and int(ids[jj]) not in taken)
    done = set()
    for dd, r, oid in cand:
        if r in done or oid in taken: continue
        done.add(r); taken.add(oid); got.append((r, oid, dd))
    if got:
        gr = u.iloc[[r for r, _, _ in got]].assign(nn=[oid for _, oid, _ in got])
        holder = c.dropna(subset=["object_id"]).set_index("object_id").component
        same = gr.nn.map(holder) == gr.component
        print(name, "nearest links", len(got), "nucleus held by a copy of the same 1026 component", int(same.sum()), "held by no copy", int(gr.nn.map(holder).isna().sum()))
    else:
        print(name, "nearest links 0")
