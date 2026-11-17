# Item 962: OPS merge strategy, tile-by-tile hash or well-level stitch, with an automatic fallback

- **Number:** 962 (reserved by master on 2026-11-16). The master session builds it in its own lane after the current landings.
- **Maintainer decision (2026-11-16):** "these two strategies should be integrated into spacr". The reference text is Brieflow, Nat Commun 2026 (doi 10.1038/s41467-026-73643-7), merge module: "The module supports two registration strategies: a tile-by-tile approach that matches cell patterns between corresponding tiles, and a well-level stitching approach for screens with low cell density that fail to produce a robust cell pattern at the tile level. The tile approach constructs a nine-edge hash directly from each tile, while the stitch method first stitches together the tiles into complete well mosaics using their recorded stage coordinates from microscope metadata."
- **Readiness:** merge is a Ready stage. The default must stay byte-identical unless the maintainer approves a change. The proof design below was **signed off by the maintainer on 2026-11-16 as written**.
- **What already exists on nightly (checked by master):**
  - `spacr/ops_merge.py`: `triangle_descriptors`, `align_by_triangles`, `match_cells`, `merge_field`, plus the tile-affine path.
  - `spacr/ops_stitch.py`: `stitch_well`, which places tiles by registering them against each other.
  - Related settings: `ops_merge_threshold`, `ops_stitch_*`.
- **Gap:** there is no strategy setting and no automatic fallback for sparse tiles. There is also no way to build a well mosaic from metadata stage coordinates alone; Brieflow's stitch route does this, while spaCR's stitch registers tiles.

## Spec
1. **`ops_merge_strategy`:**
   - `current` (default): today's path, byte-identical output.
   - `tile`: a nine-edge (Delaunay-neighbourhood) hash per phenotype/SBS tile pair, using the existing triangle-hash functions. One implementation, no copy.
   - `stitch`: build both well mosaics, then match cells on the mosaics. The tile placement source is `ops_merge_stitch_source`: `metadata` (stage coordinates only, as in Brieflow) or `registered` (the existing `stitch_well`).
   - `auto`: tile first, then fall back to stitch for any well whose tile route fails the quality rule in point 2.
2. **Fallback rule.** A tile pair is weak when either its hash votes fall below `ops_merge_min_votes` or its matched-cell fraction falls below `ops_merge_min_match_fraction`. A well falls back when the share of weak tile pairs exceeds `ops_merge_fallback_share`. The default thresholds are set by the proof check below, not guessed.
3. **Per-well strategy report** (`merge_strategy_report.csv`), one row per well:
   - strategy used and the reason;
   - number of weak tile pairs;
   - hash vote and matched-fraction distributions;
   - residual alignment error (px);
   - matched SBS and phenotype percentages;
   - time.
   It also appears in the run summary and the HTML report.
4. **Metadata stage coordinates.** Read them from the existing acquisition metadata readers (nd2/tiff tags already parsed for stitching); there is no new reader. A clear error is raised when the coordinates are missing.
5. **Alpha and records.** The new values ship as alpha until the proof passes; `current` stays the default. Tooltips and help are written in 9 locales, and the arrival contracts and catalogue records go in the same commit. The item 960 grid search can sweep the fallback thresholds.

## Proof check (pre-set; experiment design for sign-off)
- **Data:**
  - **Dense case:** Funk plate A, wells A1–A3, all fields (the same data as the stage-1 Ready check).
  - **Low-density case (controlled):** the same A1–A3 fields with cells randomly thinned before matching to 50, 20, 10, 5 and 2 % (seeded, 3 seeds). The full-density merge serves as truth for which cells pair, so recovered pairs can be scored for precision and recall.
  - **Low-density case (natural):** the lowest-density 10 % of plate A fields by nuclei count (edge fields). The full-well stitched merge serves as reference.
- **Threshold choice:**
  - The fallback thresholds are tuned on A1 only. Pick the point where the tile route's pair recall drops below the stitch route's on the thinning curve; pick the smallest share that avoids falling back on any dense A1 field.
  - Validate on A2 and A3 (held out), and report the thresholds and their curves.
- **Comparator:** Brieflow's own merge code (its tile-hash and stitch routes, config as in our plate-1 run in `reanalysis/brieflow/run_plateA/`), rerun by us on the same wells and on the thinned sets. The values are shown as computed alongside published:
  - SBS cells matched per tile: published 92.3–93.0 % for plate 3 (Nat Commun Fig. 3b; Source Data MOESM10). Our plate-1 rerun gives 92.5 / 91.9 / 92.8 % (A1–A3).
  - Phenotype recovery: published 76.4 % for the whole screen (p. 5, Supplementary Fig. 5a). Our rerun gives 76.2 / 75.7 / 76.0 %.
  - Where the paper gives no value for a thinned set, report "n/a in paper".
- **Metrics:**
  - matched SBS and phenotype %;
  - pair precision and recall against the truth;
  - one-gene cells kept after merging;
  - residual alignment error;
  - time per well against Brieflow on the same machine.
- **Off-switch:** with `ops_merge_strategy=current`, the merge output on plate A A1 is byte-identical to nightly (hash of the merge tables).
- **Controls:**
  - shifted-coordinate negative: random tile offsets, with the matched fraction expected near 0;
  - identity rerun: a byte-identical repeat;
  - for thinned sets, the mean over 3 seeds with the range.
- **Pass:**
  - `auto` is no worse than `current` on dense wells (pair recall within 0.5 points);
  - `auto` beats `tile` on thinned sets at ≤ 10 % density;
  - spaCR matches or beats Brieflow's best route at every density level.
