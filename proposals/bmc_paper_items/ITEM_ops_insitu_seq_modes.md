# Item 961: PerturbView-, Zombie- and T7-style in situ sequencing modes (combinatorial barcodes, peak-prioritised calling, compartment-specific read assignment)

- **Number:** 961 (confirmed by master on 2026-11-16). If the work proves large, it may be built as sub-parts 961a–d, one per mode, each with its own proof against that method's published data (computed and published side by side).
- **Maintainer decision (2026-11-16):** "spacr should support all of this". It responds to Brieflow (Nat Commun 2026, doi 10.1038/s41467-026-73643-7), which:
  - supports "combinatorial barcode schemes used in PerturbView and Zombie in situ sequencing approaches";
  - notes that "PerturbView-style in situ sequencing produces single high-intensity nuclear peaks rather than distributed cytoplasmic signal; both characteristics are accommodated through Brieflow's configurable parameters for peak detection prioritization and compartment-specific read assignment";
  - validated this on "three known barcode constructs sequenced across three cycles using T7-based chemistry" (78.2 % of cells uniquely mapped; standard spots, peak threshold 400, peak width 5, peak-prioritised calling, reads assigned to the nucleus).
- **When:** alpha, built after the genotyping freeze. The current Funk and PERISCOPE screens use one barcode per cell, which is the sgRNA itself (CROP-seq style). Brieflow ran Funk with `barcode_type: simple`, so no current comparison depends on this item.
- **Gap:** spaCR decodes one contiguous barcode per read against a whitelist (prefix or full), with no multi-segment or recombination model. Reads are assigned by cell territory (`ops_assign_by_cell_territory`), with no explicit nucleus-only or cytoplasm-only option. Calls rank barcodes by count or quality, with no peak-intensity prioritisation mode.
- **Comparator:** Brieflow `workflow/lib/sbs/call_cells.py`:
  - `barcode_type` (simple or multi);
  - `map_start` / `map_end` and `recomb_start` / `recomb_end`;
  - `prefix_recomb`, `recomb_filter_col`, `recomb_q_thresh`;
  - `sort_calls` (count or peak);
  - `error_correct` with hamming or levenshtein distance;
  - the read-compartment options in its SBS config.

## Spec
1. **Barcode layouts** (`ops_barcode_layout`: `single` (default) | `combinatorial`). A layout file lists the segments. For each segment: its cycle range, its whitelist (or the guide library columns it maps to) and its role (`map`, `recomb`, `umi`). A cell's identity is the combination of segment calls.
2. **Recombination detection.** Segments marked `recomb` are checked against the expected pairing with the `map` segment. Mismatches are flagged (`recombined`) and excluded from gene calls by default. A quality threshold mirrors Brieflow's `recomb_q_thresh`. The recombination rate is reported per well.
3. **Peak-prioritised calling** (`ops_call_sort`: `count` | `quality` (current behaviour) | `peak`). `peak` ranks candidate barcodes in a cell by peak intensity, for chemistries that give one bright amplicon per cell.
4. **Compartment-specific read assignment** (`ops_read_compartment`: `cell` (current) | `nucleus` | `cytoplasm`). This uses the nuclear and cell masks spaCR already makes. A T7 preset (nucleus compartment + peak calling + spot settings) ships as a named preset.
5. **Error tolerance per segment.** Per-segment `max_distance` and metric (hamming or levenshtein), reusing `ops_prefix` matching, so there is one matcher.
6. **Reporting.** Per segment: mapping rate, decoy FDR and barcode-shuffle call rate (item 949, shuffling within each segment's whitelist). Per cell: unique-combination rate and recombination rate.
7. **Alpha:**
   - Off by default and registered in ALPHA_FEATURES.
   - No schema bump; the defaults reproduce current output byte for byte.
   - 9-locale tooltips and help, arrival contracts and catalogue records in the same commit.
   - Item 960's grid search lists the new settings.

## Proof check (pre-set)
- **Off-switch:** with the defaults, Funk plate A A1 and PERISCOPE CP186A held-out output is byte-identical to nightly.
- **Synthetic truth:** simulated tiles with known combinatorial barcodes, recombination events and nuclear-only amplicons (seeded) show that recovered identities, the recombination flag and compartment assignment match the truth. Report precision and recall.
- **Real data:** if a public PerturbView, Zombie or T7 OPS dataset with SBS images is found, run spaCR and Brieflow (its own code, same tiles) and report the unique-mapping rate, decoy FDR and shuffle rate. The value is shown as computed AND published (Brieflow's 78.2 % on its T7 set is not deposited, so it is "published only, data not available").
- **Control:** shuffled whitelists per segment (3 seeds; 500-permutation range at the default point).
