# Item spec: nuclear-translocation readout for single-marker optical pooled screens: per-cell DNA–marker correlation inside the nucleus and nuclear/cytoplasmic ratio, and a per-gene translocation-defect score (integrated difference of sgRNA vs non-targeting distributions) with permutation FDR

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY (same per-cell values, same gene scores, same permutation p with the fixed seed). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (Feldman 2019 experiment C support benchmark: NF-κB pathway recovery).
- Related: `ITEM_effector_translocation.md` (corrected nuclear effector signal in host cells; shares the nucleus / cytoplasm compartment code but answers a different question: there the signal is binned by parasite load, here it is scored per gene against NT), `ITEM_idr0071_stack_reader_decode.md` (inputs and gene calls), `ITEM_ops_reference_mask_matching.md` (per-cell comparison with their masks), `measure.py` colocalisation (`_calculate_correlation_object_level`, per-object channel-pair correlation; reuse, do not duplicate).

## Why
Feldman et al. (2019) screened 952 genes for NF-κB p65 nuclear translocation after TNFα (wells A1–A3) or IL-1β (B1–B3). Their per-cell score is the Pearson correlation of DAPI and p65 pixels inside the nucleus (high when p65 is nuclear), and the gene score is the integrated difference between each sgRNA's cumulative distribution of that score and the NT distribution, averaged over the gene's sgRNAs, with an FDR (Table S3; HeLa, A549, HCT116). Headline recovery: 18 of 19 KEGG NF-κB components in A549, 17/19 in HeLa; receptor arms are stimulus-specific (TNFRSF1A / TRADD in TNFα wells, IL1R1 / MYD88 / IRAK in IL-1β wells). spaCR measures compartment intensities and per-object channel correlations, but has no named translocation feature for OPS cell tables, no distribution-difference gene score and no permutation FDR for it.

## Reference implementation
- Feldman `ops` package (github.com/feldman4/OpticalPooledScreens), `ops.features` and the paper's analysis: per-cell nuclear DAPI–p65 correlation (`dapi_gfp_corr`-type column; **verify the exact column and its compartment against Table S3 and the experiment C Snakefile on IDR** (`Snakefile_experimentC`, a few kB) before implementing).
- Gene score as described in the paper's methods (integrated difference of CDFs; per-sgRNA then mean over the gene's sgRNAs; significance by permutation of cell labels; BH).
- Table S3 / S4: from the paper's supplement (PMC6886477 files `NIHMS1056812-supplement-08/09.xlsx`). Without them only the pathway and receptor-arm checks can run.

## What to build
1. **Per-cell features** (OPS measurement and Measure, named so tables are self-describing):
   - `nucleus_<a>_<b>_pearson`: Pearson r of channel a (DNA) and b (marker) over the nucleus pixels (reuse the existing per-object correlation; background handling as the reference: raw pixels, no subtraction unless `translocation_background` is set);
   - `translocation_nc_ratio`: mean marker in nucleus / mean marker in cytoplasm (cytoplasm = cell minus nucleus, or a ring of `translocation_ring_px` around the nucleus when there is no cell mask);
   - both written with the cell table; cells with < `translocation_min_nuc_px` nuclear pixels → NaN.
2. **Gene score** (`spacr.gene_measurement_compare.translocation_gene_scores`): per well group (stimulus), per sgRNA the integrated difference ∫ (F_sg(x) − F_NT(x)) dx over the score range (signed; negative = translocation defect), using cells with a confident guide call; gene score = mean over its sgRNAs with ≥ `translocation_min_cells` cells; p by permutation (`translocation_permutations` 10,000; seed 0) of cell labels between the sgRNA and NT pools, BH over genes; also a rank-based alternative (two-sample KS / Mann-Whitney) reported beside it.
3. **Outputs:** `translocation_cells.parquet`, `translocation_guides.csv`, `translocation_genes.csv` (gene, stimulus, score, n_guides, n_cells, p, q), `pathway_recovery.csv`, `receptor_arm_specificity.csv`.
4. **Checks built in:** recovery of a named gene set at an FDR (`translocation_check_set`, default KEGG NF-κB 19 components as listed by the paper) and stimulus specificity of named receptor arms.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `translocation_channels` | None (required) | `{'dna': i, 'marker': j}` |
| `translocation_score` | `'nucleus_pearson'` | or `'nc_ratio'` |
| `translocation_ring_px` | 5 | cytoplasm ring without a cell mask |
| `translocation_min_nuc_px` | 20 | |
| `translocation_background` | None | constant or per-field percentile |
| `translocation_groups` | None | well → stimulus map (e.g. A* TNFα, B* IL-1β) |
| `translocation_min_cells` | 20 | per sgRNA |
| `translocation_permutations` | 10000 | |
| `translocation_seed` | 0 | |
| `translocation_fdr` | [0.10, 0.20] | Table S3 levels |
| `translocation_check_set` | `'kegg_nfkb_19'` | gene set for recovery |

## Validation data and acceptance
- **Data:** Feldman experiment C (A549, p65 immunofluorescence, CROPseq), wells A1 (TNFα) and B1 (IL-1β), 533 tiles each: raw `input/10X_c0-DAPI-p65ab/*.phenotype.tif`, their `process/` `cells.tif`, `nuclei.tif`, `reads.csv` (about 135 GB); library `CROPseq_NFkB_library.csv` (ops repository); Table S3 when it arrives.
- **Acceptance:**
  - per-cell feature on their masks equals the reference implementation of the column (once identified) to float tolerance 1e-6;
  - Spearman ρ of spaCR gene scores with Table S3 (experiment C) reported, with the shuffled-label control;
  - recovery of the 19 KEGG NF-κB components at their FDR (theirs 18/19 for A549) reported; TNF-arm genes score in A1 and not B1, IL-1-arm genes the reverse;
  - rerun from the saved settings gives identical per-cell, guide and gene tables.
- **Tests (CPU):** synthetic cells with nuclear vs cytoplasmic marker → r and ratio in the expected order; planted defect guides vs NT → negative score and q < 0.1; shuffled labels → no hits beyond the FDR.

## Controls
- **Shuffled guide labels** within well (seed 0).
- **NT pseudo-genes** (NT guides in groups of the median guides per gene) scored as genes.
- **Pixel-shuffle control:** marker pixels permuted within each nucleus (seed 0) → r ≈ 0 in every cell and no gene hits, so the score is not driven by nuclear size or brightness alone.
- **Stimulus swap:** gene scores computed with the well → stimulus map swapped (receptor-arm specificity must vanish).

## Catalogue and locales
New documented callables (the two features, `translocation_gene_scores`) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Feldman D et al. (2019) Optical pooled screens in human cells. Cell 179:787–799 (idr0071).
- Kanehisa M et al. (2023) KEGG for taxonomy-based analysis of pathways and genomes. Nucleic Acids Res 51:D587 (NF-κB pathway hsa04064).
- Hayden MS, Ghosh S (2008) Shared principles in NF-κB signaling. Cell 132:344–362.

## Effort
- **Effort:** M (two features, one gene score with permutations, two built-in checks).
