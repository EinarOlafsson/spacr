# Item spec: gene graph — learned gene–gene links and gene → category membership from OPS profiles (+ JUMP), validated by held-out genes and a time split

Related: `ITEM_ops_gene_level_aggregation.md`, `ITEM_ops_contrastive_weighting.md` (S02 input), `ITEM_ops_reference_sets.md` (STRING/BioGRID/CORUM, archived versions), `ITEM_ops_retrieval_map.md` (AP definition), `ITEM_cross_screen_similarity.md` (JUMP loader), `ITEM_gene_class_training.md` (classes, folds, NT pseudo-genes).
- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs on the validation data, identically where the computation is deterministic and within the tolerances below where it is not (GPU training). Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`; W = `/mnt/wd4tb/bmc_reanalysis/gene_graph`.

## Why
spaCR has no way to learn gene–gene links and gene → category membership from OPS gene profiles (optionally with JUMP profiles) and to validate them on held-out genes and on a time split (links reported after 2022). The analysis exists only as custom scripts. To be reproducible from a settings file, it has to live in spaCR next to the OPS gene-level analysis, and it must include its controls (shuffled features/labels, rewiring, NT decoys, degree prior); without them the metrics cannot be interpreted.

Literature: graph neural networks GCN (Kipf & Welling 2017), GraphSAGE (Hamilton et al. 2017), GPS graph transformer (Rampášek et al. 2022) with random-walk structural encodings (Dwivedi et al. 2022); DistMult decoder (Yang et al. 2015); Adamic–Adar (Adamic & Adar 2003); Laplacian eigenmaps (Belkin & Niyogi 2003); target–decoy FDR (Elias & Gygi 2007).

## Target module
`spacr/ops/gene_graph/` (alpha), a sibling of the OPS gene-level aggregation and retrieval code. It is called after the gene table exists (aggregate → S02 weighting → gene graph). Entry points, each a documented callable with an arrival contract:
- `get_gene_graph_default_settings()`
- `build_gene_graph(settings)`
- `train_gene_graph(settings)`
- `evaluate_gene_graph(settings)`
- `predict_gene_functions(settings)`
- `run_gene_graph(settings)`, which runs all four in order; this is what spacr-run / the GUI calls.

| file | from (reference) | content |
|---|---|---|
| `data.py` | `gg_data.py` | blocks + missing indicators, typed edges, classes, folds, NT pseudo-genes, poorly-characterised annotations |
| `models.py` | `gg_models.py` | `GeneGraphModel` (`lr`, `gcn`, `sage`, `gps`), `rwse`, `mean_adj`, `gcn_adj`, `spmm` |
| `train.py` | `gg_train.py` | masking loop, early stopping, controls, spectral block, GPU admission/yield + resume, budget guard |
| `baselines.py` | `gg_baselines.py` | cosine + kNN vote, Adamic–Adar, degree/size prior |
| `evaluate.py` | `gg_eval.py`, `gg_tables.py` | time-split and held-out metrics, hub/fitness/NT checks, ablation table, paired fold deltas |
| `predict.py` | `gg_predict.py`, `gg_predict_time.py`, `gg_flag.py` | cross-fitted isotonic calibration, NT target–decoy FDR, JUMP support, drivers, specificity flags |
| `figures.py` | `gg_figures.py` | bars + jittered points, controls hatched (spaCR palette) |

## Settings (`get_gene_graph_default_settings`)
| key | type | default | meaning |
|---|---|---|---|
| `src` | str | – | output folder (`gene_graph/` is created under it) |
| `ops_gene_table` | str (parquet) | – | gene rows: `gene`, `n_cells`, feature columns (`w_*` for S02-weighted) |
| `ops_guide_table` | str (parquet) | – | guide rows in the same space: `guide`, `gene`, `n_cells`, features (used for NT pseudo-genes and fitness) |
| `ops_min_cells` | int | 10 | gene rows kept if `n_cells` > this |
| `ops_feature_prefix` | str | `"w_"` | feature columns |
| `ops_scale` | str | `"global_rms"` | one global RMS scale for the whole block (keeps S02 weights); `"none"` |
| `fitness_feature` | bool | True | append z-scored log mean cells per guide to the OPS block |
| `nt_grouping` | str (csv) | ops_gene_classes v3 `nt_pseudogenes.csv` | sgRNA → pseudo-gene (4 per pseudo-gene) + fold |
| `extra_blocks` | dict[str, str] | `{}` | name → gene × feature parquet (index = symbol), e.g. `{"jcr": …crispr_final_genes.parquet, "jorf": …orf_final_genes.parquet}` |
| `extra_block_nt` | dict[str, str] | `{}` | name → NT pseudo-profile parquet; NT nodes receive one at the block's gene coverage rate |
| `extra_block_pca` | dict[str, int] | `{"jorf": 128}` | PCA dims per block (fit on genes, seed 0); 0 = none |
| `sensitivity_blocks` | dict[str, str] | `{}` | e.g. `{"ops_tvn": <pilot TVN table>}`: ic_lib-style embed, first 64 PCs |
| `train_edges` | list[str] | `["STRING11.5>=0.7", "BioGRID2022", "CORUM3"]` | message-passing edge types (reference-set names from `ITEM_ops_reference_sets.md`) |
| `new_edges` | list[str] | `["STRINGnew>=0.7", "BioGRIDnew", "CORUMnew"]` | time-split positives (minus any training pair) |
| `current_edges` | list[str] | `["STRING>=0.7", "BioGRID", "CORUM"]` | second held-out reference and "already known" filter |
| `known_extra` | list[str] | `["STRING>=0.4"]` | also counts as known for output filtering |
| `class_table`, `class_members`, `gene_folds` | str (csv) | ops_gene_classes `tables_v3/` | categories (held-out task only) and 5 folds; genes missing from folds get a seeded random fold |
| `tasks` | list[str] | `["time", "fold0", …, "fold4"]` | evaluation protocols |
| `models` | list[str] | `["gps", "sage", "gcn", "lr", "spectral", "cosine", "adamicadar", "prior"]` | models to run |
| `arms` | dict[str, list[str]] | `{"full": ["ops","jcr","jorf"], "noOPS": ["jcr","jorf"], "noJUMP": ["ops"], "graph": []}` | feature ablations |
| `controls` | list[str] | `["shufX", "shufY", "rewire"]` | control arms (applied to gps `full`, gps `noJUMP` shufY, lr `full`; configurable) |
| `seeds` | list[int] | `[0]` | model seeds (reference: 0; seed 1 partial) |
| `hidden`, `layers`, `heads`, `dropout` | int, int, int, float | 128, 3, 4, 0.2 | architecture |
| `rwse_steps` | int | 16 | random-walk structural encoding |
| `node_mask`, `edge_mask`, `neg_per_pos` | float, float, int | 0.15, 0.2, 3 | self-supervised masking (node mask only on held-out tasks) |
| `pos_weight_pair`, `neg_weight_pair`, `pos_weight_cat` | float | 1.5, 0.5, 5.0 | loss weights |
| `lr_graph`, `lr_linear`, `wd_graph`, `wd_linear` | float | 1e-3, 3e-3, 1e-4, 1e-3 | AdamW |
| `lr_pos_minibatch` | int | 50000 | positives per epoch for `lr` (full batch exceeds 3 GB) |
| `max_epochs`, `eval_every`, `patience`, `min_epochs` | int | 400, 10, 6, 60 | early stopping on the inner validation |
| `val_edge_frac`, `val_gene_frac` | float | 0.05, 0.10 | inner validation (time / held-out) |
| `knn_vote_k` | int | 25 | cosine category vote |
| `spectral_dims` | int | 64 | Laplacian eigenmaps |
| `rewire_swaps` | float | 2.0 | × edges, degree-preserving double-edge swaps |
| `device` | str | `"cuda"` | `"cpu"` allowed (slow for gps/sage/gcn) |
| `gpu_cap_gb`, `gpu_limit_gb`, `admission_hold_s`, `gpu_budget_h` | float, float, int, float | 3, 19, 180, 6 | card rule (admission: card + cap + 0.5 ≤ limit held 180 s; yield rc 75 + resume) |
| `bootstrap_reps` | int | 20 | time-split gene bootstrap |
| `topk` | list[int] | `[100, 1000, 10000]` | P@k / R@k / degree-matched control |
| `fdr`, `fdr_rule` | float, str | 0.1, `"nt"` | output threshold; `"combined"` = max(q_NT, q_cal) (also always reported) |
| `poorly_characterised` | dict | `{"string04_degree_q": 0.25, "go_terms_q": 0.25}` | output gene filter (GOA direct, ND/NOT/GO:0005515 excluded) |
| `prediction_arm` | str | `"noJUMP"` | arm for output lists (JUMP stays an independent check) |
| `specificity_table` | str (csv) | refdata `gene_specificity.csv` | flags for outputs |
| `seed` | int | 20261108 | master seed (all RNG streams derive from it as in the reference) |

## Graph construction
- **Nodes:**
  - Genes: rows of `ops_gene_table` with `n_cells` > `ops_min_cells`.
  - NT pseudo-genes: the mean of their 4 guide rows in the same space.
  - Category nodes: one per class in `class_table`, held-out tasks only.
  - Node and class counts of the reference are in `data_summary.json`.
- **Blocks:** each block is projected and summed with a learned missing-indicator embedding. NT nodes get NT pseudo-profiles for the extra blocks at the genes' coverage rate (seeded).
- **Edges:**
  - Union of `train_edges`, as unordered pairs. Each source is its own edge type (STRING 0, BioGRID 1, CORUM 2).
  - Member → category (3) and category → member (4) edges, held-out tasks only.
- **Versions (reference):**
  - STRING v11.5 (≥ 0.7) and v12.0 (new ≥ 0.7 pairs, diffed on ENSP);
  - BioGRID 4.4.214 (2022-09-25) and 5.0.263 (diffed on Entrez);
  - CORUM 3.0 (2018-09-03, Wayback 2022-10-16, md5 3c50a174…) and the published `corum_group_benchmark.tsv` (cheeseman-lab/brieflow benchmark file).
  - Provenance is in `funk_phenotype/improved/refs/versions.json`.
- **Time-split protocol:**
  - No category nodes, because the classes contain post-2022 CORUM.
  - Training: 95 % of 2022 edges. Validation: 5 % hidden.
  - Scoring: every gene pair, with the final message graph = all 2022 edges.
  - Candidates: real–real pairs that are not 2022 edges. Positives: `new_edges`.
- **Held-out protocol:**
  - For fold k, every edge and membership of the fold's genes is removed. Inner validation: 10 % of the other genes, isolated.
  - At scoring, the validation genes' edges are restored and the test genes stay isolated.
  - Scored: test gene × all genes, and test gene × all classes.
- **Controls:**
  - `shufX`: block rows permuted jointly among genes.
  - `shufY`: gene permutation applied to edges and memberships before fold removal; evaluated against the truth.
  - `rewire`: double-edge swaps (gene–gene) and bipartite swaps (gene–category), 2 × E each.
  - The negative exclusion set is the arm's own (permuted/rewired) edge set.

## Models
- **gps** (main):
  - 3 layers. Each layer: relational SAGE (mean per edge type + root) ‖ full multi-head self-attention over all nodes (SDPA, 4 heads) → sum → LayerNorm → FFN (2h, GELU) → LayerNorm.
  - Inputs: block projections, missing embeddings, and Linear(RWSE16 + log degree), recomputed every epoch on that epoch's message graph.
  - bf16 autocast. Sparse ops and decoders run in float32.
- **sage / gcn:** 3 layers, residual + LayerNorm. GCN uses the union graph with symmetric normalisation and self-loops.
- **Decoders:**
  - Pairs: DistMult, s = scale · Σ z_u r z_v + b.
  - Categories: scale · (z_g ⊙ q) · z_c + b_c.
- **Training:**
  - Node mask 15 % (held-out tasks), edge mask 20 %.
  - Negatives: 3 per positive, ½ uniform and ½ degree-proportional corruption, never a known pair.
  - BCE losses with the weights above.
  - Keep the best validation state.
- **lr:** the encoder is the raw concatenation (blocks + indicators), so DistMult = Hadamard logistic regression; the category head is linear. Supervision: every training edge (minibatch 50k positives per epoch) and every training gene.
- **spectral:** Laplacian eigenmaps (64, `eigsh` "LA" on D^-½AD^-½, v0 seeded) of the training graph only, then `lr`. Isolated nodes get zeros and the missing flag.
- **cosine:** per-block cosine averaged over the blocks both genes have. Categories: weighted vote of the top-25 most similar training genes (weights clipped at 0).
- **adamicadar:** A · diag(1/log deg) · A on the 2022 graph (time split only).
- **prior:** degree product (time) or partner degree (held-out); class size for categories.

## Metrics, controls, FDR
- **AP definition:** step-wise average precision (as `sklearn.average_precision_score` and `ic_lib.pair_auprc`). Ties are broken by seeded jitter of 1e-12 × max|s|.
- **Time split:**
  - AUPRC overall and per source, with a gene bootstrap CI (20 replicates, pair weight w_i·w_j).
  - P@k and R@k.
  - Degree-matched control: 10 random candidate pairs per top-k pair, drawn from the same 2022 degree-decile combination.
  - AUPRC within fitness tertiles (pair = min tertile of log cells per guide).
  - Fraction of the top 1,000 in the top degree decile and in the low-fitness tertile.
  - NT enrichment: NT × gene pairs above the real top-1,000 and top-10,000 thresholds ÷ expected.
- **Held-out:**
  - Links (vs 2022 and current union): pooled AUPRC and gene mAP.
  - Top-5 partner hub fraction, precision, and degree-matched precision.
  - Categories: micro AUPRC, macro AUPRC (classes positive in the fold), recall@10, micro AUPRC per fitness tertile.
  - NT enrichment for categories (top 1 %) and links (top 0.1 %).
  - Paired fold deltas, with the count of folds where A beats B.
- **Calibration:**
  - Categories: per fold, isotonic fit on the other 4 folds' out-of-fold scores (subsample ≤ 3 M, seeded).
  - Pairs: same, on a 10 % pair subsample; symmetrised as the mean of both directions.
  - Time-split pairs: two seeded gene halves, cross-applied.
  - Reported: ECE (10 equal-count bins) and Brier.
- **FDR:**
  - q_NT = (d(t) + 1)/n_decoy × n_cand/n(t), running minimum, with decoys = NT rows scored by the same models on the same calibrated scale.
  - q_cal = 1 − cumulative mean calibrated P.
  - q_combined = max of the two.
  - Output rule: `fdr_rule`. Pass counts are always reported for both rules.
- **JUMP support:**
  - Categories: mean cosine of the gene to class members, against 2,000 random same-size sets.
  - Pairs: the partner's rank among the gene's cosines.
  - p = (1 + #null ≥ obs)/(1 + n).
- **Drivers:** calibrated P under full / noJUMP / noOPS / graph. delta_OPS_over_graph = noJUMP − graph; delta_JUMP = full − noJUMP; delta_OPS_given_JUMP = full − noOPS. Label = blocks with Δ > 0.05, plus "graph" if p_graph ≥ ½ P.

## Outputs (under `<src>/gene_graph/`)
- **`data/`:**
  - `graph.npz`: `X_<block>`, `H_<block>`, `E_{train,new,cur}_<source>`, `E_cur_STRING04`, `GC`.
  - `nodes.parquet`: gene, sym, is_nt, fold, log_cells_per_guide, string04_degree_genome, go_terms, has_<block>.
  - `classes.parquet`.
  - `data_summary.json`.
- **`runs/<task>__<model>__<arm>__<control>__s<seed>/`:**
  - `scores.npz`: time → `S` (N × N float32); fold → `rows`, `S` (rows × N), `C` (rows × n_classes).
  - `log.csv`: epoch, loss, val, t.
  - `done.json`: name, task, model, arm, control, seed, blocks, dims, epochs, best_val, seconds, device, peak_reserved_gb, hparams.
- **`tables/`** (columns as in the reference CSVs):
  - Metrics: `time_split_metrics.csv`, `time_split_bootstrap.csv`, `heldout_metrics_per_fold.csv`, `heldout_summary.csv` (mean/sd per model × arm × control × seed).
  - Summaries: `ablation_table.csv/.md`, `heldout_paired_deltas.csv`, `calibration.csv` (task, fold, ECE, Brier, mean_p, base, model, arm), `fdr_summary_<model>_<arm>.csv`, `fdr_summary_timesplit_<model>_<arm>.csv`, `poorly_characterised_rule.json`, `run_configs.csv`.
  - Category predictions, `predictions_gene_category_<model>_<arm>.csv`: gene, class_id, class, source, class_size, p_calibrated, q_NT, q_cal, q_combined, q, string04_degree, go_terms, fold, p_<arm>…, JUMP_{CRISPR,ORF}_{cos,p,members}, delta_*, drivers, spec_flag_gene.
  - Held-out gene–gene predictions, `predictions_gene_gene_<model>_<arm>.csv`: gene, partner, p_calibrated, q, q_cal, q_combined, gene_string04_degree, gene_go_terms, partner_poorly_characterised, partner_2022_degree, gene_fold, partner_fold, p_<arm>…, JUMP_*_{cos,p}, delta_*, drivers, spec_flag_gene, spec_flag_partner.
  - Time-split gene–gene predictions, `predictions_gene_gene_timesplit_<model>_<arm>.csv`: the same as above, plus score, gene_2022_degree and q_NT.
  - Figure source data: `fig*.csv`.
- **`figures/`:** module figures, png + svg.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, pytest -n ≤ 4, synthetic graph of about 60 genes, 6 NT, 5 classes, 2 blocks with partial coverage)
1. Hadamard LR identity: `pair_matrix` of `lr` equals x · diag(w) · yᵀ · scale + b.
2. `rwse` equals NumPy diag(P^k) on a 6-node graph, including an isolated node (row of zeros).
3. Masked supervision never contains a test-fold gene, a validation gene or an NT node, and negatives never contain a known pair (including under shufY/rewire, which use the arm's own pairs).
4. Held-out leak test: no edge or membership touching a test gene appears in any message graph, including the final scoring graph.
5. `rewire_pairs` keeps every node degree (and the bipartite degrees), with no self-loops or duplicates.
6. `qvalues`: monotone; q = 1 when every decoy outranks every candidate; the (d + 1) correction gives the minimum q = n_cand/(n_decoy · n).
7. Cross-fitted isotonic: no fold's calibration sees that fold's labels (the fold's map comes from the other folds only).
8. Yield/resume: a forced yield after epoch 1 (mock `over_limit`) resumes to identical final scores on CPU (per-epoch RNG `default_rng([seed, s, epoch])`).
9. Planted-signal test: genes in two planted modules with module-specific features, controls on. Held-out link AUPRC must be > 3× base for `gps`/`sage`/`cosine`, and must fall to base ±50 % under `shufX` and `shufY`. NT nodes must get no top-1 % category score.
10. Settings round trip: `run_gene_graph(settings)` on the synthetic data writes every listed table with the listed columns, and a rerun from the saved settings file is byte-identical on CPU.

## Reference implementation
- **Scripts:** `R/gene_graph/scripts/`: `gg_paths.py`, `gg_data.py`, `gg_models.py`, `gg_train.py`, `gg_baselines.py`, `gg_eval.py`, `gg_predict.py`, `gg_predict_time.py`, `gg_flag.py`, `gg_figures.py`, `gg_tables.py`.
- **Launcher and queues:** `run_queue.sh`; queue files `queue_gpu.txt` (as run), `queue_gpu_v1_full168.txt`, `queue_gpu_v2_trimmed138.txt`.
- **Imported read-only:**
  - `funk_plateA_full/improved/scripts/ic_lib.py`: `_ref_pairs`, `embed`, `is_nt`, `_sym`.
  - `ops_jump_cross/scripts/ojx_cpg0016.py`: `gene_table`, whose cached parquets are used.
- **Docs:** `R/gene_graph/DESIGN.md`, `PORTING.md`.
- **Intermediates:** W/data, W/runs, W/logs.

## Validation data
| input | path |
|---|---|
| OPS genes / guides (S02-weighted) | `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/eval/external/W_bf_S02.parquet`, `W_bf_S02_guides.parquet` |
| OPS TVN (sensitivity) | `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/tables/plateA/final/final_tvn/genes_interphase_A.parquet`, `guides_interphase_A.parquet` |
| JUMP cpg0016 gene + NT tables | `/mnt/wd4tb/bmc_reanalysis/ops_jump_cross/work/cpg0016/{crispr,orf}_final_{genes,nt}.parquet` (from `/nas_mnt/data/jump_cpg0016_profiles/`) |
| references | `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/improved/refs/*.parquet` (archived + new), `/mnt/wd4tb/bmc_reanalysis/refdata/prepared/{string_v12_ge400,biogrid_physical,gene_specificity}`, the published `corum_group_benchmark.tsv`, `refdata/go/goa_human.gaf.gz` |
| classes / folds / NT grouping | `R/ops_gene_classes/tables_v3/{class_table,class_members,gene_folds,nt_pseudogenes}.csv` |
| expected outputs | `R/gene_graph/tables/` and `W/runs/*/scores.npz` |

## Acceptance (plate A)
Acceptance: identical output, or within the tolerances stated below, to the reference outputs at `R/gene_graph/tables/`, `W/data/` and `W/runs/*/`.
1. **Data assembly identical:**
   - `data_summary.json` equal to the reference `W/data/data_summary.json` (node counts, block dims, block coverage, edge counts per source for train/new/union, membership count).
   - `graph.npz` arrays equal: exactly for edges, folds and indicators; allclose 1e-6 for features, because of the PCA.
2. **Deterministic models identical:** cosine, adamicadar and prior `scores.npz` and their rows in all metric tables identical (metrics to 1e-9). The spectral embedding agrees up to sign/rotation, so it is compared on metrics (≤ 1e-4).
3. **Evaluation and prediction identical given the reference `scores.npz`:** run `evaluate`/`predict` on the reference run folders and reproduce the reference tables. That covers time_split_metrics (including bootstrap), heldout_metrics_per_fold, calibration, fdr_summary and all three prediction CSVs (`predictions_gene_category_*`, `predictions_gene_gene_*`, `predictions_gene_gene_timesplit_*`), row for row (floats ≤ 1e-9; isotonic/eigsh ≤ 1e-6).
4. **Trained models (gps/sage/gcn/lr; GPU non-deterministic: sparse atomics, bf16), on retraining with the same seeds:**
   - every fold-mean held-out metric (link AUPRC, category micro/macro AUPRC) within ±0.010 absolute of `R/gene_graph/tables/heldout_summary.csv`, with per-fold values within ±0.015 of `heldout_metrics_per_fold.csv`;
   - time-split AUPRC_any within the reference bootstrap CI (`time_split_bootstrap.csv`);
   - every paired-delta sign and "folds A better" count in `heldout_paired_deltas.csv` unchanged;
   - control arms stay at or below the levels in `ablation_table.csv` (+0.01).
   The tolerance is grounded in the seed-0 vs seed-1 spread of the reference runs (see `heldout_metrics_per_fold.csv`, seeds 0 and 1). On CPU with `torch.use_deterministic_algorithms(True)` and no autocast, a rerun of the spaCR port must be identical to itself.
5. **Output lists:** on the reference scores, the same rows as the reference `predictions_gene_category_<model>_<arm>.csv` and `predictions_gene_gene_timesplit_<model>_<arm>.csv` at the output threshold. After retraining, the Jaccard overlap of (gene, class) at q < 0.1 with the reference category predictions must be ≥ 0.7.
6. **Compute rules:** the GPU queue obeys the cap, admission, yield and budget guard and logs START/YIELD/FINISH/STOP lines to `~/.spacr/gpu/log`. Configs that the budget guard cut in the reference run (compare `run_configs.csv` with the queue files) run in the port's acceptance run, and their metrics are added to the reference tables.

## Effort
- **Effort:** L.
  - Port and tests: about 3–4 days.
  - Acceptance run: about 6–7 GPU-h on the shared card (the trimmed queue plus the cut configs) plus about 2 h CPU evaluation.
  - GUI result tab (alpha): about 1 day.
