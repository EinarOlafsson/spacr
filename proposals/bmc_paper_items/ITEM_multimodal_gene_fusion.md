# Item spec: multimodal gene-function prediction: screen x prior-knowledge fusion, screen-contribution controls, and an explained hypothesis table

Depends on `ITEM_gene_graph.md` (GPS model, held-out protocol, calibration, NT target-decoy FDR) and `ITEM_protein_knowledge_blocks.md` (inputs, leakage rules, paralog mask).
- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs on the validation data, identically where the computation is deterministic and within the tolerances below where it is not (GPU training). Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **No hosted model or API.**
- **Code basis:** origin/nightly.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`; W = `/mnt/wd4tb/bmc_reanalysis/multimodal_gene`; G = `/mnt/wd4tb/bmc_reanalysis/gene_graph`.

## Why
To state what an imaging screen adds over prior protein knowledge, screen-based and knowledge-based predictions must be fused and compared under controls. A fixed winner rule, written before the runs, picks the fusion modes. A mode qualifies only if it beats the screen-only reference and the knowledge-only model in at least 4 of 5 held-out folds, and its screen-shuffled control loses in at least 4 of 5.

Three modes are built:

| mode | task | definition |
|---|---|---|
| `late_gps` | links | z-mean of GPS(screen) and GPS(knowledge blocks) |
| `knn_stack` | categories | kNN stack over all blocks + structure (ops, img, esm, dom, txt, s3di, Foldseek spectral) |
| `late_gps_knn` | categories, paralog-excluded | z-mean of GPS(knowledge) and the kNN stack (all) |

Each is reported with the screen-only reference, the knowledge-only model and the screen-shuffled control, plain and paralog-excluded, with paired fold deltas. The hypothesis table gives out-of-fold, controlled, explained calls with calibrated probabilities and drivers.

## Target module
`spacr/ops/gene_function/` (alpha). Entry points (documented callables, arrival contracts):
- `get_gene_function_default_settings()`
- `run_knn_stack(settings)`
- `run_late_fusion(settings)`
- `evaluate_gene_function(settings)`: gene graph metrics plus the paralog-excluded variants, paired deltas, the screen-contribution table and the winner rule.
- `explain_gene_function(settings)`: calibrated hypotheses with drivers, explanation and new partners.
- `run_gene_function(settings)`: runs all of the above in order.

| file | from (reference) | content |
|---|---|---|
| `stack.py` | `mm_baselines.py` (+ gene graph `gg_baselines.run`) | per-block cosine averaged over shared blocks; 25-NN similarity-weighted vote; arms incl. noOPS / noScreen / shufOPS / shufScreen |
| `train.py` | `mm_train.py` | GPS / LR on knowledge blocks via ITEM_gene_graph's trainer: task-specific blocks (`_f<k>`, `_time`), `shufOPS` / `shufScreen` controls, optional 6th edge type (Foldseek) |
| `late.py` | `mm_late.py` | z-score each run's S and C over the whole matrix, mean; screen-shuffled controls (screen model replaced by its feature-shuffled run) |
| `evaluate.py` | `mm_eval.py`, `mm_tables.py` | gene graph `time_metrics` / `fold_metrics` + paralog-excluded link and category AUPRC; paired deltas; winner rule |
| `explain.py` | `mm_predict.py`, `mm_hypotheses.py` | cross-fitted isotonic, q_NT / q_cal (gene graph functions), drivers, explanation, new partners, flags |
| `figures.py` | `mm_figures.py` | bars + jittered fold points (~60 % opacity), controls hatched; screen-contribution deltas |

## Settings (`get_gene_function_default_settings`)
| key | type | default | meaning |
|---|---|---|---|
| `src` | str | - | output folder |
| `gene_graph_src` | str | - | gene graph outputs (graph, folds, classes, reference runs) |
| `knowledge_src` | str | - | knowledge-block outputs |
| `stack_arms` | dict[str, list[str]] | the 22 `k_*` arms of `mm_baselines.STACK` | kNN stack arms |
| `stack_controls` | dict[str, str] | `k_a/k_b/k_c/k_g/k_h: shufOPS`, `k_d/k_e/k_ep: shufScreen` | |
| `knn_k` | int | 25 | vote |
| `gps_arms` | dict | `mm_train.ARMS` | e.g. `f: [esm, dom, txt]`, `c: f + [ops]`, `ep: all + [s3di, ops, struct]` |
| `gps_controls` | list | `["c:shufOPS", "ep:shufScreen"]` (reference run; extend: `g/h:shufOPS`, `e:shufScreen`) | |
| `late_pairs` | dict | `{"late_opsf": [gene_graph gps noJUMP, gps f], "late_opsf_kNNe": [gps f, cosine k_e]}` | late fusions + their shuffled controls |
| `paralog_eval` | bool | True | report every held-out metric also paralog-excluded |
| `winner_rule` | dict | >= 4/5 vs screen-only reference and vs knowledge-only; shuffled control loses >= 4/5; simpler arm wins within 0.005 if not worse in >= 3/5 | fixed before the runs |
| `hypothesis_genes` | str | `"poorly_characterised"` | gene graph rule (bottom quartile STRING >= 0.4 degree and GO terms) |
| `hypothesis_top_per_gene` | int | 4 | |
| `screen_specific_delta` | float | 0.05 | winner - knowledge-only and winner - shuffled |
| `link_partners_from` | str | `"late:late_opsf"` | partner column |
| `seed` | int | 20261109 | |

## Outputs (under `<src>/gene_function/`)
- `runs/` (gene graph score format), `tables/heldout_metrics_per_fold.csv`, `time_split_metrics.csv`, `results_table.csv`, `paired_deltas.csv`, `winner_candidates_<metric>.csv`, `winners.json`.
- `calibration_<model>_<arm>.csv`, `fdr_summary_<model>_<arm>.json`.
- `hypotheses_<model>_<arm>.csv`, with columns: gene, rank_in_gene, class_id, class, source, class_size, fold, p_calibrated, q_NT, q_cal, q_combined, string04_degree, go_terms, p_knowledge_only, p_screen_only, p_winner_shuffled, delta_screen, delta_knowledge, delta_vs_shuffled, OPS_z, OPS_similar_members, IMG_z, ESM_nearest_member, ESM_nearest_pct, shared_domains, struct_hit_members, explanation, top_new_partners_link_winner, posthoc_uniprot_mentions_class, drivers.
- `hypotheses_ranked.csv` (+ screen_specific, call_type, gene_graph_q_NT, spec_flag) and `hypotheses_top.md`.
- `figures/fig1_heldout_link`, `fig2_heldout_category`, `fig3_screen_contribution` (png + svg) with their source CSVs.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers, synthetic gene graph from ITEM_gene_graph tests + 2 knowledge blocks)
1. The kNN stack equals the gene graph cosine baseline when given one block (identity).
2. Late fusion of a run with itself has the same ranking as the run; z-scoring uses only that run's matrix.
3. `shufOPS` permutes only the ops rows among genes; `shufScreen` permutes ops and img with one shared permutation; the other blocks are untouched.
4. The paralog-excluded metrics equal the plain metrics when the paralog mask is empty, and drop exactly the masked cells otherwise.
5. Winner rule: on a planted table, the arm chosen matches a hand-computed choice, including the simpler-arm clause.
6. Explanation: no field reads the gene's own text, edges or memberships (mock data with sentinels); members used are training-fold members only.
7. Planted-signal test: with a planted screen-only module, the stack + OPS beats knowledge-only and the shuffled control in every fold. With a knowledge-only module, the screen-specific flag stays off.
8. Settings round trip: byte-identical rerun on CPU for the stack, late fusion, evaluation and the explanation tables.

## Reference implementation
- **Scripts** (`R/multimodal_gene/scripts/`): `mm_train.py`, `mm_baselines.py`, `mm_late.py`, `mm_eval.py`, `mm_tables.py`, `mm_predict.py`, `mm_hypotheses.py`, `mm_figures.py`.
- **Queues:** `queue_gpu_A.txt`, `queue_gpu_B.txt`, `queue_gpu_C.txt` (C not run), `queue_gpu_1/2/3.txt` (superseded plan), `queue_cpu_lr_*.txt`.
- **Drivers:** `run_gpu_chain.sh`, `run_gpu_chain2.sh`.
- **Runs:** `W/runs/` (every config incl. losers); reference runs reused from `G/runs/` (gps noJUMP / graph / shufY, cosine noJUMP / shufX, lr noJUMP, prior, adamicadar, gps full shufX).

## Acceptance (plate A)
1. **Deterministic modes are identical:** the kNN stack and late fusion `scores.npz`, and their rows in `heldout_metrics_per_fold.csv`, `time_split_metrics.csv` and `paired_deltas.csv` (to 1e-9), against `R/multimodal_gene/tables/`.
2. **Evaluation and explanation are identical given the reference scores:** `winners.json`, `calibration_cosine_k_ep.csv`, `fdr_summary_cosine_k_ep.json`, `hypotheses_cosine_k_ep.csv`, `hypotheses_ranked.csv` (floats <= 1e-9; isotonic <= 1e-6; the OPS/IMG z-scores use a seeded 1,000-set null and must be identical with the same seed).
3. **GPS retraining** (GPU non-deterministic): fold-mean link and category AUPRC within +- 0.010 of the reference `results_table.csv` (per fold +- 0.015) for f, c, e, ep and their controls. Every paired-delta sign and every "folds better" count in `paired_deltas.csv` is unchanged, and `winners.json` is unchanged.
4. **Hypotheses:** after GPS retraining, the rows flagged screen_specific in the reference `hypotheses_ranked.csv` must stay among the top 20 with the same predicted class.
5. **GPU rule:** the card admission / yield / budget lines appear in `~/.spacr/gpu/log`. The port's acceptance run also runs the GPS arms the reference budget cut (g, h, their noOPS and shuffled controls, e_noOPS, ep_noOPS, ep_noScreen, the time-split arms) and adds them to the reference tables.

## Effort
- **Effort:** M-L.
  - Port and tests: about 3 days on top of ITEM_gene_graph and ITEM_protein_knowledge_blocks.
  - Acceptance: about 2.5 GPU-h (reference arms), plus about 4.5 GPU-h for the cut arms, plus about 2 CPU-h.
  - GUI result tab: hypothesis table with an explanation column, alpha.
