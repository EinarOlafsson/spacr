# Item 980: Voyager3 enrichment mode, a non-LLM cluster annotation

- **Number:** 980. Approved by the maintainer on 2026-11-18 at about 07:30.
- **Status:** alpha (`ALPHA_FEATURES[980]`).
- **Builds on:** item 973 (Voyager3), with the same clusters, tables, report, provenance and GUI button.
- **Default:** `ops_voyager3_method` is `'ai'`. With the new settings unset, Voyager3 and OPS output is byte-identical to before.

## Design card

**Question.** Does a deterministic, model-free annotator name the roles of clusters as often as the LLM annotators, at the same or a lower rate on shuffled clusters?
- The LLM annotators are Voyager3 AI and MozzareLLM.
- The enrichment annotator has no model to cost, is reproducible bit for bit, and runs offline.

**Data.**
- **Clusters:** Funk plate A, the 257 tvn_r10 clusters (`cluster_assignments_final_tvn.csv`, column `leiden_r10`).
- **Shuffled control:** 257 clusters with the genes shuffled across all 257, sizes kept, seed 20261117. These are the same real and shuffled clusters as the Voyager3 AI run and the MozzareLLM rerun.
- **Background:** all 5,049 targeting genes of the screen.
- **Annotation sources:**
  - UniProt keywords and GO terms (process, component, function), complete term lists, from Voyager3's own UniProt fetch of reviewed entries.
  - CORUM complexes, human (`refdata_archive/corum/corum3_allComplexes_wayback20221016.txt.zip`, CORUM 3.0 archive).
  - Reactome pathways (`refdata_archive/reactome_go_2026-11-11/ReactomePathways.gmt`).
  - Each source is copied into `results/ops_voyager3/reference_cache/`, with its name, SHA-256, file date and the PROVENANCE note beside it. Later runs work offline.

**Method, per cluster.**
1. **Test:** a one-sided hypergeometric test of every term with at least 2 cluster genes. The background is the screen's genes.
   - p = P(X ≥ k), where k is the term's genes in the cluster.
   - The parameters are N (background size), K (the term's genes in the background) and n (cluster size).
2. **Correction:** Benjamini-Hochberg FDR across the terms tested in that cluster.
3. **Significant term:** FDR ≤ `ops_voyager3_enrichment_fdr` (default 0.05), at least 3 cluster genes, and fold enrichment (k/n)/(K/N) ≥ 2.
4. **Best term:** the significant term with the lowest FDR. Ties go to the higher fold, then the smaller K, then the term id.
   - The best term gives the cluster's role.
   - The top 3 terms are given in the summary.
5. **Confidence:** the share of cluster genes in the best term, under Voyager3's own coverage rule.
   - Above 70 % is high, 50–70 % medium and 30–50 % low.
   - Below 30 %, or no significant term, is none.
   - So the column compares directly with Voyager3 AI's recount and with MozzareLLM's 70/50/30 % rule.
6. **Supporting genes:** the best term's cluster genes.
7. **Candidates:** in high and medium clusters, the genes without the best term. Each is listed with its UniProt annotation score.
8. **Outputs:**
   - the same `voyager3_clusters.csv`, `voyager3_candidates.csv`, `voyager3_gene_facts.csv`, report and provenance as AI mode;
   - `voyager3_enrichment_terms.csv`, with every tested term: cluster, kind, source, term id, name, k, n, K, N, fold, p, FDR and genes.
   - `voyager3_replay` rebuilds all of these byte for byte.

**Settings (alpha, `ALPHA_FEATURES[980]`).**
- `ops_voyager3_method`: `'ai'` or `'enrichment'`, default `'ai'`.
- `ops_voyager3_reactome`: a GMT file. Empty means no Reactome.
- `ops_voyager3_corum`: a CORUM complexes table, as .txt or .zip. Empty means no CORUM.
- `ops_voyager3_enrichment_fdr`: default 0.05.

**Controls.**
1. The shuffled clusters, as above.
2. A planted-truth synthetic screen:
   - 2,000 background genes and 200 random terms;
   - 40 clusters of 10–40 genes, each with one planted term covering 40–90 % of it;
   - 40 random clusters with no planted term.

**Statistics.** Proportions carry 95 % Clopper-Pearson intervals. Agreement with Voyager3 AI's confidence class and with MozzareLLM's class is reported as % agreement and Cohen's κ.

**Pre-set criterion (fixed before the run).**
- (a) Planted truth: the planted term is the best term in at least 95 % of planted clusters.
- (b) At most 5 % of random synthetic clusters reach medium or high.
- (c) On the real data, shuffled clusters at medium or high stay at or below 5 % (upper CI bound at or below 8 %).
- (d) The 514-cluster run, after the UniProt fetch, takes under 5 minutes on 2 CPU threads.

**Compute.** CPU only, minutes. There are no LLM calls.

## Report
Enrichment, Voyager3 AI (Sonnet) and the MozzareLLM v0.2.0 rerun (Sonnet), each on real and shuffled clusters, are shown beside MozzareLLM's published values (Nat Commun 2026, p. 8, Supp Fig 7b). A value with no published counterpart says "n/a in paper". A glossary is included.
