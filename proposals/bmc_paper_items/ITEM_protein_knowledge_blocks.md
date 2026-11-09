# Item spec: protein-knowledge feature blocks for gene-level analysis (ESM-2, domains, fold-masked UniProt text, ProstT5-3Di, Foldseek), with out-of-library NT decoys and paralog-excluded evaluation

Related: `ITEM_gene_graph.md` (consumer: GPS, evaluation, calibration, FDR), `ITEM_multimodal_gene_fusion.md` (fusion built on these blocks), `ITEM_ops_reference_sets.md` (archived references), `ITEM_gene_class_training.md` (folds, classes, NT pseudo-genes).
- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs on the validation data, identically where the computation is deterministic and within the tolerances below where it is not (GPU fp16 embedding). Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **No hosted model or API:** every model runs locally from Hugging Face weights; there is no LLM API or provider default.
- **Code basis:** origin/nightly.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`; W = `/mnt/wd4tb/bmc_reanalysis/multimodal_gene`.

## Why
To judge what an imaging screen adds to gene-function prediction, the screen has to be compared with what prior protein knowledge alone predicts. That needs knowledge blocks (sequence embeddings, domains, text, structure) built so they cannot leak the answer:
- the text of held-out genes and every sentence naming them are removed;
- archived (earlier-release) versions are used for the time split;
- the decoy carries real protein knowledge, so missing blocks do not reveal it;
- evaluation is also available paralog-excluded, because sequence, domain and structure signal is partly paralog-driven.

## Target module
`spacr/ops/gene_knowledge/` (alpha), used by `spacr/ops/gene_graph/` (ITEM_gene_graph) and the fusion item. Entry points (documented callables, arrival contracts):
- `get_gene_knowledge_default_settings()`
- `fetch_protein_annotations(settings)`: UniProt REST current + an archived release (Swiss-Prot .dat), sequences, Pfam / InterPro cross-references, function text; out-of-library decoy proteins.
- `embed_sequences(settings)`: ESM-2.
- `embed_text(settings)`: sentence-level, with the mentions index.
- `predict_3di(settings)`: ProstT5 encoder + CNN head, plus the 3Di-string embedding. With AFDB files available: `foldseek createdb` on the structures instead, with pLDDT masking.
- `foldseek_similarity(settings)`: all-vs-all search, edges, paralog pairs.
- `build_knowledge_blocks(settings)`: per-task blocks (fold-masked text, `_time` versions), decoy blocks, structure blocks.
- `paralog_mask(settings)`.

| file | from (reference) | content |
|---|---|---|
| `fetch.py` | `mm_fetch_uniprot.py`, `mm_parse_sprot2022.py` | REST accession batches (<= 100, accessions endpoint), exact-gene fallback, seeded decoy draw (80 -> first 63 with sequence), 2022_01 .dat streaming parser |
| `embed.py` | `mm_esm.py`, `mm_text.py`, `mm_prostt5.py` | ESM-2 windows 1,022/511, per-sequence accumulation; PubMedBERT mean pooling; ProstT5 windows 1,000/750 |
| `structure.py` | `mm_structure.py` | foldseek base:createdb AA + 3Di, exhaustive search E <= 1e-3, edges, fident > 0.30 paralogs, 3Di k-mer TF-IDF, spectral block |
| `blocks.py` | `mm_data.py` | PCA/SVD per block, unit RMS, missing indicators, per-fold text masking, NT decoy assignment |
| `paralog.py` | `mm_eval.paralog_matrix` | fident > 0.30 union shared Pfam |

## Settings (`get_gene_knowledge_default_settings`)
| key | type | default | meaning |
|---|---|---|---|
| `src` | str | - | output folder |
| `genes` | str (parquet) | gene graph `nodes.parquet` | node order (genes + NT pseudo-genes) |
| `uniprot_function_cache` | str (tsv) | `refdata_archive/uniprot_function_2026-11-08.tsv` | gene -> accession, function text |
| `uniprot_archive_release` | str | `"2022_01"` | Swiss-Prot release for time-split blocks (`uniprot_sprot-only2022_01.tar.gz`, ftp.uniprot.org) |
| `decoy_proteins` | int | 63 | out-of-library reviewed human proteins (seeded) for NT decoys |
| `esm_model` | str | `facebook/esm2_t33_650M_UR50D` | local weights dir `models_dir` |
| `esm_window`, `esm_stride`, `esm_tokens_per_batch` | int | 1022, 511, 6000 | |
| `esm_pca` | int | 128 | fit on library genes, current sequences |
| `text_model` | str | `NeuML/pubmedbert-base-embeddings` | mean pooling, L2-normalised, max 256 tokens |
| `text_pca` | int | 64 | refit per fold on the masked matrix |
| `text_mask_mentions` | bool | True | drop every sentence naming a held-out gene (whole-word symbol or accession) |
| `domain_min_genes` | int | 3 | InterPro + Pfam entries kept |
| `domain_svd` | int | 64 | TF-IDF -> TruncatedSVD (seed 0), fit per version |
| `prostt5_model`, `prostt5_cnn` | str | `Rostlab/ProstT5_fp16`, ProstT5 `cnn_chkpnt/model.pt` | |
| `prostt5_window`, `prostt5_stride`, `prostt5_residues_per_batch` | int | 1000, 750, 2000 | |
| `structure_source` | str | `"prostt5"` | `"afdb"`: AFDB v4 PDB/CIF files -> `foldseek createdb --mask-bfactor-threshold 70` |
| `foldseek_evalue`, `foldseek_exhaustive` | float, bool | 1e-3, True | |
| `paralog_fident`, `paralog_shared_pfam` | float, bool | 0.30, True | paralog pairs |
| `device`, `gpu_cap_gb`, `gpu_limit_gb` | str, float, float | `"cuda"`, 4.5, 19 | card admission as ITEM_gene_graph |
| `seed` | int | 20261109 | |

## Leakage rules (must be enforced in code and tested)
1. Fold k: the fold's genes and the NT decoys in fold k get no text vector (indicator off). Every sentence of every other gene that names a fold-k gene is dropped before averaging. Text PCA is refit per fold.
2. Time split: blocks `_time` come from the archived release only (sequence, cross-references, text). ESM-2 counts as valid (UniRef50 2021_04). ProstT5 is flagged "approximately valid".
3. Sequence, domain and structure blocks are kept for held-out genes. Domain names are never embedded; InterPro2GO is never used. GO is never an input.
4. NT decoys: screen blocks = NT profiles; knowledge blocks = an out-of-library protein (not one of the screen's genes), so missing blocks cannot reveal decoys.
5. Foldseek edges are message-passing inputs only (never labels, positives or negatives).

## Outputs
- `blocks.npz`: `X_esm`, `X_esm_time`, `X_dom`, `X_dom_time`, `X_txt_f0..f4`, `X_txt_time`, `X_txt_all` (profile reference only, marked circular), `X_img_*` (pass-through), each with `H_*`.
- `structure.npz`: `X_s3di`, `X_s3di_kmer`, `X_struct_spec`, `E_struct`, `E_paralog_fident30`.
- `node_protein.parquet`; `blocks_summary.json`; `structure_summary.json`.
- `text/sentences.parquet` (version, gene, sentence, sid, mentions) and `sent_emb.npy`.
- `esm/esm_by_md5.npz`, `prostt5/{3di.fasta, aa.fasta, emb3di_by_md5.npz}`.
- `profile_metrics.csv`: SELECT / REPORT mAP per block and per concatenation, plus paralog-excluded SELECT.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers; tiny models or mocked embedders)
1. Text masking: on a synthetic table, no test gene has a text vector, and no kept sentence of any gene names a test gene (symbol or accession).
2. Per-fold PCA never sees a test row (fit matrix rows = present, non-test).
3. Time blocks read only the archive fields (mock archive with sentinel values).
4. Decoys: the decoy proteins are disjoint from the gene universe, and decoy indicator rates equal the genes' rates (+- 2 %).
5. ESM / ProstT5 windowing: the per-sequence mean equals the mean over residues of the window-averaged per-residue vectors (synthetic embedder).
6. Foldseek parsing: edges are symmetric and unique with no self pairs; paralog pairs are a subset of library x library; identical sequences on two nodes form a paralog pair.
7. Paralog mask: removes exactly the shared-Pfam union fident > 0.30 pairs.
8. Settings round trip: a rerun from the saved settings file is byte-identical on CPU (fp32) for every deterministic step.

## Reference implementation
- **Scripts** (`R/multimodal_gene/scripts/`): `mm_paths.py`, `mm_gpu.py`, `mm_fetch_uniprot.py`, `mm_parse_sprot2022.py`, `mm_download_models.py`, `mm_download_prostt5.py`, `mm_esm.py`, `mm_text.py`, `mm_prostt5.py`, `mm_structure.py`, `mm_data.py`, `mm_profile.py`.
- **Data** (W): `downloads/` (UniProt current raw, 2022_01 tarball, InterPro entry.list), `data/`, `esm/`, `text/`, `prostt5/`, `structure/`.
- **Tool:** foldseek static binary 463739e (`W/tools/foldseek`).

## Validation data and acceptance (plate A)
1. Inputs: the library genes and NT pseudo-genes in gene graph order, the UniProt cache, UniProt 2022_01, the models above.
2. Coverage, vocabulary sizes, dropped-sentence counts, Foldseek edge and paralog-pair counts must match `W/data/blocks_summary.json` and `W/structure/structure_summary.json` exactly.
3. Deterministic blocks (domains, text given the sentence embeddings, k-mer, spectral up to sign) must be equal to 1e-6. GPU fp16 embeddings must have cosine >= 0.999 to the reference per protein, and the PCA blocks must give profile SELECT mAP within +- 0.002 of the reference `profile_metrics.csv`.
4. Downstream: with these blocks, the kNN stack metrics in `R/multimodal_gene/tables/heldout_metrics_per_fold.csv` must be reproduced to 1e-9 (deterministic).

## Effort
- M-L (3-4 days port + tests; 0.5 GPU-h embeddings; 15 CPU-min Foldseek).
- AFDB mode needs the 5.1 GB AFDB human tarball, fetched on the work network.
