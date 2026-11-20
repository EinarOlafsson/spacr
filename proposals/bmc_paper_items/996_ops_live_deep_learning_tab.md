# Item spec 996: OPS Live "Deep-learning features" tab

- **Lifecycle:** future (alpha). The tab sits behind "Show alpha features" AND the OPS Live switch (item 971). Registered as `ALPHA_FEATURES[996]`. Off by default.
- **Number:** 996 was the next free number on 2026-11-20. Recheck `features/feature_list/` and the claims files before landing.
- **Rule:** a settings-file rerun gives IDENTICAL preview output (same sample, same embeddings on the CPU path, same QC table, same 2-D coordinates, same gallery order). Every documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (written against 53e91b5b17).
- **Paper-critical:** no (a tool for choosing and checking a backbone before a run).
- **Related:** 971 (OPS Live preview), 991 (shared SubCell loader), 990 / 997 (cell-cycle tab), 995.

## BUILD GATE
**Implementation starts only when both of these are "Ready" in `proposals/READINESS.md`:**
1. the SubCell shared loader (item 991, `spacr.embeddings`), and
2. the chosen backbone (`subcell_rbg` or `subcell_ybg`; DINO only if it is picked as the baseline).

Before then this file is the only deliverable. Readiness of the tab itself: Under development until acceptance A1-A3 below pass.

## 1. What it does
On the Objects & phenotype crops of the preview workspace, the tab:
- embeds a small, seeded sample of crops with the chosen backbone, through the one shared loader in `spacr.embeddings`;
- shows embedding QC;
- shows a 2-D PCA view of the sample;
- shows a nearest-neighbour crop gallery (click a point, see its k nearest crops).

It writes only into the preview workspace (`~/.cache/spacr/ops_live/<digest>/`). It never touches the run's output folder.

## 2. Dependency and integration
- `_STEP_NEEDS["embed"] = ("objects",)`. Through `_closure` this also pulls `align`; nothing new is computed for those.
- A new step key `"embed"` is added to `spacr.ops_live` **only through one function** (`_steps_for(alpha)`) that returns `_STEPS` unchanged when the alpha flag is off. The module-level `_STEPS`, `_STEP_NEEDS` and `_STEP_ORDER` tuples stay as they are, so the off state is byte-identical (T1).
- `_run_embed(ctx, cancelled=None)` follows the other `_run_*` functions: it reads `_result(ctx, "objects", cancelled)`, runs on `_preview_settings(...)` and caches under `_signature(settings, "embed")`, so a changed setting reruns only this tab.
- Input: the crops the objects step already wrote (`_write_crop` / the cell table with its crop window). No new segmentation, no new crop code.
- Workspace layout: `<workspace>/embed/` holds `sample.parquet` (ids, seed), `embedding.npy`, `qc.json`, `pca.parquet`, `neighbours.parquet`, `gallery/`. Nothing else is written.

## 3. Settings (all alpha; names prefixed `ops_live_dl_`)
| Key | Values | Default | Meaning |
|---|---|---|---|
| `ops_live_dl_backbone` | `'subcell_rbg'` \| `'subcell_ybg'` \| `'dino'` | `'subcell_rbg'` | Backbone. `'dino'` is a labelled natural-image baseline only (maintainer rule 2026-11-19). |
| `ops_live_dl_reference_channels` | two channel names | `''` (= DNA + tubulin for rbg, DNA + ER for ybg, from the run's channel map) | The two reference planes, in model order. |
| `ops_live_dl_protein_channels` | list of channel names | `''` (= every remaining phenotype channel) | One pass per protein plane; the passes are concatenated (991 item 2). |
| `ops_live_dl_pixel_size_um` | float | `0` (= the run's `pixel_size_um`) | Passed to `EmbeddingSpec.pixel_size_um` (991 item 3). |
| `ops_live_dl_sample` | int | `256` (GPU), `64` (CPU) | Crops embedded. The CPU default applies when no GPU is usable. |
| `ops_live_dl_neighbours` | int | `8` | Crops shown per query. |
| `ops_live_dl_random_control` | bool | `True` | Also embed the sample with the random-weights twin (`init="random"`, 991 item 6) and show its QC row beside the trained one. |
| `ops_live_dl_vram_gb` | float | `0` (= automatic) | VRAM cap. Read through `sched_cap` where that helper is available (guarded import; at the time of writing it is not in the repo, so the build agent checks first; otherwise the existing GPU admission helper, else no cap). |
| `ops_live_dl_seed` | int | `0` | Sample draw, PCA, neighbour ties and the random twin. |

The tab adds no switch of its own. With the alpha flag off the keys are absent from the schema, help and panel.

## 4. UI layout
New page class `_EmbedPage` in `spacr/qt/widgets/ops_live_preview.py`, built like the other pages (blurb, picture, table, sample line), registered only when the alpha flag is on.
- Top row: blurb; backbone, sample size and seed shown read-only from the form (the form stays the single place for settings).
- Left: PCA scatter (matplotlib, spaCR palette navy/teal/blue, 60 % alpha; points = crops; the random-twin PCA as a second small panel when `ops_live_dl_random_control`).
- Right: QC table (rows: trained, random twin; columns below).
- Bottom: neighbour gallery. Click a point: the query crop plus its k nearest crops (cosine on mean-centred embeddings), one fixed display range per channel (no per-crop stretch).
- A status line shows backbone, device, sample size actually used and the weight digest.

**QC table columns**
- crops embedded; embedding width;
- non-finite columns (count);
- zero-variance columns (count);
- mean centred cosine (mean pairwise cosine after subtracting the sample mean; near 0 is healthy, near 1 is collapsed);
- per-pass protein-channel summary: for each protein pass, the pass's mean per-column SD and its share of total variance.

## 5. Functions reused (no duplicated helpers)
- `spacr.embeddings.EmbeddingSpec`, `embed_array` (crops to embeddings), the SubCell passes through 991's public wrapper (`_subcell_model` / `_subcell_passes`), `encoder_entry` for provenance and the fingerprint. The tab does not load weights itself.
- `spacr.ops_live`: `_context`, `_result`, `_signature`, `_workspace`, `_preview_settings`, `_check`, `_write_crop`, `_run_step` dispatch.
- QC: existing embedding-QC code in `spacr.embeddings` (grep first at build time); the only new code is the small pure function that assembles the table.
- PCA: the call the Clustering tab already uses (one call, fixed `random_state`).
- Neighbours: the cosine k-NN already used by the retrieval code, or `sklearn.neighbors.NearestNeighbors(metric="cosine")`, one call.
- Qt: `LivePreviewContract`, the `PREVIEW_*` strings and the existing worker thread of `_OpsLivePanel`.

## 6. GPU and CPU
- GPU optional. With CUDA the default sample is 256; on CPU it is 64, and the status line says "CPU, smaller sample".
- Batch size is the smaller of the embeddings default and what `ops_live_dl_vram_gb` allows.
- A GPU run launched from a session follows the GPU turn rule (`tools/gpu_turn.sh`); inside the GUI it uses the user's own admission.
- Tests run with `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` at most.

## 7. Tests (CPU, synthetic, small)
- **T1 off switch:** with the alpha flag off, `_STEPS`, `_STEP_NEEDS`, the tab order and the output of every existing step are byte-identical to nightly; the new keys are not in the settings schema or panel.
- **T2 workspace only:** run the tab on the synthetic OPS fixture under a temporary `$HOME`; list every file created or modified outside `<workspace>/embed/` (and the existing workspace caches); the set must be empty. The run's `dst_root` stays untouched.
- **T3 smoke:** synthetic crops with a stub encoder (`embed_array(..., encoder=...)`, no weights downloaded) give a QC table with correct counts; planted non-finite and zero-variance columns are counted exactly; a constant embedding reports mean centred cosine as NaN with a message, and an all-same-direction embedding gives about 1.
- **T4 identical rerun:** run the same settings file twice (CPU, same seed): `embedding.npy`, `qc.json`, `pca.parquet`, `neighbours.parquet` byte-identical.
- **T5 gallery:** the nearest neighbours of a planted duplicate crop include the duplicate at rank 1; the query is excluded from its own list.
- **T6 backbone gate:** `'dino'` shows the "natural-image baseline" label; an unknown backbone is refused with a message.
- **T7 offscreen Qt:** the page builds, shows the "Not run yet" text, and fills after a worker run.
- **T8 random twin:** the random-weights QC row is produced without any download.
- Ratchets before handover: raw-reader, nested-function, documented-callable and hash-comment.

## 8. Acceptance (makes the tab Ready)
- **A1:** on the OPS end-to-end test data, the tab's trained-backbone embeddings equal what `spacr.embeddings.embed_array` returns for the same crops and spec (identical on CPU).
- **A2:** on a Funk plate A well A1 sample, the QC row (non-finite, zero-variance, mean centred cosine, per-pass summary) matches the SubCell arm's script QC for the same crops (card 20261119, check I5).
- **A3:** the settings-file rerun is identical (T4) on both datasets.

## 9. Strings for Codex
All translations go to Codex via `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_996_task.md` (sources with SHA-256, 9 locales). No translations are written in this item. Every user-facing string the tab adds, literally:

**Tab title**
- `Deep-learning features`

**Blurb**
- `Embed a small sample of the phenotype crops with the chosen deep-learning backbone, check the embedding and see which crops it places next to each other.`

**Labels and headings**
- `Backbone`
- `Sample size`
- `Seed`
- `Device`
- `Weights`
- `Embedding quality`
- `Trained weights`
- `Random weights (control)`
- `Crops embedded`
- `Embedding width`
- `Non-finite columns`
- `Zero-variance columns`
- `Mean centred cosine`
- `Protein channel passes`
- `Pass`
- `Mean column SD`
- `Share of variance`
- `Embedding map (PCA)`
- `PC1`
- `PC2`
- `Nearest crops`
- `Query crop`
- `Neighbour {rank}`
- `Cosine {value}`

**Backbone names shown in the picker**
- `SubCell (reference + protein: DNA, tubulin)`
- `SubCell (reference + protein: DNA, ER)`
- `DINO (natural-image baseline)`

**Tooltips**
- `The deep-learning model that turns each crop into a list of numbers. SubCell is trained on fluorescence images; DINO is trained on everyday photographs and is shown only as a baseline.`
- `How many crops are embedded. A smaller sample is used when no GPU is available.`
- `The same seed always draws the same crops and gives the same map.`
- `Columns that are not finite numbers. There should be none.`
- `Columns that have the same value in every crop. They carry no information.`
- `How alike the crops are after the average is removed. Near 0 means the crops differ; near 1 means the model gives almost the same answer for every crop.`
- `The same model with random, untrained weights. If the trained weights do not beat it, the embedding is not learning anything from your images.`
- `Click a point to see the crops closest to it in the embedding.`

**Messages**
- `CPU, smaller sample: {n} crops.`
- `GPU: {n} crops.`
- `This backbone is a natural-image baseline, not a fluorescence model.`
- `The model weights are not available. Fetch them first (see the embeddings help).`
- `Run the Objects & phenotype tab first: this tab embeds its crops.`
- `Not enough crops to embed ({n}). Increase the sample or check the objects step.`
- `Unknown backbone: {name}.`
- `Mean centred cosine is not defined: every embedding is identical.`
- `Alpha: this tab is under development.`

**Settings help (the nine `ops_live_dl_*` keys, in table order)**
- `Backbone for the Deep-learning features tab.`
- `The two reference channels, in model order (for example DNA and tubulin).`
- `The protein channels, one model pass each; the passes are joined.`
- `Pixel size in micrometres; 0 uses the run's pixel size.`
- `How many crops to embed.`
- `How many nearest crops to show per query.`
- `Also embed with random weights, as a control.`
- `GPU memory limit in GB; 0 chooses automatically.`
- `Random seed for the sample, the map and the control.`

## 10. Effort
M: step + page S-M; QC table and gallery S; tests S. Blocked by the build gate, not by code.
