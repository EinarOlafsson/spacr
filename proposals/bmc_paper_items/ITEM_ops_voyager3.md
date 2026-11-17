# Item 973: Voyager3, LLM cluster annotation from fetched UniProt facts (the last OPS step)

- **Number:** 973 (reserved by master on 2026-11-17). Alpha (`ALPHA_FEATURES[973]`), behind "Show alpha features".
- **Maintainer request (2026-11-17):** "as a last step in the OPS module, as Brieflow has MozzareLLM, I'd like to do the same thing, which as I understand is just a custom prompt to get LLMs to analyze the clustered data. I'd like to call it voyager3 and it should use the spacr AI path but with a custom prompt that gets the software to pull data from uniprot and assign biological roles to clusters."
- **Builds on:** `spacr/uniprot.py` (the existing UniProt REST fetcher and its JSON cache), `spacr/qt/ai/providers.py` (the user's configured AI provider), the OPS profile path (`spacr.ops_tvn.run_profile_phases`) and the cluster tables of items 916 and 928.
- **Supersedes, for the AI part:** `ITEM_ops_cluster_annotator.md`. The evaluation mode in that spec (held-out GO / CORUM scoring) stays a separate later item.
- **Script it replaces:** `BMCspacr/reanalysis/cluster_annotation/` (session agents with a MozzareLLM-style rubric, 2026-11-08). Voyager3 brings that analysis into spaCR.

## Hard rules
1. **No Claude API.** Voyager3 never imports the anthropic SDK, never reads an API key and never uses a Batch API. Nothing in it names a default vendor.
   - The provider is the one the user chose under Preferences → AI (`spacr.qt.preferences.get_preferred_provider`, resolved by `spacr.qt.ai.providers.get_provider` / `configured_providers`). It can also be passed in as an object with `name` and `stream_chat(messages, system, model)`.
   - The vendor CLIs (Claude Code, Codex, Gemini) use the user's own sign-in.
   - The hosted relay (`spacr_hosted`) is refused with a clear message, as in `spacr.qt.ai.manuscript`. It is an error-diagnosis route: it strips code blocks, sends only the newest question and adds its own preamble.
2. **Tests and the proof use a stub provider only.** A real-provider demo needs the maintainer's OK.
3. **Off by default.** With `ops_voyager3_enabled` off and the `voyager3` phase not named, every OPS output, `runtime.json` included, is byte-identical to before.
4. **The prompt template is English and is not translated.** Labels, tooltips, GUI strings and docstrings get the usual 9-locale records.

## MozzareLLM, what Voyager3 must match or beat
Sources:
- Paper: Nat Commun 2026, 17:6997, doi 10.1038/s41467-026-73643-7. Local PDF and supplement in `reanalysis/brieflow/natcommun_2026/`. Always cite this version, not the preprint.
- Code: github.com/cheeseman-lab/mozzarellm. Tags v0.2.0 (2026-04-07) and v0.3.0 (2026-11-03). Brieflow v1.4.10, the paper's run, used the v0.2.0 `ClusterAnalyzer`.

**What it does** (p.8, Fig 4b on p.9, Methods p.17–18):
- **Input:** one call per cluster. The call carries the cluster's gene list and, for every gene, the UniProt FUNCTION comment only (v0.2.0 `prompt_factory.py` l.24–32, `uniprot_api_client.py` l.226–261). For Brieflow clusters it also names the imaging markers and the cell population (interphase or mitotic) (p.18).
- **Steps:**
  1. Find the dominant pathway.
  2. Class each gene as established, novel role or uncharacterized.
  3. Set the confidence from the share of genes that fit: High >70 %, Medium 50–70 %, Low 30–50 %. Under 30 % means "no coherent pathway".
  4. Score each uncharacterized or novel-role gene 1–10 (p.17).
- **Output JSON** (v0.2.0 `prompts.py` l.322–345): `cluster_id`, `dominant_process`, `pathway_confidence` (High/Medium/Low), `established_genes`, `uncharacterized_genes` and `novel_role_genes`, each a list of `{gene, priority, rationale}`, plus `summary`.
- **Model:** Claude Opus 4.6 at temperature 0 (p.18), chosen over GPT-5.2 and Gemini 3 Pro. One call per cluster, 3 retries. Each cluster's JSON is saved, and a rerun skips clusters already done.

**Published results** (Nat Commun 2026):

| Result | Value | Source |
|---|---|---|
| Interphase high-confidence clusters, Brieflow | 73 / 227 (32.2 %), 1,874 genes | p.8; Supp Fig 7b |
| Interphase high-confidence clusters, Funk et al. 2022 clusters | 52 / 222 (23.4 %), 1,114 genes | p.8; Supp Fig 7b |
| Mitotic high-confidence clusters, Brieflow | 10 / 222 (4.5 %), 731 genes | p.8; Supp Fig 7b |
| Mitotic high-confidence clusters, Funk | 16 (7.2 %), 521 genes | p.8 |
| Shuffled control (cluster sizes kept) | 1 / 256 (0.4 %) high confidence | p.8 |
| Genes per high-confidence Brieflow cluster: established / novel role / uncharacterized | 16.6 / 7.4 / 1.7 | Supp Fig 7c |
| Model benchmark, dominant process correct on 11 ground-truth clusters | Opus 4.6 10/11, GPT-5.2 10/11, Gemini 3 Pro 9/11 | p.18 |
| Novel-role or uncharacterized genes in Brieflow-unique high-confidence clusters (29 of 83 Brieflow clusters unmatched in Funk) | 297 | PDF p.11; Supp Fig 9a, b |
| Agreement with experts on gene classes or priorities | n/a in paper | |
| Run-to-run repeatability; experimental validation of prioritised genes | n/a in paper | |

**Weak points Voyager3 addresses:**

| MozzareLLM (paper version) | Voyager3 |
|---|---|
| Only UniProt FUNCTION text is given. "0–1 publications" is judged from the model's memory. | Function, protein names, GO (process, component, function), pathway, subcellular location, reviewed status and the UniProt annotation score (1–5) are fetched for every gene. "Uncharacterised" is checked against the fetched annotation score. |
| The model reports its own confidence, and the schema returns no counts. | The model returns its supporting genes. spaCR computes the support fraction itself and applies the same 70/50/30 % rule (`confidence_rule`). Disagreement with the model's stated confidence is flagged. |
| Genes the model names are not checked. | Genes that are not in the cluster are removed and counted (`unknown_genes`). |
| No provenance; the closed model is pinned by name only. | A provenance JSON per run: exact prompts, template version, provider and model, UniProt release and fetch date, per-gene facts, raw responses. `voyager3_replay` rebuilds the tables byte for byte. |
| A shuffled control was run by hand. | `ops_voyager3_shuffle_control` annotates a seeded, size-preserving shuffle of the same partition, with the same prompt. |

## Behaviour
1. **Input clusters.**
   - `ops_voyager3_clusters` is a table with a `gene` column and a cluster column, named by `ops_voyager3_cluster_column` (default `cluster`).
   - Empty reads the run's own `results/ops_clusters/consensus_partition.csv` (item 928).
   - Rows whose gene starts with `ops_control_prefix` are dropped. So are rows with `targeting` False or `unassigned` True, when the table has those columns.
   - `ops_voyager3_cluster_ids` restricts the run to some clusters. Clusters with fewer than 2 genes are skipped.
2. **UniProt facts.**
   - Fetched through `spacr.uniprot`: the same request builder, retry/back-off, User-Agent (`spaCR (https://github.com/EinarOlafsson/spacr)`, no personal data) and JSON cache.
   - Voyager3 asks for extra fields: `cc_pathway`, `reviewed`, `annotation_score`, `gene_primary`.
   - Genes are fetched in sorted batches of at most 100. The organism is `ops_voyager3_organism` (default `human`, resolved by `spacr.uniprot.resolve`). The cache is `results/ops_voyager3/uniprot_cache/`.
   - The release is read from UniProt's `X-UniProt-Release` and `X-UniProt-Release-Date` headers and cached beside each batch with the fetch date.
   - **Gene match:** a reviewed entry whose primary gene name equals the symbol; else any entry whose primary name matches; else a reviewed entry listing the symbol as a synonym. Otherwise the gene is "not found".
   - **Text cleaning:** evidence tags (`{ECO:…}`) and `PubMed:` citations are removed. Text is cut deterministically: function 600 characters; at most 8 GO terms per aspect; pathway and location 300 characters each.
3. **Prompt** (template `voyager3-1`, English).
   - A system text, plus one user message per cluster containing:
     - the cluster id and size;
     - cluster statistics where the table has them: mean stability, number of genes;
     - the screen context, `ops_voyager3_context` (cell type, markers, population, as in MozzareLLM);
     - one block per gene, in sorted order, with its fetched facts;
     - the rules (confidence thresholds, use only the facts given plus well-established knowledge, mark uncharacterised genes from the annotation score);
     - the JSON answer schema.
   - **Answer fields:** `cluster_id`, `biological_role`, `confidence` (high/medium/low/none), `supporting_genes` (each with `gene` and `evidence`), `misfit_genes`, `uncharacterized_candidates` (each with `gene`, `suggested_role`, `priority` 1–10 and `rationale`), `summary`.
4. **AI call.**
   - `provider.stream_chat([{"role": "user", "content": prompt}], system=SYSTEM, model=ops_voyager3_model or None)`, collected on a worker thread with a timeout of `ops_voyager3_timeout` seconds (default 300).
   - One retry after an error or an empty answer. Every attempt is recorded.
   - Temperature cannot be set through the vendor CLIs, so repeatability rests on the saved responses and the replay.
5. **Parsing.**
   - The first fenced `json` block is used; failing that, the first balanced `{…}` object.
   - Gene names are matched to the cluster without regard to case. Confidence words are normalised.
   - A malformed answer becomes a row with `status` `unparsed` and the reason, and never stops the run.
   - `support_fraction` = matched supporting genes / cluster genes. `confidence_rule` applies the 70/50/30 % thresholds.
   - `confidence_agrees` is set when the model's stated confidence equals `confidence_rule`.
6. **Outputs** in `<dst_root>/results/ops_voyager3/`:
   - `voyager3_clusters.csv`: one row per cluster.
   - `voyager3_candidates.csv`: one row per uncharacterised candidate, with its annotation score.
   - `voyager3_gene_facts.csv`: the facts each prompt used.
   - `voyager3_report.md`: per cluster, the role, confidence, supporting genes, misfits and candidates, a provenance header and a glossary.
   - `voyager3_provenance.json`.
   - With the shuffle control, the same tables with a `_shuffled` suffix and a summary row.
7. **Replay.** `voyager3_replay(provenance, out_dir)` parses the saved raw responses and writes the tables again with no provider and no network. The run writes its tables through that same code path, so the replay is byte-identical by construction; the proof checks it anyway.

## Settings (OPS, alpha, all off or empty by default)
`ops_voyager3_enabled` (False), `ops_voyager3_clusters` (None), `ops_voyager3_cluster_column` ('cluster'), `ops_voyager3_cluster_ids` (None), `ops_voyager3_organism` ('human'), `ops_voyager3_context` (''), `ops_voyager3_provider` ('' = the Preferences choice), `ops_voyager3_model` ('' = the provider's default), `ops_voyager3_timeout` (300), `ops_voyager3_shuffle_control` (False).

- **Phase:** `voyager3` is the last of the profile phases. It also runs after `cluster_sweep`, `cluster_consensus` or `cluster_bench_test` when `ops_voyager3_enabled` is on.
- **Settings category:** at the end of "OPS phenotype", after the clustering keys. **For item 969:** Voyager3 belongs in its own subcategory, "Cluster annotation (Voyager3)", placed after the clustering subcategory.
- **GUI:**
  - The section holding `ops_voyager3_enabled` gets an "Annotate clusters (Voyager3)" button, a progress bar and a status line.
  - The button runs only the `voyager3` phase on the current settings, through the same worker thread as Run.
  - Progress lines are standard `Progress: n/total, operation_type: Voyager3` lines, so the module bar shows percent and time left too.

## Proof check (pre-set, stub provider only)
1. **Real clusters:** Funk plate A, headline TVN clustering (`funk_plateA_full/benchmarks/cluster_assignments_final_tvn.csv`, column `leiden_r10`), 6 clusters (3 coherent and 3 mixed, according to `cluster_annotation` results).
2. **UniProt:**
   - The live fetch fills the cache.
   - A second run makes 0 requests and gives identical facts.
   - Coverage: genes found, genes with function text, and the release.
   - Control: the function text agrees with `refdata_archive/uniprot_function_2026-11-08.tsv` for at least 95 % of genes found in both.
3. **Prompts:** every prompt holds every cluster gene once, the JSON schema and the template version; no control genes; size within 60,000 characters.
4. **Parsing robustness:** one valid answer and 10 malformed kinds: no JSON, truncated JSON, prose around the JSON, wrong case, unknown genes, a priority out of range, a string priority, a missing field, an empty answer, an exception. Each gives a row and never an exception, and the reasons are recorded.
5. **Replay:** all tables are byte-identical from the provenance, with the network disabled.
6. **Off-switch:** `run_profile_phases` (cluster_sweep on a synthetic profile) with the new keys at their defaults versus nightly: every output and `runtime.json` byte-identical.
7. **Shuffle control:** with the stub's rule (coherence from shared GO terms), real clusters score above the shuffled ones. This only shows that the control works; no LLM is involved.
8. **No Claude API:** the module imports no anthropic SDK and reads no API key. A test asserts this from the source.

## What to build
- `spacr/voyager3.py`: `fetch_gene_facts`, `build_cluster_prompt`, `parse_cluster_answer`, `run_voyager3`, `voyager3_replay`.
- `spacr/uniprot.py`: private extensions only (an optional field list, and the release headers recorded with the cache). Existing callers and cache keys are unchanged.
- `spacr/ops_settings.py` (defaults, types, category, tooltips), `spacr/settings.py` (`ALPHA_FEATURES[973]`), `spacr/ops_engine.py` / `spacr/ops_tvn.py` (phase), `spacr/qt/voyager3_panel.py` and the hook in `app_screen.py`.
- Tests: `tests/test_voyager3_973.py` (stub provider, fake UniProt `_get`, parsing, replay, off-switch, no-API check) and a Qt test of the button.
- Arrival contract, 9-locale records, release note, feature file, readiness row, alpha deck line.
