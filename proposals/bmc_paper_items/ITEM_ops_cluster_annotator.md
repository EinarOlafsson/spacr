# Item spec: spaCR AI cluster annotator for OPS gene clusters

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Builds on:** the spaCR AI layer in `spacr/qt/ai/` (`providers.py`, `worker.py`, `prompts.py`, `settings.py`) and `spacr/qt/widgets/ai_chat_panel.py`. Related: `ITEM_ops_cluster_viewer.md`, `ITEM_ops_reference_sets.md`.
- **Rule:** the model provider is injected through the existing provider settings. There is no default provider and no hard-coded vendor API or key in code. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (OPS gene-cluster interpretation; phase 5 of the OPS story).

## Why
OPS screens produce hundreds of gene clusters. Reading each one against the literature is slow. An assistant that summarises what a cluster's members share, how confident that summary is, and which members are poorly characterised turns a cluster map into testable hypotheses.

## Behaviour
- **Input per cluster:**
  - member genes;
  - the cluster's top phenotype features versus non-targeting controls (effect sizes);
  - stability (Jaccard over guide resamples);
  - local reference annotations: GO terms, CORUM complexes, STRING edges inside the cluster, UniProt function lines, read from the bundled reference sets and never fetched during scoring.
- **Output per cluster** (JSON plus a table):
  - proposed shared function;
  - confidence (high / medium / low), with the reason;
  - the supporting members and the reference lines that support them;
  - poorly characterised members as candidates;
  - a phenotype-based rationale linking the features to the function;
  - a "no coherent function" verdict when warranted.
- **Grounding:** each claim cites the reference record it rests on. Claims without a citation are labelled speculative. The prompt and the reference snapshot version are saved with each result.
- **Reproducibility:** the settings file records the provider, model name, temperature 0, prompt version and reference snapshot. A rerun with cached responses reproduces the outputs exactly; a fresh call is flagged as a new run.
- **Evaluation mode (for the paper):** blind the cluster labels, hold out the annotations for a gene subset, and score the annotator's function calls against held-out GO / CORUM. Report the same score for a shuffled-membership control and for a simple enrichment baseline (top GO term by hypergeometric test).

## What to build
- `spacr.ops_annotate`: `annotate_clusters(clusters, profiles, references, settings)` → table + JSON.
- GUI: an "Annotate clusters" action in the cluster viewer that writes into the cluster side panel.
- Tests: a fake provider with fixed responses; citation enforcement; cache-replay identity; shuffled control.
