# Item spec: MoA profiling evaluation (NSC / NSCB) with CellProfiler-Analyst-equivalent classifiers and controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P14).

## Why
The BBBC021 mechanism-of-action (MoA) benchmark (Ljosa et al. 2013; NSC = not-same-compound and NSCB = not-same-compound-or-batch protocols) is computed by custom scorers that do not import spaCR. It covers 1-NN cosine profile matching, supervised classifiers (including re-implementations of the CellProfiler Analyst 3.0.5 classifiers), per-MoA accuracy, and majority / shuffled controls. Two reference scorers exist; `run_and_score.py` is the scorer of record. spaCR needs one scorer, run from a settings file, with 5 seeds.

## Reference implementation
`R/bbbc021/cellprofiler_baseline/`:
- `run_and_score.py` (`normalise`, `knn_eval`, `score`; **scorer of record**);
- `ljosa_methods.py` (Ljosa 2013 profile methods: means, medians, factor-analysis variants);
- `supervised_moa.py` (`loco`: leave-one-compound-out NSC and leave-one-compound-and-batch-out NSCB; `make_model`; models logistic, LDA, shrinkage LDA, KNN, cpa_*);
- `cpa_models.py::make_cpa_model` (CellProfiler Analyst 3.0.5 `generalclassifier` / `fastgentleboosting` from `/mnt/wd4tb/bmc_reanalysis/cpa_src`);
- `cpa_percell.py` (pc50 per-cell protocol), `assemble_cpa.py`;
- `feature_qc.py::standard_qc`;
- run scripts `run_cpa_{all,rest,final,pc50,qc}.sh`;
- `R/bbbc021/spacr/score_endtoend.sh`.

Reference outputs: `R/bbbc021/cellprofiler_baseline/results/cpa_classifiers_summary.md`, `R/bbbc021/cv/results/cv_summary.md`, `R/bbbc021/spacr/results/` (of record). The LDA diagnosis scripts are in `/mnt/firecuda2/codex/scratch-docs/cpa_fix/lda_nscb.py` and `pc_lda.py`; copy these into R first.

## What to build
`spacr.sp_stats.moa_evaluation(profiles, metadata, settings)` plus a Measure-module analysis entry `profiling_evaluation` (alpha). Settings:
- `moa_eval_profiles` (path, well-level table);
- `moa_eval_label` ('moa');
- `moa_eval_compound` ('compound');
- `moa_eval_batch` ('week');
- `moa_eval_protocols` ('NSC,NSCB');
- `moa_eval_methods`: '1nn_cosine, logistic, lda, shrinkage_lda, knn, cpa_gentleboost, cpa_rf …', with CPA classifiers as an optional dependency and lazy import. The method names in `supervised_moa.make_model` are authoritative.
- `moa_eval_feature_qc` ('standard', item 807);
- `moa_eval_normalise` (as `run_and_score.normalise`);
- `moa_eval_controls` ('majority, shuffled');
- `moa_eval_seeds` ('0,1,2,3,4');
- `moa_eval_per_moa` (bool; writes the per-MoA table).

Outputs in `results/profiling_evaluation/`:
- `summary.csv` (method × protocol × seed accuracy);
- `per_moa.csv`;
- `confusion_<method>.csv`;
- `controls.csv`.

## Validation data and acceptance
- **Data:** the of-record well-level profiles. spaCR profiles: the inputs of `R/bbbc021/spacr/results` (the parquet aggregate). CellProfiler feature tables: `/nas_mnt/data/bmc_reanalysis/bbbc021_archive/tables`.
- **Acceptance:** identical output to the reference outputs at the three result paths above: with seed 0, every accuracy in `cpa_classifiers_summary.md` and `cv_summary.md` is reproduced exactly (as a fraction of the treatments of each protocol), including the controls; the "means" row follows `run_and_score.py`.
- **Tests:** leave-one-compound-out folds on a toy set; majority and shuffled controls.

## Effort
- **Effort:** L (the CPA classifier port is most of it).
