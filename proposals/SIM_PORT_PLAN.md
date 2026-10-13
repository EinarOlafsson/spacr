# Porting spaCRPower into spaCR's `sim` module — implementation plan

Design pass only. Nothing under `/mnt/firecuda2/Claude/repo/spacr` was modified.

Sources read in full:
`spaCRPower/R/{simulate_screen,fit_model,scan_parameters,plot_*}.R`, `DESCRIPTION`,
`NAMESPACE`, `README.md`, all 14 `vignettes/Tgondii_screen/*.Rmd`;
`spacr/sim.py` (1519 lines), `spacr/sequencing.py`, `spacr/ml.py` (regression path),
`spacr/settings.py`, `spacr/gui_core.py`, `spacr/gui_utils.py`, `spacr/gui.py`,
`spacr/qt/app.py`, `spacr/qt/screens/settings_model.py`, `spacr/qt/bridge.py`,
`spacr/qt/synthetic.py`, `spacr/validate.py`, `spacr/cli.py`, `setup.py`,
`tests/test_{sim,cov_sim_engine,coverage_fill_sim*,app_entry_consistency,settings_categories}.py`.

---

## 0. Decisions at a glance

| Question | Decision |
|---|---|
| Port faithfully? | Port the **model**, not the code. Five of the R functions are broken on their own simulated data (§3). Reproduce the generative process; fix the analysis. |
| Bayesian backend | **No new hard dependency.** Ship four estimators built on numpy/scipy/statsmodels (already dependencies), the closest being a regularized-horseshoe Poisson GLM fitted by EM + Laplace. `pip install spacr[stan]` optionally runs the literal Stan model via cmdstanpy for cross-validation against R. PyMC / NumPyro / TF all rejected, with reasons (§6). |
| Default estimator | `spacr_ols` — spaCR's *own* regression formula. A power analysis should measure the pipeline the user will actually run. |
| Sequencing error | Eight-layer model, each independently switchable, emitting **real FASTQ that `spacr.sequencing.generate_barecode_mapping` can read** (§7). Barcode collision and chimeras are the two that corrupt results; index hopping is the one the old `sim.py` was groping at. |
| Additions | Accept 6 (guides-within-gene, controls, plate/row/column/edge effects, classifier drift, fitness/dropout, multi-integration). Reject 4 with reasons (§8). |
| Module layout | `spacr/sim.py` → package `spacr/sim/` with 7 modules + `legacy.py` (§4). |
| GUI | New `SimScreen` at `spacr/qt/screens/sim.py`, app key `sim`, section **Data & batch runs** (6→7 of 9, no overflow). Fix the Tk panel too — it is 3 lines once the settings factory exists (§9). |
| Old API | Deprecated, not deleted. `run_multiple_simulations` keeps working for one release with a `DeprecationWarning` and a shim onto the new sweep (§11). |
| Test coverage for sim.py | Correctly deferred. Do not chase the current 45.3% — most of the uncovered statements are in code this plan deletes. |

---

## 1. What spaCRPower is actually for

The vignette is worth more than the API. The real screen it was built against:

- **452 genes**, ~4–5 gRNAs each, pooled and spotted into **1536 wells** (4 × 384-well plates).
- Plate layout: column 1 = **NC**, column 2 = **PC**, column 3 = **Sweep**, rest = **Screen**.
- A gene counts as "in a well" when its read fraction ≥ **2%** (`gene_in_well_threshold`) —
  everything below that is treated as noise. *Nobody has ever justified that 2%.* One of the
  strongest reasons to simulate sequencing error is that it lets you derive it (§7.9).
- ~**123 cells imaged per well** (mean), very over-dispersed (var ≈ 8000).
- A MaxViT classifier calls each cell positive/negative; fitted operating point
  **class_pos ≈ 0.80, class_neg ≈ 0.12** — i.e. a *bad* classifier, and that is the point:
  the question is how many wells and cells buy you a hit call at that accuracy.
- Estimated hit rate ≈ **2.5%** (~11 hit genes), inferred by inverting the well positivity rate.
- Total reads ≈ 128 318 (the parameter is used inconsistently — see §3.4).
- The answer the package produces: **AUROC of hit identification vs. number of wells**, swept
  over genes-per-well, classifier accuracy, library skew and cells-per-well.

Everything in this plan is in service of that plot, plus the two things it can't currently
answer: *what does sequencing error cost me*, and *what does it cost me in **spaCR's** analysis
rather than in a bespoke brms model*.

---

## 2. Function-by-function mapping

### 2.1 Distribution helpers

| R | Python | Trap |
|---|---|---|
| `rgamma_mean_variance(n, mean, var)` — `rgamma(shape=m²/v, rate=m/v)` | `rng.gamma(shape=m**2/v, scale=v/m, size=n)` | **R takes `rate`, numpy takes `scale = 1/rate`.** The single most common R→numpy port bug. |
| `rnbinom_mean_variance(n, mean, var)` — `rnbinom(size=m²/(v−m), prob=m/v)` | `rng.negative_binomial(n=m**2/(v-m), p=m/v, size=n)` | Conventions agree (both mean `n(1−p)/p`), but assert `v > m` first: at `v == m` the size is `inf`. |
| `rbeta_mean_variance` | `rng.beta(a, b, size=n)` with the same moment inversion | Assert `0 < v < m(1−m)`; R's `assertthat` message is right. Note the existing `spacr.sim.classifier._calc_alpha_beta` already does exactly this — reuse it. |
| `anscombe_mean` | unused, dead in R | Drop. |
| *(not in R)* | `rdirichlet_stable(alpha, n)` | `rng.dirichlet(np.full(n, 0.6))` is used at α = 0.6 with n = 452. numpy draws Gammas and normalises; at small α individual draws underflow to exactly 0.0, giving a gene abundance of exactly zero and a downstream 0/0. Draw in log space (`log g = log(rng.gamma(a+1)) + log(u)/a`), subtract the log-sum-exp, exponentiate. |

**COM-Poisson has no maintained Python equivalent** in spaCR's dependency set
(`COMPoissonReg::rcmp` is used for `sequencing_n_cells_per_well_nu` — and note it is *not even
declared in spaCRPower's DESCRIPTION*, so the R package itself fails to install-and-run that
branch). Next best in Python, and honestly better because it is closed-form:

```
count_model(mean m, var v):
    v > m  ->  NegativeBinomial(n=m²/(v−m), p=m/v)      # over-dispersed  (nu < 1)
    v ≈ m  ->  Poisson(m)                                # nu == 1
    v < m  ->  Binomial(n=round(m²/(m−v)), p=(m−v)/m)    # under-dispersed (nu > 1)
```

This spans exactly the range the vignette sweeps (`nu` from 0.9 to 20, i.e. mostly
under-dispersion) with the right first two moments and no new dependency. Document that the
third+ moments differ from COM-Poisson; the vignette only ever fits mean and sd, so nothing is
lost that was being used.

### 2.2 The simulation chain

| R function | Python | Notes / traps |
|---|---|---|
| `simulate_library(n_genes_in_library, gene_abundance_alpha, gene_hit_rate)` | `simulate_library(...) -> Library` | `abundance ~ Dirichlet(α·1_n)`, `hit ~ Bernoulli(p)`. 1-based `gene` index in R; emit **both** `gene_index` (0-based int) and `gene` (string label `TGGT1_<id>_<guide>` compatible with `spacr.ml.process_reads`, §7.10). Add guides (§8.1). |
| `simulate_spot_plate(gene_library, n_wells, well_abundance_factor_mu, var)` | `simulate_spot_plate(...) -> SpotPlate` | `gene_in_well ~ Bernoulli(gene_abundance_i × well_abundance_j)`. **`prob` can exceed 1**: at α = 0.6 the top gene's abundance × `well_abundance_factor_mu = 4.6` routinely exceeds 1. R's `rbinom` silently returns `NA`; numpy raises `ValueError`. Clip to `[0,1]`, **count the clips and warn once with the count** — a clipped run means the requested `well_abundance_factor_mu` is not achievable and the realised genes-per-well is below target. |
| | | `tidyr::expand_grid(gene, well)` varies the **last** variable fastest (unlike base R `expand.grid`). `pd.MultiIndex.from_product([genes, wells])` matches; `np.meshgrid` with default `indexing='xy'` does **not**. |
| `simulate_imaging_plate(spot_plate, n_cells_mu, n_cells_var, class_pos_mu/var, class_neg_mu/var)` | `simulate_imaging_plate(...) -> ImagingPlate` | Per well: draw the well's total cells from `count_model(mu, var)`, then `rmultinom` it across the genes present. **R's `prob = gene_in_well` is a 0/1 vector, so cells are split *uniformly* over the genes in the well** — the gene's own abundance is ignored at this step. That is a modelling choice, not a bug, but it is wrong: a gene that is 20% of the well's reads should not get the same number of cells as one at 3%. Make it a parameter: `sim_imaging_split ∈ {'uniform','abundance'}`, default `'abundance'`, with `'uniform'` for R parity. |
| | | `np.random.multinomial` requires `sum(pvals[:-1]) <= 1` and does not renormalise the way R's `rmultinom` does. Normalise explicitly; guard the all-zero well (R does, with the `sum(gene_in_well)==0` branch). |
| | | `positive ~ Binomial(n_cells, ifelse(hit, Beta(pos), Beta(neg)))` — the classifier probability is drawn **i.i.d. per (gene, well) row**. Keep, but add correlated drift (§8.4). |
| `simulate_sequencing_plate(spot_plate, lambda, nu, pcr_factor_mu, pcr_factor_var, n_reads_total)` | `simulate_sequencing_plate(...) -> SequencingPlate` | Cells contributing DNA: `gene_in_well × count_model(lambda, ...)`. Well-level PCR factor `rlnorm(meanlog, sdlog=sqrt(var))` → `rng.lognormal(mean, sigma=sqrt(var))` (conventions agree; the vignette's comment "mean is exp(mu + 1/var)" is a typo for `exp(mu + var/2)`). Reads drawn **without replacement** from the amplified barcode pool: `extraDistr::rmvhyper` → `rng.multivariate_hypergeometric(colors, nsample, method='marginals')`. Guard `nsample <= colors.sum()` — R's `min(...)` guard is defeated by per-element `round()` (§3.4). |
| `simulate_screen(...)` | `simulate_screen(...) -> Screen` | Composition + three left joins on `(well, gene)`. In pandas use explicit `merge(on=['well','gene'], validate='1:1')` — never `concat`/positional, because `dplyr::do()` returns groups in **sorted key order**, not input order, and any positional assumption downstream is wrong. |
| `scan_parameters(...)` / `scan_parameters_future(...)` | `scan_parameters(...)` (one function, `backend='process'|'serial'|'slurm'`) | `tidyr::expand_grid` over 17 parameters → `itertools.product` over the swept keys. The R `future` variant is a SLURM submission wrapper; in Python use `concurrent.futures.ProcessPoolExecutor` and emit a **`sbatch` array script** for cluster users rather than binding to a scheduler library. Note the R code's `model <<- compile_model(...)` inside `future::future({...})` never propagates back to the parent, so every remote job recompiles Stan — one more reason not to need Stan. |

### 2.3 The model / evaluation chain

| R function | Python | Notes |
|---|---|---|
| `prepare_model_data(data)` | `build_design(screen) -> DesignMatrix` | See §3.1–§3.3 — the R version is broken for simulated input and only works on the real-data frame assembled in `04.1_analyze_screen.Rmd`. Rewrite: `X[w, g] = log10(read_fraction[w,g] + pseudo)`, `y[w] = Σ_g positive`, `offset[w] = log(Σ_g cells)`. Build `X` from an explicit `(well_index, gene_index)` pivot with a fixed gene ordering — the R matrix-column construction is positional and misaligns the moment a well is missing a gene. |
| `compile_model` + `fit_model` (brms/cmdstanr) | `spacr.sim.estimate.fit(design, method=...)` | §6. |
| `gather_model_estimate(model_fit)` | returned as a DataFrame `[gene, estimate, lower, upper, se]` | R strips the `b_log10expression` prefix and returns `gene` as **character**; `evaluate_model_fit` then joins it to an **integer** `gene` column, which `dplyr` refuses. Use string gene labels end to end. |
| `evaluate_model_fit(data, model_estimate)` | `evaluate(estimates, truth) -> dict` | §3.5 — the R AP is computed for the wrong class. Use `sklearn.metrics.roc_auc_score` and `average_precision_score` with the hit class as positive, and also report `precision@k`, `recall@10%FPR`, and the **rank of the known positive control**, which is what a screener actually looks at. |
| 8 × `plot_*.R` | `spacr.sim.plots` (matplotlib) | 1:1, listed in §5.4. `plot_well_gene_reads_heatmap` in R builds a `pivot_wider` and then **throws it away** and plots the long frame — port the intent, drop the dead pivot. `plot_cells_per_well` reads a column `n_cells_per_gene_per_well` that no simulator emits (it is `imaging_n_cells_per_gene_per_well`) — port with the right name. |

---

## 3. Bugs in spaCRPower — do NOT port these

These are stated plainly because the plan is to *improve on* the R package, and because an
implementer translating line-by-line would faithfully reproduce every one.

**3.1 `prepare_model_data`: `Npositive = sum(well_data$positive[1])`.**
`sum(x[1])` is `x[1]`. This takes the positive-cell count of the **first gene row** in the well,
not the well total. It is correct for the real-data frame in `04.1` (where `positive` is a
per-well constant broadcast across gene rows) and wrong for `simulate_screen` output (where
`positive` is per gene per well). Correct: `y[w] = Σ_g positive[w,g]`.

**3.2 `prepare_model_data`: `Ntotal = well_data$imaging_n_cells_per_well[1]`.**
No such column exists in `simulate_screen` output. The three columns that start with that string
(`..._mu`, `..._var`, `..._gene_per_well`) make `$` partial matching ambiguous, so R returns
`NULL`, `tibble()` drops the column, and `filter(Ntotal > 0)` dies with *object 'Ntotal' not
found*. **The simulate → fit path in `03.1_simulate_screen.Rmd` does not run as committed.**
Correct: `offset[w] = log(Σ_g imaging_n_cells_per_gene_per_well[w,g])`.

**3.3 The matrix predictor is positionally aligned.**
`log10expression` is built per well as a 1×n_gene matrix inside `dplyr::do()` and row-bound. The
column *identity* is positional. Any well whose gene set differs in order or length silently
shifts every gene's coefficient. Build an explicit pivot on a canonical gene index.

**3.4 `n_reads_per_well <- round(n_reads_total / nrow(well_data))` divides by the number of
GENES, not wells.** With 452 genes and `n_reads_total = 128318` that is 284 reads per well —
against a real screen with a geometric mean near 3×10⁴. The parameter is also documented
inconsistently: `03.1` calls it "total reads in the screen", `03.3.1` calls it "geometric mean of
well reads". Replace with two explicit, non-overlapping parameters: `sim_reads_per_well`
(what people actually know) and `sim_read_depth_cv` (well-to-well depth variation), and derive
the total.
Related: `k = min(n_cells_in_well * pcr_factor[1], n_reads_per_well)` guards `rmvhyper`'s
`k ≤ sum(n)`, but `n` is `round(cells × pcr)` **element-wise**, whose sum can be less than
`round(Σcells × pcr)`. Compute `k = min(colors.sum(), reads)` from the actual colour vector.

**3.5 `evaluate_model_fit` scores the wrong class.**
`hit` is factored with `levels = c(0,1), labels = c("no","yes")`, and yardstick's default is
`event_level = "first"`. The event is therefore **"no"**, and the score is `mean_inv = -mean`.
AUROC is invariant under simultaneously flipping label and score, so `model_auroc` is fine — but
**average precision is not**, so `model_ap` is the average precision for detecting *non-hits*
against a 97.5% base rate. Every AP in the R package's sweeps is a near-constant ≈ 0.98 that
means nothing. Do not port; compute AP for the hit class.

**3.6 `COMPoissonReg` is called but not declared** in `DESCRIPTION`. Moot once §2.1 replaces it.

**3.7 The vignettes call a signature that no longer exists.**
`02.3` calls `simulate_imaging_plate(imaging_n_cells_per_well_lambda=, ...nu=)` against a function
whose parameters are `..._mu` / `..._var`. The R package is mid-refactor and internally
inconsistent. **Port `R/`, not `vignettes/`** — but mine the vignettes for the fitted parameter
values, which are the real contribution (§10 defaults).

**3.8 The `+ 0.0001` pseudocount in `log10(fraction + .0001)` is depth-blind.**
It pins absent genes at exactly −4 regardless of sequencing depth. Since the sweep varies depth,
the covariate is **not comparable across sweep points**, which quietly corrupts any conclusion
about how many reads you need. Use a half-read pseudocount, `pseudo = 0.5 / reads_in_well`, so
"absent" means "below the detection limit at this depth" at every depth. Keep `sim_pseudocount =
'half_read' | <float>` so the R behaviour is reproducible for comparison.

---

## 4. Module layout

`spacr/sim.py` is 1519 lines / 746 statements and this work roughly doubles it. Split into a
package. `import spacr.sim` keeps working.

```
spacr/sim/
  __init__.py      # public API re-exports + deprecation shims        (~120 lines)
  model.py         # dataclasses: Library, SpotPlate, ImagingPlate,
                   #   SequencingPlate, Screen, ScanResult; RNG stream mgmt (~250)
  screen.py        # simulate_library / _spot_plate / _imaging_plate /
                   #   _sequencing_plate / simulate_screen              (~450)
  seqerror.py      # the 8-layer error model + FASTQ/barcode-CSV writer (~500)
  estimate.py      # 4 estimators + evaluate()                          (~450)
  scan.py          # scan_parameters, progress file, parallel backends  (~300)
  plots.py         # the 8 R plots + parameter-scan plot                (~400)
  io.py            # write spaCR-consumable score_data / count_data /
                   #   FASTQ / settings CSVs                            (~200)
  legacy.py        # today's sim.py, verbatim, deprecated               (1519)
```

Why a package and not one file: `seqerror.py` and `estimate.py` have essentially no shared state
with `screen.py`, are the two parts most likely to grow, and are the two parts most worth testing
in isolation. Also, `legacy.py` needs to be a separate module so it can be deleted in one commit.

**Import-cost note.** `spacr/sim.py` currently imports `shap`, `sklearn.ensemble`,
`sklearn.inspection`, `seaborn` and `statsmodels` at module top level. `spacr/__init__.py` lists
`"sim"`, so every `import spacr` pays for `shap` today. Move all of those to function-local
imports in `legacy.py`/`plots.py`. This is a free ~1–2 s startup win for the whole package and
should be in the first commit.

---

## 5. Public API

### 5.1 Simulation

```python
@dataclass(frozen=True)
class ScreenSpec:
    # library
    n_genes: int = 452
    n_grnas_per_gene: int = 4
    gene_abundance_alpha: float = 0.6
    gene_hit_rate: float = 0.025
    grna_efficiency_mean: float = 0.7
    grna_efficiency_sd: float = 0.25
    n_control_genes: int = 10          # non-targeting
    pc_gene: str | None = "TGGT1_220950"
    nc_gene: str | None = "TGGT1_233460"
    # plate layout
    n_plates: int = 4
    plate_format: int = 384            # 96 | 384 | 1536
    control_columns: dict = ...        # {'NC': 1, 'PC': 2, 'Sweep': 3}
    well_abundance_mu: float = 4.6
    well_abundance_var: float = 1.0
    plate_effect_sd: float = 0.0
    row_effect_sd: float = 0.0
    column_effect_sd: float = 0.0
    edge_effect: float = 1.0           # multiplier on the outer ring
    # imaging
    cells_per_well_mu: float = 123.0
    cells_per_well_var: float = 8000.0
    imaging_split: str = "abundance"   # 'abundance' | 'uniform'
    well_qc_fail_rate: float = 0.0
    class_pos_mu: float = 0.80
    class_pos_var: float = 0.10
    class_neg_mu: float = 0.12
    class_neg_var: float = 0.01
    class_drift_plate_sd: float = 0.0  # logit-scale
    class_drift_well_sd: float = 0.0
    hit_fitness_lfc: float = 0.0       # log2 change in cell count for hits
    multi_integration_rate: float = 0.0
    # sequencing
    cells_per_well_seq_lambda: float = 1000.0
    cells_per_well_seq_var: float | None = None   # None -> Poisson
    pcr_factor_mu: float = 2.0
    pcr_factor_var: float = 1.0
    pcr_jackpot_sd: float = 0.0        # per-(gene,well) lognormal sdlog
    reads_per_well: int = 30_000
    read_depth_cv: float = 0.35
    seq_error: "SeqErrorSpec | None" = None

def simulate_screen(spec: ScreenSpec, *, seed: int | None = None) -> Screen
def simulate_library(spec, rng) -> Library
def simulate_spot_plate(library, spec, rng) -> SpotPlate
def simulate_imaging_plate(spot, spec, rng) -> ImagingPlate
def simulate_sequencing_plate(spot, spec, rng) -> SequencingPlate
```

`Screen` carries `.library`, `.spot`, `.imaging`, `.sequencing`, `.spec`, `.seed`, and
`.truth` (a `[gene, hit, n_wells, n_cells]` frame), plus:

```python
Screen.counts()      -> DataFrame   # count_data format: plateID,rowID,columnID,grna,count
Screen.scores()      -> DataFrame   # score_data format: plateID,rowID,columnID,objectID,pred
Screen.wells()       -> DataFrame   # per-well summary
Screen.write(dst)    -> Paths       # both CSVs + settings CSV, ready for perform_regression
Screen.write_fastq(dst, ...) -> Paths  # FASTQ.gz + row/column/grna barcode CSVs
```

### 5.2 The with/without-error comparison — the point of §7

RNG streams are split per stage via `np.random.SeedSequence(seed).spawn(8)`, so the biology
stream is unaffected by whether the error model runs and how many draws it consumes. That makes
this exact and not approximate:

```python
truth = simulate_screen(spec, seed=0)                       # seq_error=None
noisy = truth.resequence(SeqErrorSpec.illumina_novaseq())   # same library, plate, cells
cost  = compare_screens(truth, noisy)                       # -> DataFrame
```

`compare_screens` returns, per estimator: ΔAUROC, ΔAP, Δrank of the positive control, the
fraction of reads lost, the fraction of reads **misassigned** (the number that matters), and the
Spearman correlation of per-gene estimates. `SeqErrorSpec.ablate()` yields one spec per layer
turned on alone, so the GUI can show a waterfall of *which* error costs what.

### 5.3 Estimation

```python
def build_design(screen_or_frames, *, pseudocount='half_read') -> Design
def fit(design, method='spacr_ols', **kw) -> Estimates   # [gene, estimate, se, lower, upper]
def evaluate(estimates, truth) -> dict
ESTIMATORS = ('spacr_ols', 'hs_em', 'glm_l1', 'ridge_laplace', 'stan')
```

### 5.4 Sweep and plots

```python
def scan_parameters(base: ScreenSpec, grid: dict[str, Sequence],
                    *, replicates=3, method='spacr_ols', seed=0,
                    progress_file=None, backend='process', max_workers=None,
                    verbose=False) -> ScanResult
def emit_slurm_array(scan_spec, dst) -> Path

# plots.py — every one returns a Figure, none calls plt.show()
plot_genes_per_well, plot_wells_per_gene, plot_cells_per_well,
plot_positivity_rate_by_well, plot_well_gene_reads_heatmap,
plot_classification_scores, plot_model_estimate, plot_parameter_scan,
plot_error_waterfall, plot_plate_map
```

`plot_*` returning a bare `Figure` and never calling `plt.show()` is a deliberate departure from
today's `sim.py`, which calls `plt.show()` inside library functions — that hangs a headless run
and is the reason `visualize_all` cannot be used from the GUI.

---

## 6. The Bayesian fitting problem

### 6.1 What the model actually is

Not a beta-binomial. From `fit_model.R`:

```
Npositive_w ~ Poisson( Ntotal_w · exp( β₀ + Σ_g β_g · log10expression_{w,g} ) )
β_g ~ horseshoe(df = 10)                       # brms's regularized horseshoe
```

A sparse high-dimensional Poisson GLM with a log link and an exposure offset:
n ≈ 1536 observations, p ≈ 453 parameters, expected non-zeros ≈ 11. The horseshoe is doing
variable selection, and that is the only part of "Bayesian" that is load-bearing here — the
downstream use is `roc_auc(hit, β_g)`, i.e. a **ranking**, not a calibrated interval.

### 6.2 The options, honestly

| Option | Verdict |
|---|---|
| **cmdstanpy / bridgestan** | Runs the same Stan model — the only route to numerically comparable results. But it needs a working C++ toolchain at *install* time. spaCR's users are wet-lab collaborators on managed laptops; "install a C++ compiler" is where they stop. **Optional extra, not a dependency.** |
| **PyMC** | Mature, pip-installable, pure Python, would express the horseshoe directly. Rejected as a *hard* dependency on two grounds. (a) Weight: pytensor + its C backend is ~250 MB and compiles at first use — spaCR's install is already heavy. (b) Speed, which is the real argument: NUTS on 1536×453 with a horseshoe is minutes per fit, and a sweep is 100 grid points × 3 replicates. **spaCRPower needed a SLURM cluster to run its own sweeps** (`future.batchtools`, `batchtools.greatlakes.tmpl`). Reproducing that in Python reproduces the mistake. A power analysis you can only run on a cluster is one nobody runs. |
| **NumPyro** | Fastest sampler of the three, but it pulls JAX. JAX preallocates ~75% of VRAM on import by default, in a package that ships torch and runs Cellpose in the same process. That is a support burden ("Cellpose OOMs after I open the Simulation tab") for a module that doesn't need a GPU at all. **Rejected.** |
| **TensorFlow Probability** | Banned outright. Not considered. |
| **Direct implementation** | **Chosen.** |

### 6.3 What ships

Four estimators, all on existing dependencies, selected by `sim_estimator`:

**1. `spacr_ols` (default).** Reproduces spaCR's own regression, in-process, without the file and
plot machinery of `perform_regression`:
`score ~ fraction:grna + gene_fraction:gene + rowID + columnID` via `statsmodels.formula.api.ols`
(and `mixedlm` when `sim_random_row_column=True`), reusing `spacr.ml.prepare_formula` so the two
cannot drift. Per-gene score = the `gene_fraction:gene` coefficient (t-statistic for ranking).
*Why the default:* the number a user wants is "how many wells before **my** pipeline finds the
hit", and that is this model, not brms's. It is also the fastest (< 1 s).

**2. `hs_em`.** The spaCRPower-comparable one. Regularized horseshoe (Piironen & Vehtari 2017)
Poisson GLM fitted by EM over the local scales, then a Laplace approximation at the mode:

- Scale-mixture representation (Makalic & Schmidt 2016): `λ_g² | ν_g ~ IG(½, 1/ν_g)`,
  `ν_g ~ IG(½, 1)`; likewise for τ. Every conditional is inverse-gamma, so the E-step is closed
  form: `E[1/λ_g²] = (1/ν_g + β_g²/(2τ²))⁻¹`-style updates, no sampling.
- M-step: penalised IRLS for the Poisson GLM, per-coefficient ridge weights `1/(τ² λ̃_g²)`,
  `λ̃_g² = c²λ_g² / (c² + τ²λ_g²)`.
- `τ₀ = (p₀/(p−p₀)) / sqrt(n_wells · mean(y))` with `p₀ = n_genes × gene_hit_rate`.
- Laplace: `Σ ≈ (XᵀWX + D)⁻¹`, `W = diag(μ)`, `D = diag(1/(τ²λ̃²))`;
  `lower/upper = β ∓ 1.645·sqrt(diag(Σ))`.

XᵀWX is 453×453 — a Cholesky per iteration, ~50 iterations, well under a second. ~120 lines of
numpy/scipy. **Note for the implementer: a plain MAP of a horseshoe is not a thing.** The
horseshoe density has a pole at zero, so the naive posterior mode is β = 0 for every coefficient.
The EM-over-scales formulation is what makes a mode-based fit well defined; do not "simplify" it
into `scipy.optimize.minimize` on the horseshoe log-density.

**3. `glm_l1`.** `statsmodels.GLM(family=Poisson, offset=...).fit_regularized(L1_wt=1.0)`, λ by
BIC over a small path. Fast, familiar, no intervals. Included because it is the estimator
reviewers will ask about.

**4. `ridge_laplace`.** L2-penalised Poisson GLM + Laplace. The dense-effects null model — useful
precisely because comparing it against `hs_em` shows how much the sparsity assumption is buying.

**5. `stan` (optional).** `pip install spacr[stan]` → `cmdstanpy`. Ships the Stan file
(`spacr/sim/resources/screen_poisson_hs.stan`) transcribed from the brms-generated code so
results are directly comparable with the R package. Raises a clear `ImportError` naming the
extra if cmdstanpy is absent. Used by one opt-in test (§12.6).

### 6.4 What is lost — stated plainly

- **Credible intervals become approximate.** `hs_em` / `ridge_laplace` report a *Laplace* interval:
  a normal approximation at the posterior mode. For genes with a handful of wells the true
  posterior is skewed and this interval is too narrow. Column names are `lower`/`upper`, and the
  docstring and the GUI legend must both say **"Laplace approximation, not MCMC"**. `glm_l1` and
  `spacr_ols` report a confidence interval, not a credible one — different object, same columns.
- **No convergence diagnostics.** No R̂, no ESS, no divergent transitions. In exchange there is
  nothing to diverge. `hs_em` reports its own convergence (iterations, final relative change) and
  refuses to return silently on non-convergence.
- **Partial pooling survives, sampling does not.** The global-local shrinkage — the actually
  useful part — is preserved exactly. What is lost is the full posterior *shape*.
- **`hs_em` will not match brms to the third decimal.** It matches in rank ordering and shrinkage
  behaviour, which is what AUROC/AP consume. §12.6 pins the agreement empirically against the
  `stan` backend and records the observed tolerance rather than asserting one a priori.
- Where an interval is load-bearing (a single-gene claim in a paper), the docstring points at
  `method='stan'`.

**Net dependency change: zero.** One optional extra added to `setup.py`, following the existing
`trackastra`/`ultrack` pattern, comment included ("no TensorFlow").

---

## 7. The sequencing-error model

spaCRPower has none. Today's `sim.py` has `sequencing_error=0.01`, which moves a whole gene's read
block to a uniformly random well — that is neither a substitution nor index hopping, it is nothing.
Replace it.

Everything below is anchored on what `spacr/sequencing.py` actually does, so the output is
consumable by the real mapper: reads are anchored on `target_sequence`, a window of
`expected_end` bases starting at `pos + offset_start` is cut, R2 is reverse-complemented and
consensus-called base-by-base against R1, and the window is split by a named-group regex into
`columnID` / `grna` / `rowID`, each looked up in a `sequence,name` CSV.

### 7.1 The synthetic amplicon

Exactly matching the shipped default regex
`^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<rowID>.{8}).*`
with `target_sequence='TGCTGTTTCCAGCATAGCTCTTAAAC'`, `offset_start=-8`, `expected_end=89`:

```
offset  0    8                          34                55    60              76    81   89
        |----|--------------------------|-----------------|-----|---------------|-----|----|
        col8  TGCTGTTTCCAGCATAGCTCTTAAAC  gRNA barcode(21)  AACTT  spacer(16)      AGAAG row8
```
8 + 26 + 21 + 5 + 16 + 5 + 8 = 89. The spacer is a fixed 16-mer chosen to contain **no `AGAAG`**
(the trailing `.*AGAAG` is greedy and would otherwise steal the row barcode). R1 = a 0–61 nt
random 5′ stagger + this window + random 3′ filler to 150 nt; R2 = reverse complement of R1 with
independently drawn errors. Headers follow `spacr/qt/synthetic.py`'s Illumina-1.8 format —
**reuse `_fastq_header` and `_phred_run` from there rather than duplicating them**, and while
you are in that file note that `generate_map_barcodes_demo` writes barcodes as **FASTA** while
`map_sequences_to_names` reads a **CSV** with `sequence,name`; the demo cannot round-trip today.
Fixing that is a two-line change and belongs in this work.

### 7.2 The eight layers

Applied in this order, each with its own `enabled` flag and its own RNG stream so that ablation is
exact:

| # | Layer | Model | Default | Matters here? |
|---|---|---|---|---|
| 1 | **Library representation** | Dirichlet(α) on gene abundance | α = 0.6 | Already in R. Keep. |
| 2 | **PCR bias & jackpotting** | per-(gene, well) lognormal multiplier, `sdlog = pcr_jackpot_sd`; optional `'branching'` Galton–Watson over `pcr_cycles` with efficiency `p_eff` | `sdlog = 0.4`, `cycles = 22`, `p_eff = 0.85` | **Yes.** R's factor is per *well* only, which just rescales depth and cancels in the fraction. Per-species variance is what actually distorts the read fraction, i.e. the covariate the model regresses on. |
| 3 | **Depth & read sampling** | multivariate hypergeometric over the amplified pool; per-well depth `lognormal(log(reads_per_well), read_depth_cv)` | 30 000 / CV 0.35 | Yes — fixes §3.4. |
| 4 | **Per-base substitution** | position-dependent Phred `Q(i)` (Illumina ramp, reusing `_phred_run`'s shape), `P(err) = 10^(−Q/10)`, substitution uniform over the other three bases | mean Q30 → ~1e-3/base | Yes, but mostly it *drops* reads (an error inside a constant region breaks the regex). Cheap to model, needed as the substrate for layer 6. |
| 5 | **Indels** | `indel_rate` per base, length ~ Geometric(0.7) | 5e-6 | **Yes, disproportionately** — spaCR's extractor is *positional* (`offset_start`, `expected_end`). A 1-nt deletion upstream of the barcodes shifts every field by one and produces an unmappable read even though the barcode bases are perfect. That is a failure mode substitutions do not have, and it is invisible without simulating it. |
| 6 | **Barcode collision** | precompute the Hamming-1 (and -2) neighbour graph over each of the three barcode sets; a substitution that lands on another **real** barcode reassigns the read to that gene / row / column | derived, not a free parameter | **The one that corrupts results.** An error producing an unmappable read costs you power; an error producing a *valid wrong* barcode costs you correctness — a phantom gene in a well, indistinguishable from a real low-abundance one. Report `min_hamming_distance` per barcode set and `expected_collisions` as first-class QC outputs. If min distance ≥ 3, no single substitution can collide, and the simulator should say so. |
| 7 | **Index hopping** | with prob `index_hop_rate`, swap the read's (row, column) barcode pair for another well's, sampled ∝ that well's library share | 0.003 | **Yes.** The real data is NovaSeq X (patterned flow cell, ExAmp) — 0.1–2% with non-unique dual indexes is the published range. This is what the old `sequencing_error` was reaching for, done properly: it moves reads between wells in proportion to abundance, not uniformly. |
| 8 | **Chimeras / template switching** | with prob `chimera_rate`, join the gRNA of one template to the row+column barcodes of another from the same PCR pool | 0.01 | **Yes.** Rises with cycles and pool complexity. It puts a *wrong gene* into a *real well* at low frequency — which is exactly the population the vignette's 2% `gene_in_well_threshold` is silently discarding. Simulating it lets you derive that threshold instead of guessing it. |
| — | Read loss (adapter dimer, phiX, failed QC) | scalar depth deflation `read_loss_rate` | 0.05 | Included as one number; not worth a read-level model. |

**Rejected: read-level duplicates.** spaCR has no UMIs, so duplicates are unobservable and
indistinguishable from real counts. Their only statistical effect is count over-dispersion, which
layer 2 already produces. Modelling both double-counts the same variance.

### 7.3 `SeqErrorSpec`

```python
@dataclass(frozen=True)
class SeqErrorSpec:
    pcr_jackpot_sd: float = 0.4
    pcr_cycles: int = 22
    pcr_efficiency: float = 0.85
    pcr_model: str = "lognormal"        # 'lognormal' | 'branching'
    read_depth_cv: float = 0.35
    base_error_rate: float = 1e-3       # mean; profile shapes it along the read
    q_profile: str = "illumina_ramp"    # 'illumina_ramp' | 'flat'
    indel_rate: float = 5e-6
    index_hop_rate: float = 0.003
    chimera_rate: float = 0.01
    read_loss_rate: float = 0.05
    collisions: bool = True             # honour the real barcode neighbour graph
    @classmethod
    def none(cls): ...                  # every rate 0 — the clean baseline
    @classmethod
    def illumina_novaseq(cls): ...      # the defaults above
    @classmethod
    def illumina_miseq(cls): ...        # unpatterned: index_hop_rate=0.0002
    def ablate(self) -> Iterator[tuple[str, "SeqErrorSpec"]]: ...
```

### 7.4 Two modes, one model

- **Count mode (default, fast).** The layers are applied analytically to the (well × gRNA) count
  matrix: collisions become a sparse mixing matrix `M` (`counts_obs = counts_true @ M`), hopping a
  well-mixing matrix, chimeras a rank-1-plus-sparse perturbation. Sub-second for 1536 wells. This
  is what the sweep uses.
- **Read mode (`emit_fastq=True`).** Actually writes the reads, base by base, through the same
  parameters. Slow (~1e6 reads/min), used for the round-trip test and for the user who wants a
  realistic FASTQ to hand to a collaborator. **The two modes must agree**: §12.4 asserts the count
  mode reproduces the read mode's empirical mixing matrix within Monte-Carlo error. That
  consistency test is the whole justification for having a fast path.

### 7.5 What comes out

`Screen.write_fastq(dst)` produces the exact directory a `map_barcodes` run expects:

```
dst/
  fastq/  sim_R1_001.fastq.gz, sim_R2_001.fastq.gz
  barcodes/  row_barcodes.csv, column_barcodes.csv, grna_barcodes.csv   # sequence,name
  settings_map_barcodes.csv      # regex/target_sequence/offset/expected_end already correct
  truth/  counts_true.csv, collisions.csv, hops.csv, chimeras.csv
```

gRNA names are emitted as `TGGT1_<gene6>_<guide>` — three underscore-separated fields, because
`spacr.ml.process_reads` does `merged_df['grna'].str.split('_', expand=True)` into
`(org, gene, grna)` and silently falls over on anything else.

---

## 8. Additions — accepted and rejected

### Accepted

**8.1 gRNAs within genes, with variable efficiency.** *The biggest gap.* spaCRPower simulates
**genes**; spaCR maps **gRNAs** and regresses `fraction:grna + gene_fraction:gene`. Without a
guide level the simulator cannot produce data spaCR's own analysis can consume, and cannot answer
"do 4 guides per gene beat 6?" — a question that costs real money. Each guide gets an efficiency
`~ Beta(mean, sd)`; a guide below `sim_grna_dud_threshold` produces no phenotype. Genes are hits;
guides are how much of the hit shows up.

**8.2 Non-targeting and positive controls, with real plate placement.** `perform_regression` calls
hits relative to controls; today's `sim.py` has `number_of_control_genes` and the R package has
none at all. The real layout puts NC in column 1, PC in column 2, sweep in column 3. Without
controls in the simulation, the sweep cannot exercise spaCR's actual hit-calling rule, only a
bespoke one — which defeats the purpose.

**8.3 Plate / row / column / edge effects.** The real screen is 4 × 384 plates and spaCR's
regression carries `rowID + columnID` terms. If the simulator has no spatial structure, the sweep
cannot tell you whether those terms rescue you or just burn degrees of freedom, and cannot say
whether an observed edge effect is fatal. Four scalars (`plate_effect_sd`, `row_effect_sd`,
`column_effect_sd`, `edge_effect`) acting multiplicatively on well abundance and additively on the
classifier logit. Feeds `plot_plate_map` and pairs with the existing Plate Viewer's edge-effect
detection.

**8.4 Classifier calibration drift between plates and wells.** The R model draws the classifier
probability i.i.d. per (gene, well), which understates real error badly. Correlated classifier
error is what *creates false hits*, because a plate-wide shift in the operating point looks like a
well-level phenotype. Add logit-scale random effects: `class_drift_plate_sd`,
`class_drift_well_sd`. This is cheap and changes the answer more than almost anything else in the
list — with a classifier at 0.80/0.12, a 0.3-logit plate drift is comparable to the whole signal.

**8.5 Fitness effect / dropout for hits.** A knockout that kills the cell removes it from imaging
**and** from the sequencing pool. That is a selection effect that biases the read fraction — the
covariate — and in many screens it is the primary readout. `hit_fitness_lfc` (log2 change in cell
count for hit genotypes, default 0 = the R assumption). Without it the simulator assumes the
phenotype is invisible to abundance, which is often false and materially changes power.

**8.6 Multi-integration (MOI).** At lentiviral MOI > ~0.3, some cells carry two constructs and the
phenotype is attributed to both, diluting effect size. Modelled as a single scalar
`multi_integration_rate` (fraction of cells assigned a second, random genotype) rather than a full
MOI/Poisson-truncation model — the scalar captures the effect and is the thing an experimentalist
can actually estimate.

### Rejected, with reasons

- **Pixel-level imaging artefacts / focus failure.** Out of scope: this module works in counts,
  and `spacr/qt/synthetic.py` already generates images. What focus failure *does to the numbers*
  is lose a well, so it is one scalar, `well_qc_fail_rate`, which zeroes a well's imaging while
  leaving its sequencing intact — a real and under-appreciated asymmetry.
- **Gaussian-process spatial autocorrelation on plate coordinates.** Row/column effects plus an
  edge multiplier capture what people actually see on a plate map. A GP has more parameters than
  four plates can identify; it would be unfalsifiable decoration.
- **Read-level duplicate modelling.** See §7.2 — double-counts variance already produced by PCR
  jackpotting, and without UMIs it is unobservable anyway.
- **A separate cell-segmentation error model.** Segmentation error is already absorbed by
  `class_neg_mu` / `class_pos_mu`; giving it its own parameters would make the two unidentifiable
  and invite users to tune both.

---

## 9. The GUI

### 9.1 Registration

Add to `spacr/qt/app.py`:

```python
("sim", "Simulate Screen",
 "Plan a pooled screen: simulate it end to end, price the sequencing error, "
 "and sweep wells/reads/classifier accuracy for the power you need",
 SECTION_DATA),
```

**Section choice.** `SECTION_CORE` is **full at 9/9** and its nine entries are bound to
Ctrl+1..9, so Simulation cannot go there without an overflow (`MAX_APPS_PER_SECTION = 9`,
enforced by tests). Counts today: Core 9, Data 6, Models 5, Results 6, Toxo 3.
**Data & batch runs → 7/9.** It fits the section's stated purpose ("get tables into a spaCR
project, run things unattended, get numbers back out"): the simulator writes project-shaped CSVs
and FASTQ, and its sweep is an overnight batch run. Runner-up is Results & QC (6→7), which is
defensible if you read power analysis as QC. If the user prefers, renaming the section to
"Data, batch & simulation" is free — section display names are not load-bearing, only app keys are.

Icon: no bundled PNG reads as "simulate", so add `"sim"` to `_FORCE_GLYPH` and use a qtawesome
dice/flask glyph, or map `_ICON_OVERRIDES["sim"] = "sequencing.png"`. Prefer the glyph.

### 9.2 `spacr/qt/screens/sim.py` — `SimScreen`

Not a plain `AppScreen`. A generic settings-form-plus-Run screen would give the user 60 numbers
and a console, which is exactly the experience that made the current simulator unused. The screen
is a **three-tab QWidget** (~900 lines), modelled structurally on
`spacr/qt/screens/hyperparam.py` (QThread worker, signals, inline errors, never a modal):

**Tab 1 — Design.** The settings form, built by reusing `SettingsWidgets` from
`screens/settings_model.py` with the five new categories (§10), so tooltips, type hints and the
settings-CSV import/export all come for free. Alongside it, a **live plate map** and four live
summary numbers that update on every edit, computed from a fast 1-replicate draw:
*genes per well · wells per gene · cells per well · expected hits*. This is the single highest-value
piece of the screen: today a user has no way to tell that `well_abundance_mu = 4.6` means "about
5 genes per well" without running the whole thing.

**Tab 2 — Simulate.** Run one screen. Shows the eight R plots in a grid, plus the
**sequencing-error waterfall**: AUROC with no error, then each layer added, then all — the "how
much does it cost me" answer, as a bar chart with the loss annotated. Buttons: *Export CSVs*
(writes `score_data` + `count_data` + a `settings_regression.csv`), *Export FASTQ*, and
**"Open in Regression"** / **"Open in Map Barcodes"**, which navigate to those screens with the
paths pre-filled (the pattern already exists — see `_on_zoo_compare_requested` and
`_on_train_requested` in `app.py`). That hand-off is what makes this a module rather than a toy.

**Tab 3 — Power sweep.** Pick 1–2 parameters, ranges, replicates, estimator; run on a
`ProcessPoolExecutor`; stream points onto the AUROC-vs-parameter plot as they land, with a
progress bar and a Stop button; write the same progress TSV the R package writes so a partial run
is never lost. A "Generate SLURM array" button emits the sbatch script for a cluster.

**Threading.** Per the project rule: relay `worker.finished` through a bound-method signal
(`self._on_worker_finished`), never a lambda, or the handler runs off the GUI thread. Do not
`deleteLater` the worker from inside its own `finished`. Copy `hyperparam.py`'s pattern verbatim,
including its `closeEvent` guard against destroying a widget whose QThread is still running.

### 9.3 The Tk bug — fix, don't retire

Today: `spacr/validate.py`, `spacr/cli.py`, `spacr/gui_utils.run_function_gui` and
`gui_core.start_process`'s whitelist **all** know about `'simulation'`, but
`gui_core.setup_settings_panel` has no branch for it and ends at
`raise ValueError(f"Invalid settings type: {settings_type}")`. On top of that, `spacr/gui.py`
lists Simulation in **neither** `main_gui_apps` nor `additional_gui_apps`, so nothing can even
reach it. The panel is unopenable and the button does not exist.

**Fix it.** Tk and Qt coexist by policy, and once `get_simulation_default_settings` exists in
`settings.py` (which Qt needs anyway) the fix is three lines:

1. `gui_core.setup_settings_panel`: `elif settings_type == 'sim': settings = get_simulation_default_settings(settings={})`
2. `gui.py`: one entry in `additional_gui_apps`.
3. `gui_core.start_process`: change `'simulation'` to `'sim'` in the whitelist.

**Rename the key `'simulation'` → `'sim'`** across `validate.APP_FUNCTIONS`,
`cli.MODULES`/`ALIASES` (`'simulation'` becomes an alias), `gui_utils.run_function_gui`,
`bridge.resolve_pipeline_entry` and `settings_model.resolve_default_settings`, so the Qt app key,
the CLI module name and the settings type are one string. `tests/test_app_entry_consistency.py`
exists precisely to catch this class of divergence and must be extended to cover `sim`.

---

## 10. Settings keys

Reuse the already-categorised `src`, `plot`, `verbose`, `n_jobs`. Prefix everything else `sim_`
so nothing collides with the ~800 existing keys. Defaults are the vignette's fitted values for the
real *T. gondii* screen, which makes the shipped defaults a working example rather than filler.

Five new `categories` entries (each ≥ 10 keys, per the repo's "no two-entry headings" rule):
**"Screen design"**, **"Simulated imaging"**, **"Simulated sequencing"**,
**"Sequencing errors"**, **"Power analysis"**.
`tests/test_settings_categories.py` enforces that every key returned by
`get_simulation_default_settings` appears in exactly one category — write the map in the same
commit as the factory.

### Screen design

| key | type | default | tooltip |
|---|---|---|---|
| `sim_n_genes` | `int` | 452 | (int) - Genes in the library. The real *T. gondii* screen this simulator was fitted to had 452. Power falls roughly as the log of this: doubling the library costs about as much as halving the wells. |
| `sim_n_grnas_per_gene` | `int` | 4 | (int) - gRNAs per gene. More guides per gene raise the chance at least one works, but at fixed well count they dilute each guide's representation. Sweep this against `sim_grna_efficiency_mean` before committing to a library design. |
| `sim_gene_abundance_alpha` | `float` | 0.6 | (float) - Dirichlet concentration on gene abundance. Large values give an even library; small values give a skewed one where a few genes dominate. At 1.0 the Gini index is 0.5. The real screen fitted to 0.6, i.e. quite skewed - which is why some genes never reach enough wells. |
| `sim_gene_hit_rate` | `float` | 0.025 | (float) - Fraction of genes that are true hits. 0.025 was inferred from the real screen by inverting the well positivity rate against the classifier operating point. This is the single number most worth checking against your own pilot data. |
| `sim_grna_efficiency_mean` | `float` | 0.7 | (float) - Mean fraction of a hit gene's phenotype that a single guide produces. 1.0 means every guide is perfect; the literature median for CRISPR knockouts is nearer 0.6-0.8. |
| `sim_grna_efficiency_sd` | `float` | 0.25 | (float) - Spread of guide efficiency within a gene, as a Beta standard deviation. Large values mean a gene is only found if you happen to have drawn its one good guide - which is the argument for more guides per gene. |
| `sim_n_control_genes` | `int` | 10 | (int) - Non-targeting controls. spaCR's regression sets its hit cutoff from the spread of control coefficients, so a screen simulated without them cannot exercise the real hit-calling rule. Below about 8 the cutoff is itself noisy. |
| `sim_pc_gene` / `sim_nc_gene` | `str` | `'TGGT1_220950'` / `'TGGT1_233460'` | (str) - Positive / negative control gene, plated in the control columns. The rank the positive control achieves is reported alongside AUROC, because that is the number you will actually look at on the day. |
| `sim_n_plates` | `int` | 4 | (int) - Plates in the screen. |
| `sim_plate_format` | `combo 96/384/1536` | 384 | (int) - Wells per plate. Total wells is this times `sim_n_plates`, minus the control columns. |
| `sim_control_columns` | `str (dict)` | `"{'NC': 1, 'PC': 2, 'Sweep': 3}"` | (dict) - Which plate columns are reserved for controls, matching the real screen layout. These wells are simulated but excluded from the screen analysis, exactly as `well_type == 'Screen'` does in the real data. |
| `sim_well_abundance_mu` | `float` | 4.6 | (float) - Mean well abundance factor. With 452 genes this gives about 4.6 genes per well. This is the knob that trades genes-per-well against wells-per-gene, and the sweep the R package cared most about. |
| `sim_well_abundance_var` | `float` | 1.0 | (float) - Variance of the well abundance factor - how unevenly the pool spots into wells. |
| `sim_plate_effect_sd` | `float` | 0.0 | (float) - Log-scale standard deviation of a per-plate multiplier on cell abundance. Set to 0 to reproduce spaCRPower, which has no plate term; set to ~0.2 to ask whether your batch effect is fatal. |
| `sim_row_effect_sd` / `sim_column_effect_sd` | `float` | 0.0 | (float) - Per-row / per-column effects. spaCR's regression fits `rowID + columnID` terms, so turning these on is how you find out whether those terms are earning their degrees of freedom. |
| `sim_edge_effect` | `float` | 1.0 | (float) - Multiplier applied to the outer ring of each plate. 1.0 is no edge effect; 0.7 is a typical evaporation-driven cell loss. Pairs with the Plate Viewer's edge-effect detection. |

### Simulated imaging

| key | type | default | tooltip |
|---|---|---|---|
| `sim_cells_per_well_mu` | `float` | 123.0 | (float) - Mean cells imaged per well. The real screen averaged 123. This is the parameter you buy with microscope time, and the sweep tells you what it buys. |
| `sim_cells_per_well_var` | `float` | 8000.0 | (float) - Variance of cells per well. Far above the mean (8000 vs 123), so counts are drawn from a negative binomial. Variance below the mean is also allowed and draws from a binomial instead. |
| `sim_imaging_split` | `combo abundance/uniform` | `'abundance'` | (str) - How a well's cells are divided between the genes in it. 'abundance' weights by each gene's share of the well; 'uniform' splits evenly, which is what spaCRPower did. 'uniform' understates the imbalance and is offered only for comparison with the R package. |
| `sim_well_qc_fail_rate` | `float` | 0.0 | (float) - Fraction of wells whose imaging fails outright (focus loss, debris, missing field). The well's sequencing survives, which is the asymmetry that makes a failed well different from a missing one. |
| `sim_class_pos_mu` | `float` | 0.80 | (float) - Probability a cell with a hit genotype is called positive - the classifier's true positive rate at its operating point. The real MaxViT classifier sat at 0.80, which is low, and the whole point of the power analysis is that a low number here is survivable if you have enough wells. |
| `sim_class_pos_var` | `float` | 0.10 | (float) - Variance of that probability across cells. |
| `sim_class_neg_mu` | `float` | 0.12 | (float) - Probability a non-hit cell is called positive - the false positive rate. 0.12 in the real screen. The signal you are trying to detect is the gap between this and `sim_class_pos_mu`. |
| `sim_class_neg_var` | `float` | 0.01 | (float) - Variance of the false positive rate across cells. |
| `sim_class_drift_plate_sd` | `float` | 0.0 | (float) - Logit-scale drift of the classifier's operating point between plates. spaCRPower draws classifier error independently for every observation, which understates the damage badly: a plate-wide shift looks exactly like a plate-wide phenotype and manufactures false hits. With a classifier at 0.80/0.12, a drift of 0.3 is comparable to the entire signal. |
| `sim_class_drift_well_sd` | `float` | 0.0 | (float) - The same drift at the well level (staining, exposure, confluence). |
| `sim_hit_fitness_lfc` | `float` | 0.0 | (float) - log2 change in cell number for a hit genotype. Negative means the knockout costs the cell fitness, so hits are depleted from both the images and the reads - a selection effect that biases the read fraction the model regresses on. 0.0 reproduces spaCRPower, which assumes the phenotype is invisible to abundance. |
| `sim_multi_integration_rate` | `float` | 0.0 | (float) - Fraction of cells carrying a second construct. At lentiviral MOI above ~0.3 this is real, and it dilutes effect size by attributing one cell's phenotype to two genotypes. |

### Simulated sequencing

| key | type | default | tooltip |
|---|---|---|---|
| `sim_cells_per_well_seq` | `float` | 1000.0 | (float) - Cells per gene per well contributing DNA to the sequencing library. Usually far larger than the imaged count, because sequencing sees the whole well and the microscope sees a few fields. |
| `sim_cells_per_well_seq_var` | `float or None` | None | (float) - Variance of that count. None means Poisson. |
| `sim_reads_per_well` | `int` | 30000 | (int) - Mean reads per well. spaCRPower's `n_reads_total` was ambiguous - documented as the screen total in one place and the per-well geometric mean in another, and divided in the code by the number of genes rather than the number of wells. This key is unambiguously per well; the screen total is derived and reported. |
| `sim_read_depth_cv` | `float` | 0.35 | (float) - Coefficient of variation of read depth between wells. Real screens are far from uniform, and shallow wells are where hits go to die. |
| `sim_pcr_factor_mu` | `float` | 2.0 | (float) - Log-scale mean of the per-well PCR amplification factor. |
| `sim_pcr_factor_var` | `float` | 1.0 | (float) - Log-scale variance of the per-well amplification factor. |
| `sim_gene_in_well_threshold` | `float` | 0.02 | (float) - Read fraction below which a gene is treated as absent from a well. The real analysis uses 2%, chosen by eye from a histogram. Turn on chimeras and index hopping in Sequencing errors and this simulator will tell you what the threshold should actually be. |

### Sequencing errors

| key | type | default | tooltip |
|---|---|---|---|
| `sim_seq_error_preset` | `combo none/miseq/novaseq/custom` | `'novaseq'` | (str) - Preset for the whole error model. 'none' is a clean baseline with every rate at zero - run it alongside a preset to price the error. 'novaseq' assumes a patterned flow cell, which hops indexes far more than 'miseq'. |
| `sim_base_error_rate` | `float` | 0.001 | (float) - Mean per-base substitution rate (Q30). Applied through a position-dependent quality profile, so the 3' end of a read is much worse than the 5' end, as on a real instrument. Most substitutions simply break the read; the ones that matter are the ones in the next row. |
| `sim_q_profile` | `combo illumina_ramp/flat` | `'illumina_ramp'` | (str) - Shape of the quality profile along the read. 'illumina_ramp' degrades toward the 3' end like a real run; 'flat' is uniform and exists only to isolate the effect. |
| `sim_indel_rate` | `float` | 5e-06 | (float) - Per-base insertion/deletion rate. Rare, but they hurt out of proportion: spaCR extracts barcodes by fixed offset from an anchor, so one deleted base upstream shifts every field and destroys a read whose barcode bases were perfect. Substitutions cannot do that. |
| `sim_barcode_collisions` | `bool` | True | (bool) - Honour the real barcode set when a substitution lands on another valid barcode. This is the error that corrupts results rather than merely losing reads: the read is not dropped, it is silently reassigned to a different gene, row or column. spaCR reports the minimum Hamming distance of your barcode sets alongside the collision rate - if that distance is 3 or more, no single substitution can collide and you can leave this off. |
| `sim_index_hop_rate` | `float` | 0.003 | (float) - Fraction of reads whose row/column index pair is swapped with another well's. Patterned flow cells (NovaSeq) hop at 0.1-2% with non-unique dual indexes; MiSeq is ~100x lower. This is what moves reads between wells, and it moves them in proportion to abundance rather than uniformly. |
| `sim_chimera_rate` | `float` | 0.01 | (float) - Fraction of reads that are PCR chimeras - one template's gRNA joined to another's well barcodes. Rises with cycle count and pool complexity. Chimeras put a wrong gene into a real well at low frequency, which is exactly the population the 2% gene-in-well threshold is throwing away. |
| `sim_pcr_jackpot_sd` | `float` | 0.4 | (float) - Log-scale spread of amplification efficiency between templates - PCR jackpotting. spaCRPower varied amplification per well only, which just rescales depth and cancels out of the read fraction; per-template variation is what actually distorts the abundance the model regresses on. |
| `sim_pcr_cycles` | `int` | 22 | (int) - PCR cycles. Used by the branching amplification model and to scale the chimera rate. |
| `sim_read_loss_rate` | `float` | 0.05 | (float) - Fraction of reads lost to adapter dimers, phiX and failed filters. Deflates effective depth and nothing else. |
| `sim_emit_fastq` | `bool` | False | (bool) - Write real gzipped FASTQ files and barcode CSVs instead of applying the error model directly to counts. Far slower, but the output goes straight through spaCR's own Map Barcodes module, which is the only way to prove the simulated reads are actually readable. |

### Power analysis

| key | type | default | tooltip |
|---|---|---|---|
| `sim_estimator` | `combo spacr_ols/hs_em/glm_l1/ridge_laplace/stan` | `'spacr_ols'` | (str) - Which model calls the hits. 'spacr_ols' is spaCR's own regression, so the power you measure is the power of the pipeline you will actually run - which is why it is the default. 'hs_em' is the sparse Poisson model spaCRPower used, fitted by EM plus a Laplace approximation instead of MCMC. 'stan' runs the original Stan model and needs `pip install spacr[stan]`. |
| `sim_pseudocount` | `str or float` | `'half_read'` | (str) - Pseudocount added before taking log10 of the read fraction. 'half_read' scales with each well's depth, so "absent" means the same thing at every depth. spaCRPower used a fixed 0.0001, which silently changes meaning as you sweep depth and makes points on that sweep incomparable. |
| `sim_random_row_column` | `bool` | False | (bool) - Fit row and column as random effects instead of fixed. Slower, but the right choice when you have many plates and few wells per row. |
| `sim_scan_variable` | `str` | `'n_wells'` | (str) - Which setting to sweep. Any numeric key on this screen can be swept by name. |
| `sim_scan_values` | `list` | `[10, 510, 1010, 1510, 2010]` | (list) - Values for the swept setting. |
| `sim_scan_variable_2` / `sim_scan_values_2` | `str` / `list` | None / `[]` | (str/list) - An optional second swept setting, drawn as separate lines on the same plot. Two is the limit that stays readable. |
| `sim_replicates` | `int` | 3 | (int) - Independent simulations per grid point. One replicate tells you nothing about a stochastic simulator; the plot shows the spread across replicates and you should not believe a difference smaller than it. |
| `sim_seed` | `int or None` | 0 | (int) - Master random seed. Every stage draws from its own derived stream, so turning the error model on or off does not change the underlying biology - which is what makes the with/without comparison exact rather than approximate. |
| `sim_compare_error` | `bool` | True | (bool) - Also run every grid point with the error model disabled and report the difference. This is the "what does sequencing error cost me" answer, and it is nearly free because the clean run reuses the same simulated biology. |
| `sim_metric` | `combo auroc/ap/pc_rank/precision_at_k` | `'auroc'` | (str) - What the sweep plots. AUROC is robust and comparable across hit rates; average precision is more honest when hits are rare; 'pc_rank' is where your positive control landed, which is the number you will check first on the day. |
| `sim_backend` | `combo process/serial/slurm` | `'process'` | (str) - How to run the sweep. 'slurm' writes an sbatch array script instead of running anything. |
| `sim_progress_file` | `str or None` | None | (str) - TSV appended after every grid point, so a sweep killed halfway is not lost. |
| `sim_save_screens` | `bool` | False | (bool) - Keep every simulated screen on disk instead of only its score. Large; useful when a grid point behaves strangely and you want to look at it. |

`expected_types` entries follow the existing conventions
(`"sim_cells_per_well_seq_var": (float, type(None))`, `"sim_scan_values": list`,
`"sim_pseudocount": (str, float)`, `"sim_seed": (int, type(None))`, and so on), and
`sim_estimator` / `sim_seq_error_preset` / `sim_metric` / `sim_backend` / `sim_imaging_split` /
`sim_q_profile` / `sim_plate_format` go into `convert_settings_dict_for_gui.special_cases` as
combos.

---

## 11. Migration

The rule is future-first: fix the design and convert old usage. Here is exactly what changes.

**Removed outright** (all of it is either broken, unused, or reimplemented properly):

| Old symbol | Fate |
|---|---|
| `run_simulation`, `run_and_save`, `run_multiple_simulations`, `generate_paramiters` | Replaced by `simulate_screen` + `scan_parameters`. Shim for one release. |
| `run_experiment` | Replaced by `simulate_spot_plate` + `simulate_imaging_plate`. The old one materialises **one row per cell** in a Python list and loops per well — it is O(cells) in RAM and is why the current simulator cannot do 1536 wells × 123 cells at any useful sweep size. |
| `sequence_plates` | Replaced by `simulate_sequencing_plate` + `seqerror`. Its `sequencing_error` moved a whole gene's read block to a uniformly random well — not a real error mode, and it iterates `.loc` per gene per well. |
| `generate_power_law_distribution`, `power_law_dist_gen` | Replaced by the Dirichlet library model, which is what the R package fits and what actually matches the data. |
| `dist_gen`, `generate_gene_weights` | Replaced by the moment-parameterised helpers in §2.1. |
| `visualize_all` (13 panels in one row, 52 in wide), `vis_dists` | Replaced by `plots.py`. Both call `plt.show()` inside a library function, which hangs headless runs. |
| `save_data`/`append_database`/`create_database`/`read_simulations_table` (SQLite) | Replaced by tidy CSV/parquet output. The SQLite schema stores one wide row per simulation with settings and stats concatenated; it is not queryable, not diffable, and duplicated by the progress TSV. |
| `plot_simulations`, `plot_correlation_matrix`, `plot_feature_importance`, `calculate_permutation_importance`, `plot_partial_dependences`, `generate_shap_summary_plot` | **Kept**, moved to `sim/plots.py`. These analyse a completed sweep and are genuinely useful. Their hard-coded `grouping_vars` list must be re-derived from the swept keys instead of being a literal. This also removes the top-level `shap` import from the `import spacr` path. |
| `classifier`, `compute_roc_auc`, `compute_precision_recall`, `get_optimum_threshold`, `cell_level_roc_auc`, `gini*`, `normalize_array`, `generate_integers`, `generate_floats`, `remove_columns_with_single_value`, `remove_constant_columns` | **Kept**, moved and re-exported unchanged. Well tested and correct. |

**Behavioural changes that will surprise someone:**

1. `plt.show()` no longer called from library code. Notebook users must `display(fig)`.
2. Output is CSV/parquet, not `simulations.db`. Provide `spacr.sim.migrate_legacy_db(path)` to
   convert an existing database into the new tidy frame — do not ask anyone to re-run a sweep.
3. `random.seed(42)` under a truthy `settings['random_seed']` is replaced by an explicit
   `np.random.Generator` per stage. The global `random` module is no longer touched, which also
   fixes the current situation where running a simulation reseeds the caller's RNG.
4. Gene identifiers are strings (`TGGT1_000123_1`), not integers. Required by
   `spacr.ml.process_reads`, and it removes the int/str join failure the R package hits.
5. `sequencing_error` (a single float) becomes `SeqErrorSpec`. The shim maps a bare float onto
   `SeqErrorSpec(index_hop_rate=x)`, which is the closest honest reading of what the old parameter
   did, and warns saying so.
6. App key `'simulation'` → `'sim'`, with `'simulation'` kept as a CLI alias.

**Deprecation path.** One release (1.3.6 → 1.3.7) in which `spacr.sim.run_multiple_simulations`
and friends exist in `legacy.py`, are re-exported, emit a `DeprecationWarning` naming the
replacement, and are covered by the existing tests. `spacr/sim/__init__.py` carries a
`_LEGACY_NAMES` tuple and a module `__getattr__` so the warning fires on *attribute access*, not
on import — otherwise `import spacr` warns for everyone. Delete `legacy.py` in 1.3.8.

**Test migration.** Only 11 monkeypatch sites across the sim tests, and 9 of them patch
`cpu_count`. Repoint those at `spacr.sim.legacy` in the same commit that creates the package.
`tests/test_sim.py` (191 lines, pure-math assertions on gini/normalize/classifier) needs no
changes at all, since those symbols are re-exported.

---

## 12. Test plan

How do you test a stochastic simulator? Six ways, in descending order of how much they are worth.

**12.1 Seeded determinism.** `simulate_screen(spec, seed=0)` twice → identical frames
(`pd.testing.assert_frame_equal`). Then the stronger one that most simulators fail:
**stream independence** — `simulate_screen(spec_with_error, seed=0).spot` must be *bit-identical*
to `simulate_screen(spec_no_error, seed=0).spot`. If turning on the error model perturbs the
biology, every with/without comparison in §5.2 is confounded, and this test is the only thing
standing between the user and a silently wrong answer.

**12.2 Distributional assertions against analytic expectations.** Each stage has a closed-form
mean, and each gets a test at n large enough that a 5-sigma band is tight, with the seed fixed so
the test cannot flake:

| Quantity | Expectation |
|---|---|
| `gene_abundance.sum()` | exactly 1.0 |
| `Var(gene_abundance)` | `(α_i(α₀−α_i))/(α₀²(α₀+1))` for the symmetric Dirichlet |
| mean genes per well | `n_genes × E[abundance] × E[well_abundance] = well_abundance_mu` |
| mean/var of cells per well | the requested `mu`, `var`, for all three branches of `count_model` |
| `E[positive | hit]` | `n_cells × class_pos_mu` |
| `E[reads for gene g in well w]` | `reads_per_well × amplified_share(g,w)` (multivariate hypergeometric mean) |
| Gini of the library | matches the analytic Dirichlet Gini at α = 1 (= 0.5), as the R docstring claims |
| collision rate | matches `Σ_b P(1 substitution) × (#Hamming-1 neighbours in the set)/(3L)` |
| index-hop rate | recovered from the truth table to within Monte-Carlo error |

Plus a Kolmogorov–Smirnov test of the simulated well-read-fraction distribution against the
analytic marginal, at a seed-fixed p-value floor.

**12.3 Round-trip through spaCR's real barcode mapper.** The one that proves the output is worth
producing. Write FASTQ + barcode CSVs with `SeqErrorSpec.none()`, run
`spacr.sequencing.generate_barecode_mapping` on it, read back `unique_combinations.csv`, and
assert the per-`(rowID, columnID, grna_name)` counts **exactly equal** the ground truth. Then with
the NovaSeq preset, assert the recovered fraction and the misassignment rate match the analytic
prediction within tolerance. Small case (2 plates × 8 wells × 12 guides × 500 reads) so it runs in
CI in seconds; the full-size version is marked `slow`.

Second round trip, equally important: `Screen.write(dst)` → `spacr.ml.perform_regression` on the
emitted `score_data` / `count_data` → the run completes and the planted hit genes are ranked in the
top decile. That is the test that proves the simulator speaks spaCR's own dialect, and it will
catch the `TGGT1_gene_guide` three-field naming requirement the first time someone breaks it.

**12.4 Fast path vs slow path.** The count-mode error model must reproduce the read-mode empirical
mixing matrix within Monte-Carlo error (χ² on the well×well and gene×gene reassignment tables).
Without this test the fast path is an unverified shortcut, and the fast path is what every sweep
uses.

**12.5 Estimator recovery.** On a screen simulated with a known hit set and generous parameters
(200 genes, 1000 wells, class_pos 0.95 / class_neg 0.02), every estimator must reach AUROC > 0.9;
on a null screen (`gene_hit_rate = 0`) every estimator's AUROC must be indistinguishable from 0.5
(95% CI over 20 seeds covers 0.5) — the null test is the one that catches a sign error or a
leaked label, and it is the test the R package's inverted `mean_inv` would have failed. Also pin
monotonicity: AUROC must be non-decreasing in `n_wells` over a 5-point sweep, averaged over
replicates.

**12.6 Cross-validation against R / Stan** (`@pytest.mark.optional`, skipped unless cmdstanpy is
installed). `hs_em` vs the `stan` backend on the same design: Spearman correlation of per-gene
estimates > 0.95, and identical AUROC to within 0.02. Record the observed tolerance in the test
docstring rather than asserting a number pulled from the air. A companion fixture stores a small
frozen design matrix + the R package's own output for that design, so the comparison survives
cmdstanpy not being installed anywhere.

**12.7 GUI.** Following `tests/qt/` conventions: `SimScreen` constructs offscreen; the Design tab's
live summary updates on a settings change without touching disk; the worker thread starts and
stops cleanly on `closeEvent` mid-run; `sim` appears in all five dispatch tables
(`test_app_entry_consistency.py` extended); `MAX_APPS_PER_SECTION` still holds; every key from
`get_simulation_default_settings` has a category, an `expected_types` entry and a tooltip
(`test_settings_categories.py` + a new tooltip-coverage assertion).

**Runtime budget.** The whole non-slow sim suite must stay under ~60 s. `simulate_screen` at the
default 1536 wells × 452 genes should be well under a second in count mode — if it is not, the
sweep is unusable and that is a design failure, not a performance nit. Add a
`@pytest.mark.timeout`-style guard on the default-spec simulation.

---

## 13. Suggested implementation order

Each step is independently reviewable and leaves the tree green.

1. **Package split, no behaviour change.** `spacr/sim.py` → `spacr/sim/legacy.py` + `__init__.py`
   re-exporting everything; move the top-level `shap`/`sklearn`/`seaborn` imports into functions;
   repoint the 11 monkeypatch sites. Measures a real `import spacr` speedup on its own.
2. **`model.py` + `screen.py`**: the R port with every §3 bug fixed and §2 trap handled, plus the
   RNG-stream design. Tests 12.1 and 12.2.
3. **`estimate.py`**: `spacr_ols` and `hs_em` first; `glm_l1` / `ridge_laplace` / `stan` after.
   Tests 12.5.
4. **`io.py` + round trip to `perform_regression`.** Test 12.3b. This is the first point at which
   the module is genuinely useful, so it is worth reaching early.
5. **`seqerror.py`**, count mode first, then read mode + FASTQ. Tests 12.3a and 12.4. Fix the
   FASTA-vs-CSV mismatch in `qt/synthetic.py` here.
6. **`scan.py` + `plots.py`.** Tests 12.5 monotonicity.
7. **Settings**: `get_simulation_default_settings`, `expected_types`, the five categories, the
   combo special-cases, `descriptions['sim']`. Test 12.7 settings half.
8. **GUI**: `qt/screens/sim.py`, `app.py` registration, `bridge`/`validate`/`cli`/`gui_utils` key
   rename, and the three-line Tk fix. Test 12.7 GUI half.
9. **Deprecation warnings** on the legacy surface + `migrate_legacy_db`. Bump the patch version
   once, at the end, per the project rule.
10. **Docs**: port the vignette as a notebook — `examples/simulate_tgondii_screen.ipynb` —
    reproducing the R package's headline AUROC-vs-wells figure with the error model priced in.
    That notebook is the acceptance test for whether the port was worth doing.

---

## Appendix — open questions for the user

1. **Section placement.** Data & batch runs (recommended) or Results & QC? Either is 7/9.
2. **`gene_hit_rate = 0.025`** was inferred, not measured, by inverting well positivity against a
   guessed 0.80/0.12 classifier operating point. Is there a better estimate from the pilot data?
   Every power number scales with it.
3. **Index-hop rate.** Were the real libraries dual-indexed with *unique* dual indexes? If yes the
   default should be ~1e-4 rather than 0.003, and the answer changes the headline cost of error.
4. **The 2% `gene_in_well_threshold`.** Once chimeras and hopping are simulated, the simulator can
   derive it. Is the user willing to have that number change?
5. **Is the Tk GUI still worth 3 lines?** The plan says yes (they coexist by policy). If Tk is on
   its way out, say so and the Simulation panel should be removed from
   `gui_core.start_process`'s whitelist and `validate`/`cli` pointed at Qt only.
