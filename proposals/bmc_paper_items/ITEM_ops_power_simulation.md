# Item 968: OPS power submodule — simulate optical pooled screens with planted ground truth to learn what an OPS screen can and cannot detect

- **Number:** 968, reserved by master on 2026-11-17.
- **Requested by the maintainer** on 2026-11-17: "add an OPS power submodule to the spaCR power module and use it to perform OPS screen simulations to learn what we can and cannot know from OPS screens at the current scale (start with Funk scale)".
- **Placement (master, 2026-11-17):** built inside the spaCR Power module's OPS submodule, after item 969 moves the existing `ops_power_*` settings there.
- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Code basis:** origin/nightly.
- **Paper-critical:** yes. It sets the detection limits for every OPS claim, and supplies the missing power check for the DINO feature test (experiment_designs/20261116_foundation_model_features_vs_spacr.md).
- **Rule:**
  - A spaCR settings-file run must reproduce the reference outputs identically. Every seed is fixed.
  - Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Related:**
  - `spacr/power_simulate.py` + `spacr/power_model.py`: the spaCRPower port (MIT, M. O'Meara). It covers arrayed well-level pooled screens; reuse its samplers, e.g. `rnbinom_mean_variance`, `rdirichlet_stable`, `resolve_rng`.
  - `spacr/annotation_power.py`.
  - `proposals/bmc_paper_items/ITEM_ops_etest_power_curve.md` subsamples real cells. It has no ground truth, so it cannot give a miss rate. This item does, and it should reuse that item's E-test when it lands.

## Gap
- spaCR's power code answers a different question: "how many wells and cells for a well-level pooled screen?". The model there is per-well positive-cell counts against per-gene sequencing read fractions.
- An OPS screen has a different structure: per-cell genotype calls (with call rate and misassignment), a high-dimensional per-cell phenotype, guides nested in genes, and field/well batch structure.
- Today we cannot say, at a given screen scale:
  - what fraction of real phenotypes we detect;
  - how small an effect is detectable;
  - how many complexes the clustering step can recover;
  - how much each pipeline step (genotyping, segmentation, feature set) costs in power.

## Simulator (`spacr/power_ops.py`, new; imports helpers from `power_simulate`)
1. **Library:** genes, guides per gene, NT guides, guide-abundance skew (negative binomial or Dirichlet), and guide efficacy (a fraction of inactive guides, plus a per-guide knockout penetrance).
2. **Cells:**
   - cells per well and fields per well;
   - cell assignment by abundance;
   - genotyping as observed: call rate, wrong-gene rate at a set decoy FDR, and multi-barcode drop.
3. **Phenotype:** per-cell features drawn from a reference covariance.
   - **Mode A "empirical":** NT cells resampled from a real run, with features as measured.
   - **Mode B "parametric":** a fitted multivariate t.
   - **Field/well batch effects:** random offsets with variance taken from the real run.
4. **Planted biology:**
   - a fraction of genes with an effect;
   - an effect-size distribution, in the feature space or as bits;
   - penetrance (the share of cells responding);
   - effect type: mean shift, variance change, or a subpopulation;
   - the dimensionality of the effect (how many feature directions it touches).
   - **Complexes:** groups of genes that share an effect direction, with a noise term. Sizes come from CORUM, so recovery can be scored like the real analysis.
5. **Calibration from data:** `calibrate_from_run(run_dir)` reads a real spaCR OPS run and fills every nuisance parameter: cells per gene, call rate, FDR, NT covariance, batch variance, guide abundance. Start with Funk plate A (spaCR's own run). Every value it uses is written to a JSON file.

## Analysis inside each simulation
Run the same spaCR steps that real screens use, unchanged:
- per-gene tests against NT (the E-test when available; a Hotelling or permutation test until then) with BH;
- guide-replicate agreement;
- the aggregation (hybrid, 10 % trim), PHATE + Leiden clustering, and CORUM/STRING scoring;
- the conditional-information screen (I(gene; new | old)), for feature-set questions such as DINO.

## Outputs per simulated design
- detection power per effect size and penetrance;
- the false-positive rate on NT pseudo-genes;
- the minimum detectable effect at 80 % power;
- complex recovery (CORUM enriched %, STRING F1) against the planted truth;
- a "design curve": power against cells per gene, guides per gene, call rate and FDR.

## Sweep (`scan_ops_designs`)
- **Grid:** cells per gene (10–1,000), guides per gene (1–6), call rate (40–90 %), FDR (0.1–5 %), number of wells or plates, effect size and penetrance.
- **Seeds:** 3–5 replicates each.
- **Caching:** results cached per design key, as `power_model.scan_parameters` does.

## First study: Funk scale (the maintainer's request)
1. **Calibrate** on Funk plate A (and A1) from spaCR's run.
2. **Answer at Funk's library and depth:**
   - (a) the smallest mean-shift and penetrance effect detectable per gene;
   - (b) the share of planted complexes the clustering recovers;
   - (c) the gain from more cells per gene: 1 plate vs all plates; A1 depth (about 39 cells per gene) vs plate A depth (about 240);
   - (d) the cost in power of the genotyping call rate and FDR. This quantifies what spaCR's genotyping gain over Brieflow is worth.
   - (e) the smallest conditional information (bits) the information screen can detect at A1 and plate-A depth. This is the DINO power check.
3. **Validate the simulator against reality:**
   - simulated NT pseudo-gene false-positive rates must match the real run's NT controls;
   - the simulated per-gene statistic distribution for real hits must match the observed one in shape;
   - report both.

## Settings (alpha)
- `ops_power_sim` (bool)
- `ops_power_calibration_run` (path)
- `ops_power_mode` ('empirical' | 'parametric')
- `ops_power_grid` (dict)
- `ops_power_sim_seeds` ([0, 1, 2]). Renamed so it doesn't clash with the E-test item's existing `ops_power_seeds`.
- `ops_power_effect_types` (['mean', 'variance', 'subpopulation'])
- `ops_power_scorers` (['etest', 'clustering', 'information'])

## Acceptance
- The off switch changes nothing in existing outputs; byte-identical.
- **Recovery test:** a planted strong effect (e.g. ≥ 1 SD in 5 features, 100 % penetrance, 200 cells per gene) is detected at ≥ 95 % power; with no planted effects, the false-positive rate is ≤ 5 % at q < 0.05.
- The calibration JSON from Funk plate A reproduces the real run's cells-per-gene distribution and NT false-positive rate within 10 %.
- A settings-file rerun gives identical tables.
- **Speed:** one Funk-scale design replicate in ≤ 10 min on 8 CPU threads (empirical mode, without clustering), and ≤ 30 min with clustering.

## Paper use
Supplementary figure "What a Funk-scale OPS screen can detect": power curves and the minimum-detectable-effect table. It is cited wherever a negative result appears (the DINO screen, the localisation tiers): "below the detection limit of X at this depth".
