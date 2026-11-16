# Item 964: GPU feature-information screen, measuring what each new feature column adds about genes beyond the existing features

- **Number:** 964 (reserved by master on 2026-11-16).
- **Maintainer decisions (2026-11-16):**
  - New feature columns enter the OPS feature matrix only if they carry new and real gene information (`BMCspacr/reanalysis/experiment_designs/STANDARD_new_feature_acceptance.md`).
  - "Yes, add to standard": a GPU information-theory screen is that standard's fast first pass, and it becomes a spaCR feature.
- **When:** alpha, built after the genotyping and segmentation freeze. GPU use goes through `tools/gpu_turn.sh` under the 280 W power cap.
- **Gap:** spaCR has no tool that tells, column by column, how much information a new feature table adds about gene identity beyond an existing table. Redundancy and value are currently judged only by slow downstream clustering tests.

## Spec
1. **Entry points:**
   - `spacr.ops_feature_info.screen(old, new, cells, *, label='gene', guide='sgRNA', control=…, device='auto')`;
   - a phase `ops_feature_screen` in the OPS settings file;
   - a GUI panel listed under alpha.
   Inputs are the per-cell old and new feature tables, already normalised (robust z or TVN against controls). Rows are joined on cell ID.
2. **Gaussian conditional mutual information:**
   - one covariance of [old, new] on the GPU (torch, streamed over cells; float64 accumulation);
   - for each new column, its conditional variance given old from the Schur complement (unique fraction = 1 − R²);
   - added gene information I(gene; new | old) = ½ log det Σ(new|old) − ½ log det Σ(new|old, gene) (MANOVA/Wilks form), per column and per declared family, reported in bits;
   - within-gene covariances via GPU scatter-add group sums;
   - a ridge term so that singular blocks stay finite; the ridge value is recorded.
3. **Null and bias correction:**
   - 500 gene-label permutations within well, recomputing only the group sums;
   - report the observed value minus the null mean, with the null's 95 % range;
   - estimates on held-out cells (split by well).
4. **Real-signal criterion:** guide-replicated information, i.e. the information computed with one guide's cells predicting the gene profile defined by the other guides, against a null from pairs of different genes.
5. **Greedy forward selection:** at each step, add the column or family with the largest added information given old plus everything chosen so far (rank-1 or block updates). Stop at a user threshold or at the permutation-null level. The output is a ranked list.
6. **Optional nonlinear check:**
   - a small MLP predicting guide or gene from old, then from old + new; the held-out cross-entropy difference is a variational lower bound on the added information, in nats;
   - optionally a GPU kNN (KSG) estimator on a subsample.
   Both are off by default.
7. **Outputs:**
   - `feature_information.csv`: per column and per family, unique fraction, added bits with CI, null mean and range, guide-replicated bits, greedy rank;
   - a ranked bar chart (bars plus jittered permutation points);
   - a summary in the run report.
8. **Performance target:** a full Funk plate A (about 4.7 M cells × about 2,000 features) screened in minutes on one 3090 within 22 GB VRAM. There is a CPU fallback (same maths, numpy) with identical results to tolerance.
9. **Alpha and docs:** off by default, registered in ALPHA_FEATURES, no schema bump. 9-locale help, arrival contracts and catalogue records go in the same commit.

## Proof check (pre-set)
- **Exactness on synthetic data:**
  - Gaussian data with known conditional mutual information: the estimate matches the analytic value within 1 %;
  - a new column that is an exact linear combination of old columns gives 0 added bits (± null);
  - a pure-noise column gives null-level information.
- **CPU = GPU:** identical to 1e-6 relative on plate A A1.
- **Agreement with the slow standard:** on Funk plate A, compare the screen's family ranking with steps 2a–2c of the standard (cross-validated R² residuals, guide reproducibility, downstream held-out complexes and pairs). The rank correlation is reported; the pre-set expectation is Spearman ≥ 0.7 between added bits and downstream gain across families.
- **Generality:** the same run on PERISCOPE CP186A.
- **Controls:**
  - 500-permutation null;
  - shuffled-within-well copies of old features (expected about 0 added bits);
  - duplicate columns (expected 0 added bits).
- **Speed:** wall time on GPU against CPU on the same data.
