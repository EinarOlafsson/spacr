# 1012fix readout

**Overall: FAIL. Keep 1012 off nightly.** The no-plant p < 0.05 share is 187/7280 (2.56868%), below the required 3–7%. No parameter was tuned.

the plant survives into PCA; the matched graph loses it

No input or PCA routing defect was found. Graph construction and the original pipeline arms are unchanged. At 8x, target-view top-ten partner hits are 376/520 (72.31%) after PCA versus 4/520 (0.77%) without planting. The paper audit of the saved matched graph reports 14/520 (2.69%) versus 2/520 (0.38%). These representations are reported separately.

## Primary acceptance

| Criterion | Result | Evidence |
|---|---|---|
| Pooled 8x paired mAP excess, complex-bootstrap 95% lower bound > 0 | PASS | Planted 0.0095855027325; no-plant 0.00206527273288; excess 0.00752022999961; CI [0.00363228335953, 0.0126098107787] |
| Positive Spearman trend over 52 view/strength strata, p < 0.05 | PASS for both reported metrics | mAP rho 0.563863980447, p 1.34343621556e-05; target-view pair-share rho 0.287916528879, p 0.0384731772374 |
| No-plant p < 0.05 share within [0.03, 0.07] | FAIL | 187/7280 = 0.025686813186813186 |

The specified Spearman criterion passes. The pooled mAP means are not strictly monotone: the 2x mean exceeds the 4x mean. Both facts are explicit below. The no-plant calibration criterion uses p < 0.05 across all views; the primary pair-share metric additionally requires the winning view to equal the target view.

## Fallback method and evidence

FALLBACK was used. The original summary stores shape, dtype and SHA only for p/view; the original harness deletes the numerical result. No original numerical p/view file exists in the saved acceptance folders. The input matrices were reconstructed from the exact eight approved inputs and original adapter; all eight hashes and 260 planted-direction records match. This is a readout calculation, not a pipeline arm rerun.

Each view uses the existing population column-standardization owner, fills missing standardized features with zero, then uses the existing row L2-normalization owner. This is the **fallback missing-value rule, not in sim.py**. A gene with no profile contributes p = 1 in that view. Coverage flags were verified to equal non-missing profile presence.

For each view and arm, rng = default_rng(0) draws two independent sets of 200,000 indices, uniformly with replacement from covered genes outside every planted set. Self-pairs are retained. The empirical upper-tail p is mean(null >= observed), without a +1 correction. The pair p is min over views of min(1, 13*p_view); the first winning view is retained on ties.

The primary pair share uses p < 0.05 AND winning view = target view. The existing AP owner ranks all 8,139 other genes by ascending corrected p, at single precision, with tied partners first. Each member has seven partners. The blocked query adapter was byte-equal to the unchanged full-square owner on a reordered fixture with ties. Pooled 8x excess uses the 65 paired complex means, with the existing bootstrap owner, 2,000 draws and seed 0.

Random controls are matched position by position on the number of non-missing views, without replacement within each set, outside every planted set. rng = default_rng(0) resets per stratum. Each five-template stratum cycles templates 100 times for 500 sets, apart from the approved exception below.

## Pooled strength readout

| Strength | Planted target-view pair share | No-plant target-view pair share | Planted mAP | No-plant mAP |
|---|---:|---:|---:|---:|
| 1x | 0.2747253% | 0.1648352% | 0.00212224565232 | 0.00182622532787 |
| 2x | 0.7142857% | 0.1648352% | 0.00512587380601 | 0.00210047812187 |
| 4x | 0.8241758% | 0.3296703% | 0.00355579307396 | 0.00214953719712 |
| 8x | 3.736264% | 0.2747253% | 0.0095855027325 | 0.00206527273288 |

Pooled 8x target-view pairs: planted 68/1820 (3.73626%) versus no-plant 5/1820 (0.274725%). Pooled random controls average the 13 view strata for each of 500 draws; the paired JSON also gives the distribution of all 6,500 individual sets.

| Pooled 8x random metric | Mean [2.5–97.5% range of 500 pooled view means] |
|---|---|
| pair_share | 0.2587912 [0, 1.098901] % |
| map | 0.002639892 [0.001573761, 0.006328715] |

## Every primary stratum

Each row contains five planted complexes of eight members. Pair shares require the target winning view.

| View | Strength | Planted pair share (%) | No-plant pair share (%) | Planted mAP | No-plant mAP |
|---|---:|---:|---:|---:|---:|
| transcript | 1x | 0.7142857 | 0.7142857 | 0.00215965617 | 0.00177902038 |
| transcript | 2x | 0 | 0 | 0.00157311376 | 0.00158566347 |
| transcript | 4x | 0 | 0 | 0.00179400122 | 0.00174289745 |
| transcript | 8x | 0 | 0 | 0.00322740713 | 0.00168141752 |
| translation | 1x | 0 | 0 | 0.0018365303 | 0.00182200107 |
| translation | 2x | 0 | 0 | 0.00202682196 | 0.00165865013 |
| translation | 4x | 1.428571 | 1.428571 | 0.00403489422 | 0.0020989308 |
| translation | 8x | 2.142857 | 0.7142857 | 0.00706599696 | 0.001723031 |
| chromatin | 1x | 1.428571 | 0.7142857 | 0.0026698967 | 0.00198759685 |
| chromatin | 2x | 1.428571 | 0 | 0.0103912304 | 0.00372395694 |
| chromatin | 4x | 0.7142857 | 0.7142857 | 0.00989046491 | 0.00358892403 |
| chromatin | 8x | 12.85714 | 0.7142857 | 0.0216295084 | 0.00178417022 |
| protein_abundance | 1x | 0 | 0 | 0.00165095604 | 0.00152177651 |
| protein_abundance | 2x | 0.7142857 | 0 | 0.00220398205 | 0.00200889728 |
| protein_abundance | 4x | 0.7142857 | 0.7142857 | 0.00217457159 | 0.00247600232 |
| protein_abundance | 8x | 0 | 0 | 0.00284491735 | 0.00217516808 |
| fitness | 1x | 0 | 0 | 0.0020447166 | 0.00197099614 |
| fitness | 2x | 0 | 0 | 0.00199082994 | 0.00194200719 |
| fitness | 4x | 0 | 0 | 0.0026561365 | 0.00159287861 |
| fitness | 8x | 0 | 0 | 0.0069558192 | 0.00269772094 |
| place | 1x | 0.7142857 | 0 | 0.00221734978 | 0.00177167114 |
| place | 2x | 5 | 0 | 0.0303926887 | 0.00214904954 |
| place | 4x | 0.7142857 | 0 | 0.0024705872 | 0.00175227069 |
| place | 8x | 0.7142857 | 0 | 0.00507391745 | 0.00341376233 |
| interaction | 1x | 0 | 0 | 0.00184075545 | 0.0017202696 |
| interaction | 2x | 0 | 0 | 0.00186014272 | 0.00178596285 |
| interaction | 4x | 0.7142857 | 0 | 0.00227087458 | 0.00182503438 |
| interaction | 8x | 0 | 0.7142857 | 0.00283008959 | 0.00217820594 |
| sequence | 1x | 0 | 0 | 0.00183706142 | 0.00161502783 |
| sequence | 2x | 0 | 0 | 0.00201346578 | 0.00186651816 |
| sequence | 4x | 0 | 0 | 0.00220798674 | 0.00176635343 |
| sequence | 8x | 15.71429 | 0.7142857 | 0.0450434711 | 0.00219308014 |
| structure_computed | 1x | 0 | 0 | 0.00163272438 | 0.00160952049 |
| structure_computed | 2x | 0 | 0 | 0.00294591782 | 0.00251423287 |
| structure_computed | 4x | 0 | 0 | 0.00218824927 | 0.00184378536 |
| structure_computed | 8x | 0 | 0 | 0.00287861207 | 0.00169165723 |
| conservation | 1x | 0 | 0 | 0.00162263889 | 0.00149343802 |
| conservation | 2x | 0 | 0 | 0.0027886261 | 0.00184300273 |
| conservation | 4x | 0 | 0 | 0.00242557917 | 0.00194616371 |
| conservation | 8x | 7.142857 | 0 | 0.0126123211 | 0.00161230235 |
| esm | 1x | 0.7142857 | 0.7142857 | 0.00273684639 | 0.00166890258 |
| esm | 2x | 0.7142857 | 0.7142857 | 0.00189930222 | 0.00188971002 |
| esm | 4x | 0.7142857 | 0.7142857 | 0.00330595775 | 0.00195666018 |
| esm | 8x | 0 | 0 | 0.00172454462 | 0.00159274918 |
| secondary_structure | 1x | 0 | 0 | 0.00321329738 | 0.00303045754 |
| secondary_structure | 2x | 0 | 0 | 0.00299552635 | 0.00227377565 |
| secondary_structure | 4x | 0.7142857 | 0 | 0.0062982799 | 0.00326270105 |
| secondary_structure | 8x | 1.428571 | 0.7142857 | 0.0050007589 | 0.00207096931 |
| 3di | 1x | 0 | 0 | 0.00212676398 | 0.00175025111 |
| 3di | 2x | 1.428571 | 1.428571 | 0.00355471166 | 0.00206478877 |
| 3di | 4x | 5 | 0.7142857 | 0.00450772691 | 0.00209138155 |
| 3di | 8x | 8.571429 | 0 | 0.0077241716 | 0.00203431128 |

## Every matched-random stratum

Ranges are empirical 2.5–97.5 percentiles of sets, not confidence intervals for the mean.

| View | Strength | Sets | Target-view pair share (%), mean [range] | mAP, mean [range] |
|---|---:|---:|---:|---:|
| transcript | 1x | 500 | 0.4142857 [0, 3.571429] | 0.002905981 [0.00125803, 0.01217113] |
| transcript | 2x | 500 | 0.3571429 [0, 3.571429] | 0.002339084 [0.001261935, 0.007038896] |
| transcript | 4x | 500 | 0.3214286 [0, 3.571429] | 0.002898816 [0.001261605, 0.0141894] |
| transcript | 8x | 500 | 0.2428571 [0, 3.571429] | 0.002565046 [0.001233809, 0.009093473] |
| translation | 1x | 500 | 0.2714286 [0, 3.571429] | 0.002736243 [0.001282271, 0.0115101] |
| translation | 2x | 500 | 0.2857143 [0, 3.571429] | 0.002564369 [0.001232735, 0.008678819] |
| translation | 4x | 500 | 0.3214286 [0, 3.571429] | 0.002472819 [0.001269281, 0.008063262] |
| translation | 8x | 500 | 0.3142857 [0, 3.571429] | 0.002645904 [0.001257536, 0.01044757] |
| chromatin | 1x | 500 | 0.3071429 [0, 3.571429] | 0.002557997 [0.00125407, 0.01019856] |
| chromatin | 2x | 500 | 0.3928571 [0, 3.571429] | 0.002644443 [0.001264039, 0.008796996] |
| chromatin | 4x | 500 | 0.3857143 [0, 3.571429] | 0.00269093 [0.00124012, 0.00951916] |
| chromatin | 8x | 500 | 0.3142857 [0, 3.571429] | 0.002636685 [0.00127785, 0.01094205] |
| protein_abundance | 1x | 500 | 0.3571429 [0, 3.571429] | 0.00277989 [0.001295847, 0.008291961] |
| protein_abundance | 2x | 500 | 0.2071429 [0, 3.571429] | 0.00247348 [0.001258506, 0.009282375] |
| protein_abundance | 4x | 500 | 0.2714286 [0, 3.571429] | 0.002588868 [0.001225786, 0.009006111] |
| protein_abundance | 8x | 500 | 0.3214286 [0, 3.571429] | 0.00261222 [0.001244322, 0.01011069] |
| fitness | 1x | 500 | 0 [0, 0] | 0.002415714 [0.001238784, 0.008261877] |
| fitness | 2x | 500 | 0 [0, 0] | 0.002570304 [0.001267219, 0.008700566] |
| fitness | 4x | 500 | 0 [0, 0] | 0.002162331 [0.001243359, 0.007030294] |
| fitness | 8x | 500 | 0 [0, 0] | 0.002746384 [0.001269422, 0.01086807] |
| place | 1x | 500 | 0.007142857 [0, 0] | 0.002327051 [0.001278981, 0.007522207] |
| place | 2x | 500 | 0.02857143 [0, 0] | 0.002538315 [0.001267414, 0.007499676] |
| place | 4x | 500 | 0.02142857 [0, 0] | 0.002764225 [0.001287734, 0.01149709] |
| place | 8x | 500 | 0.02142857 [0, 0] | 0.002663395 [0.001285522, 0.007993824] |
| interaction | 1x | 500 | 0.9714286 [0, 10.71429] | 0.002779231 [0.001271538, 0.009600961] |
| interaction | 2x | 500 | 1.278571 [0, 10.71429] | 0.002723911 [0.001231994, 0.01144225] |
| interaction | 4x | 500 | 1.214286 [0, 10.71429] | 0.002903566 [0.001288891, 0.01237223] |
| interaction | 8x | 500 | 0.7928571 [0, 5.446429] | 0.00255918 [0.001235813, 0.007497041] |
| sequence | 1x | 500 | 0.2 [0, 3.571429] | 0.002668519 [0.001242785, 0.0110663] |
| sequence | 2x | 500 | 0.25 [0, 3.571429] | 0.002436568 [0.001241479, 0.007956364] |
| sequence | 4x | 500 | 0.2928571 [0, 3.571429] | 0.002485515 [0.001260055, 0.008628235] |
| sequence | 8x | 500 | 0.2857143 [0, 3.571429] | 0.002676591 [0.001253268, 0.009067265] |
| structure_computed | 1x | 500 | 0 [0, 0] | 0.002800692 [0.001252121, 0.0131497] |
| structure_computed | 2x | 500 | 0 [0, 0] | 0.002519896 [0.001260088, 0.00935901] |
| structure_computed | 4x | 500 | 0 [0, 0] | 0.002442389 [0.001261628, 0.007799407] |
| structure_computed | 8x | 500 | 0 [0, 0] | 0.002899431 [0.00123263, 0.0150247] |
| conservation | 1x | 400 | 0.3928571 [0, 3.571429] | 0.002372346 [0.001233864, 0.007083058] |
| conservation | 2x | 500 | 0.4 [0, 3.571429] | 0.002921555 [0.001259627, 0.0123794] |
| conservation | 4x | 500 | 0.3642857 [0, 3.571429] | 0.002463262 [0.001203867, 0.006827437] |
| conservation | 8x | 500 | 0.4071429 [0, 3.571429] | 0.002502464 [0.001246222, 0.00797297] |
| esm | 1x | 500 | 0.3428571 [0, 3.571429] | 0.002511842 [0.001262329, 0.009698381] |
| esm | 2x | 500 | 0.2428571 [0, 3.571429] | 0.002491535 [0.001259162, 0.007808561] |
| esm | 4x | 500 | 0.2428571 [0, 3.571429] | 0.002608902 [0.001264863, 0.009351966] |
| esm | 8x | 500 | 0.2714286 [0, 3.571429] | 0.002609035 [0.001241375, 0.008344691] |
| secondary_structure | 1x | 500 | 0.25 [0, 3.571429] | 0.002513645 [0.001257805, 0.00782248] |
| secondary_structure | 2x | 500 | 0.2428571 [0, 3.571429] | 0.002666972 [0.001226786, 0.009161076] |
| secondary_structure | 4x | 500 | 0.15 [0, 3.571429] | 0.002478626 [0.001247039, 0.007927263] |
| secondary_structure | 8x | 500 | 0.2142857 [0, 3.571429] | 0.002804905 [0.001283311, 0.01222153] |
| 3di | 1x | 500 | 0.3071429 [0, 3.571429] | 0.002619202 [0.001279461, 0.008844215] |
| 3di | 2x | 500 | 0.2214286 [0, 3.571429] | 0.002515249 [0.00125364, 0.00952853] |
| 3di | 4x | 500 | 0.1571429 [0, 3.571429] | 0.002489301 [0.001228073, 0.008562994] |
| 3di | 8x | 500 | 0.1785714 [0, 3.571429] | 0.00239736 [0.001271808, 0.006625539] |

## Approved coverage exception and sensitivity

1 of 5 templates in conservation 1x has no coverage-matched null (its coverage-7 member is the only coverage-7 gene and is planted); 400 random sets instead of 500. Those 400 cycle through the four feasible templates, with seed 0. No planted gene is reused and no main-analysis coverage bin or replacement rule changes.

**SENSITIVITY ONLY; excluded from PASS:** plant_conservation_s1_0 gets 100 random sets with its coverage-7 member matched to coverage 8, the nearest eligible pool of 133 genes. All other matching positions keep their exact coverage counts. Measurements and pair scores are unchanged.

Observed excluded template: planted mAP 0.0013792953109; paired no-plant mAP 0.00129074240554; target-view pair share 0 for both. Sensitivity random mAP: 0.002358698 [0.001241152, 0.004402506]; target-view pair share (%): 0.3214286 [0, 3.571429]. No primary gate changed.

## Secondary saved-partition readout

Kept together means at least two members share a cluster, via the existing complex_metrics partner metric. The 500 full-vector label permutations preserve every cluster size; the seed is SWEEP_CONTROL_SEED. No partition or graph was rebuilt.

| Arm | Strength | Kept / 65 | Share (%) | Null mean (%) | Null 95% range (%) | One-sided permutation p |
|---|---:|---:|---:|---:|---:|---:|
| no_plant | 1x | 4 | 6.153846 | 4.815385 | 0–10.76923 | 0.365269461 |
| no_plant | 2x | 3 | 4.615385 | 4.947692 | 0–10.76923 | 0.626746507 |
| no_plant | 4x | 2 | 3.076923 | 4.944615 | 0–10.76923 | 0.842315369 |
| no_plant | 8x | 1 | 1.538462 | 4.849231 | 0–10.76923 | 0.9500998 |
| planted | 1x | 2 | 3.076923 | 4.873846 | 0–10.76923 | 0.820359281 |
| planted | 2x | 3 | 4.615385 | 5.095385 | 0–10.76923 | 0.668662675 |
| planted | 4x | 5 | 7.692308 | 4.870769 | 0–10.76923 | 0.209580838 |
| planted | 8x | 7 | 10.76923 | 4.836923 | 0–10.76923 | 0.0459081836 |

## Size-aware recovery

The existing Fisher owner now offers an enrichment tail while retaining its two-sided default. Recovery requires at least two members in a cluster with one-sided Fisher p, Bonferroni-corrected across all nonempty scored clusters, below 0.05. It uses distinct set members and conditions on background and cluster sizes.

| Shuffled set size | Background genes | Clusters | Random sets | Recovered (%) |
|---|---:|---:|---:|---:|
| 4 | 10000 | 10 | 20000 | 3.755 |
| 8 | 6400 | 640 | 20000 | 4 |

The 48 focused clustering/enrichment tests and three source-contract tests passed. Full items: 1,599 passed plus 960 subtests, 10 skips. Full settings: 630 passed plus 369 subtests, two skips. All three native source consumers passed with zero new production imports. Final source-rule results are recorded in acceptance.json.

## First diagnostic receipts

The first input geometry diagnostic was logged before any source change or rescoring. It took 0.896 seconds; the unchanged PCA trace took 2.014 seconds. The exact per-view geometry and PCA values are in input-cosine-diagnostic.json and embedding-cosine-diagnostic.json. The 8/sqrt(d) expression is a large-d direction-alignment approximation; bounded low-dimensional references are reported separately, not treated as pair cosines.

The JSON receipts retain all per-complex metrics, 52 stratum summaries, null draw details, source and input hashes, the approved reporting exception, and the sensitivity results. Frozen analysis source is included for review. Persisted numerical pair arrays stay outside the branch in the recovery workspace.
