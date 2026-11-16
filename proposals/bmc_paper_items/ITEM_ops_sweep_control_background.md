# Item 959: spaCR's own cluster-sweep rows use the same Fisher background as the published-comparator rows

- **Number:** 959 (reserved by master 2026-11-16). **Maintainer decision:** "Same background, new item", following item 953.
- **Gap:** since item 953, published-comparator rows (`score_published`) count the comparator's non-targeting/control rows in the CORUM/KEGG Fisher background, which reproduces Funk's published 24.3 % / 13.5 % exactly. spaCR's own sweep rows (`score_partitions`) still use a targeting-only background, so one table mixes two backgrounds.
- **Order:** touches spacr/ops_clusters.py, as 948 does. Master builds it after 948 lands.

## Spec
1. **New setting** `ops_cluster_sweep_controls_background` (bool, default True). With True, `score_partitions` adds the run's non-targeting/control rows to the Fisher background of the CORUM/KEGG enrichment, exactly as `score_published` does since 953 (one shared helper, no second copy). Control rows are never cluster members, and complex and STRING metrics are unchanged.
2. **Schema:** a bump to the next free version at merge (read max(DEFAULT_CHANGES) on nightly), plus `DEFAULT_CHANGES[n] = {'ops_cluster_sweep_controls_background': False}`, so older settings files keep the targeting-only background.
3. **Records:** each sweep row records the background size and `control_rows_in_background`. The table legend states one background for all rows.
4. **Docs and strings:** a 9-locale tooltip and help; updated docstrings and catalogue records in the same commit.

## Proof check (pre-set)
- **Sweep rows recomputed:** on the plate A/B card-C+ inputs (`BMCspacr/reanalysis/funk_published_rows/`) and the stage-5 proof inputs, every CORUM/KEGG share is shown old vs new side by side in a table in the proof report.
- **Unchanged by construction:** STRING precision/recall/tp/fp/fn and all complex metrics are byte-identical old vs new.
- **Off-switch:** with `ops_cluster_sweep_controls_background=False`, outputs are byte-identical to nightly aa7ef4fc19.
- **Consistency:** in one table, spaCR rows and the published Funk row now share the same background size.
- **Permuted controls** (maintainer rule): the mean of 500 permutations with 95 % CI includes 0 for excess-type metrics. Fisher shares are reported as mean and range against the real row, and the single-seed control falls inside the 95 % permutation range.
- **Reporting rule:** any comparator value is shown as computed and as published (with source), or "n/a in paper".
