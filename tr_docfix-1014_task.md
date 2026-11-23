# docfix-1014 translation and API review task

Base source: `4eaa2008bb972e613040e74d5f738d07ab54b622` (nightly before train 13 landing).
Authority: Claude WORK_TASKS.md task 0z, 2026-11-22 19:10.

Three explanatory paragraphs move above the reStructuredText parameter field lists. All words and all executable code are preserved. The new paragraph boundaries require fresh review of the affected API records. Do not reuse an old whole-body hash for the revised layout.

The exact three revised bodies, signatures, prior hashes, new hashes, canonical translation blocks and layouts are in `tests/data/release_contracts/docfix_1014_api_changes_2026-11-22.json`. Historical arrival JSON files and generated translation records are unchanged. The existing item-1014 adapter composes these three revisions after the S2J adapter, so all three protected consumers receive the new sources through their existing hooks.

No new runtime message, setting, callable or parameter is introduced. Home owns the nine-language records and their review.

## spacr.ops_decode.threshold_settings

Source: `spacr/ops_decode.py:701`

Previous source SHA-256: `172cba2e797e4a0bfe195f58f5e1c5aeb8eacb113d2bdedf9f0a5c517ef6c6b4`

Current source SHA-256: `cc947d1628e9e8f41cedae8b0ac81c92c5ff389c2c38de8e5a874e2ad47d348a`

```rst
The decode threshold mode and, in auto mode, its grid and target.

Manual never-barcode calibration uses a single setting, preserving the
entered thresholds. Without decoys it requests the reported fallback.

:param settings: the run's settings, read for
    ``ops_decode_threshold_mode``, ``ops_decode_fdr_max``,
    ``ops_decode_auto_grid``, ``ops_decode_auto_spot_grid``,
    ``ops_decode_auto_tune_fraction``, ``ops_decode_auto_seed`` and the
    shared tuner's ``ops_decode_auto_ci``, ``ops_decode_auto_objective``,
    ``ops_decode_auto_f_beta`` and ``ops_decode_auto_unreachable``, and
    ``ops_decode_fdr_calibration`` and the shared controls
    ``ops_never_barcode_k``, ``ops_never_barcode_seed`` and
    ``ops_never_barcode_sets``.
:param rule: from :func:`decode_rule`; a grid without the pixel quality,
    or a rule without the pixel fallback, keeps ``pixel_quality``.
:returns: ``mode``; in auto mode or manual never calibration also ``floors`` and ``pixel_qualities``
    (sorted), ``fdr_max`` (per 100 calls), ``ci``, ``objective``,
    ``f_beta``, ``unreachable``, ``tune_fraction``, ``seed`` and
    ``spot_thresholds`` (the read thresholds tried jointly, sorted; empty
    when the read threshold stays as entered); with
    a calibration method also ``fdr_calibration``.
:raises ValueError: for an unknown mode or FDR calibration, an FDR target that is not
    above 0, a tuning fraction outside (0, 1), a grid that is not a
    mapping of the two thresholds to lists of numbers 0 or above, or
    auto mode without decoys (``ops_decoy_library_k`` 0).
```

## spacr.ops_decode.plate_fdr_summary

Source: `spacr/ops_decode.py:2306`

Previous source SHA-256: `d3e4bc6058eb0e30d4704dcc036f296d7e52998eb77d96d9eaca1172185ab752`

Current source SHA-256: `3135404d7298e05b9d97353aa05b6bdcb44e850a525619a116e8f995b129b0e5`

```rst
Assigned percentages and FDR across all field statistics for the plate.

Per-well never reports provide their applied scales; the plate FDR
and its existing field bootstrap use those same scales. Fixed and
shuffle-only plates retain their previous summary calculation.

:param folder: the ``results/ops_decode`` folder holding
    ``field_stats_<well>.csv`` files.
:param plate: the plate.
:param k: decoys per guide.
:param scale: the FDR calibration.
:param reps: field-bootstrap replicates for the 95 % FDR interval.
:param seed: the bootstrap seed.
:returns: the summary, with per-well rows; empty wells when no stats
    were written.
```

## spacr.ops_decode.write_well_reports

Source: `spacr/ops_decode.py:2387`

Previous source SHA-256: `b60a898d64c8391a419dfc1b06918647c658d4946bf75b80f17d9619c7d62280`

Current source SHA-256: `40355e9b0e51ad9dcd6a17c2d8bc0a50768a5bed57f05ccd670c71422ff56da2`

```rst
Write field statistics, FDR report and plate summary for one well.

Never calibration writes its JSON and CSV through the same report
writer used by the prefix path, with point and interval fields folded
into this report and the per-well plate summary.

:param db: the measurements database.
:param plate: the plate.
:param well: the well.
:param decoded: the decoded fields, each with ``site`` and ``rule_a``
    from :func:`field_calls`.
:param rule: from :func:`decode_rule`; after auto thresholds with a
    shuffle calibration (item 949) its ``fdr_scale`` is the calibrated
    scale and its ``threshold_choice`` holds the calibration.
:param skipped_sites: the unplaced sites.
:returns: the FDR report for one well, with the paths written under ``paths``;
    with a shuffle calibration it adds ``fdr_calibration``,
    ``fdr_scale_entered``, ``shuffle_called`` for the whole well,
    ``shuffle_call_rate_pct``, ``shuffle_ci95``, ``shuffle_limit_pct``
    (1.5 times the calibrated FDR) and ``shuffle_le_limit``.
```
