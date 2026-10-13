Hit List: exact real Regression outputs and an EXTERNAL companion window
=======================================================================

This archive contains unchanged results from the downloadable-data Regression
tutorial, plus hit_list_companion.py. It does not contain a full project,
acquired ground truth, selected discoveries or a repaired spaCR application.

The current Regression > Hits shortcut selects a page in a hidden container.
The tutorial shows that failure first. To follow the remaining demonstration,
activate your spaCR environment and run from this extracted directory:

    python hit_list_companion.py

The visibly labelled EXTERNAL window wraps the existing HitListScreen without
changing its readers, filters, tables, workers or exports. Click Browse and
select this archive's results directory. No regression or inference is run.
The helper is a tutorial companion, NOT a new built-in spaCR launch command.
Investigate integration is not wired here; do not use it as a demonstrated
route. Annotation imports and penalised-model ranking are outside this lesson.

The family contains 325 tested genes and 434 tested guide rows. Effects are
standardized marginal effects, not simultaneous multivariable coefficients.
Gene BH correction has minimum q=0.2708333333333333: no gene passes q <= .05.
The p-only significant CSV is preserved as an original file but is NOT a list
of multiple-testing-corrected discoveries. Gene and guide corrections use
separate families. Missing intervals are unavailable, not zero-width intervals.
The Hit List export's n_obs=0 is not the true well support: inspect the original
results_gene.csv wells_with_gene column instead (between 2 and 48 here).

Native double-clicks on table-header boundaries make clipped values readable.
The recorded filters are examples, not recommendations or statistical retests:
Max q=1 -> 325; Max q=.05 -> 0; restore 1 -> 325.
Min guides=2 -> 117; additionally Min agreement=1 -> 57.
Additionally Hide controls -> 56; restore controls -> 57.
With all other filters cleared, Direction=up -> 162; Min |effect|=.1 -> 16.
Search, clear it and restore the other controls to recover the original list.

For the recorded exports, Max q=1, Min guides=2, Min agreement=1, Direction=any,
Min |effect|=0, controls INCLUDED, and the search is empty. CSV, Markdown and
HTML each contain all 57 rows in this example. Markdown/HTML round displayed
numbers; CSV is appropriate for later numerical work. These exports do not
carry every filter argument: save the choices separately with the files.
Keep originals and exported subsets distinct. Agreement in direction alone
does not establish significance, causality or validation.

The original files are byte-checked in source_manifest.json. No live API key,
remote AI provider, GPU operation, result rewriting or publication is needed.
Python users can use spacr.hits.build_hit_list and HitList.filter/write_csv/
to_markdown/to_html; remember the API's default Markdown/HTML row limits when
exporting larger lists. The GUI Markdown action explicitly exports its whole
current list, while the HTML formatter has a 500-row cap.
