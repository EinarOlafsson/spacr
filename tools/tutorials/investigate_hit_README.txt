Investigate Hit — real screen example
=====================================

Everything in this archive comes from one real pooled CRISPR screen. It is not
simulated or synthetic data.

regression_run/
    A complete Regression run on Regression's own "Load test data" tables,
    with the settings used in the Regression tutorial: nonparametric guide
    permutation, 199 permutations, seed 0, guide and gene levels, analysis unit
    well, minimum guide support 2 wells, annotation source none. Hit List
    reads regression_run/results/guide_permutation. Its regression_data.csv is
    the per-well guide fraction table Investigate Hit uses.

cv_predictions.csv
    The real per-cell CV predictions for the kept cells, copied from the
    Regression test score tables. pred is the phenotype score; prcfo is the
    cell's plate_row_column_field_object key; path is the crop name used when
    the cells were scored.

measurements/measurements.db
    Real per-cell measurements of the same cells from the screen's plate
    databases: a cell table with shape, size and mean channel intensities.
    Crop images are not included, so the blinded review gallery stays empty
    and Open candidate crops has nothing to show.

To keep the download small, the database only holds the wells this
investigation needs. These are every analysed well containing a guide for gene
225160, plus three wells from the same plate for each of them, sampled at
random with a fixed seed from the analysed wells with no 225160 guide.
example_manifest.json lists the wells, the source files' SHA-256 digests and
the per-well checks. The mean score and cell count of every kept well equal the
values in the Regression run.

No gene in this quick Regression run passes q <= 0.05. With 199 permutations
the smallest possible permutation P value is 0.005, and 225160's q-value is
about 0.27. Treat the investigation as a worked example of the evidence, not
as a confirmed hit.

Running Investigate Hit writes its tables to
regression_run/results/guide_permutation/hit_investigation/225160/ and stores
a versioned attribution run in measurements.db. Keep an unzipped copy if you
want to start again from the original files.
