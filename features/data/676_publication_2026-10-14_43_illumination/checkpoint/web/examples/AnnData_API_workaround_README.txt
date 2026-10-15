AnnData tutorial: explicit missing-metadata storage workaround
============================================================

Extract AnnData_API_workaround.zip into a NEW directory. The archive contains
the exact Python helper used in the tutorial and this README, not microscopy
data. Use the real downloaded measurements example from Annotate. Keep the
original database and images untouched; put a COPY of measurements.db beside
the extracted script. Activate your spaCR Python environment first.

The unmodified GUI exporter is NOT fixed by this example. Some all-missing
object-typed observation metadata cannot be written by the tested writer.
The helper calls spacr.anndata_export.build_anndata and explicitly encodes
only those entirely missing metadata columns as empty categories. Missing
values remain missing, and unknown voxel calibration is NOT invented. This
encoding does not change X. Mean imputation below is a separate policy.

Run one or more commands from the new directory:

python anndata_missing_metadata_example.py --source measurements.db --out results/joined_keep.h5ad --nan-policy keep
python anndata_missing_metadata_example.py --source measurements.db --out results/cell_keep.h5ad --single-table cell --nan-policy keep
python anndata_missing_metadata_example.py --source measurements.db --out results/cell_mean.h5ad --single-table cell --nan-policy mean
python anndata_missing_metadata_example.py --source measurements.db --out results/cell_drop_features.h5ad --single-table cell --nan-policy drop_features
python anndata_missing_metadata_example.py --source measurements.db --out results/cell_drop_objects.h5ad --single-table cell --nan-policy drop_objects
python anndata_missing_metadata_example.py --source measurements.db --out results/nucleus_keep.h5ad --single-table nucleus --nan-policy keep

The helper refuses an existing destination. It writes and reopens a temporary
file, compares X, ordered identities and observation metadata, then installs
the new output without overwriting an existing file. It checks that the
source database hash is unchanged. Never treat a partial file from a failed
GUI/API operation as a completed export.

Recorded example (your own data may have different sizes):
joined/keep:        2341 x 1136, 400320 missing feature values
cell/keep:          2341 x 261, 28 missing feature values
cell/mean:          2341 x 261, 28 values imputed; missingness layer retained
cell/drop_features: 2341 x 257, four feature columns removed
cell/drop_objects:  2334 x 261, seven cell rows removed
nucleus/keep:       2682 x 341, 1912 missing feature values

All six files also passed an independent SQLite comparison in the tutorial
recording. This helper is not that independent oracle and does not validate
segmentation or biology. Choose missing-value policies for your analysis;
none is universally recommended. Labels remain metadata, not training
features. Images are not embedded: keep originals and paths. No UMAP, GPU
operation or AI request is performed. Keep versions, commands and settings.
