Export measurements to AnnData
==============================

Create an ``.h5ad`` file containing measured features and object metadata for
analysis in AnnData or Scanpy. Start with a spaCR project containing
``measurements/measurements.db``. For a worked example, download the project
from the `Annotate tutorial <tutorials/#lesson=09_annotate>`__.

Export from the application
---------------------------

1. From **Home**, open **Measure**, then **AnnData Export**.
2. Select the measured project as the source.
3. Set **Anndata out** to a new ``.h5ad`` filename. Leaving it empty writes
   beneath the project's ``results`` folder. A successful export replaces an
   existing file at the chosen path; use separate names to keep both versions.
4. Set **Anndata single table** to ``cell`` for one row per cell using that
   table's features. Use ``nucleus`` or ``pathogen`` for those object types.
   Leave it empty for a cell-level joined export, where child measurements
   are aggregated onto their parent cells.
5. Choose **Anndata nan policy**, then click **Run**. Read the completion
   message and open the output file to check its shape and feature names.

Choose how to handle missing features
--------------------------------------

* ``keep`` retains missing values. Choose this when you want to decide how to
  handle them in the downstream analysis.
* ``drop_features`` removes feature columns containing missing values.
* ``drop_objects`` removes object rows containing missing feature values.
* ``mean`` replaces missing values with the observed feature mean.
* ``zero`` replaces missing values with zero. Use it only when zero is an
  appropriate value for the intended analysis.

The imputing policies retain the original missing positions in
``layers['missing']``. These choices apply to the feature matrix, not to
unknown metadata such as unavailable calibration values. Check object and
feature counts after choosing a dropping policy. Scanpy operations such as
scaling and PCA need missing feature values to be handled before use.

Inspect the output
------------------

``X`` contains the object-by-feature matrix, ``obs`` contains object metadata,
and ``var`` describes the features. Keep the source images and their paths:
image pixels are not embedded in the export. With the Annotate example, a
cell-only export using ``keep`` has 2,341 objects, 261 features and 28 missing
feature values.

Export from Python
------------------

The following uses the same entry point as the application's **Run** button:

.. code-block:: python

   import anndata as ad
   from spacr.anndata_export import run_anndata_export

   result = run_anndata_export({
       "src": "/path/to/project",
       "anndata_out": "/path/to/project/results/cells_keep.h5ad",
       "anndata_single_table": "cell",
       "anndata_nan_policy": "keep",
   })
   exported = ad.read_h5ad(result.path)
   print(exported.shape)
   print(exported.obs.head())

For explicit function arguments, see
:func:`spacr.anndata_export.export_anndata`.

Follow the video
----------------

The `AnnData video <tutorials/#lesson=59_anndata_export>`__ walks through
these controls using the downloadable measurement example. It shows joined,
cell-only and nucleus-only exports and compares the missing-value policies.
The normal exporter handles entirely missing metadata directly. If writing a
replacement fails, an existing completed output is preserved.
