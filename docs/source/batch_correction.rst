Plate and batch-effect correction
=================================

spaCR can normalize acquisition-batch effects before Image UMAP, Classify
(ML), and phenotype regression. Correction is disabled by default because a
plate effect can be real biology when treatments and plates are confounded.

Choosing a method
-----------------

``none``
   Leave measurements unchanged.

``control_center``
   Recommended when every plate contains the same negative/reference control.
   spaCR estimates the median reference shift on each plate and subtracts only
   that shift. Treatment dispersion and treatment-to-control differences are
   retained.

``robust_zscore``
   Align plate medians and median absolute deviations. This is resilient to
   outliers, but can remove genuine differences if biological conditions are
   unevenly distributed across plates.

``center``
   Align plate means while preserving the overall mean.

``zscore``
   Align both plate means and standard deviations. This is the strongest
   correction and should be used only when plate composition is comparable.

Settings
--------

Set ``batch_correction`` to a method above and use ``batch_column`` to name the
batch identifier (normally ``plateID``). ``batch_min_samples`` rejects
unstable estimates from undersized batches.

For ``control_center``, ``batch_control_column`` identifies the metadata field
that contains the controls and ``batch_control_values`` selects one or more
reference values. When blank, Image UMAP follows ``col_to_compare``/``neg``
and Classify (ML) follows ``location_column``/``negative_control``.
Regression requires an explicit reference value.
``batch_missing_control=error`` is the safe default;
``skip`` leaves an affected plate unchanged and records a warning.

Every regression correction writes ``batch_correction.json`` next to the
regression outputs. Image UMAP and Classify print before/after batch-centroid
spread in the run log, while the reproducibility manifest records the exact
settings used.

Python API
----------

The shared implementation is :func:`spacr.batch_correction.correct_batch_effects`.
Most integrations use
:func:`spacr.batch_correction.correct_from_metadata`, which ensures metadata
columns never enter the numeric feature matrix.

.. code-block:: python

   from spacr.batch_correction import correct_from_metadata

   corrected, report = correct_from_metadata(
       features,
       metadata,
       batch_correction="control_center",
       batch_column="plateID",
       batch_control_column="columnID",
       batch_control_values="c1",
   )
   print(report.to_dict())
