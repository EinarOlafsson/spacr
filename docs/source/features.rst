Capabilities
============

spaCR follows an image-based screen from raw microscopy files to a ranked hit
list. This page gives the full map; the :doc:`Python API quickstart
<python_api>` and interactive tutorials show the individual routes.

Core screen workflow
--------------------

Mask
~~~~

Mask prepares TIFF, OME-TIFF, LIF, CZI and ND2 acquisitions and segments
cells, nuclei, pathogens and organelles with Cellpose. It supports 2-D,
volumetric and time-series data, estimates object diameter, and can exchange
mask corrections with the layer viewer or napari.

Measure
~~~~~~~

Measure writes per-object morphology, intensity, texture, radial, spatial and
colocalization features to ``measurements.db``. It can save classifier-ready
object crops, estimate illumination correction from a plate, restrict work to
a region of interest and report segmentation quality before a run.

Annotate and Classify
~~~~~~~~~~~~~~~~~~~~~

Annotate provides a keyboard-driven crop grid, records labels directly in the
project database and can rank an active-learning queue by uncertainty.
Classify trains PyTorch image models or classical and boosted models from
measurement tables. Checkpoints record their dataset, split rule, class
balance and held-out metrics.

Map Barcodes
~~~~~~~~~~~~

Barcode mapping decodes row, column and gRNA barcodes from FASTQ reads, joins
them to imaged wells and reports abundance, collision, unmapped-read and
library-coverage checks.

Regression
~~~~~~~~~~

Regression estimates guide, gene, condition and control effects. Its model
families cover continuous, fractional, binary and count responses, robust and
quantile fits, penalised high-dimensional designs, mixed effects and guide
permutation. Diagnostics and run summaries are written beside the result.

Planning, quality control and exploration
-----------------------------------------

- **Power and Design** estimate cell and well requirements and lay out plates,
  controls and replicates.
- **QC Dashboard** combines segmentation, plate, annotation-agreement and
  leakage checks.
- **Batch correction** provides centering, z-scoring, robust z-scoring,
  control centering and ComBat with protected biological covariates.
- **Graph Builder, gates and linked views** connect summary plots to the
  object crops behind them.
- **Feature, dose-response, control-chart and outlier views** inspect a result
  without an export/re-import cycle.
- **Layer and lineage views** connect images, masks and the cell → nucleus →
  pathogen object hierarchy.

Reproducibility and interoperability
------------------------------------

Every run can record its identifier, seed, resolved settings and outputs.
Interrupted workflows can resume from checkpoints, and the run history can
compare settings and artefacts. Measurements export to AnnData; optional
integrations read or write OME-Zarr, connect to OMERO and send masks to napari.

Maturity labels
---------------

The API uses these labels consistently:

**Stable**
   Supported entry points used by the principal Mask, Measure, Classify,
   barcode and regression workflows. Backward-incompatible changes require a
   deprecation period.

**Advanced**
   Supported specialist functionality whose defaults or result schema may
   still evolve. Release notes describe material changes.

**Experimental**
   Early interfaces intended for evaluation. They may change between minor
   releases and should be pinned before use in an automated workflow.

**Internal**
   GUI widgets, workers and implementation helpers. They are documented for
   contributors but are not a compatibility promise.

Optional dependencies
---------------------

The base package contains the headless pipelines. Install ``spacr[qt]`` for
the desktop interface. Other extras add OME-Zarr, OMERO, napari, attribution,
tracking, Zernike measurements and vendor readers. Availability varies with
Python version; the :doc:`installer guide <installers>` is the authoritative
compatibility table.
