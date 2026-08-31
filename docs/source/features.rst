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

The objects are not a fixed set of four. A project has a cell, a nucleus and a
pathogen, a cytoplasm derived from them, and as many organelle slots as
``number_of_organelles`` asks for -- from none up to twenty-six. Each slot is
independent, with its own channel, diameter, detection method and morphology.

A slot is given a morphology preset -- punctate, vesicular, spherical,
filamentous, tubular, reticular, cisternal, toroidal, crescent, or custom --
and the preset chooses the detection strategy, one of spots, network,
irregular or ring.

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

How a module is reached
-----------------------

The home screen groups modules into four categories -- **Core**, **Data**,
**Tools** and **Assays** -- and twenty-one modules have a tile in one of
them. Core is the pipeline you run in order; Data is what goes in and what
comes out of it; Tools are the instruments you point at a project rather
than steps the pipeline takes on its own; Assays are the quantitative
readouts. Make Masks is filed under **Tools**.

A TILE IS NOT THE ONLY WAY IN, and most modules do not have one. A tile says
"start here", and a module that answers a question about a run somebody else
started is not a place to start. Those open instead from:

- **a button on their host's masthead** -- as a page beside that host's
  settings, already pointed at the same project. Investigate Hit and
  Prediction Profiler open from Regression; Format Converter and External
  Masks from Import; Layer Viewer, Control Charts and Outliers from QC.
- **the Help menu**, for the ones that inspect or administer work that
  already exists rather than belonging behind any one module -- Run History,
  Pipeline Graph, Project Browser, Database Browser, Report, Data Manager,
  Plate Queue, Batch Runner and Distributed Jobs.
- **the command palette** (Ctrl+K), which reaches EVERY module, tiled or
  not. It is the one route with no exceptions, and the keyboard user's
  navigation.

None of them is second-class: they are shipped, translated and documented
like any other module, and the ones that are pipelines still run headlessly
under ``spacr-run``.

============= ==========================================================
Host          Opens from its masthead
============= ==========================================================
Mask          Timelapse
Measure       Illumination Correction, AnnData Export, Motility Assay
Annotate      Annotator Agreement
Classify      Classifier Evaluation, Explain CV Model, Activation Maps
Map Barcodes  Barcode QC
Regression    Volcano Explorer, Hit List, Methods & Results
Image UMAP    Image Scatter, PCA
Make Masks    Cellpose Workbench, Mask the whole folder, Model Compare,
              Model Zoo, Curate, Napari Bridge
============= ==========================================================

Parameter Sweep is reached a third way: it is a panel on the Regression
screen, opened by the **Parameter sweep** switch on its settings form.

Make Masks
----------

Make Masks corrects masks by hand and carries the Cellpose loop on its
masthead. Its canvas has nine tools: Brush, Erase, Erase object, Wand +,
Wand −, Draw, Divide, Zoom and Recrop.

Draw traces a free-form outline that closes and fills as a single object --
the tool a brush is not, because a brush stamps disks along the path, so
tracing a rim with it labels the rim and leaves the middle background. Divide
drags a line across a merged object and makes it two, leaving every other
object's label untouched; it is the commonest correction a segmentation needs.

Recrop is the only tool that changes which field is on screen rather than
what is painted on it. A staged crop holding several cells is not one training
example, and curating it as one teaches a network that two objects are one
picture -- so a box round an object writes that region of both the image and
the mask as a field of its own, queued straight after the current one, and the
multi-object original is retired into ``recropped_originals/`` rather than
curated. A box smaller than the minimum side, or one repeating a cut already
made, is refused; objects the box cuts through are dropped, because an object
whose boundary is where the mouse was released is not that object; and the
labels that survive are renumbered from one.

Running Cellpose-SAM from this screen shows its two intermediate outputs
beside the mask: the cell-probability map and the flow field. A mask is a
threshold applied to that probability map, and a candidate object is discarded
when its flows disagree with the ones the network predicted by more than the
flow-error threshold. When a mask is wrong, those two panes are where the
reason is visible.

Settings that apply
-------------------

The settings panel carries a control when it applies to the run being set up
and leaves it off the form when it does not:

- a slot past ``number_of_organelles`` takes its whole block of settings with
  it, its channel included -- a slot the run does not have is not a slot with
  its channel left showing;
- an object whose channel names no plane is not in the run at all, so its
  settings are not on the form;
- a setting belonging to one morphology is dropped for a slot of another: a
  punctate organelle has no ridge filter.

The 3D and Time switches declare which dimensions the plate has. ``z_stack``
declares a z axis and enables the volumetric settings -- segmentation mode,
anisotropy and voxel size -- and stops with an error rather than guessing
which axis is z. ``timelapse`` declares a time axis and reveals tracking; a
single-timepoint plate ignores it. The 4D settings apply only when the data is
both a z-stack and a time series, and appear only then.

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
Python version; the :doc:`installer guide <installer_guide>` is the authoritative
compatibility table.
