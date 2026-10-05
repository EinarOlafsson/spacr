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

Mask generation lays out the per-object settings as one table under
**Per-object settings**, with a column for each object whose channel is set;
the cell column is always shown. An object's channel is set on the ordinary
form, and a hidden object's answers are kept for when its channel is set
again. Every object has a **Remove background** check box, and the cell
column also has **Adjust cells**. **Add a filter** adds a row that filters
objects on any scikit-image regionprop, such as area or mean intensity; each
object's cell takes ``200 – 5000``, a minimum alone (``200``), a maximum
alone (``– 5000``) or a blank for no filter. These rows are the
``object_filters`` setting. Choosing a Cellpose 3 model for an object shows a
**Cellpose 3** category with that backend's own settings. **Quality Control**
holds the segmentation checks together with output, storage, runtime and
reliability settings.

Mask's **Live** preview shows the result as **Overlay**, **Masks**,
**Flows** or **Cell probability**. Right-click a picture to **Save as PNG…**
or **Save as PDF…**. Once a session has produced more than one mask,
**Compare masks…** lays the masks and images you tick over one another,
each with its own opacity and stacking order, in a panel of its own.

Measure
~~~~~~~

Measure writes per-object morphology, intensity, texture, radial, spatial and
colocalization features to ``measurements.db``. It can save classifier-ready
object crops, estimate illumination correction from a plate, restrict work to
a region of interest and report segmentation quality before a run.

Measure's settings are grouped as **Input & Experiment**, **Mask & Channel
Mapping**, **Image Preprocessing** (with **Image Deconvolution (PSF)** and
**Illumination Correction**), **Features**, **Object Filtering**, **Crop
Output**, **3D Calibration (Beta)** and **Postprocessing** (with **Runtime &
Reliability**). The segmentation-quality verdict is opt-in: the **QC** switch
beside **3D** and **Time** opens it in a popup, and closing the popup turns
the switch off.

Annotate and Classify
~~~~~~~~~~~~~~~~~~~~~

Annotate provides a keyboard-driven crop grid, records labels directly in the
project database and can rank an active-learning queue by uncertainty.
A page holds as many crops as fit and never scrolls; the rest are on the next
page. After **Suggest…**, suggested crops carry a **?** badge: click one or
press **Y** to confirm it, right-click or press **N** to reject it, and **U**
undoes. **Confirm the rest of the page** and **Reject the rest of the page**
judge every remaining suggestion at once. Rejections are kept and inform the
next round of training.
Classify trains PyTorch image models or classical and boosted models from
measurement tables. Checkpoints record their dataset, split rule, class
balance and held-out metrics. The **Essentials** view of Classify holds what
is needed to choose and train either family — an image model such as ResNet
or MaxViT, or a tabular model such as XGBoost or a random forest — and greys
the settings of the family not chosen.

Classify CV also offers optional rotations and reflections at inference time.
See :doc:`classifier_evaluation` for the aggregation methods, original and
mean probabilities, and orientation-stability flags.

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

In **Live Preview**, **QC** image views and the raw/enhanced comparison, wheel
zoom keeps the image point beneath the pointer in place. This also works when
the image is smaller than its viewport; **Fit** or opening a new image resets
the extra navigation space. Linked comparison views update together after the
zoom. Orthogonal image views zoom around the pointer without moving the
crosshair's selected position.

For Toxoplasma compartment-intensity comparisons, see :doc:`recruitment`.

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

Picking up where you left off
-----------------------------

- **Session restore**: every ordinary start reopens the module that was on
  screen when spaCR last closed or crashed, with its settings and input
  folder. No run is started. While the status bar says what was reopened, its
  **Open fresh** button puts that module's default settings back and goes to
  Home; ``spacr-qt --fresh`` skips the restore for one start. Turn it off in
  **Preferences → Modules → Session** (on by default). A module named on the
  command line, or a Force restart record, takes precedence.
- **Settings autosave**: every 30 seconds the unsaved settings of every open
  module are kept as a draft in the preference store; nothing is written when
  nothing changed. A clean exit drops the drafts. If spaCR ended without
  closing, the next start asks once whether to restore them (drafts identical
  to the reopened session are not asked about). The same behaviour applies on
  Linux, macOS and Windows, because the store is Qt's own settings store.

Undo and redo
-------------

**Ctrl+Z** undoes and **Ctrl+Shift+Z** or **Ctrl+Y** redoes in a module's
settings panel, in the Gate Editor and in Annotate:

- in the **settings panel**, each changed setting is one step. Loading
  settings, applying a template or replaying a run does not add steps.
  While you are typing in a text field, Ctrl+Z undoes the typing first;
- in the **Gate Editor**, each change to the gates (drawing, moving,
  renaming or deleting a gate) is one step;
- in **Annotate**, **U** and Ctrl+Z undo the last label and Ctrl+Shift+Z or
  Ctrl+Y puts it back;
- the **Make Masks** editor keeps its own **Undo** and **Redo**.

Running jobs
------------

**Help → Window → Jobs** opens the **Jobs** panel at the right edge of the
window. It lists every job that is running, with its module, a progress bar,
the time elapsed and its last output line. **Cancel** asks that job to stop.
With nothing running, the panel says **No jobs are running.**

Two spaCR windows
-----------------

You can open more than one spaCR window. When a module runs, spaCR claims
its source folder, which holds ``measurements.db`` and the results, and the
Queue screen claims the plate queue. If another window already uses that
folder or queue, **Project open in another window** names the other
process and offers:

- **Open read-only**: look without writing. This window then refuses to run
  a pipeline that writes there, and the Queue screen cannot change or run
  the queue;
- **Continue anyway**: write as well, when you are sure the other window is
  idle. Two windows writing the same queue, database or results folder can
  corrupt them;
- **Cancel**.

The answer is remembered for that folder until the window closes. A window
that ended without closing does not keep its claims: the next window takes
them over. The lock files are in ``~/.spacr/locks``; set ``SPACR_LOCK_DIR``
to keep them elsewhere.

Accessibility
-------------

Every control has a name that screen readers announce, taken from its label,
tooltip or placeholder, and the names follow the interface language. **Tab**
moves out of text boxes and tables to the next control instead of typing a
tab. **Preferences → Appearance → Theme** offers **High contrast**, in which
all text has a contrast ratio of at least 7:1 against its background.

Arranging the window
--------------------

- **Home** keeps its panels in a right-hand column. Drag the column's arrow
  handle to resize it, or click it to fold the column away
  and bring it back. The **Text size** slider below the lowest panel scales
  the text in those panels.
- **The dock**: while it is shown, drag its edge to make it wider or
  narrower; double-click the edge to fit it to the module names again.
- **A module's right-hand column** (console, actions and the other panels)
  resizes as one by dragging its left edge. Hold **Ctrl** and scroll over it
  to make its text larger or smaller; **Ctrl+0** over the column returns to
  the default size.
- **Tooltips** appear once the pointer has rested on a control for the
  **Tooltip delay** set in **Preferences → Appearance** (2.0 s by default;
  0 shows them at once) and stay while the pointer is on them. The same
  delay applies to setting help, hint strips and other hover help. Turn
  tooltips off with **Show tooltips** on the same page.

These sizes are remembered between sessions.

To resize a popup, point at one of its edges or corners. A thin blue guide
line appears across the middle half of each side you can drag. Drag from
anywhere along that edge or corner; the line only marks the side.

Make Masks
----------

Make Masks corrects masks by hand and carries the Cellpose loop on its
masthead. Its canvas has ten tools: Brush, Erase, Erase object, Wand +,
Wand −, Draw, Divide, Zoom, Recrop and Ruler. The
:doc:`Make Masks reference <make_masks>` covers these tools, Levels,
detection settings, primary/secondary pairing, saving and measurement.

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

Figure settings
---------------

**Preferences → Figures** sets how every spaCR figure looks and how it is
saved. The settings apply to figures drawn on screen and to the files the
pipelines and the save buttons write; only the settings you change are
applied, so a figure keeps its own look for everything else.

- **Text and lines**: font, the sizes of titles, axis labels, tick labels and
  legend entries (**Legend size**), line width and marker size.
- **Colour**: the palette for categories and the **Colormap** for heat maps,
  images and density plots. ``viridis`` (the default) and ``cividis`` stay
  readable with colour-blindness and in greyscale.
- **Layout**: background, grid, axis lines and **Despine offset**, which
  moves the axis lines away from the data. **Figure width** and **Figure
  height**, in inches, set the size of new figures; the height applies when
  the page shape is custom.
- **Saving**: **Format** is PDF, PNG, SVG or TIFF, with its resolution.
  **Also save** writes a second copy in another format beside each saved
  figure, for example a PNG next to a PDF. **Vector text** keeps text
  editable in PDF and SVG files instead of turning it into outlines.
- **Bar and point graphs**: **Error bars** shows the standard error
  (``sem``, the default), the standard deviation (``sd``), a confidence
  interval at the **CI level** (``ci``), a 95% interval (``ci95``) or nothing.
  **Error-bar cap size** sets the cap width. **Point overlay** draws the
  individual points over bar and box graphs, spread by **Jitter width** and
  drawn with the opacity in **Point alpha**.

Settings can also differ per graph type. A figure is styled once, so a change
you make with its right-click menu is kept when it is saved. Figures that
are already drawn with several panels keep their size and resolution; the
size and resolution settings apply to new figures and to saved files.

Working with figures
--------------------

Right-click any figure in spaCR, in a module's figure list or in an embedded
plot such as Graph Builder, the UMAP explorer or a training comparison, to
open its figure menu. The Volcano explorer and the Gate Editor keep their own
right-click menus and add these entries to them.

- **Edit figure…** changes the title, axis labels, limits, scales, fonts,
  colours, legend, size and resolution of this figure. Changes appear as you
  make them.
- **Change graph type** redraws the figure as another graph that fits its
  data. Groups can be shown as box, violin, strip, swarm, bar, point or boxen
  plots, a box or bar plot with points, or as histogram, KDE or ECDF; two
  measurements as scatter, line, hexbin, KDE or regression plots; one
  distribution as histogram, KDE, ECDF, box, violin, strip or boxen; counts as
  a count plot or heat map; and a matrix as a heat map or clustered heat map.
  The figure is redrawn from its data and keeps your **Figure settings**.
- **Statistics…** shows the test spaCR chose for the data and lets you
  override the **Test**, the **Subject column**, **Paired / repeated
  measures** and the **Multiple-comparison correction**. **Show on the plot**
  draws the test and significance brackets on the figure.
- **Save figure (zip)…** writes one zip file with the figure in your saved
  format and as PNG, ``data.csv`` with the plotted data, ``statistics.csv``
  and ``statistics.txt``, ``spec.json`` with the figure's full recipe, and
  ``recreate_figure.py``, a script that draws the same figure from the CSV and
  JSON files without spaCR.

How the statistics are chosen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic choice follows the data:

- **Groups**: each group is tested for normality (Shapiro–Wilk) and the
  groups for equal variance (Bartlett, or Levene when a group is not normal).
  Two groups are compared with Student's t, Welch's t or Mann–Whitney, and
  paired data with a paired t test or Wilcoxon. Three or more groups get a
  one-way ANOVA followed by Tukey HSD, Welch's ANOVA followed by
  Games–Howell, Kruskal–Wallis followed by Dunn, or Friedman followed by
  pairwise Wilcoxon tests, with pairwise p-values corrected for multiple
  comparisons.
- **Two measurements**: Pearson or Spearman correlation, chosen by
  normality.
- **Counts**: chi-square, or Fisher's exact test when an expected count is
  below 5; tables larger than 2×2 use the Fisher–Freeman–Halton test with a
  fixed-seed Monte Carlo p-value.
- **Proportions** (a 0/1 outcome): a two-proportion z test or Fisher's exact
  test; three or more groups get chi-square followed by corrected pairwise z
  tests.

``statistics.csv`` has one row per test, with the test name, groups,
statistic, degrees of freedom, p-value, adjusted p-value and correction,
effect size, number of observations, whether the test was chosen
automatically or by you, and the reason for the choice.

Figures in Graph Builder and the Volcano explorer carry their data rows. For
other figures, spaCR reads the plotted values back from the figure; box and
violin plots drawn this way keep only their summary, not the individual
rows. A new graph type replaces a faceted Graph Builder grid with a single
plot until Graph Builder draws the chart again.

Workers and free memory
-----------------------

Before a module starts its workers, spaCR estimates how much memory each
worker needs from one input, such as a field, mask, crop batch or table. If
the ``n_jobs`` you set would leave less than 12.5% of the computer's RAM
free, a popup shows the estimate, the free memory and the number of workers
that fit. Choose **Use N workers (recommended)** to run with that
number, or keep your own number; Measure then pauses new fields whenever
free memory drops below the reserve. **Free RAM by closing applications…**
lists your own largest programs. Nothing is ticked at first, and only the
programs you tick are asked to quit, after you confirm; they can usually
save first, but unsaved work in them may still be lost.

Runs started without the GUI lower ``n_jobs`` to the safe number and print a
warning. Turn ``ram_guard`` off in **Advanced** to keep your ``n_jobs``
unchanged; Measure still waits while free memory is below the reserve.

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
