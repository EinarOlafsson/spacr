# Notes from `spacr/qt/widgets/regression_results.py`

Prose lifted out of `spacr/qt/widgets/regression_results.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [find_results_tables](#find_results_tables) (2 entries)
- [read_run_tables](#read_run_tables) (1 entry)
- [for_table](#for_table) (1 entry)
- [_summary_filenames](#_summary_filenames) (2 entries)
- [find_summary_file](#find_summary_file) (1 entry)
- [Module level](#module-level) (1 entry)
- [_with_spacr_summary](#_with_spacr_summary) (1 entry)
- [_spacr_summary_text](#_spacr_summary_text) (1 entry)
- [backend_of](#backend_of) (1 entry)
- [RegressionResultsPanel.__init__](#regressionresultspanel__init__) (36 entries)
- [RegressionResultsPanel.set_run_settings](#regressionresultspanelset_run_settings) (3 entries)
- [RegressionResultsPanel.ask_refit](#regressionresultspanelask_refit) (1 entry)
- [RegressionResultsPanel.show_panel](#regressionresultspanelshow_panel) (1 entry)
- [RegressionResultsPanel._clear_diagnostic_views](#regressionresultspanel_clear_diagnostic_views) (1 entry)
- [RegressionResultsPanel.set_summary](#regressionresultspanelset_summary) (1 entry)
- [RegressionResultsPanel._name_the_model](#regressionresultspanel_name_the_model) (1 entry)
- [RegressionResultsPanel._no_model_reason](#regressionresultspanel_no_model_reason) (1 entry)
- [RegressionResultsPanel.set_diagnostics](#regressionresultspanelset_diagnostics) (2 entries)
- [RegressionResultsPanel.judge_homogeneity](#regressionresultspaneljudge_homogeneity) (1 entry)
- [RegressionResultsPanel._extra_colour_column](#regressionresultspanel_extra_colour_column) (1 entry)
- [RegressionResultsPanel._select_many_from_a_plot](#regressionresultspanel_select_many_from_a_plot) (1 entry)
- [RegressionResultsPanel._folder_of](#regressionresultspanel_folder_of) (1 entry)
- [RegressionResultsPanel._set_loading](#regressionresultspanel_set_loading) (1 entry)
- [RegressionResultsPanel.cancel_load](#regressionresultspanelcancel_load) (1 entry)
- [RegressionResultsPanel.start_load](#regressionresultspanelstart_load) (2 entries)
- [RegressionResultsPanel._finish_load](#regressionresultspanel_finish_load) (1 entry)
- [RegressionResultsPanel._apply_loaded_run](#regressionresultspanel_apply_loaded_run) (1 entry)
- [RegressionResultsPanel._name_the_levels](#regressionresultspanel_name_the_levels) (3 entries)
- [RegressionResultsPanel.set_frame](#regressionresultspanelset_frame) (24 entries)
- [RegressionResultsPanel.plot_state](#regressionresultspanelplot_state) (2 entries)
- [RegressionResultsPanel.apply_plot_state](#regressionresultspanelapply_plot_state) (2 entries)
- [RegressionResultsPanel.apply_workspace_state](#regressionresultspanelapply_workspace_state) (1 entry)
- [RegressionResultsPanel._mark_the_level_on_the_plots](#regressionresultspanel_mark_the_level_on_the_plots) (1 entry)
- [RegressionResultsPanel.refresh_views](#regressionresultspanelrefresh_views) (4 entries)
- [RegressionResultsPanel._analysis_path](#regressionresultspanel_analysis_path) (1 entry)
- [RegressionResultsPanel._gene_terms](#regressionresultspanel_gene_terms) (1 entry)
- [RegressionResultsPanel._draw_guide_support](#regressionresultspanel_draw_guide_support) (5 entries)
- [RegressionResultsPanel._draw_guide_support.verdict](#regressionresultspanel_draw_guide_supportverdict) (1 entry)
- [RegressionResultsPanel._show_significance](#regressionresultspanel_show_significance) (1 entry)
- [RegressionResultsPanel.level_counts](#regressionresultspanellevel_counts) (1 entry)
- [RegressionResultsPanel._offer_levels](#regressionresultspanel_offer_levels) (1 entry)
- [RegressionResultsPanel.family_note](#regressionresultspanelfamily_note) (1 entry)
- [RegressionResultsPanel.set_level](#regressionresultspanelset_level) (1 entry)
- [RegressionResultsPanel._say_which_family](#regressionresultspanel_say_which_family) (1 entry)
- [RegressionResultsPanel._levels_of](#regressionresultspanel_levels_of) (1 entry)
- [RegressionResultsPanel._offer_p_values](#regressionresultspanel_offer_p_values) (1 entry)
- [RegressionResultsPanel._offer_compartments](#regressionresultspanel_offer_compartments) (1 entry)
- [RegressionResultsPanel.set_compartment](#regressionresultspanelset_compartment) (1 entry)
- [RegressionResultsPanel.set_baseline](#regressionresultspanelset_baseline) (1 entry)
- [RegressionResultsPanel._redraw_volcano](#regressionresultspanel_redraw_volcano) (7 entries)
- [RegressionResultsPanel._tested_mask](#regressionresultspanel_tested_mask) (1 entry)
- [RegressionResultsPanel._draw_controls](#regressionresultspanel_draw_controls) (1 entry)
- [RegressionResultsPanel._select_from_a_plot](#regressionresultspanel_select_from_a_plot) (1 entry)
- [RegressionResultsPanel._reachable](#regressionresultspanel_reachable) (1 entry)
- [RegressionResultsPanel._select_key](#regressionresultspanel_select_key) (1 entry)

## find_results_tables

### lines 165-169

```python
home = os.path.dirname(root)
```

AND THE REST OF THE RUN. A permutation run writes its guides to results.csv and its genes to results_gene.csv beside it; handed the one file, returning it alone hides the level the reader asked for. The siblings follow RESULT_FILENAMES order with the chosen file first, so it stays the primary table `read_run_tables` merges into.

### line 195, trailing  _(unsure)_

```python
except OSError:
```

vanished between listing and stat

## read_run_tables

### line 250, trailing  _(unsure)_

```python
continue
```

a different run, not a second half of this one

## for_table

### lines 303-308

```python
if ("coefficient" in columns
```

THE SAME NUMBER UNDER TWO NAMES IS WORSE THAN EITHER NAME ALONE. The permutation path copies `standardized_marginal_effect` into `coefficient` so the rest of the screen can read one name, and a reader then sees two identical columns and asks which is the real one -- and whether a quantity bounded in [-1, 1] is a coefficient at all. It is not: it is a partial correlation. The accurate name is the one kept.

## _summary_filenames

### line 358, trailing  _(unsure)_

```python
except Exception:
```

ml unavailable

### lines 359-361

```python
return ()
```

Named rather than guessed: without the writer there is nothing to agree with, and inventing the list here would be the second source of truth this indirection exists to avoid.

## find_summary_file

### lines 393-395

```python
table = find_results_table(root)
```

A PARENT OF A RUN FOLDER IS ALSO A LEGAL ANSWER, and it is the one `load` accepts, so the summary is looked for beside the table that was actually chosen rather than only in the folder the user typed.

## Module level

### line 450  _(unsure)_

```python
_CANONICAL_TABLE = "results.csv"
```

Canonical results tables share the run folder's remembered plot state.

## _with_spacr_summary

### lines 557-559

```python
return f"No summary: {statsmodels_text}" if missing else statsmodels_text
```

NOTHING AT ALL, and the tab says exactly that. "No summary" is the sentinel every caller tests for; the qualified wording below is only honest when there IS a spaCR summary above it.

## _spacr_summary_text

### lines 581-582  _(unsure)_

```python
if not text or not text.startswith(SPACR_SUMMARY_HEADING):
```

A file that is ONLY statsmodels text is a run from before spaCR wrote its own summary. Nothing to put in front.

## backend_of

### line 617, trailing  _(unsure)_

```python
except Exception:
```

hits unavailable

## RegressionResultsPanel.__init__

### lines 677-681

```python
self._load_button = QPushButton("Load results…")
```

A WAY IN THAT DOES NOT REQUIRE STARTING A RUN. The panel used to be filled from exactly one place -- successful run completion -- so a user whose results were already on disk, or whose run finished while the settings pointed somewhere else, had no way to open them at all and no reason given for the empty table.

### lines 691-705

```python
controls_row = FlowHost()
```

WHICH RUN IS ON SCREEN, SAID WHERE THE RESULTS ARE. Instruction 157: the loaded mark lived in the Runs tab and the coefficients lived here, so the only way to notice the two had diverged was to compare two views -- and the maintainer did exactly that ("even if the ols model is marked as loaded i still see the mixed results"). A panel that names its own run makes the disagreement visible in the view that is wrong, rather than in the one that is right. THE SECOND ROW, AND IT WRAPS. Everything below has a minimum width -- three combo boxes at 140, 120 and 120 plus two labels and a QHBoxLayout asked for more than the panel has does not shrink its children below their minimum: it lets them OVERLAP. Measured on the real screen at a 577 px panel, the second combo began 48 px inside the first, the third began 27 px inside the second, and the third ran 32 px off the right edge of the panel. `FlowLayout` puts the overflow on a new line instead (236 C10).

### lines 716-730

```python
self._level_label = QLabel("level")
```

WHICH HALF OF THE RUN IS ON SCREEN, ON THE PANEL ITSELF.

"in the regression module i still only get gRNA level coefficients i cant plot the gene level coefficients ...or see the gene level coefficients." Both halves were there and both were reachable from the VOLCANO's own header and from the coefficient table's right-click menu. Neither is on screen while the reader is on the p-value tab, the Q-Q or the effect ranks, and the volcano's control is not in this panel at all when the host places the plot itself (`external_volcano`). A filter that decides what SEVEN tabs draw belongs on the panel that holds them.

THE SAME `set_level`, so the three entry points cannot end up with three opinions about which rows are being shown -- the rule the table's menu was added under, applied once more.

### lines 747-748

```python
self._level_box.activated.connect(self._level_chosen)
```

`activated`, not `currentIndexChanged`: refilling the box on every redraw must not look like a choice. See `_offer_levels`.

### lines 751-767

```python
self._colour_by_label = QLabel("colour by")
```

THE CONTROL BELONGS ON THE FIGURE IT CHANGES. Asked for 2026-08-19: "the color by in results can be removed as its alos in the right click for the volcano graph the only place it is used i think" and it is: the volcano's own menu already offers "Colour by a column…" and "Colour by localisation", and the volcano is the only thing this combo redraws.

HIDDEN, NOT DELETED, and the difference matters. `_redraw_volcano` reads `currentData()`, `_restore_plot_state` writes it back, and a saved run carries a `colour_by` key -- so the object stays and keeps answering, while the header loses a duplicate. Deleting it would have meant unpicking a saved-state field for a cosmetic win. VISIBLE AGAIN, because it no longer duplicates the menu. It was hidden when it offered exactly what the volcano's own "Colour by a column…" offered; it now offers something that menu cannot -- three ORDERED channels at once (instruction 222) -- and a feature nobody can reach is not a feature.

### lines 769-774

```python
self._colour_by_label.setToolTip(
```

ONE TOOLTIP, ON THE NAME. The combo carried a second one saying what the first channel does, and `retarget_field_tooltips` leaves a field alone when its label already has help of its own -- so this was the one control on the panel whose hover help stayed on the field. Both sentences belong to the same name, so they are one tooltip on it.

### lines 789-793

```python
self._colour_by_2 = QComboBox()
```

THE SECOND AND THIRD COLUMNS (instruction 222). Separate combos rather than one checkable list, because the ORDER is the encoding: first is hue, second is shape, third is opacity, and a checklist has no way to say which is which. Each offers the same columns and "nothing", so the user adds a channel by filling in the next box.

### lines 812-816

```python
self._missing_level = QLabel("")
```

WHY A LEVEL IS EMPTY, WHERE THE EMPTY THING IS. Above the tabs, so it is beside whichever tab the reader is looking at rather than only on the one that happens to carry a status line -- and hidden whenever there is nothing to say, because a banner that is always there is a banner nobody reads. Filled by :meth:`_offer_levels`.

### lines 827-828  _(unsure)_

```python
self.volcano = VolcanoPlot()
```

Volcano and table share a splitter: the two views of one table belong beside each other, and the divider is the user's to move.

### lines 833-838

```python
self.table.table.setContextMenuPolicy(Qt.CustomContextMenu)
```

THE SAME GESTURE ON THE TABLE AS ON THE PLOT. Instruction 128 L: "i should be able to right click on the coeffisients table and only see grna or genes and this should also filer the subsequent data/graphs in the subsequent tabs". Wired to the SAME :meth:`set_level`, so the two entry points cannot end up with two opinions about which rows the panel is showing.

### line 842  _(unsure)_

```python
self._volcano_tab, self._volcano_tab_name = self.table, "Coefficients"
```

The table is the panel; the graph is the caller's to place.

### lines 849-855

```python
self._model_line = QLabel("")
```

WHICH MODEL DREW THIS (189 B), directly under the graph and not three tabs away. Two volcanoes can be correctly identical glm and quasi_binomial share every coefficient, because dispersion moves the standard errors and not the point estimates -- and without a label that is indistinguishable from a bug. It was, and it is what "all the plots look the same no matter which regression type i do" was about.

### lines 864-868

```python
self.volcano.setMinimumHeight(240)
```

Floors, not preferences. Without them the panel's share of the window is divided by the widgets' own size hints and BOTH end up too short to read -- a volcano with no room for its axes and a table showing its header and one row, which is what this looked like on the real screen before the numbers were put in.

### lines 874-886

```python
self.effect_rank = EffectRankPlot()
```

THE TWO EFFECT PANELS, which the sheet has drawn since it existed and the screen had no twin of -- instruction 129 B, where the gap was measured at exactly these two. They sit directly after the volcano because they read in the same order the sheet does: the result, then HOW BIG it is and how sure, then whether the model was entitled to say it.

A volcano ranks by significance and cannot answer either question. On the TSG101 screen its top guide by p is not its top guide by effect, and the strongest effect in the screen (4.37) has q = 3e-05 while the third strongest (-4.22) has q = 0.063 -- one is called and the other is not, and only a ranked list with intervals shows that they are the same size.

### lines 909-911

```python
from PySide6.QtGui import QFontDatabase
```

THE STATSMODELS SUMMARY. Monospace, read-only and SELECTABLE: the reason to want it is usually to paste a number into a methods section, and a summary you cannot select is a summary you retype.

### lines 913-918

```python
from .folding_summary import FoldingSummaryView
```

168 D: "The Summary tab shows the verdict expanded and each section collapsed, with the section headings as the outline." A drop-in for the QPlainTextEdit that was here -- same setPlainText/toPlainText -- so nothing that fills or reads it changes, and text with no spaCR headings (the statsmodels summary) is still shown whole.

### lines 924-927

```python
self.residuals = ResidualPlot()
```

ADDED AFTER THE DIAGNOSTICS, not before them. Q-Q, Controls, Residuals, Scale-location and Influence are one group that reads in order, and a test asserts they sit together; dropping the Summary into the middle of it split the group.

### lines 929-945

```python
self.residuals = ResidualPlot()
```

THE RESIDUAL DIAGNOSTICS, LIVE. "in the tabs Q-Q and Controls, there should be Tabs like residuals showing the residuals and regression controll graphs like that."

Every tab so far is drawn from the COEFFICIENT table, which is one row per guide and says nothing about how well the model fitted the WELLS it was given. These three are the well-level half, and they are the three the QC report leads with: is the mean right, is the variance flat, is the answer resting on one well. They come from `spacr.regression_qc` -- the same arrays the saved report draws rather than being recomputed here, because a live panel that named different influential wells than the PDF beside it would be worse than no live panel.

THEY ARE ALWAYS TABS, even before there is a model to fill them, and each one SAYS why it is empty. A diagnostic that appears only once it happens to be computable is one nobody knows to look for.

### lines 951-962

```python
self.homogeneity = QLabel(self.NO_HOMOGENEITY_VERDICT)
```

THE HOMOGENEITY VERDICT, BESIDE THE PICTURE. Instruction 128 M: the Scale-location tab shipped the plot and never asked the question the plot exists to answer -- is the residual spread constant across the fitted range? A reader who cannot answer that from a scatter of 610 points (and most cannot) has no way to know that every standard error in the Summary tab, and so every p-value in the coefficient table, is optimistic.

A LABEL, NOT A STATUS LINE. The plot's own status is overwritten by whatever was last clicked (`FastPlot.note_selection`), and a verdict that disappears when the user interacts with the panel is a verdict they will not have read.

### lines 974-983

```python
try:
```

WHERE THE ANNOTATED CELLS LAND (215). An independent check: the annotation is made from sequencing fractions plus a phenotype call, and this asks where the cell sits among the CONTROLS using every measurement at once. Agreement between two routes that different is worth more than either alone.

Nothing is computed until the tab is opened and its button pressed. A UMAP over a screen's cells is seconds of work, and a tab that embedded on construction would charge that to every user who never looks at it.

### line 990

```python
self.annotation_umap = None
```

A panel that cannot build its optional tab is still a panel.

### lines 993-994  _(unsure)_

```python
self.tabs.addTab(self._summary, "Summary")
```

The statsmodels summary closes the diagnostic group: it is the model-level readout the panels above it are pictures of.

### lines 1010-1017

```python
self.agreement = GuideAgreementPlot()
```

GUIDE SUPPORT. The one thing the volcano structurally cannot show: a gene carried by a single surviving guide and a gene whose guides agree are the same dot, ranked by the same number, and only one of them is independent evidence.

Plot above table, the same arrangement as the volcano tab and for the same reason: the picture says which genes rest on one guide at a glance, and the numbers behind any one of them are a click away.

### lines 1032-1046

```python
from .gene_panel import GenePanel
```

THE GENE TILE. Instruction 121. The volcano answers "which guides moved" and structurally cannot answer "what IS 411710", which is the question the user has the instant they click one. The frame is reached through a callable rather than stored, so a newly loaded regression is never answered from the previous one.

`GenePanel` and not `GeneTilePanel`: the tile alone says WHICH gene a dot is and what THIS screen measured about it, both read out of the frame already on screen. The panel adds the other half the instruction asks for -- product, DeepTMHMM topology with each segment's coordinates, hyperLOPIT compartment, the published fitness screens and the stage expression -- out of `spacr.annotation`, and it loads those five CSVs on a worker thread. Cold that read is 360 ms, and 360 ms inside a mouse press is a plot that reads as broken.

### lines 1051-1055

```python
for plot in self._keyed_plots():
```

When the volcano is external the tile goes WITH IT, not behind a tab: "when a gene is clicked a tile should appear with all the information on that gene" -- appear, beside the point that was clicked. A tile the user has to go and find is a tile they will not look at. The caller places it, exactly as it places the volcano.

### lines 1057-1076

```python
for plot in self._keyed_plots():
```

THE TWO DIRECTIONS OF THE SAME LINK, JOINED ON THE KEY.

Not on a position. The table is sorted by whatever column the user clicked last and filtered by whatever is in the search box; the volcano is drawn in input order and, since it stopped plotting the nuisance terms, does not even hold the same number of rows. Two frames in two orders joined by index highlight the wrong guide silently, and in the one direction nobody questions, because a point did light up.

`feature` is the key: 1,213 rows and 1,213 distinct values on the real screen. It is checked, not assumed -- see _key_column.

EVERY PLOT WHOSE POINTS ARE COEFFICIENTS, not just the volcano. Instruction 124 F: "id like to be able to presson the datapoints of all graphs where data is represented as genes and grnas, e.g. like the Q-Q plots". A Q-Q point IS a coefficient; so is a dot in the control panel and a gene in the agreement plot. They all reach the table by the same one-line route, which is why they cannot disagree about what a click means.

### lines 1079-1083

```python
self.p_values.key_selected.connect(self.table.select_key)
```

A BAR IS NOT A POINT. The histogram is the one mark here that stands for many rows, so it narrows the table to them rather than pretending to pick one -- see PValueHistogram.select_bin. When a bar happens to hold exactly ONE coefficient there is nothing to guess between, so it selects it like any other point and takes the same route as the rest.

### lines 1086-1089

```python
self.effect_distribution.key_selected.connect(self.table.select_key)
```

The effect distribution is the OTHER histogram, and it takes the same route for the same reason: its marks are bars of many coefficients, so it narrows the table rather than guessing which of them the user meant.

### lines 1093-1097

```python
self.table.key_selected.connect(self.gene.show_feature)
```

ON THE TABLE, NOT THE VOLCANO, and one connection rather than two: table.key_selected is the funnel BOTH directions already pass through -- volcano.key_selected -> table.select_key -> selection change -> re-emit. Connecting the volcano as well would build the tile twice for every click.

### lines 1099-1106

```python
for plot in self._keyed_plots():
```

THE MULTI-SELECT ROUTE, THE SAME SHAPE AS THE SINGLE ONE

(instruction 206). Every keyed plot's band and modifier-click reach the table, and the table re-emits to the consumers -- so there is still exactly one place that decides what "the selection" is, and the gene tile cannot be showing one guide while the image tabs show another. The two histograms are NOT on this route: their `keys_selected` means "the rows behind this bar", which narrows the table rather than selecting, and is already connected above.

### lines 1138-1144

```python
from ..job_runner import JobRunner
```

LOADING A RUN GOES OFF THE GUI THREAD (instruction 159). `load` walked the folder, read the CSV and rebuilt every view inline, and this file contained no JobRunner at all -- so a big table or a deep folder stopped the window with no spinner and no cancel, which is indistinguishable from a crash. It is the same defect 154 A fixed for the merge, and this is the same machinery rather than a second one.

### lines 1152-1170

```python
self._level = self._default_level()
```

A DEFAULT, NOT A PIN, and there is exactly one statement of it: :meth:`_default_level`, which every table goes through. This line used to be a second, unconditional `= "grna"`, and that is what made the gene half of a `level='both'` run unreachable to a reader who never right-clicked the plot.

GUIDES ARE STILL WHERE A TABLE OPENS. The guide is the unit the screen measures, and a permutation run reports guides only, so it is the level the two inference modes agree on. What has changed is that the other two are on the panel beside it.

The duplication the pin was fixing -- `225160` drawn four times, once per guide plus once for itself, every one of them labelled `225160` -- is fixed at its source now: the rows are keyed and labelled by the design-matrix TERM, so the four read `gene_fraction:gene[225160]`, `fraction:grna[225160_1]`, `[_2]` and `[_3]`, and a `level` column says which fit each came from. Checked on the real screen before this default was relaxed, not assumed.

### lines 1191-1192

```python
self.clear_diagnostics()
```

The diagnostics start out saying what they are waiting for. An empty plot with no sentence is indistinguishable from a broken one.

### lines 1197-1200

```python
self.volcano.offer_refit(self.ask_refit)
```

RE-FITTING IS OFFERED FROM THE PLOT, under its own heading, because that is where it was asked for: "right click on the regression plot and choose a different regression". It is separated from the restyling entries above it for the reason `offer_refit` gives.

### lines 1206-1208

```python
self._colour_by_label.setToolTip("\n\n".join([
```

These three editors share one visible setting name. Keep each channel's detailed help, but expose it from that name instead of requiring the pointer to discover help on the editable fields.

### line 1218  _(unsure)_

```python
from ..screens.settings_model import retarget_field_tooltips
```

Move any remaining editor help to the label that names its setting.

## RegressionResultsPanel.set_run_settings

### lines 1233-1235

```python
self._from_live_run = True
```

ONLY THE LIVE PATH CALLS THIS, so it is also the answer to "did this table come from a run in this session or off the disk" -- a question the Summary tab used to answer with a guess.

### lines 1237-1240

```python
self._offer_levels()
```

THE SETTINGS ARE EVIDENCE ABOUT AN EMPTY LEVEL, so the sentence that reads them is rebuilt now rather than at the next click: `level='grna'` is the plainest possible answer to "why is Gene empty", and it arrives after the table it describes.

### lines 1242-1244

```python
if self._model is None:
```

The reason for an absent summary has just changed, so the sentence on the tab has to. Only when there is no model to render: a summary already on screen is the run's own and is not rewritten.

## RegressionResultsPanel.ask_refit

### lines 1263-1267

```python
self.say(str(error))
```

No settings, or no count data in them. SAID ON THE PANEL and

BEFORE the dialog opens: a form whose only content is a disabled button and an error is worse than a sentence, and the user right-clicked a graph -- a traceback is not an answer to that either.

## RegressionResultsPanel.show_panel

### lines 1416-1418

```python
if self.tabs.indexOf(page) < 0:
```

-1 when the widget exists but was never added, which is the ordinary state of the volcano and gene tabs on a screen that shows the volcano outside the panel.

## RegressionResultsPanel._clear_diagnostic_views

### lines 1449-1451

```python
self._homogeneity = {}
```

THE VERDICT GOES WITH THE PICTURE IT JUDGED. Left behind, it would be a sentence about the previous fit sitting under an empty plot, which is the worst of both: authoritative and about nothing.

## RegressionResultsPanel.set_summary

### lines 1469-1472

```python
model = self._model
```

THE MODEL THIS PANEL ALREADY HAS. A caller handing over a payload whose `model` key is empty is saying "the run returned none", not "throw away the one you were given" -- and the fit kept by `set_diagnostics` is the same fit this table came from.

## RegressionResultsPanel._name_the_model

### lines 1504-1506

```python
label.setVisible(bool(said))
```

HIDDEN WHEN IT HAS NOTHING TO SAY. An empty muted strip under the graph is a row of pixels that means nothing, and a caption that guessed would be worse than no caption.

## RegressionResultsPanel._no_model_reason

### lines 1524-1528

```python
return (f"the diagnostics failed and the fit was not kept "
```

Defensive rather than expected: `set_diagnostics` now stores the fit BEFORE it tries to draw anything, so a diagnostics failure no longer takes the model with it. If some other path ever loses one, the tab says which error did it rather than telling the disk story again.

## RegressionResultsPanel.set_diagnostics

### lines 1558-1567

```python
self._model = model
```

THE FIT IS STORED BEFORE ANYTHING IS DRAWN FROM IT.

It used to be assigned at the bottom, after the context had been built, so every early return below threw the model away -- and the Summary tab then explained its absence with "this panel was opened from a results table on disk", which for a live run whose diagnostics could not be built is simply untrue. A failure in the VIEW must not destroy the thing being viewed: the model is the fit, the diagnostics are one way of looking at it, and the summary is another that works perfectly well when this one does not.

### lines 1605-1607

```python
self._note_the_diagnostics()
```

The three plots above each wrote their own headline, which clears whatever note was on them -- so the "these are wells, not coefficients" sentence is put back last or it is not there at all.

## RegressionResultsPanel.judge_homogeneity

### lines 1649-1651

```python
self._homogeneity = {}
```

A real answer about the fit -- a quantile or hinge fit has no error scale, so it has no standardised residual to judge -- and it must not read like a broken panel.

## RegressionResultsPanel._extra_colour_column

### lines 1766-1768

```python
if chosen == self.LOPIT_KEY:
```

LOPIT is materialised onto the frame copy under its own name by the caller, and only for the FIRST channel. Asking for it here would name a column that is not there.

## RegressionResultsPanel._select_many_from_a_plot

### lines 1792-1794

```python
return
```

One key is the ordinary single click, which already has a route. Taking it here as well would select it twice and build the gene tile twice for every click.

## RegressionResultsPanel._folder_of

### lines 1847-1853

```python
if os.path.splitext(os.path.basename(path))[1]:
```

A PATH THAT NO LONGER EXISTS IS STILL EVIDENCE. A run folder that was deleted (146) or moved should name the run it named yesterday rather than reading as "this table came from nowhere" -- and the state keyed on it has to be reachable to be forgotten. The two shapes are told apart by the only thing left to read: a results table is `results.csv` and a run folder is `ols_3`, so a suffix means a file and no suffix means the folder.

## RegressionResultsPanel._set_loading

### line 1977, trailing  _(unsure)_

```python
pass
```

the widget went away under a close

## RegressionResultsPanel.cancel_load

### lines 1993-1994

```python
self.load_finished.emit(False)
```

THE SAME ENDING AS EVERY OTHER, because a caller waiting on

`load_finished` must not be left waiting by a cancel either.

## RegressionResultsPanel.start_load

### line 2019, trailing  _(unsure)_

```python
if not started:
```

JobRunner

### line 2020, trailing  _(unsure)_

```python
self._set_loading(False)
```

always returns True today

## RegressionResultsPanel._finish_load

### lines 2063-2064

```python
self.load_finished.emit(bool(ok))
```

ALWAYS, on both endings. A caller waiting on this must not be left waiting by a failure -- that is a spinner nothing clears.

## RegressionResultsPanel._apply_loaded_run

### lines 2125-2126

```python
self.say(f"{self._status} ({found}, found under {searched})")
```

set_frame said why -- an empty table is not the same failure as a missing one -- but the folder it came from is worth adding.

## RegressionResultsPanel._name_the_levels

### line 2200, trailing  _(unsure)_

```python
except Exception:
```

hits unavailable

### lines 2209-2214

```python
levels = levels.replace("", "nuisance")
```

A NAME FOR THE ROWS IN NEITHER FAMILY. The intercept and the plate row/column terms are covariates, not hypotheses, so they belong to no level -- and a blank in a column offered as a colouring is a legend entry with no name on it. `coefficient_levels` does not recognise the word, so reading the column back still places those rows by their term, which is where the blank came from.

### lines 2216-2217

```python
frame.insert(columns.index("feature") + 1, "level", levels)
```

BESIDE THE IDENTIFIER IT QUALIFIES, not appended after thirty annotation columns where a reader scanning the row never reaches it.

## RegressionResultsPanel.set_frame

### lines 2229-2237

```python
self._say_if_no_p_values(frame)
```

ROWS BUT NO P VALUES IS NOT AN EMPTY TABLE, and saying nothing about it produced the report "with guides i see nothing in the graph" (2026-08-21) against a run that had worked perfectly.

A MIXED MODEL MAKES THE GUIDE A RANDOM EFFECT. Each guide gets a shrunken BLUP -- a prediction -- and a BLUP has no p value, so a volcano, whose vertical axis IS the p value, has nothing to draw. The run already says this in the console; the panel the user is looking at did not.

### line 2239

```python
frame = self._name_the_levels(frame)
```

THE LEVEL IS WRITTEN INTO THE TABLE, so it survives the panel.

### lines 2241-2244

```python
self._remember_plot_state()
```

THE OUTGOING RUN KEEPS WHAT THE USER BUILT ON IT. Saved BEFORE anything is replaced, because every line below this one resets a piece of it -- and saved against the OLD path, which is the key the user will come back through.

### lines 2248-2250

```python
self._name_the_run()
```

NAMED THE MOMENT THE TABLE CHANGES, not at the end: every early return below this line would otherwise leave the header naming the previous run over the new one's coefficients.

### lines 2254-2258

```python
self._from_live_run = False
```

A NEW TABLE IS A NEW FIT, so the old fit's residuals have to go. The caller that HAS a model calls `set_diagnostics` immediately after this; the caller that loaded a CSV has none, and leaving the last run's residuals on the tabs would describe a fit the user is no longer looking at -- with nothing on screen saying so.

### lines 2262-2269

```python
self.set_summary(None)
```

AND THE SUMMARY IS THE SAME KIND OF STALENESS. It was left untouched here, so a table opened from disk sat under the PREVIOUS run's statsmodels output with nothing saying whose it was. Refreshed from the new table's own folder: `perform_regression` writes the summary beside `results.csv`, so a run re-opened from disk shows the text it showed while it was running -- byte for byte, because it is the same bytes. A live run overrides this a moment later with the model itself, which is better still.

### lines 2272-2277

```python
from ...refit import settings_of_run
```

WHICH SETTINGS PRODUCED THIS TABLE. Read from beside the table, and REPLACED rather than kept: a new table is a new experiment, and carrying the last run's settings over would offer to re-fit a screen the panel is no longer showing. A live run overrides this by calling `set_run_settings` afterwards, which is better still the shared settings/ copy on disk is overwritten by every later run.

### lines 2285-2286  _(unsure)_

```python
self._colour_by.blockSignals(True)
```

Offer every column that could sensibly colour the points, without guessing: a column with one value per point is not a category.

### lines 2304-2310

```python
self._colour_by.addItem(f"{name} ({distinct})", name)
```

OFFERED ANYWAY, AND THE COUNT SAYS WHY IT IS USELESS. A `condition` column with ONE value is not a boring column, it is a FINDING: it means the negative/positive control names matched no feature, so nothing got labelled. Dropping it silently hides that, and the maintainer reported exactly this as "the color by doesn't include condition".

### lines 2317-2319

```python
try:
```

LOPIT IS NOT A COLUMN, so the walk above cannot see it. It is joined from the bundled TAGM table, and only offered when this screen actually has compartments in it.

### lines 2330-2331  _(unsure)_

```python
preferred = self._colour_by.findData("condition")
```

'condition' is what a screen labels its controls with, so it is the colouring a reader wants first.

### lines 2335-2337

```python
for extra in (self._colour_by_2, self._colour_by_3):
```

THE SAME OFFER IN ALL THREE, so a user can put any column on any channel. Built from the first rather than by walking the frame again: two walks are two chances to offer different lists.

### lines 2350-2357

```python
self._selected_key = None
```

A new table is a new experiment; carrying the old selection over would ring a point that means something else now.

EVERY PLOT, not just the volcano. Each one re-marks `_selected_key` at the end of its own draw so a restyle does not lose the user's place -- which means a plot whose selection is NOT cleared here cheerfully re-rings the new run at the old key. Caught by exporting the control panel after a reload and finding a ring still on it.

### lines 2359-2365

```python
try:
```

AND THE TABLE'S OWN ROW, because the table is what re-establishes a selection. Clearing `_selected_key` and the rings was not enough: rebuilding the table leaves a row highlighted, that row re-emits `key_selected`, and the panel comes back from the reset holding a key -- on the new run, but one nobody chose, and a mark nobody chose is exactly what this reset exists to prevent. Blocked, so the clear itself does not emit a THIRD time.

### lines 2372-2375

```python
pass
```

A table whose C++ half has gone raises RuntimeError; one that was never built raises AttributeError. Neither is a reason to refuse the frame -- the selection being cleared is housekeeping ahead of repopulating it.

### lines 2381-2384

```python
self.gene.clear()
```

AND THE GENE TILE, for the same reason and it is the worst offender: a plot re-rings a point, but the tile keeps a whole paragraph about a gene from the previous screen, with this screen's effect nowhere near it and nothing on it saying which run it came from.

### lines 2386-2388

```python
self.gene.warm_for(frame)
```

Warm the annotation for THIS screen's genes, off the GUI thread. One join covers the whole table -- 400 genes cost the same 21 ms as one -- so every click afterwards is a dictionary lookup.

### lines 2391-2395

```python
self._compartment = None
```

THE COMPARTMENT MENU IS BUILT FROM THE TABLE, so it has to be rebuilt when the table changes. Built once in __init__ it is built from no frame at all, which is an empty submenu that never appears and a new screen would otherwise be offered the last one's compartments.

### lines 2397-2405

```python
self._threshold_method = DEFAULT_THRESHOLD_METHOD
```

AND SO DO THE EFFECT CUT AND THE AXIS WINDOW, which they did not. Measured on the real panel: after typing an x-range of (-1.5, 1.5) and a 2-spread cut on run A, opening run B drew run B inside run A's window with run A's cut -- a picture nobody chose, with nothing on screen saying where it came from. The other four resets in this block exist for exactly that reason; these two were missing.

A run RETURNED TO gets its own back at the end of this method (`_restore_plot_state`), so the reset costs nothing a user built.

### lines 2414-2423

```python
self._level = self._default_level()
```

THE DEFAULT LEVEL IS READ OFF THE TABLE, not asserted.

Defaulting to "grna" unconditionally is what fixed the four-fold duplication -- a gene drawn once per guide -- and it is right for the table a mixed or hierarchical run writes, which carries both levels. It is WRONG for the table instruction 128 R produces: the separate gene fit writes `results_gene.csv`, whose every row is a gene term, and a guide filter over it selects nothing. The panel then draws an empty volcano with a full coefficient table beside it, which reads as a broken plot rather than as an empty filter.

### lines 2430-2433

```python
self.refresh_views()
```

EVERY TAB THROUGH ONE PATH. A new table and a change of the gene/guide filter draw the panel with the same method, so the two cannot leave it in two different states -- which is the whole of instruction 128 L.

### lines 2436-2439

```python
note = self.both_levels_note()
```

BOTH FITS ARE ANNOUNCED ON LOAD. A run at level='both' writes two tables and the panel opens on one; until this line nothing said the other existed, and a user who ran glm reported "it only runs once" about a run that had written both.

### lines 2444-2446

```python
self.say(f"{self._status} Colouring: {self._colour_by_note}.")
```

SAID, not swallowed. A colouring the user expected and cannot find is a bug report; the same colouring listed with the reason it is useless is an answer.

### lines 2448-2452

```python
self._restore_plot_state(source)
```

AND A RUN COME BACK TO GETS ITS PLOT BACK. Last, so it wins over every default the lines above chose from the table -- which is the whole point: those defaults are right the FIRST time a run is opened and wrong every time after, because by then the user has said what they want to look at.

## RegressionResultsPanel.plot_state

### lines 2473-2475

```python
reader = getattr(self.volcano, "pinned_limits", None)
```

THE PLOT'S OWN ANSWER, through its public method (116's last private coupling). `getattr(volcano, "_pinned")` was reaching into another module's attribute for a question that module can answer.

### lines 2488-2490

```python
"x_limits": pinned.get("x"),
```

ONLY WHAT THE USER PINNED. Storing the view range would freeze an auto-ranged plot at whatever it happened to show, so a run returned to would stop following its own data.

## RegressionResultsPanel.apply_plot_state

### lines 2518-2520

```python
blocked = self._colour_by.blockSignals(True)
```

Blocked: the redraw below covers it, and letting the combo fire here would draw the panel against a level that has not been re-offered yet.

### lines 2540-2543

```python
try:
```

THE ROW MAY NOT BE THERE. A saved selection is a feature NAME, and the level restored above may filter it out -- a gene picked at level=None is not in the guide table. Missing is not an error; it is a row the user cannot currently see.

## RegressionResultsPanel.apply_workspace_state

### lines 2591-2593

```python
if path and os.path.exists(path):
```

LOADED, not just assigned. `_path` without the table behind it is a panel claiming to show a run it has not read, and every diagnostic tab would draw the previous run's numbers under the new run's name.

## RegressionResultsPanel._mark_the_level_on_the_plots

### lines 2677-2688

```python
self._offer_levels()
```

THE VOLCANO ONLY, and deliberately. The diagnostics carry the numbers they exist for -- the inflation factor, the control medians, how many genes rest on one guide -- and writing the level sentence over those would trade a panel's whole content for something the header already says. The volcano is the plot the report was about and the one a user looks at first.

THROUGH `_offer_levels`, NOT `set_status_note`. The click slot is rewritten by every click, so the sentence was gone the first time the plot was used; `offer_levels`' own note slot is durable. One call keeps the control, its counts and its sentence in step, which is why this is a delegation and not a second copy of the sentence.

## RegressionResultsPanel.refresh_views

### lines 2709-2712

```python
self._say_which_family()
```

THE LABELS STILL GO ON. A panel with no table yet is still a panel whose filter is set to something, and tab labels that disagreed with `level()` for as long as it took a run to finish would be the exact failure this method exists to prevent.

### lines 2718-2722

```python
self.table.configure(significance_filter=(kind == "p-value"))
```

"significant only" cuts on `value <= alpha`. That is the right way round for a p-value and exactly backwards for a selection frequency, where the interesting rows are the HIGH ones -- so on a penalised backend the checkbox is taken away rather than left to hide every feature the bootstrap kept.

### lines 2724-2726

```python
significance = None
```

The table prefers a CORRECTED column when there is one, and that is the right cut; the detected column is only needed when the p-value is spelled some way the table would not recognise.

### lines 2738-2741

```python
self._draw_guide_support(frame)
```

THE FULL TABLE, on purpose. A gene's concordance is how its GUIDES agree, so the guide rows are what the number is made of -- see `_draw_guide_support`, which narrows which genes are LISTED without touching how any of them was computed.

## RegressionResultsPanel._analysis_path

### lines 2755-2760

```python
if columns:
```

THE TABLE DECIDES, and only a table with no columns at all defers to the settings. A saved settings file carries the MODULE's default inference, not what the run did -- an OLS folder beside this one says inference='nonparametric' and was fitted, so reading the settings first labelled an unbounded OLS coefficient a partial correlation. The permutation writes its columns every time.

## RegressionResultsPanel._gene_terms

### line 2785, trailing  _(unsure)_

```python
except Exception:
```

hits unavailable

## RegressionResultsPanel._draw_guide_support

### line 2800, trailing  _(unsure)_

```python
except Exception:
```

module unavailable

### line 2804, trailing  _(unsure)_

```python
except Exception:
```

odd table shape

### lines 2813-2815

```python
terms = self._gene_terms(frame)
```

The term each gene is called in the coefficient table, put in the table as a column so the support rows and the agreement points join on the SAME key every other view here uses.

### lines 2831-2837

```python
if self._level == "gene":
```

THE FILTER NARROWS WHICH GENES ARE LISTED, never how one was measured. "genes only" keeps the genes the fit gave a gene-level term -- the ones whose dot is on the filtered volcano -- and drops the genes that exist here only as a bundle of guides. "guides only" drops nothing, because every row of this table IS a gene's guides; that is stated on the tab rather than left to look like a filter that failed to fire.

### lines 2848-2851

```python
usable = table["feature"].notna().all() and table["feature"].is_unique
```

A gene with no gene-level term has no key. Offering `feature` as the key column anyway would make every such row unselectable AND make the column non-unique on None; the table checks, so say nothing and let it fall back.

## RegressionResultsPanel._draw_guide_support.verdict

### lines 2819-2820

```python
def verdict(row):
```

A verdict column, because "n_guides=1" is a fact and "this hit rests on one guide" is what the reader needs to take from it.

## RegressionResultsPanel._show_significance

### lines 2906-2910

```python
keys = self._keys_for(frame)
```

THE KEYS GO IN WITH THE VALUES, IN FRAME ORDER. Both are taken positionally out of the same frame, so they stay aligned however the plot reorders them afterwards -- and the Q-Q reorders them completely, which is the whole point of handing them over rather than letting the plot infer a row from a drawing position.

## RegressionResultsPanel.level_counts

### lines 3146-3152

```python
return {None: int(len(frame)),
```

NOT A COMPLEMENT. `gene` used to be "every row that is not a guide", which counts the intercept and the plate row/column terms as genes and on a run fitted at both levels it counts the GUIDE fit's intercept as one. The two families are counted separately, so a row in neither is in neither count and the menu can be read as an inventory: 790 + 381 = 1171 on the real screen, where the complement gave 789 + 382.

## RegressionResultsPanel._offer_levels

### lines 3190-3192

```python
blocked = self._level_box.blockSignals(True)
```

REFILLED WITHOUT FIRING. `activated` is a person's choice only, so this cannot re-enter `set_level` -- blocked as well, because a future `currentIndexChanged` here would, and silently.

## RegressionResultsPanel.family_note

### lines 3286-3288

```python
missing = self.missing_level_note()
```

A LEVEL WITH NO ROWS IS NOT A NARROWER FAMILY, IT IS NO FAMILY. Saying "the inflation figure is this family's" about zero tests describes a diagnostic that is not on screen and cannot be.

## RegressionResultsPanel.set_level

### lines 3320-3323

```python
missing = self.missing_level_note()
```

THE REASON LEADS WHEN THERE IS NOTHING TO DRAW. "0 of 789 coefficients — genes only" is arithmetic; it does not tell the reader that this run has no gene fit, which is what they are looking at an empty tab wondering about.

## RegressionResultsPanel._say_which_family

### lines 3365-3368

```python
plot.plot.setTitle(
```

THE TITLE, NOT THE STATUS LINE. A plot's status is overwritten by whatever was last clicked -- `note_selection` does exactly that -- so a family written there is gone the moment the reader uses the panel. A title is not.

## RegressionResultsPanel._levels_of

### line 3404, trailing  _(unsure)_

```python
except Exception:
```

hits unavailable

## RegressionResultsPanel._offer_p_values

### line 3437  _(unsure)_

```python
self.volcano.offer_p_values([])
```

The column is there and it is the raw p under another name.

## RegressionResultsPanel._offer_compartments

### lines 3555-3556

```python
options.append(("all localisations",
```

ALL, asked for on 2026-08-20. Second, so the one-at-a-time reading the house style prefers is still what a user lands on first.

## RegressionResultsPanel.set_compartment

### lines 3577-3580

```python
from ...localisation import of as compartment_of
```

ITS OWN SENTENCE. `mask` takes ONE compartment, so the branch below would hand it the sentinel and report "0 annotated \x00all-localisations" -- a number about nothing, printed confidently.

## RegressionResultsPanel.set_baseline

### lines 3623-3625

```python
self.say(chosen.sentence
```

THE REASON, WHEN THERE IS ONE. A request that could not be honoured silently falling back to zero is a user who believes they are reading control-relative effects and is not.

## RegressionResultsPanel._redraw_volcano

### lines 3641-3645

```python
try:
```

WHAT THE HORIZONTAL AXIS IS. The permutation path copies its partial correlation into `coefficient` so the rest of the screen can read one name, which leaves the axis calling a bounded correlation a coefficient. Named per redraw rather than at load, because a panel can be handed a different run without being rebuilt.

### lines 3651-3655

```python
p_column = column if kind == "p-value" else "\0no p-value"
```

A volcano's y-axis IS -log10(p). Where there is no p-value the axis has nothing to be, so the plot is left empty on purpose and says why -- rather than plotting the OLS-style number a penalised fit carries, which would look exactly like a volcano and be one of a quantity nobody tested.

### lines 3662-3663

```python
from ...baseline import apply as apply_baseline
```

MEASURED FROM WHATEVER THE USER CHOSE, on a copy. The run's own table is not shifted under the coefficient table beside it.

### lines 3673-3675

```python
category = self._colour_by.currentData()
```

THE LOPIT OPTION IS DERIVED, so it is materialised onto the copy rather than looked up as a column. On the copy, not the run's own table -- the same rule the baseline follows.

### lines 3702-3710

```python
effect_threshold=self._current_threshold(),
```

THE CUT THE MENU COMPUTED, actually drawn.

Reported 2026-08-17: "the coefficient threshold still dosnt work". It did not: the seven methods, the multiplier and the status sentence all landed, and the NUMBER was never handed to the plot -- `set_results`'s `effect_threshold` defaults to None, so every method redrew the same volcano with no line and only the sentence changed. A feature whose every visible part works except the one that draws it.

### lines 3714-3720

```python
self.volcano.set_keys(())
```

AN EMPTY DRAW HAS TO EMPTY THE IDENTIFIERS TOO. `set_results` returns early on a frame with no rows, and the early return is the one path through the plot that does not re-key it -- so after choosing a level this run has none of, the plot showed nothing and still answered `highlight_key` for the 789 guides it drew a moment ago. A selection that rings a point on an empty plot is the linkage reporting a hit it does not have.

### lines 3726-3728

```python
if self._selected_key is not None:
```

THE SELECTION SURVIVES A SETTINGS CHANGE. Changing the colouring redraws from scratch; without this the ring the user was reading disappears and they have to find their guide again.

## RegressionResultsPanel._tested_mask

### line 3797, trailing  _(unsure)_

```python
except Exception:
```

hits unavailable

## RegressionResultsPanel._draw_controls

### lines 3818-3820

```python
if key_column is not None:
```

Sliced out of the frame WITH their values, so a dot's row travels with it into a group that is drawn in a different order from the table.

## RegressionResultsPanel._select_from_a_plot

### lines 3858-3859

```python
QTimer.singleShot(0, lambda k=str(key): self.table.select_key(k))
```

A single shot rather than a direct call: see the docstring. The bound method keeps the panel alive for the one turn it needs.

## RegressionResultsPanel._reachable

### lines 3879-3883

```python
if str(key) not in set(features):
```

A KEY THE TABLE DOES NOT HOLD AT ANY LEVEL IS REACHABLE, which reads oddly and is right: moving the filter cannot produce a row that does not exist, so the only thing it would achieve is rearranging the panel around a click nobody can honour. The plot that emitted it reports the miss; that is its job, not this one's.

## RegressionResultsPanel._select_key

### lines 3904-3907

```python
for histogram in (self.p_values, self.effect_distribution):
```

A histogram has no point to ring, but it can outline the bar the coefficient falls in, which is the honest equivalent. No note goes with it: a bar is a hundred rows, and printing one row's name beside it would read as a claim that the bar IS that row.
