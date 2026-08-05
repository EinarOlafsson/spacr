# spaCR notebooks

One notebook per workflow, each with what it does, when to reach for it,
what it leaves behind, and a runnable example.

Every function name, signature and default in these notebooks was read out
of the installed package rather than written by hand, so they describe the
version you have. Where a notebook prints the settings, treat that output
as the reference — not the prose around it.

## The main pipeline, in order

| | Notebook | Step |
|---|---|---|
| 1 | `01_generate_masks` | raw images → per-object masks |
| 2 | `02_measure_and_crop` | masks → measurements + single-object crops |
| 3 | `03_classify_computer_vision` | crops → a trained network and per-object scores |
| 3′ | `04_classify_machine_learning` | measurements → a tabular model, when the phenotype is already measured |
| 4 | `05_map_barcodes` | sequencing reads → per-well guide counts |
| 5 | `06_regression` | scores + counts → the ranked gene list |

Steps 3 and 3′ are alternatives. Use the network when the phenotype is
visible but hard to write down; use the tabular model when the features you
already measured capture it.

## Segmentation

| Notebook | Use |
|---|---|
| `08_train_cellpose` | fine-tune on your own annotations when stock models mis-segment |
| `09_apply_cellpose` | run a trained model over a folder |
| `10_test_cellpose` | score a model against ground truth before trusting it |

## Looking at a screen

| Notebook | Use |
|---|---|
| `07_image_umap` | embed crops in 2-D with the images shown — clusters and batch effects first appear here |
| `25_score_heatmap` | per-well scores in plate geometry, where edge effects are obvious |
| `30_volcano_plot` | effect size against significance |
| `11_activation_maps` | which pixels drove a classifier's decision |
| `20_activation_analysis` | summarise those maps per class |
| `24_interpret_vision_model` | relate network scores back to readable features |

## Assays

| Notebook | Readout |
|---|---|
| `12_recruitment` | marker localisation to the pathogen |
| `13_plaque_assay` | plaque count and size |
| `14_motility_assay` | movement through a timelapse |
| `15_replication` | parasites per vacuole |
| `16_invasion` | inside versus outside the host cell |
| `17_endodyogeny` | division state from morphology |
| `18_percent_positive` | fraction of objects above threshold |
| `19_count_phenotypes` | counts per annotated class |

## Sequencing and downstream

| Notebook | Use |
|---|---|
| `26_sequencing_stats` | read quality and depth — run before believing any count |
| `23_compare_reads_to_scores` | catch plate mix-ups before they become a result |
| `27_post_regression_analysis` | coefficients → figures and rankings |
| `28_go_term_enrichment` | what the hits have in common |
| `29_gene_phenotype_plots` | one gene against the screen distribution |

## Models

| Notebook | Use |
|---|---|
| `21_model_knowledge_transfer` | seed a new model from a trained one |
| `22_model_fusion` | combine models whose errors differ |

## Two shapes of notebook

Most workflows take a **settings dictionary**. Those notebooks write out
the dictionary in full — every key the function accepts, on its own line,
with its real default and a comment saying what it controls. Edit values
in place and delete nothing: a key left at its default behaves exactly as
if it were absent. About 630 settings are documented this way across the
fourteen of them.

The rest take **plain arguments**. Their notebooks show the real signature
and leave the call commented out, deliberately: filling in arguments that
were never verified against your data would be a notebook that looks
runnable and is not. Read the signature, then write the call.

## Before you run anything

* `src` is a placeholder in every notebook. Change it first.
* spaCR writes beside the source folder, so a plate stays self-contained
  and re-running does not clobber a different experiment.
* Want more log detail? Preferences → Logging, or `SPACR_LOG_LEVEL=DEBUG`
  before starting Jupyter.

The previous generation lives in `legacy/`, kept because existing links
point at it.

## Elsewhere

* GUI, same workflows as forms — `python -m spacr`
* Narrated walkthroughs — <https://einarolafsson.github.io/spacr/tutorials/>
* API reference — <https://einarolafsson.github.io/spacr/>
