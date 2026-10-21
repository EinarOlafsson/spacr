# Count annotated phenotypes

**Purpose.** Count objects assigned to each annotation or predicted phenotype class within experimental groups.

**Recommended use.** Use after manual annotation or classification when class abundance is the required outcome.

**Primary outputs.** A tidy table of phenotype counts by experimental group.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.count_phenotypes`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.count_phenotypes)

```python
count_phenotypes(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.count_phenotypes`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.count_phenotypes)

> This function does not yet expose a settings factory. The list below is recovered from direct settings access in its source, so a key used only on a dynamic branch may be absent.


#### Essential inputs

- **`src`** *(optional)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Training Classes (1 settings)</summary>

#### Training Classes

- **`annotation_column`** *(optional)* — (str) - Integer column in the png_list table that stores manual class labels. The Annotate app adds it with ALTER TABLE if it is absent and writes labels to it. This column provides the reference labels when dataset_mode is 'annotation' and is the fallback when annotation_columns is unset. Supplying it while dataset_mode is unset also selects annotation mode for compatibility with older settings files. Default None. Default: `None`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A tidy table of phenotype counts by experimental group.
