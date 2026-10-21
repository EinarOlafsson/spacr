# Map sequencing reads to microplate wells

**Purpose.** Process amplicon sequencing reads and quantify guide-RNA assignments for each microplate well.

**Recommended use.** Use after pooled-screen sequencing to construct the guide-count matrix required for genotype-phenotype analysis.

**Primary outputs.** Per-well guide counts and sequencing quality-control summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.sequencing.generate_barecode_mapping`](https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html#spacr.sequencing.generate_barecode_mapping)

```python
generate_barecode_mapping(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.sequencing.generate_barecode_mapping`](https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html#spacr.sequencing.generate_barecode_mapping)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`regex`** *(required)* — (str) - Regex applied with re.match to each extracted read window; it must define the named groups columnID, grna and rowID, whose captured sequences are looked up in the three barcode CSVs. Non-matching reads are silently dropped, so a wrong group name or barcode orientation yields zero counts. The default captures an 8 bp column, 20-21 bp gRNA and 8 bp row barcode.
- **`column_csv`** *(required)* — (path) - CSV mapping column barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim against it with no reverse-complementing, so the sequences must be in the same orientation as the reads - run barecodes_reverse_complement on the file if they are not. Unmatched reads get NA for columnID. Default the bundled spacr/resources/data/barcodes_column.csv; barcode QC (sequencing_qc) instead defaults this key to empty, where the reference is optional.
- **`grna_csv`** *(required)* — (path) - CSV mapping gRNA barcode sequences to gRNA names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so orientation must match the reads (barecodes_reverse_complement flips a file). Rows whose gRNA does not match are written as NA and dropped from the counts. Default: the bundled spacr/resources/data/grna_barcodes.csv.
- **`row_csv`** *(required)* — (path) - CSV mapping row barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so the sequences must be in the same orientation as the reads - use barecodes_reverse_complement to flip the file if needed. Unmatched reads get NA for rowID. Default: the bundled spacr/resources/data/barcodes_row.csv.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Sequencing Input (2 settings)</summary>

#### Sequencing Input

- **`mode`** *(optional)* — (str) - Read-pairing strategy for barcode extraction: 'paired' locates target_sequence in R1 and in the reverse complement of R2 and merges them base-by-base into a quality-weighted consensus; 'single' scans one mate alone, chosen by single_direction. Paired calls barcodes more accurately but discards any read whose anchor is missing from either mate. Default 'paired'. Default: `'paired'`.
- **`single_direction`** *(optional)* — (str) - Which mate to scan when mode is 'single': 'R1' or 'R2'. The chosen file is read as-is with no reverse-complementing, so selecting 'R2' means target_sequence and regex must be written in R2 orientation or nothing will match. Ignored when mode is 'paired'. Default 'R1'. Default: `'R1'`.

</details>

<details>
<summary>Read Parsing (4 settings)</summary>

#### Read Parsing

- **`target_sequence`** *(optional)* — (str) - Constant vector sequence used as the anchor: every read is scanned for an exact match and the barcode window is then sliced relative to that hit using offset_start and window_length. Reads without an exact match are skipped entirely, so it must be error-free and given in the orientation of the read being scanned. Default 'TGCTGTTTCCAGCATAGCTCTTAAAC'. Default: `'TGCTGTTTCCAGCATAGCTCTTAAAC'`.
- **`offset_start`** *(optional)* — (int) - Bases to shift from the start of the target_sequence match to the start of the extracted window; negative values move upstream to capture a barcode preceding the anchor. The start is clamped at position 0, so an over-negative value silently shifts the reading frame and the regex stops matching. Default -8. Default: `-8`.
- **`window_length`** *(optional)* — (int) - Number of bases sliced out of each read starting at offset_start relative to the target_sequence hit; this window is what the regex is matched against. It must span the whole barcode block (column + gRNA + row) or the regex stops matching and reads are dropped; shorter reads are padded with 'N'. Default 89. Default: `89`.
- **`barcode_mismatches`** *(optional)* — (int) - How many mismatched bases a barcode may carry and still be matched. 0, the default and prior behavior, requires an exact match, so a single sequencing error anywhere in a barcode discards the read. A read that falls within the mismatch budget of two barcodes remains unassigned because its source barcode cannot be resolved without risking guide misattribution. Default: `0`.

</details>

<details>
<summary>Output &amp; Storage (4 settings)</summary>

#### Output & Storage

- **`save_h5`** *(optional)* — (bool) - Also write every annotated read (consensus sequence plus its parsed row/column/gRNA barcodes and IDs) to annotated_reads.h5. The per-well counts in unique_combinations.csv and qc.csv are written either way, so set it False unless you need read-level data; True produces a very large file and compression can dominate runtime. Default True. Default: `True`.
- **`comp_type`** *(optional)* — (str) - PyTables compression library used when writing annotated_reads.h5, passed to pandas HDFStore as complib: 'zlib', 'lzo', 'bzip2' or 'blosc'. 'blosc' is far faster at similar file size, 'bzip2' is smallest but slowest. Ignored entirely when save_h5 is False. Default 'zlib'. Default: `'zlib'`.
- **`comp_level`** *(optional)* — (int) - complevel passed to the HDF5 store, 0-9. 0 disables compression (fastest write, largest file); higher values shrink annotated_reads.h5 at increasing CPU cost, and at the top of the range saving can take longer than the barcode mapping itself. Ignored when save_h5 is False. Default 5. Default: `5`.
- **`fill_na`** *(optional)* — (bool) - When a barcode does not match any entry in its reference CSV, count it under its raw sequence instead of dropping it. When disabled, grouping discards unmatched reads without warning; a reference in the incorrect orientation can therefore produce a reduced table rather than an empty one. Enable this setting to quantify the unmapped fraction. Default False. Default: `False`.

</details>

<details>
<summary>Runtime &amp; Reliability (4 settings)</summary>

#### Runtime & Reliability

- **`chunk_size`** *(optional)* — (int) - Number of FASTQ reads read into memory and handed to each worker batch. Larger chunks cut per-batch overhead and make the progress bar coarser but raise peak RAM per job; smaller chunks stream more gently on low-memory machines. Also sets how many reads are processed when test is True. Default 100000. Default: `100000`.
- **`n_jobs`** *(optional)* — (int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None. Default: `None`.
- **`ram_guard`** *(optional)* — (bool) - Keep worker pools from filling RAM. Before any module starts its workers, spaCR estimates each worker's memory from one input (a field, mask, crop batch or table) and lowers n_jobs to the count that leaves 12.5% of RAM free, printing a warning; the GUI asks first. Off keeps your n_jobs, and Measure fields still wait while free RAM is below that reserve. Default True. Default: `True`.
- **`test`** *(optional)* — (bool) - In classifier training, run the held-out evaluation pass, either with train or alone to score an existing model. In the sequencing barcode mapper, process only the first read chunk and print a preview so the regex and barcode CSVs can be validated quickly. Default False. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-well guide counts and sequencing quality-control summaries.
