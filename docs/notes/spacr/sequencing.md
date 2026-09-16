# Notes from `spacr/sequencing.py`

Prose lifted out of `spacr/sequencing.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [display](#display) (1 entry)
- [map_sequences_to_names](#map_sequences_to_names) (1 entry)
- [_map_within.resolve](#_map_withinresolve) (1 entry)
- [save_df_to_hdf5](#save_df_to_hdf5) (1 entry)
- [save_unique_combinations_to_csv](#save_unique_combinations_to_csv) (1 entry)
- [get_consensus_base](#get_consensus_base) (2 entries)
- [process_chunk](#process_chunk) (4 entries)
- [process_chunk.single_find_sequence_in_chunk_reads](#process_chunksingle_find_sequence_in_chunk_reads) (3 entries)
- [saver_process](#saver_process) (1 entry)
- [paired_read_chunked_processing](#paired_read_chunked_processing) (7 entries)
- [single_read_chunked_processing](#single_read_chunked_processing) (8 entries)
- [generate_barecode_mapping](#generate_barecode_mapping) (8 entries)
- [barecodes_reverse_complement](#barecodes_reverse_complement) (3 entries)
- [graph_sequencing_stats.find_and_visualize_fraction_threshold._line_plot](#graph_sequencing_statsfind_and_visualize_fraction_threshold_line_plot) (1 entry)
- [graph_sequencing_stats.find_and_visualize_fraction_threshold](#graph_sequencing_statsfind_and_visualize_fraction_threshold) (4 entries)
- [graph_sequencing_stats](#graph_sequencing_stats) (7 entries)

## Module level

### lines 63-64  _(unsure)_

```python
from .runctx import run_context
```

One run id on every log line and every artifact, one seed, and the on_error policy at the per-sample boundary. See spacr.runctx.

### lines 71-73

```python
from .figures.style import ROLES, figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### line 91  _(unsure)_

```python
BARCODE_MISMATCHES = 0
```

Function to map sequences to names (same as your original)

## display

### lines 78-81

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## map_sequences_to_names

### lines 147-162

```python
duplicate_sequences = df["sequence"].dropna().duplicated(keep=False)
```

A SEQUENCE ON TWO NAMES IS DROPPED, NOT A REASON TO REFUSE THE LIBRARY.

The safety property is that such a read must never be attributed: it genuinely cannot be told which guide it came from, and picking one would put another gene's counts on it. That property is kept -- those sequences map to NA, so the reads carrying them fall out of the per-well counts.

Refusing the whole FILE was too blunt. The real tsg101 gRNA library is 1,385 guides of which THREE sequences are shared -- two guides of TGGT1_241310 also appear under TGGT1_411210 and TGGT1_411710 -- so spaCR would not map the maintainer's own screen at all, and the 1,382 unambiguous guides were unusable because of eight rows. A library with a few shared sequences is an ordinary thing; one with NOTHING left after they are dropped is a mis-built or mis-columned file, and that still raises rather than mapping every read to nothing in silence.

## _map_within.resolve

### line 234, trailing  _(unsure)_

```python
if found > 1:
```

ambiguous: two references fit

## save_df_to_hdf5

### line 243  _(unsure)_

```python
def save_df_to_hdf5(df, hdf5_file, key='df', comp_type='zlib', comp_level=5):
```

Functions to save data (same as your original)

## save_unique_combinations_to_csv

### lines 294-299

```python
unique_combinations.to_csv(csv_file, index=False)
```

index=False: the frame comes out of a groupby with as_index=False, so its index is a RangeIndex carrying nothing. Written out, the next chunk read it back as a column named 'Unnamed: 0', summed it with the counts, and wrote a fresh index beside it -- one junk column per chunk, in the count table the whole run exists to produce. A real run is hundreds of chunks.

## get_consensus_base

### line 382  _(unsure)_

```python
if bases[0][0] == 'N':
```

Prefer non-'N' bases, if 'N' exists, pick the other one.

### line 388  _(unsure)_

```python
return bases[0][0] if bases[0][1] >= bases[1][1] else bases[1][0]
```

Return the base with the highest quality score

## process_chunk

### line 399  _(unsure)_

```python
def process_chunk(chunk_data):
```

Core logic for processing a chunk (same as your original)

### lines 460-465

```python
legacy = {
```

THE THREE THIS MODULE HAS ALWAYS DECODED, WRITTEN THE WAY THE FRAME LISTS THEM: the name column each barcode fills, and beside it the barcode's name, its reference table, the regex group that captures it, and the older spelling of that group still accepted. A run that names a barcode set of its own replaces this collection entirely; it is what a run that names none decodes.

### lines 476-480

```python
count_columns=('rowID', 'columnID', 'grna_name'))
```

COUNTED BY ROW, THEN COLUMN, THEN GUIDE, which is not the order the reads are listed in. Both orders are older than barcode sets and both are in files people already have, so the set carries the counting order rather than re-sorting every count table that has ever been written.

### lines 692-697

```python
frame = {'read': consensus_sequences}
```

ONE PAIR OF COLUMNS PER BARCODE, in the order the set lists them. The three shipped barcodes carry the column names this frame has always had, in the order it has always had them, so a run that names no set writes the frame it always wrote. Every entry contributes both of its columns, which is why the fill below never has to ask whether a column it is about to read is there.

## process_chunk.single_find_sequence_in_chunk_reads

### line 650  _(unsure)_

```python
r1_seq, r1_qual = extract_sequence_and_quality(r1_sequence, r1_quality, r1_start, r1_end)
```

Extract the sequence and quality within the defined region

### line 653  _(unsure)_

```python
if len(r1_seq) < expected_end:
```

If the sequence is shorter than expected, pad with 'N's and '!' for quality

### lines 672-673  _(unsure)_

```python
if consensus_seq:
```

Every assigned consensus is padded to ``expected_end`` above; only the absence of an anchored read can skip this fallback.

## saver_process

### line 725  _(unsure)_

```python
def saver_process(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type...
```

Function to save data from the queue

## paired_read_chunked_processing

### line 915  _(unsure)_

```python
save_queue = Queue()
```

Queue for saving

### line 918  _(unsure)_

```python
save_process = Process(target=saver_process, args=(save_queue, hdf5_file, save_h5, unique_combina...
```

Start the saving process

### line 935

```python
r1_lines = [r1.readline().strip() for _ in range(4)]
```

Read the next 4 lines for both R1 and R2 files

### lines 939-940

```python
r1_done, r2_done = not r1_lines[0], not r2_lines[0]
```

Paired files must end together; truncating to the shorter input silently changes per-well counts.

### line 953  _(unsure)_

```python
if not r1_chunk:
```

If the chunks are empty, break the outer while loop

### line 963  _(unsure)_

```python
result = pool.apply_async(process_chunk, (chunk_data,))
```

Process chunks in parallel-

### line 983  _(unsure)_

```python
pool.close()
```

Cleanup the pool

## single_read_chunked_processing

### line 1040  _(unsure)_

```python
save_queue = Queue()
```

Queue for saving

### line 1043  _(unsure)_

```python
save_process = Process(target=saver_process, args=(save_queue, hdf5_file, save_h5, unique_combina...
```

Start the saving process

### line 1057

```python
r1_lines = [r1.readline().strip() for _ in range(4)]
```

Read the next 4 lines for both R1 and R2 files

### line 1060  _(unsure)_

```python
if not r1_lines[0]:
```

Break if we've reached the end of either file

### line 1066  _(unsure)_

```python
if not r1_chunk:
```

If the chunks are empty, break the outer while loop

### line 1076  _(unsure)_

```python
result = pool.apply_async(process_chunk, (chunk_data,))
```

Process chunks in parallel

### line 1085  _(unsure)_

```python
save_queue.put((df, unique_combinations, qc_df))
```

Queue the results for saving

### line 1098  _(unsure)_

```python
pool.close()
```

Cleanup the pool

## generate_barecode_mapping

### lines 1239-1242

```python
global BARCODE_MISMATCHES
```

THE MISMATCH BUDGET FOR THIS RUN. Set here, before any worker is forked, because the mapping happens inside worker processes whose arguments are a fixed tuple assembled in three places -- and a budget is one value for the whole run by definition.

### lines 1254-1257

```python
barcode_set = barcode_set_from_settings(settings)
```

THE BARCODES THIS RUN DECODES. None is the ordinary answer: no settings file written before barcode sets existed names one, and None means the run decodes the plate column, the guide and the plate row from the three reference CSVs, which is exactly what it decoded before.

### lines 1260-1265

```python
barcode_set.resolve_groups(regex)
```

CHECKED ONCE, HERE, rather than in the first worker that reaches it. A regex naming no group for one of the barcodes is a settings mistake, and a set of five barcodes with four groups is the shape of mistake this whole change makes possible -- so it costs a user a second before any FASTQ is opened instead of a chunk's work and a traceback out of a worker process.

### lines 1276-1277  _(unsure)_

```python
with run_context('sequencing', settings) as run:
```

One run over every sample: one id on the log lines and the artifacts, and the on_error policy at the per-sample boundary. See spacr.runctx.

### lines 1280-1283

```python
for attempt in run.policy.attempts_for(key, stage='sample'):
```

on_error, at the per-sample boundary. Until now a single unreadable FASTQ pair took every later sample down with it, and the run still exited 0 -- the folder simply had fewer outputs in it than it had samples.

### lines 1286-1291

```python
reads = samples_dict[key]
```

`.get`, not `[...]`. A sample whose mate could not be identified from its filename used to raise `KeyError: 'R1'` from inside this condition, several frames from the cause and naming neither the file nor the problem. Reported 2026-09-01 after downloading the project's own reads, which ENA names `<run>_1.fastq.gz`.

### lines 1341-1349

```python
expected_end=settings['window_length'],
```

THE SETTING IS `window_length` NOW (364); the PARAMETER keeps its name because these two chunked-processing functions are public and renaming a keyword argument breaks every external caller for a word. The parameter's own docstring already says "window *length*, not an end coordinate", which is what the setting rename fixes for the person reading the panel.

### lines 1366-1369

```python
_run_barcode_qc(settings, dst,
```

The table exists now; QC it while we know which sample it belongs to. Inside the attempt so a per-sample on_error policy still applies, but itself never raising -- see _run_barcode_qc.

## barecodes_reverse_complement

### line 1373  _(unsure)_

```python
def barecodes_reverse_complement(csv_file):
```

Function to read the CSV, compute reverse complement, and save it

### line 1394  _(unsure)_

```python
file_dir, file_name = os.path.split(csv_file)
```

Create the new filename

### line 1399  _(unsure)_

```python
df.to_csv(new_filename, index=False)
```

Save the DataFrame with the reverse complement sequences

## graph_sequencing_stats.find_and_visualize_fraction_threshold._line_plot

### lines 1516-1524

```python
with figure_style(theme_target()):
```

No "are x and y in df.columns?" guard: this is a closure with one call site eight lines below, and `df` there is the results_df built two lines above it with exactly these two columns. The check could not fire, so it was a branch no test could ever reach honestly -- removed rather than excused. THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## graph_sequencing_stats.find_and_visualize_fraction_threshold

### line 1542  _(unsure)_

```python
for threshold in fraction_thresholds:
```

Iterate through the fraction thresholds

### lines 1552-1557

```python
print(f"Closest Fraction Threshold: "
```

THE NUMBER CARRIES THE TARGET IT WAS CHOSEN AGAINST. "Closest Fraction Threshold: 0.0168" on its own is a bare number a reader will attach to whichever threshold they were last thinking about, and a screen that asked for the control-well calibration and did not get it has two candidate sources for exactly this line. The caller says the rest; this says which target it hit.

### lines 1565-1566

```python
plt.axvline(x=closest_threshold['fraction_threshold'],
```

178 A: the reference role rather than black, which spaCR's dark theme makes invisible.

### lines 1580-1581

```python
from .plot import save_figure
```

108 point 6. `format='pdf', dpi=600` was a preference written into a call site: a user who chose PNG at 300 got neither.

## graph_sequencing_stats

### lines 1618-1626

```python
filter_column = _resolve_column(df, settings.get('filter_column'))
```

THE SETTING IS CANONICALISED TOO, NOT ONLY THE FRAME. The line above renames the frame's headers to spaCR's vocabulary, and `settings['filter_column']` is the user's own spelling of one of them so a settings CSV saying `ColumnID` indexed a frame holding `columnID` and every run died with `KeyError: 'ColumnID'`, four frames deep, after the counts had already been read. Found in ~/.spacr/logs/spacr.log.

Instruction 145's rule is one vocabulary; applying it to the data and not to the setting that indexes the data is half a rule.

### lines 1633-1637

```python
closest_threshold = find_and_visualize_fraction_threshold(
```

`.get`, because instruction 135 retired log_x/log_y as settings the axes are chosen automatically and changed on the plot now. This runs on the DEFAULT regression path, whenever fraction_threshold is None, so a subscript here killed every run 25 lines after the one that killed it first.

### line 1643  _(unsure)_

```python
df = df[df['fraction'] >= closest_threshold]
```

Apply the closest threshold to the DataFrame

### line 1646

```python
unique_counts = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().reset_index(name='...
```

Group by 'plateID', 'rowID', 'columnID' and compute unique counts of 'grna'

### lines 1651-1656

```python
df = pd.merge(df, unique_counts, on=['plateID', 'rowID', 'columnID'],
```

Merge the unique counts back into the original DataFrame. unique_counts is one row per well by construction (groupby on exactly this key), df is one row per (well, gRNA): many-to-one. If the right side ever gained a duplicate the plate heatmap below would average a well's gRNA rows more than once and simply show the wrong number, with nothing in the output saying so.

### lines 1662-1672

```python
df['rowID'] = (df['rowID'].astype(str)
```

rowID sometimes arrives as the composite '<plate>_<row>' that count CSVs carry in their 'plate_row' column; plot_plates wants the row alone.

This was guarded by `df['rowID'].str.contains('_').any()` and then run over EVERY row with `x.split('_')[1]`, so one composite value anywhere in the table made the whole column go through an index that the plain values do not have: ['plate1_r1', 'r2', 'r3'] raises IndexError and the caller loses the threshold it had already computed. The [1] was also the wrong token for a plate whose own name contains a separator ('exp1_plate1_r2' gave 'plate1'). Taking the token after the LAST separator is right for both, needs no guard, and leaves a plain 'r2' untouched.

### lines 1678-1684

```python
print(f"fraction_threshold={closest_threshold} chosen from "
```

WHICH QUESTION THIS ANSWERED, SAID WHERE THE ANSWER IS HANDED BACK.

This function does not read `calibrate_fraction_threshold` and must not: see the docstring. What it owes a reader is the source of the number it returns, because a screen that ticked the calibration box and could not run it falls through to exactly this line, and a bare threshold gives them no way to tell the two apart.
