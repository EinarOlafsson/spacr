# Notes from `spacr/feature_dict.py`

Prose lifted out of `spacr/feature_dict.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (16 entries)
- [Concept](#concept) (1 entry)
- [_entry](#_entry) (2 entries)
- [_parse_organelle_summary](#_parse_organelle_summary) (1 entry)
- [parse_column](#parse_column) (12 entries)
- [_example_columns](#_example_columns) (2 entries)
- [search_features](#search_features) (7 entries)
- [_table_measurement_units](#_table_measurement_units) (3 entries)
- [describe_database](#describe_database) (2 entries)
- [_jsonable](#_jsonable) (1 entry)
- [_markdown](#_markdown) (1 entry)

## Module level

### lines 51-60

```python
__all__ = [
```

PANDAS IS NOT IMPORTED HERE. Everything above the export section is strings and parsing: what a measured column is called, what it means and which object it belongs to. Only `describe_database` and the export path below build or read a DataFrame, and they import pandas themselves.

The Feature Dictionary panel imports this module for those strings, and that panel registers its app and its stylesheet block at launch -- so a module-level pandas here was several hundred modules and a good fraction of a second spent before the window drew, on behalf of a user who may never open it.

### lines 404-409

```python
_INTENSITY = (
```

Intensity is read from the merged stack, which _merge_file (io.py:2367) builds from ``<src>/stack`` — the *raw* concatenated channel arrays. The percentile-normalised copies produced by concatenate_and_normalize live in a separate folder and are used for segmentation, not for measurement. The only transformation measure.py applies is a dtype promotion to uint16 for arrays that are neither uint8 nor uint16 (measure.py:914-917).

### lines 414-416

```python
_INTENSITY_SUM_3D = (
```

The sum is np.sum(region.intensity_image[region.image]) — a plain sum over the object's elements with no spacing factor, so in 3-D it is intensity x voxel count and NOT intensity x um^3, whatever the voxel size says.

### lines 701-702  _(unsure)_

```python
"centroid_weighted_z": PropertyInfo(
```

3-D centroids, named by axis

(measure.py:_CENTROID_AXES_3D / _rename_3d_centroids)

### line 1001

```python
"manders_m1": PropertyInfo(
```

Manders (measure.py; unconditional since 2026-09-02)

### lines 1304-1307

```python
"periphery_<p>_percentile": PropertyInfo(
```

Retained so that a column read out of a database that has not been migrated yet — an old file opened outside spaCR, a CSV exported by an older release — is still explained rather than reported as unknown. The description is the same measurement; only the spelling differs.

### line 1531

```python
"organelle_summary_organelle_ch<c>_mean_intensity_per_<parent>": PropertyInfo(
```

Legacy spellings, kept so an un-migrated database is still described.

### lines 1584-1588

```python
"before_filtration": PropertyInfo(
```

pivoted_counts (mask stage, one row per FIELD) spacr.io._save_object_counts_to_database writes one object_counts row per (file, count_type) with count_type = f'{object_type}{added_string}'; spacr.utils._pivot_counts_table then pivots count_type into columns, so each suffix below becomes '<object><suffix>' in pivoted_counts.

### lines 2017-2022

```python
"measurement_ndim": PropertyInfo(
```

measurement provenance stamp spacr.measure.MEASUREMENT_STAMP_COLUMNS, written onto every row of every object table by spacr.utils._merge_and_save_to_database. These five are what makes the conditional units above resolvable: measure.py records the unit on the row instead of renaming <object>_area, so the columns below are the only thing that says which quantity that column holds.

### lines 2091-2093

```python
_LINK_COLUMNS: dict[str, PropertyInfo] = {
```

Per-object-type parent/child link columns produced by the morphology merge: measure.py:183 (nucleus <- cell_to_nucleus), 194 (pathogen <- cell_to_pathogen) and 208 (organelle <- _map_child_to_parent), all prefixed at measure.py:225.

### lines 2174-2184

```python
_ALL_OBJECTS = OBJECT_TYPES
```

scope — which objects a feature exists for, and how channels enter it

A column name says which object it came from; it does not say which objects the feature is written for AT ALL. `nucleus_periphery_mean` exists and `cell_periphery_mean` does not, because `_intensity_measurements` guards the periphery block with `if ls[j] in ('nucleus', 'pathogen', 'organelle')` (measure.py:1207) — a fact nobody can read off the name, and the reason a user searching for "the periphery of a cell" finds nothing and assumes the dictionary is broken. Every entry below was read off the emitter.

### lines 2667-2668

```python
_DUP_SUFFIX_RE = re.compile(r"^(?P<base>.+)__dup(?P<idx>\d+)$")
```

Current form written by spacr.utils._check_integrity. Unambiguous, so it is matched BEFORE the legacy positional form above.

### lines 2671-2672

```python
_PARAMETERIZED: tuple[tuple[re.Pattern[str], str], ...] = (
```

(regex, KNOWN_PROPERTIES key). Order matters only in that each regex is anchored and mutually exclusive.

### lines 2679-2680  _(unsure)_

```python
(re.compile(r"^periphery_(?P<p>\d+)_percentile$"), "periphery_<p>_percentile"),
```

Pre-migration word order. Still matched so an un-migrated database is described; spacr.utils.rename_columns_in_db renames these on first read.

### lines 2685-2687

```python
(re.compile(r"^neighbors_within_(?P<r>\d+)$"), "neighbors_within_<r>"),
```

The neighbourhood radius is part of the name, the same way the percentile is in percentile_<p>: two plates measured at different radii carry different columns rather than the same column meaning two things.

### lines 3617-3618

```python
"key",
```

Appended, never inserted: the exported CSV is a file people diff, and `measurement_units` is pinned to the slot right after `unit`.

## Concept

### lines 2448-2454

```python
@dataclass(frozen=True)
```

concepts — the words a user actually searches with

Nobody types "equivalent_diameter_area". They type "size", or "how big", or "shape". A dictionary that only matches the naming scheme is a dictionary for people who already know the naming scheme.

## _entry

### lines 2744-2745

```python
basis = measurement_units
```

Only a conditional unit is affected by the stamp, so a column whose unit is fixed never claims to have been resolved against one.

### lines 2750-2751  _(unsure)_

```python
objects: tuple[str, ...] = ()
```

Metadata and link columns are not per-object features and have no scope row; their provenance string already names the writer.

## _parse_organelle_summary

### lines 2832-2835

```python
m = _ORG_SUMMARY_LEGACY_CH_RE.match(name)
```

The pre-migration ch<c> spelling resolves to its own curated entry, which says so — reporting it under the canonical key would tell a user reading an old database that they are looking at a name their file does not contain.

## parse_column

### lines 2963-2964

```python
info = META_COLUMNS.get(name)
```

1. exact metadata match wins over any structural interpretation, so that e.g. cell_id is an identifier and not a 'cell' feature named 'id'.

### lines 2969-2973

```python
embedding = _parse_embedding(name)
```

1b. embedding dimensions (spacr/embeddings.py). BEFORE the object prefix is looked for, because these carry no object prefix at all they are keyed to an object id rather than named for an object type, and falling through to the structural parse would classify every one of them as "unknown" and hide the whole family from the pickers.

### lines 2978-2996

```python
bystander = KNOWN_PROPERTIES.get(name) if name in _BYSTANDER_COLUMNS \
```

1c. infection-neighbourhood columns (spacr/bystanders.py), for the same reason as the embeddings above and with the same consequence if it is skipped. `is_bystander`, `is_distal` and `distance_to_infected` carry NO OBJECT PREFIX -- they are written onto the cell table by `measure._with_bystanders` and named for what they say rather than for the object they say it about -- so the structural parse at step 3 returns `object_type is None` and they come back as "unknown".

Measured before this branch existed: all three landed in

`family/unknown` with no object group at all, so the whole family was invisible to `column_groups.classify` and therefore to every picker, regression selection and hit call that groups by family. Their `KNOWN_PROPERTIES` entries have said `"spatial"` since they were written; nothing was reading them.

NOT A NEW FAMILY. `spatial` already covers this in its own words "where an object ... sits relative to ... other segmented object types ... same-type neighbourhood and touching measurements" -- and a sixth vocabulary for the same concept is the mistake 377 names.

### lines 3003-3004

```python
summary = _parse_organelle_summary(name, measurement_units)
```

2. per-parent organelle summaries, before the object prefix is stripped (the prefix 'organelle_' would otherwise swallow the family name).

### line 3009

```python
object_type: str | None = None
```

3. object prefix (measure.py:225 and measure.py:395)

### line 3021  _(unsure)_

```python
link = _LINK_COLUMNS.get(rest)
```

4. parent/child link columns, e.g. nucleus_cell_id, organelle_cell

### lines 3027-3028

```python
m = _RAD_DIST_RE.match(rest)
```

5. radial distribution: the channel index sits AFTER the family token (measure.py:444), so it needs its own rule.

### line 3041

```python
channels: list[int] = []
```

6. up to two channel_<n> infixes (measure.py:395 and measure.py:429)

### lines 3053-3056

```python
m = _DOUBLE_PREFIX_BLUR_RE.match(rest)
```

7. blur, whose emitted name carries the object/channel prefix twice: measure.py:393 writes '<obj>_channel_<i>_blur' into the frame, then measure.py:395 prefixes every non-label column with '<obj>_channel_<i>_' again, so the stored column is '<obj>_channel_<i>_<obj>_channel_<i>_blur'.

### lines 3097-3098  _(unsure)_

```python
for obj in _OBJECT_TYPE_MATCH:
```

8. a pandas merge suffix appended when object tables are joined (spacr.io._read_and_join_tables uses suffixes=('', '_<entity>')).

### lines 3118-3119

```python
m = _DUP_SUFFIX_RE.match(rest)
```

8b. Current spacr.utils._check_integrity suffixes a repeated column with '__dup<n>'. Unambiguous, unlike the legacy positional form below.

### lines 3136-3137

```python
m = _DEDUP_SUFFIX_RE.match(rest)
```

9. Databases written before that change carry the positional index instead, which produces names that look parameterised.

## _example_columns

### lines 3338-3339  _(unsure)_

```python
if key.startswith("organelle_summary_"):
```

The organelle summaries are written under their own prefix into their own tables and take no object prefix at all.

### lines 3347-3348  _(unsure)_

```python
if key.startswith("rad_dist_"):
```

rad_dist carries its own channel token after the family name, which is exactly why it needs its own parsing rule.

## search_features

### line 3506

```python
return []
```

An unknown concept filter must not silently widen to everything.

### lines 3509-3510  _(unsure)_

```python
exact_key: str | None = None
```

A whole column name beats every text match: the user pasted the thing they are looking at.

### lines 3513-3515

```python
entry = parse_column(raw)
```

The RAW query, not the lower-cased one: the emitted names are case-sensitive (`M1_correlation_85`, `Pearson_correlation`) and lower-casing here made every colocalisation column unresolvable.

### lines 3534-3535  _(unsure)_

```python
return []
```

A stopword-only query carries no searchable meaning. Letting the raw substring rules below see it makes ``of`` match ``centre_offset``.

### lines 3567-3571

```python
score += max(_concept_rank(name, doc.key)
```

Within a concept, rank by that concept's own key order: each

CONCEPTS entry lists its most characteristic feature first, so a search for "texture" leads with the GLCM homogeneity columns rather than with whichever intensity statistic happens to come first in KNOWN_PROPERTIES.

### lines 3577-3579

```python
if (text not in doc.key.lower()
```

Only when the key itself did NOT match: a metadata doc's example column IS its key, so awarding both would score "voxel_size_z_um" twice for the word "size" and float it above the size features.

### lines 3584-3587

```python
hay = _haystack(doc)
```

ALL the meaningful terms, not any of them. With "any", the query "zzzzz-not-a-feature" scored every entry in the dictionary, because `a` and `not` appear in all of them — a nonsense search came back with 137 confident results.

## _table_measurement_units

### lines 3696-3699

```python
return _LEGACY_UNITS, (
```

Not a guess: before 3-D measurement existed a 3-D mask crashed the morphology pass outright, so an unstamped row cannot be anything but a 2-D pixel measurement. Same rule as spacr.utils._LEGACY_STAMP.

### line 3708, trailing

```python
except sqlite3.Error as exc:
```

unreadable table: describe it, do not fail

### line 3713

```python
resolved = {_LEGACY_UNITS if v is None else v for v in found}
```

A NULL stamp is the legacy 2-D/px row, exactly as spacr.utils reads it.

## describe_database

### lines 3774-3776

```python
df["measurement_units"] = pd.Series(
```

Pandas 3 infers ``str`` for text columns and exposes missing values from that dtype as ``nan``.  Keep the public contract for an unstamped unit: callers receive the Python ``None`` stored by ``FeatureEntry``.

### lines 3783-3784

```python
for col in ("channel", "channel_2"):
```

Keep channel indices as integers-or-missing rather than letting pandas promote them to float and print "channel 0.0" in the exports.

## _jsonable

### line 3819  _(unsure)_

```python
item = getattr(value, "item", None)
```

numpy / pandas scalars

## _markdown

### lines 3904-3905  _(unsure)_

```python
missing = df["object_type"].isna()
```

Collapse every missing marker (None / NaN) onto a single None bucket, so the "no object" section is emitted exactly once.
