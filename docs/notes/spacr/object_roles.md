# Notes from `spacr/object_roles.py`

Prose lifted out of `spacr/object_roles.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [_recase](#_recase) (1 entry)
- [setting_label](#setting_label) (1 entry)

## Module level

### line 14  _(unsure)_

```python
from .schema import (  # noqa: F401
```

These schema sets remain re-exported for existing ``object_roles`` consumers.

### lines 106-118

```python
"log_x": "Logarithmic x",
```

THE THIRD SENSE OF "log", and the reason these are spelled out rather than left to the humaniser. `dog` and the Laplacian-of-Gaussian `log_*` suffixes are handled in CASED_TERMS and CASED_PHRASES below; these three are the ORDINARY logarithm, and "Log x" is ambiguous in English before any translator sees it. Every machine translator read the logbook: German "Protokoll x" and "Protokollieren y", Spanish "Registro x", French "Journal x", and Chinese rendered `log_x` as "彩票X" -- lottery.

Saying "logarithmic" removes the ambiguity at the source instead of correcting nine locales separately, which is the same fix as `dog` and for the same reason: the translators were not wrong about the word they were given. It is also more accurate English -- all three are log10 or log(x + 1e-6), not "log" in any other sense.

### lines 122-152

```python
"src": "Source",
```

"Src" is an abbreviation of an abbreviation: the humaniser capitalises the key and stops there, so the field that asks for the images read "Src". Asked for on 2026-09-01 -- "path should always just say path". The KEY stays `src`; every settings CSV in existence uses it.

NOT applied to regression, which overrides this to "Output directory" in `settings_model._label_for`. That one is not an abbreviation, it is a more specific true statement -- regression's `src` is where results are written, not where images are read from. SOURCE, NOT PATH, AND THIS IS THE SECOND RENAME. The humaniser capitalises the key and stops, so the field that asks for the images read "Src" -- an abbreviation of an abbreviation. It was renamed to "Path" on 2026-09-01 ("path should always just say path"), and to "Source" on 2026-09-04, to standardise one word across every surface that names where a run reads its input. Both are recorded because the second reverses the first, and a reader who finds only the current answer cannot tell a decision from an oversight.

THE KEY STAYS `src`. Every settings CSV in existence uses it, and this renames the label, not the setting.

ONLY `src`. The other path-like keys -- `model_path`, `custom_model_path`, `organelle_unet_model_path` -- name a MODEL, not a source, and calling them Source would be a lie that reads as a standard. Measured across all 508 keys in `spacr.settings`: `src` is the only source among them.

NOT applied to regression, which overrides this to "Output directory" in `settings_model._label_for`. That one is not an abbreviation, it is a more specific true statement -- regression's `src` is where results are written, not where images are read from.

### lines 154-156

```python
"sample": "Sample size limit",
```

"Sample" reads as the thing being sampled; it is a CAP on how many crops are drawn. The key stays `sample` -- every settings CSV in existence uses it, and this renames the label, not the setting.

### lines 158-163

```python
"power": "Statistical power",
```

These keys belong to the statistical-power simulator, not electrical power.  Leaving the ordinary underscore humaniser to infer their labels produced awkward English ("Power n genes") and led translation models to choose electrical/mechanical terminology in several languages. Qualify the whole family at its single label source so every settings surface and every source-hashed locale catalog carries the same meaning.

### lines 199-206

```python
"dog": "DoG",
```

DIFFERENCE OF GAUSSIANS, and the only reason it is spelled out here is that the machine translators read the lower-case word as the animal. "Organelle 1 — Dog sigma high" became "Hundesigma hoch" in German, "Chien sigma haut" in French, "강아지 Sigma" (puppy) in Korean and "狗 Sigma" in Chinese, in the label of a blob-detector parameter. `dog` appears in exactly two suffixes in the whole vocabulary, `dog_sigma_high` and `dog_sigma_low`, so there is no ambiguity to weigh -- it is always skimage's `blob_dog`.

### lines 318-320

```python
"FT": "flow_threshold",
```

All six landed in b7ae412af (2026-09-02), "retire organelle's duplicate size settings, rename four families" -- 90 settings across 26 organelle slots. None of them got a migration, which is what this table repairs.

### lines 327-340

```python
"min_split_area": "minimum_area_to_split",
```

391, 2026-09-12. Both are honest corrections: `min_split_area` is NOT a minimum object area -- an object below it is kept, it is simply never split -- and `min_distance` is the minimum separation between watershed seeds, which means nothing outside that algorithm.

NOTE THE CHAIN THIS CREATES, and that it is why the resolver walks to a fixed point rather than taking one step:

<role>_min_object_area  ->  <role>_min_split_area >  <role>_minimum_area_to_split

A file written before b7ae412af needs BOTH hops. One hop would leave the value on `_min_split_area`, which nothing reads any more -- the same silent loss this table exists to prevent, one rename later.

### lines 482-484  _(unsure)_

```python
ANCHOR_COLUMN: Dict[str, str] = {
```

The anchor: one concept, two column names

## _recase

### lines 233-235

```python
lowered = text.lower()
```

PHRASES FIRST. A phrase rule exists because its word is ambiguous on its own, so applying the word rules first would settle the ambiguity the wrong way and leave nothing for the phrase rule to match.

## setting_label

### lines 269-272

```python
role = organelle_role_of(key)
```

RESOLVED FROM THE KEY, not by looping the four roles the schema segments. A settings file may carry any slot the vocabulary allows, and one that fell outside those four rendered as "Organellee channel" -- the raw suffix -- instead of "Organelle 5 — Channel".
