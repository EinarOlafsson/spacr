# Notes from `spacr/qt/regex_detect.py`

Prose lifted out of `spacr/qt/regex_detect.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [validate_records](#validate_records) (1 entry)
- [auto_detect_regex](#auto_detect_regex) (5 entries)
- [_synthesise_regex](#_synthesise_regex) (10 entries)
- [tabulate_records](#tabulate_records) (3 entries)

## Module level

### lines 50-51  _(unsure)_

```python
CQ1 = (
```

Yokogawa CQ1 naming: W<well>F<field>T<time>Z<slice>C<chan> (well is numeric). Matches spacr.utils._get_regex('cq1', ...).

### line 57  _(unsure)_

```python
CANONICAL = (
```

Bare-bones canonical form spaCR generates when auto-normalising

## validate_records

### line 196  _(unsure)_

```python
if "plateID" not in all_group_names:
```

plateID is optional but VERY handy — soft warn

## auto_detect_regex

### line 245  _(unsure)_

```python
best_label = "none"
```

1 + 2: try built-ins, remember the best

### lines 265-277

```python
try:
```

3: INFER IT FROM THE NAMES (instruction 137 B, `spacr.regex_infer`).

BEFORE the old synthesiser, and the difference is what it works from: `_synthesise_regex` builds a template from ONE filename and routinely fails to match its siblings, which is why the hit count above had to be corrected. `regex_infer.propose` aligns the whole set -- the slots that VARY become groups and the parts that never vary are literals -- so on cellvoyager, cq1 and a microscope spaCR has never met it reaches 100% coverage with the right group NAMES.

It is tried against the best built-in rather than instead of it: a bundled pattern that already matches everything is a known-good answer and does not need improving.

### lines 282-290

```python
named = {n for n in _group_names(proposal.pattern)
```

HIT COUNT ALONE IS NOT A RANKING, and this is what it cost:

given `a.txt`, one cellvoyager image and `zz.txt`, inference offered `(?P<group0>[A-Za-z]+)\.txt` -- matching the two files that are not images at all -- and beat the cellvoyager pattern 2 to 1. A catch-all that captures NO ROLE is not a metadata regex however many names it happens to match.

So a proposal has to name at least one of the fields the import actually reads before its count is even compared.

### lines 296-297

```python
pass
```

Inference is an improvement on the fallback, never a requirement: a failure here leaves the old path exactly as it was.

### lines 303-306

```python
try:
```

Report the TRUE hit count. The synthesiser works from a single template filename, so it routinely fails to match its siblings — blindly returning ``n`` told the caller "matched 3/3" for a regex that matched nothing, and the regex editor printed that lie.

## _synthesise_regex

### lines 334-335  _(unsure)_

```python
template = min(filenames, key=lambda s: (len(s), s))
```

Take the "shortest, alphanumeric-only" as template — least likely to have noise like an underscore-suffixed acquisition tag.

### line 338  _(unsure)_

```python
prefix_map = {
```

Map single-char prefixes to standard field names

### line 357  _(unsure)_

```python
wm = re.fullmatch(r"([A-Za-z])(\d{2,3})", tok)
```

Well id shape: single letter + 2-3 digits, or two-letter + digits

### line 363  _(unsure)_

```python
m = re.fullmatch(r"([A-Za-z])(\d+)", tok)
```

Single-letter prefix + digits combos (F001, T0001, ...)

### line 371  _(unsure)_

```python
multi = re.findall(r"([A-Za-z])(\d+)", tok)
```

Multi-prefix runs like "T0001F001L01A01Z01C01"

### lines 382-386

```python
if m:
```

Single letter + digits with an unrecognised prefix (e.g. `W1`, `M12`) — keep the letter literal, let the digits vary so plates that use W2, W3, … still match. Otherwise we'd bake the exact digits from the template into the regex and only match one file.

### line 390  _(unsure)_

```python
if re.fullmatch(r"[A-Za-z0-9]+", tok) and "plateID" not in used_groups:
```

Pure identifier → plateID (only first free identifier)

### lines 395-398

```python
if tok.isdigit():
```

Bare digit run with plateID already spent (e.g. the `0001` in `IMG_0001.tif`) — vary the digits for the same reason we do it for `W1`/`M12` above. Escaping the template's literal digits here produced a regex that matched exactly one file.

### line 402  _(unsure)_

```python
parts.append(re.escape(tok))
```

Fallback: literal escaped shape

### line 404  _(unsure)_

```python
exts = r"(?:tif|tiff|png|jpg|jpeg)$"
```

Suffix: allow any of the common image extensions

## tabulate_records

### lines 409-411  _(unsure)_

```python
def tabulate_records(
```

tabulate_records — plain-text table for the Console

### line 435  _(unsure)_

```python
found: Set[str] = set()
```

Deterministic order — use KNOWN_FIELDS then any extras

### line 452  _(unsure)_

```python
widths = {c: max(len(c), max(
```

Compute per-column widths
