# Notes from `spacr/_v1_v2_bridge.py`

Prose lifted out of `spacr/_v1_v2_bridge.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [v2_channels_from_settings](#v2_channels_from_settings) (2 entries)
- [report_disk_savings](#report_disk_savings) (2 entries)
- [v2_mask_source](#v2_mask_source) (1 entry)
- [v2_mask_source, 2026-09-19](#v2_mask_source-2026-09-19) (1 entry)

## Module level

### line 35  _(unsure)_

```python
("nucleus_channel",       "nucleus"),
```

(settings key,          human name)

## v2_channels_from_settings

### line 74  _(unsure)_

```python
raw = settings.get("channels")
```

Fall back to a top-level `channels` list if the user set that

### line 84  _(unsure)_

```python
chans = [0, 1, 2, 3]
```

Last-ditch default — 4-channel plate

## report_disk_savings

### line 120  _(unsure)_

```python
for extra in (src / "filename_map.csv",
```

Add the filename map + channel-order sidecars

### line 130, trailing  _(unsure)_

```python
v1_estimated_bytes = v2_bytes * 4
```

see docstring for rationale

## v2_mask_source

### lines 203-206

```python
offset = 0
```

One mask and a name that does not match: score it anyway. The channel was written by the run being scored, and refusing over a naming difference would report "no masks" about a plate that has them.

## v2_mask_source, 2026-09-19

```python
if name.startswith(".") or not name.lower().endswith(".npy"):
```

v2 scores its masks by listing `merged/`. On a macOS external volume (GitHub #121 and #117) every `stack_<field>.npy` there has an AppleDouble sidecar, `._stack_<field>.npy`. Each sidecar became a field of its own, failed to load, and was scored as `unreadable_mask`. Measured on the code before this line, with two good fields: "verdict: FAIL - 2 of 4 fields failed (50%): fix the segmentation before running Measure". `seg_qc` defaults to `report`, so every v2 run on such a drive got that false FAIL. It is the same false FAIL `seg_qc._iter_masks` gave v1. Filtered inline because this module imports nothing heavier than numpy, and `spacr.io` imports torch. Reasons for the dot-file rule are in `docs/notes/spacr/io.md` under `_listdir_visible`.
