# Notes from `spacr/qt/widgets/qc_summary.py`

Prose lifted out of `spacr/qt/widgets/qc_summary.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_segmentation](#_read_segmentation) (2 entries)
- [_flag_explanations](#_flag_explanations) (1 entry)
- [_read_leakage](#_read_leakage) (1 entry)
- [_read_units](#_read_units) (1 entry)

## _read_segmentation

### lines 133-135  _(unsure)_

```python
def _read_segmentation(src: Any, reader=None) -> QCCard:
```

Readers -- one per source. Each is cheap: listdir, stat, parse.

### line 149, trailing  _(unsure)_

```python
except Exception as exc:
```

defensive

## _flag_explanations

### line 180, trailing  _(unsure)_

```python
except Exception:
```

defensive

## _read_leakage

### line 240, trailing  _(unsure)_

```python
except Exception:
```

defensive

## _read_units

### line 333, trailing  _(unsure)_

```python
except Exception:
```

defensive
