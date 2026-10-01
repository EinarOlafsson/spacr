# Notes from `spacr/figure_font.py`

Prose lifted out of `spacr/figure_font.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## use_open_sans_for_figures

### lines 75-77

```python
return _resolved
```

ALREADY DONE THIS PROCESS. The check below is a set comprehension over every font matplotlib knows, which is not free either, and this function is called before every figure is styled.

### lines 90-94

```python
if FAMILY not in available:
```

ASK BEFORE ADDING. `addfont` is expensive -- it reads the file, builds a FontProperties and resolves alternative family names, and measured on the Mask screen the eight bundled faces cost 23 SECONDS of a 13-second module open, because this ran while the settings panel was being built. A machine that already has Open Sans installed needs none of it.

### line 100

```python
continue
```

One unreadable face must not cost the other seven.


---

# Notes from `spacr/figure_font.py`

Prose lifted out of `spacr/figure_font.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_default_font_params](#_default_font_params) (1 entry)

## Module level

### lines 107-128

```python
_RUN_MARKER = "SPACR_FIGURES_IN_OPEN_SANS"
```

Item 291: Open Sans as matplotlib's DEFAULT -- in the app and in pipeline runs, and nowhere else.

Decided 2026-09-15, verbatim: "Global in the app only (Recommended)". The option read: "The spaCR GUI and spaCR's pipeline runs set Open Sans as matplotlib's default; plain `import spacr` in a notebook leaves the user's matplotlib alone."

So nothing below runs on import (`use_open_sans_for_figures` above still changes no rcParam). The ENTRY POINTS call it: `spacr.qt.run`, which every GUI console script goes through; `spacr.cli.cmd_run`, which is `spacr-run` and also what every batch-queue job executes; `spacr-repro`; `spacr-tutorial`; and the parameter sweep's worker processes.

HELD FOR THE RUN, NOT WRITTEN BARE. In those processes the run IS the process, so an `rc_context` held for the whole run is the process default. But `spacr.cli.main` and `spacr.batch.inprocess_runner` are also called in-process -- by the test suite, and by a frozen build -- and a bare `rcParams.update` there would restyle every later figure of whoever called it. The context hands the caller its matplotlib back when the run ends.

## _default_font_params

### lines 164-165  _(unsure)_

```python
"font.sans-serif": [FAMILY, *others],
```

A caller that names the generic family itself, `fontfamily="sans-serif"`, gets Open Sans as well.
