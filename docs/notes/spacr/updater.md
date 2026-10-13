# Notes from `spacr/updater.py`

Prose lifted out of `spacr/updater.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_lt](#_lt) (1 entry)
- [find_uv](#find_uv) (1 entry)
- [editable_install_location](#editable_install_location) (1 entry)
- [run_pip_upgrade](#run_pip_upgrade) (1 entry)
- [Module level](#module-level) (1 entry)
- [installed_version](#installed_version) (1 entry)
- [DryRun.summary](#dryrunsummary) (1 entry)
- [_parse_pip_report](#_parse_pip_report) (1 entry)

## _lt

### line 141  _(unsure)_

```python
n = max(len(pa), len(pb))
```

Pad to same length

## find_uv

### lines 163-165

```python
for name in ("uv.exe", "uv") if os.name == "nt" else ("uv", "uv.exe"):
```

The Windows bootstrap writes uv.exe; POSIX installers write uv. Check both names rather than relying on PATHEXT, because this directory is deliberately private and is not added to PATH.

## editable_install_location

### line 231  _(unsure)_

```python
return here if os.path.isdir(os.path.join(here, ".git")) or \
```

Not under a site-packages: this is a checkout.

## run_pip_upgrade

### lines 245-257

```python
editable = editable_install_location()
```

NEVER UPGRADE OVER A DEVELOPMENT CHECKOUT. `pip install --upgrade spacr` uninstalls whatever is there and installs from the index -- including when what is there is an EDITABLE install pointing at a working tree. The developer's source stops being what runs, nothing says so, and every change they make afterwards has no effect they can see. Reported 2026-08-18: an update check ran mid-session and the console showed "Uninstalling spacr-1.5.0.4 ... Successfully installed spacr-1.5.0.4", which is that exact operation.

An editable install is a statement that this checkout IS the package, so the upgrade is refused rather than confirmed -- there is no version of "yes" that leaves the checkout in charge, and `git pull` is the upgrade for a checkout.

## Module level

### lines 298-310

```python
PROTECTED_PACKAGES = ("numpy", "torch", "pandas", "scikit-learn")
```

Offering to install something spaCR needs but does not have

Instruction 158. A greyed-out option has to be able to say what would ungrey it, and -- where that is honestly possible HERE -- to do it. The machinery lives in this module because this module already knows how to install into spaCR's own environment (:func:`find_uv`, :func:`upgrade_command`) and already handles the install where ``python -m pip`` was never seeded.

It is deliberately Qt-free. The GUI half is :mod:`spacr.qt.widgets.availability_panel`, and the split is what lets the three answers below be tested without a screen.

## installed_version

### line 339  _(unsure)_

```python
return None
```

A bundler that ships only what it saw imported can leave this out.

## DryRun.summary

### lines 511-513

```python
lines.append(f"  add {len(additions)} package(s): "
```

WITH THEIR VERSIONS. "adds cuml-cu12" and "adds cuml-cu12 26.2.0" are different amounts of evidence, and the second is what lets a reader check the wheel they are about to take.

## _parse_pip_report

### lines 592-597

```python
decoder = json.JSONDecoder()
```

`json.loads` CANNOT BE USED HERE. Measured 2026-08-18 against pip 25.3: `--report -` writes the document to stdout with pip's own progress BEFORE it and a "Would install ..." line AFTER it, so a whole-string parse fails on trailing data and the plan is lost -- the failure mode being "the packaging tool produced no readable plan" on a resolve that succeeded. `raw_decode` stops at the end of the first valid document.
