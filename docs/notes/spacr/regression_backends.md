# Notes from `spacr/regression_backends.py`

Prose lifted out of `spacr/regression_backends.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_backend_name](#resolve_backend_name) (1 entry)
- [backend_status](#backend_status) (6 entries)
- [Module level](#module-level) (1 entry)
- [backend_install_offer](#backend_install_offer) (4 entries)

## resolve_backend_name

### lines 120-121

```python
aliases = {'lme4': 'pymer4', 'rapids': 'cuml', 'pytorch': 'torch',
```

'statsmodels (cpu)' with the suffix mangled, 'lme4', 'rapids' -- the spellings a person actually types.

## backend_status

### lines 241-244

```python
if regression_type is not None and not backend_supports(key,
```

THE TYPE FIRST, because it is about the choice the user just made rather than about the machine, and it is the one they can fix from the same panel. "cuML has no mixed model" is instruction 141 C's own example.

### lines 254-266

```python
listed = ', '.join(spec['types'])
```

SAY WHICH TYPES, AND WHETHER IT IS INSTALLED. Reported 2026-08-21: with the family left to be chosen from the response, every optional backend read "unavailable: needs an explicit regression type" seven identical lines saying what was MISSING and nothing about what any of them does or whether it is even on the machine.

"write the explisit regression type and what needs to be done if it is not installed. if it is intalled write installed."

Both facts belong here because they are answered differently: the types tell the user which choice would make this row selectable, and the install state tells them whether making that choice would be enough.

### lines 274-276

```python
if not spec['implemented']:
```

Keep the machine state explicit in the dropdown itself; a package command alone does not say whether it is required or merely an optional upgrade.

### line 289

```python
if installed:
```

LONG FORM: room to say both, and to say what to DO.

### lines 307-314

```python
missing = not package_installed(spec['package'])
```

NOT INSTALLED IS SAID ALONGSIDE NOT WIRED UP, not instead of it.

The unimplemented test used to return first, so on a machine with none of the six optional packages -- which is every machine, since they are extras -- the pip command instruction 141 C asks for was never shown by anything. Both facts are true of the same entry and both are what a reader needs: installing the package alone would not make it choosable, and neither would wiring it up alone.

### lines 317-321

```python
reason = (
```

"not wired up" in the long form as well as the short one. The long form used to carry the fact only as a paraphrase, and the install command was what a reader matched on -- so once a backend became a core dependency and had no install command to offer, the entry stopped naming its own state in the words the menu uses.

## Module level

### lines 468-477

```python
INSTALL_RECIPES = {
```

What would make an unavailable backend available, and can it be done HERE

Instruction 158. `backend_status` already says WHY an entry is greyed out; this half says what would ungrey it, and whether that is honestly possible in this environment. The three answers are install-here, possible-elsewhere and not-possible, and the shapes are `spacr.updater.InstallOffer` so the Image UMAP's GPU acceleration (`spacr.gpu_reduce.install_offer`) answers in the same vocabulary and one shared panel serves both.

## backend_install_offer

### lines 665-666

```python
if key == 'pymer4' and not installed:
```

1. NOT POSSIBLE BY INSTALLING -- said first when it is true of the package itself rather than of this machine.

### line 675

```python
if key == 'cuml' and not installed and not _cuml_python_supported():
```

2. POSSIBLE, BUT NOT HERE.

### line 686

```python
if not installed:
```

3. INSTALLABLE HERE.

### line 698  _(unsure)_

```python
if spec['device'] == 'gpu' and not cuda_present_without_importing_torch():
```

The package is here. Whatever is left is not an install problem.
