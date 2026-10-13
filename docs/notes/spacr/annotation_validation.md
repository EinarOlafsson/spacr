# Notes from `spacr/annotation_validation.py`

Prose lifted out of `spacr/annotation_validation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Verdict](#verdict) (4 entries)
- [synthesise](#synthesise) (4 entries)
- [permuted](#permuted) (1 entry)
- [benchmark](#benchmark) (3 entries)
- [default_scenarios](#default_scenarios) (7 entries)
- [mixture_proportion](#mixture_proportion) (2 entries)
- [mixed_ratio_calibration](#mixed_ratio_calibration) (1 entry)

## Verdict

### line 83, trailing  _(unsure)_

```python
coverage: float
```

share of cells annotated at all

### line 84, trailing  _(unsure)_

```python
precision: float
```

share of ANNOTATED cells that are right

### line 85, trailing  _(unsure)_

```python
recall: float
```

share of ALL cells annotated correctly

### line 86, trailing  _(unsure)_

```python
per_guide: Dict[str, Tuple[float, float]]
```

guide -> (precision, recall)

## synthesise

### lines 96-98  _(unsure)_

```python
def synthesise(*,
```

1. a screen whose truth is known

### lines 149-155

```python
block: List[Tuple[np.ndarray, float, str]] = []
```

SHUFFLED WITHIN THE WELL, and this is not tidiness -- it is the difference between a benchmark and a lie. Emitting each guide's cells as a contiguous block leaves the ROW ORDER carrying the answer, and any method that hands out contiguous runs then scores far above what the data can support. It was caught by the `no_effect` scenario reporting 85% precision on features that contain no information at all, which is the scenario's whole job.

### line 162  _(unsure)_

```python
value = float(np.linalg.norm(centre)) + rng.normal(scale=0.5)
```

The score tracks the phenotype, then the classifier errs.

### line 174  _(unsure)_

```python
seen = {names[g]: float(p) for g, p in zip(here, weights)}
```

What sequencing REPORTS: thresholded, biased, renormalised.

## permuted

### lines 282-284  _(unsure)_

```python
def permuted(screen: Screen, *, seed: int = 0) -> Screen:
```

3. the null -- the check that runs on REAL data

## benchmark

### lines 347-349  _(unsure)_

```python
def benchmark(strategies: Mapping[str, Callable[[Screen], Sequence[str]]],
```

4. every strategy, on the same screens

### lines 366-368

```python
everything = {**BASELINES, **dict(strategies)}
```

THE BASELINES ARE NOT OPTIONAL. They are what separates "the method works" from "the fractions work", and a caller who forgot them would read the second as the first.

### line 391  _(unsure)_

```python
"gain": float(real.precision - chance.precision),
```

What the data was worth, which is the number to read.

## default_scenarios

### line 463

```python
"clean": synthesise(seed=seed),
```

Nothing wrong: the ceiling. A method that fails here is broken.

### lines 465-466  _(unsure)_

```python
"no_effect": synthesise(effect=0.0, seed=seed + 1),
```

No signal at all: the floor. Everything must be at chance, and a method that beats chance here is reading its own anchors.

### line 468

```python
"penetrance_0.5": synthesise(penetrance=0.5, seed=seed + 2),
```

Half the cells do not show the phenotype.

### line 470

```python
"inflated_fractions": synthesise(fraction_threshold=0.10,
```

The 207 mechanism: threshold, then renormalise, then inflate.

### line 473  _(unsure)_

```python
"classifier_0.94": synthesise(classifier_accuracy=0.94, seed=seed + 4),
```

The maintainer's stated classifier.

### line 475  _(unsure)_

```python
"crowded": synthesise(guides_per_well=8, guides=16, seed=seed + 5),
```

Crowded wells: more guides sharing, so more chances to confuse.

### line 477  _(unsure)_

```python
"realistic": synthesise(penetrance=0.6, fraction_threshold=0.10,
```

Everything at once, which is the real screen.

## mixture_proportion

### lines 484-515

```python
def mixture_proportion(features: np.ndarray,
```

5. mixed-ratio control wells -- ground truth on REAL data

The maintainer's proposal, 2026-08-21: "can the hold out be the mixed ratio wells, where we dont know the identity of each cell but we do know how many cells are PC and how many are NC from the sequencing. these were not use for training."

IT IS BETTER THAN THE SIMULATION AND BETTER THAN INSTRUCTION 214's SINGLE POSITIVE CONTROL, for a reason worth stating precisely.

214 records that a single positive control cannot separate PENETRANCE from FRACTION BIAS: the slope of imaging-fraction on sequencing-fraction is their product. A RATIO SERIES separates them, because the two enter at different places. A well that is a proportion `pi` of PC cells has a feature distribution that is exactly the mixture

F_w  =  pi * F_PC  +  (1 - pi) * F_NC

and `F_PC` is estimated from the extreme wells INCLUDING its non-penetrant cells -- a PC cell showing no phenotype is still a PC cell and is still in `F_PC`. So the mixture fit recovers the true CELLULAR proportion with penetrance already absorbed, and comparing that to what sequencing reported isolates the fraction bias on its own.

WHAT IT DOES NOT SHOW, said here because it is the easy thing to forget: PC-versus-NC is a two-class problem with the largest phenotype difference in the screen. A method can be perfect on it and still fail at six guides in one well, which is the actual task. This validates the calibration and the discrimination; the simulation above remains the only check of the multi-guide assignment.

### lines 545-546

```python
return float("nan")
```

The two controls are indistinguishable: there is no line to project onto, and any number would be invented.

## mixed_ratio_calibration

### lines 578-584

```python
circular = pure_pc_wells is None or pure_nc_wells is None
```

WHICH WELLS ARE PURE IS A FACT ABOUT THE PLATE, not about the numbers under test. Picking them by the REPORTED fraction is circular -- that fraction is precisely the biased quantity this is measuring, so a bias large enough to matter moves a pure well below the cut-off and the fit refuses to run on exactly the screens that need it. Caught that way: a 0.55 bias made every 100%-PC well report 0.55 and no pure well was found.
