# Notes from `spacr/guide_attribution.py`

Prose lifted out of `spacr/guide_attribution.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_beta_density](#_beta_density) (2 entries)
- [posterior](#posterior) (4 entries)
- [attribute_well](#attribute_well) (1 entry)
- [attributable](#attributable) (2 entries)
- [Assignment](#assignment) (1 entry)
- [assign_well](#assign_well) (6 entries)
- [effective_dimension](#effective_dimension) (2 entries)
- [posterior_multivariate](#posterior_multivariate) (3 entries)

## _beta_density

### lines 120-125

```python
mean = 0.0 if shifted < -700.0 else 1.0 / (1.0 + math.exp(-shifted))
```

`1 / (1 + exp(-shifted))` RAISES OverflowError once shifted drops below about -709, so a guide with a large negative effect killed the whole well instead of being attributed. Saturating there is not an approximation: the clamp on the next line already pins the mean at `eps` for every shift below about -14, so the value returned is identical to what the unclamped expression produced.

### line 129  _(unsure)_

```python
variance = min(max(spread * spread, 1e-9), mean * (1.0 - mean) * 0.999)
```

Concentration from the spread: var = mean(1-mean)/(1+nu) inverted.

## posterior

### lines 188-189

```python
dead = density.sum(axis=1) <= 0
```

A cell no guide can explain is given the prior rather than dropped: it is a cell, it carries something, and the honest answer is "no idea".

### line 195, trailing  _(unsure)_

```python
r = density * weights
```

start from plain Bayes

### line 199, trailing  _(unsure)_

```python
r = r / rows
```

every cell carries one guide

### line 206, trailing  _(unsure)_

```python
r = r * factor
```

pin each guide to its reads

## attribute_well

### lines 282-283

```python
tied = np.flatnonzero(row == best)
```

EXACT ties only. `np.isclose` here would make an arbitrary choice on values that are merely similar, which is a different claim.

## attributable

### lines 331-337

```python
rest = [pair for pair in
```

EACH WEIGHT IS READ ONCE. Converting in the filter and again in the stored tuple let a weight whose conversion is not pure -- a lazily fetched count, a mutable proxy, a value re-read from a stream -- pass the positivity test and then be stored non-positive. The weight the filter approved would not be the weight used, and the well could be reported "this guide can never be called" on data that looked positive when it was checked. Reading once makes the two agree by construction.

### lines 346-347  _(unsure)_

```python
effects = [mine] + [e for e, _ in rest]
```

The range of scores a cell could plausibly take: `span` sigmas either side of the centre, widened to cover every component's own centre.

## Assignment

### lines 453-481

```python
@dataclass(frozen=True)
```

The constrained assignment -- the Sudoku one

"my mind always goes to suduko where you have rules and conditions that must be met and you use the little information you have within the confines of the rules to do your inference."

That is the better framing, and the soft posterior above throws away its central mechanism. Sudoku's power is not probability, it is EXCLUSION: if this cell takes that value, no other cell in the region can. `posterior` gives each cell an independent marginal and then notices that none of them is confident. The constraints here are exactly Sudoku's shape:

every cell carries EXACTLY ONE guide guide g occupies EXACTLY round(N * pi_g) cells of the well a guide absent from the well occupies NONE of it

Solved as a minimum-cost assignment: expand each guide into its own integer number of slots and match cells to slots so the total -log likelihood is smallest. Every count is then exactly right BY CONSTRUCTION, and every cell has a definite guide -- which the marginal posterior can never deliver when the priors are small.

WHAT IT DOES NOT DO, and this is the honest half. An assignment being OPTIMAL does not make it CERTAIN. When the evidence is weak, many assignments are nearly as good, and swapping two cells costs almost nothing. `Assignment.degeneracy` reports exactly that, so a reader can tell a solved grid from one that merely satisfies the rules.

## assign_well

### line 543

```python
exact = np.array([priors[g] * n for g in names], dtype=float)
```

HOW MANY SLOTS EACH GUIDE GETS, summing to n exactly.

### lines 547-553

```python
if short > 0:
```

ONLY THE SHORT-BY-SOME CASE. There was an `elif short < 0` arm undoing an overshoot, marked "rare"; it is not rare, it is impossible. `priors` sums to 1, so `exact` sums to n, and `floor(x) <= x` gives `slots.sum() <= n` -- `short` cannot be negative. Argued, then checked over 30,000 random fraction sets with magnitudes spanning twelve orders and well sizes to 400: the most negative value seen was 0.

### lines 565-571

```python
columns = np.repeat(np.arange(len(names)), slots)
```

One column per SLOT, so the counts are a property of the matrix rather than something checked afterwards. `slots` sums to exactly n after the correction above, so this has exactly n entries. The truncation that used to follow could not fire for the same reason the removed arm could not: the only way to get more than n columns is `slots.sum() > n`, which requires the negative `short` that cannot happen.

### lines 578-583

```python
order = np.empty(n, dtype=int)
```

HOW ARBITRARY IS IT: what this cell would cost under its best ALTERNATIVE GUIDE, not its second-cheapest slot. Slots of the same guide have identical cost, so the second-cheapest slot is almost always another slot of the guide already chosen and the difference is exactly zero which made this read "arbitrary" for a perfectly decided grid. Caught by the test that asks a decided assignment to score above an undecided one.

### line 595, trailing

```python
del rng
```

ties are broken by the

### lines 596-597  _(unsure)_

```python
assigned = tuple(str(names[c]) for c in chosen)
```

solver deterministically; the seed is accepted so the signature matches `attribute_well` and a caller can pass one without thinking about it.

## effective_dimension

### lines 604-630

```python
def effective_dimension(matrix: np.ndarray) -> float:
```

Option C -- every measurement, not just the score

"best case i can use all the fraction information and all the measurement and classefication data to estimate which grna is linked to which cell ... eaven if it only holds a timy little bit of information it still might work, right?"

Right in principle, and the arithmetic below is what makes the "might" honest. Two things have to be got correct or this produces confident nonsense.

1. LOG SPACE. A product of 785 densities underflows to exactly zero in double precision long before it reaches the end, and every cell then looks equally impossible -- which the code above answers by handing back the prior. The bug would present as "option C always says ambiguous".

2. THE MEASUREMENTS ARE NOT INDEPENDENT, and pretending otherwise is the difference between a method and a fiction. `cell_area` and `cell_perimeter` are one measurement wearing two names; multiplying their likelihoods counts the same evidence twice. Measured on the maintainer's own screen, 785 measurement columns carry an effective dimension in the low tens. So the summed log-likelihood is SCALED by n_eff / n_measured, which is the standard design-effect correction. Without it the posterior saturates at 0 or 1 for every cell and the 0.55 threshold becomes decorative.

### lines 651-653

```python
alive = spread > 0
```

A column that does not vary carries no information and must not be allowed to divide by zero; it is dropped rather than kept at scale 1, which would have made it look like an independent measurement.

## posterior_multivariate

### lines 728-731

```python
log_density = np.zeros((n_cells, len(guides)), dtype=float)
```

LOG DENSITY, SUMMED. `_density` is per-measurement, so this is a loop over columns rather than one vectorised call -- 785 columns is nothing beside the per-cell work, and reusing the same densities is what keeps option C's answer commensurable with option A's.

### lines 734-736

```python
present = finite[:, column]
```

A measurement missing for a cell contributes NOTHING for that cell rather than a zero score, which would be a real and usually extreme value.

### lines 752-754

```python
log_density -= log_density.max(axis=1, keepdims=True)
```

Subtract the per-cell maximum before exponentiating: the shift cancels in the normalisation and is the difference between a usable number and exp(-4000).
