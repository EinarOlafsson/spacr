# Notes from `spacr/qt/widgets/formula.py`

Prose lifted out of `spacr/qt/widgets/formula.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_Parser._power](#_parser_power) (1 entry)
- [_Parser._call](#_parser_call) (1 entry)
- [_numeric_column](#_numeric_column) (1 entry)
- [evaluate.walk](#evaluatewalk) (1 entry)
- [_binary](#_binary) (1 entry)
- [ColumnFormula.__post_init__](#columnformula__post_init__) (1 entry)
- [ColumnFormula.ast](#columnformulaast) (1 entry)
- [_apply_one](#_apply_one) (1 entry)

## _Parser._power

### lines 723-725

```python
return self._count(Binary("**", node, self._nest(self._unary)))
```

Right-associative, and the exponent goes through `_unary` so

`2 ** -1` parses. `-a ** 2` is `-(a ** 2)`, as in Python and as in every maths textbook.

## _Parser._call

### line 784, trailing  _(unsure)_

```python
self._advance()
```

the '('

## _numeric_column

### lines 886-887

```python
return coerced.to_numpy(dtype=float)
```

Numbers stored as text — a CSV column read as object. Usable, and the values that are not numbers become NaN rather than an error.

## evaluate.walk

### lines 938-940

```python
raise FormulaError(
```

A NODE THE PARSER DOES NOT PRODUCE TODAY. Named rather than merely refused: a formula error the user sees has to say what it could not do.

## _binary

### lines 986-987

```python
return a ** b
```

`**`. Floats throughout, so a huge exponent is `inf` rather than a bignum allocation that never returns — see the module docstring.

## ColumnFormula.__post_init__

### lines 1084-1085

```python
object.__setattr__(self, "_ast", parse(self.expression))
```

Parse now, so an unparseable formula cannot be stored, serialised, or reach a redraw.

## ColumnFormula.ast

### line 1096, trailing  _(unsure)_

```python
return self._ast
```

set in __post_init__; not a field

## _apply_one

### lines 1245-1248

```python
raise FormulaError(
```

`area = area * 2` with ``replace`` on is a rescale, and legitimate: `compute` always starts from a fresh copy of the loaded table, so it reads the measured column and is idempotent however often it is re-applied. `x = x + 1` where no `x` exists is a genuine circle.
