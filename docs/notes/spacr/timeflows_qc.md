# Notes for `spacr/timeflows_qc.py`

The module carries no comments; the reasons behind it live here and in its
docstrings. Item 426
(`features/future/426_timeflows_a_temporal_cellpose_for_timelapse.txt`) is
where the history is, and step 1 of its plan is what this module is.

## Why a table with verdicts rather than a plot or a summary number

The plan asks for "a QC script that produces a table, not a one-off look", and
the reason is in the defect it is hunting. A recycled label -- the same number
given to an unrelated object after the first one left -- is a correct frame
followed by a correct frame. Nothing about either frame is wrong; only the
pair is. So there is no image to look at that shows it, and no single number
that survives averaging: one recycled id in a thousand objects disappears into
any mean and still teaches the model to point at the wrong cell.

Every verdict therefore has a detail table behind it and every detail table is
public (`recycled_labels`, `division_events`, `displacement_table`). A flagged
row is a row somebody reads.

## How a gap is told from a recycled id, and why the threshold is measured

Both look identical in one frame: a label that is not there. What separates
them is what happens on the far side of the gap. Two measurements are taken
for every gap:

* the overlap (IoU) between the mask the label had before the gap and the mask
  it has after it -- a real object missed for a frame is usually still
  overlapping itself;
* the distance the centroid moved, divided by the number of frames that
  passed.

A gap is called `reuse-suspected` only when BOTH say so: zero overlap AND a
per-frame distance above what continuous objects in the same data cover.

The threshold is measured from the data rather than chosen: `3 x` the 95th
percentile of the per-frame displacement of every object that survives a
frame (`REUSE_DISPLACEMENT_FACTOR`). Three is deliberately generous. The flag
is an accusation against somebody's annotation, and a false one costs a day of
reading; a missed one is caught later by the displacement rows, which are
reported whether or not anything is flagged. When nothing in the stack
survives a frame there is no displacement distribution, and the fallback is
the median object diameter -- an object cannot move much less than its own
size and still be called the same object.

`--max-displacement` overrides the measurement for data whose own
distribution is untrustworthy, which is the case if the reuse is systematic
enough to inflate the percentile it is being compared against.

## Why divisions are read by overlap and not by counting labels

A division is not "one label became two". Annotation that renumbers on every
frame would look like that everywhere. The test is that at least half of each
child's own area came from the same parent object in the previous frame
(`CHILD_OVERLAP = 0.5`). Half is the Cell Tracking Challenge's detection
criterion and has the property that makes it usable: two parents cannot both
hold more than half of the same child, so the parent is unique and no tie
needs breaking.

The convention is then read off the ids alone: if one child carries the
parent's number the annotation inherits, otherwise both children are new. The
plan needs this because the loss has to know which one it is being trained on;
a mixture is a `fail` rather than a `check`, because no single rule is right
for it.

## The displacement bands, and what they decide

`DISPLACEMENT_EASY = 0.5` and `DISPLACEMENT_HARD = 1.0`, in multiples of the
object's own area-equivalent diameter, at the 95th percentile.

This ratio, not the pixel count, is what the plan's stated failure mode is
about: "If per-frame displacement is routinely larger than an object, the
vector has to point further than the receptive field can see, and the field
becomes ambiguous where several identical cells are candidates." Below half a
diameter the target is inside the neighbourhood the head already sees for the
spatial flows. Above one diameter the head is being asked to point past what
it can see, and step 1 is supposed to say so before any model is written --
hence `fail`, in a QC script, for a property of the data rather than a bug.

## Why the frame interval is a check at all

A displacement field encodes pixels per frame and nothing else. A dataset
imaged every 30 s and one imaged every 10 min produce different fields for the
same biology, and training on the mixture teaches the average of two
incompatible problems. So the interval is reported per dataset AND compared
across datasets (`audit_datasets`), which is the only row that can see the
second failure.

A stated interval is reported as `stated` and verdict `check`, never `ok`: it
is the caller's claim, not a measurement. Nothing here overrides a measurement
with a claim.

## Why an exit code

`main` returns 1 when any verdict is `fail`. The failure is a statement about
the ANNOTATION, not about the program, which is why it is documented on the
function: a run that exits 1 has found what it was sent to look for.

## What it shares with `spacr/timeflows_baseline.py`

`overlap_counts` lives in the baseline module and is imported here. Both
halves of item 426 need the same cross-tabulation of shared pixels between two
frames, and one implementation with one set of tests is the only way the two
agree by construction. The direction of the import is arbitrary and was chosen
so the scoring module, which a training loop imports every epoch, depends on
nothing.
