# Notes for `spacr/timeflows_baseline.py`

The module carries no comments; the reasons behind it live here and in its
docstrings. Item 426
(`features/future/426_timeflows_a_temporal_cellpose_for_timelapse.txt`) is
where the history is, and step 2 of its plan is what this module is: "the
plain stitcher ... and the tracking metrics: TRA/DET from the Cell Tracking
Challenge, plus a count of identity switches. This number is what the fork has
to beat."

## What the metrics are, exactly

DET and TRA are the Cell Tracking Challenge measures of Matula et al. (PLoS
ONE 10(12): e0144959, 2015), built on AOGM -- the weighted number of edit
operations that turn the computed acyclic oriented graph into the ground truth
one, normalised by the cost of building the ground truth from nothing:

    DET = 1 - min(AOGM_D, AOGM_D0) / AOGM_D0
    TRA = 1 - min(AOGM,   AOGM_0)  / AOGM_0

Weights, from the same paper and reproduced in `AOGM_WEIGHTS`: split a merged
detection 5, false negative 10, false positive 1, delete an edge 1, add an
edge 1.5, change an edge's semantics 1. They are not knobs. A score reported
with other weights is not comparable with a published one, which is why
`score_tracking` carries the counts beside the scores and why the weights
argument is documented as changing what the number MEANS.

Vertices are matched by the challenge's detection criterion: a computed
segment detects a ground-truth object when it covers MORE THAN HALF of that
object's pixels (`DETECTION_OVERLAP`). More than half is what makes the
matching a function -- two segments cannot both hold more than half of the
same object -- so nothing here needs a tie broken, and exactly 50% is not a
detection.

## Where this differs from the official evaluator, honestly

Three places, all of them deliberate and all of them conservative:

1. **Edges are mapped only through vertices that detect exactly one
   ground-truth object.** A merged segment (one prediction over two objects)
   is already paid for as a split; its edges are deleted rather than being
   assigned to one of the two objects it covers. The official implementation
   makes the same restriction, and the effect either way is that a merge
   cannot earn credit for a link.
2. **A label whose frames are not contiguous is scored rather than refused,
   and the gap earns no edge.** The challenge's format has no way to say "the
   same object, not visible for a while": tracks are contiguous and a bridged
   gap is a separate track with a parent link. The official evaluator rejects
   such a file. Refusing it here would make the module unusable for exactly
   the trackers step 2 exists to measure, so the tracking is scored on the
   links it can show -- and the identity it claims across the gap still counts
   in `identity_switches`, where that claim belongs.
3. **A parent-child edge is only built when the child's first frame directly
   follows the parent's last.** A lineage entry that does not line up is
   ignored rather than turned into an edge that skips frames.

The consequence of 2 and 3 is that a tracking with gaps scores no better here
than the official evaluator would allow, and possibly slightly worse. That is
the right direction for a baseline the fork has to beat: it must not be
flattered.

## Why identity switches are counted separately when TRA already pays for them

TRA is one number dominated by its vertex terms -- a false negative costs ten
and a redundant edge costs one -- so a tracker can lose a great many links and
still score well if its segmentation is good. An identity switch is the
failure a biologist actually sees ("the cell became another cell") and the one
that corrupts every downstream per-track measurement. The count is reported
next to the score so the two cannot be traded silently.

The definition: each ground-truth track is walked in frame order over the
frames where it was detected, and every change of computed label is one
switch. Frames where the object was missed are SKIPPED rather than ending the
track, so an object lost and found again under a new id counts. A tracker
should not score better for dropping the frame in which it lost the thread.

## Why it does not call `spacr.timelapse.link_by_iou`

`spacr/timelapse.py` has the same linking and it is what the pipeline uses.
Two reasons it is not imported:

* importing `spacr.timelapse` pulls matplotlib, OpenCV, the figure style and
  the tracking backends. This module is meant to be importable inside a
  training loop, and step 6 of the plan reports its score every epoch.
* its overlap arithmetic is a Python double loop over every pair of boolean
  masks -- fine for a run's own frames, hopeless for scoring a held-out set
  once per epoch. `overlap_counts` here is one cross-tabulation per frame
  pair, over the pixels where both frames are labelled.

The two are meant to agree on WHICH labels match at a given threshold, and
`tests/test_timeflows_baseline_metrics.py::test_link_frames_agrees_with_the_pipeline_linker`
holds that down. If that test ever goes red, one of the two is wrong and it is
not safe to assume which.

## Why the stitcher has no divisions

`stitch_by_iou` is one-to-one Hungarian assignment: when a cell divides, one
child inherits the track and the other starts a new one with no parent
recorded. That is not an oversight to be patched -- it is the limitation the
third head is supposed to remove, and TRA charges it for the parent edges it
did not produce. Patching it here would hide the gap the fork has to close.

Hungarian rather than greedy matters for the same reason: greedy takes the
best single pair and can leave a second object with nothing above the
threshold, losing a link that the assignment maximising the total keeps. The
monkeypatched test in the suite is that case, because every simple fixture
passes either way.
