Timeflows training data and supervision
========================================

Timeflows training uses consecutive image pairs with object identities shared
between frames. Inspect those identities before interpreting a predicted
displacement or disappearance. A correction to the training reader changes
future training inputs; it does not retrain an existing checkpoint.

Build pairs from Cell Tracking Challenge annotations
-----------------------------------------------------

:func:`spacr.timeflows_model.ctc_pairs` reads raw images, full segmentation
masks and tracking markers. It forms pairs only from consecutive frame numbers
for which all three inputs exist. Slice-mask filenames are excluded. Duplicate
frame numbers, invalid sequence names or limits, and invalid annotation arrays
raise errors instead of silently selecting a file.

:func:`spacr.timeflows_model.track_masks_from_ctc` assigns a track identity
only when a segmented object contains exactly one marker identity and that
marker overlaps no other segmented object. Unmarked objects, merged objects
with multiple markers, and split objects sharing a marker are excluded. The
output uses ``int64`` labels, with zero outside retained objects. Inputs must
be matching two-dimensional non-negative integer arrays, and marker identities
must fit in ``int64``.

The private ``_ctc_track_masks`` helper supplies the same assignment policy to
the training reader and evaluator. Its exclusion counts include marker
identities without retained full masks. Categories can overlap, so their
counts must not be added as if they described disjoint objects.

For each pair, a source object is excluded from supervision when its next-frame
marker exists but has no valid full mask. Missing or ambiguous annotation must
not become a disappearance label. This exclusion uses a copy of the source
labels and preserves cached labels used by other pairs.

The nested :func:`spacr.timeflows_model.ctc_pairs.indexed` helper has its own
API entry for the frame-number mapping. The private top-level assignment
helper remains an implementation detail of the public reader.

Understand crop boundaries
---------------------------

:func:`spacr.timeflows_model.train_timeflows` samples a shared image window
before constructing temporal targets. A partially cropped source object can
have the wrong diameter; a partially cropped successor can have the wrong
centroid. A successor outside the window can otherwise look like a
disappearance even though it is present in the full frame.

The private ``_training_window`` helper therefore removes supervision for
partial source masks, partial successor masks and successors outside the tile.
An absence in the supplied full-frame labels remains supervised. This policy
checks consistency with the supplied labels; it does not establish that those
labels are biologically correct.

If a crop contains no usable source supervision, training samples another
window. After 32 unsuccessful attempts for one optimizer step, it raises
``ValueError``. Inspect full masks, motion and tile size before retrying.
Removing affected examples avoids incorrect crop-boundary targets but also
removes some fast-motion examples. It does not establish performance on full
movies or large displacements.

The stage loop is documented separately as
:func:`spacr.timeflows_model.train_timeflows.run`. Shared image augmentations
are applied through :func:`spacr.timeflows_model.augment_pair.apply`.

Interpret links and unmatched objects
--------------------------------------

:func:`spacr.timeflows_model.link_by_timeflows` first removes candidate links
outside the configured distance limit, then solves the assignment with
explicit unmatched choices. A rejected edge cannot occupy a target and
displace a valid link. The objective minimizes distance plus unmatched costs;
it does not maximize the number of links. Distances use source-object
diameters, with a one-pixel minimum diameter. A zero limit allows only exact
predicted-centre matches.

Both thresholds must be finite. ``min_successor`` must lie in ``[0, 1]`` and
``max_distance`` must be non-negative. When both frames contain objects, the
prediction arrays must match the source frame. Foreground vectors and derived
centres must be finite, and foreground successor probabilities must lie in
``[0, 1]``. Invalid values raise ``ValueError``; background predictions are
ignored. An empty object set in either frame returns no links without reading
the prediction arrays.

Keep the evaluator's ``temporal_assignment`` policy and thresholds with its
results. Scores obtained with the previous decoder describe that decoder;
they do not establish the accuracy of the corrected assignment. Compare both
decoders on identical predictions before attributing a difference to matching.

Keep checkpoint provenance with the results
--------------------------------------------

The training command records ``window_supervision`` metadata, including the
policy, tile size, maximum attempts per step and treatment of full-frame
absences. Its ``annotation_assignment`` metadata records the strict assignment
policy and treatment of unknown successors. Retain this metadata with the
checkpoint and the source annotation identities when comparing training runs.

Existing checkpoints are unchanged by these reader and supervision fixes.
Assignment audits and consistency checks are separate from retraining,
independent annotation review and accuracy evaluation. Evaluate a newly
trained checkpoint on independent complete sequences before making a claim
about improved tracking.
