Power / Design: a critical reading of a REAL simulator run
=======================================================

The JSON is the unchanged recorded output of the actual spaCR simulator.
Its numbers are simulated teaching results, not acquired biological data,
validated statistical power, a sample-size recommendation or evidence that
the simulated assumptions match your experiment. Do not use the misleading
headline as a design recommendation. No new simulation is run by this archive.

The tutorial's bounded run has two marginal scans, nine points and eighteen
fits, with only two replicates per point. The displayed detection criterion
is a usable fit reaching AUROC 0.8; this is not a hypothesis-test P threshold.
Four nonconverged fits remain in the detection denominator as non-detections.
Means of AUROC/AP include usable fits only. Average precision needs its
prevalence baseline for context. A curve at 100% here means only 2 of 2.

Two estimates refer to the SAME nominal baseline: 120 cells per well and
96 wells. Their 22 recorded input fields agree, but their recorded random
seeds differ. The cells scan finds 1/2 detections and the wells scan 2/2.
The headline's "Going to 96 wells would take that to 100%" does NOT describe
an increase in wells. The tutorial explicitly rejects that advice. The
application code and headline are NOT repaired or silently overwritten.

Two simulated screens clipped probabilities above 1, so realised occupancy
does not match the requested occupancy there. This and nonconvergence limit
interpretation. Read all native caveats; their presence does not validate the
simulator. The example is too small for a reliable precision claim, and the
model is not calibrated against new held-out experimental observations here.

recorded_simulation_result.json preserves the visible specification, actual
replicate outputs, displayed curves/table, status and contradictory headline.
It is an external evidence download, NOT a native GUI export demonstration
or a file that the GUI imports as a completed run. No results were injected
into the app. The captured implementation remains byte-identical in power.py,
power_design.py, power_model.py and power_simulate.py. It was recorded with
spaCR 1.5.0.4; reuse does not assert a new run on a newer version.

For developers: DesignSpec and simulator_kwargs in spacr.qt.widgets.power_design
map assumptions; spacr.qt.screens.power.run_power_sweep coordinates the scans,
and spacr.power_simulate and spacr.power_model
perform simulation and fitting. Inspect actual API signatures before use.
No new GPU task, provider request, app edit or publication accompanies this
read-only tutorial review. Review every assumption before any real planning.
