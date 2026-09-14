# Notes from `spacr/parameter_sweep.py`

Prose lifted out of `spacr/parameter_sweep.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (4 entries)
- [_default_filters.aggregation_belongs_to_wells](#_default_filtersaggregation_belongs_to_wells) (1 entry)
- [_default_filters.permutation_ignores_the_family](#_default_filterspermutation_ignores_the_family) (1 entry)
- [_default_filters.permutation_at_cell_level_exhausts_memory](#_default_filterspermutation_at_cell_level_exhausts_memory) (1 entry)
- [_default_filters.penalty_belongs_to_penalised_families](#_default_filterspenalty_belongs_to_penalised_families) (1 entry)
- [build_trials](#build_trials) (1 entry)
- [_named_control_rows](#_named_control_rows) (1 entry)
- [correction_rows](#correction_rows) (1 entry)
- [_recommended_worker_budget](#_recommended_worker_budget) (2 entries)
- [_pin_threads](#_pin_threads) (3 entries)
- [be_polite](#be_polite) (4 entries)
- [containment_available](#containment_available) (1 entry)
- [free_memory_gb](#free_memory_gb) (1 entry)
- [run_trial_contained](#run_trial_contained) (5 entries)
- [_trial_settings](#_trial_settings) (3 entries)
- [_execute_trial](#_execute_trial) (5 entries)
- [_register_resource_workers](#_register_resource_workers) (1 entry)
- [run_sweep_parallel](#run_sweep_parallel) (5 entries)
- [run_sweep](#run_sweep) (8 entries)
- [rank_trials](#rank_trials) (1 entry)
- [summarise_sweep](#summarise_sweep) (2 entries)
- [settings_for_trial](#settings_for_trial) (3 entries)
- [rerun_trial](#rerun_trial) (2 entries)

## Module level

### lines 48-50

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 91-94

```python
"alpha": ["auto", 1],
```

'auto' cross-validates the penalty. The literal 1 is spaCR's default and is far larger than the scale of a fraction design -- it shrinks every coefficient to exactly zero -- so sweeping only the default would report the three penalised families as uniformly useless.

### lines 101-103

```python
"random_row_column_effects": [False, True],
```

nuisance structure: row, column and plate

False keeps rowID + columnID as FIXED effects in the formula; True moves them to random effects, which replaces the backend with a mixed model.

### lines 105-106  _(unsure)_

```python
"batch_correction": ["none", "center", "zscore"],
```

The plate effect. 'none' leaves plates alone; the rest remove a per-plate shift before fitting.

## _default_filters.aggregation_belongs_to_wells

### lines 177-178

```python
return "analysis_unit='cell' ignores agg_type"
```

agg_type is forced to None for a per-cell fit, so sweeping it would run the same analysis several times under different labels.

## _default_filters.permutation_ignores_the_family

### lines 192-193

```python
if trial.get("inference") == "nonparametric" and \
```

The permutation test is its own estimator: it does not read regression_type, and sweeping it would repeat one analysis 13 times.

## _default_filters.permutation_at_cell_level_exhausts_memory

### lines 201-211

```python
if trial.get("inference") == "nonparametric" and \
```

THE COMBINATION THAT TOOK THE MACHINE DOWN.

The permutation test builds `x_unit.T @ permuted_outcomes` in batches. At WELL level that is 606 rows and costs nothing. At CELL level it is ~116,000 rows against 200,000 permutations, and a single trial was measured holding 57 GB of resident memory before the host ran out -- one fit, on its own, with nothing running in parallel.

Rejected rather than merely discouraged: there is no worker count, thread limit or nice level that makes it survivable, because the allocation happens inside one process regardless.

## _default_filters.penalty_belongs_to_penalised_families

### lines 228-230

```python
if trial.get("alpha") not in (None, 1) and \
```

alpha is refused outright by every family that cannot read it, so sweeping it against them would turn one axis into a wall of identical rejections.

## build_trials

### lines 301-315

```python
trial.update(space.fixed)
```

THE FIXED VALUES ARE PART OF THE TRIAL BEFORE IT IS JUDGED.

This used to run AFTER accept(), which meant every filter decided on a half-built trial. The GUI pins each UNTICKED axis into `fixed` (qt/screens/parameter_sweep.py), so the settings a user did not vary were exactly the ones the filters could not see -- and `permutation_at_cell_level_exhausts_memory` read `analysis_unit` as None and passed.

Reproduced with stock filters and nothing exotic: axes {"inference": ["parametric", "nonparametric"]} with fixed {"analysis_unit": "cell"} emitted the nonparametric x cell permutation, which is the ~57 GiB run. Unticking one checkbox was enough to schedule it, and the filter written to prevent exactly that was already in the default list.

## _named_control_rows

### lines 381-382  _(unsure)_

```python
row = ranked.loc[position].iloc[0]
```

ranked_labels is a permutation of labels, so a control present in labels is necessarily present here as well.

## correction_rows

### lines 446-449

```python
adjusted, reject = adjust_p_values(
```

(adjusted, reject). The reject mask is the METHOD'S OWN verdict: step-down procedures like holm and hommel do not reduce to "adjusted <= alpha" applied afterwards, so re-thresholding here would quietly report different hits than the method called.

## _recommended_worker_budget

### line 501, trailing  _(unsure)_

```python
except Exception:
```

psutil is a dependency, but be safe

### line 505, trailing  _(unsure)_

```python
except AttributeError:
```

non-Linux

## _pin_threads

### lines 651-662

```python
try:
```

THE ENVIRONMENT ALONE IS NOT ENOUGH, and this is the part that bit.

OpenBLAS reads OMP_NUM_THREADS once, when numpy first imports it, and sizes its pool from the core count if the variable is not set yet. Any module that imports numpy before this runs -- which is most of them leaves the pool at 32 threads no matter what is put in os.environ afterwards. Measured: env-then-numpy gives 1 thread, numpy-then-env gives 32.

threadpool_limits resizes the LIVE pool, so it works whatever the import order was. It is held open for the life of the process rather than used as a context manager, because the fitting happens after this returns.

### line 667, trailing  _(unsure)_

```python
except Exception:
```

threadpoolctl may be absent

### line 672, trailing  _(unsure)_

```python
except Exception:
```

torch may be absent

## be_polite

### lines 693-694  _(unsure)_

```python
with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as handle:
```

Linux only, and best effort: a container or a hardened kernel may refuse the write, which is not a reason to fail a sweep.

### line 697, trailing  _(unsure)_

```python
except OSError:
```

not Linux, or not permitted

### lines 701-702  _(unsure)_

```python
subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
```

Linux idle I/O class: a sweep reads gigabytes of CSV and should yield the disk to anything interactive.

### line 705, trailing  _(unsure)_

```python
except Exception:
```

best effort

## containment_available

### line 738, trailing  _(unsure)_

```python
except Exception:
```

no user manager

## free_memory_gb

### line 786, trailing  _(unsure)_

```python
except OSError:
```

not Linux

## run_trial_contained

### lines 829-832

```python
payload = {"settings": dict(settings), "trial_id": trial_id,
```

The control ALIASES travel with the trial. Without them a contained row loses the `{alias}_rank` columns -- and `positive_rank` is one of the columns the sweep screen puts in its table, so containing a trial would have quietly emptied the one column the run is judged on.

### lines 851-852

```python
print("WARNING: systemd-run is unavailable, so this trial runs "
```

Say so rather than pretending the cap is there: an uncapped sweep is a decision the user should get to make knowingly.

### line 875, trailing  _(unsure)_

```python
except Exception:
```

truncated by a kill

### lines 878-880

```python
import multiprocessing
```

A pool worker must carry the stamp one hop farther to the real parent. A direct/main-process caller can register it now and must not expose a private transport column in its public row.

### lines 885-887

```python
return {"status": "timeout" if tail == "timed out" else "killed",
```

No result file: the child was killed before it could write one. The cap is the likeliest reason and worth naming, because "killed" and "crashed" want different responses from the user.

## _trial_settings

### lines 909-932

```python
settings["regression_qc"] = bool(qc)
```

THE QC SUITE IS OFF FOR A SWEEP UNLESS IT IS ASKED FOR.

It costs ~5.8 s and writes ~19 figures plus a combined PDF per fit right for one analysis, and roughly ten minutes and two thousand files across a hundred trials, almost none of which anyone will open. The SCALAR diagnostics that make a row judgeable are a different thing: summarise_trial computes them in ~150 ms and they are unaffected by this, so a sweep still sorts by control rank, inflation and R^2.

ASSIGNED, NOT setdefault -- and this is the whole bug. `setdefault` does nothing when the key is already there, and EVERY base dict reaching here already carries it at True: the Tk panel, the Qt panel and `spacr-run` all build theirs from `get_perform_regression_default_settings`, which defaults it on. Measured 2026-08-18: `_trial_settings(get_perform_regression_default_settings({}), ...)` came back with regression_qc=True, so a sweep driven from the application paid the full suite on every trial -- roughly ten minutes and two thousand files per hundred trials -- while the comment above said it did not.

`qc` is the caller's explicit say. A sweep that wants the pictures asks for them; reopening one interesting trial goes through settings_for_trial, which does not pass here, so the trial you choose to look at again is still fitted WITH the diagnostics.

### lines 934-937

```python
try:
```

Write the settings the way the GUI writes them, so a trial worth a second look can be opened straight in the regression module -- point it at the trial folder and press run. A sweep whose interesting rows cannot be reopened is only half an answer.

### line 941, trailing

```python
except Exception:
```

never lose a trial over its record

## _execute_trial

### lines 953-958

```python
base_settings, trial, destination, controls, contained = payload[:5]
```

SIX OR FIVE. `qc` was appended on 2026-08-18 and this tuple is an internal, pickled, POSITIONAL contract -- growing it broke four tests that build a payload by hand, and would break any caller pickled by an older process mid-sweep. Reading it with a default costs one line and makes the addition backward compatible; the default is False, which is what a sweep wants.

### line 961

```python
be_polite()
```

Before anything expensive: this work yields to the user's machine.

### lines 965-969

```python
from .fit_resources import _worker_stamp
```

Returned through the existing future rather than a new IPC channel. The parent sampler may already have seen this PID; registering the creation-time stamp retroactively attaches the trial name to that process and to its eventual disappearance event without a PID-reuse race.

### lines 983-997

```python
child = run_trial_contained(settings, trial_id=trial["trial_id"],
```

THE CAP APPLIES TO THE SWEEP THE GUI ACTUALLY RUNS.

run_sweep has been contained by default since the kernel-cap work; this path had not been, and this is the one the sweep screen calls. So every guarantee that work bought -- MemoryMax, MemorySwapMax=0, CPUQuota, TasksMax -- was absent from the only sweep a user starts by clicking Start, and what remained was recommended_workers() and the free-memory floor. Those are ACCOUNTING, and this module's own history is that accounting is not containment: every previous fix was a better estimate of what a trial would use, and each one was wrong in a way that took the desktop with it.

The pool worker now only waits on the child, so it holds no design matrix of its own and the worker count stops being a memory multiplier as well.

### lines 1016-1023

```python
row.update(summarise_trial(output, settings))
```

THE SAME COLUMNS WHICHEVER WAY THE TRIAL RAN.

This is the path the GUI uses (run_sweep_parallel), and it was the only one that never called summarise_trial: a contained trial got fit quality, residual tests, design rank and control recovery, and a trial run from the sweep screen got a hit count. Same sweep, same question, different table depending on which entry point produced it.

## _register_resource_workers

### line 1060

```python
pass
```

Accounting must never change whether a trial is a result.

## run_sweep_parallel

### lines 1129-1136

```python
import multiprocessing
```

A CALLER WITHOUT A MAIN GUARD FORK-BOMBS ITSELF.

This pool spawns rather than forks (torch and OpenMP are not safe to fork), and a spawned child re-imports the module it was launched from. If that module is a script whose sweep call sits at top level, every child starts its own sweep, which starts more children. What the user sees is "BrokenProcessPool: A child process terminated abruptly", which says nothing about the actual mistake.

### lines 1148-1151

```python
_pin_threads()
```

Set here, in the parent, rather than in the worker: a spawned child inherits this environment at exec, whereas a pool initializer would run only AFTER the child has imported numpy and torch and they have already sized their thread pools from the core count.

### lines 1161-1162  _(unsure)_

```python
n_jobs, reason = recommended_workers(requested=n_jobs)
```

The worker count is a REQUEST, clamped to what the machine can afford. Honouring it literally is what killed the user's editor twice.

### lines 1177-1179

```python
context = multiprocessing.get_context("spawn")
```

'spawn', not the default fork: perform_regression imports torch, and a forked child that inherits a torch/OpenMP runtime deadlocks or segfaults rather than failing cleanly.

### lines 1208-1210

```python
future = next(as_completed(tuple(futures)))
```

``futures`` is non-empty here, so as_completed necessarily yields one item. Taking exactly one lets _fill recheck memory after every completion without an unreachable for-loop exit.

## run_sweep

### lines 1312-1315

```python
runner = None
```

Each trial in its own kernel-capped process. This is the default because the alternative -- trusting this module's own accounting took the user's machine down seven times, and every one of those was a fix to the accounting.

### lines 1333-1338

```python
exhausted: dict[tuple, dict] = {}
```

A family that cannot fit this response fails the same way every time 'poisson' needs integer counts, and a fractional score will never become one. Sampling would rediscover that hundreds of times at full cost, so after `learn_from_failures` identical failures the family is skipped and RECORDED as skipped. The finding is kept; only the repetition is dropped, and setting learn_from_failures=0 turns the shortcut off.

### lines 1355-1360

```python
settings, folder = _trial_settings(base_settings, trial, destination,
```

THE ONE HELPER, not a second copy of it. This branch had its own inline version of _trial_settings and had drifted from it: it never set `regression_qc` at all, so an in-process sweep paid the full ~5.8 s diagnostic suite and ~19 figures on every trial while the parallel branch was trying not to. Two copies of "build one trial's settings" is how a sweep ends up meaning two different things.

### line 1367  _(unsure)_

```python
if runner is None and contained and free_memory_gb() < memory_floor_gb:
```

Stop BEFORE the machine is in trouble, not once it is.

### line 1379  _(unsure)_

```python
child = run_trial_contained(settings, trial_id=trial["trial_id"],
```

Contained: the child returns a finished ROW, not a model.

### lines 1410-1412

```python
row.update(_design_summary(output))
```

The in-process path skipped the design summary as well as every diagnostic, so an uncontained sweep could not even say how many wells reached the fit.

### lines 1434-1436

```python
for extra in correction_rows(output, corrections,
```

One row per correction, all from this single fit. Each row still carries every setting, so it reproduces its own regression when the user opens it -- see settings_for_trial.

### lines 1455-1458

```python
try:
```

Written every trial, so a sweep killed halfway still leaves a usable table rather than nothing. The file is a best-effort checkpoint: a full or disconnected results disk must not discard the in-memory rows the sweep can still return to its caller.

## rank_trials

### line 1500

```python
ran = (frame["status"] == "ok") if "status" in frame.columns else True
```

NaN last, and a failed trial never outranks one that ran.

## summarise_sweep

### lines 1556-1558

```python
if "positive_control_rank" in ok.columns and len(ok):
```

THE ANSWER, WHEN THE SCREEN HAS A YARDSTICK. Stated before the hit counts, because a configuration that loses the positive control is not improved by reporting more hits.

### lines 1575-1577

```python
for axis in ("multiple_testing_method", "regression_type",
```

The spread across settings IS the result: a screen whose hit count ranges from 2 to 400 depending on the correction has not been analysed, it has been chosen.

## settings_for_trial

### lines 1623-1629

```python
aliases = [key[: -len("_present")] for key in row
```

The control-alias columns cannot be listed in advance, because the aliases are the CALLER'S: run_sweep(controls={"gra14": "239740"}) makes gra14_rank, gra14_q and the rest. They are recoverable exactly, though, because _named_control_rows always writes `{alias}_present` for every alias whether or not it found one -- so the row names its own aliases and nothing has to be guessed from suffixes. Guessing would be wrong anyway: spaCR has twenty-two real settings ending in `_percentile`.

### line 1645, trailing  _(unsure)_

```python
pass
```

genuinely a string

### lines 1650-1652

```python
settings["src"] = str(folder)
```

No mkdir here: this builds a settings dict and nothing else, so it stays callable from a test, a dry run or a preview without leaving directories behind. rerun_trial creates the folder it writes to.

## rerun_trial

### line 1674  _(unsure)_

```python
settings["verbose"] = True
```

Plots are the entire reason for this call.

### lines 1684-1687

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.
