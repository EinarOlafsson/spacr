"""Run one regression trial in a fresh worker process.

    python -m spacr.sweep_child <settings.json> <result.json>

The parent may place this process in a kernel-enforced resource scope. Thread
limits are set before NumPy imports so BLAS libraries initialize with one
thread per trial. Results and failures are returned through a JSON file,
including when no in-memory return value is available.
"""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

# Volunteer for the OOM killer, before importing anything large. Same reason
# as spacr.parameter_sweep.be_polite: left alone the kernel scores by resident
# size and kills the biggest process on the box, which during a sweep is the
# user's editor and not this child. Repeated here because a contained trial is
# exec'd into a fresh interpreter and never runs be_polite.
try:
    with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as _handle:
        _handle.write("800")
except OSError:  # not Linux, or not permitted
    pass

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402


def main(argv=None) -> int:
    """Run one sweep trial in this process and write its result as JSON.

    Parameters
    ----------
    argv : sequence of str, optional
        Settings and output JSON paths. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when a result was produced, ``2`` for invalid arguments, or a
        nonzero status when the trial raised. Trial exceptions are also
        written to the output file.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m spacr.sweep_child <settings.json> <out.json>",
              file=sys.stderr)
        return 2
    settings_path, out_path = argv

    with open(settings_path) as handle:
        payload = json.load(handle)
    settings = payload.get("settings", payload)

    trial_id = payload.get("trial_id")
    from .fit_resources import _worker_stamp

    result = {
        "status": "failed",
        "trial_id": trial_id,
        # The parent removes this private transport field before writing the
        # sweep table. PID plus creation time lets the run sampler attach the
        # trial name to samples it took while this short-lived child existed.
        "_resource_worker": _worker_stamp(
            "parameter_sweep_trial", trial_id),
    }
    began = time.time()
    try:
        import matplotlib
        matplotlib.use("Agg")

        from .parameter_sweep import _pin_threads
        # Belt as well as braces: the environment above is read at import,
        # this resizes the pool that already exists.
        _pin_threads()

        from .ml import perform_regression

        output = perform_regression(dict(settings))
        result["status"] = "ok"

        from .trial_metrics import summarise_trial
        result.update(summarise_trial(output, settings))

        # The caller's own control ALIASES, on top of the canonical
        # positive_control_* columns. The sweep screen puts `positive_rank` in
        # its table, and that column is built from this mapping -- so a
        # contained trial that did not compute it would leave the one column
        # the run is judged on blank, which looks exactly like a control that
        # was never recovered.
        controls = payload.get("controls") or {}
        if controls:
            try:
                import pandas as pd

                from .parameter_sweep import _named_control_rows

                results = output.get("results") \
                    if hasattr(output, "get") else None
                if isinstance(results, pd.DataFrame):
                    result.update(_named_control_rows(results, controls))
            except Exception:  # noqa: BLE001 - an alias must not sink a trial
                pass
    except BaseException as error:  # noqa: BLE001 - a failed trial is a result
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = (str(error).splitlines() or [""])[0][:400]
        folder = settings.get("src")
        if folder:
            try:
                with open(os.path.join(folder, "error.txt"), "w",
                          encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
            except OSError:
                pass
    result["seconds"] = round(time.time() - began, 2)

    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
