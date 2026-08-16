"""Run ONE regression in a fresh process, under whatever cap the parent set.

    python -m spacr.sweep_child <settings.json> <result.json>

This exists so a sweep trial can be contained by the kernel rather than by
spaCR's own accounting. Seven attempts to sweep this screen took the user's
desktop down, and each time the fix was a better ESTIMATE of what a trial
would use. Estimates are not containment; a cgroup is.

The thread environment is set at the very top, before anything imports numpy,
because OpenBLAS reads OMP_NUM_THREADS exactly once -- when numpy first
imports it -- and sizes its pool from the core count if the variable is not
there yet. Measured: env-then-numpy gives one thread, numpy-then-env gives
thirty-two, and thirty-two per process is what reached load 35 on 32 cores.
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
except OSError:  # pragma: no cover - not Linux, or not permitted
    pass

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m spacr.sweep_child <settings.json> <out.json>",
              file=sys.stderr)
        return 2
    settings_path, out_path = argv

    with open(settings_path) as handle:
        payload = json.load(handle)
    settings = payload.get("settings", payload)

    result = {"status": "failed", "trial_id": payload.get("trial_id")}
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
