"""Table-driven test of the contained-trial worker ``python -m spacr.sweep_child in.json out.json``.

A sweep runs each trial in a fresh interpreter and reads the trial's answer
back from a JSON file, so every outcome -- a successful fit, a failed or
interrupted one, the control aliases the sweep table is judged on -- has to
end in that file.  The regression is replaced per case; the metric summary
(:mod:`spacr.trial_metrics`) and the control-alias lookup run for real on a
real DataFrame.  Thread pinning is recorded instead of performed, because it
would resize the pytest process's own BLAS pools.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

import spacr.ml as ml
import spacr.parameter_sweep as parameter_sweep
import spacr.sweep_child as sweep_child
import spacr.trial_metrics as trial_metrics

USAGE = "usage: python -m spacr.sweep_child <settings.json> <out.json>\n"


def _results_frame():
    """A minimal regression result table with a recoverable positive control."""
    return pd.DataFrame({
        "grna": ["TGGT1_POS_1", "TGGT1_POS_2", "guide_a", "guide_b"],
        "coefficient": [2.5, 2.1, 0.2, -0.1],
        "p_value": [0.0001, 0.0004, 0.4, 0.9],
        "q_value": [0.001, 0.002, 0.6, 0.95],
        "n": [40, 38, 41, 39],
    })


def _raises(error):
    """A regression double that raises ``error``."""
    def fit(settings):
        raise error
    return fit


class _Opaque:
    """A regression output with no ``get``."""


OK_METRICS = {"n_results": 4, "n_raw_below_alpha": 2, "n_below_alpha": 2}
POSITIVE = {"positive_present": True, "positive_rank": 1,
            "positive_effect": 2.5, "positive_q": 0.001, "positive_p": 0.0001}


def test_sweep_child_main_always_ends_in_a_result_file(tmp_path, monkeypatch,
                                                       capsys, subtests):
    """Usage errors return 2; every trial, ok or failed, writes one exact result document and returns 0."""
    with subtests.test(msg="argument-count"):
        for argv in ([], ["only-one.json"], ["a", "b", "c"]):
            assert sweep_child.main(argv) == 2
            assert capsys.readouterr().err == USAGE
        monkeypatch.setattr(sys, "argv", ["spacr.sweep_child", "one-path"])
        assert sweep_child.main() == 2
        assert capsys.readouterr().err == USAGE

    real_open = open

    def refuse_error_txt(path, *args, **kwargs):
        if str(path).endswith("error.txt"):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    def broken_alias(results, names):
        raise RuntimeError("alias lookup is broken")

    cases = [
        # id, payload (src=True adds a run folder), regression, patches,
        # expected document (minus seconds/worker stamp), traceback written
        ("success", {"settings": {"fdr_alpha": 0.05}, "trial_id": 7},
         lambda s: {"results": _results_frame(), "model": None}, [],
         {"status": "ok", "trial_id": 7, **OK_METRICS}, None),
        ("settings-only-payload", {"gene_weights": "sum", "fdr_alpha": 0.1},
         lambda s: {"results": _results_frame()}, [],
         {"status": "ok", "trial_id": None, **OK_METRICS}, None),
        ("control-aliases", {"settings": {},
                             "controls": {"positive": "TGGT1_POS_1"}},
         lambda s: {"results": _results_frame()}, [],
         {"status": "ok", "trial_id": None, **OK_METRICS, **POSITIVE}, None),
        ("broken-alias-lookup", {"settings": {}, "controls": {"positive": "POS"}},
         lambda s: {"results": _results_frame()},
         [(parameter_sweep, "_named_control_rows", broken_alias)],
         {"status": "ok", "trial_id": None, **OK_METRICS}, None),
        ("non-frame-results", {"settings": {}, "controls": {"positive": "POS"}},
         lambda s: {"results": ["not", "a", "frame"]}, [],
         {"status": "ok", "trial_id": None}, None),
        ("output-without-get", {"settings": {}, "controls": {"positive": "POS"}},
         lambda s: _Opaque(), [],
         {"status": "ok", "trial_id": None}, None),
        ("unserialisable-metric", {"settings": {}},
         lambda s: {"results": _results_frame()},
         [(trial_metrics, "summarise_trial",
           lambda output, settings: {"odd": object()})],
         {"status": "ok", "trial_id": None, "odd": "<object object"}, None),
        ("failed-fit-with-run-folder", {"settings": {"src": True},
                                        "trial_id": 3},
         _raises(ValueError("singular design matrix\nsecond line is dropped")),
         [], {"status": "failed", "trial_id": 3, "_overload": False,
              "error_type": "ValueError", "error": "singular design matrix"},
         "ValueError: singular design matrix"),
        ("failure-without-src", {"settings": {}},
         _raises(MemoryError("out of memory")), [],
         {"status": "failed", "trial_id": None, "_overload": True,
          "error_type": "MemoryError", "error": "out of memory"}, None),
        ("unwritable-run-folder", {"settings": {"src": True}},
         _raises(RuntimeError("fit blew up")),
         [("builtins", "open", refuse_error_txt)],
         {"status": "failed", "trial_id": None, "_overload": False,
          "error_type": "RuntimeError", "error": "fit blew up"}, None),
        ("keyboard-interrupt", {"settings": {}}, _raises(KeyboardInterrupt()),
         [], {"status": "failed", "trial_id": None, "_overload": False,
              "error_type": "KeyboardInterrupt", "error": ""}, None),
    ]
    for case_id, payload, regression, patches, expected, traceback_line in \
            cases:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            run = folder / "run"
            run.mkdir(parents=True)
            settings = payload.get("settings", payload)
            if settings.get("src") is True:
                settings["src"] = str(run)
            settings_path = folder / "settings.json"
            out_path = folder / "result.json"
            settings_path.write_text(json.dumps(payload), encoding="utf-8")

            order = []
            seen = {}

            def fit(settings, regression=regression):
                order.append("regression")
                seen["settings"] = dict(settings)
                return regression(settings)

            with monkeypatch.context() as patch:
                patch.setattr(parameter_sweep, "_pin_threads",
                              lambda *a, **k: order.append("pin"))
                patch.setattr(ml, "perform_regression", fit)
                for target, name, value in patches:
                    patch.setattr(target if target != "builtins"
                                  else __import__("builtins"), name, value)
                assert sweep_child.main([str(settings_path),
                                         str(out_path)]) == 0

            assert order == ["pin", "regression"]
            assert seen["settings"] == {k: v for k, v in settings.items()}
            document = json.loads(out_path.read_text(encoding="utf-8"))
            seconds = document.pop("seconds")
            assert isinstance(seconds, float) and seconds >= 0
            stamp = document.pop("_resource_worker")
            assert (stamp["worker_kind"], stamp["worker_id"]) == (
                "parameter_sweep_trial", str(payload.get("trial_id")))
            if case_id == "unserialisable-metric":
                assert document.pop("odd").startswith(expected.pop("odd"))
            assert document == expected
            written = sorted(p.name for p in folder.rglob("error.txt"))
            if traceback_line is None:
                assert written == []
            else:
                text = (run / "error.txt").read_text(encoding="utf-8")
                assert text.startswith("Traceback (most recent call last):")
                assert text.rstrip().splitlines()[-2:] == [
                    traceback_line, "second line is dropped"]
