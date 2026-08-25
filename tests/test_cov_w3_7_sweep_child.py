"""The contained-trial worker: ``python -m spacr.sweep_child in.json out.json``.

A sweep runs each trial in a fresh interpreter so a runaway fit cannot take
the parent down with it. The parent reads the trial's answer back out of a
JSON file, so everything this module does has to end in that file -- a
successful fit, a failed one, and the control aliases the sweep table is
judged on alike.

The regression itself is replaced per test; the metric summarisation
(:mod:`spacr.trial_metrics`) and the control-alias lookup
(``spacr.parameter_sweep._named_control_rows``) run for real against a real
DataFrame, because those are what fill the row.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

import spacr.sweep_child as sweep_child


@pytest.fixture
def trial_files(tmp_path):
    """Return a writer for the settings file plus the two paths ``main`` takes."""
    settings_path = tmp_path / "settings.json"
    out_path = tmp_path / "result.json"

    def write(payload):
        settings_path.write_text(json.dumps(payload), encoding="utf-8")
        return [str(settings_path), str(out_path)]

    write.out_path = out_path
    write.tmp_path = tmp_path
    return write


@pytest.fixture(autouse=True)
def _quiet_thread_pinning(monkeypatch):
    """Record the ``_pin_threads`` call instead of resizing this process's pools.

    ``parameter_sweep._pin_threads`` rewrites ``OMP_NUM_THREADS`` and friends
    and shrinks the live BLAS pool. That is right for a one-trial child
    process and wrong for the pytest process every other test shares, so the
    call is observed rather than performed.
    """
    import spacr.parameter_sweep as parameter_sweep

    calls = []
    monkeypatch.setattr(parameter_sweep, "_pin_threads",
                        lambda *a, **k: calls.append((a, k)))
    return calls


def _results_frame():
    """A minimal regression result table with a recoverable positive control."""
    return pd.DataFrame({
        "grna": ["TGGT1_POS_1", "TGGT1_POS_2", "guide_a", "guide_b"],
        "coefficient": [2.5, 2.1, 0.2, -0.1],
        "p_value": [0.0001, 0.0004, 0.4, 0.9],
        "q_value": [0.001, 0.002, 0.6, 0.95],
        "n": [40, 38, 41, 39],
    })


def test_two_arguments_are_required(capsys):
    """Anything but exactly two paths is a usage error, and says so on stderr."""
    assert sweep_child.main([]) == 2
    assert sweep_child.main(["only-one.json"]) == 2
    assert sweep_child.main(["a", "b", "c"]) == 2
    err = capsys.readouterr().err
    assert "usage: python -m spacr.sweep_child" in err


def test_argv_defaults_to_the_process_arguments(monkeypatch, capsys):
    """With no ``argv`` the worker reads ``sys.argv``, the way the parent execs it."""
    monkeypatch.setattr(sys, "argv", ["spacr.sweep_child", "one-path-only"])
    assert sweep_child.main() == 2
    assert "usage:" in capsys.readouterr().err


def test_a_successful_trial_writes_status_ok_and_its_metrics(
        trial_files, monkeypatch):
    """A fit that returns results lands in the output file with its metrics."""
    import spacr.ml as ml

    seen = {}

    def fake_regression(settings):
        seen["settings"] = dict(settings)
        return {"results": _results_frame(), "model": None}

    monkeypatch.setattr(ml, "perform_regression", fake_regression)

    argv = trial_files({"settings": {"src": str(trial_files.tmp_path),
                                     "fdr_alpha": 0.05},
                        "trial_id": 7})
    assert sweep_child.main(argv) == 0

    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["trial_id"] == 7
    assert result["seconds"] >= 0
    # summarise_trial ran for real: the hit counts come from the frame above.
    assert result["n_results"] == 4
    assert seen["settings"]["fdr_alpha"] == 0.05


def test_thread_pinning_happens_before_the_fit(trial_files, monkeypatch,
                                               _quiet_thread_pinning):
    """Each trial pins itself to one compute thread before it imports the model."""
    import spacr.ml as ml

    order = []

    def fake_regression(settings):
        order.append("regression")
        return {"results": _results_frame()}

    monkeypatch.setattr(ml, "perform_regression", fake_regression)
    sweep_child.main(trial_files({"settings": {}}))

    assert _quiet_thread_pinning, "the child never pinned its thread pools"
    assert order == ["regression"]


def test_a_settings_only_payload_is_accepted_as_the_settings(
        trial_files, monkeypatch):
    """A file with no ``settings`` key is itself the settings mapping."""
    import spacr.ml as ml

    seen = {}

    def fake_regression(settings):
        seen["settings"] = dict(settings)
        return {"results": _results_frame()}

    monkeypatch.setattr(ml, "perform_regression", fake_regression)

    argv = trial_files({"gene_weights": "sum", "fdr_alpha": 0.1})
    assert sweep_child.main(argv) == 0
    assert seen["settings"]["gene_weights"] == "sum"
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["trial_id"] is None


def test_control_aliases_reach_the_row(trial_files, monkeypatch):
    """``controls`` in the payload adds the caller's own ``{alias}_*`` columns.

    ``positive_rank`` is the column the sweep table is judged on, and it is
    built from this mapping, so a contained trial that skipped it would look
    exactly like a control that was never recovered.
    """
    import spacr.ml as ml

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: {"results": _results_frame()})

    argv = trial_files({"settings": {},
                        "controls": {"positive": "TGGT1_POS_1"}})
    assert sweep_child.main(argv) == 0

    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["positive_present"] is True
    assert result["positive_effect"] == pytest.approx(2.5)


def test_a_broken_alias_lookup_does_not_sink_the_trial(trial_files, monkeypatch):
    """The alias block is best-effort: its failure leaves ``status`` at ``ok``."""
    import spacr.ml as ml
    import spacr.parameter_sweep as parameter_sweep

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: {"results": _results_frame()})

    def explode(results, names):
        raise RuntimeError("alias lookup is broken")

    monkeypatch.setattr(parameter_sweep, "_named_control_rows", explode)

    argv = trial_files({"settings": {}, "controls": {"positive": "POS"}})
    assert sweep_child.main(argv) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert "positive_present" not in result


def test_controls_with_a_non_frame_result_add_nothing(trial_files, monkeypatch):
    """An output whose ``results`` is not a DataFrame skips the alias columns."""
    import spacr.ml as ml

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: {"results": ["not", "a", "frame"]})

    argv = trial_files({"settings": {}, "controls": {"positive": "POS"}})
    assert sweep_child.main(argv) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert not any(key.startswith("positive_") for key in result)


def test_an_output_without_get_skips_the_alias_block(trial_files, monkeypatch):
    """A regression that returns a bare object still produces a row."""
    import spacr.ml as ml

    class Opaque:
        pass

    monkeypatch.setattr(ml, "perform_regression", lambda settings: Opaque())

    argv = trial_files({"settings": {}, "controls": {"positive": "POS"}})
    assert sweep_child.main(argv) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert not any(key.startswith("positive_") for key in result)


def test_a_failed_fit_is_a_result_not_a_crash(trial_files, monkeypatch):
    """The trial's exception is written to the output file the parent reads."""
    import spacr.ml as ml

    def explode(settings):
        raise ValueError("singular design matrix\nsecond line is dropped")

    monkeypatch.setattr(ml, "perform_regression", explode)

    folder = trial_files.tmp_path / "run"
    folder.mkdir()
    argv = trial_files({"settings": {"src": str(folder)}, "trial_id": 3})
    # Zero, because the parent judges the trial by the file it wrote, not by
    # the child's exit status; it only falls back to the status when the
    # child was killed before writing anything.
    assert sweep_child.main(argv) == 0

    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["trial_id"] == 3
    assert result["error_type"] == "ValueError"
    assert result["error"] == "singular design matrix"

    traceback_text = (folder / "error.txt").read_text(encoding="utf-8")
    assert "ValueError: singular design matrix" in traceback_text
    assert "Traceback" in traceback_text


def test_a_failure_with_no_src_writes_no_traceback_file(trial_files,
                                                        monkeypatch):
    """Without a run folder the error still reaches the result file."""
    import spacr.ml as ml

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: (_ for _ in ()).throw(
                            MemoryError("out of memory")))

    assert sweep_child.main(trial_files({"settings": {}})) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "MemoryError"
    assert not list(trial_files.tmp_path.glob("**/error.txt"))


def test_an_unwritable_run_folder_does_not_lose_the_result(trial_files,
                                                           monkeypatch):
    """A traceback that cannot be written is dropped; the result file is not."""
    import spacr.ml as ml

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: (_ for _ in ()).throw(
                            RuntimeError("fit blew up")))

    real_open = open

    def refuse_error_txt(path, *args, **kwargs):
        if str(path).endswith("error.txt"):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", refuse_error_txt)

    argv = trial_files({"settings": {"src": str(trial_files.tmp_path)}})
    assert sweep_child.main(argv) == 0
    monkeypatch.undo()
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"


def test_a_keyboard_interrupt_is_recorded_rather_than_propagated(
        trial_files, monkeypatch):
    """``BaseException`` is caught: an interrupted trial still leaves a row."""
    import spacr.ml as ml

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: (_ for _ in ()).throw(
                            KeyboardInterrupt()))

    assert sweep_child.main(trial_files({"settings": {}})) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "KeyboardInterrupt"


def test_the_result_file_survives_unserialisable_metrics(trial_files,
                                                         monkeypatch):
    """``json.dump`` uses ``default=str`` so an exotic metric cannot lose the row."""
    import spacr.ml as ml
    import spacr.trial_metrics as trial_metrics

    monkeypatch.setattr(ml, "perform_regression",
                        lambda settings: {"results": _results_frame()})
    monkeypatch.setattr(trial_metrics, "summarise_trial",
                        lambda output, settings: {"odd": object()})

    assert sweep_child.main(trial_files({"settings": {}})) == 0
    result = json.loads(trial_files.out_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["odd"].startswith("<object object")


def test_the_worker_volunteers_for_the_oom_killer():
    """Importing the child raises its own OOM score so the parent is not the target.

    The write happens at import; this reads back what the running interpreter
    now carries. Non-Linux kernels have no such file and the module tolerates
    that, so the assertion only applies where the file exists.
    """
    path = f"/proc/{os.getpid()}/oom_score_adj"
    if not os.path.exists(path):
        pytest.skip("no /proc/<pid>/oom_score_adj on this kernel")
    with open(path) as handle:
        assert int(handle.read().strip()) == 800


def test_blas_thread_limits_are_set_before_numpy_can_size_its_pool():
    """The module sets a one-thread default for every BLAS variable it names."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(variable), f"{variable} was left unset"
