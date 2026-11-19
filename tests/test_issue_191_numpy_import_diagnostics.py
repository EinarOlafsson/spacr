"""Preserve NumPy ABI error types and the import stack without private paths."""

import io
import json
import threading
import types

import pytest

from spacr import crashreport as cr
from spacr.logging_util import _ErrorStream


NUMPY_MESSAGE = """A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.4.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
"""
IMPORT_STACK = (
    'Traceback (most recent call last):  File "/home/private_person/private_project/run.py", line 7, in main\n'
    '    import array_backend\n'
    '  File "/home/private_person/env/lib/python3.12/site-packages/array_backend/_loader.py", line 11, in load_extension\n'
    '    import array_backend._native\n'
)
GUARD_TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "/home/private_person/env/lib/python3.12/site-packages/numpy/core/_multiarray_umath.py", line 46, in __getattr__\n'
    '    raise ImportError(msg)\n'
    'ImportError:\n'
)


def _numpy_write(stream, text):
    """Write a diagnostic from a simulated NumPy ABI guard caller."""
    stream.write(text)


@pytest.fixture(autouse=True)
def isolated_reporter(monkeypatch):
    """Keep reporter state and transport isolated from the user's installation."""
    monkeypatch.setattr(cr, "_ERROR_DISPATCH_LOCAL", threading.local())
    monkeypatch.setattr(cr, "install_id", lambda: "00000000-0000-4000-8000-000000000001")
    monkeypatch.setenv(cr.NO_NETWORK_ENV, "1")


def test_multiline_numpy_importerror_keeps_its_type_and_complete_message():
    """The report must not replace ImportError with the final prose sentence."""
    kind, message, frames = cr._parse_traceback_text(GUARD_TRACEBACK + NUMPY_MESSAGE)
    assert kind == "ImportError"
    assert message == " ".join(NUMPY_MESSAGE.split())
    assert frames == [{"file": "_multiarray_umath.py", "function": "__getattr__", "line": 46}]


def test_numpy_import_stack_survives_the_following_guard_traceback():
    """Keep NumPy's separately printed importer stack beside the guard frame."""
    text = NUMPY_MESSAGE + "\n" + IMPORT_STACK + GUARD_TRACEBACK + NUMPY_MESSAGE
    summary = cr.public_summary(traceback_text=text, error_text=text)
    assert summary["exception_type"] == "ImportError"
    assert [frame["file"] for frame in summary["frames"]] == [
        "<user-code>", "_loader.py", "_multiarray_umath.py",
    ]
    assert summary["frames"][1]["function"] == "load_extension"
    assert "compiled using NumPy 1.x" in summary["message"]
    public = json.dumps(summary)
    for secret in ("private_person", "private_project", "/home/", "run.py"):
        assert secret not in public


@pytest.mark.parametrize("chunking", ["whole", "lines", "characters"])
def test_terminal_numpy_diagnostic_retains_importer_before_dispatch(monkeypatch, chunking):
    """The terminal capture must retain the preamble and stack across writes."""
    summaries = []

    def capture(text, **kwargs):
        """Capture sanitized summaries without starting any report worker."""
        summaries.append(cr.public_summary(traceback_text=kwargs.get("traceback_text", text)))

    monkeypatch.setattr(cr, "_dispatch_error", capture)
    diagnostic = "\n" + NUMPY_MESSAGE + "\n" + IMPORT_STACK
    raised = GUARD_TRACEBACK + NUMPY_MESSAGE
    text = diagnostic + raised
    original = io.StringIO()
    stream = _ErrorStream(original)
    numpy_write = types.FunctionType(_numpy_write.__code__, {"__name__": "numpy.core._multiarray_umath"})
    for part, write in ((diagnostic, numpy_write), (raised, _numpy_write)):
        chunks = ([part] if chunking == "whole" else part.splitlines(keepends=True)
                  if chunking == "lines" else part)
        for chunk in chunks:
            write(stream, chunk)
    stream.flush()
    assert original.getvalue() == text
    assert len(summaries) == 1
    assert summaries[0]["exception_type"] == "ImportError"
    assert [frame["file"] for frame in summaries[0]["frames"]] == [
        "<user-code>", "_loader.py", "_multiarray_umath.py",
    ]
    assert "compiled using NumPy 1.x" in summaries[0]["message"]
    assert "private_person" not in json.dumps(summaries)
    stream.write("Traceback (most recent call last):\n")
    stream.write('  File "/usr/lib/python3.12/other.py", line 9, in other\n')
    stream.write("ValueError: independent failure\n")
    assert len(summaries) == 2
    assert summaries[1]["message"] == "independent failure"
    assert summaries[1]["frames"] == [{"file": "other.py", "function": "other", "line": 9}]
    assert "NumPy" not in json.dumps(summaries[1])


def test_ordinary_chained_exception_still_keeps_only_its_final_frames():
    """An ordinary cause chain must not borrow the earlier exception's stack."""
    text = (
        'Traceback (most recent call last):\n'
        '  File "/usr/lib/python3.12/first.py", line 5, in first\n'
        'ValueError: earlier failure\n\n'
        'The above exception was the direct cause of the following exception:\n\n'
        'Traceback (most recent call last):\n'
        '  File "/usr/lib/python3.12/last.py", line 9, in last\n'
        'RuntimeError: final failure\n'
    )
    kind, message, frames = cr._parse_traceback_text(text)
    assert (kind, message) == ("RuntimeError", "final failure")
    assert frames == [{"file": "last.py", "function": "last", "line": 9}]


def test_real_numpy_guard_emission_preserves_its_calling_stack(monkeypatch):
    """Exercise NumPy's actual stderr protocol before its one-frame exception."""
    import importlib
    import numpy as np
    import traceback

    if int(np.__version__.split(".")[0]) < 2:
        pytest.skip("NumPy 1 has no NumPy 2 ABI guard")
    _multiarray_umath = importlib.import_module("numpy.core._multiarray_umath")

    summaries = []

    def capture(text, **kwargs):
        """Observe the public result without dispatching it to any service."""
        summaries.append(cr.public_summary(traceback_text=kwargs.get("traceback_text", text)))

    monkeypatch.setattr(cr, "_dispatch_error", capture)
    stream = _ErrorStream(io.StringIO())
    monkeypatch.setattr("sys.stderr", stream)
    try:
        _multiarray_umath.__getattr__("_ARRAY_API")
    except ImportError as exc:
        last_frame = exc.__traceback__
        while last_frame.tb_next is not None:
            last_frame = last_frame.tb_next
        traceback.print_exception(type(exc), exc, last_frame, file=stream)
    else:
        pytest.fail("The installed NumPy did not emit its ABI guard")
    stream.flush()
    assert len(summaries) == 1
    assert summaries[0]["exception_type"] == "ImportError"
    assert "compiled using NumPy 1.x" in summaries[0]["message"]
    assert any(frame["function"] == "test_real_numpy_guard_emission_preserves_its_calling_stack"
               for frame in summaries[0]["frames"])
    assert summaries[0]["frames"][-1]["file"] == "_multiarray_umath.py"
