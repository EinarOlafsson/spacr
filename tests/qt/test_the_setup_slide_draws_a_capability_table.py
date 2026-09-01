"""The first setup slide shows a table, not four sentences.

Asked for on 2026-08-31: the slide "describes the capabilities of the
computer more granularly but instead of the way you show it in lines do a
table instead".

The CONTENT was already right -- it comes from
``accelerator.capabilities()``, which reports per task because one verdict
cannot be honest: on Metal the segmentation and the classifier are
accelerated while the cuML reductions are not. It is the SHAPE that
changed. Four sentences in a column read as four separate remarks rather
than one answer with an axis.

The middle column is derived from the same function ``spacr-doctor`` and
the README table render, so the three surfaces cannot disagree about one
machine. These tests fake two backends and assert the table DIFFERS where
the backends differ -- which is the entire reason it exists.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from spacr import accelerator as acc
from spacr.qt.widgets import setup_slides as slides


@pytest.fixture
def unprobed(monkeypatch):
    """No cached accelerator, so each test states its own machine."""
    monkeypatch.setattr(acc, "_CACHED", None, raising=False)
    yield
    monkeypatch.setattr(acc, "_CACHED", None, raising=False)


def _rows(html):
    """``[(library, where, task)]`` from the rendered table."""
    return [tuple(re.sub(r"<[^>]+>", "", cell).strip()
                  for cell in re.findall(r"<td[^>]*>(.*?)</td>", row))
            for row in re.findall(r"<tr>(.*?)</tr>", "".join(html))]


def _as(monkeypatch, **kwargs):
    """Render the capability table as it would look on that machine."""
    monkeypatch.setattr(acc, "_CACHED", acc.Accelerator(**kwargs),
                        raising=False)
    return _rows(slides.SetupSlides._what_this_machine_can_do())


def test_it_is_a_table_and_not_a_list_of_lines(unprobed, monkeypatch):
    """The change asked for, asserted on the markup."""
    monkeypatch.setattr(acc, "_CACHED",
                        acc.Accelerator(kind="cuda", device="cuda",
                                        label="NVIDIA"), raising=False)
    html = "".join(slides.SetupSlides._what_this_machine_can_do())
    assert "<table" in html and "<tr>" in html


def test_every_row_names_a_library_a_place_and_a_task(unprobed, monkeypatch):
    """Three columns, in the order asked for."""
    rows = _as(monkeypatch, kind="cuda", device="cuda", label="NVIDIA")
    assert rows, "the table drew no rows"
    for library, where, task in rows:
        assert library and task
        assert where in ("GPU", "CPU")


def test_cuda_and_metal_differ_where_the_backends_differ(unprobed,
                                                          monkeypatch):
    """THE WHOLE REASON THE TABLE EXISTS.

    cuML ships for CUDA only, so the reductions row is GPU on NVIDIA and
    CPU on Metal -- while the Cellpose row is GPU on both. A table that
    showed one verdict for the machine would be wrong about one of them,
    and a test that only checked it rendered would not notice.
    """
    cuda = dict((task, where) for _lib, where, task in
                _as(monkeypatch, kind="cuda", device="cuda",
                    label="NVIDIA"))
    metal = dict((task, where) for _lib, where, task in
                 _as(monkeypatch, kind="mps", device="mps", label="AMD",
                     float64=False))
    assert cuda["Machine learning"] == "GPU"
    assert metal["Machine learning"] == "CPU", (
        "cuML is CUDA-only, so this row cannot be GPU on Metal")
    assert cuda["Segmentation"] == metal["Segmentation"] == "GPU", (
        "Cellpose is accelerated on both, so this row must agree")


def test_a_machine_with_no_compute_gpu_says_cpu_for_the_compute_rows(
        unprobed, monkeypatch):
    """The COMPUTE rows, and the backdrop is deliberately not one.

    Written first as "CPU everywhere" and that was wrong: the backdrop
    asks whether OpenGL is available for DISPLAY, which is a different
    question from whether torch has a compute device. A machine with a
    display GPU and a CPU-only torch build draws the shader backdrop and
    segments on the CPU, and reporting the backdrop as CPU there would
    be false.

    So the three compute rows are asserted and the visualization row is
    asserted to be answered at all, not to any particular value.
    """
    rows = {task: where for _lib, where, task in
            _as(monkeypatch, kind="cpu", device="cpu", label="CPU")}
    for task in ("Segmentation", "Classification", "Machine learning"):
        assert rows[task] == "CPU", task
    assert rows["Visualization"] in ("GPU", "CPU")


def test_the_cellpose_row_takes_its_version_from_the_package(unprobed,
                                                             monkeypatch):
    """A hardcoded "Cellpose 4" is a claim the next Cellpose falsifies,
    and this label is on the first screen a new user sees."""
    label = slides.SetupSlides._cellpose_label()
    assert label.startswith("Cellpose")
    try:
        import cellpose
    except Exception:                                        # noqa: BLE001
        pytest.skip("cellpose is not installed")
    version = str(getattr(cellpose, "version", None)
                  or getattr(cellpose, "__version__", ""))
    assert version.split(".")[0] in label


def test_a_renamed_capability_drops_its_row_rather_than_drawing_a_blank(
        unprobed, monkeypatch):
    """A blank middle column reads as "spaCR does not know", which is a
    worse thing to say than nothing."""
    monkeypatch.setattr(acc, "_CACHED",
                        acc.Accelerator(kind="cuda", device="cuda",
                                        label="NVIDIA"), raising=False)
    monkeypatch.setattr(slides, "GPU_TABLE_ROWS",
                        (("Nonsense", "NoSuchPrefix", "Nothing"),))
    assert _rows(slides.SetupSlides._what_this_machine_can_do()) == []


@pytest.mark.parametrize("kind,expected", [
    ("cuda", "CUDA"), ("rocm", "ROCm"), ("mps", "Metal"),
    ("xpu", "XPU"), ("directml", "DirectML"), ("cpu", ""),
])
def test_the_library_is_named_for_every_backend(unprobed, monkeypatch,
                                                kind, expected):
    """The LIBRARY, not the vendor.

    A user reading "Metal" beside an AMD card learns why ROCm is
    irrelevant on their machine -- which is exactly what 319's own
    backend table got wrong, and it hid a 139x speedup.
    """
    monkeypatch.setattr(acc, "_CACHED",
                        acc.Accelerator(kind=kind, device=kind,
                                        label="card"), raising=False)
    assert slides._gpu_library() == expected
