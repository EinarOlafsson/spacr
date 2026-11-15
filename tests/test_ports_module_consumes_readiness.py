"""A module's declared input ports decide whether a folder can feed it.

:func:`spacr.ports.module_ports` declares what each module consumes. The
chaining layer reads those declarations twice: :func:`spacr.chaining.satisfies`
answers "does this folder hold what the module needs?", and
``chaining._problems_for`` explains a drop that found nothing in the module's
own ``check_ready`` words (or says the folder is no project when there is no
port to check).
"""
from __future__ import annotations

import os

from spacr import chaining, ports


def _project(root, name="plate1"):
    """A plate folder with ``merged/field1.npy`` and ``measurements/``."""
    project = os.path.join(str(root), name)
    for directory in ("merged", "measurements"):
        os.makedirs(os.path.join(project, directory), exist_ok=True)
    with open(os.path.join(project, "merged", "field1.npy"), "wb") as handle:
        handle.write(b"\x00")
    return project


def test_measure_consumes_merged_arrays_and_explains_their_absence(subtests,
                                                                   tmp_path):
    """Exact port declaration, satisfaction and problem wording."""
    spec = ports.module_ports("measure")
    consumes = tuple(spec.consumes)
    empty = tmp_path / "bare"
    empty.mkdir()

    with subtests.test(msg="measure consumes one required merged-arrays port"):
        assert spec.key == "measure"
        assert [(p.kind, p.role, p.path, p.pattern, p.required)
                for p in consumes] == [
            ("merged-arrays", "merged", "merged", "*.npy", True)]

    with subtests.test(msg="a folder missing the port does not satisfy it"):
        assert chaining.satisfies(str(empty), consumes) is False

    with subtests.test(msg="the layout written by mask satisfies it"):
        assert chaining.satisfies(_project(tmp_path), consumes) is True

    with subtests.test(msg="check_ready explains the empty folder"):
        problems = chaining._problems_for("measure", consumes, str(empty), None)
        assert [(p.severity, p.setting, p.message, p.fix)
                for p in problems] == [
            ("error", "merged",
             f"no merged-arrays at {empty}{os.sep}merged{os.sep}*.npy",
             "Run mask or timelapse on this project first.")]

    with subtests.test(msg="no ports means the folder is not a project"):
        problems = chaining._problems_for("measure", (), str(empty), None)
        assert len(problems) == 1
        assert problems[0].message == f"{empty} is not a spaCR project folder"
        assert problems[0].fix.startswith("Drop the plate folder itself")
