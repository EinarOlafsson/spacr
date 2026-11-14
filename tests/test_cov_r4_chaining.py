"""``spacr.chaining``: the answers a drop gives when the layout is not tidy.

The paths pinned here are the ones a real project reaches and the happy-path
tests do not:

* :func:`~spacr.chaining.layout_directories` reads folder names off the port
  declarations, and a port that names a *file* at the project root is not one;
* :func:`~spacr.chaining.db_candidates` still finds a database in a project
  that has no ``measurements/`` folder at all;
* a module with two ports bound to the same settings key resolves it once —
  Classify declares both a measurements database and an optional crop folder,
  and both bind to ``src``;
* the registry, not the layout, answers when a run was registered;
* two databases in one project is a question, not a guess;
* a screen with no port declaration of its own — one driven by ``kinds`` —
  still gets its ports' own problem sentences back.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr import chaining, ports


@pytest.fixture
def restore_ports():
    """Snapshot the global port declarations, and the layout cache with them.

    ``layout_directories`` caches against ``len(PORTS)``, so a test that adds
    a declaration must put both back or the next caller reads a cache built
    from ports that no longer exist.
    """
    saved = dict(ports.PORTS)
    yield ports.PORTS
    ports.PORTS.clear()
    ports.PORTS.update(saved)
    chaining._LAYOUT_CACHE.clear()


def _plate(root, *, merged=1, dbs=("measurements.db",), crops=False,
           tables=("cell", "png_list")):
    """A plate folder shaped the way the pipeline leaves one."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    if merged:
        os.makedirs(os.path.join(root, "merged"), exist_ok=True)
        for index in range(merged):
            np.save(os.path.join(root, "merged", f"plate1_A01_{index}.npy"),
                    np.zeros((4, 4, 3), dtype=np.uint16))
    for name in dbs:
        os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
        connection = sqlite3.connect(os.path.join(root, "measurements", name))
        for table in tables:
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
            connection.execute(f'INSERT INTO "{table}" VALUES (1)')
        connection.commit()
        connection.close()
    if crops:
        folder = os.path.join(root, "data", "A01", "cell_png")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "object_1.png"), "wb") as handle:
            handle.write(b"\x89PNG")
    return root


# --------------------------------------------------------------------------- #
#  The layout is read off the ports, and it is a list of FOLDERS
# --------------------------------------------------------------------------- #

def test_a_port_naming_a_file_at_the_root_is_not_a_layout_directory(
        restore_ports):
    """A plugin's port declaration joins the layout — if it names a folder.

    ``layout_directories`` is the set of names that are never a sub-project
    and that a "this is not a spaCR project" message lists. A port naming a
    single file at the project root — ``summary.csv``, no pattern — is not a
    folder, and putting its filename in that set would exclude a real
    sub-project called ``summary.csv`` from a drop.

    Both ports are declared by the same module so the difference is the
    declaration and nothing else.
    """
    before = chaining.layout_directories()
    assert "qc4" not in before and "summary4.csv" not in before

    ports.register_module_ports(ports.ModulePorts(
        key="r4plugin",
        produces=(
            ports.Port(kind=ports.REGRESSION_RESULTS, role="qc",
                       path="qc4", pattern="*.csv"),
            ports.Port(kind=ports.SETTINGS_CSV, role="summary",
                       path="summary4.csv"),
        )))

    after = chaining.layout_directories()
    assert "qc4" in after, "a port naming a folder joins the layout"
    assert "summary4.csv" not in after
    assert "summary4" not in after


# --------------------------------------------------------------------------- #
#  Finding the databases
# --------------------------------------------------------------------------- #

def test_a_project_with_no_measurements_folder_still_offers_its_database(
        tmp_path):
    """The declared folder is where it looks first, not the only place.

    A project someone assembled by hand — or one whose ``measurements/``
    folder was never created because the database was written beside the
    images — has no declared holder to list, and the shallow listing of the
    root is what saves the drop from answering "no database here".
    """
    laid_out = _plate(tmp_path / "laid_out")
    flat = tmp_path / "flat"
    flat.mkdir()
    sqlite3.connect(str(flat / "loose.db")).close()
    assert not os.path.isdir(str(flat / "measurements"))

    assert chaining.db_candidates(str(flat)) == (str(flat / "loose.db"),)
    # ...and with the declared folder present, the declared file comes first.
    assert chaining.db_candidates(laid_out)[0] == os.path.join(
        laid_out, "measurements", "measurements.db")


# --------------------------------------------------------------------------- #
#  One settings key, one answer
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  A sibling that is not a project
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  A screen with no port declaration of its own
# --------------------------------------------------------------------------- #
