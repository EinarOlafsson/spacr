"""The per-object grid writes THROUGH to the widgets the pipeline reads.

The grid edits values; the settings panel is a dictionary of widgets, and
``collect()`` -- what the run actually uses -- reads the widgets. Mount the
grid without a binding and the screen holds two answers to every per-object
question: the table showing what was typed, and ``collect()`` still returning
what the widget holds. The run would use the second one silently.

So these tests are all one property: **after any edit in the table, the
widget behind that cell holds the new value**. The grid is a view; the widget
stays the source of truth; nothing downstream learns the grid exists.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.object_grid_binding import ObjectGridBinding  # noqa: E402
from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid  # noqa: E402


class FakePanel:
    """``collect`` and ``set_value_for_key``, which is all a binding needs."""

    def __init__(self, settings):
        self._settings = dict(settings)
        self.writes = []

    def collect(self):
        return dict(self._settings)

    def set_value_for_key(self, key, value):
        if key not in self._settings:
            return False
        self._settings[key] = value
        self.writes.append((key, value))
        return True


class ReseedingPanel(FakePanel):
    """A panel that reseeds its grid whenever a widget changes, as a screen does."""

    def __init__(self, settings):
        super().__init__(settings)
        self._binding = None
        self.reseeds_during_write = 0
        self._writing = False

    def bind(self, binding):
        self._binding = binding

    def set_value_for_key(self, key, value):
        ok = super().set_value_for_key(key, value)
        if ok and self._binding is not None:
            if self._writing:
                self.reseeds_during_write += 1
            self._writing = True
            try:
                self._binding.seed()
            finally:
                self._writing = False
        return ok


@pytest.fixture
def mask_settings():
    """Mask's own defaults, not a fixture written to suit."""
    from spacr.settings import get_timelapse_settings

    return get_timelapse_settings()


@pytest.fixture
def bound(qtbot, qt_theme_applied, mask_settings):
    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    panel = FakePanel(mask_settings)
    binding = ObjectGridBinding(grid, panel)
    binding.seed()
    return grid, panel, binding


# ---------------------------------------------------------------------------
# What the grid speaks for
# ---------------------------------------------------------------------------

def test_the_grid_claims_only_the_per_object_keys(bound):
    """``src`` and ``verbose`` are not table cells and are not claimed."""
    _, panel, binding = bound
    owned = binding.owned_keys()

    assert owned, "the grid claimed nothing at all"
    for key in ("src", "verbose", "n_jobs", "timelapse"):
        if key in panel.collect():
            assert key not in owned, f"{key} is not a per-object question"


def test_every_claimed_key_is_a_real_setting(bound):
    """A claimed key the panel does not have would be a key nothing reads."""
    _, panel, binding = bound
    settings = panel.collect()

    for key in binding.owned_keys():
        assert key in settings, f"{key} is not in the settings"


# ---------------------------------------------------------------------------
# Grid -> widget
# ---------------------------------------------------------------------------

def test_an_edited_cell_reaches_the_widget_behind_it(bound):
    """The property the whole binding exists for."""
    grid, panel, binding = bound
    key = _a_numeric_key(panel, binding)
    was = panel.collect()[key]

    _edit(grid, key, float(was) + 3)
    binding.write_through()

    assert panel.collect()[key] != was
    assert float(panel.collect()[key]) == float(was) + 3


def test_a_value_that_did_not_change_is_not_written(bound):
    """Setting a widget to what it holds still makes it emit."""
    grid, panel, binding = bound

    binding.write_through()

    assert panel.writes == [], f"wrote without an edit: {panel.writes}"


def test_only_the_edited_key_is_written(bound):
    """One cell edited is one widget touched, not seventy-eight."""
    grid, panel, binding = bound
    key = _a_numeric_key(panel, binding)

    _edit(grid, key, float(panel.collect()[key]) + 5)
    changed = binding.write_through()

    assert list(changed) == [key], f"touched more than one widget: {changed}"


def test_an_int_setting_stays_an_int(bound):
    """A cell edited in a table arrives as a string; the file must not."""
    grid, panel, binding = bound
    key = _a_numeric_key(panel, binding, want=int)

    _edit(grid, key, 17)
    binding.write_through()

    assert panel.collect()[key] == 17


def test_the_settings_the_table_does_not_cover_are_untouched(bound):
    """The grid edits a corner of the file, not the whole of it."""
    grid, panel, binding = bound
    before = panel.collect()
    key = _a_numeric_key(panel, binding)

    _edit(grid, key, float(before[key]) + 1)
    binding.write_through()
    after = panel.collect()

    for other, value in before.items():
        if other == key:
            continue
        assert after[other] == value, f"{other} moved"


# ---------------------------------------------------------------------------
# Widget -> grid
# ---------------------------------------------------------------------------

def test_seeding_shows_what_the_panel_now_holds(bound):
    """A settings file loaded behind the grid reaches the table."""
    grid, panel, binding = bound
    key = _a_numeric_key(panel, binding)
    panel.set_value_for_key(key, 99)

    binding.seed()

    assert grid.settings()[key] == 99


def test_seeding_twice_says_the_same_thing(bound):
    """Idempotent, so a screen may reseed whenever something wrote."""
    grid, panel, binding = bound

    binding.seed()
    once = grid.settings()
    binding.seed()

    assert grid.settings() == once


# ---------------------------------------------------------------------------
# The loop that would eat the cursor
# ---------------------------------------------------------------------------

def test_the_table_is_not_rebuilt_while_it_is_being_written(
        qtbot, qt_theme_applied, mask_settings):
    """Write-back changes widgets, and a screen reseeds when widgets change.

    Reseeding mid-write resets the model the user is typing into: the cell
    loses focus and the edit under the cursor is redrawn from the panel. No
    VALUE is lost -- the write works from a snapshot taken before the first
    widget moves -- but the table rebuilding under the hands is the whole
    reason the guard is there.
    """
    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    panel = ReseedingPanel(mask_settings)
    binding = ObjectGridBinding(grid, panel)
    panel.bind(binding)
    binding.seed()
    first, second = _two_numeric_keys(panel, binding)
    _edit(grid, first, float(panel.collect()[first]) + 2)
    _edit(grid, second, float(panel.collect()[second]) + 3)

    rebuilds = []
    original = grid.set_settings
    grid.set_settings = lambda values: (rebuilds.append(values),
                                        original(values))[1]
    binding._write_back()

    assert rebuilds == [], f"the table was rebuilt {len(rebuilds)}x mid-write"


def test_two_edits_both_survive_the_write(qtbot, qt_theme_applied,
                                          mask_settings):
    """A screen reseeds when its widgets change, and write-back changes them.

    So the honest wiring is a cycle: write a widget -> the widget emits ->
    the screen reseeds the table from the panel. Reseeding in the MIDDLE of a
    write-back reloads the table over the edits that have not been written
    yet, and the second cell the user changed is silently dropped. The guard
    is what holds the whole write together, so this edits two cells.
    """
    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    panel = ReseedingPanel(mask_settings)
    binding = ObjectGridBinding(grid, panel)
    panel.bind(binding)
    binding.seed()
    first, second = _two_numeric_keys(panel, binding)
    want_first = float(panel.collect()[first]) + 2
    want_second = float(panel.collect()[second]) + 3

    _edit(grid, first, want_first)
    _edit(grid, second, want_second)
    binding._write_back()

    assert float(panel.collect()[first]) == want_first, "first edit lost"
    assert float(panel.collect()[second]) == want_second, "second edit lost"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _a_numeric_key(panel, binding, want=None):
    """A per-object key holding a number, so a test can move it."""
    settings = panel.collect()
    for key in sorted(binding.owned_keys()):
        value = settings[key]
        if isinstance(value, bool):
            continue
        if want is int and not isinstance(value, int):
            continue
        if isinstance(value, (int, float)):
            return key
    pytest.skip("no numeric per-object setting to move")


def _two_numeric_keys(panel, binding):
    """Two distinct per-object keys holding numbers."""
    settings = panel.collect()
    found = []
    for key in sorted(binding.owned_keys()):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        found.append(key)
        if len(found) == 2:
            return found
    pytest.skip("need two numeric per-object settings")


def _edit(grid, key, value):
    """Write ``value`` into the cell for ``key``, through the model."""
    from spacr.object_settings_table import _split

    obj, question = _split(key)
    table = grid.table()
    table[question][obj] = value
    grid._model.set_table(table)
