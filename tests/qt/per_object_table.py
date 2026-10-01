"""Read-backs for Mask generation's per-object table, shared by the Qt tests.

Item 592 (2026-09-28) made the per-object table the ONLY layout of the
per-object settings on Mask generation and Timelapse: every key the table
answers (``cell_diameter``, ``nucleus_channel``, ``organellec_model_name``,
``remove_background_cell``, ``adjust_cells`` ...) has its flat row hidden
behind the table, and the object is a COLUMN. A test that asked "is this
row on the form?" of such a key now has to ask the table instead. These
helpers answer that question from the built screen, not from the rule.
"""
from __future__ import annotations

from typing import Optional, Tuple


def grid_section(screen):
    """The section the per-object table sits in, or ``None``.

    :param screen: a built :class:`spacr.qt.screens.app_screen.AppScreen`.
    """
    grid = getattr(screen, "_object_grid", None)
    if grid is None:
        return None
    node = grid.parentWidget()
    while node is not None and not hasattr(node, "add_prose_row"):
        node = node.parentWidget()
    return node


def grid_shown(screen) -> bool:
    """Whether the per-object table's section is on the form.

    :param screen: a built module screen.
    """
    section = grid_section(screen)
    return section is not None and not section.isHidden()


def table_cell(key: str) -> Optional[Tuple[str, str]]:
    """``(question, object)`` of the table cell that answers ``key``.

    :param key: a settings key such as ``"nucleus_diameter"``.
    :returns: ``None`` for a key that is not a per-object question.
    """
    from spacr.object_settings_table import to_table

    for question, row in to_table({key: None}).items():
        for obj in row:
            return question, obj
    return None


def the_table_answers(screen, key: str) -> bool:
    """Whether ``key`` is a cell the per-object table shows.

    True when the table claims the key, its section is on the form, the
    object has a column, and that column asks the question.

    :param screen: a built module screen.
    :param key: the settings key.
    """
    binding = getattr(screen, "_object_grid_binding", None)
    if binding is None or key not in binding.owned_keys():
        return False
    if not grid_shown(screen):
        return False
    cell = table_cell(key)
    if cell is None:
        return False
    question, obj = cell
    grid = screen._object_grid
    return obj in grid.objects() and obj in grid.table().get(question, {})


def table_value(screen, key: str):
    """The value the per-object table shows for ``key``.

    :param screen: a built module screen.
    :param key: the settings key.
    """
    question, obj = table_cell(key)
    return screen._object_grid.table().get(question, {}).get(obj)
