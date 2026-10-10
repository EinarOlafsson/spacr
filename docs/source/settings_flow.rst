Where a setting goes
====================

Every setting spaCR reads, and the call paths that carry it from an entry
point to the functions that actually read it.

Asked for on 2026-09-08: *"in the API when a user clicks a settings they
should see the setting and the function it goes to and the function(s) that
the settings get passed along to. when they click the setting itself they
should get the tool tip text."* This page lists those call paths. When the
function that reads a setting has no published API anchor, the **API** link
in the setting's tooltip opens the setting's section on this page instead.

Each line of a branch is a call, indented under its caller. A step marked
**-- reads it** is where the value is used rather than passed on. Branches
that reach no reader are omitted. A step marked ``[UNRESOLVED]`` is a call
that static analysis cannot follow -- a ``getattr``, a dispatch table, a Qt
signal, a callback passed as a value. These steps are shown rather than
omitted, so the tree marks where the analysis is incomplete.

The page is generated from the source by ``tools/settings_flow.py``.
``tests/test_the_settings_flow_artefacts_are_current.py`` fails when the
generated page no longer matches a fresh run of the generator.

.. include:: _generated/settings_flow.rst
