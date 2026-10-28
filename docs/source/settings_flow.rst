Where a setting goes
====================

This page lists settings and call paths identified by spaCR's static source
analysis, with their published tooltip text. It is not a runtime trace or a
guarantee that every dynamic setting access was found. When the function that
reads a listed setting has no published API anchor, the **API** link in its
tooltip opens the setting's section on this page instead.

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
