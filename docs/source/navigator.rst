Navigator: choose a tool from your input
========================================

Navigator is the default Home for a fresh profile; a saved opt-out is respected. Launch it with ``spacr-nav`` or ``spacr-navigator``, or enable **Navigator** in **Help**. Explicit module launches still open that module.

Find a starting point
---------------------

Drop a local file or folder, browse for one, or type a path or task and press Enter. Typing alone does not inspect data or open a module. For example, “detect features” offers Measure; tool names and supported task phrases match without case sensitivity. Recognized commands, such as opening tutorials, can follow their own command route on Enter.

Suggestions appear as a grid of named cards. The first click anchors the selected card and reveals a sparse workflow; it does not run analysis. Hover to read the fixed, scrollable description. A second click on the selected tool opens its native screen. Review its settings and input before pressing Run.

Explore without running
------------------------

Select a downstream card to reveal its branches. **More next steps** exposes additional direct choices; **All tools** includes the other available registered tools and subtools. Availability follows maturity and species preferences. A connector describes a workflow relationship, not proof that the current input satisfies every prerequisite. Modes such as 3D, Time and OPS label connections rather than inventing separate processing steps.

Use **Back** or Escape to return to suggestions. Returning Home starts a fresh Navigator visit. The original card stays anchored while branches grow; the description remains below, and the app dock and side panes can be collapsed without changing the input.

Check what the input supports
-----------------------------

Local discovery uses bounded filenames, headers and project metadata, not a full analysis. An explicit raw image stays scoped to that input; a verified generated array inside a canonical merged folder can inherit its plate context. Handoffs use the selected file or project root required by the destination. An annotation-only database is not evidence of measured features, and a format keyword is not a promise that every file with that suffix can be read. Missing readers or calibrated-data prerequisites still need attention in the native workflow.

An offered filename-pattern editor requires explicit validation before its pattern is passed onward. Navigator does not silently convert data, run models or complete scientific prerequisite review. Optional AI assistance sends explicitly submitted unmatched questions only when enabled with a chosen provider; local tool and path matches retain their normal routes.

Desktop entry points and API reference
--------------------------------------

The launchers use :func:`spacr.qt.run`. Registered screen metadata is described by :class:`spacr.qt.app_catalog.DeclaredApp` and retrieved with :func:`spacr.qt.app_catalog.declared_app`; :func:`spacr.qt.app_catalog.register_declared` registers an existing declaration. Resolving :func:`spacr.qt.app.registered_factory` can import its screen. These are desktop registry contracts, not a headless Navigator inference API.

For scripted processing, use the supported entry points in :doc:`python_api`. For the files and prerequisites of each scientific step, see :doc:`workflows`.
