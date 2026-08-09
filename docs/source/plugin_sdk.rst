spaCR plugin SDK
================

The plugin SDK lets a separately installed Python package add assays,
importers, analysis utilities, settings panels, model-zoo entries and report
sections without editing spaCR. Plugins use Python package entry points, so
spaCR discovers them at startup and keeps them available to both the Qt
application and ``spacr-run``.

The current SDK API is ``1.0``. Only the major version controls
compatibility: a plugin declaring API ``1.x`` works with spaCR's ``1.x`` SDK.
A malformed or failing plugin is isolated and reported by
``spacr-plugins doctor``; it cannot replace a built-in module or prevent spaCR
from starting.

Minimal plugin
--------------

Declare the entry point in the plugin package's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."spacr.plugins"]
   my_assays = "my_spacr_plugin:plugin"

Then expose a manifest:

.. code-block:: python

   from spacr.plugins import AppContribution, SpacrPlugin

   plugin = SpacrPlugin(
       name="My laboratory assays",
       version="0.1.0",
       api_version="1.0",
       apps=(
           AppContribution(
               key="organelle_contact",
               name="Organelle Contact",
               description="Measure contact sites from a processed plate.",
               kind="assay",
               section="results",
               stage="alpha",
               entrypoint="my_spacr_plugin.pipeline:run",
               defaults="my_spacr_plugin.pipeline:default_settings",
               categories={
                   "Input": ("src", "table"),
                   "Detection": ("distance_px", "min_area"),
                   "Output": ("save_figures",),
               },
               tooltips={
                   "src": "Processed spaCR plate folder.",
                   "distance_px": "Maximum membrane-to-membrane distance.",
               },
               labels={"distance_px": "Contact distance (px)"},
               docs_url="https://example.org/my-plugin/api/",
               aliases=("contacts",),
               validator="my_spacr_plugin.pipeline:validate",
               drop_handler="my_spacr_plugin.qt:PlateDropHandler",
               requires=("src — a processed spaCR plate",),
               writes=("<src>/measurements/contact_sites.csv",),
           ),
       ),
   )

The pipeline callable receives one settings dictionary. The defaults callable
must accept an optional settings dictionary and return a dictionary. A
validator returns :class:`spacr.validate.Problem` objects (or equivalent
mappings). All processing still runs through spaCR's worker, journal,
cancellation and reproducibility paths.

Settings and custom screens
---------------------------

The generic settings screen is preferred: declare ``categories``, ``tooltips``
and optional ``labels`` on :class:`spacr.plugins.AppContribution`. The UI then
provides the normal Run/Stop controls, console, progress reporting, API links,
drag-and-drop fallback and remote-submit action.

For a layout the generic screen cannot express, set ``screen_factory`` to a
``module:callable``. The callable is invoked as ``factory(app_key=...)`` and
must return a ``PySide6.QtWidgets.QWidget``. A custom ``drop_handler`` must be
a subclass of ``spacr.qt.dnd_handlers.DropHandler``. ``icon`` may name a
spaCR semantic icon or an absolute image path; otherwise the normal puzzle
piece fallback is used.

Model providers
---------------

Add a :class:`spacr.plugins.ModelProviderContribution`. Its zero-argument
callable returns an iterable of :class:`spacr.model_zoo.ModelEntry` objects or
the mappings accepted by spaCR's JSON model catalogue.

.. code-block:: python

   from spacr.plugins import ModelProviderContribution

   ModelProviderContribution(
       key="lab_models",
       provider="my_spacr_plugin.models:catalogue",
   )

Providers must be read-only during catalogue discovery. Downloads remain an
explicit Model Zoo action. A provider exception is shown by
``spacr-plugins doctor`` and built-in models remain available.

Report sections
---------------

A :class:`spacr.plugins.ReportSectionContribution` builder receives a
read-only :class:`spacr.plugins.ReportContext` and returns a
:class:`spacr.report.Section`:

.. code-block:: python

   from spacr.plugins import ReportSectionContribution
   from spacr.report import Section

   def build_contacts(context):
       return Section(
           key="organelle_contacts",
           title="Organelle contacts",
           body_html="<p>Contact-site results.</p>",
           text_lines=["Contact-site results."],
       )

   contact_report = ReportSectionContribution(
       key="organelle_contacts",
       title="Organelle contacts",
       builder="my_spacr_plugin.report:build_contacts",
       after="statistics",
   )

If the builder raises, the generated report contains a visible problem
chapter with the exception instead of silently omitting the section.

Translations
------------

``SpacrPlugin.translations`` maps spaCR language codes to English-source /
translated-text mappings. Supported codes are ``sv``, ``de``, ``es``,
``zh_CN``, ``pt``, ``hi``, ``ko``, ``is`` and ``fr``. Missing strings fall
back to English.

Development and diagnostics
---------------------------

During local development only, point ``SPACR_PLUGIN_MODULES`` at a
comma-separated list of ``module:attribute`` references. Set
``SPACR_DISABLE_PLUGINS=1`` to start spaCR without third-party plugins.

.. code-block:: bash

   SPACR_PLUGIN_MODULES=my_spacr_plugin:plugin spacr-plugins list
   spacr-plugins doctor
   spacr-plugins doctor --json

The public SDK lives in :mod:`spacr.plugins`; the diagnostics command lives in
:mod:`spacr.cli_plugins`.

API reference
-------------

.. automodule:: spacr.plugins
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
