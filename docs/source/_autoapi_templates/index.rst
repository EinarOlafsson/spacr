API reference
=============

Start with the workflow you want to run. These pages contain the supported
pipeline entry points; implementation modules remain available in the
complete reference below.

New to the Python interface? Follow the :doc:`../python_api` first.

Navigator and desktop launch
----------------------------

See the :doc:`Navigator guide <../navigator>` for local evidence, workflow
exploration and native source handoff. The existing public interface is
:func:`spacr.qt.run`, with :func:`spacr.qt.spaceout.main` for process-local
Spaceout launch. :func:`spacr.qt.preferences.get_easy_mode` and
:func:`spacr.qt.preferences.set_easy_mode` read and persist the Navigator
preference; the historical function names remain unchanged. These functions
do not submit AI requests or run an analysis. Launch Navigator with
``spacr-nav`` or ``spacr-navigator``; ``spaceout-nav`` and
``spaceout-navigator`` select the Spaceout presentation for that process.

Applications and workflow
-------------------------

Every tile links to the API page used by that application's in-product help.
The bands below are the categories the home screen groups its applications
into, in the same order and with the same members.

.. include:: ../_generated/workflow_grid.rst

.. include:: ../_generated/folded_modules.rst

Stability
---------

The pages above contain spaCR's stable workflow entry points. Advanced and
experimental functions state that status in their documentation. GUI widgets,
workers, names beginning with an underscore and modules described as internal
are implementation details rather than a compatibility promise.

Complete module reference
-------------------------

The complete reference is generated from the source so contributors can
inspect every documented module. It is an alphabetical list of Python
modules rather than of the applications above, so a reader who knows a
module by the name it carries in the product should start from a tile:
Mask is ``spacr.core`` and Recruitment is a function inside
``spacr.submodules``, and neither is findable here by that name.
Application code should prefer the curated workflow entry points above.

.. toctree::
   :hidden:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. dropdown:: Every documented module, alphabetically
   :name: complete-module-reference

   {% for page in pages|selectattr("is_top_level_object") %}
   * :doc:`{{ page.id }} <{{ page.include_path }}>`
   {% endfor %}
