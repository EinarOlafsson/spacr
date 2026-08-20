API reference
=============

Start with the workflow you want to run. These pages contain the supported
pipeline entry points; implementation modules remain available in the
complete reference below.

Applications and workflow
-------------------------

Every tile links to the API page used by that application's in-product help.

.. include:: ../_generated/workflow_grid.rst

New to the Python interface? Follow the :doc:`../python_api` first.

Stability
---------

The pages above contain spaCR's stable workflow entry points. Advanced and
experimental functions state that status in their documentation. GUI widgets,
workers, names beginning with an underscore and modules described as internal
are implementation details rather than a compatibility promise.

Complete module reference
-------------------------

The complete reference is generated from the source so contributors can
inspect every documented module. Application code should prefer the curated
workflow entry points above.

.. toctree::
   :hidden:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

{% for page in pages|selectattr("is_top_level_object") %}
* :doc:`{{ page.id }} <{{ page.include_path }}>`
{% endfor %}
