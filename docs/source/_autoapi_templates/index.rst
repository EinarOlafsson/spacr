API reference
=============

Start with the workflow you want to run. These pages contain the supported
pipeline entry points; implementation modules remain available in the
complete reference below.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Typed workflow API
      :link: spacr/api/index
      :link-type: doc

      Configure and run Mask or Measure from a lightweight interface.

   .. grid-item-card:: Mask and image preparation
      :link: spacr/core/index
      :link-type: doc

      Segment microscopy images and write mask stacks.

   .. grid-item-card:: Measure and crop
      :link: spacr/measure/index
      :link-type: doc

      Extract object features, build SQLite projects and save crops.

   .. grid-item-card:: Classification
      :link: spacr/deep_spacr/index
      :link-type: doc

      Train and apply image classifiers.

   .. grid-item-card:: Barcode mapping
      :link: spacr/sequencing/index
      :link-type: doc

      Decode FASTQ reads and connect guides to wells.

   .. grid-item-card:: Regression and hits
      :link: spacr/ml/index
      :link-type: doc

      Estimate guide and gene effects and inspect diagnostics.

   .. grid-item-card:: Projects and provenance
      :link: spacr/artifacts/index
      :link-type: doc

      Inspect runs, settings, outputs and reproducibility records.

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
