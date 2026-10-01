Model zoo
=========

spaCR ships a catalogue of trained models and fetches them on demand. Name a
key in a settings file — ``pathogen_model: toxoplasma_pv_v1`` — and the model
is downloaded and checksum-verified the first time it is needed, or open
**Model Zoo** from the **Make Masks** masthead to browse and install them.

Every published entry carries a SHA-256. An entry without one is refused
rather than installed, because a truncated or substituted checkpoint cannot
be told from the real one.

.. include:: _generated/model_zoo_table.rst

Models are hosted on their author's own Hugging Face account, so contributing
one does not mean handing write access to anyone else's.
``spacr.model_zoo``'s ``publish_model`` performs the upload and prints the
catalogue row to add.

Choose a model from a form
--------------------------

A model setting in Mask generation, Make Masks and Plaque Assay has a
**Model zoo…** button. Its list is grouped by origin under five headings —
the stock Cellpose-SAM weights, spaCR's own models, unvetted spaCR community
uploads, Cellpose models published on bioimage.io, and the Cellpose 3
backend's cyto, cyto2, cyto3 and nuclei models. Click a heading to show or
fold its rows. **Download** fetches the selected model, or **Install** its
backend; **Use this model** then writes the model into the setting. A model that needs another backend, such as
Cellpose 3 or Cellpose-DINO, says so on its row and offers to install that
backend into its own environment under ``~/.spacr/backends``; spaCR's own
environment is not changed. A row spaCR cannot run says so instead of being
offered.

Opened from Mask generation, the popup also has **Measure diameters…**: it
estimates the cell, nucleus and pathogen diameters from a few fields of your
own images and can write them into the settings. See
:class:`spacr.qt.prerun.DiameterDialog`.

Per-model detail
----------------

.. include:: _generated/model_zoo_sections.rst
