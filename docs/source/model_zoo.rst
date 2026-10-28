Model zoo
=========

Select a registered model key, for example
``pathogen_model: toxoplasma_pv_v1``. A missing registered checkpoint can be
downloaded and checksum-verified when automatic acquisition is allowed.
``SPACR_MODEL_DOWNLOAD=off`` disables that automatic acquisition; compatible
local paths and verified cached checkpoints remain usable. A missing model
requested directly on the GUI thread is refused. Open **Model Zoo** from the
**Make Masks** masthead to browse and explicitly download models.

Downloads through :func:`spacr.model_zoo.fetch` require a published or
explicitly supplied SHA-256 by default. The digest checks the acquired bytes
against that expected identity; it does not validate a model's biological
performance. The explicit low-level Python override
``require_checksum=False`` permits acquisition without a published digest;
:func:`spacr.model_zoo.install` then reports ``verified=False`` when no expected
digest was supplied. This is not a verified installation.

.. include:: _generated/model_zoo_table.rst

Each model is hosted in a repository under its author's own Hugging Face
account; contributing a model does not require write access to another
account's repository.
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
