Model zoo
=========

spaCR ships a catalogue of trained models and fetches them on demand. Name a
key in a settings file — ``pathogen_model: toxoplasma_pv_v1`` — and the model
is downloaded and checksum-verified the first time it is needed, or open
**Model Zoo** from the home screen to browse and install them.

Every published entry carries a SHA-256. An entry without one is refused
rather than installed, because a truncated or substituted checkpoint cannot
be told from the real one.

.. include:: _generated/model_zoo_table.rst

Models are hosted on their author's own Hugging Face account, so contributing
one does not mean handing write access to anyone else's.
``spacr.model_zoo``'s ``publish_model`` performs the upload and prints the
catalogue row to add.

Per-model detail
----------------

.. include:: _generated/model_zoo_sections.rst
