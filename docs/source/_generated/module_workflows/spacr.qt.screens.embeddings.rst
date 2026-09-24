Workflow inputs and outputs
---------------------------

Embeddings
~~~~~~~~~~

Encode object images with a chosen model and channel policy. Retain object identities and encoder provenance when supplying the features to downstream exploration.

**Open:** Home → Embeddings.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Image embeddings** — Object-indexed encoder features, with channel policy and encoder provenance. The encoding API returns features; persistence is caller-dependent.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Retain encoder and channel-policy provenance.

**After this module**

* :ref:`Image UMAP <workflow-module-umap>`: Supply the encoder feature table with matching object IDs; do not assume every GUI route automatically persists it.

:doc:`API reference </api/spacr/qt/screens/embeddings/index>`.

`Module tutorial <../../../../../tutorials/#lesson=77_embeddings>`__.

