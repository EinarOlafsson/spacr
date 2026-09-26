---
title: spaCR model upload
sdk: gradio
app_file: app.py
pinned: false
---

The upload endpoint behind spaCR's Model Zoo **Add** button and its
**Contribute training data** buttons. It holds the Hugging Face token so
contributors do not need an account of their own.

- `upload`: a model, held in `staging/` of `einarolafsson/user-models` for
  review.
- `contribute`: a community training-data folder (images, masks or YOLO
  labels, meta, `contribution.json`) as one tar. It is checked (known
  target, size and file caps, file types, every image paired with its mask
  or labels) and opened as a **pull request** on
  `einarolafsson/community_<name>` or Plaque Assay's two plaque datasets;
  the maintainer reviews and merges. See `contribution.py`.

See the module docstring in `app.py` for how to deploy it and why it is
built this way.
