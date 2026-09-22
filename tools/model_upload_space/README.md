---
title: spaCR model upload
sdk: gradio
app_file: app.py
pinned: false
---

The upload endpoint behind spaCR's Model Zoo **Add** button. It holds the
Hugging Face token so contributors do not need an account of their own, and
holds every submission in `staging/` for review. See the module docstring in
`app.py` for how to deploy it and why it is built this way.
