## Model zoo

spaCR ships a catalogue of trained models and fetches them on demand. Open <strong>Model Zoo</strong> from the home screen to browse and install them, or name a key in a settings file -- ` pathogen_model: toxoplasma_pv_v1 ` -- and the model is downloaded and checksum-verified the first time it is needed. Every published entry carries a SHA-256; an entry without one is refused rather than installed, because a truncated or substituted checkpoint cannot be told from the real one.
