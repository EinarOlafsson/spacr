# spaCR tutorial release contract

## Current accepted matrix

- 69 lessons
- 487 scenes
- 8 narration languages
- 50 voices: 24 English and 26 localized
- 3,450 narration tracks
- 69 4K silent masters
- derived Pages media below 700 MiB
- cache key `20260813-69-lessons-final`

## Editable-workspace gates

Run from `/mnt/firecuda2/Claude/toxoplasma_projects/tutorials`:

```bash
python tools/verify_tutorial_release.py
python tools/verify_audio_release.py --workers 16 --freshness-languages all
pytest -q \
  tests/test_visual_master.py \
  tests/test_web_player_contract.py \
  tests/test_verify_audio_release.py \
  tests/test_pronunciation.py \
  tests/test_narration_audio.py
```

Audit content beyond structural tests. In particular, verify current Home and
API guidance; real Batch flow; Preview in Mask, Timelapse, Motility, Cellpose
Masks, and Plaque; bounded searches in Image UMAP, Activation, Classify CV, and
Classify ML; and Gate Editor 2D/3D/xD plus Walk.

## Repository gates

Run from the spaCR repository:

```bash
python tools/docs_media_budget.py --report
QT_QPA_PLATFORM=offscreen pytest -q \
  tests/test_docs_media_budget.py \
  tests/test_tutorial_link_and_install_recipe.py \
  tests/qt/test_tutorial_cli.py \
  tests/qt/test_help_menu_tutorial_link.py \
  tests/qt/test_tutorial_director.py \
  tests/qt/test_tutorial_scripts.py \
  tests/qt/test_tutorial_overlay_geometry.py \
  tests/qt/test_tutorial_engine.py
python tools/sample_tutorial_frames.py --all --output /tmp/spacr-tutorial-audit
```

Build Sphinx with `--keep-going`, but report its real exit status, warning
count, `index.html` existence, and HTML page count.

## Publication gates

1. Run `python tools/publish_tutorials.py --skip-hf` and require no unexpected
   changed media.
2. Run `python tools/publish_tutorials.py` and record the Hugging Face commit.
3. Commit the derived bundle explicitly and push nightly.
4. Merge through a clean main-based PR because Pages deploys from main only.
5. Require the standard docs/localization gate; never create a second bypass
   deployment path.
6. Run `python tools/verify_tutorial_live.py --browser` after deployment.

The live check must prove 69 lessons, 487 scenes, 8 languages, 50 voices, no
retired voices, exact committed asset hashes, caption and narration Blob URLs,
audible clock progress, native pause, narration-authoritative end, replay from
zero, mobile viewport/user agent, and no narration fallback toast.
