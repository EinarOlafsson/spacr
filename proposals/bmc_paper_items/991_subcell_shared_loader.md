# Item (number assigned by master): one shared SubCell loader in spacr.embeddings

- **Requested:** maintainer sign-off of `reanalysis/experiment_designs/20261119_subcell_phenotype_arm.md`, decision D7 (2026-11-19): "file the spaCR shared-loader item now".
- **Rule behind it:** when a transformer model is used on our immunofluorescence (IF) or optical pooled screen (OPS) images, it is SubCell (maintainer, 2026-11-19). DINO stays a labelled natural-image baseline only.
- **Ships as alpha.** The GUI picker is unchanged: the new backbones are reachable from the API and settings files only, until a GUI step follows.

## Problem
spaCR had two partial SubCell loaders:
- `spacr.embeddings._subcell_model`:
  - backbones `subcell` (bg ViT-ProtS-Pool) and `subcell_rybg`;
  - `torch.load(weights_only=False)` on an unchecked download;
  - no rbg or ybg model, so it did not fit Funk (DNA, tubulin, γH2AX, phalloidin) or PERISCOPE (DNA, ER, Mito, phalloidin, WGA);
  - no MAE-CellS-ProtS-Pool, the paper's main model;
  - no rescale to the Human Protein Atlas pixel size, which the authors apply to every outside dataset;
  - no multi-pass protein channels, no layer read-out and no random-weights control.
- A script copy, `reanalysis/plateA_spatial_net/scripts/net.py`, used for the 2026-11-17 card.

## Change
1. **Weights.** `_SUBCELL_WEIGHTS` registry of all 8 published encoders (bytes, sha256 when pinned).
   - Pinned now: MT-DNA-Protein MAE-CellS-ProtS-Pool (db26dad7…4911) and MT-DNA-Protein ViT-ProtS-Pool (8d1e299f…8812).
   - The others are pinned when downloaded (ybg at the PERISCOPE step, card D1).
   - `_subcell_state` reads the hub copy or `EmbeddingSpec.checkpoint_path`, checks the digest (the pinned one, or `checkpoint_sha256` when none is pinned), loads with `weights_only=True`, and refuses anything that is not a mapping of names to tensors.
2. **Three-plane backbones.** `subcell_rbg` (MT-DNA-Protein) and `subcell_ybg` (ER-DNA-Protein), MAE-CellS-ProtS-Pool, in `_SUBCELL_VARIANTS`.
   - `channels` = two references in model order, then one or more protein planes.
   - Each protein plane is one pass `(ref0, ref1, protein)`, and the passes' 1,536-d embeddings are concatenated.
   - This is the authors' JUMP convention.
3. **Pixel size.** New `EmbeddingSpec.pixel_size_um`. Crops are resized by `pixel_size_um / 0.0800885` (bilinear), each side rounded to whole 16-px patches. Funk 96 px at 0.2125 µm/px becomes 256 px.
4. **Normalisation.**
   - `normalize=False`: the authors' whole-pass min–max.
   - `normalize=True`: the fixed plate × channel scale, clipped, so intensity stays comparable between crops.
5. **Layer read-out.** `Pooled.forward(x, layers=...)` returns, for each layer, the class token and the mean patch token through the final LayerNorm. `_subcell_passes(..., layers=...)` returns them per pass.
6. **Random-weights twin.** `_subcell_model(..., init="random", seed=...)`: the same architecture untrained, with nothing downloaded. It is the control for every SubCell result, as R1 is for DINO.
7. **Provenance.** `encoder_entry` and `_weights_on_disk` cover the new backbones and their local `checkpoint_path`. The fingerprint includes the pixel size and the weights digest.
8. **Retired:** `plateA_spatial_net/scripts/net.py` gets a header note saying it is superseded. The SubCell arm's identity check I5 shows the shared loader reproduces its embeddings. No other script carries a SubCell copy.

## Users (one loader, no copies)
- **OPS phenotype embeddings:** `EmbeddingSpec(backbone="subcell_rbg", channel_policy="project", channels=(GFP, DAPI, A594, AF750), pixel_size_um=0.2125, normalize=False)`.
- **The SubCell phenotyping arm** (card 20261119): its scripts import `spacr.embeddings` (`_subcell_model`, `_subcell_passes`).
- **The transformer arm of the cell-cycle classifier** (item being specced in this folder): the same encoder and transform, with a classifier head on the pooled or block embeddings.

## Tests
- `tests/test_subcell_three_plane_loader.py`:
  - pass plane order and the authors' min–max;
  - fixed scale;
  - pixel-size resize to 256;
  - pinned-digest mismatch, explicit digest, missing file and non-tensor state refused;
  - spec refusals (too few, repeated or per-channel planes; bad pixel size);
  - the real-ViT random twin is seed-stable, and the layer read-out has width 768.
- The existing foundation-loader tests pass unchanged.
- Ratchets: raw-reader, nested-function, documented-callable and hash-comment.

## 8-step workflow position
- Steps 1–4 (script, small iteration, test, dev slide) are the card's probe: identity checks I4–I8 and the 640 canvas probe.
- This item is steps 5–6 (spaCR implementation and test).
- Step 8 (production on source data) is the card's extraction, run through this loader once it is on nightly.
