CURATE PRACTICE: SYNTHETIC MASK AND TRACKS, NOT BIOLOGICAL CORRECTIONS

These are outputs of the synthetic Timelapse demonstration. The integer TIFF
is exactly label plane 2 of the recorded four-plane NPY; 16 original labels
are numbered 2 through 17 in a 256 x 256 image. tracks.csv contains 128 rows:
16 tracks over 8 frames. They are teaching data, not real organism tracks.

Work on fresh COPIES of mask.tif and tracks.csv. Keep the downloaded originals.
Open Make Masks > Curate from the nested tool strip. Browse to mask.tif.
Use TIFF here: the current picker offers NPY but that path failed to load.

For brush practice use radius 8 and New (label 18). Draw in empty space near
pixel x=64,y=64, Undo, then draw again. Your mouse position may differ slightly
from the recording, so the exact painted pixel count is not a universal target.
The practice disk is invented for learning controls, not a real missing cell.
Save mask writes back to the selected file and writes mask.tif.curation.json.

IMPORTANT CURRENT LIMIT: reopening a saved mask resets the in-memory edit log.
The next save replaces the sidecar with only the new session's edits. A message
that the mask was curated does NOT mean its complete history was restored.

Before reopening, explicitly preserve both saved files using the companion:

  python curate_checkpoint.py /path/to/working/mask.tif /path/to/NEW-checkpoint

The script copies the saved data and ledger exactly, checks both SHA-256 hashes,
and refuses an existing destination. This is a separate external backup, NOT
an application feature, history repair or automatic replay. Do not edit the
source while it copies. Keep each session's checkpoint with your own notes;
do not claim the latest sidecar is the full history. Opening a checkpoint file
in Curate can still lose its in-memory history; do not edit the backups.

Tracks: Browse to the working tracks.csv. Selecting tracks 2 and 3 and Join
is refused because both exist in the same frames. Select track 2, set frame 4,
and Split; its tail becomes track 18. Select 2 and 18 and Join to restore the
original rows. Save tracks writes the CSV plus its own curation sidecar.
Deleting track 3 without saving removes 8 rows only in memory. Reload the saved
CSV to discard that unsaved deletion: 128 rows and the saved split/join log return.

The comparison checks every coordinate, label, frame and track ID; it does not
prove that these practice operations would be scientifically correct on real
data. No GPU, AI provider or new segmentation is involved. See Timelapse and
Make Masks for upstream data generation and the API lesson for programmatic use.
