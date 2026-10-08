N673 native 3D Make Masks CPU checkpoint

The committed model and real Qt editor use N672's shared physical boundary for
rendering, voxel containment, edits, undo and persistence. Ordinary 2D and YOLO
paths remain separate and passed the selected legacy regression files.

Evidence is explicitly phased: 116 tests at ecf, followed by 13 final GUI tests
and three branch-traced close tests at 2720. These overlap; do not sum them.
The final close follow-up protects completed unsaved edits while another edit
is busy. Cancel preserves the native dialog/worker/session; explicit Discard
cancels the pending operation. All actual regression files are frozen here.

The two 1300x900 stills are actual offscreen Qt captures. Capture saved/reopened
exact anisotropic ZYX labels and geometry, preserved the source bytes and
ended with a destroyed native dialog, idle worker and zero top-level widgets.
There is no GPU, aesthetics, native4K FPS, whole-suite or measured-RSS claim.

The receipt lists explicit unsupported topology/axis/size cases, publication
race limitations, structural memory bounds and remaining changed-source
coverage gaps. Full normal API/UI regeneration belongs to the workstation;
the included +11 API/+41 UI/+10 objectname delta is scoped to two files only.

Run python verify.py --git after integration to verify immutable payloads,
frozen source/test bytes and current Git source/test bindings. --frozen verifies
only the historical checkpoint if current app bytes have legitimately changed.
Original agent commit objects are not required by the verifier.
