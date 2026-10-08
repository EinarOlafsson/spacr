"""Replay only the thirteen capped CPU boundary/protocol cases."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--python", default=sys.executable)
parser.add_argument("--minimum", help="Directory containing the extracted official Cellpose4.0.7 wheel")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
nodes = [
    "tests/test_plaque_segmentation_diagnostics.py::test_prediction_retains_two_flow_axes_across_cellpose_return_contracts",
    "tests/test_plaque_segmentation_diagnostics.py::test_unmeasurable_vectors_and_logits_never_invent_alignment",
    "tests/test_plaque_segmentation_diagnostics.py::test_incompatible_flow_and_metric_returns_refuse_before_model_evaluation",
    "tests/test_plaque_segmentation_diagnostics.py::test_unreadable_saved_diagnostics_keep_mask_analysis_but_clear_metrics",
    "tests/test_mask_engine_recovery_contracts.py::test_noninteger_hole_fill_is_refused_without_mutating_pixels",
    "tests/test_cov_8_plate_view.py::test_live_plate_rejects_nonmapping_ledger_without_overwriting_committed_state",
    "tests/test_cov_8_plate_view.py::test_finishing_inactive_live_plate_stops_timer_without_reading_old_source",
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SPACR_DEVICE="cpu", QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
if args.minimum:
    env["PYTHONPATH"] = str(Path(args.minimum).resolve())
    nodes = ["tests/test_plaque_segmentation_diagnostics.py"]
raise SystemExit(subprocess.call([str(root / "tools/run_capped.sh"), "4G", args.python, "-m", "pytest", "-q", "-p", "no:randomly", *nodes], cwd=root, env=env))
