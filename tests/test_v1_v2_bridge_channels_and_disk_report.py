"""Two readers of :mod:`spacr._v1_v2_bridge` that get values in an unexpected shape.

The channel list is hand-edited, so one unparseable entry costs that entry
and a list of nothing usable costs the list, not the run.  The disk report is
a log line at the end of a finished run, so a stack deleted in between or a
sidecar on a share that will not answer costs that file's bytes, never the
report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from spacr._v1_v2_bridge import report_disk_savings, v2_channels_from_settings

DEFAULT = ([0, 1, 2, 3], ["ch0", "ch1", "ch2", "ch3"])


def test_v2_channels_from_settings_skips_what_does_not_parse(subtests):
    """Named assignments win in fixed order; ``channels`` keeps original positions; nothing usable is the default."""
    cases = [
        ("named_assignments_in_fixed_order",
         {"cell_channel": 2, "nucleus_channel": "0", "pathogen_channel": None,
          "organelle_channel": "x", "channels": [5]},
         ([0, 2], ["nucleus", "cell"])),
        ("one_word_in_the_channel_list", {"channels": ["red", 2]}, ([2], ["ch1"])),
        ("a_channel_tuple", {"channels": (3, "4")}, ([3, 4], ["ch0", "ch1"])),
        ("only_words", {"channels": ["red", "green"]}, DEFAULT),
        ("a_channel_string_is_not_a_list", {"channels": "0,1"}, DEFAULT),
        ("nothing_at_all", {}, DEFAULT),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert v2_channels_from_settings(settings) == expected


def _report(src, stacks, caplog):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="spacr.pipeline_v2.bridge"):
        report = report_disk_savings(src, stacks)
    return report, [r.getMessage() for r in caplog.records
                    if r.name == "spacr.pipeline_v2.bridge"]


def test_the_disk_report_counts_only_files_it_can_still_read(tmp_path, monkeypatch, caplog, subtests):
    """Missing stacks, path-less entries and stale sidecars add nothing; the totals and log line are exact."""
    with subtests.test(msg="a_deleted_stack_and_a_pathless_entry"):
        src = tmp_path / "plate"
        (src / "merged").mkdir(parents=True)
        stack = src / "merged" / "plate1.npy"
        stack.write_bytes(b"\x00" * 4096)
        stacks = [SimpleNamespace(path=str(stack)),
                  SimpleNamespace(path=str(src / "merged" / "deleted.npy")),
                  object()]
        report, logged = _report(src, stacks, caplog)
        assert report == {"v2_bytes": 4096, "v1_estimated_bytes": 16384,
                          "saved_bytes": 12288, "saved_pct": 75.0}
        assert logged == ["v2 pipeline finished — used 4.10 KB. Estimated v1 disk "
                          "use for the same run: 16.38 KB. Saved: 12.29 KB (75.0%)."]

    with subtests.test(msg="sidecars_count_and_a_stale_one_is_skipped"):
        src = tmp_path / "share"
        (src / "merged").mkdir(parents=True)
        stack = src / "plate1.npy"
        stack.write_bytes(b"\x00" * 2048)
        (src / "filename_map.csv").write_text("a,b\n", encoding="utf-8")
        (src / "filename_map.json").write_bytes(b"{}" * 10)
        (src / "merged" / "channel_order.json").write_bytes(b"[]" * 5)
        report, _ = _report(src, [SimpleNamespace(path=str(stack))], caplog)
        assert report["v2_bytes"] == 2048 + 4 + 20 + 10

        real_exists = Path.exists

        def stale(self, *args, **kwargs):
            if self.name == "filename_map.csv":
                raise OSError(116, "Stale file handle")
            return real_exists(self, *args, **kwargs)

        monkeypatch.setattr(Path, "exists", stale)
        report, _ = _report(src, [SimpleNamespace(path=str(stack))], caplog)
        assert report == {"v2_bytes": 2078, "v1_estimated_bytes": 8312,
                          "saved_bytes": 6234, "saved_pct": 75.0}

    with subtests.test(msg="nothing_readable_is_zero_bytes"):
        monkeypatch.undo()
        report, logged = _report(tmp_path / "empty", [], caplog)
        assert report == {"v2_bytes": 0, "v1_estimated_bytes": 0,
                          "saved_bytes": 0, "saved_pct": 0.0}
        assert logged == ["v2 pipeline finished — used 0 B. Estimated v1 disk use "
                          "for the same run: 0 B. Saved: 0 B (0.0%)."]
