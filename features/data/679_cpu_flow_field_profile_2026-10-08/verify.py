"""Verify the immutable CPU frame-stage profile payloads."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = HERE.relative_to(ROOT)


def read(name, committed):
    if committed:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{REL / name}"], cwd=ROOT)
    return (HERE / name).read_bytes()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(committed=False):
    receipt = json.loads(read("receipt.json", committed))
    payloads = {}
    for name, expected in receipt["payloads"].items():
        packed = read(name, committed)
        assert len(packed) == expected["gzip_bytes"], name
        assert digest(packed) == expected["gzip_sha256"], name
        raw = gzip.decompress(packed)
        assert len(raw) == expected["raw_bytes"], name
        assert digest(raw) == expected["raw_sha256"], name
        payloads[name] = raw

    source = payloads["ambient.py.gz"]
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    assert blob == receipt["source"]["ambient_git_blob"]
    assert digest(source) == receipt["source"]["ambient_sha256"]

    script = payloads["stage_probe.py.gz"]
    assert digest(script) == receipt["probe_sha256"]
    assert b"engine.shade(width, height)" in script
    assert b"for _ in range(3):" in script
    assert b"for _ in range(5):" in script
    assert b"_PACKED_SCATTER" in script

    for result_name, log_name, themes, cases in (
        ("results.json.gz", "steady-final.log.gz",
         ("data_art_genetic_advection", "data_art_impulse_lens"), 12),
        ("spaceout.json.gz", "spaceout.log.gz",
         ("data_art_spaceout_field",), 3),
    ):
        result = json.loads(payloads[result_name])
        assert result["source_sha256"] == receipt["source"]["ambient_sha256"]
        assert result["pyside"] == "6.10.0"
        assert result["qt_platform"] == "offscreen"
        assert result["cuda_visible_devices"] == ""
        assert result["scatter_ready_before_cases"] is True
        assert len(result["rows"]) == cases
        assert {row["theme"] for row in result["rows"]} == set(themes)
        lines = payloads[log_name].decode().splitlines()
        assert len(lines) == cases
        for row, line in zip(result["rows"], lines):
            assert json.loads(line) == row
            assert row["samples"] == 5
            assert len(row["wall_ms"]) == len(row["cpu_ms"]) == 5
            assert row["packed_scatter_ready"] is True
            assert math.isclose(statistics.median(row["wall_ms"]),
                                row["median_wall_ms"], rel_tol=0, abs_tol=1e-9)
            for stage, values in row["stage_median_ms"].items():
                assert values is None or math.isfinite(values), stage
        assert result["final_memory_kib"]["VmHWM"] < 8 * 1024 * 1024
    print("verified: six compressed payloads, source blob, 15 warm CPU cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    verify(parser.parse_args().git)
