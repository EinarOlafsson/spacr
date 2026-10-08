# 5636 required coverage phase

Source `5636eac60938f5d6021eaf73546ff8a30ae17c32`, run `37763414112`, attempt 1. Eleven of twelve selected-test shards succeeded. Coverage shard 0 failed after xdist gw1 segfaulted in the parent-source load test; serial coverage recovery passed but does not turn that required test into success. The numerical ratchet independently passed for all 664 shipped modules, with one recovered signal loss and no numerical regression or unconfirmed module. The original report and native evidence ZIPs, complete raw logs (deterministic gzip), job records, and phase snapshots are preserved here. The systemd core was extracted on the runner but refused by the old ELF identity parser, leaving no native C++ stack. This archive records failure; it does not claim CI green or native cause.

`python verify.py` checks filesystem payloads; `python verify.py --git` checks committed Git bytes and source bindings.
