# Complete numerical coverage failure on source 683

Run `37698599898` measured all 664 shipped modules with complete integrity:
12 of 12 coverage shards supplied process data, no lost-worker batch needed
recovery, and no module rise was unconfirmed. The unchanged numerical ratchet
failed 11 modules; the required-shards gate also failed because selected tests
failed. This is neither a green run nor an incomplete measurement.

The full aggregate job log names every module, old allowance, current missing
statement and branch. The full Coverage2 log adds only the already owned
callable/i18n inventory and Swedish `Animation detail` glossary failures. The
aggregate report ZIP was published as GitHub artifact `11519648871`, but the
Azure blob download route failed twice during this archive; no ZIP bytes are
claimed here.

The focused local coverage JSON is a seven-case, CPU-only validation of the
four Home-owned gaps in `deep_spacr.py`, `model_zoo.py`, `torch_artifacts.py`,
and `validate.py`. It hits every exact line and arc missing for those four in
the hosted report. It does not combine with hosted process data or establish
the remaining seven modules' verdict. A real no-improvement resume bug was
found and fixed: a catalogue checkpoint key previously returned a fake path
under the checkout and wrote its card there; the resolved checkpoint path is
now selected. Commit `ebfdb47132b` contains the source and focused tests.
