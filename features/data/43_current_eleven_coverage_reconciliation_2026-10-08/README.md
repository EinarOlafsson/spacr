# Current-source eleven-module numerical reconciliation

This small archive links two already frozen coverage proofs to source
`7f321bf3d9ca663ce9e76c16aaa9c32105fbd4e4`. It contains no copied
coverage payloads and does not change an allowance.

The original eleven-module proof is rerun at its exact target revision
`11229945a6e803f5fa1d5ed6683d446a2b068132`, where its eleven source
hashes and coordinate unions apply. Nine of those production files are
byte-identical at the current source. The remaining Preferences and ambient
files are checked by the later strict background/fungal proof, including the
two-line positive-cost guard. Its AppScreen check is additional to the eleven
modules. Both referenced manifests and payload hashes are verified from Git.

| Module | Missing statements / branches | Original limit |
| --- | ---: | ---: |
| deep_spacr | 0 / 0 | 0 / 0 |
| model_zoo | 0 / 0 | 4 / 1 |
| plaque | 0 / 0 | 0 / 0 |
| qt/mask_engine | 8 / 11 | 11 / 11 |
| qt/preferences | 84 / 25 | 89 / 25 |
| qt/screens/make_masks | 11 / 28 | 59 / 28 |
| qt/screens/plate_view | 1 / 1 | 1 / 1 |
| qt/widgets/ambient | 0 / 0 | 1 / 0 |
| submodules | 0 / 0 | 0 / 0 |
| torch_artifacts | 0 / 0 | 0 / 0 |
| validate | 2 / 2 | 2 / 4 |

Run `python features/data/43_current_eleven_coverage_reconciliation_2026-10-08/verify.py --git`
from a source-identical descendant. The command executes both original
strict verifiers and compares each current production blob with the matching
frozen receipt. This is local numerical reconciliation, not a passing hosted
aggregate or a complete GitHub acceptance result. The original 683 hosted
run failed, and a complete current-source hosted verdict remains open.
