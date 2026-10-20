N328 / general CI: stale README checkout-size claim

Hosted 5727: 4,609,990,455 tracked bytes (4396.4295 MiB), exactly
reproducing the existing hosted failure against the 3416 MB claim.
Current 4108: 4,736,742,564 bytes (4517.3097 MiB), 24.38% drift.

This is a document owner request, not a proposed tolerance or test change.
Home-B should update the dated canonical estimate and nine exact reviews,
then use the normal README image/text generator. Old download/time data
was not remeasured and must stay separately dated. No owner file changed.

Reproduce without materializing 4GB or downloading:
python features/data/ci_readme_checkout_size_owner_request_2026-10-15/reproduce_blob_budget.py 5727b5354de46a2ac24f434afb2e0cb4671028ec 4108d3f99dbe326837aa45a222a1b023609ca81b
