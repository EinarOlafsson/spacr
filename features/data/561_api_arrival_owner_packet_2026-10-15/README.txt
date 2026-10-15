UNAPPLIED exact-source owner proposal

Original CI owns tests/test_api_i18n_extractor.py. No authoritative test or
release-contract file was modified by this packet. The patch targets immutable
public20a; verify manifest preimage or review any owner-local drift first.

After owner acceptance, copy the included JSON to
tests/data/release_contracts/561_classifier_api_arrivals_2026-10-15.json, then
apply extractor-test-owner.patch in the owner private tree. Use git apply
--check before git apply. If the owner test changed, reconcile these two named
additions normally, preserving every newer owner assertion.

The first hunk validates exactly19 source-bound helper bodies and disjoint
identities before subtracting them as named arrivals. The second admits only
the new normal-rendered local Classifier.__init__. Neither the historical
13218 floor nor the211 excluded-symbol boundary changes.

The actual API23/all10 catalogs and Help19 arrivals are separately generated
by the normal tools. This packet does not claim full extractor acceptance.
