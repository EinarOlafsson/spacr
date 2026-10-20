# Proposed N703 identity, preservation and rollback receipt contract

The proposal makes eight acceptance paths concrete: requested/effective version,
actual imported code and runtime, native GUI process/window readiness, preservation,
selected cleanup, retirement and rollback evidence. Installer exit 0, an isolated
version probe or an accepted launch request do not establish GUI success.

The refined proposal distinguishes upgrade, same-version repair, fresh install,
retaining a verified newer runtime and restoring the previous runtime after
failure. Successful rollback is recovery; it does not accept the new release.
A venv interpreter may legitimately resolve outside its prefix, so executable,
prefix and import bindings are checked together.

Private path/process/data inventories remain private. Public receipts permit
bounded normalized versions and roles, and do not publish raw user paths,
settings, credentials or individual user-data hashes. Challenges correlate an
attempt; they are not cryptographic attestation against a compromised app.

Original and corrected proposals are preserved. Bounded semantic review and
final consistency cover the stated outcomes and privacy refinements; mechanical
checks bind eight flow IDs and 12 source/test files. Root rechecked those files.
The added state graph remains proposed. No schema was approved or executed;
no installer, native GUI, preservation, rollback, tests or source changes ran.
Home owns final implementation after the release hold.
