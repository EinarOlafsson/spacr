Cellpose fail-fast double complete-signature repair, 2026-10-08

Corrected 4133 Fast1 really failed the strict signature sweep: the previous
axis-aware double still had **kwargs. Declare all real parameters/defaults
without changing the axis sentinel/check, original refusal assertion or
hard failure upon any unexpected model call. No production/guard edits.
Four exact focused checks pass on actual installed Cellpose 4.2.1.1;
separate native metadata comparison proves names/order/defaults, with the
intentional axis sentinel substitution. Full failed hosted raw log is
preserved in scratch; its complete byte count/SHA and exact excerpt are
bound here, with whole-phase archival assigned separately. No hosted
green or hardware acceptance is claimed. Initial probe path error retained.
