Read-only GitHub CI follow-up, 2026-10-15

The latest retrieved Tests run was 38816815119, source
99789fd25c44fd94b488ce3fca63e6aa5303ad9c. GitHub reported the overall
run as QUEUED while its actual job list contained 19 in progress,
7 queued and 1 completed successfully (lint/types/complexity). There
were no failed jobs in this snapshot. This is not a final CI result.

Latest retrieved nightly source 8993ec4cc544b4f2c20e12a5836fc13770632416
had compat-matrix run 38823391128 queued (all nine jobs queued), and
docs run 38823390986 pending with no jobs. Older docs run 38822178568,
source 77e1c175b87ce18a3ff566778862ad5af0cd644a, was building docs;
its English manifest refresh and verification steps had passed.

The older failed docs run 38819608264 at b51fb194bb183990ab25c981d70e3030063bf9ac
passed documentation build, translated user guides, tooltip API-link checks
and branch artifact upload. Only subsequent Hugging Face publication failed:
the Space reported its 1 GB storage limit reached. This is an external
publication dependency, not evidence of a current source fix.

JSON files preserve retrieved gh JSON response values. latest-tests.json
and latest-runs.json contain the relevant rows selected from the retrieved
run-list responses; tests-jobs.json is the complete retrieved gh --jq job
projection. They were transcribed from already-returned tool outputs; no
further request was made while sealing this packet.

Timestamp limitation: original gh invocations did not print a wall-clock
retrieval timestamp. It is unavailable and is not inferred from job events.
GitHub createdAt and the quota log's event timestamps are retained verbatim.
Evidence sealing began at 2026-10-15T23:30:13Z. These statuses are snapshots,
not a claim about the scheduler at sealing or publication time.

Commands already executed:
  gh run list --workflow tests.yml --branch nightly --limit 3 --json databaseId,headSha,status,conclusion,createdAt,url
  gh run view 38816815119 --json databaseId,headSha,status,conclusion,jobs --jq '{databaseId,headSha,status,conclusion,jobs:[.jobs[]|{id:.databaseId,name,status,conclusion}]}'
  gh run list --branch nightly --limit 8 --json databaseId,headSha,status,conclusion,name,createdAt,url
  gh run list --branch nightly --limit 24 --json databaseId,headSha,status,conclusion,name,createdAt,url
  gh run view 38823391128 --json databaseId,headSha,status,conclusion,jobs,url
  gh run view 38823390986 --json databaseId,headSha,status,conclusion,jobs,url
  gh run view 38822178568 --json databaseId,headSha,status,conclusion,jobs,url
  gh run view 38819608264 --json headSha,status,conclusion,jobs,url
  gh workflow list --json id,name,path,state
  gh run view 38819608264 --job 116721148285 --log-failed | tail -n 65

No dispatch, cancellation, rerun, source/test/workflow edit, artifact download,
or current failed-job repair occurred. No handoff, item or tracker was changed.
