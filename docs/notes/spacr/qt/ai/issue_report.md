# Notes from `spacr/qt/ai/issue_report.py`

Prose lifted out of `spacr/qt/ai/issue_report.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [sanitize_path](#sanitize_path) (1 entry)
- [_traceback_hash](#_traceback_hash) (1 entry)
- [build_report](#build_report) (5 entries)
- [issue_url](#issue_url) (3 entries)
- [submit_report](#submit_report) (4 entries)

## Module level

### line 37, trailing

```python
MAX_URL_LEN = 7500
```

GitHub caps the pre-filled issue URL at ~8 KB

## sanitize_path

### line 113  _(unsure)_

```python
s = re.sub(r"[/\\][^\s'\"]+\.db\b", "<DB>", s)
```

Redact any `.db` path suffix even if not under $HOME

## _traceback_hash

### line 194  _(unsure)_

```python
lines.append(stripped.split(":", 1)[0])
```

"ValueError: channels must be a list" -> "ValueError"

## build_report

### line 337  _(unsure)_

```python
err_line = ""
```

First non-empty error-type-looking line for the title

### lines 361-371

```python
analysis = sanitize_path(str(ai_response or "")).strip()
```

AFTER THE TRACEBACK, BEFORE THE ENVIRONMENT. When the AI is on it has usually already diagnosed the crash by the time the user files, and that analysis is the most useful thing in the report after the traceback itself -- it is what a reader would otherwise spend the first hour reproducing. It goes below the traceback because `issue_url` trims the tail, and the traceback must survive that trim.

MARKED AS MACHINE-GENERATED, and folded shut. It is a lead, not a finding: the analysis in the session this was written for was right about the cause and wrong about the fix, in a way that would have changed behaviour silently for every run that left a field blank.

### lines 405-414

```python
saved = save_log_bundle(tb_hash)
```

THE LOG DOES NOT GO IN THE ISSUE. An issue on the public tracker is world-readable and permanent, and a log line carries whatever the run happened to be about -- a gene name, a plate barcode, a collaborator's folder, the name of an unpublished screen. None of that is credential-shaped, so no redaction pass catches it, and the person filing the bug has no way to know it is there.

So the log is written BESIDE the report instead: a file on the user's own disk, whose path the issue names. The maintainer can ask for it, and the user decides then, having read it.

### lines 423-425

```python
body_parts.append(f"It was saved on the reporter's machine at "
```

Through the same sanitiser as everything else: the bundle lives under the user's home, and the home path carries their account name.

### lines 433-435

```python
return {"title": title, "body": "\n".join(body_parts),
```

`fingerprint` is returned, not just embedded in the body, so the caller can look for an existing issue carrying it before opening a new one. Without that the hash was written and never read.

## issue_url

### line 458  _(unsure)_

```python
scaffold_len = (
```

Reserve room for the fixed URL scaffolding + title

### lines 464-474

```python
note = (
```

Trim body — keep the traceback (most valuable), drop subsequent details blocks.

Measured against the ENCODED length, not the raw one. This used to slice `body[:head_len]` with head_len computed from the URL budget in raw characters, which is a different unit: quoting expands, and a traceback is mostly newlines at three characters each (`%0A`). A realistic crash report came out at 11,924 characters against a 7,500 limit AFTER "truncation", and GitHub answers an over-long issues/new with a page that reads "page not found" -- which is the 404 users were getting.

### lines 480-483

```python
head = body
```

Shrink until the QUOTED body fits. Halving converges in a few passes for any expansion ratio, where a fixed guess cannot: the ratio is 1x for plain ASCII and 3x for newline-dense text, and the body that matters most here is the newline-dense one.

## submit_report

### lines 510-512

```python
try:
```

If the user is signed in to GitHub (stored token / env / gh CLI), create the issue directly via the API — no browser needed. Otherwise fall back to opening the pre-filled issues/new URL in the browser.

### lines 515-519

```python
refusal = github_auth._transport_refusal()
```

This check uses the module instance resolved NOW.  A broad batch once left this module holding a different instance from the one a test had patched; its process-wide allow flag then sent four real comments to issue #114.  A real transport is refused before credential discovery, while an explicitly substituted offline seam can exercise the flow.

### lines 524-536

```python
searched, existing = github_auth.find_issue_by_fingerprint(
```

DEDUPE BY FINGERPRINT FIRST. `_traceback_hash` exists so the same bug hashes the same across runs and machines, and nothing consumed it: one crash produced one issue per occurrence -- ten in a single day on 2026-08-11 (#79-#81, #84-#90), which buries the reports that matter.

A hit gets a COMMENT rather than a new issue, because the second occurrence is still information: it says the bug is reproducible and carries that run's environment.

`searched` is distinguished from "found nothing" deliberately. If the search could not run we still file, because losing a crash report is worse than filing a duplicate.

### line 550, trailing

```python
return result
```

the created issue's html_url
