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

## file_without_review

### added 2026-09-19 (autofile-default, follow-up to 432 and issue #117)

```python
def file_without_review(report: Dict[str, str]) -> Dict[str, str]:
```

The maintainer decided on 2026-09-19: "Do real auto-filing, and make this the default, and add the user agreeing to this in the user agreement, if set to always." From 807ba9e0a (instruction 45, 2026-08-14) until then, nothing read 'always', so it behaved exactly like 'ask'.

It never opens a browser. The browser form is the fallback `submit_report` keeps for 'ask'. It is a prompt: the user has to press Submit on GitHub. 'always' is the choice not to be prompted, so a profile with no GitHub sign-in files nothing and the console says so. The alternative was a browser window opening over spaCR on every new crash. That was the pre-August behaviour, and it was not what "real auto-filing" asked for.

`_post_report` is shared by both paths, so the open-issue search by fingerprint, the "Seen again" comment and the labels are the same whichever way a report is filed.

## public_report

### added 2026-09-19

```python
def public_report(report: Dict[str, str]) -> Dict[str, str]:
```

The instruction was "the SAME REDACTION the manual path uses". The manual path redacts in two places. `build_report` does the home folder, `.db` paths and credentials. The preview then applies `strip_report_paths`, because its "Remove file and folder names" switch starts on. So what a reviewer sends by pressing Send without editing is the stripped body. `tests/qt/test_the_report_files_itself.py` checks that the automatic body equals it character for character. The title is stripped as well, which the preview does not do, because nobody reads it before it goes.

## redact_identity

### added 2026-09-19

```python
def redact_identity(s: str) -> str:
```

Checked on 2026-09-19 against what the manual path redacted: home folders yes (to `~`, then `<PATH>` in the preview); other absolute paths yes (`<PATH>`); `.db` paths yes (`<DB>`); login names only inside the home path; host names never. A login name also appears as another user folder (`/srv/lab/<name>/`, which `<PATH>` covers), as a value in a setting, or in an error message. A host name appears in network paths and connection errors. `platform.platform()` carries no host name, so the Environment section was already clean.

The login name is taken from `getpass.getuser()` and the name of the home folder. The host name is taken from `socket.gethostname()` and `platform.node()`, in full and as its first dotted label. None of these touches the network. Names under three characters, and generic ones like `root`, `runner` and `localhost`, are left alone: they identify nobody, and redacting `root` would take the word out of every traceback that mentions a root folder. On a machine whose login name is an ordinary word the report loses that word everywhere. That is the accepted cost.

It sits inside `sanitize_path`, so the manual preview gets it too, and so does the log copy saved beside a report.

## strip_report_paths

### changed 2026-09-19

```python
value = re.sub(r"(?<![\w~<])(?:[A-Za-z]:[\\/]|/)[^\s'\"`]+", "<PATH>",
```

Issue #121 shows `<details><summary>spaCR AI's analysis of this error<<PATH>` and `<<PATH>`. The path pattern took the `/` of `</summary>` and `</details>` for the start of an absolute path. Every collapsible section after the first stayed open, so the environment, the settings and the log note all rendered inside the AI analysis. A `/` straight after `<` is now never a path.

## _frame_key

### added 2026-09-19

```python
where = "/".join(parts[-2:]) if parts else match.group("path")
```

The fingerprint hashed each frame's whole path after `~` substitution. So the same crash had a different fingerprint on every machine whose conda environment, Python minor version, install prefix or path separator differed. The open-issue search could therefore only find a duplicate from the same computer, and `find_issue_by_fingerprint` was cross-machine dedupe in name only. Two components (`spacr/core.py`) are enough to tell modules apart, and the whole stack plus the exception type is hashed, so two crashes that share a module name in one frame still differ.

This changes every fingerprint whose frames had more than two components, which is all real ones. An issue opened before 2026-09-19 will not be found by a client after it. The search covers open issues only, and those are few.

### changed again 2026-09-19, after review (a path can contain a space)

```python
_BARE_PATH_RE = re.compile(
```

The rule stopped a path at the first whitespace, so any path with a space in it was only partly replaced and the rest was published. Measured on the branch before this change, with `HOME` in a scratch folder:

| in | out |
|---|---|
| `src = 'C:\Users\anna\OneDrive - Karolinska Institutet\Screens\Patient 042'` | `src = '<PATH> - Karolinska Institutet\Screens\Patient 042'` |
| `src = '~/My Data/TB screen 2026/plate 1'` | `src = '<PATH> Data/TB screen 2026/plate 1'` |
| `Permission denied: '/mnt/lab share/Smith lab/raw'` | `'<PATH> share/Smith lab/raw'` |
| `\\LAB-NAS\screens\plate 1` | untouched: nothing here started at `\\` |

Spaced paths are the normal case on the two operating systems most users are on: `/Volumes/<name with spaces>` on macOS, `OneDrive - <institution>` and `Program Files` on Windows, and any share named after a lab. With 'always' there is no preview between the crash and the public tracker, and Section 5.5 of the terms accepted in this same change tells the user that file and folder names are replaced.

Three rules now, in order. A traceback's `File "..."` field, whatever is inside it. A quoted value that STARTS with a path root, taken to its closing quote or to the end of the line -- everything that quotes a path quotes the whole of it (`repr` of a settings value, the file name in an `OSError`, the backticked log path), and the end-of-line branch is for the title, which `build_report` cuts to 80 characters and can therefore hand over an opening quote whose closing one was cut off. Then unquoted paths, where a component may contain spaces PROVIDED a separator follows it: only the separator proves the space is inside the path rather than after it.

What that deliberately leaves is the tail of the LAST component of an unquoted path -- `\\LAB-NAS\screens\plate 1` still ends `<PATH> 1`. Taking it would mean eating the sentence after every path, which turns `opening /mnt/data/x.tif failed` into `<PATH>` and makes the report useless. Quoted paths have no such limit, and the title and every settings value are quoted.

The quoted rule requires the root directly after the quote, so an apostrophe in prose cannot pair with a later one and swallow the words between them.

### changed 2026-09-19, after review (a machine named after a word)

```python
_GENERIC_NAMES = frozenset({
```

`redact_identity` replaces any login or host name of three characters or more wherever it is not surrounded by `[A-Za-z0-9]`, so `_` and `-` are boundaries. On a machine called `gpu` that rewrote `use_gpu = True` in the settings block to `use_<HOST> = True`, and since `sanitize_path` runs before `_traceback_hash`, a substitution on a non-indented line moved the fingerprint on that machine alone -- the opposite of what `_frame_key` was added to fix.

Widening the boundary to include `_` was the wrong fix: it would leave a real login name in `/data/<name>_backup`. The word list is the right one. A host called `gpu`, `server`, `desktop` or `nas` identifies nobody, which is the same reason `root` and `localhost` are already there.

Two neighbours fixed with it: a name that is all digits is dropped (an IP-shaped host contributes the label `192`, which would otherwise be replaced wherever it stood, `line 192, in run` included), and a name that is both the login and the host name -- a workstation named after its user -- is listed once as the login name instead of twice with the host entry winning.
