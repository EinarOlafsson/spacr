# Security Policy

## Supported versions

spaCR is developed on `nightly` and released from `main`. Only the latest
released version receives security fixes; there are no maintained older
branches.

| version | supported |
|---|---|
| latest release on [PyPI](https://pypi.org/project/spacr/) / [conda-forge](https://anaconda.org/conda-forge/spacr) | yes |
| anything older | no — please upgrade |

Check what you are running with `python -c "import spacr; print(spacr.__version__)"`.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately, either through
[GitHub's private vulnerability reporting](https://github.com/EinarOlafsson/spacr/security/advisories/new)
if it is enabled on this repository, or by email to **einar.olafsson@gmail.com**
with `spaCR security` in the subject line.

Please include, as far as you can:

* what an attacker can achieve, not only what is technically wrong;
* the version of spaCR and the platform;
* a reproduction — a minimal file, dataset or sequence of steps;
* whether you have published anything about it already.

**What to expect.** This is a small academic project maintained by one person,
so please calibrate accordingly: acknowledgement within about a week, an
assessment of whether it is exploitable and how badly within about two, and a
fix in the next release once there is one. If a report needs a coordinated
disclosure date, say so and it will be agreed with you rather than imposed.
Credit is given in the release notes unless you prefer otherwise.

## What spaCR actually does, so a report can be aimed properly

spaCR is a desktop and command-line scientific application, not a network
service. It has no server, listens on no port, and stores no credentials for
anybody but the person running it. The parts with a genuine security surface
are these, and they are the interesting places to look:

* **It opens local SQLite measurement databases** and reads paths recorded
  inside them, including image paths written by an earlier run on a different
  machine.
* **It downloads model weights** from Hugging Face and other model hosts, and
  loads them. A model file is executable content in the general case.
* **It can hold a GitHub token.** The auto-issue feature can file a crash
  report directly, using either a personal access token you paste or the
  GitHub CLI's own credentials.
* **It auto-files crash reports** containing a traceback and pipeline
  settings, with filesystem paths redacted to `<PATH>` and `<DB>`. A leak of
  something not redacted is in scope, and worth reporting.
* **It loads third-party code through the plugin SDK**, which is a documented
  extension point. Code you install deliberately is not a vulnerability; a
  path by which spaCR loads a plugin you did *not* install is.
* **It writes into project folders you choose**, and a path-traversal that
  escapes a chosen destination is in scope.

## Out of scope

* Crashes on malformed local input where the only person affected is the one
  who supplied the file. Please still report those as ordinary bugs — they are
  wanted, just not security reports.
* Vulnerabilities in dependencies, unless spaCR's own use of the dependency is
  what makes it exploitable. Report those upstream; tell us if we should pin
  or bound a version in the meantime.
* Anything requiring an attacker to already have the ability to run code as
  the user running spaCR.
* The absence of a hardening measure with no demonstrated impact.

## For users handling sensitive data

spaCR processes microscopy images and measurement tables locally. Nothing is
transmitted anywhere unless you ask for it — the two exceptions are model
downloads and, if you opt in, auto-filed issue reports. If you work with data
that must not leave your institution, leave issue auto-filing off and fetch
model weights on a machine where that is permitted.
