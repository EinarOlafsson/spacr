## What this changes

<!-- What the change does, and why. If it fixes an issue, write "Fixes #123". -->

## How it was verified

<!--
The command you ran and what it printed. "Tests pass" is not a measurement.
If this touches something a user clicks, say that you opened the application
and pressed it — a green Qt suite does not prove a control is reachable.
-->

```
```

## Checklist

- [ ] Targets `nightly`, not `main`
- [ ] Tests added or updated, and the relevant suite passes locally
      (`xvfb-run -a python -m pytest tests/qt/` for anything Qt)
- [ ] No generated file was hand-edited — see the table in
      [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] If a public module was added, it is in `spacr/__init__.py::_SUBMODULES`
      and `tools/build_documentation_i18n.py --sources-only` has been run
- [ ] New user-facing strings go through `tr()` and are not typed into a
      translation catalog by hand
- [ ] No AI attribution in commit messages (no `Co-Authored-By` naming an
      assistant, no "Generated with …" line)

## Anything deliberately left undone

<!--
Optional, and genuinely useful: something you considered and rejected, with
the reason. A decision not to act is a result — without the reason, the next
person re-derives it or "fixes" something that was intentional.
-->
