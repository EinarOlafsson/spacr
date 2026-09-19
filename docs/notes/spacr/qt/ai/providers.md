# Notes from `spacr/qt/ai/providers.py`

Prose lifted out of `spacr/qt/ai/providers.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ChatProvider](#chatprovider) (5 entries)
- [ChatProvider.__init__](#chatprovider__init__) (1 entry)
- [Module level](#module-level) (1 entry)
- [_stream_process](#_stream_process) (2 entries)
- [ClaudeCliProvider](#claudecliprovider) (2 entries)
- [GeminiCliProvider.stream_chat](#geminicliproviderstream_chat) (1 entry)
- [_stream_process and ProviderFailed, 2026-09-19](#_stream_process-and-providerfailed-2026-09-19) (1 entry)

## ChatProvider

### line 47, trailing

```python
name: str = ""
```

short id: "claude" / "codex" / "gemini"

### line 48, trailing  _(unsure)_

```python
label: str = ""
```

human-readable label

### line 49, trailing  _(unsure)_

```python
cli_name: str = ""
```

the executable on PATH

### line 50, trailing  _(unsure)_

```python
install_hint: str = ""
```

shell one-liner to install

### line 51, trailing  _(unsure)_

```python
login_command: str = ""
```

shell one-liner the user should run

## ChatProvider.__init__

### lines 54-56

```python
"""Create the provider with no child process running.
```

Tracks the currently-running child process so cancel_stream() can actually terminate it — otherwise `for line in proc.stdout` blocks indefinitely and the worker thread never exits.

## Module level

### lines 119-120  _(unsure)_

```python
_NOISE_LINE_PREFIXES = (
```

Noise the vendor CLIs emit that we drop before showing to the user. Match on line prefix (case-sensitive).

## _stream_process

### line 240, trailing  _(unsure)_

```python
bufsize=1,
```

line-buffered

### lines 265-266  _(unsure)_

```python
try:
```

Always tear the child down cleanly — cancel_stream() may have already terminated it; ok to call terminate again defensively.

## ClaudeCliProvider

### lines 316-326

```python
install_hint = (
```

ONE COMMAND, WHOLE, PER PLATFORM. It was a single line carrying both forms joined by "# or", which pastes correctly -- the shell comments the rest away -- and copies badly: the maintainer took the curl half without `| bash` on 2026-09-10, and curl then printed the installer to the terminal instead of running it. Nothing failed and nothing installed, which is the worst shape a copied command can have.

CURL ON EVERY SYSTEM, WINDOWS INCLUDED (item 414, 2026-09-15). Windows used to get `npm install -g @anthropic-ai/claude-code`, because install.sh refuses Windows outright ("Windows is not supported by this script"). That form fails with "'npm' is not recognized" on any machine without Node.js, which is what the maintainer met on Windows. Windows now gets Anthropic's own CMD installer from https://code.claude.com/docs/en/setup (retrieved 2026-09-15), `curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`, run as `cmd /c "..."`. The wrapper is the point: pasted into PowerShell the bare command fails twice, because `curl` can be an alias for Invoke-WebRequest there and Windows PowerShell 5.1 rejects `&&`. Named as a `cmd /c` command, the one copied line runs whole from CMD or PowerShell. Windows 10 (1803+) and 11 ship curl.exe, and install.cmd ends with `exit /b`, so the trailing `del` still runs. macOS and Linux keep the documented `curl -fsSL https://claude.ai/install.sh | bash`. `tests/qt/test_claude_code_install_hint.py` pins both.

### login_command, 2026-09-19

```python
login_command = "claude auth login"
```

IT WAS `claude setup-token` FROM f3506ff13 (2026-07-21) UNTIL 2026-09-19, AND THAT COMMAND DOES NOT SIGN THE CLI IN. `claude setup-token --help` in Claude Code 2.1.274 says "Set up a long-lived authentication token". After the browser step it prints the token with "Store this token securely. You won't be able to see it again" and "export CLAUDE_CODE_OAUTH_TOKEN=<token>". That token is meant for CI and scripts, and `claude` stays signed out until the variable is set in its environment. A spaCR started as a macOS app would not inherit a variable exported in a terminal anyway. `claude auth --help` lists `login` as "Sign in to your Anthropic account". It is the command-line form of the `/login` that a signed-out `claude` asks for ("Not logged in · Please run /login"). The strings above come from `--help` and from the 2.1.274 binary. Neither command was taken through its browser step here, because that needs an Anthropic account.

Four places read this attribute, and all four were wrong in the same way: the `[AI error]` hint that `ProviderFailed` builds (GitHub #117), the Login row of the Providers dialog, first-run setup's "Sign in now" button (`setup_slides._prompt_to_set_up`, which runs the command in a terminal), and the manuscript writer's "run `...`" advice. `tests/qt/test_a_failed_run_offers_the_report_it_promised.py` pins the command. When a `claude` is installed, it also asks that binary whether `auth login` is the sign-in.

## GeminiCliProvider.stream_chat

### line 423  _(unsure)_

```python
args = ai_settings.provider_args(self.name)
```

SPEED_MAP uses --model; translate to -m for the gemini CLI

## _stream_process and ProviderFailed, 2026-09-19

```python
if (isinstance(exit_status, int) and exit_status != 0
```

A provider CLI that exits non-zero has failed, and the line it printed is its error message, not an answer. Before this, `_stream_process` never looked at the exit status, so the worker reported success with that line as the reply. GitHub #117 (jak18015, macOS, 1.5.0.8) is the case: `claude` printed `Failed to authenticate: OAuth session expired and could not be refreshed` and exited, and the console showed it as spaCR AI's answer, with no `[AI error]` and no hint to sign in again. The console kept it as the explanation of the crash, so `ai_explanation_of` handed it to the bug reporter, and issues #118 and #121 were filed with that line under "spaCR AI's analysis of this error".

Measured 2026-09-19 with Claude Code 2.1.274 in an empty HOME: `claude -p "say hi"` prints `Not logged in · Please run /login` and exits 1. The expired-session message in #117 needs an expired session, which could not be produced here, so its exit status is not measured. The check depends only on the status, never on the wording.

The lines are still streamed as they arrive, because a long answer has to show while it is written. The failure is raised after the last line, once the child has been reaped. The message quotes the last three non-blank lines, cut to 400 characters. When a provider is given, it also names that provider's `login_command`. The command is the one the Providers dialog already shows, so the two places agree. For Claude it is `claude auth login`. It was `claude setup-token` until review found that `setup-token` only mints a token for an environment variable (see `login_command` under ClaudeCliProvider).

A child that spaCR itself ended does not count as failing. Cancel and quitting through `terminate_all_streams` mark the Popen before they signal it. This function's own escalation marks it too, although on that path the reader never reads a status, because the wait that would return one is the wait that timed out. The status such a child exits with is spaCR's doing. On POSIX it is a negative signal number. On Windows it is 1, because `Popen.terminate` is `TerminateProcess(handle, 1)`, so it looks exactly like a CLI reporting a failure. The mark is the only way to tell the two apart.

## INSTALL_METHODS, CommandLineTool and GitHubCli, 2026-09-19 (item 420)

```python
INSTALL_METHODS: Dict[str, Dict[str, Tuple[InstallMethod, ...]]] = {
```

ONE DEFINITION FOR THE HINT AND THE BUTTON. Item 420 asked for an Install button that runs "the curl link" instead of printing it, and said the button and the hint must not drift apart. So each way of installing a tool is a row (`InstallMethod`: the programs it needs, the command as typed, and how it is started), `install_hint` is built from the rows, and `cli_install.plan_install` runs the first row whose programs are on `PATH`. The hints that existed before are unchanged on Linux and macOS, character for character (a test pins them). On Windows only the first row is shown: `cmd` has no `#` comment, so the old `npm install -g @openai/codex   # or brew install codex` handed `#`, `or`, `brew` ... to npm as package names if pasted there.

WINDOWS AND MACOS ARE UNVERIFIED (`UNVERIFIED_PLATFORMS`). Their rows come from the vendors' documentation: Anthropic's `install.cmd` (item 414), npm and Homebrew package names, `winget install --id GitHub.cli` and `conda install gh --channel conda-forge` from cli.github.com. None has been run on those systems by spaCR. On Linux the runner was exercised with real processes standing in for the installers (a shell script named `npm` or `conda`); nothing real was installed.

THE GITHUB CLI HAS NO ROOT-FREE OFFICIAL INSTALLER ON LINUX. Its documented Linux route is the distribution's package manager, which needs `sudo` and so a terminal. The rows spaCR can run without one are Homebrew and conda (`--yes`, because the child has no input and conda would otherwise stop at "Proceed ([y]/n)?"). With neither present the prompt says so and offers cli.github.com.

THE CONDA ROW NAMES SPACR'S OWN ENVIRONMENT (review, 2026-09-19). `conda install --yes gh --channel conda-forge` installs into whichever environment conda treats as active, with the plan never shown. spaCR started from a desktop icon has no `CONDA_PREFIX`, so that was `base`, and `cli_install.likely_folders` looks only in `sys.prefix` and `CONDA_PREFIX`: a successful install ended as "cannot find gh". When `sys.prefix` is a conda environment (it has `conda-meta`), `gh_conda_row` now writes `--prefix <sys.prefix>` into the command, so the prompt shows which environment changes, and `--freeze-installed`, so conda adds `gh` without updating the packages spaCR runs on (if it cannot, it fails with its own message instead). Checked with `conda install --help` on this machine that both options exist; the command itself was not run. Outside a conda environment the row is conda's plain command, as before. A Windows path with a space is written in double quotes, and `cli_install.split_command` splits Windows commands without treating the backslashes as escapes.

STATUS COMMANDS, MEASURED 2026-09-19. `claude auth status` exits 1 in an empty HOME ("Not logged in. Run claude auth login to authenticate.") and 0 signed in. `gh auth status` exits 1 with an empty `GH_CONFIG_DIR` and 0 signed in. `gh auth token` is what `github_auth` already uses and is what `GitHubCli.status_command` runs, with its output discarded because the output is the token. Beware when probing it: with a fake HOME and `GH_CONFIG_DIR`, `gh auth token` still answered 0 from the system keyring, so an empty config directory is not a signed-out `gh` for that subcommand. `codex login status` is taken from Codex's documentation and was not run here (no `codex` on this machine). Gemini has no status command, so its sign-in ends with the user pressing Done.

`ChatProvider.is_logged_in` is still "installed", not a status command. It is called on the GUI thread by the setup screen's marks and by `configured_providers`, and a process per call there would stall the screen. The status commands are asked only off the GUI thread, by the install panel.
