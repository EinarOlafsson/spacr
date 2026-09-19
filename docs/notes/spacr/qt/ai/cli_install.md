# Notes from `spacr/qt/ai/cli_install.py`

Written with the module on 2026-09-19 (item 420). The module carries no comments, so its reasons live here and in its docstrings.

## launch_command

```python
return [shutil.which("bash") or "bash", "-o", "pipefail", "-c",
```

`curl -fsSL https://claude.ai/install.sh | bash` EXITS 0 WHEN THE DOWNLOAD FAILS. `bash` reading an empty pipe runs nothing and succeeds, so without `pipefail` a machine with no network would report a finished install. With it, the pipeline's status is curl's (6 for an unresolved host, 22 for an HTTP error under `--fail`), which is what `classify` reads.

The Windows row is handed to `Popen` as the shown line itself, `cmd /c "curl ... && install.cmd && del install.cmd"`, so what runs is character for character what the prompt showed. Not run on Windows (see `providers.UNVERIFIED_PLATFORMS`).

## run_install

```python
proc = _spawn(command, stdin=subprocess.DEVNULL,
```

NO INPUT, A SCRATCH FOLDER, A GROUP OF ITS OWN. A child started from a GUI has no terminal, so a prompt would wait forever; with `stdin` at `/dev/null` it reads end-of-file and fails, which is a message instead of a hang. The Windows command downloads `install.cmd` into its working folder, so the working folder is a temporary one, removed afterwards. `start_new_session` makes the installer a process-group leader so Cancel reaches the `curl` and the `bash` both; `tests/test_an_installer_runs_the_command_it_shows.py` starts a real `sleep` under a real `bash` and checks the `sleep` is gone after Cancel.

Output is read with `read1` and split on `\r` as well as `\n`, because npm and curl redraw progress on one line with carriage returns and a line-based read shows nothing until the end.

## _watch

A THREAD OF ITS OWN, because the thread running the install is blocked in `read1` and cannot notice Cancel or the time limit. It polls every 0.2 s, which is also the longest Cancel takes to act.

## locate and put_on_path

Claude's native installer puts `claude` in `~/.local/bin`. A spaCR started from a desktop icon often has a `PATH` without that folder, so the install would succeed and the mark would still say "not installed" until a restart. `locate` looks in the folders installers use, and `put_on_path` adds the one it found to this process's `PATH` -- on the GUI thread, because changing the environment while another thread reads it is not safe.
