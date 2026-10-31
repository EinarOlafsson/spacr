Remote and distributed execution
================================

spaCR can submit any module exposed by ``spacr-run`` to another workstation,
a Slurm, PBS or LSF cluster, or a cloud/HPC command-line client. Jobs remain
on the remote system when the spaCR GUI closes. Their identifiers, status, settings hashes
and latest logs are stored locally and appear in **Help → Distributed Jobs**.

Execution profiles
------------------

An execution profile describes *how to reach compute*, not an analysis.
Connection options and command/script templates are saved in the local profile
JSON. Keep passwords and API keys out of those templates: their text is stored
as entered. Built-in SSH sign-in can optionally remember the login password in
the operating-system keychain, not in the profile; one-time codes are not saved.
The execution backends are:

``SSH workstation``
   Uploads the resolved settings JSON over SSH, starts ``spacr-run`` as a
   detached background process (output redirected to a log, stdin closed) so
   it continues if the SSH connection drops, and records its exit code.

``Slurm cluster``
   Uploads settings, submits an ``sbatch`` script, polls ``squeue`` and then
   ``sacct``, and cancels with ``scancel``. The SSH host may be blank when the
   Slurm commands are installed locally.

``PBS cluster``
   Uploads settings and submits a job script with ``qsub``, polls ``qstat``
   and cancels with ``qdel``. Choose ``--backend pbs`` for a CLI profile.

``LSF cluster``
   Uploads settings and submits a job script with ``bsub``, polls ``bjobs``
   and cancels with ``bkill``. Choose ``--backend lsf`` for a CLI profile.

``Cloud / custom command``
   Runs configured submit/status/cancel argument templates. This supports
   cloud CLIs and site-specific schedulers without embedding vendor
   credentials in spaCR. The command must print a job identifier containing only letters, digits
   and ``_.:+-``, and the
   status command should print a conventional state such as ``PENDING``,
   ``RUNNING``, ``SUCCEEDED``, ``FAILED`` or ``CANCELLED``.

Configure SSH keys, cloud CLI authentication and VPN access outside spaCR.
Built-in interactive SSH sign-in can instead prompt for a password and second
factor in spaCR. Check the selected sign-in route before submitting a long run.

Shared data and path mapping
----------------------------

spaCR transfers the settings document, not an image dataset. Images must
already be visible to the execution target through shared storage, a mirrored
mount or a cloud-aware wrapper command.

When the mount paths differ, set a *Local dataset root* and *Remote dataset
root*. Every absolute path nested in the settings below the local root is
rewritten beneath the remote root. Paths outside it are preserved. For
example::

   local root:   /mnt/microscopy
   remote root:  /cluster/projects/microscopy
   local src:    /mnt/microscopy/experiment-7/plate-A
   remote src:   /cluster/projects/microscopy/experiment-7/plate-A

Settings can be dragged onto the Distributed Jobs screen, selected with
**Browse**, or handed off directly with **Submit remote…** in any ordinary
module.

.. _settings-files-from-the-gui:

Settings files from the GUI
---------------------------

Every module screen with **Import settings…** has **Export settings…** next
to it. It saves the values currently in the form, exactly what **Run** would
use, to a CSV you choose. Nothing runs: the source folder is not listed or
opened, so a form set up for a slow network share is saved at once. The
dialog suggests ``<module>_settings.csv`` in the source folder.

The file has the layout spaCR writes into ``<src>/settings/`` at the start
of every run: a ``Key,Value`` header and one row per setting, with lists in
Python form (``[0, 1, 2]``), ``True``/``False`` and empty cells for unset
values. Paths are saved exactly as the form shows them. A ``.json`` copy
with the same settings is written beside it. **Import settings…** reads the
file back to the same form values, and the command line runs it unchanged::

   spacr-run measure --settings measure_settings.csv --dry-run
   spacr-run measure --settings measure_settings.csv

From Python, ``spacr.cli.load_settings_file`` reads it and
``spacr.cli.write_settings_file`` writes one. When the cluster mounts the
data under another path, change ``src`` with
``--set src=/cluster/projects/plate-A`` or use the dataset roots of an
execution profile.

After **Submit remote…**, **Save settings CSV…** on the Distributed Jobs
screen saves the snapshot it received as the same file. Submitting is
unchanged.

Model checkpoints on the execution target
-----------------------------------------

Exported settings keep the model path and any recorded zoo key, publisher version and SHA-256. Existing pinned files must match that checksum. Missing files resolve only to the exact recorded identity, or an exact zoo key/filename when unpinned; no fuzzy substitution is made. Models are cached below the spaCR home directory (normally ``~/.spacr/models``), unless ``SPACR_MODEL_CACHE`` selects another directory. Processes share a download lock and verify cached checkpoints before downloading. On an offline node set ``SPACR_MODEL_DOWNLOAD=off`` before ``spacr-run``: model metadata and weights are not downloaded. Existing local files and verified cached models remain usable; an uncached known model reports the checksum and cache path to copy. Unknown requests fail clearly. Custom paths do not gain a trusted catalogue identity from their filenames.

Jobs and nodes that share one model cache download a missing model once. The first job holds a lock file in the cache and refreshes it while it downloads; the others wait, then reuse the verified file. A lock that a crashed node left behind is removed after ``SPACR_MODEL_LOCK_STALE_SECONDS`` without a refresh (600 by default). When ``SPACR_MODEL_DOWNLOAD=off`` stops a run, the error ends with a shell command that downloads the checkpoint to the cache path and checks its SHA-256. Run it on a machine with network access that sees the same folder, or run it elsewhere and copy the file to that path.

Command-line workflow
---------------------

The GUI and CLI share the same profile and job stores. Create a workstation
profile and submit a settings export. The execution target must already
have spaCR and the analysis dependencies installed; submitting does not
install them. Set ``--runner`` to its ``spacr-run`` executable, or use
``--env-setup`` for the shell commands that activate the target environment.
For example, a conda profile can use
``--env-setup 'eval "$(conda shell.bash hook)"; conda activate spacr'``.
A job that cannot find its runner exits with code 127; changing the local
GUI environment does not change the target environment.

Create a profile::

   spacr-remote profile add gpu-box \
       --backend ssh \
       --host scientist@gpu-box \
       --workdir /shared/spacr \
       --local-root /mnt/lab \
       --remote-root /shared/lab

Inspect the generated job script before submitting. The command prints it
without submitting a job::

   spacr-remote script mask --profile gpu-box

Submit and monitor the job::

   spacr-remote submit mask \
       --settings mask-settings.csv \
       --profile gpu-box

   spacr-remote list --refresh
   spacr-remote status JOB_ID --logs
   spacr-remote watch JOB_ID --logs
   spacr-remote cancel JOB_ID

A Slurm profile can be local or reached through a login host::

   spacr-remote profile add lab-slurm \
       --backend slurm \
       --host scientist@login.cluster \
       --workdir /project/lab \
       --runner /project/lab/env/bin/spacr-run \
       --scheduler-options "--partition=gpu --gres=gpu:1 --time=12:00:00"

Before submitting through the cluster profile, inspect its resources::

   spacr-remote resources --profile lab-slurm

This reads the cluster configuration and reports resource problems; it does
not request an allocation. Use ``spacr-remote watch JOB_ID --follow`` to stream
new log output while watching; ``--logs`` prints the log tail when watching
finishes.

Custom/cloud templates
----------------------

Templates are parsed as argument vectors; spaCR does not execute them with
``shell=True``. Supported placeholders are:

``{job_id}``
   The permanent local spaCR identifier.

``{module}``
   The canonical ``spacr-run`` module name.

``{settings}``
   The absolute local resolved-settings JSON. A cloud wrapper is responsible
   for uploading it if needed.

``{external_id}``
   The identifier printed by the submit command, used by status/cancel/log
   commands.

``{profile}``
   The profile display name.

If a cloud CLI prints JSON, set a Job-ID regular expression with a named
``id`` group. For example ``"jobId":\s*"(?P<id>[A-Za-z0-9-]+)"``.

Reliability and limitations
---------------------------

* A temporary polling failure is retained as a visible error and does not
  falsely mark a remote job failed.
* SSH jobs write an exit-code file atomically. Slurm uses accounting after a
  job leaves the queue.
* Local job records and profiles use advisory locks plus atomic replacement,
  so a GUI and ``spacr-remote`` process do not partially write JSON.
* Cancellation is a request. A scheduler or remote process may take time to
  stop, and pipeline-level checkpoint semantics still determine how much work
  can be resumed.
* Cloud templates intentionally do not provide an embedded shell. Put
  pipelines, uploads, quoting and vendor-specific JSON handling in a reviewed
  wrapper executable.

Python API
----------

.. automodule:: spacr.remote_execution
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
