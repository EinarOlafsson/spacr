Remote and distributed execution
================================

spaCR can submit any module exposed by ``spacr-run`` to another workstation,
a Slurm cluster, or a cloud/HPC command-line client. Jobs remain on the remote
system when the spaCR GUI closes. Their identifiers, status, settings hashes
and latest logs are stored locally and appear in **Data & batch → Distributed
Jobs**.

Execution profiles
------------------

An execution profile describes *how to reach compute*, not an analysis.
Profiles never store passwords or API keys:

``SSH workstation``
   Uploads the small resolved settings JSON over SSH, starts ``spacr-run`` in
   a durable background process and records its exit code.

``Slurm cluster``
   Uploads settings, submits an ``sbatch`` script, polls ``squeue`` and then
   ``sacct``, and cancels with ``scancel``. The SSH host may be blank when the
   Slurm commands are installed locally.

``Cloud / custom command``
   Runs configured submit/status/cancel argument templates. This supports
   cloud CLIs and site-specific schedulers without embedding vendor
   credentials in spaCR. The command must print a safe job identifier and the
   status command should print a conventional state such as ``PENDING``,
   ``RUNNING``, ``SUCCEEDED``, ``FAILED`` or ``CANCELLED``.

Configure SSH keys, cloud authentication and VPN access outside spaCR. Test
the same connection in a terminal before submitting a long run.

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

Command-line workflow
---------------------

The GUI and CLI share the same profile and job stores. Create a workstation
profile and submit a settings export::

   spacr-remote profile add gpu-box \
       --backend ssh \
       --host scientist@gpu-box \
       --workdir /shared/spacr \
       --local-root /mnt/lab \
       --remote-root /shared/lab

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
   :members:
   :undoc-members:
   :show-inheritance:
