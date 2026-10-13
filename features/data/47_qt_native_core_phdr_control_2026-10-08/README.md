# Controlled Qt/scientific ELF identity probe, 2026-10-08

This is a **local control**, not a reproduction of the hosted coverage abort.
Its purpose is to check whether a real process importing the Qt 6.12 Make
Masks screen and scientific stack necessarily exceeds the native collector's
unchanged limit of 1,024 ELF program headers.

The process used Python 3.12.13, PySide6/Qt 6.12.0, CPU/offscreen mode, and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. It imported
`spacr.qt.screens.make_masks`, NumPy, SciPy, pandas, scikit-image, imageio,
and tifffile, then sent itself SIGUSR1 under GDB. GDB generated a core at that
stop; the process was not an app-test or native-crash reproduction. The source
commit was `79d8311908fe1081a0aceeefca9bd52eb668644b`.

The exact bounded launcher, run from the repository root, was:

```bash
CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay prlimit --fsize=2147483648 -- tools/run_capped.sh 4G timeout --signal=TERM --kill-after=5s 90s /usr/bin/gdb -nx -nh -batch -x /mnt/wd4tb/scratch/native-core-phdr-probe-20261008/gdb.commands --args /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/native-core-phdr-probe-20261008/probe.py
```

The process reported PID `3952166`, resolved executable
`/home/olafsson/anaconda3/envs/spacr/bin/python3.12`, and 934 mapped regions.
The 399,765,008-byte ELF core had 599 program headers, each 56 bytes.
`_core_process_ids` returned exactly `{3952166}` with the executable path
present and reason `accepted`; see the independent `readelf` header and
collector diagnostic. Full-core SHA-256:
`260fe07f4174587c4d0679711adb2e939054ac2b0fa07fb8d2a28ecf74105b5a`.

The core remains only in `/mnt/wd4tb/scratch/native-core-phdr-probe-20261008/`.
It is excluded from Git because a process memory image can contain sensitive
environment data. This real core did not hit the 1,024-header ceiling. The
hosted 657 MB core was deleted after the earlier collector rejected its
internal identity, so this control cannot identify that rejection's cause.
