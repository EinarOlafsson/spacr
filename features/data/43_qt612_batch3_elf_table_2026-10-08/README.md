# Controlled ELF header-table proof, 2026-10-08

This is a local **collection-only control**, not a reproduction of the hosted
Coverage0 SIGABRT. It uses the exact 32 files in batch 3 of Coverage0 at
`7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab`. The file manifest is
SHA-256 `d5f347bdb25b692cf207f18526b333d01e58a952a923842881367ad77ff65826`;
it was reconstructed with the unchanged shard/batch functions and matched to
the archived hosted batch log. Pytest collected 503 items and executed none.

The isolated 7a worktree used Python 3.12.13, a local PySide6/Qt 6.12.0
overlay, pytest 8.4.2, CPU/offscreen mode, and OMP/MKL/OPENBLAS each set to
one. A scratch pytest plugin stopped the owned process with SIGUSR1 at
`pytest_collection_finish`; GDB wrote a core at that stop. The launcher ran
under `tools/run_capped.sh 4G`, a 240-second timeout and a 2 GiB file limit.
No application test body ran.

The bounded launch from the isolated 7a worktree was:

```bash
mapfile -t batch < /mnt/wd4tb/scratch/native-qt612-7a-collection-20261008/batch3.txt
CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_ADDOPTS='' \
  PYTHONPATH="/mnt/wd4tb/scratch/native-qt612-7a-collection-20261008:/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay:$PWD" \
  prlimit --fsize=2147483648 -- tools/run_capped.sh 4G \
  timeout --signal=TERM --kill-after=5s 240s /usr/bin/gdb -nx -nh -batch \
  -x /mnt/wd4tb/scratch/native-qt612-7a-collection-20261008/gdb.commands \
  --args /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest \
  "${batch[@]}" --collect-only -q -m 'not gui' -p collection_probe \
  --cov=spacr --cov-branch --cov-report=
```

The process had 1,627 mapped regions. Its 1,407,100,200-byte ELF core has
1,091 program headers of 56 bytes each: a 61,096-byte header table. The old
collector rejected the core at its **count** check before inspecting identity.
With the new header **byte** bound, the same core yields exactly one process
leader PID (`3969483`) and the expected executable path. The new bound is
262,144 bytes, equal to the largest header table the former 1,024-count and
256-byte-stride limits could read. The 4 MiB aggregate note budget, strict
stride, exact PID/executable identity, core size, extraction time and GDB
limits are unchanged.

The full core stays only in scratch because process memory may contain
credentials. Its SHA-256 is
`6a14b079b63fc96159134d16e87ea73341ce48a5965ef67f6c5eaec2baf6b0d7`.
The compact receipt, exact file manifest, original ELF header and collection
facts are in this directory. The old hosted 657 MB core was deleted after a
nonspecific identity refusal; its actual rejection reason remains unknown.
