# Repairing a checkout that is missing one object

A checkout here went missing a single object and could no longer fetch at
all, because every pack the server sends has deltas based on it (item 434).
The ordinary answers do not work on that box: `git fetch` is the broken
thing, and re-cloning means coordinating a swap of a checkout other
sessions are working in.

This folder holds the object itself, as text.

## Why a text file and not a packfile

The missing object is a **commit**, which is a few lines of text. Git names
every object by the SHA-1 of `"<type> <length>\0<content>"`, so a commit
written back byte for byte gets the same name it had. That makes the repair
reviewable in a diff, verifiable by anyone, and recoverable through a web
browser on a machine whose `git fetch` does not work — which is exactly the
machine that needs it.

An 840-byte packfile also works and was tried first; it is not here because
a binary blob in the repository is harder to check and no easier to use.

## Repairing a checkout

On the damaged machine, with `<file>` the `.commit` file from this folder
(download it from the GitHub web UI — **not** with `git fetch`, which is
the thing that is broken):

```sh
cd /path/to/the/damaged/checkout
git hash-object -t commit -w --stdin < <file>      # prints the object's name
git cat-file -t 984b725b5a11346ebf011808812d562a43869fc7   # -> commit
git fetch origin nightly                            # works again
```

`git hash-object -w` writes the object and prints its name. **If the name
it prints is not the one in the file's own name, stop**: the file was
altered in transit and writing it has put a different object into the
store, which is a new problem rather than a fix. Remove it and fetch the
file again.

## Checking this file without a damaged repository

`tests/test_a_missing_object_can_be_put_back.py` recomputes the name from
the bytes on disk, so a file corrupted in the repository fails there rather
than on the machine that needs it.
