#!/bin/sh
#
# Install spaCR from source WITHOUT downloading the whole repository.
#
# Instruction 328. A plain `git clone` of spaCR fetches a multi-gigabyte
# history and lays down ~427 MB of tracked files, of which ~76 MB is what
# you actually need to run the program. This script fetches only that.
#
# It uses two independent git features, and they cut different things:
#
#   --depth 1 --filter=blob:none   cuts the DOWNLOAD. Shallow takes one
#                                  commit instead of the whole history;
#                                  the blob filter then declines even that
#                                  commit's file contents until something
#                                  asks for them.
#   sparse-checkout                cuts the WORKING TREE. Combined with the
#                                  filter, the paths it excludes are never
#                                  requested, so they are never downloaded.
#
# Neither alone is enough: sparse-checkout on its own still downloads
# everything and merely declines to write it to disk.
#
# Usage:
#   ./install_from_source.sh [options]
#
#   --dir PATH            where to put the checkout (default ./spacr)
#   --branch REF          branch or tag to install (default main)
#   --repo URL            source repository
#   --with-translations   keep the extended translation catalogs (+33 MB)
#   --with-tests          keep the test suite (+32 MB)
#   --with-docs           keep the documentation sources (+236 MB)
#   --no-install          fetch only; skip `pip install`
#   --help
#
set -eu

REPO="https://github.com/EinarOlafsson/spacr.git"
BRANCH="main"
DIR="spacr"
KEEP_TRANSLATIONS=0
KEEP_TESTS=0
KEEP_DOCS=0
DO_INSTALL=1

usage() { sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)              DIR="$2"; shift 2 ;;
        --branch)           BRANCH="$2"; shift 2 ;;
        --repo)             REPO="$2"; shift 2 ;;
        --with-translations) KEEP_TRANSLATIONS=1; shift ;;
        --with-tests)       KEEP_TESTS=1; shift ;;
        --with-docs)        KEEP_DOCS=1; shift ;;
        --no-install)       DO_INSTALL=0; shift ;;
        -h|--help)          usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# The exclusion list lives beside this script so that the script and the
# test that pins it read the SAME file and cannot drift.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXCLUDES="$HERE/source_install_excludes.txt"

if [ ! -f "$EXCLUDES" ]; then
    # Running the script standalone -- curl'd on its own, say -- is a
    # supported way to use it, so fetch the list rather than failing.
    EXCLUDES=$(mktemp)
    RAW="https://raw.githubusercontent.com/EinarOlafsson/spacr/$BRANCH/packaging/source_install_excludes.txt"
    echo "fetching exclusion list from $RAW"
    curl -fsSL "$RAW" -o "$EXCLUDES"
fi

if [ -e "$DIR" ]; then
    echo "error: $DIR already exists -- pass --dir to choose somewhere else" >&2
    exit 1
fi

echo "fetching $BRANCH from $REPO into $DIR"

mkdir -p "$DIR"
cd "$DIR"
git init -q
git remote add origin "$REPO"

# `promisor` and `partialclonefilter` are what a real `git clone --filter`
# writes into the config. Setting them by hand is what makes the filter
# stick for LATER fetches too, not just the first one.
git config core.sparseCheckout true
git config remote.origin.promisor true
git config remote.origin.partialclonefilter blob:none

# Strip comments and blank lines; git reads this file itself and is
# happier without them.
grep -v '^[[:space:]]*#' "$EXCLUDES" | grep -v '^[[:space:]]*$' \
    > .git/info/sparse-checkout

drop_exclusion() {
    grep -v -x -F "$1" .git/info/sparse-checkout > .git/info/sparse-checkout.new
    mv .git/info/sparse-checkout.new .git/info/sparse-checkout
}

# Written as `if` blocks, NOT as `[ ... ] && drop_exclusion`. Under
# `set -eu` an AND-OR list whose test is false returns non-zero, and that
# aborts the script -- so the short form would make every install that did
# NOT ask for translations exit right here.
if [ "$KEEP_TRANSLATIONS" = 1 ]; then drop_exclusion '!/spacr/qt/i18n_catalogs/'; fi
if [ "$KEEP_TESTS" = 1 ];        then drop_exclusion '!/tests/'; fi
if [ "$KEEP_DOCS" = 1 ];         then drop_exclusion '!/docs/'; fi

# `|| true` on the fetch: servers that do not implement filtering warn and
# send everything instead. That is slower, not broken, so it must not stop
# the install.
git fetch -q --depth 1 --filter=blob:none origin "$BRANCH" || \
    git fetch -q --depth 1 origin "$BRANCH"
git checkout -q FETCH_HEAD

# Leave the checkout on a real branch rather than a detached FETCH_HEAD,
# so `git pull` works afterwards.
git checkout -q -B "$BRANCH"
git branch -q --set-upstream-to="origin/$BRANCH" "$BRANCH" 2>/dev/null || true

echo "checked out $(git rev-parse --short HEAD)"
echo "working tree: $(du -sh --exclude=.git . 2>/dev/null | cut -f1)"
echo "git objects:  $(du -sh .git 2>/dev/null | cut -f1)"

if [ "$DO_INSTALL" = 1 ]; then
    echo "installing with pip"
    python -m pip install -e .
    echo
    echo "done. Start spaCR with:  spacr"
else
    echo
    echo "done (fetch only). To install:  cd $DIR && python -m pip install -e ."
fi
