#!/bin/sh
#
# Install spaCR from source WITHOUT downloading the whole repository.
#
# Instruction 328. A plain `git clone` of spaCR downloads 5.8 GB and lays
# down ~941 MB of tracked files, of which ~55 MB is what you actually need
# to run the program. This script fetches only that: measured 2026-09-14
# against `main`, 86 MB on disk in 5 s, against 6.7 GB in 131 s. Re-measure
# with packaging/measure_clone_forms.sh; the old numbers in this header
# (427 MB and 76 MB) were a year of tree-growth out of date.
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
#   --with-translations   keep the extended translation catalogs (+17 MB)
#   --with-tests          keep the test suite (+35 MB)
#   --with-docs           keep the documentation sources (+419 MB)
#   --no-install          fetch only; skip `pip install`
#   --show-exclusions     print the exclusion list and where it came from
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
SHOW_EXCLUSIONS=0

# WHERE THIS SCRIPT LIVES, resolved HERE and not inside the function that
# wants it, because the function is called after `cd "$DIR"` and `$0` is a
# RELATIVE path for the ordinary invocation:
#
#     sh packaging/install_from_source.sh
#
# `dirname "$0"` was then resolved against the new checkout, found no
# `packaging/`, and fell back to the embedded copy of the list -- silently.
# Editing packaging/source_install_excludes.txt and running the script the
# obvious way did nothing at all, and nothing said so. An absolute `$0`
# happened to work, which is why the end-to-end test never saw it.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd) || SCRIPT_DIR=""

usage() { sed -n '3,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)              DIR="$2"; shift 2 ;;
        --branch)           BRANCH="$2"; shift 2 ;;
        --repo)             REPO="$2"; shift 2 ;;
        --with-translations) KEEP_TRANSLATIONS=1; shift ;;
        --with-tests)       KEEP_TESTS=1; shift ;;
        --with-docs)        KEEP_DOCS=1; shift ;;
        --no-install)       DO_INSTALL=0; shift ;;
        --show-exclusions)  SHOW_EXCLUSIONS=1; shift ;;
        -h|--help)          usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# The exclusion list. It lives in packaging/source_install_excludes.txt
# WITH its justifications, and is embedded here as well.
#
# The embedding is not duplication for its own sake: this script is meant
# to be curl'd on its own, and the first version fetched the list over the
# network as a second request. That broke immediately, and in the way a
# second request always breaks -- the script was curl'd from `nightly`,
# defaulted its branch to `main`, asked `main` for a file that only exists
# on `nightly`, and died on a bare `curl: (56) 404` under `set -e`.
#
# A self-contained script has no such failure mode. A test asserts this
# copy is character-for-character the same as the file, so the two cannot
# drift.
# Which of the two the run will use, as a value rather than as a variable
# set inside a function: `list=$(read_the_exclusions)` runs the function in
# a subshell, so anything it assigned was lost and --show-exclusions printed
# an empty origin. One function answers where, the other answers what.
exclusions_file() {
    if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/source_install_excludes.txt" ]; then
        echo "$SCRIPT_DIR/source_install_excludes.txt"
    fi
}

read_the_exclusions() {
    from=$(exclusions_file)
    if [ -n "$from" ]; then
        # Running from inside a checkout: prefer the file, so someone
        # editing the list sees their edit take effect immediately.
        grep -v '^[[:space:]]*#' "$from" | grep -v '^[[:space:]]*$'
        return
    fi
    cat <<'SPACR_EXCLUDES'
/*
!/docs/
!/tests/
!/features/
!/Notebooks/
!/proposals/
!/tools/
!/data/
!/spacr/resources/models/
!/spacr/resources/home/versions/
!/spacr/resources/icons/backup_icons/
!/spacr/qt/i18n_catalogs/
SPACR_EXCLUDES
}

# Answering --show-exclusions BEFORE anything is created: it is the flag
# somebody reaches for when they have edited the list and want to know
# whether the script can see the edit, and it should not leave a directory
# behind to answer that.
if [ "$SHOW_EXCLUSIONS" = 1 ]; then
    from=$(exclusions_file)
    echo "exclusions read from: ${from:-the copy embedded in this script}"
    read_the_exclusions
    exit 0
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

read_the_exclusions > .git/info/sparse-checkout

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
# Two fallbacks, and they are for different failures.
#
# A server that does not implement filtering warns and sends everything;
# that is slower, not broken, so the plain fetch is tried next. A branch
# that does not exist is a different problem and deserves to be said out
# loud rather than surfacing as a bare git error.
if ! git fetch -q --depth 1 --filter=blob:none origin "$BRANCH" 2>/dev/null; then
    if ! git fetch -q --depth 1 origin "$BRANCH" 2>/dev/null; then
        echo "error: could not fetch branch '$BRANCH' from $REPO" >&2
        echo "       available branches:" >&2
        git ls-remote --heads "$REPO" 2>/dev/null \
            | sed 's#.*refs/heads/#         #' >&2
        echo "       pass --branch to choose one" >&2
        exit 1
    fi
fi
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
