#!/bin/sh
#
# Measure what each documented clone form actually costs.
#
# Instruction 328. The README quotes a number for the full clone and a
# number for the lightweight one, and a number in prose is a claim about a
# tree that moves. The first pair went stale inside a fortnight: the README
# said "Full clone: 427 MB" while the checkout had grown to ~975 MB, and it
# was quoting the size of the CHECKOUT anyway, when what a reader choosing
# between two commands wants to know is the size of the DOWNLOAD.
#
# This script re-measures. Run it, read the table, put the numbers in the
# README with the date you ran it.
#
# It reports three things per form, because they are not the same number:
#
#   downloaded   bytes git says it received, summed over every fetch the
#                form performs -- a partial clone performs two, and the
#                second one is the expensive one.
#   .git         objects on disk afterwards.
#   worktree     the files you can actually see.
#
# Usage:
#   ./measure_clone_forms.sh [options]
#
#   --repo URL       repository to measure (default: the spaCR remote)
#   --branch REF     branch to measure (default: main)
#   --dir PATH       working directory for the clones (default: a temp dir)
#   --forms LIST     comma-separated subset of: full,depth1,depth1-filter,light
#   --keep           do not delete the clones afterwards
#   --dry-run        print the commands and exit, touching no network
#   --help
#
# A full clone of spaCR is several gigabytes. That is the point of the
# measurement, but do not run this on a metered connection by accident.
set -eu

REPO="https://github.com/EinarOlafsson/spacr.git"
BRANCH="main"
WORKDIR=""
FORMS="full,depth1,depth1-filter,light"
KEEP=0
DRY=0

usage() { sed -n '3,35p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)    REPO="$2"; shift 2 ;;
        --branch)  BRANCH="$2"; shift 2 ;;
        --dir)     WORKDIR="$2"; shift 2 ;;
        --forms)   FORMS="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd) || HERE="."
INSTALLER="$HERE/install_from_source.sh"

# The command each form runs, with $1 as the destination. Keeping them in
# one place is what lets --dry-run print the very thing that would run,
# rather than a description of it that can drift from it.
command_for() {
    case "$1" in
        full)          echo "git clone --progress --branch $BRANCH $REPO DEST" ;;
        depth1)        echo "git clone --progress --depth 1 --branch $BRANCH $REPO DEST" ;;
        depth1-filter) echo "git clone --progress --depth 1 --filter=blob:none --branch $BRANCH $REPO DEST" ;;
        light)         echo "sh $INSTALLER --repo $REPO --branch $BRANCH --no-install --dir DEST" ;;
        *) echo "unknown form: $1" >&2; exit 2 ;;
    esac
}

FORM_LIST=$(echo "$FORMS" | tr ',' ' ')

# Cloning a PATH is not cloning a URL, and the difference is the whole
# measurement. Git clones a local repository by hardlinking its object
# store, so `--depth 1` against a path came back with a 6.5 GB .git and a
# download of nothing at all. Measure against the real remote, or against
# a `file://` URL, which forces the transport.
case "$REPO" in
    *://*) ;;
    *) echo "warning: $REPO is a local path -- git hardlinks local clones" >&2
       echo "         and ignores the filter, so these numbers are not a" >&2
       echo "         download. Use a URL, or file://$REPO to force it." >&2 ;;
esac

if [ "$DRY" = 1 ]; then
    for form in $FORM_LIST; do
        printf '%-14s %s\n' "$form" "$(command_for "$form" | sed "s#DEST#<dir>/$form#")"
    done
    exit 0
fi

if [ -z "$WORKDIR" ]; then
    WORKDIR=$(mktemp -d)
    CLEANUP="$WORKDIR"
else
    mkdir -p "$WORKDIR"
    CLEANUP=""
fi

# `git` reports progress on stderr and rewrites the line with \r. Summing
# every "Receiving objects: 100% ..., <n> <unit>" is what makes a partial
# clone's SECOND fetch visible: its first fetch is a few hundred KiB of
# trees and looks like a triumph until the checkout goes back for the
# blobs.
received_bytes() {
    tr '\r' '\n' < "$1" \
    | sed -n 's/.*Receiving objects: 100% ([0-9/]*), \([0-9.]*\) \([KMG]*\)iB .*/\1 \2/p' \
    | awk '{ mult = 1
             if ($2 == "K") mult = 1024
             if ($2 == "M") mult = 1048576
             if ($2 == "G") mult = 1073741824
             total += $1 * mult }
           END { printf "%d", total + 0 }'
}

human() { awk -v b="$1" 'BEGIN { printf "%.1f MB", b / 1048576 }'; }

# install_from_source.sh fetches with `git fetch -q`, and a quiet fetch
# prints no progress at all -- so `downloaded` comes back as zero for it,
# which would read as "it downloaded nothing" rather than "git did not
# say". Print `n/a` there and let the .git column answer instead; on the
# forms that DO report, the two agree to within a per cent (595 MiB
# received against a 596 MiB .git), so .git is a fair stand-in.
downloaded_column() {
    if [ "${1:-0}" -eq 0 ] 2>/dev/null; then echo "n/a"; else human "$1"; fi
}

printf '%-14s %10s %12s %12s %12s\n' form seconds downloaded .git worktree
for form in $FORM_LIST; do
    dest="$WORKDIR/$form"
    rm -rf "$dest"
    cmd=$(command_for "$form" | sed "s#DEST#$dest#")
    start=$(date +%s)
    # shellcheck disable=SC2086
    if ! sh -c "$cmd" > "$WORKDIR/$form.out" 2> "$WORKDIR/$form.err"; then
        printf '%-14s %10s  FAILED -- see %s\n' "$form" "-" "$WORKDIR/$form.err"
        continue
    fi
    seconds=$(( $(date +%s) - start ))
    got=$(received_bytes "$WORKDIR/$form.err")
    gitb=$(du -sb "$dest/.git" 2>/dev/null | cut -f1)
    treeb=$(du -sb --exclude=.git "$dest" 2>/dev/null | cut -f1)
    printf '%-14s %10s %12s %12s %12s\n' \
        "$form" "$seconds" "$(downloaded_column "${got:-0}")" "$(human "${gitb:-0}")" "$(human "${treeb:-0}")"
    if [ "$KEEP" = 0 ]; then rm -rf "$dest"; fi
done

if [ "$KEEP" = 0 ] && [ -n "$CLEANUP" ]; then rm -rf "$CLEANUP"; fi
