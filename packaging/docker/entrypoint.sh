#!/bin/sh
# spaCR container entrypoint.
#
# Three jobs, in order, and nothing else:
#
#   1. Give the process a writable HOME, whatever UID it is running as.
#   2. Point every cache spaCR and its dependencies write at that HOME, so a
#      container started with --read-only or with a foreign UID does not fail
#      six frames inside matplotlib.
#   3. Make a mounted /models the folder the Model Zoo downloads into, and the
#      folder Cellpose loads from.
#
# WHY 1 IS NOT AUTOMATIC. The image's own user is `spacr`, and `docker run
# --user "$(id -u):$(id -g)"` -- which is what a user on a shared filesystem
# should do, and what the documentation tells them to do -- replaces that UID
# with one that has no passwd entry. Python's os.path.expanduser then falls
# back to the HOME variable, which still names /home/spacr, which that UID
# cannot write to. The symptom is a matplotlib font-cache traceback, or a
# Cellpose download that "works" and writes nothing.
#
# WHY 3 IS A SYMLINK AND AN ENVIRONMENT VARIABLE. Two different readers ask
# for the model folder two different ways. Cellpose reads
# CELLPOSE_LOCAL_MODELS_PATH (cellpose/models.py). spaCR's Model Zoo resolves
# Path.home()/".cellpose"/"models" and Path.home()/".spacr"/"models" directly
# (spacr/model_zoo.py, default_local_roots), which no variable reaches. Setting
# only the variable would leave the zoo listing an empty folder; linking only
# the paths would leave Cellpose downloading into HOME. Both, or one of them is
# wrong.
set -eu

# ---------------------------------------------------------------------------
# 1. A writable HOME.
# ---------------------------------------------------------------------------
: "${HOME:=/home/spacr}"
if ! ( [ -d "$HOME" ] && [ -w "$HOME" ] ); then
    fallback="/tmp/spacr-home-$(id -u)"
    mkdir -p "$fallback" 2>/dev/null || true
    if [ -w "$fallback" ]; then
        if [ "${SPACR_CONTAINER_QUIET:-}" != "1" ]; then
            echo "spacr: HOME ($HOME) is not writable for UID $(id -u);" \
                 "using $fallback instead." >&2
            echo "spacr: mount a writable home with" \
                 "-v \"\$HOME/.spacr-container:/home/spacr\"" \
                 "to keep caches between runs." >&2
        fi
        HOME="$fallback"
    fi
fi
export HOME

# ---------------------------------------------------------------------------
# 2. Caches, all under HOME, all only when the caller has not chosen already.
# ---------------------------------------------------------------------------
: "${XDG_CONFIG_HOME:=$HOME/.config}"
: "${XDG_DATA_HOME:=$HOME/.local/share}"
: "${XDG_CACHE_HOME:=$HOME/.cache}"
: "${MPLCONFIGDIR:=$XDG_CACHE_HOME/matplotlib}"
: "${HF_HOME:=$XDG_CACHE_HOME/huggingface}"
export XDG_CONFIG_HOME XDG_DATA_HOME XDG_CACHE_HOME MPLCONFIGDIR HF_HOME
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" \
         "$MPLCONFIGDIR" "$HF_HOME" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. A mounted /models becomes the model folder.
#
# /models exists in the image whether or not anything is mounted on it, and
# that is deliberate: with nothing mounted, a model downloaded during the run
# lands in the container's own writable layer and is discarded with the
# container, which is what an unconfigured `docker run --rm` should do. It is
# mode 1777 so this holds for any --user as well.
# ---------------------------------------------------------------------------
if [ -d /models ]; then
    : "${CELLPOSE_LOCAL_MODELS_PATH:=/models}"
    export CELLPOSE_LOCAL_MODELS_PATH
    for leaf in .cellpose/models .spacr/models; do
        target="$HOME/$leaf"
        if [ ! -e "$target" ]; then
            mkdir -p "$(dirname "$target")" 2>/dev/null || true
            ln -s /models "$target" 2>/dev/null || true
        fi
    done
fi

# A container has no display unless the user passed one in, and a pipeline
# that reaches plt.show() on a headless node must not block. spacr.cli forces
# Agg itself; this covers everything a user runs by hand beside it.
: "${MPLBACKEND:=Agg}"
export MPLBACKEND

exec "$@"
