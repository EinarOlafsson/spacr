#!/usr/bin/env bash
#
# Small online installer for Linux and macOS.
#
# The file itself contains no Python runtime or scientific dependencies. It
# downloads a pinned uv bootstrap over TLS, lets uv install a private CPython
# 3.12 runtime, then resolves spaCR and the appropriate PyTorch backend.

set -Eeuo pipefail
# Expanded below as ${SPACR_ORIGINAL_ARGS[@]+"${SPACR_ORIGINAL_ARGS[@]}"}:
# macOS ships bash 3.2, where `set -u` makes "${array[@]}" of an EMPTY array
# an unbound-variable error. A plain `curl ... | bash` from a Terminal running
# under Rosetta passes no arguments and died on the arm64 re-exec.
SPACR_ORIGINAL_ARGS=("$@")
GUI_REQUEST=0

# @SPACR_INSTALLER_MESSAGES_BEGIN@
SPACR_INSTALLER_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=generated/installer_messages.sh
source "$SPACR_INSTALLER_DIR/generated/installer_messages.sh"
# @SPACR_INSTALLER_MESSAGES_END@

# Where this script lives. The macOS package's first-launch helper runs it from
# the package's own support folder.
SPACR_SOURCE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# @SPACR_CLEANUP_MODULE_BEGIN@
# spacr/install_cleanup.py finds and removes older spaCR installations. A
# rendered standalone installer carries the module itself between these
# markers; run from a checkout, the script copies it from the source tree.
spacr_write_cleanup_module() {
    cp "$SPACR_SOURCE_DIR/../../spacr/install_cleanup.py" "$1"
}
# @SPACR_CLEANUP_MODULE_END@

# @SPACR_PROFILE_MODULE_BEGIN@
# The installer owns the profile schema even when it installs an older spaCR.
spacr_write_profile_module() {
    cp "$SPACR_SOURCE_DIR/../../spacr/install_profile.py" "$1"
}
# @SPACR_PROFILE_MODULE_END@

UV_VERSION="0.11.32"
PYTHON_VERSION="3.12"
DEFAULT_SPACR_VERSION="@SPACR_VERSION@"
TORCH_BACKEND="${SPACR_TORCH_BACKEND:-}"
DETECTED_ACCELERATOR="unknown"
LAUNCHER_DIR="${SPACR_LAUNCHER_DIR:-}"
# SHAP 0.52 leaves these dependencies unbounded and uv can otherwise select
# 2021 source releases whose metadata admits Python 3.12 but build scripts do
# not. Keep the released 1.4.9.9 wheel installable even though PyPI artifacts
# are immutable; setup.py carries the same guards for subsequent releases.
RESOLVER_GUARDS=("numba>=0.60,<1.0" "llvmlite>=0.43,<1.0")

PLATFORM=""
INSTALL_ROOT=""
PACKAGE_SPEC="${SPACR_PACKAGE_SPEC:-}"
SKIP_SYSTEM_DEPS=0
NO_LAUNCH=0
NO_COMMAND_LAUNCHER=0
DRY_RUN="${SPACR_INSTALL_DRY_RUN:-0}"
CONSENT_COLLECTED=0
SHARE_DIAGNOSTICS=0
REPORT_ISSUES=0
# wgpu draws the True 3D view through Vulkan, Metal or Direct3D 12. Without
# it spaCR draws 3D through OpenGL (VisPy) or the CPU. Off by default.
WITHOUT_WGPU="${SPACR_WITHOUT_WGPU:-0}"
WGPU_SKIP_REASON=""
# An offline bundle (packaging/offline/build_offline_bundle.py) carries uv,
# the private Python, every wheel, Cellpose weights and Mask test data, so
# nothing is downloaded. Without it the installer is the online one.
OFFLINE_BUNDLE="${SPACR_OFFLINE_BUNDLE:-}"
CHECK_MASK=0

usage() {
    spacr_say usage
    printf '\n%s:\n' "$(spacr_say options)"
    printf '  --platform linux|macos  %s\n' "$(spacr_say help_platform)"
    printf '  --install-root PATH     %s\n' "$(spacr_say help_install_root)"
    printf '  --package-spec SPEC     %s\n' "$(spacr_say help_package_spec)"
    printf '  --torch-backend NAME    %s\n' "$(spacr_say help_torch_backend)"
    printf '  --launcher-dir PATH     %s\n' "$(spacr_say help_launcher_dir)"
    printf '  --no-command-launcher   %s\n' "$(spacr_say help_no_command_launcher)"
    printf '  --skip-system-deps      %s\n' "$(spacr_say help_skip_system_deps)"
    printf '  --no-launch             %s\n' "$(spacr_say help_no_launch)"
    printf '  --gui                   %s\n' "$(spacr_say gui_explanation)"
    printf '  --dry-run               %s\n' "$(spacr_say help_dry_run)"
    printf '  --without-wgpu          %s\n' "$(spacr_say help_without_wgpu)"
    printf '  --share-diagnostics     Include redacted logs in report previews.\n'
    printf '  --report-issues         Show the public GitHub report action.\n'
    printf '  --consent-collected     Record that these choices were reviewed.\n'
    printf '  --offline-bundle PATH   Install from an offline bundle, with no network.\n'
    printf '  --check-mask            After an offline install, run Mask on its test data.\n'
    printf '  -h, --help              %s\n' "$(spacr_say help_help)"
}

require_option_value() {
    if (($# < 2)) || [[ -z "${2:-}" ]]; then
        spacr_say option_requires "$1" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --platform)
            require_option_value "$1" "${2:-}"
            PLATFORM="$2"
            shift 2
            ;;
        --install-root)
            require_option_value "$1" "${2:-}"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --package-spec)
            require_option_value "$1" "${2:-}"
            PACKAGE_SPEC="$2"
            shift 2
            ;;
        --torch-backend)
            require_option_value "$1" "${2:-}"
            TORCH_BACKEND="$2"
            shift 2
            ;;
        --launcher-dir)
            require_option_value "$1" "${2:-}"
            LAUNCHER_DIR="$2"
            shift 2
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            shift
            ;;
        --no-command-launcher)
            NO_COMMAND_LAUNCHER=1
            shift
            ;;
        --gui)
            GUI_REQUEST=1
            shift
            ;;
        --no-launch)
            NO_LAUNCH=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --share-diagnostics)
            SHARE_DIAGNOSTICS=1
            shift
            ;;
        --without-wgpu)
            WITHOUT_WGPU=1
            shift
            ;;
        --report-issues)
            REPORT_ISSUES=1
            shift
            ;;
        --consent-collected)
            CONSENT_COLLECTED=1
            shift
            ;;
        --offline-bundle)
            require_option_value "$1" "${2:-}"
            OFFLINE_BUNDLE="$2"
            shift 2
            ;;
        --check-mask)
            CHECK_MASK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            spacr_say unknown_option "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    case "$(uname -s)" in
        Linux)  PLATFORM="linux" ;;
        Darwin) PLATFORM="macos" ;;
        *)
            spacr_say unsupported_platform >&2
            exit 2
            ;;
    esac
fi
if [[ "$PLATFORM" != "linux" && "$PLATFORM" != "macos" ]]; then
    spacr_say unsupported_platform_value "$PLATFORM" >&2
    exit 2
fi

# An online managed install follows the Mac's native architecture, including
# when a terminal or caller is translated by Rosetta. Offline bundles retain
# their explicitly selected architecture and never change process mode here.
if [[ "$PLATFORM" == "macos" && -z "$OFFLINE_BUNDLE" && "$(uname -s)" == "Darwin" ]] && \
        [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" == "1" ]]; then
    exec /usr/bin/arch -arm64 /bin/bash "${BASH_SOURCE[0]}" ${SPACR_ORIGINAL_ARGS[@]+"${SPACR_ORIGINAL_ARGS[@]}"}
fi

PYTHON_REQUEST="$PYTHON_VERSION"
if [[ "$PLATFORM" == "macos" && -z "$OFFLINE_BUNDLE" ]]; then
    case "$(uname -m)" in
        arm64|aarch64) PYTHON_REQUEST="cpython-$PYTHON_VERSION-macos-aarch64-none" ;;
        x86_64) PYTHON_REQUEST="cpython-$PYTHON_VERSION-macos-x86_64-none" ;;
    esac
fi

# Record detected hardware for validation and the install profile. Automatic
# backend selection is the user-facing default: uv chooses CUDA on compatible
# NVIDIA systems and falls back to the portable wheel elsewhere. An explicit
# environment variable or --torch-backend always wins.
if [[ "$PLATFORM" == "linux" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        DETECTED_ACCELERATOR="nvidia"
    else
        DETECTED_ACCELERATOR="none"
    fi
elif [[ "$(uname -m)" == "arm64" ]]; then
    DETECTED_ACCELERATOR="apple-silicon"
else
    DETECTED_ACCELERATOR="none"
fi
if [[ -z "$TORCH_BACKEND" ]]; then
    TORCH_BACKEND="auto"
fi

# An offline bundle was locked for one PyTorch wheel line when it was built,
# and no package manager can reach its mirrors, so both come from the bundle.
if [[ -n "$OFFLINE_BUNDLE" ]]; then
    if [[ ! -f "$OFFLINE_BUNDLE/bundle.json" ]]; then
        printf 'Not an offline bundle (no bundle.json): %s\n' "$OFFLINE_BUNDLE" >&2
        exit 2
    fi
    OFFLINE_BUNDLE="$(CDPATH= cd -- "$OFFLINE_BUNDLE" && pwd)"
    TORCH_BACKEND="$(sed -n 's/.*"torch_backend": *"\([a-z0-9]*\)".*/\1/p' "$OFFLINE_BUNDLE/bundle.json")"
    PACKAGE_SPEC="$(sed -n 's/.*"package_spec": *"\([^"]*\)".*/\1/p' "$OFFLINE_BUNDLE/bundle.json")"
    SKIP_SYSTEM_DEPS=1
fi

# llvmlite 0.46+ no longer publishes Intel macOS wheels. Without this
# architecture-specific ceiling uv selects the latest release, attempts a
# source build, and the otherwise self-contained installer fails looking for
# a Homebrew LLVMConfig.cmake. Numba 0.63+ requires that unavailable llvmlite
# line, so keep the pair on the newest mutually compatible Intel wheels.
if [[ "$PLATFORM" == "macos" && "$(uname -m)" == "x86_64" ]]; then
    RESOLVER_GUARDS=(
        "numpy>=1.26,<2.0"
        "opencv-python-headless<4.12"
        "numba>=0.60,<0.63"
        "llvmlite>=0.43,<0.46"
        "moderngl>=5.12,<5.13"
        "glcontext>=3.0,<3.1"
        "cryptography<49"
    )
fi
# wgpu publishes Linux wheels for glibc 2.28 and newer only; on an older
# glibc the resolver would pick its source release, which does not build.
# Such a system installs without wgpu, decided before resolving. An unknown
# C library (musl, or no getconf/ldd) is not treated as old.
spacr_glibc_version() {
    local text=""
    if [[ -n "${SPACR_GLIBC_VERSION:-}" ]]; then
        printf '%s\n' "$SPACR_GLIBC_VERSION"
        return 0
    fi
    if command -v getconf >/dev/null 2>&1; then
        text="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
    fi
    if [[ ! "$text" =~ [Gg][Ll][Ii][Bb][Cc] ]] && command -v ldd >/dev/null 2>&1; then
        text="$(ldd --version 2>&1 | head -n 1 || true)"
        if [[ ! "$text" =~ [Gg][Ll][Ii][Bb][Cc]|GNU ]]; then
            text=""
        fi
    fi
    if [[ "$text" =~ ([0-9]+)\.([0-9]+)[^0-9]*$ ]]; then
        printf '%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    fi
}
if [[ "$PLATFORM" == "linux" ]]; then
    GLIBC_VERSION="$(spacr_glibc_version)"
    if [[ "$GLIBC_VERSION" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        glibc_major="${BASH_REMATCH[1]}"
        glibc_minor="${BASH_REMATCH[2]}"
        if (( glibc_major < 2 || (glibc_major == 2 && glibc_minor < 28) )); then
            WITHOUT_WGPU=1
            WGPU_SKIP_REASON="glibc"
            spacr_say wgpu_glibc_too_old "$GLIBC_VERSION"
        fi
    fi
fi
if [[ "$WITHOUT_WGPU" != "0" && "$WITHOUT_WGPU" != "1" ]]; then
    WITHOUT_WGPU=1
fi

if [[ ! "$TORCH_BACKEND" =~ ^[a-z0-9]+$ ]]; then
    spacr_say invalid_backend "$TORCH_BACKEND" >&2
    exit 2
fi

ask_yes_no() {
    local prompt="$1"
    local reply=""
    printf '%s [y/N] ' "$prompt"
    IFS= read -r reply || reply=""
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# A terminal-launched .run (and the macOS first-user helper) can collect the
# choices here. Package-manager and CI invocations have no TTY, remain fully
# non-interactive, and leave both choices off for the app to ask later.
if [[ "$CONSENT_COLLECTED" == "0" && -t 0 && "$DRY_RUN" != "1" && "$GUI_REQUEST" != "1" ]]; then
    printf '\nspaCR privacy choices\n'
    printf '%s\n' \
        'Anonymised error reports are on by default. They are filed as public GitHub issues by spaCR'"'"'s relay, with no account needed, and never include paths, names, IP addresses, emails, tokens, logs, images or your content.' \
        'spaCR AI (free) is also on by default. Turn reports off in Preferences > AI > AI Error report. The terms of use, Sections 5.6-5.7 and 6, give the details.'
    ask_yes_no 'Include redacted diagnostic logs in report previews?' && SHARE_DIAGNOSTICS=1
    ask_yes_no 'Enable the public GitHub issue-report action?' && REPORT_ISSUES=1
    CONSENT_COLLECTED=1
fi

if [[ -z "$INSTALL_ROOT" ]]; then
    if [[ "$PLATFORM" == "macos" && "$(id -u)" -eq 0 ]]; then
        INSTALL_ROOT="/Library/Application Support/spaCR"
    else
        INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/spacr"
    fi
fi

case "${INSTALL_ROOT%/}" in
    ""|"/"|"$HOME"|"/home"|"/Users"|"/Library"|"/usr"|"/usr/local")
        spacr_say unsafe_root "$INSTALL_ROOT" >&2
        spacr_say choose_directory >&2
        exit 2
        ;;
esac

if [[ -z "$PACKAGE_SPEC" ]]; then
    if [[ "$DEFAULT_SPACR_VERSION" == @*@ ]]; then
        PACKAGE_SPEC="spacr"
    else
        PACKAGE_SPEC="spacr==$DEFAULT_SPACR_VERSION"
    fi
fi

BOOTSTRAP_DIR="$INSTALL_ROOT/bootstrap"
PYTHON_DIR="$INSTALL_ROOT/python"
VENV_DIR="$INSTALL_ROOT/venv"
CACHE_DIR="$INSTALL_ROOT/cache"
UV_BIN="$BOOTSTRAP_DIR/uv"
UV_INSTALL_URL="https://astral.sh/uv/$UV_VERSION/install.sh"

if [[ "$PLATFORM" == "linux" ]]; then
    USER_BIN_DIR="${LAUNCHER_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
    DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    LAUNCHER="$USER_BIN_DIR/spacr"
else
    USER_BIN_DIR="${LAUNCHER_DIR:-/usr/local/bin}"
    DESKTOP_DIR=""
    LAUNCHER="$USER_BIN_DIR/spacr"
fi

spacr_say installer_title
printf '  %s:       %s\n' "$(spacr_say platform)" "$PLATFORM"
printf '  %s:    %s\n' "$(spacr_say application)" "$PACKAGE_SPEC"
printf '  %s: %s\n' "$(spacr_say private_python)" "$PYTHON_VERSION"
printf '  %s:   %s\n' "$(spacr_say install_root)" "$INSTALL_ROOT"
printf '  %s: %s\n' "$(spacr_say pytorch_backend)" "$TORCH_BACKEND"
printf '  GPU benchmark: RTX 3090 measured 13x faster Cellpose segmentation and 20x faster ResNet classification than CPU; hardware varies.\n'
printf '  %s: %s\n' "$(spacr_say resolver_guards)" "${RESOLVER_GUARDS[*]}"
if [[ "$WITHOUT_WGPU" == "1" ]]; then
    printf '  %s: %s\n' "$(spacr_say wgpu_plan)" "$(spacr_say wgpu_plan_skip)"
else
    printf '  %s: %s\n' "$(spacr_say wgpu_plan)" "$(spacr_say wgpu_plan_install)"
fi

if [[ -n "$OFFLINE_BUNDLE" ]]; then
    printf '  Offline bundle: %s\n' "$OFFLINE_BUNDLE"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    spacr_say dry_remove_old
    if [[ -z "$OFFLINE_BUNDLE" ]]; then
        spacr_say dry_download "$UV_INSTALL_URL"
    fi
    spacr_say dry_create "$VENV_DIR"
    if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
        spacr_say dry_launcher "$LAUNCHER"
    fi
    exit 0
fi

# @SPACR_LINUX_GUI_BEGIN@
spacr_linux_gui_install() {
    local installer="$1"
    shift
    if ! command -v zenity >/dev/null 2>&1; then
        spacr_say required_command zenity >&2
        return 3
    fi
    if ! command -v setsid >/dev/null 2>&1; then
        spacr_say required_command setsid >&2
        return 3
    fi
    if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
        spacr_say required_command "Bash 5.1+" >&2
        return 3
    fi
    local title choices intro log_folder log fifo backend window completed status
    title="$(spacr_say installer_title)"
    local wgpu_default=FALSE
    if [[ "${WITHOUT_WGPU:-0}" == "1" ]]; then wgpu_default=TRUE; fi
    intro="$(spacr_say gui_explanation)"$'\n\n'"$(spacr_say gui_preserve)"
    if ! choices="$(zenity --list --checklist --hide-header --width=760 --height=520 \
        --title="$title" --text="$intro" --ok-label="$(spacr_say gui_install)" \
        --cancel-label="$(spacr_say gui_cancel)" \
        --column="" --column="" --column="" --hide-column=2 --print-column=2 --separator='|' \
        TRUE diagnostics "$(spacr_say gui_share_diagnostics)" \
        TRUE issues "$(spacr_say gui_report_issues)" \
        "$wgpu_default" without_wgpu "$(spacr_say without_wgpu)")"; then
        return 130
    fi
    local diagnostics=0 issues=0 without_wgpu=0
    case "|$choices|" in *'|diagnostics|'*) diagnostics=1 ;; esac
    case "|$choices|" in *'|issues|'*) issues=1 ;; esac
    case "|$choices|" in *'|without_wgpu|'*) without_wgpu=1 ;; esac
    umask 077
    log_folder="${XDG_CACHE_HOME:-$HOME/.cache}/spacr/installer"
    mkdir -p "$log_folder"
    local work
    work="$(mktemp -d "$log_folder/gui.XXXXXXXX")"
    log="$work/install.log"
    fifo="$work/output.fifo"
    : > "$log"
    mkfifo "$fifo"
    zenity --text-info --auto-scroll --width=900 --height=600 --title="$title" \
        --ok-label="$(spacr_say gui_cancel)" < "$fifo" &
    window=$!
    local -a child_args=() child_options=(--consent-collected --skip-system-deps --no-launch)
    local argument
    for argument in "$@"; do
        case "$argument" in
            --gui|--consent-collected|--share-diagnostics|--report-issues|--without-wgpu) ;;
            *) child_args+=("$argument") ;;
        esac
    done
    if [[ "$diagnostics" == "1" ]]; then child_options+=(--share-diagnostics); fi
    if [[ "$issues" == "1" ]]; then child_options+=(--report-issues); fi
    if [[ "$without_wgpu" == "1" ]]; then child_options+=(--without-wgpu); fi
    SPACR_INSTALL_GUI_CHILD=1 SPACR_GUI_LOG="$log" SPACR_GUI_FIFO="$fifo" \
        setsid bash -c 'set -o pipefail; trap "wait; exit 130" TERM INT; bash "$@" 2>&1 | tee "$SPACR_GUI_LOG" > "$SPACR_GUI_FIFO"' \
        spacr-gui "$installer" ${child_args[@]+"${child_args[@]}"} "${child_options[@]}" < /dev/null &
    backend=$!
    completed=""
    status=0
    wait -n -p completed "$backend" "$window" || status=$?
    if [[ "${completed:-}" != "$backend" ]]; then
        kill -TERM -- "-$backend" 2>/dev/null || true
        zenity --progress --pulsate --title="$title" --text="$(spacr_say gui_stopping)" < /dev/null &
        local stopping=$!
        wait "$backend" || true
        kill "$stopping" 2>/dev/null || true
        wait "$stopping" 2>/dev/null || true
        rm -f "$fifo"
        zenity --info --no-markup --title="$title" --text="$(spacr_say gui_cancelled)" || true
        return 130
    fi
    kill "$window" 2>/dev/null || true
    wait "$window" 2>/dev/null || true
    rm -f "$fifo"
    if [[ "$status" != "0" ]]; then
        zenity --error --no-markup --title="$title" --text="$(spacr_say gui_failed)" || true
        zenity --text-info --width=900 --height=600 --title="$title" \
            --filename="$log" --ok-label="$(spacr_say gui_close)" || true
        return "$status"
    fi
    zenity --info --no-markup --title="$title" --text="$(spacr_say installed)" || true
    if [[ "$NO_LAUNCH" == "0" ]]; then
        nohup "$INSTALL_ROOT/venv/bin/python" -I -m spacr.qt >/dev/null 2>&1 &
    fi
    return 0
}
# @SPACR_LINUX_GUI_END@

if [[ "$PLATFORM" == "linux" && "${SPACR_INSTALL_GUI_CHILD:-0}" != "1" ]]; then
    if [[ "$GUI_REQUEST" == "1" ]] || \
       [[ "$CONSENT_COLLECTED" == "0" && "$NO_LAUNCH" == "0" && ! -t 0 && \
          -n "${DISPLAY:-${WAYLAND_DISPLAY:-}}" ]]; then
        spacr_linux_gui_install "$SPACR_SOURCE_DIR/$(basename -- "${BASH_SOURCE[0]}")" ${SPACR_ORIGINAL_ARGS[@]+"${SPACR_ORIGINAL_ARGS[@]}"}
        exit $?
    fi
fi

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        spacr_say required_command "$1" >&2
        exit 3
    fi
}

if [[ -z "$OFFLINE_BUNDLE" ]]; then
    require_command curl
fi
require_command sh
require_command tee

disk_probe="$(dirname "$INSTALL_ROOT")"
while [[ ! -e "$disk_probe" && "$disk_probe" != "/" ]]; do
    disk_probe="$(dirname "$disk_probe")"
done
available_kb="$(df -Pk "$disk_probe" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
if [[ "$available_kb" =~ ^[0-9]+$ ]] && ((available_kb < 5 * 1024 * 1024)); then
    spacr_say needs_free_space >&2
    spacr_say available_space "$((available_kb / 1024 / 1024))" "$INSTALL_ROOT" >&2
    exit 4
fi

install_linux_system_dependencies() {
    [[ "$PLATFORM" == "linux" && "$SKIP_SYSTEM_DEPS" == "0" ]] || return 0

    local elevate=()
    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            elevate=(sudo)
        else
            spacr_say no_sudo
            spacr_say qt_help
            return 0
        fi
    fi

    local arm_build=0
    local build_deps=()
    case "$(uname -m)" in
        aarch64|arm64|ARM64) arm_build=1 ;;
    esac

    spacr_say installing_linux_deps
    if command -v apt-get >/dev/null 2>&1; then
        [[ "$arm_build" == "0" ]] || build_deps=(build-essential libx11-dev)
        "${elevate[@]}" apt-get update
        "${elevate[@]}" apt-get install --no-install-recommends -y \
            libegl1 libgl1 libxkbcommon0 libdbus-1-3 libpulse0 \
            libx11-xcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
            libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
            libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
            libxkbcommon-x11-0 libnss3 libxcomposite1 libxcursor1 \
            libxdamage1 libxi6 libxtst6 libsm6 libxext6 libxrender1 ffmpeg \
            ${build_deps[@]+"${build_deps[@]}"}
    elif command -v dnf >/dev/null 2>&1; then
        [[ "$arm_build" == "0" ]] || build_deps=(gcc-c++ libX11-devel)
        "${elevate[@]}" dnf install -y \
            mesa-libGL mesa-libEGL libxkbcommon libxkbcommon-x11 \
            libxcb xcb-util-cursor xcb-util-image xcb-util-keysyms \
            xcb-util-renderutil dbus-libs pulseaudio-libs nss ffmpeg \
            ${build_deps[@]+"${build_deps[@]}"}
    elif command -v zypper >/dev/null 2>&1; then
        [[ "$arm_build" == "0" ]] || build_deps=(gcc-c++ libX11-devel)
        "${elevate[@]}" zypper --non-interactive install \
            Mesa-libGL1 Mesa-libEGL1 libxkbcommon0 libxkbcommon-x11-0 \
            libxcb1 libpulse0 libnss3 ffmpeg \
            ${build_deps[@]+"${build_deps[@]}"}
    elif command -v pacman >/dev/null 2>&1; then
        [[ "$arm_build" == "0" ]] || build_deps=(base-devel libx11)
        "${elevate[@]}" pacman -S --needed --noconfirm \
            libglvnd libxkbcommon-x11 libxcb xcb-util-cursor \
            xcb-util-image xcb-util-keysyms xcb-util-renderutil \
            dbus libpulse nss ffmpeg \
            ${build_deps[@]+"${build_deps[@]}"}
    else
        spacr_say unknown_package_manager
    fi
}

# @SPACR_INSTALLER_LEASE_BEGIN@
SPACR_INSTALLER_LEASE=""
SPACR_INSTALLER_LEASE_ID=""
SPACR_RUNTIME_LEASE=""
SPACR_RUNTIME_LEASE_ID=""
SPACR_INSTALLER_STATUS_LEASE=""
SPACR_INSTALLER_STATUS_LEASE_ID=""
SPACR_INSTALLER_RETAIN_LEASE=0
SPACR_ABANDONED_OWNERS=()
# GNU stat first, then BSD stat: PLATFORM alone does not say which one is on
# PATH (Homebrew's gnubin puts GNU stat first on a Mac).
spacr_installer_lease_identity() {
    stat -c '%d:%i' "$1" 2>/dev/null || stat -f '%d:%i' "$1"
}
# Each lease records its owner: pid, start stamp, host and (Linux) pid
# namespace. A later installer takes a lease over only when it is this
# user's, written on this host in this pid namespace, holds nothing but
# that record, and the system says the pid is gone or now belongs to a
# process with another start stamp of the same kind. Linux stamps the start
# tick from /proc (immune to clock steps, locale and time zone); macOS reads
# ps lstart in the C locale and UTC. A rollback marker, a journal, any other
# file, another user's, host's or container's lease, an unreadable process
# or a lease that recorded no owner is never taken over. One takeover at a
# time: a second installer judging the same lease refuses.
spacr_process_started() {
    local text fields
    if [[ "$(uname -s)" == "Linux" ]]; then
        text="$(cat "/proc/$1/stat" 2>/dev/null)" || return 0
        text="${text##*) }"
        read -r -a fields <<< "$text"
        if [[ "${fields[19]:-}" =~ ^[0-9]+$ ]]; then printf 'ticks:%s\n' "${fields[19]}"; fi
        return 0
    fi
    text="$(LC_ALL=C TZ=UTC0 ps -o lstart= -p "$1" 2>/dev/null | tr -s ' \t' ' ' | sed 's/^ //; s/ $//')" || true
    if [[ -n "$text" ]]; then printf 'lstart:%s\n' "$text"; fi
    return 0
}
spacr_process_gone() {
    local message
    kill -0 "$1" 2>/dev/null && return 1
    message="$( (export LC_ALL=C; kill -0 "$1") 2>&1 )" || true
    [[ "$message" == *"No such process"* ]]
}
spacr_pid_namespace() {
    if [[ "$(uname -s)" == "Linux" ]]; then readlink "/proc/$$/ns/pid" 2>/dev/null || printf 'unknown\n'; fi
    return 0
}
# The process group a lease records, when the system names one. Children of
# a killed installer (uv, its log tee) keep that group after the installer
# itself is gone, so a lease that recorded a group stays live while any
# member owned by this user survives, other than this installer's own
# ancestors and descendants. A lease that recorded no group is judged by its
# pid alone, as before.
spacr_process_group() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d ' \t' || true
}
spacr_group_live_member() {
    local listing
    listing="$(ps -A -o pid=,ppid=,pgid=,uid= 2>/dev/null)" || { printf 'unknown\n'; return 0; }
    [[ -n "$listing" ]] || { printf 'unknown\n'; return 0; }
    awk -v group="$1" -v me="$$" -v uid="$(id -u)" '
        { pid[NR] = $1; parent[$1] = $2; pg[$1] = $3; owner[$1] = $4 }
        END {
            for (p = me; p != "" && p != 0 && !(p in mine); p = parent[p]) mine[p] = 1
            for (i = 1; i <= NR; i++) {
                p = pid[i]
                if (pg[p] != group || owner[p] != uid || (p in mine)) continue
                steps = 0
                for (q = parent[p]; q != "" && q != 0 && q != 1 && steps++ < 4096; q = parent[q]) if (q == me) break
                if (q == me) continue
                print p; exit
            }
        }' <<< "$listing"
}
# Fields a lease written by a later installer may add are unknown here. Such a
# lease is never taken over, and when its pid is gone it is reported as a
# lease this version cannot check, not as held by that dead pid.
spacr_lease_has_unknown_fields() {
    [[ -f "$1/owner" ]] && grep -qvE '^(pid|started|host|pidns|pgid)=' "$1/owner"
}
spacr_lease_field() {
    [[ -f "$1/owner" ]] || return 0
    sed -n "s/^$2=//p" "$1/owner" 2>/dev/null | head -n 1 || true
}
spacr_stamp_installer_lease() {
    local group
    group="$(spacr_process_group "$$")"
    printf 'pid=%s\nstarted=%s\nhost=%s\npidns=%s\n' "$$" "$(spacr_process_started "$$")" "$(uname -n)" \
        "$(spacr_pid_namespace)" > "$1/owner"
    if [[ "$group" =~ ^[1-9][0-9]*$ ]]; then printf 'pgid=%s\n' "$group" >> "$1/owner"; fi
}
spacr_lease_owner_dead() {
    local lease="$1" key pid started now namespace group
    [[ -d "$lease" && ! -L "$lease" && -f "$lease/owner" && ! -L "$lease/owner" ]] || return 1
    [[ -O "$lease" && -O "$lease/owner" ]] || return 1
    [[ "$(ls -A "$lease")" == "owner" ]] || return 1
    grep -qvE '^(pid|started|host|pidns|pgid)=' "$lease/owner" && return 1
    for key in pid started host pidns; do
        [[ "$(grep -c "^$key=" "$lease/owner")" == "1" ]] || return 1
    done
    [[ "$(grep -c '^pgid=' "$lease/owner")" -le 1 ]] || return 1
    pid="$(spacr_lease_field "$lease" pid)"
    started="$(spacr_lease_field "$lease" started)"
    [[ "$pid" =~ ^[1-9][0-9]*$ && "$pid" != "$$" && -n "$started" ]] || return 1
    [[ "$(spacr_lease_field "$lease" host)" == "$(uname -n)" ]] || return 1
    namespace="$(spacr_pid_namespace)"
    [[ "$namespace" != "unknown" && "$(spacr_lease_field "$lease" pidns)" == "$namespace" ]] || return 1
    if ! spacr_process_gone "$pid"; then
        now="$(spacr_process_started "$pid")"
        [[ -n "$now" && "${now%%:*}" == "${started%%:*}" && "$now" != "$started" ]] || return 1
    fi
    group="$(spacr_lease_field "$lease" pgid)"
    [[ -z "$group" ]] && return 0
    [[ "$group" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ -z "$(spacr_group_live_member "$group")" ]]
}
spacr_take_installer_lease() {
    local lease="$1" pid member
    if (umask 077; mkdir "$lease") 2>/dev/null; then
        spacr_stamp_installer_lease "$lease"
        return 0
    fi
    if spacr_lease_owner_dead "$lease" && (umask 077; mkdir "$lease.takeover") 2>/dev/null; then
        spacr_stamp_installer_lease "$lease.takeover"
        if spacr_lease_owner_dead "$lease"; then
            SPACR_ABANDONED_OWNERS+=("$(spacr_lease_field "$lease" pid)")
            rm -f "$lease/owner"
            if rmdir "$lease" 2>/dev/null && (umask 077; mkdir "$lease") 2>/dev/null; then
                spacr_stamp_installer_lease "$lease"
                rm -f "$lease.takeover/owner"
                rmdir "$lease.takeover" 2>/dev/null || true
                spacr_say lease_cleared "$lease" >&2
                return 0
            fi
        fi
        rm -f "$lease.takeover/owner"
        rmdir "$lease.takeover" 2>/dev/null || true
    fi
    spacr_say already_running >&2
    pid="$(spacr_lease_field "$lease" pid)"
    if [[ -n "$pid" ]] && spacr_process_gone "$pid"; then
        member="$(spacr_lease_field "$lease" pgid)"
        [[ -n "$member" ]] && member="$(spacr_group_live_member "$member")"
        if [[ "$member" =~ ^[0-9]+$ ]]; then
            pid="$member"
        elif spacr_lease_has_unknown_fields "$lease"; then
            pid=""
        fi
    fi
    if [[ -n "$pid" ]]; then spacr_say lease_held "$lease" "$pid" >&2; else spacr_say lease_held_unknown "$lease" >&2; fi
    return 1
}
spacr_release_installer_lease() {
    local lease saved current
    if [[ "$SPACR_INSTALLER_RETAIN_LEASE" != "0" ]]; then
        for lease in "$SPACR_INSTALLER_STATUS_LEASE" "$SPACR_RUNTIME_LEASE" "$SPACR_INSTALLER_LEASE"; do
            [[ -n "$lease" && -d "$lease" && ! -L "$lease" ]] || continue
            printf '%s\n' "${work_dir:-retained}" > "$lease/rollback" 2>/dev/null || true
        done
        return 0
    fi
    for lease in "$SPACR_INSTALLER_STATUS_LEASE" "$SPACR_RUNTIME_LEASE" "$SPACR_INSTALLER_LEASE"; do
        [[ -n "$lease" && -d "$lease" && ! -L "$lease" ]] || continue
        if [[ "$lease" == "$SPACR_INSTALLER_STATUS_LEASE" ]]; then
            saved="$SPACR_INSTALLER_STATUS_LEASE_ID"
        elif [[ "$lease" == "$SPACR_RUNTIME_LEASE" ]]; then
            saved="$SPACR_RUNTIME_LEASE_ID"
        else
            saved="$SPACR_INSTALLER_LEASE_ID"
        fi
        current="$(spacr_installer_lease_identity "$lease")" || continue
        if [[ -n "$saved" && "$current" == "$saved" ]]; then
            rm -f "$lease/owner"
            rmdir "$lease" 2>/dev/null || true
        fi
    done
}
spacr_acquire_installer_lease() {
    local canonical parent
    parent="$(dirname -- "$INSTALL_ROOT")"
    mkdir -p "$parent"
    if [[ -d "$INSTALL_ROOT" ]]; then
        canonical="$(CDPATH= cd -- "$INSTALL_ROOT" && pwd -P)"
    else
        canonical="$(CDPATH= cd -- "$parent" && pwd -P)/$(basename -- "$INSTALL_ROOT")"
    fi
    case "$canonical" in
        ""|"/"|"$HOME"|"/home"|"/Users"|"/Library"|"/usr"|"/usr/local")
            spacr_say unsafe_root "$INSTALL_ROOT" >&2
            return 2
            ;;
    esac
    SPACR_INSTALLER_LEASE="$(dirname -- "$canonical")/.$(basename -- "$canonical").spacr-installer-lock"
    if ! spacr_take_installer_lease "$SPACR_INSTALLER_LEASE"; then
        SPACR_INSTALLER_LEASE=""
        return 73
    fi
    SPACR_INSTALLER_LEASE_ID="$(spacr_installer_lease_identity "$SPACR_INSTALLER_LEASE")"
    trap spacr_release_installer_lease EXIT
    mkdir -p "$INSTALL_ROOT"
    SPACR_RUNTIME_LEASE="$INSTALL_ROOT/.venv.spacr-update-lock"
    if ! spacr_take_installer_lease "$SPACR_RUNTIME_LEASE"; then
        SPACR_RUNTIME_LEASE=""
        return 73
    fi
    SPACR_RUNTIME_LEASE_ID="$(spacr_installer_lease_identity "$SPACR_RUNTIME_LEASE")"
    if [[ "$PLATFORM" == "macos" ]]; then
        SPACR_INSTALLER_STATUS_LEASE="$INSTALL_ROOT/.installing"
        if ! spacr_take_installer_lease "$SPACR_INSTALLER_STATUS_LEASE"; then
            SPACR_INSTALLER_STATUS_LEASE=""
            return 73
        fi
        SPACR_INSTALLER_STATUS_LEASE_ID="$(spacr_installer_lease_identity "$SPACR_INSTALLER_STATUS_LEASE")"
    fi
    spacr_remove_abandoned_staging
}
# A killed installer leaves its staging runtime behind. It is removed only
# for an owner whose leases this installer just took over -- pid and group
# provably gone -- and only now that this installer holds every lease.
spacr_remove_abandoned_staging() {
    local owner
    for owner in "${SPACR_ABANDONED_OWNERS[@]+"${SPACR_ABANDONED_OWNERS[@]}"}"; do
        [[ "$owner" =~ ^[1-9][0-9]*$ && "$owner" != "$$" ]] || continue
        spacr_process_gone "$owner" || continue
        if [[ -d "$INSTALL_ROOT/.venv-staging-$owner" && ! -L "$INSTALL_ROOT/.venv-staging-$owner" ]]; then
            rm -rf "$INSTALL_ROOT/.venv-staging-$owner" && spacr_say staging_removed "$INSTALL_ROOT/.venv-staging-$owner" >&2
        fi
        rm -f "$INSTALL_ROOT/.install-profile-staging-$owner.json"
    done
}
spacr_acquire_installer_lease
# @SPACR_INSTALLER_LEASE_END@

mkdir -p "$INSTALL_ROOT"
INSTALL_LOG="$INSTALL_ROOT/install.log"
touch "$INSTALL_LOG"
exec > >(tee -a "$INSTALL_LOG") 2>&1
spacr_say detailed_log "$INSTALL_LOG"
install_linux_system_dependencies
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/spacr-install.XXXXXX")"
work_dir="$(cd "$work_dir" && pwd -P)"
installer_tmp="$work_dir/uv-installer.sh"
stage_venv="$INSTALL_ROOT/.venv-staging-$$"
stage_profile="$INSTALL_ROOT/.install-profile-staging-$$.json"
old_venv=""
swap_started=0
metadata_touched=0
metadata_backup="$work_dir/previous-metadata"
cleanup() {
    local retain_work=0
    if [[ "$swap_started" == "1" && -d "$old_venv" ]]; then
        rm -rf "$VENV_DIR"
        mv "$old_venv" "$VENV_DIR"
    fi
    if [[ "$metadata_touched" == "1" ]]; then
        if ! "$cleanup_python" -I "$cleanup_module" installer-metadata restore --folder "$metadata_backup"; then
            retain_work=1
            spacr_say gui_failed >&2
            printf '%s\n' "$metadata_backup" >&2
        fi
    fi
    if [[ "$retain_work" == "0" ]]; then rm -rf "$work_dir"; else SPACR_INSTALLER_RETAIN_LEASE=1; fi
    rm -f "$stage_profile"
    if [[ -d "$stage_venv" ]]; then
        rm -rf "$stage_venv"
    fi
}
trap 'cleanup; spacr_release_installer_lease' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# find old spaCR files --> delete old spaCR files --> install new spaCR.
#
# uv and a private Python are fetched into a temporary folder first, outside
# every installation: a failed download stops the install here, before
# anything old is deleted, and the Python that runs the removal is not one of
# the files it removes. The finder and remover are spacr/install_cleanup.py,
# the same module the in-app update uses, so a layout added for one is never
# missed by the other. It never runs an old version's own uninstaller, keeps
# preferences and user data, and lists environments the user made without
# touching them.
work_uv="$work_dir/bootstrap/uv"
if [[ -n "$OFFLINE_BUNDLE" ]]; then
    # Every file is checked against the bundle's SHA256SUMS before any of it
    # is run: a bundle carried on a USB stick has no TLS to vouch for it.
    printf 'Verifying the offline bundle...\n'
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$OFFLINE_BUNDLE" && sha256sum --check --quiet SHA256SUMS)
    else
        (cd "$OFFLINE_BUNDLE" && shasum -a 256 --check --quiet SHA256SUMS)
    fi
    mkdir -p "$work_dir/bootstrap"
    cp "$OFFLINE_BUNDLE/uv/uv" "$work_uv"
    chmod 755 "$work_uv"
    export UV_PYTHON_INSTALL_MIRROR="file://$OFFLINE_BUNDLE/python"
else
    spacr_say downloading_uv
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
        --retry 3 --retry-all-errors \
        "$UV_INSTALL_URL" --output "$installer_tmp"
    UV_UNMANAGED_INSTALL="$work_dir/bootstrap" UV_NO_MODIFY_PATH=1 \
        sh "$installer_tmp"
fi
if [[ ! -x "$work_uv" ]]; then
    spacr_say uv_missing "$work_uv" >&2
    exit 5
fi

export UV_SYSTEM_CERTS=true
export UV_PYTHON_CACHE_DIR="$work_dir/python-downloads"

spacr_say downloading_python "$PYTHON_VERSION"
UV_PYTHON_INSTALL_DIR="$work_dir/python" UV_CACHE_DIR="$work_dir/cache" \
    "$work_uv" python install "$PYTHON_REQUEST" --managed-python --no-bin
cleanup_python="$(UV_PYTHON_INSTALL_DIR="$work_dir/python" \
    "$work_uv" python find "$PYTHON_REQUEST" --managed-python)"
cleanup_module="$work_dir/install_cleanup.py"
spacr_write_cleanup_module "$cleanup_module"
profile_module="$work_dir/install_profile.py"
spacr_write_profile_module "$profile_module"

# The log of this install, and the lock of the macOS first-launch helper, are
# not old files.
cleanup_args=(remove --keep "$INSTALL_ROOT" --keep "$INSTALL_LOG" --keep "$INSTALL_ROOT/.installing")
if [[ "$PLATFORM" == "macos" && -f "$SPACR_SOURCE_DIR/install-for-user.sh" ]]; then
    # Run by the package's first-launch helper: the package has just put the
    # new application bundle, its command link and this folder in place.
    cleanup_args+=(--keep "$SPACR_SOURCE_DIR" --keep "/Applications/spaCR.app"
                   --keep "/usr/local/bin/spacr")
fi
if [[ "$(id -u)" -ne 0 && -t 0 ]] && command -v sudo >/dev/null 2>&1; then
    # A terminal install may ask for a password to remove the Debian package.
    cleanup_args+=(--sudo)
fi
if [[ -n "${SPACR_CLEANUP_ROOT:-}" ]]; then
    cleanup_args+=(--root "$SPACR_CLEANUP_ROOT")
fi


mkdir -p "$BOOTSTRAP_DIR" "$PYTHON_DIR" "$CACHE_DIR"
mv "$work_uv" "$UV_BIN"

export UV_PYTHON_INSTALL_DIR="$PYTHON_DIR"
export UV_CACHE_DIR="$CACHE_DIR"

"$UV_BIN" python install "$PYTHON_REQUEST" --managed-python --no-bin

spacr_say creating_environment
rm -rf "$stage_venv"
"$UV_BIN" venv "$stage_venv" \
    --python "$PYTHON_REQUEST" --managed-python --relocatable

stage_python="$stage_venv/bin/python"

# CVXOPT's newer macOS wheels can require a newer OS than this machine.
# Select the newest compatible wheel instead of compiling against external
# SuiteSparse headers, including when system dependencies are skipped.
binary_args=()
if [[ "$PLATFORM" == "macos" ]]; then
    binary_args=(--only-binary=cvxopt)
fi

# Skipping wgpu at resolve time, not uninstalling it afterwards: an override
# whose marker never holds removes wgpu from the resolution, so a glibc that
# would otherwise send uv to the wgpu source release never sees it. Nothing
# else spaCR needs depends on wgpu or on rendercanvas, which only wgpu pulls.
override_args=()
if [[ "$WITHOUT_WGPU" == "1" ]]; then
    wgpu_overrides="$work_dir/without-wgpu-overrides.txt"
    printf '%s\n' 'wgpu; sys_platform == "never"' > "$wgpu_overrides"
    override_args=(--override "$wgpu_overrides")
fi

spacr_say downloading_dependencies
if [[ -n "$OFFLINE_BUNDLE" ]]; then
    "$UV_BIN" pip install \
        --python "$stage_python" \
        --compile-bytecode \
        --offline --no-index \
        --find-links "$OFFLINE_BUNDLE/wheels" \
        ${override_args[@]+"${override_args[@]}"} \
        ${binary_args[@]+"${binary_args[@]}"} \
        -r "$OFFLINE_BUNDLE/requirements.txt"
else
    "$UV_BIN" pip install \
        --python "$stage_python" \
        --compile-bytecode \
        --torch-backend "$TORCH_BACKEND" \
        ${override_args[@]+"${override_args[@]}"} \
        ${binary_args[@]+"${binary_args[@]}"} \
        "$PACKAGE_SPEC" \
        "${RESOLVER_GUARDS[@]}"
fi

spacr_say validating_install
# An install without wgpu leaves one declared dependency out on purpose, so
# uv's sole "spacr requires wgpu" diagnostic is accepted then and nothing else
# is (install_cleanup._only_saved_wgpu_omission, as the updater does).
pip_check_log="$work_dir/pip-check.txt"
pip_check_code=0
"$UV_BIN" pip check --python "$stage_python" > "$pip_check_log" 2>&1 || pip_check_code=$?
cat "$pip_check_log"
if [[ "$pip_check_code" != "0" ]]; then
    if [[ "$WITHOUT_WGPU" != "1" ]] || ! "$stage_python" -I -c \
        'import runpy, sys; only=runpy.run_path(sys.argv[1])["_only_saved_wgpu_omission"]; sys.exit(0 if only(int(sys.argv[2]), open(sys.argv[3], encoding="utf-8", errors="replace").read()) else 1)' \
        "$cleanup_module" "$pip_check_code" "$pip_check_log"; then
        exit "$pip_check_code"
    fi
fi
QT_QPA_PLATFORM=offscreen "$stage_python" -I -c \
    "import spacr, PySide6, torch; import numpy as np; assert torch.from_numpy(np.zeros(1, dtype=np.float32)).numel() == 1; print('spaCR', spacr.__version__, '| torch', torch.__version__, '| numpy', np.__version__)"
if [[ "$TORCH_BACKEND" != "cpu" && "$DETECTED_ACCELERATOR" == "nvidia" ]]; then
    "$stage_python" -I -c \
        "import torch; assert torch.cuda.is_available(), 'GPU install selected but CUDA is unavailable'"
elif [[ "$TORCH_BACKEND" != "cpu" && "$DETECTED_ACCELERATOR" == "apple-silicon" ]]; then
    "$stage_python" -I -c \
        "import torch; assert torch.backends.mps.is_available(), 'Apple GPU selected but MPS is unavailable'"
fi

"$stage_python" -I "$profile_module" \
    --path "$stage_profile" \
    --requested "$TORCH_BACKEND" \
    --detected "$DETECTED_ACCELERATOR" \
    --consent-collected "$CONSENT_COLLECTED" \
    --share-diagnostics "$SHARE_DIAGNOSTICS" \
    --report-issues "$REPORT_ISSUES" \
    --without-wgpu "$WITHOUT_WGPU"

identity_args=(verify-runtime)
if [[ "$DEFAULT_SPACR_VERSION" != @*@ ]]; then
    identity_args+=(--version "$DEFAULT_SPACR_VERSION")
fi
"$cleanup_python" -I "$cleanup_module" "${identity_args[@]}" \
    --python "$stage_python" --retiring-prefix "$VENV_DIR"
if ! "$cleanup_python" -I "$cleanup_module" "${cleanup_args[@]}"; then
    spacr_say old_copy_not_removed >&2
    exit 6
fi
metadata_paths=("$INSTALL_ROOT/runtime-identity.json" "$INSTALL_ROOT/install-profile.json")
if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then metadata_paths+=("$LAUNCHER"); fi
if [[ "$PLATFORM" == "linux" ]]; then
    metadata_paths+=("$DESKTOP_DIR/io.github.olafssonlab.spacr.desktop" "$INSTALL_ROOT/uninstall-spacr.sh")
fi
if [[ -n "$OFFLINE_BUNDLE" ]]; then
    metadata_paths+=("$INSTALL_ROOT/offline_mask_check.py" "$INSTALL_ROOT/offline-bundle.json")
fi
"$cleanup_python" -I "$cleanup_module" installer-metadata snapshot --folder "$metadata_backup" --paths "${metadata_paths[@]}"
old_venv="$INSTALL_ROOT/.venv-previous-$$"
if [[ -d "$VENV_DIR" ]]; then
    mv "$VENV_DIR" "$old_venv"
    swap_started=1
fi
mv "$stage_venv" "$VENV_DIR"
if ! "$cleanup_python" -I "$cleanup_module" "${identity_args[@]}" \
    --python "$VENV_DIR/bin/python" > "$work_dir/runtime-identity.new.json"; then
    rm -rf "$VENV_DIR"
    if [[ -d "$old_venv" ]]; then
        mv "$old_venv" "$VENV_DIR"
    fi
    exit 7
fi
metadata_touched=1
mv "$work_dir/runtime-identity.new.json" "$INSTALL_ROOT/runtime-identity.json"
mv "$stage_profile" "$INSTALL_ROOT/install-profile.json"

if [[ -n "$OFFLINE_BUNDLE" ]]; then
    # Cellpose looks for its weights here before it would download them.
    # Weights already present are left as they are.
    cellpose_dir="${CELLPOSE_LOCAL_MODELS_PATH:-$HOME/.cellpose/models}"
    if [[ -d "$OFFLINE_BUNDLE/models/cellpose" ]]; then
        mkdir -p "$cellpose_dir"
        for weights in "$OFFLINE_BUNDLE/models/cellpose"/*; do
            [[ -e "$cellpose_dir/$(basename "$weights")" ]] || cp "$weights" "$cellpose_dir/"
        done
        printf 'Cellpose models: %s\n' "$cellpose_dir"
    fi
    if [[ -d "$OFFLINE_BUNDLE/test_data" && ! -e "$INSTALL_ROOT/test_data" ]]; then
        cp -R "$OFFLINE_BUNDLE/test_data" "$INSTALL_ROOT/test_data"
        printf 'Mask test data: %s\n' "$INSTALL_ROOT/test_data/plate1"
    fi
    cp "$OFFLINE_BUNDLE/offline_mask_check.py" "$INSTALL_ROOT/offline_mask_check.py"
    cp "$OFFLINE_BUNDLE/bundle.json" "$INSTALL_ROOT/offline-bundle.json"
    if [[ "$CHECK_MASK" == "1" ]]; then
        printf 'Running Mask on the bundled test data...\n'
        "$VENV_DIR/bin/python" -I "$INSTALL_ROOT/offline_mask_check.py" \
            --data "$INSTALL_ROOT/test_data/plate1"
    fi
fi

if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
    mkdir -p "$USER_BIN_DIR"
    launcher_tmp="$INSTALL_ROOT/.spacr-launcher-$$"
    cat > "$launcher_tmp" <<EOF
#!/usr/bin/env sh
exec "$VENV_DIR/bin/python" -I -m spacr.qt "\$@"
EOF
    chmod 755 "$launcher_tmp"
    mv "$launcher_tmp" "$LAUNCHER"
fi

if [[ "$PLATFORM" == "linux" ]]; then
    mkdir -p "$DESKTOP_DIR"
    icon_path="$("$VENV_DIR/bin/python" -I -c \
        "from pathlib import Path; import spacr; d=Path(spacr.__file__).parent/'resources/icons'; p=d/'app_icon.png'; print(p if p.is_file() else d/'logo_spacr.png')")"
    desktop_tmp="$INSTALL_ROOT/.spacr-desktop-$$"
    desktop_comment="$(spacr_say desktop_comment)"
    desktop_exec="$("$VENV_DIR/bin/python" -I - "$VENV_DIR/bin/python" <<'PY'
import sys

def quote_argument(value):
    for character in ('\\', '"', '`', '$'):
        value = value.replace(character, '\\' + character)
    value = value.replace('%', '%%')
    value = value.replace('\\', '\\\\')
    value = value.replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
    return '"' + value + '"'

arguments = ('/bin/sh', '-c', 'exec "$@"', 'spacr', sys.argv[1], '-I', '-m', 'spacr.qt')
print(' '.join(quote_argument(argument) for argument in arguments))
PY
)"
    cat > "$desktop_tmp" <<EOF
[Desktop Entry]
Type=Application
Name=spaCR
Comment=$desktop_comment
Exec=$desktop_exec
Icon=$icon_path
StartupWMClass=spaCR
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF
    chmod 644 "$desktop_tmp"
    mv "$desktop_tmp" "$DESKTOP_DIR/io.github.olafssonlab.spacr.desktop"

    uninstall_path="$INSTALL_ROOT/uninstall-spacr.sh"
    cat > "$uninstall_path" <<EOF
#!/usr/bin/env sh
# Usage: uninstall-spacr.sh [--purge [--yes]]
# --purge also deletes spaCR's caches, backends and user data, after
# listing them and asking (spacr/install_cleanup.py purge).
set -eu
for arg in "\$@"; do
    if [ "\$arg" = "--purge" ]; then
        yes_flag=""
        for other in "\$@"; do
            [ "\$other" = "--yes" ] && yes_flag="--yes"
        done
        "$VENV_DIR/bin/python" -I -m spacr.install_cleanup purge \$yes_flag || true
    fi
done
# SPACR_UNINSTALL_PATH $LAUNCHER
# SPACR_UNINSTALL_PATH $DESKTOP_DIR/io.github.olafssonlab.spacr.desktop
exec "$VENV_DIR/bin/python" -I -m spacr.install_cleanup remove --only-root "$INSTALL_ROOT"
EOF
    chmod 755 "$uninstall_path"
fi

swap_started=0
metadata_touched=0
rm -rf "$old_venv"
if [[ "$PLATFORM" == "linux" ]]; then
    "$cleanup_python" -I "$cleanup_module" register-installer-cache --install-root "$INSTALL_ROOT" || true
fi
echo
spacr_say installed
if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
    spacr_say launcher "$LAUNCHER"
fi
if [[ "$PLATFORM" == "linux" && "$NO_LAUNCH" == "0" ]]; then
    nohup "$LAUNCHER" >/dev/null 2>&1 &
fi
