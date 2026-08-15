#!/usr/bin/env bash
#
# Small online installer for Linux and macOS.
#
# The file itself contains no Python runtime or scientific dependencies. It
# downloads a pinned uv bootstrap over TLS, lets uv install a private CPython
# 3.12 runtime, then resolves spaCR and the appropriate PyTorch backend.

set -Eeuo pipefail

# @SPACR_INSTALLER_MESSAGES_BEGIN@
SPACR_INSTALLER_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=generated/installer_messages.sh
source "$SPACR_INSTALLER_DIR/generated/installer_messages.sh"
# @SPACR_INSTALLER_MESSAGES_END@

UV_VERSION="0.11.32"
PYTHON_VERSION="3.12"
DEFAULT_SPACR_VERSION="@SPACR_VERSION@"
DEFAULT_EXTRAS="qt"
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
SIGN_IN_NOW=0

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
    printf '  --dry-run               %s\n' "$(spacr_say help_dry_run)"
    printf '  --share-diagnostics     Include redacted logs in report previews.\n'
    printf '  --report-issues         Show the public GitHub report action.\n'
    printf '  --sign-in-now           Open account setup on first launch.\n'
    printf '  --consent-collected     Record that these choices were reviewed.\n'
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
        --report-issues)
            REPORT_ISSUES=1
            shift
            ;;
        --sign-in-now)
            SIGN_IN_NOW=1
            shift
            ;;
        --consent-collected)
            CONSENT_COLLECTED=1
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

# Choose the accelerated wheel only when the machine identifies a supported
# accelerator. An explicit environment variable or --torch-backend always
# wins, which keeps unattended and reproducible installs possible.
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
    if [[ "$DETECTED_ACCELERATOR" == "nvidia" || "$DETECTED_ACCELERATOR" == "apple-silicon" ]]; then
        TORCH_BACKEND="auto"
    else
        TORCH_BACKEND="cpu"
    fi
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
    )
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
# non-interactive, and leave all three choices off for the app to ask later.
if [[ "$CONSENT_COLLECTED" == "0" && -t 0 && "$DRY_RUN" != "1" ]]; then
    printf '\nspaCR privacy and account setup\n'
    printf '%s\n' \
        'Crash reports go to the PUBLIC spaCR GitHub repository. They are world-readable, indexed, and cannot be reliably unpublished.' \
        'Every report is redacted, shown in an editable preview, and sent only when you press Send for that report.' \
        'Account setup runs the official GitHub, Claude, Codex (GPT), and Gemini login tools; spaCR never stores their passwords or tokens.' \
        'All choices are optional, default off, and can be changed later in Preferences.'
    ask_yes_no 'Include redacted diagnostic logs in report previews?' && SHARE_DIAGNOSTICS=1
    ask_yes_no 'Enable the public GitHub issue-report action?' && REPORT_ISSUES=1
    ask_yes_no 'Open GitHub, Claude, GPT/Codex, and Gemini setup on first launch?' && SIGN_IN_NOW=1
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
        PACKAGE_SPEC="spacr[$DEFAULT_EXTRAS]"
    else
        PACKAGE_SPEC="spacr[$DEFAULT_EXTRAS]==$DEFAULT_SPACR_VERSION"
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

if [[ "$DRY_RUN" == "1" ]]; then
    spacr_say dry_download "$UV_INSTALL_URL"
    spacr_say dry_create "$VENV_DIR"
    if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
        spacr_say dry_launcher "$LAUNCHER"
    fi
    exit 0
fi

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        spacr_say required_command "$1" >&2
        exit 3
    fi
}

require_command curl
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

    spacr_say installing_linux_deps
    if command -v apt-get >/dev/null 2>&1; then
        "${elevate[@]}" apt-get update
        "${elevate[@]}" apt-get install --no-install-recommends -y \
            libegl1 libgl1 libxkbcommon0 libdbus-1-3 libpulse0 \
            libx11-xcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
            libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
            libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
            libxkbcommon-x11-0 libnss3 libxcomposite1 libxcursor1 \
            libxdamage1 libxi6 libxtst6 libsm6 libxext6 libxrender1 ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        "${elevate[@]}" dnf install -y \
            mesa-libGL mesa-libEGL libxkbcommon libxkbcommon-x11 \
            libxcb xcb-util-cursor xcb-util-image xcb-util-keysyms \
            xcb-util-renderutil dbus-libs pulseaudio-libs nss ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
        "${elevate[@]}" zypper --non-interactive install \
            Mesa-libGL1 Mesa-libEGL1 libxkbcommon0 libxkbcommon-x11-0 \
            libxcb1 libpulse0 libnss3 ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        "${elevate[@]}" pacman -S --needed --noconfirm \
            libglvnd libxkbcommon-x11 libxcb xcb-util-cursor \
            xcb-util-image xcb-util-keysyms xcb-util-renderutil \
            dbus libpulse nss ffmpeg
    else
        spacr_say unknown_package_manager
    fi
}

mkdir -p "$BOOTSTRAP_DIR" "$PYTHON_DIR" "$CACHE_DIR"
INSTALL_LOG="$INSTALL_ROOT/install.log"
touch "$INSTALL_LOG"
exec > >(tee -a "$INSTALL_LOG") 2>&1
spacr_say detailed_log "$INSTALL_LOG"
install_linux_system_dependencies
installer_tmp="$(mktemp "${TMPDIR:-/tmp}/spacr-uv-installer.XXXXXX")"
stage_venv="$INSTALL_ROOT/.venv-staging-$$"
stage_profile="$INSTALL_ROOT/.install-profile-staging-$$.json"
cleanup() {
    rm -f "$installer_tmp"
    rm -f "$stage_profile"
    if [[ -d "$stage_venv" ]]; then
        rm -rf "$stage_venv"
    fi
}
trap cleanup EXIT

spacr_say downloading_uv
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --retry 3 --retry-all-errors \
    "$UV_INSTALL_URL" --output "$installer_tmp"
UV_UNMANAGED_INSTALL="$BOOTSTRAP_DIR" UV_NO_MODIFY_PATH=1 \
    sh "$installer_tmp"
if [[ ! -x "$UV_BIN" ]]; then
    spacr_say uv_missing "$UV_BIN" >&2
    exit 5
fi

export UV_PYTHON_INSTALL_DIR="$PYTHON_DIR"
export UV_CACHE_DIR="$CACHE_DIR"
export UV_SYSTEM_CERTS=true

spacr_say downloading_python "$PYTHON_VERSION"
"$UV_BIN" python install "$PYTHON_VERSION" --managed-python --no-bin

spacr_say creating_environment
rm -rf "$stage_venv"
"$UV_BIN" venv "$stage_venv" \
    --python "$PYTHON_VERSION" --managed-python --relocatable

stage_python="$stage_venv/bin/python"

spacr_say downloading_dependencies
"$UV_BIN" pip install \
    --python "$stage_python" \
    --torch-backend "$TORCH_BACKEND" \
    "$PACKAGE_SPEC" \
    "${RESOLVER_GUARDS[@]}"

spacr_say validating_install
"$UV_BIN" pip check --python "$stage_python"
QT_QPA_PLATFORM=offscreen "$stage_python" -I -c \
    "import spacr, PySide6, torch; import numpy as np; assert torch.from_numpy(np.zeros(1, dtype=np.float32)).numel() == 1; print('spaCR', spacr.__version__, '| torch', torch.__version__, '| numpy', np.__version__)"
if [[ "$TORCH_BACKEND" != "cpu" && "$DETECTED_ACCELERATOR" == "nvidia" ]]; then
    "$stage_python" -I -c \
        "import torch; assert torch.cuda.is_available(), 'GPU install selected but CUDA is unavailable'"
elif [[ "$TORCH_BACKEND" != "cpu" && "$DETECTED_ACCELERATOR" == "apple-silicon" ]]; then
    "$stage_python" -I -c \
        "import torch; assert torch.backends.mps.is_available(), 'Apple GPU selected but MPS is unavailable'"
fi

"$stage_python" -I -m spacr.install_profile \
    --path "$stage_profile" \
    --requested "$TORCH_BACKEND" \
    --detected "$DETECTED_ACCELERATOR" \
    --consent-collected "$CONSENT_COLLECTED" \
    --share-diagnostics "$SHARE_DIAGNOSTICS" \
    --report-issues "$REPORT_ISSUES" \
    --sign-in-now "$SIGN_IN_NOW"

old_venv="$INSTALL_ROOT/.venv-previous"
rm -rf "$old_venv"
if [[ -d "$VENV_DIR" ]]; then
    mv "$VENV_DIR" "$old_venv"
fi
mv "$stage_venv" "$VENV_DIR"
rm -rf "$old_venv"
mv "$stage_profile" "$INSTALL_ROOT/install-profile.json"

if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
    mkdir -p "$USER_BIN_DIR"
    launcher_tmp="$INSTALL_ROOT/.spacr-launcher-$$"
    cat > "$launcher_tmp" <<EOF
#!/usr/bin/env sh
exec "$VENV_DIR/bin/python" -m spacr.qt "\$@"
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
    cat > "$desktop_tmp" <<EOF
[Desktop Entry]
Type=Application
Name=spaCR
Comment=$desktop_comment
Exec=$LAUNCHER
Icon=$icon_path
StartupWMClass=spaCR
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF
    chmod 644 "$desktop_tmp"
    mv "$desktop_tmp" "$DESKTOP_DIR/io.github.olafssonlab.spacr.desktop"

    uninstall_path="$INSTALL_ROOT/uninstall-spacr.sh"
    removed_message="$(spacr_say removed)"
    cat > "$uninstall_path" <<EOF
#!/usr/bin/env sh
set -eu
rm -f "$LAUNCHER"
rm -f "$DESKTOP_DIR/io.github.olafssonlab.spacr.desktop"
rm -rf "$INSTALL_ROOT"
echo "$removed_message"
EOF
    chmod 755 "$uninstall_path"
fi

echo
spacr_say installed
if [[ "$NO_COMMAND_LAUNCHER" == "0" ]]; then
    spacr_say launcher "$LAUNCHER"
fi
if [[ "$PLATFORM" == "linux" && "$NO_LAUNCH" == "0" ]]; then
    nohup "$LAUNCHER" >/dev/null 2>&1 &
fi
