#!/usr/bin/env bash
# Siphon Studio installer for Ubuntu / Debian.
#
#   curl -fsSL https://raw.githubusercontent.com/Manohar20006/siphon.cpp/v1.0.0/install.sh | bash
#
# Installs build dependencies (apt, asks for sudo), clones siphon.cpp, builds the CPU engine for this
# machine, downloads the Qwen3.6-35B-A3B model (21.1 GiB, resumable) and adds Siphon Studio to the
# applications menu. Re-running it updates an existing install and resumes an interrupted download.
#
# Environment overrides:
#   SIPHON_DIR=~/siphon.cpp      install location (the model is stored in $SIPHON_DIR/models)
#   SIPHON_REF=v1.0.0            git tag or branch to install
#   SIPHON_REPO=<url>            git repository to clone
#   SIPHON_MODEL=/path/x.gguf    use an already downloaded model file instead of downloading
#   SIPHON_SKIP_MODEL=1          install without the model (download it later by re-running)
#   SIPHON_JOBS=N                parallel build jobs
set -euo pipefail

SIPHON_REPO="${SIPHON_REPO:-https://github.com/Manohar20006/siphon.cpp.git}"
SIPHON_REF="${SIPHON_REF:-v1.0.0}"
SIPHON_DIR="${SIPHON_DIR:-$HOME/siphon.cpp}"

MODEL_NAME="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/${MODEL_NAME}"
MODEL_BYTES=22663387424
MODEL_SHA256="0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
PROFILE="siphon-studio/profiles/qwen3.6-moe-tq3.json"

if [ -t 1 ]; then BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RESET=$'\e[0m'
else BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""; fi
step() { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$RESET"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '    %s! %s%s\n' "$YELLOW" "$1" "$RESET"; }
die()  { printf '\n%serror:%s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# stdin is the script itself under `curl | bash`; questions read from the terminal instead.
confirm() {
    local answer=""
    if [ -r /dev/tty ]; then read -r -p "    $1 [y/N] " answer < /dev/tty || true; fi
    [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]
}

gib() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

# ── 1. System checks ──────────────────────────────────────────────────────────────────────────────
step "Checking this computer"
[ "$(uname -s)" = "Linux" ] || die "Siphon Studio runs on Linux (Ubuntu 24.04 or newer)."
[ "$(uname -m)" = "x86_64" ] || die "only x86_64 PCs are supported (found $(uname -m))."
[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root; it asks for sudo when it needs it."
[ -r /etc/os-release ] || die "cannot identify the Linux distribution (/etc/os-release missing)."
. /etc/os-release
case " ${ID:-} ${ID_LIKE:-} " in
    *" ubuntu "*|*" debian "*) info "System: ${PRETTY_NAME:-$ID}" ;;
    *) die "this installer supports Ubuntu and Debian (found ${PRETTY_NAME:-unknown})." ;;
esac
command -v apt-get >/dev/null || die "apt-get not found."

mem_kib=$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)
mem_gib=$(( mem_kib / 1048576 ))
cores=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$cores" -ge 1 ] || cores=$(nproc)
cpu_name=$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo)
info "CPU: ${cpu_name:-unknown} (${cores} cores, $(nproc) threads)"
info "RAM: ${mem_gib} GiB"
if ! grep -qw avx2 /proc/cpuinfo; then
    warn "this CPU has no AVX2; the model will run, but very slowly."
fi
if [ "$mem_kib" -lt 14000000 ]; then
    warn "less than 16 GB of RAM: expect about 1-2 tokens per second."
fi

gpu=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then gpu=1; fi
if [ "$gpu" = 1 ]; then
    warn "an NVIDIA GPU was found, but this installer builds the CPU engine only."
    info "  GPU builds need the CUDA toolkit; see README.md in the install folder."
else
    info "No NVIDIA GPU: Siphon will run the model on the CPU."
fi

parent=$(dirname -- "$SIPHON_DIR")
mkdir -p -- "$parent"
free_bytes=$(( $(df -Pk -- "$parent" | awk 'NR==2 { print $4 }') * 1024 ))
need_bytes=$(( 3 * 1073741824 ))
model_path="$SIPHON_DIR/models/$MODEL_NAME"
if [ -z "${SIPHON_MODEL:-}" ] && [ "${SIPHON_SKIP_MODEL:-0}" != 1 ]; then
    have=0
    if [ -f "$model_path" ]; then have=$(stat -L -c %s -- "$model_path"); fi
    if [ -f "$model_path.part" ]; then have=$(stat -L -c %s -- "$model_path.part"); fi
    need_bytes=$(( need_bytes + MODEL_BYTES - have ))
fi
info "Free disk space at $parent: $(gib "$free_bytes") GiB (needs about $(gib "$need_bytes") GiB)"
[ "$free_bytes" -ge "$need_bytes" ] || die "not enough free disk space."

root_dev=$(df -P -- "$parent" | awk 'NR==2 { print $1 }')
rot=$(lsblk -ndo ROTA "$root_dev" 2>/dev/null || lsblk -ndo ROTA "/dev/$(lsblk -ndo PKNAME "$root_dev" 2>/dev/null)" 2>/dev/null || echo "")
if [ "$rot" = "1" ]; then
    warn "the install folder is on a spinning hard disk. The model is read from disk while it runs,"
    warn "so it will be extremely slow. Use an SSD (set SIPHON_DIR to a folder on the SSD)."
    confirm "Continue anyway?" || exit 1
fi

# ── 2. Packages ───────────────────────────────────────────────────────────────────────────────────
step "Installing system packages (needs sudo)"
packages=(git cmake build-essential curl ca-certificates python3 python3-gi gir1.2-gtk-4.0 gir1.2-webkit-6.0)
sudo apt-get update -qq || die "apt-get update failed."
if ! apt-cache show gir1.2-webkit-6.0 >/dev/null 2>&1; then
    die "this release of ${PRETTY_NAME:-Linux} has no WebKitGTK 6.0 (gir1.2-webkit-6.0). Siphon Studio needs Ubuntu 24.04 / Debian 13 or newer."
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${packages[@]}" \
    || die "package installation failed."
info "Installed: ${packages[*]}"

# ── 3. Source ─────────────────────────────────────────────────────────────────────────────────────
step "Getting siphon.cpp ($SIPHON_REF)"
if [ -d "$SIPHON_DIR/.git" ]; then
    info "Updating $SIPHON_DIR"
    if [ -n "$(git -C "$SIPHON_DIR" status --porcelain --untracked-files=no -- . ":!$PROFILE")" ]; then
        die "$SIPHON_DIR has local changes; move them away or choose another SIPHON_DIR."
    fi
    git -C "$SIPHON_DIR" fetch --quiet --depth 1 origin "+refs/tags/$SIPHON_REF:refs/tags/$SIPHON_REF" 2>/dev/null \
        || git -C "$SIPHON_DIR" fetch --quiet --depth 1 origin "$SIPHON_REF"
    git -C "$SIPHON_DIR" stash push --quiet -- "$PROFILE" 2>/dev/null || true
    git -C "$SIPHON_DIR" -c advice.detachedHead=false checkout --quiet FETCH_HEAD 2>/dev/null \
        || git -C "$SIPHON_DIR" -c advice.detachedHead=false checkout --quiet "$SIPHON_REF"
    git -C "$SIPHON_DIR" stash pop --quiet 2>/dev/null || true
elif [ -e "$SIPHON_DIR" ] && [ -n "$(ls -A -- "$SIPHON_DIR" 2>/dev/null)" ]; then
    die "$SIPHON_DIR exists and is not a siphon.cpp checkout; choose another SIPHON_DIR."
else
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$SIPHON_REF" "$SIPHON_REPO" "$SIPHON_DIR" \
        || die "could not clone $SIPHON_REPO ($SIPHON_REF)."
fi
info "Source: $SIPHON_DIR ($(git -C "$SIPHON_DIR" rev-parse --short HEAD))"

# ── 4. Build ──────────────────────────────────────────────────────────────────────────────────────
step "Building the CPU engine (5-15 minutes)"
jobs="${SIPHON_JOBS:-}"
if [ -z "$jobs" ]; then
    jobs=$(nproc)
    by_mem=$(( mem_gib / 2 )); [ "$by_mem" -ge 1 ] || by_mem=1   # ~2 GiB per compiler process
    [ "$jobs" -le "$by_mem" ] || jobs=$by_mem
fi
build_log="$SIPHON_DIR/build-cpu.log"
info "Using $jobs parallel jobs; log: $build_log"
{
    cmake -S "$SIPHON_DIR" -B "$SIPHON_DIR/build-cpu" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=OFF
    cmake --build "$SIPHON_DIR/build-cpu" --target llama-server -j "$jobs"
} > "$build_log" 2>&1 || { tail -n 25 "$build_log" >&2; die "the build failed; the full log is $build_log"; }
server="$SIPHON_DIR/build-cpu/bin/llama-server"
"$server" --version > /dev/null 2>&1 || die "the built server does not start; see $build_log"
info "Built $server"

# ── 5. Model ──────────────────────────────────────────────────────────────────────────────────────
step "Model: $MODEL_NAME"
mkdir -p -- "$SIPHON_DIR/models"
verify_model() {
    local file="$1" size
    size=$(stat -L -c %s -- "$file")
    [ "$size" = "$MODEL_BYTES" ] || { warn "size is $size bytes, expected $MODEL_BYTES."; return 1; }
    info "Checking the file (about a minute)..."
    echo "$MODEL_SHA256  $file" | sha256sum --check --status || { warn "checksum mismatch."; return 1; }
}
if [ -n "${SIPHON_MODEL:-}" ]; then
    [ -f "$SIPHON_MODEL" ] || die "SIPHON_MODEL does not exist: $SIPHON_MODEL"
    verify_model "$SIPHON_MODEL" || die "$SIPHON_MODEL is not the expected $MODEL_NAME."
    ln -sfn -- "$(readlink -f -- "$SIPHON_MODEL")" "$model_path"
    info "Linked $model_path -> $SIPHON_MODEL"
elif [ "${SIPHON_SKIP_MODEL:-0}" = 1 ]; then
    warn "skipped (SIPHON_SKIP_MODEL=1). Re-run the installer to download it."
elif [ -f "$model_path" ] && [ "$(stat -L -c %s -- "$model_path")" = "$MODEL_BYTES" ]; then
    info "Already downloaded."
else
    info "Downloading 21.1 GiB from Hugging Face. If this stops, re-run the installer to resume."
    curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - --progress-bar \
        -o "$model_path.part" "$MODEL_URL" || die "download failed; re-run the installer to resume."
    verify_model "$model_path.part" || {
        mv -f -- "$model_path.part" "$model_path.bad"
        die "the download is damaged (kept as $model_path.bad). Delete it and re-run the installer."
    }
    mv -f -- "$model_path.part" "$model_path"
    info "Saved $model_path"
fi

# ── 6. Settings for this machine ──────────────────────────────────────────────────────────────────
step "Configuring Siphon Studio"
threads=$(( cores < 8 ? cores : 8 ))   # one per physical core; more rarely helps and hybrid E-cores slow it down
python3 - "$SIPHON_DIR/$PROFILE" "$threads" <<'PY'
import json, sys
path, threads = sys.argv[1], sys.argv[2]
with open(path) as f:
    profile = json.load(f)
env = profile["env"]
env["SIPHON_EXECUTION_MODE"] = "auto"   # GPU when one is usable, otherwise CPU
env["CPU_THREADS"] = threads
with open(path, "w") as f:
    json.dump(profile, f, indent=2)
    f.write("\n")
PY
info "CPU threads: $threads"
python3 -c 'import gi; gi.require_version("Gtk", "4.0"); gi.require_version("WebKit", "6.0"); from gi.repository import Gtk, WebKit' \
    || die "the desktop libraries (GTK 4 / WebKitGTK 6) do not load."
# Hide bundled model profiles whose model file is not on this machine, as Studio's "Remove" button would.
PYTHONPATH="$SIPHON_DIR/siphon-studio" python3 - <<'PY' || die "could not prepare the Studio model list."
from siphon.core.profiles import discover
from siphon.desktop.app import _profile_map, _profile_model_available
from siphon.desktop.catalogue import load_excluded_model_keys, save_excluded_model_keys
profiles = discover()
missing = {key for key, name in _profile_map(profiles).items() if not _profile_model_available(profiles[name])}
hidden = load_excluded_model_keys()
if missing - hidden:
    save_excluded_model_keys(hidden | missing)
PY

bin_dir="$HOME/.local/bin"
app_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p -- "$bin_dir" "$app_dir"
ln -sfn -- "$SIPHON_DIR/siphon-studio/bin/siphon-studio" "$bin_dir/siphon-studio"
ln -sfn -- "$SIPHON_DIR/siphon-studio/bin/siphon" "$bin_dir/siphon"
cat > "$app_dir/com.siphon.studio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Siphon Studio
Comment=Run Qwen3.6 locally with siphon.cpp
Exec=$SIPHON_DIR/siphon-studio/bin/siphon-studio
Path=$SIPHON_DIR
Icon=utilities-terminal
Terminal=false
Categories=Development;
Keywords=LLM;AI;chat;Siphon;
StartupNotify=true
DESKTOP
update-desktop-database "$app_dir" >/dev/null 2>&1 || true
info "Added Siphon Studio to the applications menu and $bin_dir/siphon-studio"

# ── Done ──────────────────────────────────────────────────────────────────────────────────────────
step "${GREEN}Siphon Studio is installed${RESET}"
info "Open it from the applications menu (search \"Siphon\"), or run: siphon-studio"
case ":$PATH:" in *":$bin_dir:"*) ;; *) info "${DIM}(log out and back in once so $bin_dir is on your PATH)${RESET}" ;; esac
info "Pick the Qwen3.6 model and press Start. The first answer takes longer while the model warms up."
info "To update later, run the same install command again."
info "To uninstall: rm -rf \"$SIPHON_DIR\" \"$bin_dir/siphon-studio\" \"$bin_dir/siphon\" \"$app_dir/com.siphon.studio.desktop\""
