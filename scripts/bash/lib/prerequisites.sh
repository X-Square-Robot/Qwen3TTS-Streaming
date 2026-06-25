#!/bin/bash
# ===========================================================================
#  prerequisites.sh — System prerequisites checking
#
#  Functions: check_prerequisites, ensure_git_lfs, detect_cuda_version,
#             cuda_to_torch_tag, _flash_attn_compatible_tag,
#             check_disk_space, check_gpu_memory, check_docker
#  Depends:   lib/logging.sh, lib/utils.sh, lib/docker.sh
# ===========================================================================

[[ -n "${_LIB_PREREQUISITES_LOADED:-}" ]] && return 0
_LIB_PREREQUISITES_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"
source "${_LIB_DIR}/docker.sh"

# ---------------------------------------------------------------------------
#  detect_cuda_version
#  Echoes the CUDA toolkit version (e.g. "12.6").  Returns 1 if not found.
# ---------------------------------------------------------------------------
detect_cuda_version() {
    local cuda_ver=""

    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version 2>/dev/null \
            | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    fi

    if [ -z "$cuda_ver" ] && command -v nvidia-smi &>/dev/null; then
        cuda_ver=$(nvidia-smi 2>/dev/null \
            | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    fi

    if [ -z "$cuda_ver" ]; then
        return 1
    fi
    echo "$cuda_ver"
}

# ---------------------------------------------------------------------------
#  cuda_to_torch_tag <cuda_version>
#  Maps a CUDA version to the PyTorch index URL tag (e.g. "12.4" → "cu124").
#  Pins to the nearest PyTorch-supported CUDA build.
#  See: https://download.pytorch.org/whl/
# ---------------------------------------------------------------------------
cuda_to_torch_tag() {
    local cuda_ver="$1"
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    if [ "$major" -le 11 ]; then
        echo "cu118"
    elif [ "$major" -eq 12 ]; then
        if   [ "$minor" -le 1 ]; then echo "cu121"
        elif [ "$minor" -le 4 ]; then echo "cu124"
        elif [ "$minor" -le 6 ]; then echo "cu126"
        else                          echo "cu128"
        fi
    elif [ "$major" -eq 13 ]; then
        echo "cu130"
    else
        echo "cu${major}${minor}"
    fi
}

# ---------------------------------------------------------------------------
#  flash-attn compatibility constants
#  Update these when a new flash-attn release adds support for newer CUDA/torch.
#  Wheel matrix: https://github.com/Dao-AILab/flash-attention/releases
# ---------------------------------------------------------------------------
_FLASH_ATTN_VERSION="2.8.3"
_FLASH_ATTN_MAX_CUDA="12.8"

# ---------------------------------------------------------------------------
#  _flash_attn_compatible_tag <cuda_version>
#  Given a system CUDA version, checks whether the PyTorch CUDA tag that
#  cuda_to_torch_tag would select has a flash-attn pre-built wheel.
#
#  flash-attn 2.8.3 ships wheels for: cu118 cu121 cu122 cu124 cu126 cu128
#  The max supported CUDA tag is cu128 (i.e. CUDA 12.8).
#
#  If the mapped tag is within range, echoes it and returns 0.
#  If the mapped tag exceeds flash-attn's max, echoes the max compatible
#  tag (cu128) as the downgrade candidate and returns 1.
# ---------------------------------------------------------------------------
_flash_attn_compatible_tag() {
    local cuda_ver="$1"
    local tag
    tag=$(cuda_to_torch_tag "$cuda_ver")

    local max_tag
    max_tag=$(cuda_to_torch_tag "$_FLASH_ATTN_MAX_CUDA")

    local tag_num max_num
    tag_num=$(echo "$tag" | sed 's/^cu//')
    max_num=$(echo "$max_tag" | sed 's/^cu//')

    if [ "$tag_num" -le "$max_num" ]; then
        echo "$tag"
        return 0
    fi
    echo "$max_tag"
    return 1
}

# ---------------------------------------------------------------------------
#  ensure_git_lfs
#  Installs git-lfs if missing, then runs `git lfs install`.
# ---------------------------------------------------------------------------
ensure_git_lfs() {
    if command -v git-lfs &>/dev/null; then
        git lfs install --skip-repo &>/dev/null
        return 0
    fi

    log_step "Installing git-lfs..."

    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
    elif command -v yum &>/dev/null; then
        sudo yum install -y git-lfs
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y git-lfs
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm git-lfs
    elif command -v brew &>/dev/null; then
        brew install git-lfs
    else
        log_error "Cannot auto-install git-lfs: no supported package manager found"
        log_error "Please install git-lfs manually: https://git-lfs.com"
        return 1
    fi

    if ! command -v git-lfs &>/dev/null; then
        log_error "git-lfs installation failed"
        return 1
    fi

    git lfs install --skip-repo &>/dev/null
    log_info "git-lfs installed"
}

# ---------------------------------------------------------------------------
#  ensure_sox
#  Installs SoX (Sound eXchange) if missing. Required by Qwen3-TTS 25Hz tokenizer
#  for audio processing. Non-fatal: logs warning if install fails.
# ---------------------------------------------------------------------------
ensure_sox() {
    if command -v sox &>/dev/null; then
        log_info "sox: $(command -v sox)"
        return 0
    fi

    log_step "Installing SoX (required by Qwen3-TTS audio processing)..."

    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq sox libsox-dev 2>/dev/null || {
            log_warn "SoX install failed. Install manually: sudo apt install sox libsox-dev"
            return 1
        }
    elif command -v yum &>/dev/null; then
        sudo yum install -y sox sox-devel 2>/dev/null || {
            log_warn "SoX install failed. Install manually: sudo yum install sox sox-devel"
            return 1
        }
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y sox sox-devel 2>/dev/null || {
            log_warn "SoX install failed. Install manually: sudo dnf install sox sox-devel"
            return 1
        }
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm sox 2>/dev/null || {
            log_warn "SoX install failed. Install manually: sudo pacman -S sox"
            return 1
        }
    elif command -v brew &>/dev/null; then
        brew install sox 2>/dev/null || {
            log_warn "SoX install failed. Install manually: brew install sox"
            return 1
        }
    else
        log_warn "Cannot auto-install SoX: no supported package manager found"
        log_warn "Install manually: https://sox.sourceforge.net/"
        return 1
    fi

    if ! command -v sox &>/dev/null; then
        log_warn "SoX not found after install attempt"
        return 1
    fi
    log_info "SoX installed: $(command -v sox)"
    return 0
}

# ---------------------------------------------------------------------------
#  check_disk_space <path> <required_gb>
#  Returns 1 if available space is below the threshold.
# ---------------------------------------------------------------------------
check_disk_space() {
    local path="${1:-.}"
    local required_gb="${2:-20}"

    local available_gb
    available_gb=$(df -BG "$path" 2>/dev/null \
        | awk 'NR==2 {gsub(/G/,"",$4); print $4}')

    if [ -z "$available_gb" ]; then
        log_warn "Could not determine disk space at $path"
        return 0
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Disk space insufficient: ${available_gb}GB available, need ${required_gb}GB (at $path)"
        return 1
    fi
    log_info "Disk space: ${available_gb}GB available (need ${required_gb}GB)"
}

# ---------------------------------------------------------------------------
#  check_gpu_memory [required_mb]
#  Warns if GPU memory is below the threshold (default 8 GB).
# ---------------------------------------------------------------------------
check_gpu_memory() {
    local required_mb="${1:-8000}"

    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi

    local total_mb
    total_mb=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

    if [ -z "$total_mb" ]; then
        log_warn "Could not determine GPU memory"
        return 0
    fi

    if [ "$total_mb" -lt "$required_mb" ]; then
        log_warn "GPU memory: ${total_mb}MB (recommended: ${required_mb}MB+)"
        log_warn "Small GPU may limit batch size and performance"
    else
        log_info "GPU memory: ${total_mb}MB"
    fi
}

# ---------------------------------------------------------------------------
#  check_docker
#  Checks Docker installation, daemon status, NVIDIA GPU runtime, and
#  prints the recommended NGC container for Phase B (engine build).
#  Returns 0 with warnings (Docker is needed for Phase B + deployment,
#  not for Phase A model export).
# ---------------------------------------------------------------------------
check_docker() {
    log_step "Checking Docker environment..."

    if ! command -v docker &>/dev/null; then
        log_warn "Docker 未安装 — Phase B (engine build) 和部署 Triton 需要 Docker"
        log_warn "安装指南: https://docs.docker.com/engine/install/"
        return 0
    fi
    log_info "docker: $(docker --version 2>/dev/null | head -1)"

    if ! docker info &>/dev/null 2>&1; then
        log_warn "Docker daemon 未运行或当前用户无权限"
        log_warn "尝试: sudo systemctl start docker && sudo usermod -aG docker \$USER"
        return 0
    fi
    log_info "Docker daemon: running"

    local gpu_runtime_ok=false
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        gpu_runtime_ok=true
    elif command -v nvidia-container-cli &>/dev/null; then
        gpu_runtime_ok=true
    elif [ -f /etc/nvidia-container-runtime/config.toml ]; then
        gpu_runtime_ok=true
    fi

    if $gpu_runtime_ok; then
        log_info "NVIDIA Container Toolkit: detected"
    else
        log_warn "NVIDIA Container Toolkit 未检测到 — Phase B 和 GPU 推理需要此组件"
        log_warn "安装指南: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    # Show recommended NGC container (non-fatal, informational only)
    print_container_recommendation
}

# ---------------------------------------------------------------------------
#  check_prerequisites
#  Runs all prerequisite checks.  Returns 1 on hard failure.
# ---------------------------------------------------------------------------
check_prerequisites() {
    local errors=0

    log_step "Checking system prerequisites..."

    for cmd in git curl; do
        if command -v "$cmd" &>/dev/null; then
            log_info "$cmd: $(command -v "$cmd")"
        else
            log_error "$cmd not found (required)"
            errors=$((errors + 1))
        fi
    done

    if command -v python3 &>/dev/null; then
        log_info "python3: $(python3 --version 2>&1) ($(command -v python3))"
    else
        log_warn "python3 not found (will be provided by miniforge if installed later)"
    fi

    if check_gpu; then
        check_gpu_memory 8000
        local cuda_ver
        if cuda_ver=$(detect_cuda_version); then
            log_info "CUDA version: $cuda_ver (PyTorch tag: $(cuda_to_torch_tag "$cuda_ver"))"
        else
            log_warn "CUDA toolkit not detected (nvcc not found)"
            log_warn "GPU inference requires CUDA — install the CUDA toolkit"
        fi
    else
        log_warn "No NVIDIA GPU detected — inference will be CPU-only"
    fi

    check_docker

    ensure_git_lfs || errors=$((errors + 1))

    # SoX: required by Qwen3-TTS 25Hz tokenizer. Non-fatal (some variants use 12Hz without it)
    ensure_sox || log_warn "SoX not installed — install manually: sudo apt install sox libsox-dev"

    check_disk_space "." 20 || errors=$((errors + 1))

    if [ "$errors" -gt 0 ]; then
        log_error "Prerequisites check failed ($errors error(s))"
        return 1
    fi

    log_info "All prerequisites satisfied"
}
