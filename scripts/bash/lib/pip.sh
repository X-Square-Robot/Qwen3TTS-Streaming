#!/bin/bash
# ===========================================================================
#  pip.sh — pip install helpers
#
#  Functions: pip_install, pip_install_requirements, install_torch_cuda,
#             install_flash_attn, install_qwen3_tts,
#             install_safetensors, install_onnx_export_deps,
#             validate_python_env
#  Depends:   lib/logging.sh, lib/prerequisites.sh, lib/network.sh
# ===========================================================================

[[ -n "${_LIB_PIP_LOADED:-}" ]] && return 0
_LIB_PIP_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/prerequisites.sh"
source "${_LIB_DIR}/network.sh"

# ---------------------------------------------------------------------------
#  pip_install <package> [package...]
# ---------------------------------------------------------------------------
pip_install() {
    if [ $# -eq 0 ]; then
        log_error "Usage: pip_install <package> [package...]"
        return 1
    fi
    log_step "pip install $*"
    python3 -m pip install --upgrade "$@" \
        || { log_error "pip install failed"; return 1; }
}

# ---------------------------------------------------------------------------
#  pip_install_requirements [requirements_file]
#  Defaults to requirements.txt in the current directory.
# ---------------------------------------------------------------------------
pip_install_requirements() {
    local req_file="${1:-requirements.txt}"
    if [ ! -f "$req_file" ]; then
        log_error "Requirements file not found: $req_file"
        return 1
    fi
    log_step "Installing from $req_file..."
    python3 -m pip install --upgrade -r "$req_file" \
        || { log_error "pip install from $req_file failed"; return 1; }
}

# ---------------------------------------------------------------------------
#  _resolve_tensorrt_pip_package [explicit_package]
#  Chooses the CUDA-specific TensorRT pip package name when we already know
#  the target CUDA tag from the NGC / PyTorch compatibility plan.
# ---------------------------------------------------------------------------
_resolve_tensorrt_pip_package() {
    local explicit="${1:-${STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE:-}}"
    if [ -n "$explicit" ]; then
        printf '%s\n' "$explicit"
        return 0
    fi

    local cuda_tag="${STANDALONE_ENGINE_TENSORRT_CUDA_TAG:-${ENGINE_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-${TRITON_PYTORCH_CUDA_TAG:-}}}}"
    case "$cuda_tag" in
        cu13*) printf '%s\n' 'tensorrt-cu13' ;;
        cu12*) printf '%s\n' 'tensorrt-cu12' ;;
        *)     printf '%s\n' 'tensorrt' ;;
    esac
}

# ---------------------------------------------------------------------------
#  _resolve_tensorrt_runtime_package
#  Maps CUDA-specific TensorRT package families to the CUDA runtime package
#  actually published on PyPI.  CUDA 13 currently ships a deprecated
#  nvidia-cuda-runtime-cu13 placeholder, so we must install nvidia-cuda-runtime
#  directly and then install TRT packages with --no-deps.
# ---------------------------------------------------------------------------
_resolve_tensorrt_runtime_package() {
    local package="${1:-}"
    case "$package" in
        tensorrt-cu13) printf '%s\n' 'nvidia-cuda-runtime' ;;
        *)             printf '%s\n' '' ;;
    esac
}

# ---------------------------------------------------------------------------
#  _resolve_tensorrt_runtime_version
#  Returns a compatible CUDA runtime wheel version for the selected TRT family.
# ---------------------------------------------------------------------------
_resolve_tensorrt_runtime_version() {
    local package="${1:-}"
    case "$package" in
        tensorrt-cu13) printf '%s\n' '13.0.96' ;;
        *)             printf '%s\n' '' ;;
    esac
}

# ---------------------------------------------------------------------------
#  _install_tensorrt_cu13_fallback <python_exe> <version> <common_args...>
#  Recovery path for TensorRT CUDA 13 packaging on PyPI.  Current 10.13.3.9
#  metadata pulls the deprecated nvidia-cuda-runtime-cu13 source package, so
#  we install the supported runtime wheel first and then install the TRT meta /
#  libs / bindings packages without dependency resolution.
# ---------------------------------------------------------------------------
_install_tensorrt_cu13_fallback() {
    local pyexe="$1"
    local want="$2"
    shift 2
    local common_args=("$@")
    local runtime_pkg runtime_ver
    runtime_pkg=$(_resolve_tensorrt_runtime_package "tensorrt-cu13")
    runtime_ver=$(_resolve_tensorrt_runtime_version "tensorrt-cu13")
    [ -n "$runtime_pkg" ] || return 1
    [ -n "$runtime_ver" ] || return 1

    log_warn "Detected deprecated nvidia-cuda-runtime-cu13 dependency; using CUDA 13 fallback install path"

    "$pyexe" "${common_args[@]}" \
        --index-url "https://pypi.org/simple" \
        --extra-index-url "https://pypi.nvidia.com" \
        --trusted-host "pypi.org" \
        --trusted-host "pypi.nvidia.com" \
        "${runtime_pkg}==${runtime_ver}" \
        || return 1

    "$pyexe" "${common_args[@]}" \
        --no-deps \
        --index-url "https://pypi.org/simple" \
        --extra-index-url "https://pypi.nvidia.com" \
        --trusted-host "pypi.org" \
        --trusted-host "pypi.nvidia.com" \
        "tensorrt==${want}" \
        "tensorrt_cu13_bindings==${want}" \
        "tensorrt_cu13_libs==${want}" \
        "tensorrt-cu13==${want}" \
        || return 1

    return 0
}

# ---------------------------------------------------------------------------
#  install_tensorrt_for_python <python_bin> [version]
#  Installs/pins the TensorRT Python package for a specific interpreter.
#  Prefers CUDA-specific package names from the official TensorRT pip docs and
#  forces the public PyPI index so mirror overrides do not redirect us to a
#  stale university mirror. NVIDIA's package index stays configured as the
#  extra source used by TensorRT's Python packaging flow.
# ---------------------------------------------------------------------------
install_tensorrt_for_python() {
    local pybin="${1:-python3}"
    local want="${2:-${STANDALONE_ENGINE_TENSORRT_PIP_VERSION:-10.15.1.29}}"
    local pyexe=""
    local package=""

    if [[ "$pybin" == */* ]]; then
        if [ -x "$pybin" ]; then
            pyexe="$pybin"
        fi
    else
        pyexe=$(command -v "$pybin" 2>/dev/null || true)
    fi

    if [ -z "$pyexe" ] || [ ! -x "$pyexe" ]; then
        log_error "Python interpreter not found or not executable: $pybin"
        return 1
    fi

    package=$(_resolve_tensorrt_pip_package)

    local have=""
    have=$("$pyexe" -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || echo "")
    if [ -n "$have" ] && [ "$have" = "$want" ]; then
        log_info "TensorRT $want already installed for $pybin, skipping"
        return 0
    fi

    local timeout="${TENSORRT_PIP_TIMEOUT:-15}"
    local retries="${TENSORRT_PIP_RETRIES:-1}"
    local common_args=(
        -m pip install --upgrade
        --disable-pip-version-check
        --no-cache-dir
        --timeout "$timeout"
        --retries "$retries"
    )

    log_step "Installing TensorRT $want for $pyexe ..."
    log_info "Using TensorRT package ${package}==${want} from official PyPI with NVIDIA index enabled"

    if [ "$package" = "tensorrt-cu13" ]; then
        if _install_tensorrt_cu13_fallback "$pyexe" "$want" "${common_args[@]}"; then
            log_info "TensorRT $want installed for $pyexe via CUDA 13 fallback path"
            return 0
        fi
    fi

    local install_output=""
    if install_output=$(NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=0 \
        "$pyexe" "${common_args[@]}" \
        --index-url "https://pypi.org/simple" \
        --extra-index-url "https://pypi.nvidia.com" \
        --trusted-host "pypi.org" \
        --trusted-host "pypi.nvidia.com" \
        "${package}==${want}" 2>&1); then
        printf '%s\n' "$install_output"
        log_info "TensorRT $want installed for $pyexe"
        return 0
    fi
    printf '%s\n' "$install_output"

    log_error "TensorRT install failed for $pyexe (${package}==${want})"
    log_error "If the required TensorRT Python package is no longer published for this NGC tag, use --gateway engine-docker or --gateway triton."
    return 1
}

# ---------------------------------------------------------------------------
#  install_torch_cuda [cuda_tag_or_version]
#  Installs PyTorch + torchaudio with the correct CUDA build.
#
#  Argument can be:
#    - A CUDA tag like "cu130", "cu128" (used directly as index URL suffix)
#    - A CUDA version like "13.0", "12.8" (auto-converted via cuda_to_torch_tag)
#    - Omitted: auto-detects CUDA version from the system
#
#  When called from setup_env.sh with an env_plan, the pre-resolved cuda_tag
#  is passed in and the wheel index reachability has already been validated.
#
#  Skips if PyTorch with CUDA is already importable.
# ---------------------------------------------------------------------------
install_torch_cuda() {
    local arg="${1:-}"

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local existing
        existing=$(python3 -c "import torch; print(torch.__version__)")
        log_info "PyTorch $existing (CUDA) already installed, skipping"
        return 0
    fi

    local tag=""
    if [[ "$arg" == cu* ]]; then
        tag="$arg"
    elif [ -n "$arg" ]; then
        tag=$(cuda_to_torch_tag "$arg")
    else
        local cuda_ver
        cuda_ver=$(detect_cuda_version) || true
        if [ -z "$cuda_ver" ]; then
            log_warn "No CUDA detected — installing CPU-only PyTorch"
            pip_install torch torchaudio
            return $?
        fi
        tag=$(cuda_to_torch_tag "$cuda_ver")
    fi

    local whl_index="https://download.pytorch.org/whl/${tag}"
    log_step "Installing PyTorch ($tag)..."

    local pip_args=(--upgrade torch torchaudio
        --only-binary=:all:
        --index-url "$whl_index")

    python3 -m pip install "${pip_args[@]}" \
        || { log_error "PyTorch install failed for $tag (no pre-built wheel available?)"; return 1; }

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local tv cr
        tv=$(python3 -c "import torch; print(torch.__version__)")
        cr=$(python3 -c "import torch; print(torch.version.cuda)")
        log_info "PyTorch $tv installed (CUDA runtime: $cr)"
    else
        log_warn "PyTorch installed but torch.cuda.is_available() == False"
        log_warn "Check that your NVIDIA driver is compatible"
    fi
}

# ---------------------------------------------------------------------------
#  install_flash_attn
#  Installs flash-attn with the correct pre-built wheel for the current
#  PyTorch / CUDA / Python combination.  Downloads from GitHub releases
#  via the project's mirror infrastructure (github_url).
#  Refuses to fall back to source build (10-30 min, often fails).
# ---------------------------------------------------------------------------
install_flash_attn() {
    if python3 -c "import flash_attn" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
        log_info "flash-attn ($ver) already installed, skipping"
        return 0
    fi

    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch not installed — install_flash_attn requires PyTorch first"
        return 1
    fi

    local fa_ver torch_ver cuda_major py_ver abi
    fa_ver=$(python3 -c "
import json, urllib.request
data = json.loads(urllib.request.urlopen(
    'https://pypi.org/pypi/flash-attn/json', timeout=10).read())
print(data['info']['version'])
" 2>/dev/null) || true

    if [ -z "$fa_ver" ]; then
        log_warn "无法从 PyPI 获取 flash-attn 最新版本，使用默认 2.8.3"
        fa_ver="2.8.3"
    fi

    read -r torch_ver cuda_major py_ver abi < <(python3 -c "
import torch, sys
tv = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
cu = torch.version.cuda.split('.')[0] if torch.version.cuda else ''
pv = f'{sys.version_info.major}{sys.version_info.minor}'
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
print(tv, cu, pv, abi)
")

    if [ -z "$cuda_major" ]; then
        log_warn "CUDA not available, skipping flash-attn"
        return 0
    fi

    local wheel="flash_attn-${fa_ver}+cu${cuda_major}torch${torch_ver}cxx11abi${abi}-cp${py_ver}-cp${py_ver}-linux_x86_64.whl"
    local gh_release="https://github.com/Dao-AILab/flash-attention/releases/download/v${fa_ver}/${wheel}"
    local download_url
    download_url=$(github_url "$gh_release")

    log_step "Installing flash-attn ${fa_ver} (torch${torch_ver} cu${cuda_major} cp${py_ver} abi=${abi})..."
    log_info "Wheel URL: $download_url"

    local tmp_wheel="/tmp/${wheel}"
    if curl -fSL -o "$tmp_wheel" --connect-timeout 15 --max-time 600 \
            --retry 2 --retry-delay 5 "$download_url" 2>/dev/null \
       && [ -s "$tmp_wheel" ] \
       && python3 -c "import zipfile; zipfile.ZipFile('$tmp_wheel')" 2>/dev/null; then
        log_info "预编译 wheel 下载成功，安装中..."
        if python3 -m pip install "$tmp_wheel"; then
            rm -f "$tmp_wheel"
            log_info "flash-attn ${fa_ver} installed"
            return 0
        fi
        rm -f "$tmp_wheel"
        log_error "预编译 wheel 安装失败"
    else
        rm -f "$tmp_wheel"
    fi

    log_error "No pre-built flash-attn wheel for: torch=${torch_ver} cuda=${cuda_major} python=${py_ver} abi=${abi}"
    log_error "Wheel filename: $wheel"
    log_error "Source build disabled (unreliable, 10-30 min). Options:"
    log_error "  1. Check flash-attn releases: https://github.com/Dao-AILab/flash-attention/releases"
    log_error "  2. Adjust PyTorch/CUDA/Python versions to match an available wheel"
    log_error "  3. Build manually: pip install flash-attn --no-build-isolation"
    return 1
}

# ---------------------------------------------------------------------------
#  install_qwen3_tts <source_dir>
#  Editable-installs the Qwen3-TTS package from the submodule.
# ---------------------------------------------------------------------------
install_qwen3_tts() {
    local src_dir="$1"

    if [ -z "$src_dir" ] || [ ! -f "$src_dir/pyproject.toml" ]; then
        log_error "Qwen3-TTS source not found at: ${src_dir:-<not set>}"
        return 1
    fi

    if python3 -c "import qwen_tts" 2>/dev/null; then
        local ver
        ver=$(python3 -c \
            "import importlib.metadata; print(importlib.metadata.version('qwen-tts'))" \
            2>/dev/null || echo "unknown")
        log_info "qwen-tts ($ver) already installed, skipping"
        return 0
    fi

    log_step "Installing Qwen3-TTS from $src_dir..."
    python3 -m pip install -e "$src_dir" \
        || { log_error "Qwen3-TTS install failed"; return 1; }
    log_info "Qwen3-TTS installed"
}

# ---------------------------------------------------------------------------
#  install_safetensors
#  Installs safetensors — needed for loading/saving model weight files.
#  Idempotent, lightweight, no CUDA dependency.
# ---------------------------------------------------------------------------
install_safetensors() {
    if python3 -c "import safetensors" 2>/dev/null; then
        log_info "safetensors already installed, skipping"
        return 0
    fi

    log_info "Installing safetensors ..."
    python3 -m pip install safetensors -q \
        || log_warn "safetensors install failed (non-fatal, checkpoint will fall back to .bin)"
}

# ---------------------------------------------------------------------------
#  install_onnx_export_deps
#  Installs packages needed for ONNX export:
#    onnx, onnxruntime, onnxscript (torch.onnx 内部依赖), onnxsim (简化)
#  Idempotent — skips if already installed.
# ---------------------------------------------------------------------------
install_onnx_export_deps() {
    local missing=()

    python3 -c "import onnx" 2>/dev/null \
        || missing+=(onnx)

    python3 -c "import onnxruntime" 2>/dev/null \
        || missing+=(onnxruntime)

    python3 -c "import onnxscript" 2>/dev/null \
        || missing+=(onnxscript)

    python3 -c "import onnxsim" 2>/dev/null \
        || missing+=(onnxsim)

    if [ ${#missing[@]} -eq 0 ]; then
        log_info "ONNX export dependencies already installed"
        return 0
    fi

    log_step "Installing ONNX export dependencies: ${missing[*]}"
    python3 -m pip install --upgrade "${missing[@]}" \
        || { log_error "Failed to install ONNX dependencies"; return 1; }
    log_info "ONNX export dependencies installed"
}

# ---------------------------------------------------------------------------
#  install_tensorrt_for_standalone_engine
#  Pins the TensorRT Python wheel to match Phase B trtexec inside the default
#  NGC image (build_engines.sh / resolve_ngc_image).  Engine plans are not
#  portable across TRT minor versions; e.g. tritonserver:26.02 ships libnvinfer
#  10.15.1 while pip's latest tensorrt may be 10.16.x, which cannot load those
#  engines.  Override pin with STANDALONE_ENGINE_TENSORRT_PIP_VERSION.
# ---------------------------------------------------------------------------
install_tensorrt_for_standalone_engine() {
    local want="${STANDALONE_ENGINE_TENSORRT_PIP_VERSION:-10.15.1.29}"
    install_tensorrt_for_python python3 "$want"
}

# ---------------------------------------------------------------------------
#  validate_python_env
#  Quick smoke-test: imports the core packages and reports versions.
# ---------------------------------------------------------------------------
validate_python_env() {
    log_step "Validating Python environment..."

    python3 - <<'PYEOF'
import sys, importlib

required = {
    "torch":        "torch",
    "torchaudio":   "torchaudio",
    "transformers": "transformers",
    "qwen_tts":     "qwen-tts",
}

optional = {
    "safetensors":  "safetensors",
    "onnx":         "onnx",
    "onnxruntime":  "onnxruntime",
    "aiohttp":      "aiohttp",
    "grpc":         "grpcio",
    "requests":     "requests",
}

ok, fail = 0, 0
for mod, label in required.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:20s} {ver}")
        ok += 1
    except ImportError:
        print(f"  {label:20s} ** MISSING **")
        fail += 1

for mod, label in optional.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:20s} {ver}")
    except ImportError:
        print(f"  {label:20s} (not installed)")

# CUDA check
try:
    import torch
    if torch.cuda.is_available():
        print(f"  {'CUDA':20s} {torch.version.cuda}  "
              f"(GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  {'CUDA':20s} not available")
except Exception:
    pass

sys.exit(1 if fail else 0)
PYEOF

    if [ $? -ne 0 ]; then
        log_error "Some packages are missing — see above"
        return 1
    fi
    log_info "Python environment OK"
}
