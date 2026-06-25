#!/bin/bash
# ===========================================================================
#  venv.sh — Python virtual-environment management
#
#  Functions: ensure_venv_cmd, create_venv, ensure_venv
#  Depends:   lib/logging.sh, lib/utils.sh
# ===========================================================================

[[ -n "${_LIB_VENV_LOADED:-}" ]] && return 0
_LIB_VENV_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

VENV_CMD=""

# ---------------------------------------------------------------------------
#  _nounset_safe <command...>
#  Runs a command with 'nounset' (set -u) temporarily disabled.
#  conda/mamba shell hooks and activate scripts reference variables that
#  may be unset, which conflicts with set -u.
# ---------------------------------------------------------------------------
_nounset_safe() {
    local _had_u=""
    [[ "$-" == *u* ]] && { _had_u=1; set +u; }
    "$@"
    local _rc=$?
    [ -n "$_had_u" ] && set -u
    return $_rc
}

# ---------------------------------------------------------------------------
#  _persist_conda_init <conda_exe>
#  Runs `conda init` for the user's login shell so that future terminal
#  sessions have conda/mamba available without manual setup.
# ---------------------------------------------------------------------------
_persist_conda_init() {
    local conda_exe="$1"

    _conda_init_shell "$conda_exe" bash "$HOME/.bashrc"

    local user_shell
    user_shell=$(basename "${SHELL:-bash}")
    if [ "$user_shell" != "bash" ]; then
        local rc_file
        case "$user_shell" in
            zsh)    rc_file="$HOME/.zshrc" ;;
            fish)   rc_file="$HOME/.config/fish/config.fish" ;;
            tcsh)   rc_file="$HOME/.tcshrc" ;;
            xonsh)  rc_file="$HOME/.xonshrc" ;;
            *)      return 0 ;;
        esac
        _conda_init_shell "$conda_exe" "$user_shell" "$rc_file"
    fi
}

_conda_init_shell() {
    local conda_exe="$1" shell_name="$2" rc_file="$3"
    if [ -f "$rc_file" ] && grep -q '# >>> conda initialize >>>' "$rc_file" 2>/dev/null; then
        return 0
    fi
    "$conda_exe" init "$shell_name" 2>/dev/null || true
    log_info "已为 ${shell_name} 配置 conda 初始化 (${rc_file})"
}

# ---------------------------------------------------------------------------
#  _find_conda_exe
#  Locates a conda executable: PATH → standard install paths → CONDA_EXE env.
#  Prints the path on success, returns 1 on failure.
# ---------------------------------------------------------------------------
_find_conda_exe() {
    local exe
    exe=$(command -v conda 2>/dev/null) && { echo "$exe"; return 0; }

    local prefix
    for prefix in \
        "${HOME}/miniforge3" \
        "${HOME}/miniforge" \
        "${HOME}/mambaforge" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3"; do
        if [ -x "${prefix}/bin/conda" ]; then
            echo "${prefix}/bin/conda"
            return 0
        fi
    done

    if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
        echo "$CONDA_EXE"
        return 0
    fi

    return 1
}

# ---------------------------------------------------------------------------
#  _init_conda_shell
#  Ensures the conda shell function is available in the current session.
#  Finds conda via _find_conda_exe and runs the shell hook.
# ---------------------------------------------------------------------------
_init_conda_shell() {
    declare -f conda &>/dev/null && return 0

    local conda_exe
    conda_exe=$(_find_conda_exe) || return 1

    _nounset_safe eval "$("$conda_exe" shell.bash hook 2>/dev/null)" || return 1
    declare -f conda &>/dev/null
}

# ---------------------------------------------------------------------------
#  _install_miniforge
#  Downloads and installs Miniforge3 (mamba + conda).
#  Uses GitHub mirror when github.com is unreachable.
# ---------------------------------------------------------------------------
_install_miniforge() {
    if [ -z "${HOME:-}" ]; then
        log_error "HOME 环境变量未设置，无法确定安装路径"
        return 1
    fi

    local MINIFORGE_PREFIX="${HOME}/miniforge3"

    if [ -x "${MINIFORGE_PREFIX}/bin/mamba" ]; then
        log_info "Miniforge3 已存在: ${MINIFORGE_PREFIX}"
        _init_conda_shell
        _persist_conda_init "${MINIFORGE_PREFIX}/bin/conda"
        VENV_CMD="mamba"
        return 0
    fi

    log_info "正在安装 Miniforge3 到 ${MINIFORGE_PREFIX} ..."
    local INSTALLER="Miniforge3-$(uname)-$(uname -m).sh"
    local DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"

    if declare -f github_url &>/dev/null; then
        DOWNLOAD_URL=$(github_url "$DOWNLOAD_URL") || \
            DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    fi

    retry 3 10 curl -fgSL --connect-timeout 10 --max-time 600 \
        -O "$DOWNLOAD_URL" \
        || { log_error "Miniforge 下载失败"; return 1; }
    bash "${INSTALLER}" -b -p "${MINIFORGE_PREFIX}" \
        || { log_error "Miniforge 安装失败"; return 1; }
    rm -f "${INSTALLER}"

    _nounset_safe eval "$("${MINIFORGE_PREFIX}/bin/conda" shell.bash hook 2>/dev/null)" || true
    _persist_conda_init "${MINIFORGE_PREFIX}/bin/conda"

    if command -v mamba &>/dev/null; then
        VENV_CMD="mamba"
    else
        VENV_CMD="conda"
    fi
    log_info "Miniforge3 安装完成 (${MINIFORGE_PREFIX})，使用: ${VENV_CMD}"
}

# ---------------------------------------------------------------------------
#  ensure_venv_cmd
#  Detects a venv tool or interactively installs one.
#
#  Priority: existing mamba/conda (PATH or standard paths) →
#            user-approved miniforge install → system python3 venv → error.
#
#  Sets the global VENV_CMD variable.
# ---------------------------------------------------------------------------
ensure_venv_cmd() {
    if [ -n "$VENV_CMD" ]; then
        return 0
    fi

    # ---- Priority 1a: mamba/conda already in PATH ----

    if command -v mamba &>/dev/null; then
        VENV_CMD="mamba"
        log_info "检测到已安装的环境工具: mamba"
        return 0
    fi

    if command -v conda &>/dev/null; then
        VENV_CMD="conda"
        log_info "检测到已安装的环境工具: conda"
        return 0
    fi

    # ---- Priority 1b: un-initialised install at standard locations ----

    local conda_exe
    if conda_exe=$(_find_conda_exe); then
        local prefix
        prefix="$(dirname "$(dirname "$conda_exe")")"
        log_info "检测到未初始化的 conda/mamba 安装: ${prefix}"
        _nounset_safe eval "$("$conda_exe" shell.bash hook 2>/dev/null)" || true
        _persist_conda_init "$conda_exe"
        if command -v mamba &>/dev/null; then
            VENV_CMD="mamba"
        else
            VENV_CMD="conda"
        fi
        log_info "已激活环境工具: ${VENV_CMD}"
        return 0
    fi

    # ---- No conda/mamba — check python3 venv as fallback ----

    local has_pyvenv=false
    if command -v python3 &>/dev/null && python3 -m venv -h &>/dev/null 2>&1; then
        has_pyvenv=true
    fi

    # ---- Non-interactive terminal (CI / piped stdin) ----

    if [ ! -t 0 ]; then
        if $has_pyvenv; then
            VENV_CMD="venv"
            log_info "非交互模式，使用 python3 venv"
            return 0
        fi
        log_error "未找到 conda/mamba/python3-venv，非交互模式无法自动安装"
        log_error "请预装 conda/mamba 或确保 python3 -m venv 可用"
        return 1
    fi

    # ---- Priority 2: interactive — ask user ----

    echo ""
    log_step "未检测到 conda 或 mamba"
    echo ""

    if $has_pyvenv; then
        echo "  推荐安装 Miniforge（mamba/conda），可获得更好的依赖管理和环境隔离。"
        echo "  如不安装，将使用系统 python3 的 venv 模块作为替代。"
        echo ""
        echo "  [1] 安装 Miniforge（推荐，提供 mamba + conda）"
        echo "  [2] 使用系统 python3 venv（轻量，但功能有限）"
        echo ""

        local choice=""
        read -rp "  请选择 [1/2] (默认: 1, 30s 后自动选择默认): " -t 30 choice || true
        echo ""

        case "$choice" in
            2)
                VENV_CMD="venv"
                log_info "将使用 python3 venv"
                return 0
                ;;
            *)
                _install_miniforge || return 1
                return 0
                ;;
        esac
    else
        echo "  系统中也未找到 python3 venv 模块。"
        echo "  需要安装 Miniforge 以提供 Python 环境管理工具。"
        echo ""
        echo "  [1] 安装 Miniforge（推荐，提供 mamba + conda + Python）"
        echo "  [2] 退出，稍后自行安装 Python 环境"
        echo ""

        local choice=""
        read -rp "  请选择 [1/2] (默认: 1, 30s 后自动选择默认): " -t 30 choice || true
        echo ""

        case "$choice" in
            2)
                log_error "未找到可用的 Python 环境管理工具"
                log_error "请安装 conda/mamba 或确保 python3 + venv 可用后重试"
                return 1
                ;;
            *)
                _install_miniforge || return 1
                return 0
                ;;
        esac
    fi
}

_venv_exists() {
    local env_name="$1"
    case "$VENV_CMD" in
        mamba|conda)
            $VENV_CMD env list 2>/dev/null | awk '{print $1}' | grep -qx "$env_name"
            ;;
        venv)
            [ -d "$env_name" ] && [ -f "$env_name/bin/activate" ]
            ;;
        *)
            return 1
            ;;
    esac
}

_activate_venv() {
    local env_name="$1"
    case "$VENV_CMD" in
        mamba|conda)
            _init_conda_shell || {
                log_error "conda 初始化失败，请检查 conda/mamba 安装"
                return 1
            }

            if _nounset_safe conda activate "$env_name" 2>/dev/null; then
                return 0
            fi
            log_warn "conda activate 失败，尝试直接 source activate ..."

            local env_path
            env_path=$($VENV_CMD env list 2>/dev/null \
                | awk -v name="$env_name" '$1==name {print $NF}')
            if [ -n "$env_path" ] && [ -f "$env_path/bin/activate" ]; then
                _nounset_safe source "$env_path/bin/activate" && return 0
            fi
            log_error "无法激活 conda 环境 '$env_name'"
            return 1
            ;;
        venv)
            source "$env_name/bin/activate"
            ;;
    esac
}

# ---------------------------------------------------------------------------
#  create_venv <env_name> [python_version]
#  Creates a new environment (idempotent — skips if it exists).
# ---------------------------------------------------------------------------
create_venv() {
    local ENV_NAME="$1"
    local PYTHON_VERSION="${2:-3.10}"

    if [ -z "$ENV_NAME" ]; then
        log_error "Usage: create_venv <env_name> [python_version]"
        return 1
    fi

    ensure_venv_cmd || return 1

    if _venv_exists "$ENV_NAME"; then
        log_warn "Environment '$ENV_NAME' already exists, skipping creation"
        return 0
    fi

    log_step "Creating environment '$ENV_NAME' (python=$PYTHON_VERSION)..."

    case "$VENV_CMD" in
        mamba)
            mamba create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y \
                || { log_error "mamba create failed"; return 1; }
            ;;
        conda)
            conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y \
                || { log_error "conda create failed"; return 1; }
            ;;
        venv)
            local SYS_PY_VER
            SYS_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            if [ "$SYS_PY_VER" != "$PYTHON_VERSION" ]; then
                log_warn "System python is $SYS_PY_VER, requested $PYTHON_VERSION"
                log_warn "venv cannot switch python version, will use $SYS_PY_VER"
            fi
            python3 -m venv "$ENV_NAME" \
                || { log_error "python3 -m venv failed"; return 1; }
            ;;
    esac

    log_info "Environment '$ENV_NAME' created successfully"
}

# ---------------------------------------------------------------------------
#  ensure_venv <env_name> [python_version]
#  Creates (if needed) AND activates the environment.
# ---------------------------------------------------------------------------
ensure_venv() {
    local ENV_NAME="$1"
    local PYTHON_VERSION="${2:-3.10}"

    if [ -z "$ENV_NAME" ]; then
        log_error "Usage: ensure_venv <env_name> [python_version]"
        return 1
    fi

    ensure_venv_cmd || return 1

    if _venv_exists "$ENV_NAME"; then
        log_info "Environment '$ENV_NAME' already exists"
    else
        create_venv "$ENV_NAME" "$PYTHON_VERSION" || return 1
    fi

    _activate_venv "$ENV_NAME" || { log_error "Failed to activate '$ENV_NAME'"; return 1; }
    log_info "Activated: $ENV_NAME ($(python3 --version 2>&1))"
}
