#!/bin/bash
# ===========================================================================
#  mirrors.sh — Package mirror detection & configuration
#
#  Functions: configure_mirrors
#  Depends:   lib/logging.sh
#
#  If the user already has custom conda/pip mirror configurations, they are
#  respected without modification.  Otherwise, the user is prompted to
#  optionally switch to Chinese (BFSU) mirrors for faster downloads.
#
#  Non-interactive control via CONFIGURE_MIRRORS env:
#    china | cn   → apply Chinese mirrors without asking
#                   (also tells network.sh to prefer GitHub/HF mirrors)
#    skip | no    → skip mirror configuration entirely
#    auto | ""    → detect + interactive prompt (default)
# ===========================================================================

[[ -n "${_LIB_MIRRORS_LOADED:-}" ]] && return 0
_LIB_MIRRORS_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"

# ---- Detection ------------------------------------------------------------

_conda_has_custom_mirrors() {
    local condarc="$HOME/.condarc"
    [ -f "$condarc" ] && grep -qE 'https?://' "$condarc" 2>/dev/null
}

_pip_has_custom_mirrors() {
    [ -n "${PIP_INDEX_URL:-}" ] && return 0

    local conf
    for conf in \
        "$HOME/.config/pip/pip.conf" \
        "$HOME/.pip/pip.conf" \
        "/etc/pip.conf"; do
        if [ -f "$conf" ] && grep -qi 'index-url' "$conf" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

# ---- Apply -----------------------------------------------------------------

_apply_conda_china_mirrors() {
    local condarc="$HOME/.condarc"
    log_info "配置 conda/mamba 国内镜像源 (BFSU)..."

    if [ -f "$condarc" ]; then
        cp "$condarc" "${condarc}.bak.$(date +%s)"
        log_info "已备份原有配置到 ${condarc}.bak.*"
    fi

    cat > "$condarc" << 'CONDARC'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
CONDARC

    log_info "conda/mamba 镜像源已配置: $condarc"
}

_apply_pip_china_mirrors() {
    local pip_conf_dir="$HOME/.config/pip"
    local pip_conf="$pip_conf_dir/pip.conf"
    log_info "配置 pip 国内镜像源 (BFSU)..."

    mkdir -p "$pip_conf_dir"

    if [ -f "$pip_conf" ]; then
        cp "$pip_conf" "${pip_conf}.bak.$(date +%s)"
        log_info "已备份原有配置到 ${pip_conf}.bak.*"
    fi

    cat > "$pip_conf" << 'PIPCONF'
[global]
index-url = https://mirrors.bfsu.edu.cn/pypi/web/simple
trusted-host = mirrors.bfsu.edu.cn
PIPCONF

    log_info "pip 镜像源已配置: $pip_conf"
}

# ---- Main entry point ------------------------------------------------------

# ---------------------------------------------------------------------------
#  configure_mirrors
#
#  1. Detects existing custom configs → uses them silently
#  2. If unconfigured, either follows CONFIGURE_MIRRORS env or asks the user
# ---------------------------------------------------------------------------
configure_mirrors() {
    local conda_custom=false
    local pip_custom=false

    if _conda_has_custom_mirrors; then
        conda_custom=true
        log_info "检测到自定义 conda/mamba 镜像配置，将直接使用"
    fi

    if _pip_has_custom_mirrors; then
        pip_custom=true
        log_info "检测到自定义 pip 镜像配置，将直接使用"
    fi

    if $conda_custom && $pip_custom; then
        return 0
    fi

    # ---- Non-interactive overrides ----

    case "${CONFIGURE_MIRRORS:-}" in
        china|cn)
            log_info "CONFIGURE_MIRRORS=${CONFIGURE_MIRRORS}, 自动配置国内镜像..."
            $conda_custom || _apply_conda_china_mirrors
            $pip_custom   || _apply_pip_china_mirrors
            return 0
            ;;
        skip|no|off)
            log_info "CONFIGURE_MIRRORS=${CONFIGURE_MIRRORS}, 跳过镜像配置"
            return 0
            ;;
    esac

    # ---- Non-interactive terminal (CI / piped stdin) → skip ----

    if [ ! -t 0 ]; then
        log_info "非交互模式，跳过镜像源配置（可设置 CONFIGURE_MIRRORS=china 启用）"
        return 0
    fi

    # ---- Interactive prompt ----

    local need_config=()
    $conda_custom || need_config+=("conda/mamba")
    $pip_custom   || need_config+=("pip")

    echo ""
    log_step "以下包管理器尚未配置镜像源: ${need_config[*]}"
    echo ""
    echo "  如果您在中国大陆，配置国内镜像可以显著加速包下载。"
    echo "  如果您在海外或有稳定的网络环境，保持默认即可。"
    echo ""
    echo "  [1] 配置国内镜像源（BFSU 北外镜像，推荐中国大陆用户）"
    echo "  [2] 保持默认源不变（海外用户 / 网络畅通 / 稍后自行配置）"
    echo ""

    local choice=""
    read -rp "  请选择 [1/2] (默认: 2, 30s 后自动选择默认): " -t 30 choice || true
    echo ""

    case "$choice" in
        1)
            $conda_custom || _apply_conda_china_mirrors
            $pip_custom   || _apply_pip_china_mirrors
            log_info "国内镜像源配置完成"
            ;;
        *)
            log_info "保持默认源配置"
            ;;
    esac
}
