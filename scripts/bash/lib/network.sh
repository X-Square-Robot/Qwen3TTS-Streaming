#!/bin/bash
# ===========================================================================
#  network.sh — Network / mirror helpers
#
#  Functions: github_url, hf_url, download_hf_model,
#             download_model, download_qwen3_tts_models,
#             select_model_variant, pin_model_versions,
#             list_model_variants
#  Depends:   lib/logging.sh, lib/utils.sh
#  Config:    ../model_versions.conf (version pinning)
#
#  Mirror selection strategy (GitHub & HuggingFace):
#    1. CONFIGURE_MIRRORS=china|cn  → always use mirrors (no probe)
#    2. Otherwise: probe origin site, measure response time
#       - Fast (≤ threshold)  → use origin directly
#       - Slow (> threshold)  → prefer mirror, fall back to slow origin
#       - Unreachable         → prefer mirror, fail if mirror also down
#    3. Results are cached for the script lifetime
#
#  Tuning env vars:
#    GITHUB_MIRROR             mirror base URL  (default: gh-proxy.org)
#    GITHUB_SPEED_THRESHOLD_MS probe threshold   (default: 2000)
#    HF_MIRROR                 mirror base URL  (default: hf-mirror.com)
#    HF_SPEED_THRESHOLD_MS     probe threshold   (default: 2000)
# ===========================================================================

[[ -n "${_LIB_NETWORK_LOADED:-}" ]] && return 0
_LIB_NETWORK_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

# ---- Shared: China network detection --------------------------------------

# Speed threshold (ms) — if a probe takes longer than this, prefer mirror.
# GitHub homepage / HF homepage are small; >2s means the connection is poor
# and actual downloads (git clone, LFS) will be much worse.
GITHUB_SPEED_THRESHOLD_MS="${GITHUB_SPEED_THRESHOLD_MS:-2000}"
HF_SPEED_THRESHOLD_MS="${HF_SPEED_THRESHOLD_MS:-2000}"

_is_china_mirrors() {
    case "${CONFIGURE_MIRRORS:-}" in china|cn) return 0 ;; esac
    return 1
}

# _probe_url <url> <threshold_ms>
#   Prints "fast" | "slow" | "unreachable"
_probe_url() {
    local url="$1" threshold="$2"
    local result http_code time_s time_ms

    result=$(curl -s -o /dev/null -w "%{http_code} %{time_total}" \
        --connect-timeout 5 --max-time 10 "$url" 2>/dev/null) \
        || result="000 99.0"

    http_code=${result%% *}
    time_s=${result##* }
    time_ms=$(awk "BEGIN {printf \"%d\", ${time_s} * 1000}")

    if [ "$http_code" != "200" ] && [ "$http_code" != "301" ] && [ "$http_code" != "302" ]; then
        echo "unreachable ${http_code} ${time_ms}"
        return
    fi
    if [ "$time_ms" -gt "$threshold" ]; then
        echo "slow ${http_code} ${time_ms}"
        return
    fi
    echo "fast ${http_code} ${time_ms}"
}

# ---- GitHub ---------------------------------------------------------------

GITHUB_MIRROR="${GITHUB_MIRROR:-https://v6.gh-proxy.org}"

# Cache: "direct" | "mirror" | "failed"
_GITHUB_MODE_CACHE=""

_resolve_github_mode() {
    [ -n "$_GITHUB_MODE_CACHE" ] && return

    if _is_china_mirrors; then
        log_info "CONFIGURE_MIRRORS=${CONFIGURE_MIRRORS} → GitHub 使用镜像"
        _try_github_mirror "china-config" && return
        _GITHUB_MODE_CACHE="direct"
        log_warn "GitHub 镜像不可用, 回退到直连"
        return
    fi

    local probe github_reachable=false
    probe=$(_probe_url "https://github.com" "$GITHUB_SPEED_THRESHOLD_MS")
    local status=${probe%% *}
    local rest=${probe#* }
    local code=${rest%% *}
    local ms=${rest##* }

    case "$status" in
        fast)
            log_info "GitHub 连接正常 (${ms}ms)"
            _GITHUB_MODE_CACHE="direct"
            return
            ;;
        slow)
            github_reachable=true
            log_warn "GitHub 响应缓慢 (${ms}ms > ${GITHUB_SPEED_THRESHOLD_MS}ms), 尝试镜像加速..."
            ;;
        unreachable)
            log_warn "GitHub 不可达 (HTTP: ${code}), 尝试镜像..."
            ;;
    esac

    if _try_github_mirror "$status"; then
        return
    fi

    if $github_reachable; then
        log_warn "GitHub 镜像不可用, 回退到直连（速度可能较慢）"
        _GITHUB_MODE_CACHE="direct"
    else
        _GITHUB_MODE_CACHE="failed"
    fi
}

_try_github_mirror() {
    local reason="$1"
    local mirror_probe
    mirror_probe=$(_probe_url "${GITHUB_MIRROR}" 10000)
    local m_status=${mirror_probe%% *}

    if [ "$m_status" = "fast" ] || [ "$m_status" = "slow" ]; then
        log_info "使用 GitHub 镜像: ${GITHUB_MIRROR}"
        _GITHUB_MODE_CACHE="mirror"
        return 0
    fi

    local m_rest=${mirror_probe#* }
    local m_code=${m_rest%% *}
    log_warn "GitHub 镜像 (${GITHUB_MIRROR}) 不可用 (HTTP: ${m_code})"
    return 1
}

# ---------------------------------------------------------------------------
#  github_url <repo_url>
#  Returns the best URL for a GitHub resource: original if fast, mirror if
#  GitHub is slow/unreachable, or fails if neither works.
#  Result is cached for the script lifetime.
# ---------------------------------------------------------------------------
github_url() {
    local REPO_URL="$1"

    _resolve_github_mode

    case "$_GITHUB_MODE_CACHE" in
        mirror)
            echo "${GITHUB_MIRROR}/${REPO_URL}"
            return 0
            ;;
        direct)
            echo "$REPO_URL"
            return 0
            ;;
        failed)
            log_error "GitHub 及镜像均不可用"
            return 1
            ;;
    esac
}

# ---- HuggingFace ----------------------------------------------------------

HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

# Cache: "direct" | "mirror"
_HF_MODE_CACHE=""

# ---------------------------------------------------------------------------
#  _hf_reachable
#  Returns 0 if huggingface.co is fast enough for direct use, 1 otherwise.
#  Also considers CONFIGURE_MIRRORS=china.  Cached for script lifetime.
# ---------------------------------------------------------------------------
_hf_reachable() {
    if [ -z "$_HF_MODE_CACHE" ]; then
        if _is_china_mirrors; then
            log_info "CONFIGURE_MIRRORS=${CONFIGURE_MIRRORS} → HuggingFace 使用镜像"
            _HF_MODE_CACHE="mirror"
        else
            local probe
            probe=$(_probe_url "https://huggingface.co" "$HF_SPEED_THRESHOLD_MS")
            local status=${probe%% *}
            local rest=${probe#* }
            local code=${rest%% *}
            local ms=${rest##* }

            case "$status" in
                fast)
                    log_info "HuggingFace 连接正常 (${ms}ms)"
                    _HF_MODE_CACHE="direct"
                    ;;
                slow)
                    log_warn "HuggingFace 响应缓慢 (${ms}ms > ${HF_SPEED_THRESHOLD_MS}ms), 使用镜像"
                    _HF_MODE_CACHE="mirror"
                    ;;
                unreachable)
                    log_warn "HuggingFace 不可达 (HTTP: ${code}), 使用镜像 (${HF_MIRROR})"
                    _HF_MODE_CACHE="mirror"
                    ;;
            esac
        fi
    fi
    [ "$_HF_MODE_CACHE" = "direct" ]
}

# ---------------------------------------------------------------------------
#  hf_url <original_url>
#  Rewrites a huggingface.co URL to use the mirror when HF is unreachable.
# ---------------------------------------------------------------------------
hf_url() {
    local ORIGINAL_URL="$1"
    if _hf_reachable; then
        echo "$ORIGINAL_URL"
    else
        echo "$ORIGINAL_URL" | sed "s|https://huggingface.co|${HF_MIRROR}|g"
    fi
}

# ---------------------------------------------------------------------------
#  _hf_endpoint
#  Returns the HF API endpoint (respects HF_ENDPOINT env, falls back to
#  mirror when HuggingFace is unreachable).
# ---------------------------------------------------------------------------
_hf_endpoint() {
    if [ -n "${HF_ENDPOINT:-}" ]; then
        echo "$HF_ENDPOINT"
    elif _hf_reachable; then
        echo "https://huggingface.co"
    else
        echo "$HF_MIRROR"
    fi
}

# ---------------------------------------------------------------------------
#  download_hf_model <model_id> [target_dir]
#  Shallow-clones a HuggingFace repo (idempotent).
# ---------------------------------------------------------------------------
download_hf_model() {
    local MODEL_ID="$1"
    local TARGET_DIR="${2:-.}"

    if [ -z "$MODEL_ID" ]; then
        log_error "Usage: download_hf_model <model_id> [target_dir]"
        return 1
    fi

    require_cmd git || return 1
    mkdir -p "$TARGET_DIR"

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_ID")
    local DEST="$TARGET_DIR/$MODEL_NAME"

    if [ -d "$DEST" ] && [ -d "$DEST/.git" ]; then
        log_info "Model '$MODEL_NAME' already exists at $DEST, skipping"
        return 0
    fi

    local REPO_URL
    REPO_URL=$(hf_url "https://huggingface.co/${MODEL_ID}")

    log_step "Downloading model: $MODEL_ID"
    log_info "From: $REPO_URL → $DEST"

    GIT_LFS_SKIP_SMUDGE=0 \
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=60 \
        git clone --depth 1 "$REPO_URL" "$DEST" \
        || { log_error "Failed to clone model $MODEL_ID"; return 1; }

    log_info "Model downloaded to: $DEST"
}

# ---- High-level model download -------------------------------------------

# ---------------------------------------------------------------------------
#  download_model <model_id> <target_dir> [source]
#  Downloads a model using the best available method.
#
#  source: "auto" (default) | "hf" | "modelscope"
#
#  Strategy (source=auto):
#    1. huggingface-cli  (resume-capable, progress bar)
#    2. modelscope download  (for China users)
#    3. git clone (fallback)
# ---------------------------------------------------------------------------
download_model() {
    local MODEL_ID="$1"
    local TARGET_DIR="$2"
    local SOURCE="${3:-auto}"
    local REVISION="${4:-}"

    if [ -z "$MODEL_ID" ] || [ -z "$TARGET_DIR" ]; then
        log_error "Usage: download_model <model_id> <target_dir> [hf|modelscope|auto] [revision]"
        return 1
    fi

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_ID")
    local DEST="$TARGET_DIR/$MODEL_NAME"
    mkdir -p "$TARGET_DIR"

    # Only skip when we have proof of prior successful completion (.model_revision).
    # Interrupted downloads leave partial files; _model_dir_valid may be true but we
    # must run ModelScope to validate hashes and re-download only corrupted files.
    if _model_dir_valid "$DEST"; then
        if _revision_matches "$DEST" "$REVISION" && [ -f "$DEST/.model_revision" ]; then
            log_info "Model '$MODEL_NAME' already at $DEST (revision OK, validated), skipping"
            return 0
        fi
        if ! _revision_matches "$DEST" "$REVISION"; then
            log_warn "Model '$MODEL_NAME' exists but revision mismatch, re-downloading..."
            rm -rf "$DEST"
        else
            log_info "Model '$MODEL_NAME' at $DEST may be incomplete; validating and resuming..."
        fi
    fi

    local rev_info=""
    [ -n "$REVISION" ] && [ "$REVISION" != "main" ] && rev_info=" (revision: ${REVISION:0:12})"
    log_step "Downloading: $MODEL_ID${rev_info} → $DEST"

    local ok=false
    case "$SOURCE" in
        modelscope|ms)
            _download_via_modelscope "$MODEL_ID" "$DEST" "$REVISION" && ok=true
            if ! $ok; then
                log_warn "ModelScope download failed, trying HuggingFace..."
                _download_via_hf "$MODEL_ID" "$DEST" "$REVISION" && ok=true
            fi
            ;;
        hf)
            _download_via_hf "$MODEL_ID" "$DEST" "$REVISION" && ok=true
            ;;
        auto|*)
            _download_via_modelscope "$MODEL_ID" "$DEST" "$REVISION" && ok=true
            if ! $ok; then
                log_warn "ModelScope failed, trying HuggingFace..."
                _download_via_hf "$MODEL_ID" "$DEST" "$REVISION" && ok=true
            fi
            ;;
    esac

    if $ok; then
        _record_revision "$DEST" "$REVISION"
        return 0
    fi

    log_error "All download methods failed for $MODEL_ID"
    return 1
}

# Internal: check if a model directory looks valid
_model_dir_valid() {
    local dir="$1"
    [ -d "$dir" ] && {
        [ -f "$dir/config.json" ] \
            || [ -f "$dir/tokenizer_config.json" ] \
            || [ -d "$dir/.git" ] \
            || ls "$dir"/*.safetensors &>/dev/null 2>&1 \
            || ls "$dir"/*.bin &>/dev/null 2>&1
    }
}

# Internal: check if the downloaded revision matches the requested one
_revision_matches() {
    local dest="$1" requested="$2"
    # No specific revision or "main" → any existing version is acceptable
    [ -z "$requested" ] || [ "$requested" = "main" ] && return 0

    local recorded=""
    [ -f "$dest/.model_revision" ] && recorded=$(cat "$dest/.model_revision" 2>/dev/null)
    [ -z "$recorded" ] && return 1

    # Prefix match (short hash matches full hash and vice versa)
    [[ "$recorded" = "$requested"* ]] || [[ "$requested" = "$recorded"* ]]
}

# Internal: record revision info after successful download
_record_revision() {
    local dest="$1" revision="$2"
    local actual_rev="${revision:-main}"

    if [ -d "$dest/.git" ]; then
        actual_rev=$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo "$actual_rev")
    fi

    echo "$actual_rev" > "$dest/.model_revision"
}

# Internal: download via huggingface-cli (preferred)
_download_via_hf() {
    local model_id="$1"
    local dest="$2"
    local revision="${3:-}"
    local endpoint
    endpoint=$(_hf_endpoint)

    local rev_args=()
    [ -n "$revision" ] && [ "$revision" != "main" ] && rev_args=(--revision "$revision")

    if command -v huggingface-cli &>/dev/null; then
        log_info "Using huggingface-cli (endpoint: $endpoint)"
        HF_ENDPOINT="$endpoint" \
            huggingface-cli download "$model_id" "${rev_args[@]}" --local-dir "$dest" \
            && return 0
        log_warn "huggingface-cli failed"
    fi

    log_info "Falling back to git clone..."
    local repo_url
    repo_url=$(hf_url "https://huggingface.co/${model_id}")

    local branch_args=(--depth 1)
    [ -n "$revision" ] && [ "$revision" != "main" ] && branch_args=(--depth 1 --branch "$revision")

    rm -rf "$dest"
    GIT_LFS_SKIP_SMUDGE=0 \
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=60 \
        retry 2 10 git clone "${branch_args[@]}" "$repo_url" "$dest"
}

# Internal: download via modelscope (official recommended method)
_download_via_modelscope() {
    local model_id="$1"
    local dest="$2"
    local revision="${3:-}"

    if ! command -v modelscope &>/dev/null; then
        if ! python3 -c "import modelscope" 2>/dev/null; then
            log_info "Installing modelscope CLI..."
            python3 -m pip install -q modelscope 2>/dev/null || {
                log_warn "Could not install modelscope"
                return 1
            }
        fi
    fi

    if ! command -v modelscope &>/dev/null; then
        log_warn "modelscope CLI not available after install"
        return 1
    fi

    local rev_args=()
    [ -n "$revision" ] && [ "$revision" != "main" ] && rev_args=(--revision "$revision")

    # Enable SHA256 hash validation to detect truncated/corrupted files from interrupted
    # downloads; ModelScope >= 1.21 will auto-redownload mismatched files.
    log_info "Using ModelScope to download: $model_id (hash validation enabled)"
    MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION=True retry 2 10 \
        modelscope download --model "$model_id" "${rev_args[@]}" --local_dir "$dest"
}

# ---------------------------------------------------------------------------
#  download_qwen3_tts_models <target_dir> [variant] [source]
#
#  variant: base-1.7b (default) | custom-1.7b | design-1.7b
#           | base-0.6b | custom-0.6b | all-1.7b | all
#
#  Always downloads the Tokenizer; then downloads the TTS model(s) matching
#  the selected variant.
# ---------------------------------------------------------------------------

_QWEN3_TTS_TOKENIZER="Qwen/Qwen3-TTS-Tokenizer-12Hz"

declare -A _QWEN3_TTS_MODELS 2>/dev/null || true
_QWEN3_TTS_MODELS=(
    [base-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    [custom-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    [design-1.7b]="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    [base-0.6b]="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    [custom-0.6b]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)

# ---- Model version manifest -----------------------------------------------

declare -A _MODEL_REVISIONS 2>/dev/null || true
_MODEL_VERSIONS_LOADED=""

_load_model_versions() {
    [ -n "$_MODEL_VERSIONS_LOADED" ] && return 0
    _MODEL_VERSIONS_LOADED=1

    local conf_file="${_LIB_DIR}/../model_versions.conf"
    if [ ! -f "$conf_file" ]; then
        log_warn "Model version manifest not found: $conf_file"
        return 0
    fi

    local key value
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [ -z "$key" ] && continue
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        _MODEL_REVISIONS[$key]="$value"
    done < "$conf_file"
}

_get_model_revision() {
    local variant="$1"
    _load_model_versions
    echo "${_MODEL_REVISIONS[$variant]:-main}"
}

# ---------------------------------------------------------------------------
#  pin_model_versions <model_dir>
#  Captures the actual downloaded revision hashes and writes them
#  back to model_versions.conf for reproducibility.
# ---------------------------------------------------------------------------
pin_model_versions() {
    local model_dir="$1"
    local conf_file="${_LIB_DIR}/../model_versions.conf"

    local -A variant_to_dirname=(
        [tokenizer]="Qwen3-TTS-Tokenizer-12Hz"
        [base-1.7b]="Qwen3-TTS-12Hz-1.7B-Base"
        [custom-1.7b]="Qwen3-TTS-12Hz-1.7B-CustomVoice"
        [design-1.7b]="Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        [base-0.6b]="Qwen3-TTS-12Hz-0.6B-Base"
        [custom-0.6b]="Qwen3-TTS-12Hz-0.6B-CustomVoice"
    )

    {
        cat <<'HEADER'
# ===================================================================
#  Qwen3-TTS Model Version Manifest
#
#  Pin specific revisions (commit hashes) to ensure reproducibility.
#  When upstream models are updated (e.g., base → base-v2), this
#  project continues to use the tested versions until this file is
#  explicitly updated.
#
#  Format:  variant=revision
#    - Use a full commit hash for production stability
#    - Use "main" to track the latest (NOT recommended for production)
#
#  To pin currently downloaded versions:
#    bash scripts/bash/download_models.sh --pin
#
#  To update: change the revision below, then re-run the download.
#  The downloader will detect the version mismatch and re-download.
# ===================================================================

HEADER

        local pinned=0
        for variant in tokenizer base-1.7b custom-1.7b design-1.7b base-0.6b custom-0.6b; do
            local dirname="${variant_to_dirname[$variant]}"
            local dir="$model_dir/$dirname"
            local rev="main"

            if [ -f "$dir/.model_revision" ]; then
                rev=$(cat "$dir/.model_revision" 2>/dev/null)
                pinned=$((pinned + 1))
            fi
            echo "$variant=$rev"
        done

        if [ "$pinned" -eq 0 ]; then
            log_warn "No downloaded models found in $model_dir, nothing to pin"
            return 1
        fi
    } > "${conf_file}.tmp"

    mv "${conf_file}.tmp" "$conf_file"
    log_info "Pinned $pinned model revision(s) to: $conf_file"
}

# ---- Model variant listing / selection ------------------------------------

list_model_variants() {
    echo "Available model variants:"
    echo "  base-1.7b     Qwen3-TTS-12Hz-1.7B-Base        (voice clone, recommended)"
    echo "  custom-1.7b   Qwen3-TTS-12Hz-1.7B-CustomVoice (9 premium voices + instructions)"
    echo "  design-1.7b   Qwen3-TTS-12Hz-1.7B-VoiceDesign (voice design from description)"
    echo "  base-0.6b     Qwen3-TTS-12Hz-0.6B-Base        (lightweight voice clone)"
    echo "  custom-0.6b   Qwen3-TTS-12Hz-0.6B-CustomVoice (lightweight, 9 premium voices)"
    echo "  all-1.7b      All 1.7B models"
    echo "  all           All models"
}

# ---------------------------------------------------------------------------
#  select_model_variant
#  Interactively asks the user which model variant to download.
#  Outputs the variant key on stdout (for capture with $()).
#  Falls back to "base-1.7b" in non-interactive environments.
# ---------------------------------------------------------------------------
select_model_variant() {
    if [ ! -t 0 ]; then
        log_info "非交互模式，使用默认模型: base-1.7b"
        echo "base-1.7b"
        return 0
    fi

    echo "" >&2
    echo -e "  ${_CLR_BLUE}请选择要下载的 Qwen3-TTS 模型:${_CLR_RESET}" >&2
    echo "" >&2
    echo "  ── 1.7B 系列 ──────────────────────────────────────────────" >&2
    echo "  1) base-1.7b      Base         — 语音克隆 (推荐，Triton 部署核心)" >&2
    echo "  2) custom-1.7b    CustomVoice  — 9 种预置音色 + 指令控制" >&2
    echo "  3) design-1.7b    VoiceDesign  — 通过自然语言描述设计声音" >&2
    echo "" >&2
    echo "  ── 0.6B 系列 ──────────────────────────────────────────────" >&2
    echo "  4) base-0.6b      Base         — 轻量语音克隆" >&2
    echo "  5) custom-0.6b    CustomVoice  — 轻量预置音色" >&2
    echo "" >&2
    echo "  ── 组合选项 ───────────────────────────────────────────────" >&2
    echo "  6) all-1.7b       全部 1.7B 模型 (3 个)" >&2
    echo "  7) all            全部模型 (5 个)" >&2
    echo "" >&2

    local choice=""
    read -rp "  请输入选项 [1-7] (默认: 1, 30s 后自动选择默认): " -t 30 choice || true
    echo "" >&2

    local variant
    case "${choice:-1}" in
        1) variant="base-1.7b" ;;
        2) variant="custom-1.7b" ;;
        3) variant="design-1.7b" ;;
        4) variant="base-0.6b" ;;
        5) variant="custom-0.6b" ;;
        6) variant="all-1.7b" ;;
        7) variant="all" ;;
        *)
            log_warn "无效选项 '$choice', 使用默认 base-1.7b"
            variant="base-1.7b"
            ;;
    esac

    log_info "已选择模型: $variant"
    echo "$variant"
}

# ---------------------------------------------------------------------------
#  download_qwen3_tts_models <target_dir> [variant] [source]
#
#  Always downloads the Tokenizer; then downloads the TTS model(s)
#  matching the selected variant. Uses pinned revisions from
#  model_versions.conf when available.
# ---------------------------------------------------------------------------
download_qwen3_tts_models() {
    local TARGET_DIR="$1"
    local VARIANT="${2:-base-1.7b}"
    local SOURCE="${3:-auto}"

    if [ -z "$TARGET_DIR" ]; then
        log_error "Usage: download_qwen3_tts_models <target_dir> [variant] [source]"
        return 1
    fi

    mkdir -p "$TARGET_DIR"

    # Tokenizer is always required
    local tok_rev
    tok_rev=$(_get_model_revision "tokenizer")
    log_step "Downloading Tokenizer: $_QWEN3_TTS_TOKENIZER"
    download_model "$_QWEN3_TTS_TOKENIZER" "$TARGET_DIR" "$SOURCE" "$tok_rev" \
        || { log_error "Tokenizer download failed"; return 1; }

    # Resolve variant keys to download
    local -a variants_to_download=()

    case "$VARIANT" in
        all-1.7b)
            variants_to_download=(base-1.7b custom-1.7b design-1.7b)
            ;;
        all)
            variants_to_download=(base-1.7b custom-1.7b design-1.7b base-0.6b custom-0.6b)
            ;;
        *)
            if [ -z "${_QWEN3_TTS_MODELS[$VARIANT]:-}" ]; then
                log_error "Unknown model variant: $VARIANT"
                list_model_variants >&2
                return 1
            fi
            variants_to_download=("$VARIANT")
            ;;
    esac

    local failed=0
    for vkey in "${variants_to_download[@]}"; do
        local model_id="${_QWEN3_TTS_MODELS[$vkey]}"
        local rev
        rev=$(_get_model_revision "$vkey")
        download_model "$model_id" "$TARGET_DIR" "$SOURCE" "$rev" \
            || { log_error "Failed to download $model_id"; failed=$((failed + 1)); }
    done

    if [ "$failed" -gt 0 ]; then
        log_error "$failed model(s) failed to download"
        return 1
    fi

    log_info "All models downloaded to: $TARGET_DIR"
    ls -1 "$TARGET_DIR"/
}
