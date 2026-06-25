#!/bin/bash
# ===========================================================================
#  env_plan.sh — Environment version pre-resolution
#
#  Detects hardware, derives compatible software versions, validates wheel
#  availability, and writes a deterministic env_plan.json BEFORE any
#  pip install or conda create runs.
#
#  Functions:
#    resolve_env_plan   — detect + derive + verify + write JSON
#    print_env_plan     — pretty-print the plan summary
#    read_env_plan_val  — read a value from the plan JSON (jq-free)
#    confirm_env_plan   — interactive: show plan, handle degraded items, wait
#    _handle_degraded   — ask user about a degraded check at install time
#
#  Depends: lib/logging.sh, lib/prerequisites.sh, lib/docker.sh
# ===========================================================================

[[ -n "${_LIB_ENV_PLAN_LOADED:-}" ]] && return 0
_LIB_ENV_PLAN_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/prerequisites.sh"
source "${_LIB_DIR}/docker.sh"

# ---------------------------------------------------------------------------
#  _json_str <key> <value>
#  Emit "key": "value" (with proper escaping for JSON)
# ---------------------------------------------------------------------------
_json_str() { printf '"%s": "%s"' "$1" "$2"; }

# ---------------------------------------------------------------------------
#  _json_int <key> <value>
# ---------------------------------------------------------------------------
_json_int() { printf '"%s": %s' "$1" "${2:-0}"; }

# ---------------------------------------------------------------------------
#  _detect_gpu_info
#  Sets _GPU_NAME and _GPU_MEM_MB from nvidia-smi.
# ---------------------------------------------------------------------------
_detect_gpu_info() {
    _GPU_NAME=""
    _GPU_MEM_MB="0"
    if ! command -v nvidia-smi &>/dev/null; then return 1; fi

    _GPU_NAME=$(nvidia-smi --query-gpu=name \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    _GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _GPU_MEM_MB="${_GPU_MEM_MB:-0}"
}

# ---------------------------------------------------------------------------
#  _check_pytorch_wheel <cuda_tag>
#  Sets _PYTORCH_STATUS ("ok" / "degraded" / "warning")
#  and _PYTORCH_CHECK_MSG with details.
# ---------------------------------------------------------------------------
_check_pytorch_wheel() {
    local tag="$1"
    local whl_index="https://download.pytorch.org/whl/${tag}"
    _PYTORCH_CHECK_MSG=""
    _PYTORCH_STATUS="ok"

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 \
        "${whl_index}/torch/" 2>/dev/null) || http_code="000"

    if [ "$http_code" = "200" ] || [ "$http_code" = "301" ] || [ "$http_code" = "302" ]; then
        _PYTORCH_CHECK_MSG="wheel index reachable (HTTP $http_code)"
        _PYTORCH_STATUS="ok"
    elif [ "$http_code" = "404" ]; then
        _PYTORCH_CHECK_MSG="wheel index 404: ${whl_index}/torch/ — no pre-built wheels for $tag"
        _PYTORCH_STATUS="degraded"
    else
        _PYTORCH_CHECK_MSG="wheel index unreachable (HTTP $http_code) — network issue or offline"
        _PYTORCH_STATUS="warning"
    fi
}

# ---------------------------------------------------------------------------
#  _check_disk <path> <warn_gb> <error_gb>
#  Sets _DISK_STATUS ("ok" / "degraded" / "error"),
#  _DISK_CHECK_MSG, and _DISK_AVAIL_GB.
# ---------------------------------------------------------------------------
_check_disk() {
    local path="${1:-.}" warn_gb="${2:-20}" error_gb="${3:-10}"
    _DISK_CHECK_MSG=""
    _DISK_STATUS="ok"
    _DISK_AVAIL_GB=$(df -BG "$path" 2>/dev/null \
        | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
    _DISK_AVAIL_GB="${_DISK_AVAIL_GB:-0}"

    if [ "$_DISK_AVAIL_GB" -ge "$warn_gb" ]; then
        _DISK_CHECK_MSG="${_DISK_AVAIL_GB}GB available (need ${warn_gb}GB)"
        _DISK_STATUS="ok"
    elif [ "$_DISK_AVAIL_GB" -ge "$error_gb" ]; then
        _DISK_CHECK_MSG="${_DISK_AVAIL_GB}GB available — tight (recommend ${warn_gb}GB)"
        _DISK_STATUS="degraded"
    else
        _DISK_CHECK_MSG="${_DISK_AVAIL_GB}GB available — insufficient (need ${error_gb}GB minimum)"
        _DISK_STATUS="error"
    fi
}

# ---------------------------------------------------------------------------
#  resolve_env_plan <plan_file> [--force]
#
#  Core resolver.  Detects hardware, derives versions, validates, writes JSON.
#  Returns 0 on success (may have warnings/degraded), 1 on hard errors.
#
#  If plan_file exists and hardware matches, reuses it (unless --force).
# ---------------------------------------------------------------------------
resolve_env_plan() {
    local plan_file="${1:?Usage: resolve_env_plan <plan_file> [--force]}"
    local force="${2:-}"

    # --- Reuse existing plan if hardware unchanged ---
    if [ -f "$plan_file" ] && [ "$force" != "--force" ]; then
        local existing_driver existing_cuda existing_target
        existing_driver=$(_plan_get "$plan_file" "driver_version") || true
        existing_cuda=$(_plan_get "$plan_file" "cuda_version") || true
        existing_target=$(_plan_get "$plan_file" "target_driver") || true
        local cur_driver cur_cuda cur_target="${TARGET_DRIVER:-}"
        cur_driver=$(detect_driver_version 2>/dev/null) || cur_driver=""
        cur_cuda=$(detect_cuda_version 2>/dev/null) || cur_cuda=""
        local existing_fa_ver
        existing_fa_ver=$(_plan_get "$plan_file" "flash_attn_version") || true

        if [ -n "$existing_driver" ] && [ "$existing_driver" = "$cur_driver" ] \
           && [ -n "$existing_cuda" ] && [ "$existing_cuda" = "$cur_cuda" ] \
           && [ "$existing_target" = "$cur_target" ] \
           && [ -n "$existing_fa_ver" ]; then
            log_info "Reusing existing env plan: $plan_file (hardware unchanged)"
            log_info "  Use --force-resolve to regenerate"
            return 0
        fi
        if [ -z "$existing_fa_ver" ]; then
            log_info "Regenerating env plan (missing flash-attn fields from older version)"
        fi
    fi

    log_step "Resolving environment version plan..."

    # ===== Phase 1: Detect hardware =====
    local driver_ver cuda_ver
    local target_driver="${TARGET_DRIVER:-}"
    driver_ver=$(detect_driver_version 2>/dev/null) || driver_ver=""
    cuda_ver=$(detect_cuda_version 2>/dev/null) || cuda_ver=""
    _detect_gpu_info

    # When TARGET_DRIVER is set, use it for NGC resolution instead of host driver.
    # Host driver is still recorded for reference; NGC selection uses ngc_driver_ver.
    local ngc_driver_ver="${target_driver:-$driver_ver}"
    if [ -n "$target_driver" ]; then
        log_info "Target driver override: $target_driver (host driver: ${driver_ver:-not detected})"
    fi

    # ===== Phase 2: Derive versions (no network) =====
    local errors=() warnings=()
    local ngc_tag="" ngc_min_drv="" ngc_trt="" ngc_cuda="" ngc_pyver="" ngc_size=""
    local driver_status="ok" driver_msg=""
    local cuda_status="ok" cuda_msg=""
    local ngc_status="ok" ngc_msg=""

    # 2a: Driver check (against ngc_driver_ver — the version used for NGC selection)
    if [ -z "$ngc_driver_ver" ]; then
        driver_status="error"
        driver_msg="NVIDIA driver not detected"
        errors+=("$driver_msg")
    elif ! _driver_ge "$ngc_driver_ver" "$_QWEN3_MIN_DRIVER"; then
        driver_status="error"
        driver_msg="Driver $ngc_driver_ver < $_QWEN3_MIN_DRIVER"
        errors+=("$driver_msg")
    else
        driver_msg="$ngc_driver_ver >= $_QWEN3_MIN_DRIVER"
    fi

    # 2b: CUDA check
    if [ -z "$cuda_ver" ]; then
        cuda_status="error"
        cuda_msg="CUDA toolkit not detected (nvcc / nvidia-smi)"
        errors+=("$cuda_msg")
    else
        cuda_msg="CUDA $cuda_ver detected"
    fi

    # 2c: NGC resolution (uses ngc_driver_ver which may be TARGET_DRIVER override)
    local ngc_entry=""
    if [ "$driver_status" = "ok" ]; then
        ngc_entry=$(_resolve_best_entry "$ngc_driver_ver" 2>/dev/null) || true
        if [ -n "$ngc_entry" ]; then
            read -r ngc_tag ngc_min_drv ngc_trt ngc_cuda ngc_pyver ngc_size <<< "$ngc_entry"
            ngc_msg="NGC $ngc_tag (CUDA $ngc_cuda, Python $ngc_pyver)"
        else
            ngc_status="error"
            ngc_msg="No compatible NGC container for driver $driver_ver (need driver >= $_QWEN3_MIN_DRIVER)"
            errors+=("$ngc_msg")
        fi
    else
        ngc_status="error"
        ngc_msg="Skipped (driver check failed)"
    fi

    # 2d: Python version
    local python_ver="${ngc_pyver:-3.12}"

    # 2e: PyTorch CUDA tag + flash-attn compatibility pre-check
    local torch_tag="" system_cuda_tag="" flash_cuda_tag="" flash_needs_downgrade=false
    if [ "$cuda_status" = "ok" ]; then
        system_cuda_tag=$(cuda_to_torch_tag "$cuda_ver")
        torch_tag="$system_cuda_tag"
        flash_cuda_tag=$(_flash_attn_compatible_tag "$cuda_ver") || flash_needs_downgrade=true
    fi

    # 2f: NGC image URI (single image for both Phase B and C)
    local ngc_image=""
    if [ -n "$ngc_tag" ]; then
        ngc_image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_PY3_SUFFIX}"
    fi

    # ===== Phase 3: Verify feasibility (network, no installs) =====

    # 3a: PyTorch wheel
    local pytorch_status="ok" pytorch_msg=""
    local pytorch_fallback="" pytorch_fallback_risk=""
    if [ -n "$torch_tag" ]; then
        _check_pytorch_wheel "$torch_tag"
        pytorch_status="$_PYTORCH_STATUS"
        pytorch_msg="$_PYTORCH_CHECK_MSG"
        if [ "$pytorch_status" = "degraded" ]; then
            pytorch_fallback="pip_default_index"
            pytorch_fallback_risk="pip may resolve a different CUDA version (e.g. cu128 instead of $torch_tag)"
        fi
    else
        pytorch_status="error"
        pytorch_msg="Cannot determine PyTorch CUDA tag (CUDA not detected)"
        errors+=("$pytorch_msg")
    fi

    # 3b: flash-attn wheel availability
    local flash_status="deferred" flash_msg=""
    local flash_fallback="" flash_fallback_risk=""
    if [ -z "$torch_tag" ]; then
        flash_status="deferred"
        flash_msg="Will verify after PyTorch installation"
    elif ! $flash_needs_downgrade; then
        flash_status="ok"
        flash_msg="flash-attn $_FLASH_ATTN_VERSION wheel available ($flash_cuda_tag)"
    else
        flash_status="degraded"
        flash_msg="CUDA $cuda_ver ($system_cuda_tag) exceeds flash-attn $_FLASH_ATTN_VERSION max (CUDA <= $_FLASH_ATTN_MAX_CUDA)"
        flash_fallback="downgrade_cuda_tag"
        flash_fallback_risk="Downgrade PyTorch CUDA tag $system_cuda_tag -> $flash_cuda_tag (safe on CUDA $cuda_ver, enables flash-attn)"
        warnings+=("flash-attn: $flash_msg")
    fi

    # 3c: Disk space
    local disk_status="" disk_msg=""
    local plan_dir
    plan_dir=$(dirname "$plan_file")
    mkdir -p "$plan_dir" 2>/dev/null || true
    _check_disk "$plan_dir" 20 10
    disk_status="$_DISK_STATUS"
    disk_msg="$_DISK_CHECK_MSG"
    if [ "$disk_status" = "error" ]; then
        errors+=("Disk: $disk_msg")
    elif [ "$disk_status" = "degraded" ]; then
        warnings+=("Disk: $disk_msg")
    fi

    # ===== Phase 4: Collect errors into list =====
    if [ "$pytorch_status" = "degraded" ]; then
        warnings+=("PyTorch: $pytorch_msg")
    fi

    # ===== Phase 5: Write JSON =====
    local timestamp
    timestamp=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')

    local errors_json="[]" warnings_json="[]"
    if [ ${#errors[@]} -gt 0 ]; then
        errors_json="["
        local first=true
        for e in "${errors[@]}"; do
            $first || errors_json+=", "
            errors_json+="\"$(echo "$e" | sed 's/"/\\"/g')\""
            first=false
        done
        errors_json+="]"
    fi
    if [ ${#warnings[@]} -gt 0 ]; then
        warnings_json="["
        local first=true
        for w in "${warnings[@]}"; do
            $first || warnings_json+=", "
            warnings_json+="\"$(echo "$w" | sed 's/"/\\"/g')\""
            first=false
        done
        warnings_json+="]"
    fi

    # Build checks object
    local checks_json
    checks_json=$(cat <<CHECKS_EOF
{
    "driver_sufficient": { "status": "$driver_status", "message": "$(echo "$driver_msg" | sed 's/"/\\"/g')" },
    "cuda_detected": { "status": "$cuda_status", "message": "$(echo "$cuda_msg" | sed 's/"/\\"/g')" },
    "pytorch_wheel_available": {
      "status": "$pytorch_status",
      "message": "$(echo "$pytorch_msg" | sed 's/"/\\"/g')"$(
      if [ "$pytorch_status" = "degraded" ]; then
        printf ',\n      "fallback": "%s",\n      "fallback_risk": "%s"' \
            "$pytorch_fallback" "$(echo "$pytorch_fallback_risk" | sed 's/"/\\"/g')"
      fi)
    },
    "flash_attn_wheel_available": {
      "status": "$flash_status",
      "message": "$(echo "$flash_msg" | sed 's/"/\\"/g')"$(
      if [ "$flash_status" = "degraded" ]; then
        printf ',\n      "fallback": "%s",\n      "fallback_risk": "%s"' \
            "$flash_fallback" "$(echo "$flash_fallback_risk" | sed 's/"/\\"/g')"
      fi)
    },
    "ngc_container_compatible": { "status": "$ngc_status", "message": "$(echo "$ngc_msg" | sed 's/"/\\"/g')" },
    "disk_space": { "status": "$disk_status", "message": "$(echo "$disk_msg" | sed 's/"/\\"/g')" }
  }
CHECKS_EOF
    )

    cat > "$plan_file" <<JSON_EOF
{
  "resolved_at": "$timestamp",
  "target_driver": "${target_driver}",
  "hardware": {
    "driver_version": "$driver_ver",
    "cuda_version": "$cuda_ver",
    "gpu_name": "$(echo "$_GPU_NAME" | sed 's/"/\\"/g')",
    "gpu_memory_mb": ${_GPU_MEM_MB:-0}
  },
  "versions": {
    "python": "$python_ver",
    "pytorch_cuda_tag": "$torch_tag",
    "system_cuda_tag": "$system_cuda_tag",
    "pytorch_wheel_index": "https://download.pytorch.org/whl/${torch_tag}",
    "flash_attn_version": "$_FLASH_ATTN_VERSION",
    "flash_attn_compatible_tag": "$flash_cuda_tag",
    "ngc_tag": "$ngc_tag",
    "ngc_cuda_version": "$ngc_cuda",
    "ngc_image": "$ngc_image"
  },
  "checks": $checks_json,
  "warnings": $warnings_json,
  "errors": $errors_json,
  "user_decisions": {}
}
JSON_EOF

    log_info "Plan written: $plan_file"

    if [ ${#errors[@]} -gt 0 ]; then
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  _plan_get <plan_file> <key>
#  Lightweight JSON value extraction (no jq dependency).
#  Handles both "key": "value" (string) and "key": 123 (number/bool).
# ---------------------------------------------------------------------------
_plan_get() {
    local file="$1" key="$2"
    local val
    val=$(sed -n 's/.*"'"$key"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$file" | head -1)
    if [ -z "$val" ]; then
        val=$(sed -n 's/.*"'"$key"'"[[:space:]]*:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' "$file" | head -1)
    fi
    echo "$val"
}

# ---------------------------------------------------------------------------
#  _plan_get_status <plan_file> <check_name>
#  Extracts the status of a check item.
# ---------------------------------------------------------------------------
_plan_get_status() {
    local file="$1" check="$2"
    # Match the check block and extract its status
    awk -v check="$check" '
        $0 ~ "\"" check "\"" { found=1 }
        found && /"status"/ {
            gsub(/.*"status"[[:space:]]*:[[:space:]]*"/, "")
            gsub(/".*/, "")
            print
            exit
        }
    ' "$file"
}

# ---------------------------------------------------------------------------
#  read_env_plan_val <plan_file> <key>
#  Public API for reading plan values.
# ---------------------------------------------------------------------------
read_env_plan_val() {
    _plan_get "$1" "$2"
}

# ---------------------------------------------------------------------------
#  print_env_plan <plan_file>
#  Pretty-prints the resolved plan for the user.
# ---------------------------------------------------------------------------
print_env_plan() {
    local plan_file="$1"
    [ -f "$plan_file" ] || { log_error "Plan file not found: $plan_file"; return 1; }

    local drv cuda gpu gpu_mem py tag ngc ngc_img
    drv=$(_plan_get "$plan_file" "driver_version")
    cuda=$(_plan_get "$plan_file" "cuda_version")
    gpu=$(_plan_get "$plan_file" "gpu_name")
    gpu_mem=$(_plan_get "$plan_file" "gpu_memory_mb")
    py=$(_plan_get "$plan_file" "python")
    tag=$(_plan_get "$plan_file" "pytorch_cuda_tag")
    ngc=$(_plan_get "$plan_file" "ngc_tag")
    ngc_img=$(_plan_get "$plan_file" "ngc_image")

    local gpu_gb=""
    if [ -n "$gpu_mem" ] && [ "$gpu_mem" != "0" ]; then
        gpu_gb=" ($(( gpu_mem / 1024 )) GB)"
    fi

    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║          Environment Version Plan                       ║${_CLR_RESET}"
    echo -e "${_CLR_BLUE}╚══════════════════════════════════════════════════════════╝${_CLR_RESET}"
    echo ""
    local target_drv
    target_drv=$(_plan_get "$plan_file" "target_driver")

    echo "  Hardware:"
    echo "    Driver:   ${drv:-?}"
    if [ -n "$target_drv" ]; then
        echo "    Target:   ${target_drv}  (NGC selection uses this)"
    fi
    echo "    CUDA:     ${cuda:-?}"
    echo "    GPU:      ${gpu:-?}${gpu_gb}"
    echo ""
    local sys_tag fa_ver fa_tag
    sys_tag=$(_plan_get "$plan_file" "system_cuda_tag")
    fa_ver=$(_plan_get "$plan_file" "flash_attn_version")
    fa_tag=$(_plan_get "$plan_file" "flash_attn_compatible_tag")

    echo "  Versions:"
    echo "    Python:   ${py:-?}    (from NGC matrix)"
    if [ -n "$sys_tag" ] && [ "$sys_tag" != "$tag" ]; then
        echo "    PyTorch:  ${tag:-?}    (downgraded from ${sys_tag} for flash-attn)"
    else
        echo "    PyTorch:  ${tag:-?}    (CUDA ${cuda:-?})"
    fi
    if [ -n "$fa_ver" ]; then
        if [ -n "$fa_tag" ]; then
            echo "    flash-attn: ${fa_ver}  (${fa_tag})"
        else
            echo "    flash-attn: ${fa_ver}  (no wheel for ${sys_tag:-$tag})"
        fi
    fi
    echo "    NGC:      ${ngc:-?}"
    echo ""
    echo "  Container (Phase B + C):"
    echo "    Image:    ${ngc_img:-?}"
    echo ""
    echo "  Checks:"

    local checks=("driver_sufficient" "cuda_detected" "pytorch_wheel_available"
                   "flash_attn_wheel_available" "ngc_container_compatible" "disk_space")
    local labels=("Driver >= $_QWEN3_MIN_DRIVER"
                  "CUDA detected"
                  "PyTorch wheel ($tag)"
                  "flash-attn wheel"
                  "NGC container (driver >= $_QWEN3_MIN_DRIVER)"
                  "Disk space")

    local i=0
    for check in "${checks[@]}"; do
        local st msg
        st=$(_plan_get_status "$plan_file" "$check")
        msg=$(awk -v check="$check" '
            $0 ~ "\"" check "\"" { found=1 }
            found && /"message"/ {
                gsub(/.*"message"[[:space:]]*:[[:space:]]*"/, "")
                gsub(/".*/, "")
                print
                exit
            }
        ' "$plan_file")

        case "$st" in
            ok)       echo -e "    ${_CLR_GREEN}[OK]${_CLR_RESET} ${labels[$i]}" ;;
            degraded) echo -e "    ${_CLR_YELLOW}[!!]${_CLR_RESET} ${labels[$i]}: $msg" ;;
            deferred) echo -e "    ${_CLR_BLUE}[--]${_CLR_RESET} ${labels[$i]}: $msg" ;;
            error)    echo -e "    ${_CLR_RED}[ERR]${_CLR_RESET} ${labels[$i]}: $msg" ;;
            warning)  echo -e "    ${_CLR_YELLOW}[~~]${_CLR_RESET} ${labels[$i]}: $msg" ;;
            *)        echo "    [??] ${labels[$i]}: unknown status '$st'" ;;
        esac
        i=$((i + 1))
    done

    echo ""
    echo "  Plan saved: $plan_file"
    echo ""
}

# ---------------------------------------------------------------------------
#  confirm_env_plan <plan_file>
#
#  1. Prints the plan.
#  2. For each "degraded" check, asks user to choose a fallback or abort.
#  3. Writes user decisions back to plan JSON.
#  4. If errors exist, exits with code 1.
#  5. Otherwise waits for Enter to proceed.
# ---------------------------------------------------------------------------
confirm_env_plan() {
    local plan_file="$1"
    [ -f "$plan_file" ] || { log_error "Plan file not found: $plan_file"; return 1; }

    print_env_plan "$plan_file"

    # Check for hard errors
    local error_count
    error_count=$(grep -c '"status": "error"' "$plan_file" 2>/dev/null) || error_count=0
    if [ "$error_count" -gt 0 ]; then
        log_error "Environment plan has $error_count unresolvable error(s) — cannot proceed"
        log_error "Fix the issues above and re-run"
        return 1
    fi

    # Handle degraded items interactively
    local has_degraded=false
    local checks=("flash_attn_wheel_available" "pytorch_wheel_available" "disk_space")
    for check in "${checks[@]}"; do
        local st
        st=$(_plan_get_status "$plan_file" "$check")
        if [ "$st" = "degraded" ]; then
            has_degraded=true
            if [ "$check" = "flash_attn_wheel_available" ]; then
                _prompt_flash_attn_decision "$plan_file" || return 1
            else
                _prompt_degraded_decision "$plan_file" "$check" || return 1
            fi
        fi
    done

    # Wait for user confirmation (auto-proceed after 30s)
    if [ -t 0 ]; then
        echo -e "  Press ${_CLR_GREEN}Enter${_CLR_RESET} to proceed, or ${_CLR_RED}Ctrl+C${_CLR_RESET} to abort (auto-continue in 30s)..."
        read -r -t 30 || echo ""
    else
        log_info "Non-interactive mode — proceeding with plan"
    fi

    return 0
}

# ---------------------------------------------------------------------------
#  _prompt_degraded_decision <plan_file> <check_name>
#  Interactive prompt for a degraded check. Writes decision to plan.
# ---------------------------------------------------------------------------
_prompt_degraded_decision() {
    local plan_file="$1" check="$2"

    local msg fallback risk
    msg=$(awk -v check="$check" '
        $0 ~ "\"" check "\"" { found=1 }
        found && /"message"/ { gsub(/.*"message"[[:space:]]*:[[:space:]]*"/, ""); gsub(/".*/, ""); print; exit }
    ' "$plan_file")
    fallback=$(awk -v check="$check" '
        $0 ~ "\"" check "\"" { found=1 }
        found && /"fallback"/ { gsub(/.*"fallback"[[:space:]]*:[[:space:]]*"/, ""); gsub(/".*/, ""); print; exit }
    ' "$plan_file")
    risk=$(awk -v check="$check" '
        $0 ~ "\"" check "\"" { found=1 }
        found && /"fallback_risk"/ { gsub(/.*"fallback_risk"[[:space:]]*:[[:space:]]*"/, ""); gsub(/".*/, ""); print; exit }
    ' "$plan_file")

    echo ""
    echo -e "  ${_CLR_YELLOW}[!!] $check${_CLR_RESET}"
    echo "       Problem:  $msg"
    if [ -n "$fallback" ]; then
        echo "       Fallback: $fallback"
        [ -n "$risk" ] && echo "       Risk:     $risk"
    fi
    echo ""

    if [ ! -t 0 ]; then
        log_warn "Non-interactive mode — using fallback for $check"
        _write_decision "$plan_file" "$check" "fallback"
        return 0
    fi

    local choice=""
    while true; do
        echo -n "       [Y] Use fallback  [S] Skip  [N] Abort  > "
        read -r choice
        case "$choice" in
            [Yy]) _write_decision "$plan_file" "$check" "fallback"; return 0 ;;
            [Ss]) _write_decision "$plan_file" "$check" "skip"; return 0 ;;
            [Nn]) log_error "Aborted by user"; return 1 ;;
            *)    echo "       Please enter Y, S, or N" ;;
        esac
    done
}

# ---------------------------------------------------------------------------
#  _prompt_flash_attn_decision <plan_file>
#  Specialized prompt for flash-attn degraded status (CUDA too new).
#  Offers [D] Downgrade CUDA tag or [S] Skip flash-attn.
#  On downgrade: updates pytorch_cuda_tag in the plan JSON.
# ---------------------------------------------------------------------------
_prompt_flash_attn_decision() {
    local plan_file="$1"
    local check="flash_attn_wheel_available"

    local sys_tag fa_tag msg
    sys_tag=$(_plan_get "$plan_file" "system_cuda_tag")
    fa_tag=$(_plan_get "$plan_file" "flash_attn_compatible_tag")
    msg=$(awk -v check="$check" '
        $0 ~ "\"" check "\"" { found=1 }
        found && /"message"/ { gsub(/.*"message"[[:space:]]*:[[:space:]]*"/, ""); gsub(/".*/, ""); print; exit }
    ' "$plan_file")

    echo ""
    echo -e "  ${_CLR_YELLOW}[!!] flash-attn compatibility${_CLR_RESET}"
    echo "       Problem:  $msg"
    echo ""
    echo "       Option D: Downgrade PyTorch CUDA tag ${sys_tag} -> ${fa_tag}"
    echo "                 (backward compatible, enables flash-attn pre-built wheel)"
    echo "       Option B: Keep ${sys_tag}, build flash-attn from source"
    echo "                 (slow: 10-30 min, may fail; needs ninja + CUDA toolkit)"
    echo "       Option S: Keep ${sys_tag}, skip flash-attn"
    echo "                 (Talker backbone falls back to SDPA/manual attention)"
    echo ""

    if [ ! -t 0 ]; then
        log_warn "Non-interactive mode — downgrading CUDA tag for flash-attn"
        _write_decision "$plan_file" "$check" "downgrade"
        _apply_flash_attn_downgrade "$plan_file" "$fa_tag"
        return 0
    fi

    local choice=""
    while true; do
        echo -n "       [D] Downgrade  [B] Build from source  [S] Skip  [N] Abort  > "
        read -r choice
        case "$choice" in
            [Dd])
                _write_decision "$plan_file" "$check" "downgrade"
                _apply_flash_attn_downgrade "$plan_file" "$fa_tag"
                return 0
                ;;
            [Bb])
                _write_decision "$plan_file" "$check" "source_build"
                return 0
                ;;
            [Ss])
                _write_decision "$plan_file" "$check" "skip"
                return 0
                ;;
            [Nn]) log_error "Aborted by user"; return 1 ;;
            *)    echo "       Please enter D, B, S, or N" ;;
        esac
    done
}

# ---------------------------------------------------------------------------
#  _apply_flash_attn_downgrade <plan_file> <new_tag>
#  Updates pytorch_cuda_tag and pytorch_wheel_index in the plan JSON
#  after user chooses to downgrade for flash-attn compatibility.
# ---------------------------------------------------------------------------
_apply_flash_attn_downgrade() {
    local file="$1" new_tag="$2"
    local old_tag
    old_tag=$(_plan_get "$file" "pytorch_cuda_tag")

    sed -i 's|"pytorch_cuda_tag": "'"$old_tag"'"|"pytorch_cuda_tag": "'"$new_tag"'"|' "$file"
    sed -i 's|"pytorch_wheel_index": "https://download.pytorch.org/whl/'"$old_tag"'"|"pytorch_wheel_index": "https://download.pytorch.org/whl/'"$new_tag"'"|' "$file"

    log_info "PyTorch CUDA tag downgraded: $old_tag -> $new_tag (flash-attn compatible)"
}

# ---------------------------------------------------------------------------
#  _write_decision <plan_file> <check_name> <decision>
#  Sets or updates user decision in the plan JSON's user_decisions block.
#  If the key already exists, only its value is updated (no duplicate keys).
# ---------------------------------------------------------------------------
_write_decision() {
    local file="$1" check="$2" decision="$3"
    # Replace empty user_decisions {} with the decision
    if grep -q '"user_decisions": {}' "$file"; then
        sed -i 's/"user_decisions": {}/"user_decisions": { "'"$check"'": "'"$decision"'" }/' "$file"
    elif grep -q '"user_decisions":' "$file"; then
        # If this check key already exists, update its value only (avoid duplicate keys)
        if awk '/"user_decisions":/,/}/' "$file" | grep -q '"'"$check"'":'; then
            sed -i '/"user_decisions":/,/}/ s/"'"$check"'": "[^"]*"/"'"$check"'": "'"$decision"'"/' "$file"
        else
            # Add new key before closing brace
            sed -i '/"user_decisions":/,/}/ s/}$/, "'"$check"'": "'"$decision"'" }/' "$file"
        fi
    fi
    log_info "Decision recorded: $check = $decision"
}

# ---------------------------------------------------------------------------
#  _handle_degraded <plan_file> <check_name>
#
#  Called at install time when a deferred check becomes degraded.
#  Reads existing decision or prompts user.
#  Returns 0 if fallback chosen, 1 if skip, 2 if abort.
#
#  Usage in pip.sh:
#    _handle_degraded "$PLAN_FILE" "flash_attn_wheel_available"
#    case $? in
#      0) <use fallback> ;;
#      1) <skip> ;;
#      2) return 1 ;;
#    esac
# ---------------------------------------------------------------------------
_handle_degraded() {
    local plan_file="$1" check="$2"

    # Check for existing decision
    local existing
    existing=$(awk -v check="$check" '
        /"user_decisions"/ { found=1 }
        found && $0 ~ "\"" check "\"" {
            gsub(/.*"'"$check"'"[[:space:]]*:[[:space:]]*"/, "")
            gsub(/".*/, "")
            print
            exit
        }
    ' "$plan_file" 2>/dev/null)

    case "$existing" in
        fallback) return 0 ;;
        skip)     return 1 ;;
    esac

    # No prior decision — prompt now
    if [ -t 0 ]; then
        echo ""
        echo -e "  ${_CLR_YELLOW}[!!] $check needs your decision${_CLR_RESET}"
        local choice=""
        while true; do
            echo -n "       [Y] Use fallback  [S] Skip  [N] Abort  > "
            read -r choice
            case "$choice" in
                [Yy])
                    [ -f "$plan_file" ] && _write_decision "$plan_file" "$check" "fallback"
                    return 0
                    ;;
                [Ss])
                    [ -f "$plan_file" ] && _write_decision "$plan_file" "$check" "skip"
                    return 1
                    ;;
                [Nn]) return 2 ;;
                *)    echo "       Please enter Y, S, or N" ;;
            esac
        done
    else
        log_warn "Non-interactive: defaulting to skip for $check"
        return 1
    fi
}
