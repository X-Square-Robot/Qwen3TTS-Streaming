#!/bin/bash
# ===========================================================================
#  ngc_updater.sh — Update the NGC compatibility matrix conf
#
#  Thin wrapper: the fetch + HTML parsing + merge live in
#  scripts/python/ngc_matrix_update.py (bash orchestrates, Python parses).
#
#  Functions: update_ngc_matrix [conf_path]
#  Best-effort: on any failure the existing conf is left untouched.
# ===========================================================================

[[ -n "${_LIB_NGC_UPDATER_LOADED:-}" ]] && return 0
_LIB_NGC_UPDATER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"

# ---------------------------------------------------------------------------
#  update_ngc_matrix [conf_path]
#  Fetches the latest NVIDIA Triton release-notes matrix and merges it into the
#  local conf (default: scripts/bash/ngc_matrix.conf).  Never overwrites on
#  failure.  Returns the helper's exit code.
# ---------------------------------------------------------------------------
update_ngc_matrix() {
    local conf_path="${1:-}"
    if [ -z "$conf_path" ]; then
        conf_path="$(cd "${_LIB_DIR}/.." && pwd)/ngc_matrix.conf"
    fi

    local repo_root helper
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    helper="${repo_root}/scripts/python/ngc_matrix_update.py"

    if [ ! -f "$conf_path" ]; then
        log_error "Matrix conf not found: $conf_path"
        return 1
    fi
    if [ ! -f "$helper" ]; then
        log_error "Missing helper: $helper"
        return 1
    fi

    log_step "Updating NGC compatibility matrix"
    log_info "Target: $conf_path"
    python3 "$helper" --conf "$conf_path"
}
