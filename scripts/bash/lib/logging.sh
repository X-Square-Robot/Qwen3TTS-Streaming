#!/bin/bash
# ===========================================================================
#  logging.sh — Colored log output
#
#  Functions: log_info, log_warn, log_error, log_step
#  Usage:     source this file, then call log_* anywhere.
# ===========================================================================

[[ -n "${_LIB_LOGGING_LOADED:-}" ]] && return 0
_LIB_LOGGING_LOADED=1

_CLR_RED='\033[0;31m'
_CLR_GREEN='\033[0;32m'
_CLR_YELLOW='\033[0;33m'
_CLR_BLUE='\033[0;34m'
_CLR_RESET='\033[0m'

log_info()  { echo -e "${_CLR_GREEN}[INFO]${_CLR_RESET}  $*" >&2; }
log_warn()  { echo -e "${_CLR_YELLOW}[WARN]${_CLR_RESET}  $*" >&2; }
log_error() { echo -e "${_CLR_RED}[ERROR]${_CLR_RESET} $*" >&2; }
log_step()  { echo -e "${_CLR_BLUE}[STEP]${_CLR_RESET}  $*" >&2; }
