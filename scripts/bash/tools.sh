#!/usr/bin/env bash
# tools.sh — Source all lib/ modules in one call.
# Usage: source "${SCRIPT_DIR}/tools.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for _lib in "${SCRIPT_DIR}/lib/"*.sh; do
    # shellcheck source=/dev/null
    source "${_lib}"
done
unset _lib
