#!/bin/bash
# ===========================================================================
#  tools.sh — Convenience aggregator
#
#  Sources every module under lib/ so callers can just:
#      source "path/to/tools.sh"
#  and get all utility functions at once.
#
#  Alternatively, scripts may source individual lib/*.sh files directly
#  for a lighter footprint.
# ===========================================================================

_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_TOOLS_DIR}/lib/logging.sh"
source "${_TOOLS_DIR}/lib/utils.sh"
source "${_TOOLS_DIR}/lib/prerequisites.sh"
source "${_TOOLS_DIR}/lib/venv.sh"
source "${_TOOLS_DIR}/lib/pip.sh"
source "${_TOOLS_DIR}/lib/mirrors.sh"
source "${_TOOLS_DIR}/lib/network.sh"
source "${_TOOLS_DIR}/lib/docker.sh"
source "${_TOOLS_DIR}/lib/env_plan.sh"
source "${_TOOLS_DIR}/lib/cross_host.sh"
source "${_TOOLS_DIR}/lib/trtexec_runner.sh"
source "${_TOOLS_DIR}/lib/build_pipeline.sh"
source "${_TOOLS_DIR}/lib/triton.sh"
source "${_TOOLS_DIR}/lib/engine.sh"
source "${_TOOLS_DIR}/lib/status.sh"
