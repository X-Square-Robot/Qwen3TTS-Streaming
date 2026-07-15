#!/bin/bash
# ===========================================================================
#  release_client_wheel.sh — Build the distributable client SDK wheel on a tag
#
#  Channel 2 of client delivery (channel 1 is `pip install "... @ git+ssh://
#  ...@<tag>#subdirectory=client"`). The wheel version is derived from the git
#  tag via hatch-vcs, so engine image and wheel built from the same tag carry
#  the same version — that pairing is the whole point. This script therefore
#  refuses to build from a dirty tree or an untagged commit.
#
#  Usage:
#    git tag v0.2.0 && bash scripts/bash/release_client_wheel.sh
#    bash scripts/bash/release_client_wheel.sh --out /tmp/delivery
#
#  Options:
#    --out DIR   Output directory (default: client/dist)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/lib/logging.sh"

OUT_DIR="$REPO_ROOT/client/dist"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^#  \{0,1\}//'
            exit 0
            ;;
        *)
            log_error "unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    log_error "working tree is dirty; commit or stash before building a release wheel"
    exit 1
fi

# --match "v[0-9]*": only version tags qualify — being on a personal marker
# tag (rime/xxx) must not look like being on a release tag.
tag="$(git -C "$REPO_ROOT" describe --tags --exact-match --match "v[0-9]*" HEAD 2>/dev/null)" || {
    log_error "HEAD is not on a version tag (got '$(git -C "$REPO_ROOT" describe --tags --always --match "v[0-9]*")')."
    log_error "Release wheels must be built on a tag: git tag vX.Y.Z && re-run."
    exit 1
}

log_info "Building client wheel at tag $tag ..."
mkdir -p "$OUT_DIR"
python3 -m pip wheel --no-deps --wheel-dir "$OUT_DIR" "$REPO_ROOT/client"

wheel_file="$(ls -t "$OUT_DIR"/qwen3_tts_client-*.whl | head -1)"
wheel_version="$(basename "$wheel_file" | cut -d- -f2)"
expected="${tag#v}"
if [[ "$wheel_version" != "$expected" ]]; then
    log_error "wheel version $wheel_version != tag $tag — check hatch-vcs setup"
    exit 1
fi

log_info "Release wheel ready: $wheel_file"
log_info "Deliver it alongside engine image tag $tag (engine /health reports 'version': $tag)"
