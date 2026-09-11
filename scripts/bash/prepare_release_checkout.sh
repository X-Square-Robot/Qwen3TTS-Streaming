#!/usr/bin/env bash
# Prepare an exact release candidate checkout.
#
# The candidate pipeline uses a local version tag so hatch-vcs produces the
# same wheel version that the eventual promoted tag will produce. The tag is
# deliberately never pushed by this script.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: %s <vX.Y.Z[preN]> [commit-or-ref]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

release_version="$1"
release_ref="${2:-HEAD}"

if [[ ! "$release_version" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)((a|b|rc)(0|[1-9][0-9]*))?$ ]]; then
    printf 'Unsupported release version: %s\n' "$release_version" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

status_args=(--porcelain)
if [[ "${RELEASE_ALLOW_UNTRACKED:-0}" == "1" ]]; then
    status_args+=(--untracked-files=no)
fi

if [[ -n "$(git status "${status_args[@]}")" ]]; then
    printf 'Release checkout must be clean before selecting a candidate commit\n' >&2
    exit 1
fi

target_commit=""
if [[ "$release_ref" == HEAD ]]; then
    target_commit="$(git rev-parse --verify HEAD^{commit})"
else
    target_commit="$(git rev-parse --verify "${release_ref}^{commit}" 2>/dev/null || true)"
    if [[ -z "$target_commit" ]]; then
        if git fetch --no-tags origin "$release_ref" 2>/dev/null; then
            target_commit="$(git rev-parse --verify FETCH_HEAD^{commit} 2>/dev/null || true)"
        fi
        if [[ -z "$target_commit" ]]; then
            target_commit="$(
                git rev-parse --verify "origin/${release_ref}^{commit}" 2>/dev/null ||
                    git rev-parse --verify "${release_ref}^{commit}" 2>/dev/null ||
                    true
            )"
        fi
    fi
fi

if [[ -z "$target_commit" ]]; then
    printf 'Could not resolve release source ref: %s\n' "$release_ref" >&2
    exit 1
fi

git checkout --detach "$target_commit" >/dev/null
head_commit="$(git rev-parse --verify HEAD^{commit})"
if [[ "$head_commit" != "$target_commit" ]]; then
    printf 'Release checkout resolved to %s, expected %s\n' \
        "$head_commit" "$target_commit" >&2
    exit 1
fi

if git show-ref --verify --quiet "refs/tags/$release_version"; then
    existing_commit="$(git rev-parse --verify "${release_version}^{commit}")"
    if [[ "$existing_commit" != "$head_commit" ]]; then
        printf 'Release tag %s already points to %s, expected %s\n' \
            "$release_version" "$existing_commit" "$head_commit" >&2
        exit 1
    fi
    printf 'Using existing local/formal tag %s at %s\n' \
        "$release_version" "$head_commit"
else
    git tag "$release_version" "$head_commit"
    printf 'Created local candidate tag %s at %s\n' \
        "$release_version" "$head_commit"
fi

if [[ -n "$(git status "${status_args[@]}")" ]]; then
    printf 'Release checkout became dirty while preparing candidate tag\n' >&2
    exit 1
fi

printf 'RELEASE_VERSION=%s\nRELEASE_COMMIT_SHA=%s\n' \
    "$release_version" "$head_commit"
