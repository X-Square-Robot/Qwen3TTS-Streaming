#!/usr/bin/env bash
# Promote an already-verified candidate image to its immutable release tag.
#
# The candidate and release references must be in the same registry/repository.
# Existing release tags are never overwritten: their remote digest must match
# the candidate digest exactly.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s <candidate-image> <release-image> [expected-digest]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 2
fi

candidate_image="$1"
release_image="$2"
expected_candidate_digest="${3:-}"

image_digest() {
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$1" |
        awk -F@ 'NF == 2 { print $2; exit }'
}

docker pull "$candidate_image" >/dev/null
candidate_digest="$(image_digest "$candidate_image")"
if [[ -z "$candidate_digest" ]]; then
    printf 'Candidate image has no registry digest: %s\n' "$candidate_image" >&2
    exit 1
fi
if [[ -n "$expected_candidate_digest" && "$candidate_digest" != "$expected_candidate_digest" ]]; then
    printf 'Candidate image digest changed after build: %s (expected=%s actual=%s)\n' \
        "$candidate_image" "$expected_candidate_digest" "$candidate_digest" >&2
    exit 1
fi

manifest_error="$(mktemp)"
trap 'rm -f "$manifest_error"' EXIT
if docker manifest inspect "$release_image" >/dev/null 2>"$manifest_error"; then
    docker pull "$release_image" >/dev/null
    release_digest="$(image_digest "$release_image")"
    if [[ "$release_digest" != "$candidate_digest" ]]; then
        printf 'Release image already contains different bytes: %s (release=%s candidate=%s)\n' \
            "$release_image" "$release_digest" "$candidate_digest" >&2
        exit 1
    fi
    printf 'Release image already matches candidate: %s (%s)\n' \
        "$release_image" "$candidate_digest"
    exit 0
fi

if ! grep -Eiq 'manifest unknown|no such manifest|not found|404' "$manifest_error"; then
    cat "$manifest_error" >&2
    printf 'Could not determine whether release image exists: %s\n' "$release_image" >&2
    exit 1
fi

docker tag "$candidate_image" "$release_image"
docker push "$release_image" >/dev/null
docker pull "$release_image" >/dev/null
release_digest="$(image_digest "$release_image")"
if [[ "$release_digest" != "$candidate_digest" ]]; then
    printf 'Promoted release image digest does not match candidate: %s\n' "$release_image" >&2
    exit 1
fi
printf 'Promoted candidate %s to %s (%s)\n' \
    "$candidate_image" "$release_image" "$candidate_digest"
