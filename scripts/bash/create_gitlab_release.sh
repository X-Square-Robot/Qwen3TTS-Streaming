#!/bin/sh

set -eu

missing_release_env=

require_env() {
  required_name=$1
  required_value=$2
  if [ -z "$required_value" ]; then
    missing_release_env="${missing_release_env}${missing_release_env:+ }${required_name}"
  fi
}

require_release_environment() {
  require_env RELEASE_TAG "${release_tag-}"
  require_env RELEASE_COMMIT_SHA "${release_sha-}"
  require_env CI_PROJECT_ID "${CI_PROJECT_ID-}"
  require_env CI_PROJECT_PATH "${CI_PROJECT_PATH-}"
  require_env CI_PROJECT_URL "${CI_PROJECT_URL-}"
  require_env CI_SERVER_FQDN "${CI_SERVER_FQDN-}"
  require_env CI_SERVER_PROTOCOL "${CI_SERVER_PROTOCOL-}"
  require_env ENGINE_RELEASE_IMAGE "${ENGINE_RELEASE_IMAGE-}"
  require_env TRITON_RELEASE_IMAGE "${TRITON_RELEASE_IMAGE-}"
  require_env WHEEL_FILENAME "${WHEEL_FILENAME-}"
  require_env WHEEL_REGISTRY_URL "${WHEEL_REGISTRY_URL-}"
  require_env WHEEL_RELEASE_URL "${WHEEL_RELEASE_URL-}"
  require_env WHEEL_SHA256 "${WHEEL_SHA256-}"
  require_env BROWSER_SDK_TARBALL "${BROWSER_SDK_TARBALL-}"
  require_env BROWSER_SDK_GENERIC_URL "${BROWSER_SDK_GENERIC_URL-}"
  require_env DEMO_ARCHIVE "${DEMO_ARCHIVE-}"
  require_env DEMO_ARCHIVE_URL "${DEMO_ARCHIVE_URL-}"

  if [ -n "$missing_release_env" ]; then
    printf 'Missing required release environment variables: %s\n' \
      "$missing_release_env" >&2
    return 1
  fi
}

validate_url() {
  url_label=$1
  url_value=$2

  case "$url_value" in
    http://?*|https://?*) ;;
    *)
      printf '%s must be an HTTP(S) URL, got: %s\n' "$url_label" "$url_value" >&2
      return 1
      ;;
  esac
}

validate_direct_asset_path() {
  asset_path=$1

  case "$asset_path" in
    /*) ;;
    *)
      printf 'Invalid GitLab direct asset path: %s\n' "$asset_path" >&2
      return 1
      ;;
  esac
  case "$asset_path" in
    *[!A-Za-z0-9._/-]*|*//*|*/../*|*/..)
      printf 'Invalid GitLab direct asset path: %s\n' "$asset_path" >&2
      return 1
      ;;
  esac
}

validate_link_spec() {
  spec_name=$1
  spec_url=$2
  spec_direct_path=$3

  if [ -z "$spec_name" ]; then
    echo 'GitLab release link name must not be empty' >&2
    return 1
  fi
  validate_url "GitLab release link URL for $spec_name" "$spec_url"
  validate_direct_asset_path "$spec_direct_path"
}

create_or_validate_link() {
  release_link_name=$1
  release_link_url=$2
  release_link_path=$3

  release_links_json="$(glab api --hostname "$CI_SERVER_FQDN" "$links_endpoint")"
  release_link_matches="$(printf '%s' "$release_links_json" | jq \
    --arg name "$release_link_name" --arg url "$release_link_url" \
    '[.[] | select(.name == $name or .url == $url)]')"

  case "$(printf '%s' "$release_link_matches" | jq 'length')" in
    0)
      # GitLab documents release-link creation as form fields. Using --form is
      # also compatible with older self-managed GitLab versions that do not
      # decode this endpoint's JSON body consistently.
      release_link_json="$(glab api --hostname "$CI_SERVER_FQDN" \
        --method POST "$links_endpoint" \
        --form "name=$release_link_name" \
        --form "url=$release_link_url" \
        --form "direct_asset_path=$release_link_path" \
        --form 'link_type=package')"
      ;;
    1)
      release_link_json="$(printf '%s' "$release_link_matches" | jq '.[0]')"
      ;;
    *)
      printf 'Release has conflicting asset links for %s\n' "$release_link_name" >&2
      return 1
      ;;
  esac

  expected_direct_url="${CI_PROJECT_URL}/-/releases/${release_tag}/downloads${release_link_path}"
  if [ "$(printf '%s' "$release_link_json" | jq -r '.name')" != "$release_link_name" ] || \
     [ "$(printf '%s' "$release_link_json" | jq -r '.url')" != "$release_link_url" ] || \
     [ "$(printf '%s' "$release_link_json" | jq -r '.direct_asset_url')" != "$expected_direct_url" ] || \
     [ "$(printf '%s' "$release_link_json" | jq -r '.link_type')" != package ]; then
    printf 'Existing GitLab Release asset does not match %s\n' "$release_link_name" >&2
    return 1
  fi
}

main() {
  release_tag="${RELEASE_VERSION:-${CI_COMMIT_TAG-}}"
  release_sha="${RELEASE_COMMIT_SHA:-${CI_COMMIT_SHA-}}"
  release_candidate="${CANDIDATE_ID-}"
  export release_tag release_sha
  require_release_environment

  wheel_path="/client-sdk/$WHEEL_FILENAME"
  browser_path="/browser-sdk/$BROWSER_SDK_TARBALL"
  demo_path="/demo/$DEMO_ARCHIVE"
  expected_wheel_release_url="${CI_PROJECT_URL}/-/releases/${release_tag}/downloads${wheel_path}"
  validate_url WHEEL_REGISTRY_URL "$WHEEL_REGISTRY_URL"
  validate_url WHEEL_RELEASE_URL "$WHEEL_RELEASE_URL"
  if [ "$WHEEL_RELEASE_URL" != "$expected_wheel_release_url" ]; then
    printf 'WHEEL_RELEASE_URL does not match the direct asset contract: %s\n' \
      "$WHEEL_RELEASE_URL" >&2
    return 1
  fi
  validate_link_spec "$WHEEL_FILENAME" "$WHEEL_REGISTRY_URL" "$wheel_path"
  validate_link_spec "$BROWSER_SDK_TARBALL" "$BROWSER_SDK_GENERIC_URL" "$browser_path"
  validate_link_spec "$DEMO_ARCHIVE" "$DEMO_ARCHIVE_URL" "$demo_path"

  case "$CI_SERVER_PROTOCOL" in
    http|https) ;;
    *)
      printf 'Unsupported GitLab API protocol: %s\n' "$CI_SERVER_PROTOCOL" >&2
      return 1
      ;;
  esac

  GLAB_CHECK_UPDATE=false
  GLAB_SEND_TELEMETRY=false
  GLAB_SHOW_WHATS_NEW=false
  NO_PROMPT=true
  export GLAB_CHECK_UPDATE GLAB_SEND_TELEMETRY GLAB_SHOW_WHATS_NEW NO_PROMPT

  glab config set api_protocol "$CI_SERVER_PROTOCOL" --host "$CI_SERVER_FQDN"
  glab api --hostname "$CI_SERVER_FQDN" job --silent

  # Candidate pipelines are not tag pipelines. Let the Releases API create the
  # formal tag only after every build, image promotion, and package promotion
  # has succeeded. Existing releases are accepted only as an idempotent retry.
  if [ -n "${RELEASE_VERSION-}" ]; then
    tag_endpoint="projects/${CI_PROJECT_ID}/repository/tags/${release_tag}"
    tag_json="$(glab api --hostname "$CI_SERVER_FQDN" "$tag_endpoint" 2>/dev/null || true)"
    if [ -n "$tag_json" ]; then
      actual_tag_sha="$(printf '%s' "$tag_json" | jq -r '.commit.id // empty')"
      if [ "$actual_tag_sha" != "$release_sha" ]; then
        printf 'GitLab tag %s already points to %s, expected %s\n' \
          "$release_tag" "$actual_tag_sha" "$release_sha" >&2
        return 1
      fi
    fi
  fi

  release_notes="$(printf 'Source commit: `%s`\n\nCandidate: `%s`\n\nSDK wheel: `%s`\n\nSDK SHA256: `%s`\n\nStandalone image: `%s`\n\nTriton image: `%s`\n' \
    "$release_sha" "$release_candidate" "$WHEEL_FILENAME" "$WHEEL_SHA256" \
    "$ENGINE_RELEASE_IMAGE" "$TRITON_RELEASE_IMAGE")"
  release_endpoint="projects/${CI_PROJECT_ID}/releases/${release_tag}"
  release_json="$(glab api --hostname "$CI_SERVER_FQDN" "$release_endpoint" 2>/dev/null || true)"
  release_json_type="$(printf '%s' "$release_json" | jq -r 'type' 2>/dev/null || true)"
  existing_release_tag="$(printf '%s' "$release_json" | jq -r '.tag_name // empty' 2>/dev/null || true)"
  if [ "$release_json_type" != object ] || [ -z "$existing_release_tag" ]; then
    glab release create "$release_tag" \
      --repo "$CI_PROJECT_PATH" \
      --ref "$release_sha" \
      --name "Qwen3TTS-Streaming $release_tag" \
      --notes "$release_notes"
  else
    if [ "$existing_release_tag" != "$release_tag" ]; then
      printf 'GitLab Release response has unexpected tag: %s\n' \
        "$existing_release_tag" >&2
      return 1
    fi
    printf 'GitLab Release already exists: %s\n' "$release_tag"
  fi

  links_endpoint="projects/${CI_PROJECT_ID}/releases/${release_tag}/assets/links"
  readonly links_endpoint
  create_or_validate_link "$WHEEL_FILENAME" "$WHEEL_REGISTRY_URL" "$wheel_path"
  create_or_validate_link "$BROWSER_SDK_TARBALL" "$BROWSER_SDK_GENERIC_URL" "$browser_path"
  create_or_validate_link "$DEMO_ARCHIVE" "$DEMO_ARCHIVE_URL" "$demo_path"
}

main "$@"
