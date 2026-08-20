#!/usr/bin/env bash

set -euo pipefail

readonly REQUIRED_RELEASE_ENV=(
  CI_COMMIT_SHA
  CI_COMMIT_TAG
  CI_PROJECT_ID
  CI_PROJECT_PATH
  CI_PROJECT_URL
  CI_SERVER_FQDN
  CI_SERVER_PROTOCOL
  ENGINE_RELEASE_IMAGE
  TRITON_RELEASE_IMAGE
  WHEEL_FILENAME
  WHEEL_REGISTRY_URL
  WHEEL_RELEASE_URL
  WHEEL_SHA256
  BROWSER_SDK_TARBALL
  BROWSER_SDK_GENERIC_URL
  DEMO_ARCHIVE
  DEMO_ARCHIVE_URL
)

require_release_environment() {
  local name
  local -a missing=()

  for name in "${REQUIRED_RELEASE_ENV[@]}"; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("$name")
    fi
  done

  if ((${#missing[@]})); then
    printf 'Missing required release environment variables: %s\n' "${missing[*]}" >&2
    return 1
  fi
}

validate_url() {
  local label=$1
  local value=$2

  case "$value" in
    http://*|https://*) ;;
    *)
      printf '%s must be an HTTP(S) URL, got: %s\n' "$label" "$value" >&2
      return 1
      ;;
  esac
}

validate_direct_asset_path() {
  local value=$1

  if [[ ! "$value" =~ ^/[A-Za-z0-9._/-]+$ ]] || \
     [[ "$value" == */../* ]] || [[ "$value" == */.. ]] || \
     [[ "$value" == *//* ]]; then
    printf 'Invalid GitLab direct asset path: %s\n' "$value" >&2
    return 1
  fi
}

validate_link_spec() {
  local name=$1
  local url=$2
  local direct_path=$3

  if [[ -z "$name" ]]; then
    echo 'GitLab release link name must not be empty' >&2
    return 1
  fi
  validate_url "GitLab release link URL for $name" "$url"
  validate_direct_asset_path "$direct_path"
}

create_or_validate_link() {
  local name=$1
  local url=$2
  local direct_path=$3
  local expected_direct_url
  local links_json
  local matches
  local link

  links_json="$(glab api --hostname "$CI_SERVER_FQDN" "$links_endpoint")"
  matches="$(printf '%s' "$links_json" | jq \
    --arg name "$name" --arg url "$url" \
    '[.[] | select(.name == $name or .url == $url)]')"

  case "$(printf '%s' "$matches" | jq 'length')" in
    0)
      # GitLab documents release-link creation as form fields. Using --form is
      # also compatible with older self-managed GitLab versions that do not
      # decode this endpoint's JSON body consistently.
      link="$(glab api --hostname "$CI_SERVER_FQDN" \
        --method POST "$links_endpoint" \
        --form "name=$name" \
        --form "url=$url" \
        --form "direct_asset_path=$direct_path" \
        --form 'link_type=package')"
      ;;
    1)
      link="$(printf '%s' "$matches" | jq '.[0]')"
      ;;
    *)
      printf 'Release has conflicting asset links for %s\n' "$name" >&2
      return 1
      ;;
  esac

  expected_direct_url="${CI_PROJECT_URL}/-/releases/${CI_COMMIT_TAG}/downloads${direct_path}"
  if [[ "$(printf '%s' "$link" | jq -r '.name')" != "$name" ]] || \
     [[ "$(printf '%s' "$link" | jq -r '.url')" != "$url" ]] || \
     [[ "$(printf '%s' "$link" | jq -r '.direct_asset_url')" != "$expected_direct_url" ]] || \
     [[ "$(printf '%s' "$link" | jq -r '.link_type')" != 'package' ]]; then
    printf 'Existing GitLab Release asset does not match %s\n' "$name" >&2
    return 1
  fi
}

main() {
  local notes
  local wheel_path
  local browser_path
  local demo_path
  local expected_wheel_release_url

  require_release_environment
  wheel_path="/client-sdk/$WHEEL_FILENAME"
  browser_path="/browser-sdk/$BROWSER_SDK_TARBALL"
  demo_path="/demo/$DEMO_ARCHIVE"
  expected_wheel_release_url="${CI_PROJECT_URL}/-/releases/${CI_COMMIT_TAG}/downloads${wheel_path}"
  validate_url WHEEL_REGISTRY_URL "$WHEEL_REGISTRY_URL"
  validate_url WHEEL_RELEASE_URL "$WHEEL_RELEASE_URL"
  if [[ "$WHEEL_RELEASE_URL" != "$expected_wheel_release_url" ]]; then
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

  export GLAB_CHECK_UPDATE=false
  export GLAB_SEND_TELEMETRY=false
  export GLAB_SHOW_WHATS_NEW=false
  export NO_PROMPT=true

  glab config set api_protocol "$CI_SERVER_PROTOCOL" --host "$CI_SERVER_FQDN"
  glab api --hostname "$CI_SERVER_FQDN" job --silent

  notes="$(printf 'SDK wheel: `%s`\n\nSDK SHA256: `%s`\n\nStandalone image: `%s`\n\nTriton image: `%s`\n' \
    "$WHEEL_FILENAME" "$WHEEL_SHA256" "$ENGINE_RELEASE_IMAGE" "$TRITON_RELEASE_IMAGE")"
  glab release create "$CI_COMMIT_TAG" \
    --repo "$CI_PROJECT_PATH" \
    --ref "$CI_COMMIT_SHA" \
    --name "Qwen3TTS-Streaming $CI_COMMIT_TAG" \
    --notes "$notes"

  readonly links_endpoint="projects/${CI_PROJECT_ID}/releases/${CI_COMMIT_TAG}/assets/links"
  create_or_validate_link "$WHEEL_FILENAME" "$WHEEL_REGISTRY_URL" "$wheel_path"
  create_or_validate_link "$BROWSER_SDK_TARBALL" "$BROWSER_SDK_GENERIC_URL" "$browser_path"
  create_or_validate_link "$DEMO_ARCHIVE" "$DEMO_ARCHIVE_URL" "$demo_path"
}

main "$@"
