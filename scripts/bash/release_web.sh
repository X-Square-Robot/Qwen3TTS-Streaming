#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
release_version="${1:-}"
if [[ -z "$release_version" ]]; then
    echo "Usage: release_web.sh <PEP440 version or v-prefixed release tag>" >&2
    exit 2
fi

node_major="$(node -p 'process.versions.node.split(".")[0]')"
if (( node_major < 22 )); then
    echo "Node.js 22 or newer is required; found $(node --version)" >&2
    exit 1
fi

release_version="${release_version#v}"
semver="$(node "$repo_root/web/scripts/version.mjs" "$release_version")"
case "$semver" in
    *-dev.*) dist_tag="dev" ;;
    *-alpha.*) dist_tag="alpha" ;;
    *-beta.*) dist_tag="beta" ;;
    *-rc.*) dist_tag="rc" ;;
    *) dist_tag="latest" ;;
esac
export DOCS_SOURCE_BASE="${DOCS_SOURCE_BASE:-https://github.com/X-Square-Robot/Qwen3TTS-Streaming/blob/v${release_version}}"
cd "$repo_root/web"
npm ci
npm version "$semver" --workspace @xmultimodalinteraction/qwen3tts-browser --no-git-tag-version
npm version "$semver" --workspace @xmultimodalinteraction/qwen3tts-demo --no-git-tag-version
npm run typecheck
npm run lint
npm test
npm run build
npm run pack:browser

mkdir -p dist
tar -C packages/demo/dist -czf "dist/qwen3tts-demo-${semver}.tar.gz" .
printf 'BROWSER_SDK_VERSION=%s\nBROWSER_SDK_SEMVER=%s\nBROWSER_SDK_DIST_TAG=%s\n' \
    "$release_version" "$semver" "$dist_tag" > dist/web-release.env
