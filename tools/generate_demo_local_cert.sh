#!/usr/bin/env bash
# Generate an explicitly development-only certificate for direct Demo HTTPS.
# Extra SANs let a remote test browser use the same endpoint after accepting
# the self-signed certificate once:
#   SAN_EXTRA_DNS=demo.example.test ./tools/generate_demo_local_cert.sh

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
tls_dir="${DEMO_TLS_DIR:-${repo_root}/workspace/tls}"
mkdir -p "$tls_dir"

san="DNS:localhost,IP:127.0.0.1"
if [[ -n "${SAN_EXTRA_DNS:-}" ]]; then
    IFS=',' read -ra names <<< "$SAN_EXTRA_DNS"
    for name in "${names[@]}"; do
        name="${name//[[:space:]]/}"
        [[ -n "$name" ]] && san="${san},DNS:${name}"
    done
fi
if [[ -n "${SAN_EXTRA_IPS:-}" ]]; then
    IFS=',' read -ra addresses <<< "$SAN_EXTRA_IPS"
    for address in "${addresses[@]}"; do
        address="${address//[[:space:]]/}"
        [[ -n "$address" ]] && san="${san},IP:${address}"
    done
fi

openssl req -x509 -newkey rsa:4096 -nodes -days 825 \
    -keyout "${tls_dir}/key.local.pem" \
    -out "${tls_dir}/cert.local.pem" \
    -subj "/O=Qwen3 TTS Development/CN=localhost" \
    -addext "subjectAltName=${san}"
chmod 600 "${tls_dir}/key.local.pem"
chmod 644 "${tls_dir}/cert.local.pem"

echo "Development certificate: ${tls_dir}/cert.local.pem"
echo "Development private key: ${tls_dir}/key.local.pem"
echo "SAN: ${san}"
echo "This certificate is self-signed; do not use it as a public release certificate."
