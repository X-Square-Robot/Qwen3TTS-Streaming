#!/usr/bin/env bash
# Build, verify, and publish the slow-changing Triton Python dependency base.
# Release candidate pipelines consume this image and never install dependencies.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: publish_triton_runtime_base.sh <runtime-base-image> [cache-image]

Environment:
  TRITON_BASE_IMAGE                 NVIDIA Triton py3 base image
  TRITON_TENSORRT_PYTHON_VERSION    TensorRT Python package version
  TRITON_PYTORCH_CUDA_TAG           PyTorch CUDA wheel channel
  TRITON_RUNTIME_PARENT_IMAGE       Optional preceding immutable runtime base

The caller must log in to the destination Registry before running this script.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

runtime_base_image="$1"
cache_image="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

triton_base_image="${TRITON_BASE_IMAGE:-nvcr.io/nvidia/tritonserver:26.02-py3}"
tensorrt_version="${TRITON_TENSORRT_PYTHON_VERSION:-10.15.1.29}"
pytorch_cuda_tag="${TRITON_PYTORCH_CUDA_TAG:-cu130}"
runtime_parent_image="${TRITON_RUNTIME_PARENT_IMAGE:-}"

if [[ "${TRITON_RUNTIME_BASE_ALLOW_OVERWRITE:-0}" != "1" ]] \
    && docker manifest inspect "$runtime_base_image" >/dev/null 2>&1; then
    echo "Triton runtime base already exists; keeping immutable image: $runtime_base_image"
    exit 0
fi

if [[ -n "$cache_image" ]]; then
    docker pull "$cache_image" || {
        echo "Warning: no previous Triton build cache is available at $cache_image" >&2
    }
fi

if [[ -n "$runtime_parent_image" ]]; then
    docker pull "$runtime_parent_image"
fi

build_args=(
    --pull
    --file "$repo_root/infra/docker/Dockerfile.triton"
    --target triton-deps
    --build-arg "BUILDKIT_INLINE_CACHE=1"
    --build-arg "BASE_IMAGE=$triton_base_image"
    --build-arg "TENSORRT_PYTHON_VERSION=$tensorrt_version"
    --build-arg "PYTORCH_CUDA_TAG=$pytorch_cuda_tag"
    --tag "$runtime_base_image"
)
if [[ -n "$runtime_parent_image" ]]; then
    build_args+=(--build-arg "TRITON_RUNTIME_PARENT_IMAGE=$runtime_parent_image")
fi
if [[ -n "$cache_image" ]]; then
    build_args+=(--cache-from "$cache_image")
fi

docker build "${build_args[@]}" "$repo_root"
docker run --rm --entrypoint python3 "$runtime_base_image" -c \
    "import aiohttp, attr, packaging, torch, tokenizers, tensorrt, yaml, grpc, numpy, soxr, wetext, pypinyin; print(f'torch={torch.__version__} cuda={torch.version.cuda} tensorrt={tensorrt.__version__} attrs={attr.__version__} packaging={packaging.__version__} wetext=ready pypinyin={pypinyin.__version__}')"
docker push "$runtime_base_image"

echo "Published Triton runtime base: $runtime_base_image"
