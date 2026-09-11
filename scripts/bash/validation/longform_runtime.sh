#!/usr/bin/env bash
# Isolated launch helpers for the 0818 long-form hallucination comparison.
# This script intentionally never calls build_triton.sh/assemble/autorun.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
package_root="$repo_root/workspace/model_repository"
package_version_dir="$package_root/tts_orchestrator/2"
plan_path="$package_version_dir/runtime/model.plan"
expected_model_version="${QWEN3TTS_LONGFORM_MODEL_VERSION:-}"
expected_plan_sha256="18f0e6d324bae9bdc446cae580f63cf83c981909408927fc2d01d8b0be26a181"
expected_text_token_ids_sha256="fcf7b0c06027375b7e02eb3e58aa7f96d8654320e0d0e80978ed1ff862e1ca87"
eval_image="qwen3tts-longform-eval:triton25.10-pytorch25.10-v0.1.2a6"
expected_eval_image_id="sha256:4b6d77171820de0319320703e4da836d870fb93a79d9a2fa8fcb8231fc4c35da"
triton_source_image="nvcr.io/nvidia/tritonserver:25.10-py3"
triton_source_id="sha256:9ff4dd7a1c52487b35e98208ce9bbdb6580b036cdc8943a64587e1c2685de662"
pytorch_source_image="cr.x2robot.cn/audio/qwen3tt-streaming:trt25.10_580_cu13_v0.1.2a6"
pytorch_source_id="sha256:4c314475904ebc446addc3450a37348fb0bbf02efdc0f93908f079bf50b1a084"

usage() {
  echo "usage: $0 {build-triton-image|image-versions|smoke-triton-empty|current|triton|official|current-matched-sample|current-matched-greedy|triton-matched-sample|triton-matched-greedy|official-matched-sample|official-matched-greedy|verify-triton-mount} OUTPUT_DIR" >&2
}

verify_frozen_package() {
  local observed
  test -f "$package_version_dir/MODEL_VERSION"
  test -f "$plan_path"
  observed="$(sha256sum "$plan_path" | cut -d' ' -f1)"
  if [[ "$observed" != "$expected_plan_sha256" ]]; then
    echo "frozen TRT plan hash mismatch: $observed" >&2
    exit 1
  fi
  if [[ -n "$expected_model_version" ]]; then
    if [[ "$(tr -d '\r\n' < "$package_version_dir/MODEL_VERSION")" != "$expected_model_version" ]]; then
      echo "frozen MODEL_VERSION does not match QWEN3TTS_LONGFORM_MODEL_VERSION" >&2
      exit 1
    fi
  fi
}

require_exclusive_gpu() {
  local runtime_name="${1:-$command_name}"
  local processes
  if ! processes="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>&1)"; then
    echo "cannot verify GPU exclusivity because nvidia-smi failed:" >&2
    echo "$processes" >&2
    exit 1
  fi
  if [[ -z "${processes//[[:space:]]/}" ]]; then
    return
  fi

  local allowed_pids="${QWEN3TTS_LONGFORM_ALLOWED_GPU_PIDS:-}"
  local waiver_reason="${QWEN3TTS_LONGFORM_GPU_WAIVER_REASON:-}"
  allowed_pids="${allowed_pids//[[:space:]]/}"
  if [[ -z "$allowed_pids" ]]; then
    echo "GPU 0 is not exclusive; stop/coordinate these processes before running an arm:" >&2
    echo "$processes" >&2
    exit 1
  fi
  if [[ ! "$allowed_pids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "QWEN3TTS_LONGFORM_ALLOWED_GPU_PIDS must be a comma-separated PID list" >&2
    exit 1
  fi
  if [[ -z "${waiver_reason//[[:space:]]/}" ]]; then
    echo "QWEN3TTS_LONGFORM_GPU_WAIVER_REASON is required with a GPU PID allowlist" >&2
    exit 1
  fi

  local unexpected=""
  local pid process_name used_memory
  while IFS=',' read -r pid process_name used_memory; do
    pid="${pid//[[:space:]]/}"
    if [[ ",$allowed_pids," != *",$pid,"* ]]; then
      unexpected+="${pid},${process_name},${used_memory}"$'\n'
    fi
  done <<< "$processes"
  if [[ -n "$unexpected" ]]; then
    echo "GPU 0 has processes outside the explicit allowlist:" >&2
    printf '%s' "$unexpected" >&2
    exit 1
  fi

  local evidence_timestamp waiver_path
  evidence_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  waiver_path="$output_dir/runtime/${runtime_name}_gpu_process_waiver_${evidence_timestamp}_$$.txt"
  umask 022
  {
    printf 'schema_version=1\n'
    printf 'runtime=%s\n' "$runtime_name"
    printf 'allowed_pids=%s\n' "$allowed_pids"
    printf 'reason=%s\n' "$waiver_reason"
    printf 'observed_processes_csv=pid,process_name,used_memory_mib\n'
    printf '%s\n' "$processes"
  } > "$waiver_path"
  sha256sum "$waiver_path" > "$waiver_path.sha256"
  echo "GPU process waiver accepted and recorded: $waiver_path" >&2
}

verify_source_image() {
  local reference="$1"
  local expected_id="$2"
  local observed_id
  observed_id="$(docker image inspect --format '{{.Id}}' "$reference")"
  if [[ "$observed_id" != "$expected_id" ]]; then
    echo "source image identity mismatch for $reference: $observed_id" >&2
    exit 1
  fi
}

record_eval_image() {
  local inspect_path="$output_dir/runtime/triton_image_inspect.json"
  local versions_path="$output_dir/runtime/triton_runtime_versions.json"
  local token_identity_path="$output_dir/runtime/triton_text_tokenization.json"
  verify_source_image "$eval_image" "$expected_eval_image_id"
  docker image inspect "$eval_image" > "$inspect_path"
  sha256sum "$inspect_path" > "$inspect_path.sha256"
  docker run --rm --entrypoint python3 "$eval_image" -c \
    'import grpc,json,numpy,soxr,tensorrt,tokenizers,torch,yaml; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"tokenizers":tokenizers.__version__,"tensorrt":tensorrt.__version__,"grpcio":grpc.__version__,"numpy":numpy.__version__,"soxr":soxr.__version__,"pyyaml":yaml.__version__,"triton_server":open("/opt/tritonserver/TRITON_VERSION").read().strip()},sort_keys=True))' \
    > "$versions_path"
  docker run --rm \
    --entrypoint python3 \
    --mount "type=bind,src=$repo_root,dst=/repo,readonly" \
    --workdir /repo \
    -e PYTHONPATH=/repo/workspace/model_repository/tts_orchestrator/2 \
    "$eval_image" -c \
    'from pathlib import Path; from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer; import hashlib,json,tokenizers; text=Path("resources/dataset/badcase/verylong.txt").read_bytes().decode("utf-8"); ids=LightQwen3TTSTokenizer("workspace/model_repository/tts_orchestrator/2/tokenizer").encode_ids(text,add_special_tokens=False); print(json.dumps({"tokenizer_class":"LightQwen3TTSTokenizer","tokenizers_version":tokenizers.__version__,"add_special_tokens":False,"token_count":len(ids),"ids_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()},sort_keys=True))' \
    > "$token_identity_path"
  /home/rime/miniforge3/envs/qwen3-tts/bin/python -c \
    'import json,sys; payload=json.load(open(sys.argv[1],encoding="utf-8")); expected=sys.argv[2]; assert payload["token_count"] == 677 and payload["ids_sha256"] == expected, payload' \
    "$token_identity_path" "$expected_text_token_ids_sha256"
  sha256sum "$versions_path" "$token_identity_path" \
    > "$output_dir/runtime/triton_runtime_evidence.sha256"
}

record_gpu_snapshot() {
  local runtime_name="$1"
  local gpu_path="$output_dir/runtime/${runtime_name}_gpu_before_start.csv"
  nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$gpu_path"
}

record_host_runtime() {
  local runtime_name="$1"
  local versions_path="$output_dir/runtime/${runtime_name}_versions.json"
  local gpu_path="$output_dir/runtime/${runtime_name}_gpu_before_start.csv"
  /home/rime/miniforge3/envs/qwen3-tts/bin/python -c \
    'import json,platform,sys; import grpc,numpy,soxr,tensorrt,tokenizers,torch,yaml; print(json.dumps({"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"tokenizers":tokenizers.__version__,"tensorrt":tensorrt.__version__,"grpcio":grpc.__version__,"numpy":numpy.__version__,"soxr":soxr.__version__,"pyyaml":yaml.__version__},sort_keys=True))' \
    > "$versions_path"
  record_gpu_snapshot "$runtime_name"
  sha256sum "$versions_path" "$gpu_path" \
    > "$output_dir/runtime/${runtime_name}_evidence.sha256"
}

command_name="${1:-}"
output_dir="${2:-$repo_root/workspace/validation/longform_0818}"
if [[ -z "$command_name" ]]; then
  usage
  exit 2
fi
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
mkdir -p "$output_dir/runtime"

case "$command_name" in
  build-triton-image)
    verify_frozen_package
    verify_source_image "$triton_source_image" "$triton_source_id"
    verify_source_image "$pytorch_source_image" "$pytorch_source_id"
    DOCKER_BUILDKIT=1 docker build --pull=false --provenance=false --progress=plain \
      --build-arg TRITON_SOURCE_IMAGE="$triton_source_image" \
      --build-arg PYTORCH_SOURCE_IMAGE="$pytorch_source_image" \
      -t "$eval_image" \
      -f "$repo_root/infra/docker/Dockerfile.longform-eval" "$repo_root"
    verify_source_image "$eval_image" "$expected_eval_image_id"
    record_eval_image
    ;;

  image-versions)
    docker run --rm --entrypoint python3 "$eval_image" -c \
      'import grpc, numpy, soxr, tensorrt, tokenizers, torch, yaml; print({"torch": torch.__version__, "cuda": torch.version.cuda, "tokenizers": tokenizers.__version__, "tensorrt": tensorrt.__version__, "grpcio": grpc.__version__, "numpy": numpy.__version__, "soxr": soxr.__version__, "pyyaml": yaml.__version__})'
    docker run --rm --entrypoint cat "$eval_image" /opt/tritonserver/TRITON_VERSION
    ;;

  smoke-triton-empty)
    container_name="qwen3tts-0818-triton-empty-smoke"
    if docker container inspect "$container_name" >/dev/null 2>&1; then
      echo "refusing to replace existing container: $container_name" >&2
      exit 1
    fi
    docker run --rm --detach \
      --name "$container_name" \
      --tmpfs /models:rw,noexec,nosuid,size=16m \
      -p 127.0.0.1:59100:8000 \
      -p 127.0.0.1:59101:8001 \
      -p 127.0.0.1:59102:8002 \
      "$eval_image" \
      tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --disable-auto-complete-config \
        --allow-gpu-metrics=false \
        --strict-readiness=true \
        --log-verbose=0 >/dev/null
    cleanup_empty_smoke() {
      docker logs "$container_name" > "$output_dir/runtime/triton_empty_smoke.log" 2>&1 || true
      docker stop --time 10 "$container_name" >/dev/null 2>&1 || true
    }
    trap cleanup_empty_smoke EXIT
    for _ in $(seq 1 30); do
      if curl -fsS http://127.0.0.1:59100/v2/health/ready >/dev/null; then
        curl -fsS http://127.0.0.1:59100/v2 > "$output_dir/runtime/triton_empty_metadata.json"
        exit 0
      fi
      sleep 1
    done
    echo "empty-repository Triton smoke did not become ready" >&2
    exit 1
    ;;

  current)
    verify_frozen_package
    require_exclusive_gpu "current_default"
    record_host_runtime "current_default"
    config_path="$output_dir/runtime/current_engine.yaml"
    cp "$repo_root/engine.yaml" "$config_path"
    sed -i -E \
      -e "s|^  model_package_dir:.*$|  model_package_dir: $package_version_dir|" \
      -e 's|^  port: [0-9]+.*$|  port: 55151|' \
      -e 's|^  websocket_port: [0-9]+.*$|  websocket_port: 55152|' \
      -e 's|^  health_port: [0-9]+.*$|  health_port: 58080|' \
      "$config_path"
    sha256sum "$config_path" > "$output_dir/runtime/current_engine.yaml.sha256"
    cd "$repo_root"
    exec env -i \
      PATH=/home/rime/miniforge3/envs/qwen3-tts/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHONPATH="$repo_root:$repo_root/client/src" \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
      /home/rime/miniforge3/envs/qwen3-tts/bin/python -m engine.server \
        --config "$config_path" \
        --model-package-dir "$package_version_dir" \
        --device 0 --max-batch 1 --max-sessions 1 --max-seq-len 512 \
        --port 55151 --ws-port 55152
    ;;

  official)
    verify_frozen_package
    require_exclusive_gpu "official_default"
    record_host_runtime "official_default"
    cd "$repo_root"
    exec env -i \
      PATH=/home/rime/miniforge3/envs/qwen3-tts/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHONPATH="$repo_root:$repo_root/client/src:$repo_root/third_party/Qwen3-TTS" \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
      /home/rime/miniforge3/envs/qwen3-tts/bin/python \
        -m tools.validation.hallucination.longform collect-official \
        --output-dir "$output_dir"
    ;;

  official-matched-sample|official-matched-greedy)
    verify_frozen_package
    require_exclusive_gpu "official_matched_${command_name#official-matched-}"
    matched_mode="${command_name#official-matched-}"
    record_host_runtime "official_matched_${matched_mode}"
    cd "$repo_root"
    exec env -i \
      PATH=/home/rime/miniforge3/envs/qwen3-tts/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHONPATH="$repo_root:$repo_root/client/src:$repo_root/third_party/Qwen3-TTS" \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
      /home/rime/miniforge3/envs/qwen3-tts/bin/python \
        -m tools.validation.hallucination.longform collect-matched-official \
        --output-dir "$output_dir" --mode "$matched_mode"
    ;;

  current-matched-sample|current-matched-greedy)
    verify_frozen_package
    require_exclusive_gpu "current_matched_${command_name#current-matched-}"
    matched_mode="${command_name#current-matched-}"
    record_host_runtime "current_matched_${matched_mode}"
    config_path="$output_dir/runtime/current_matched_${matched_mode}.yaml"
    if [[ -e "$config_path" ]]; then
      echo "refusing to overwrite matched runtime config: $config_path" >&2
      exit 1
    fi
    cp "$repo_root/engine.yaml" "$config_path"
    sed -i -E \
      -e "s|^  model_package_dir:.*$|  model_package_dir: $package_version_dir|" \
      -e 's|^  port: [0-9]+.*$|  port: 55151|' \
      -e 's|^  websocket_port: [0-9]+.*$|  websocket_port: 55152|' \
      -e 's|^  health_port: [0-9]+.*$|  health_port: 58080|' \
      -e 's|^  guarded_delivery_default:.*$|  guarded_delivery_default: false|' \
      "$config_path"
    if [[ "$matched_mode" == "sample" ]]; then
      sed -i -E 's|^  do_sample:.*$|  do_sample: true|' "$config_path"
    else
      sed -i -E 's|^  do_sample:.*$|  do_sample: false|' "$config_path"
    fi
    sha256sum "$config_path" > "$config_path.sha256"
    cd "$repo_root"
    exec env -i \
      PATH=/home/rime/miniforge3/envs/qwen3-tts/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHONPATH="$repo_root:$repo_root/client/src" \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
      /home/rime/miniforge3/envs/qwen3-tts/bin/python -m engine.server \
        --config "$config_path" \
        --model-package-dir "$package_version_dir" \
        --device 0 --max-batch 1 --max-sessions 1 --max-seq-len 512 \
        --port 55151 --ws-port 55152
    ;;

  triton)
    verify_frozen_package
    require_exclusive_gpu "triton_default"
    record_gpu_snapshot "triton_default"
    record_eval_image
    staging_root="$output_dir/runtime/triton_default_repo"
    staging_model="$staging_root/tts_orchestrator"
    mkdir -p "$staging_model"
    if [[ -e "$staging_model/config.pbtxt" ]]; then
      if ! cmp -s "$package_root/tts_orchestrator/config.pbtxt" "$staging_model/config.pbtxt"; then
        echo "existing default Triton staging config differs from frozen config" >&2
        exit 1
      fi
    else
      cp "$package_root/tts_orchestrator/config.pbtxt" "$staging_model/config.pbtxt"
    fi
    if [[ -L "$staging_model/2" ]]; then
      if [[ "$(readlink "$staging_model/2")" != "/frozen/tts_orchestrator/2" ]]; then
        echo "existing default Triton staging symlink has unexpected target" >&2
        exit 1
      fi
    elif [[ -e "$staging_model/2" ]]; then
      echo "refusing non-symlink default Triton staging version" >&2
      exit 1
    else
      ln -s /frozen/tts_orchestrator/2 "$staging_model/2"
    fi
    sha256sum "$staging_model/config.pbtxt" \
      > "$staging_model/config.pbtxt.sha256"
    exec docker run --rm --init \
      --name qwen3tts-0818-triton-eval \
      --label qwen3tts.eval.model-version=2 \
      --gpus 'device=0' \
      --shm-size=4g \
      --ulimit memlock=-1 \
      --mount "type=bind,src=$staging_root,dst=/models,readonly" \
      --mount "type=bind,src=$package_root,dst=/frozen,readonly" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e MAX_BATCH_SLOTS=1 \
      -e MAX_SESSIONS=1 \
      -e ENGINE_MAX_DECODE_LEN=512 \
      -e TRITON_MODEL_VERSION=2 \
      -p 127.0.0.1:58100:8000 \
      -p 127.0.0.1:58101:8001 \
      -p 127.0.0.1:58102:8002 \
      "$eval_image" \
      tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --disable-auto-complete-config \
        --log-verbose=0
    ;;

  triton-matched-sample|triton-matched-greedy)
    verify_frozen_package
    require_exclusive_gpu "triton_matched_${command_name#triton-matched-}"
    matched_mode="${command_name#triton-matched-}"
    record_gpu_snapshot "triton_matched_${matched_mode}"
    staging_root="$output_dir/runtime/triton_matched_${matched_mode}_repo"
    staging_model="$staging_root/tts_orchestrator"
    if [[ -e "$staging_model/config.pbtxt" ]]; then
      echo "refusing to overwrite matched Triton config: $staging_model/config.pbtxt" >&2
      exit 1
    fi
    mkdir -p "$staging_model"
    cp "$package_root/tts_orchestrator/config.pbtxt" "$staging_model/config.pbtxt"
    ln -s /frozen/tts_orchestrator/2 "$staging_model/2"
    sed -i -E \
      -e 's|string_value: "128"|string_value: "1"|' \
      "$staging_model/config.pbtxt"
    if [[ "$matched_mode" == "sample" ]]; then
      sed -i -E \
        '/key: "do_sample"/{n;s|string_value: "false"|string_value: "true"|;}' \
        "$staging_model/config.pbtxt"
    fi
    sha256sum "$staging_model/config.pbtxt" > "$staging_model/config.pbtxt.sha256"
    record_eval_image
    exec docker run --rm --init \
      --name "qwen3tts-0818-triton-matched-${matched_mode}" \
      --label qwen3tts.eval.model-version=2 \
      --gpus 'device=0' \
      --shm-size=4g \
      --ulimit memlock=-1 \
      --mount "type=bind,src=$staging_root,dst=/models,readonly" \
      --mount "type=bind,src=$package_root,dst=/frozen,readonly" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e MAX_BATCH_SLOTS=1 \
      -e MAX_SESSIONS=1 \
      -e ENGINE_MAX_DECODE_LEN=512 \
      -e ENGINE_SAMPLING_TEMPERATURE=0.9 \
      -e ENGINE_SAMPLING_REPETITION_PENALTY=1.05 \
      -e RATIO_INITIAL=4.5 \
      -e TRITON_MODEL_VERSION=2 \
      -p 127.0.0.1:58100:8000 \
      -p 127.0.0.1:58101:8001 \
      -p 127.0.0.1:58102:8002 \
      "$eval_image" \
      tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --disable-auto-complete-config \
        --log-verbose=0
    ;;

  verify-triton-mount)
    curl -fsS http://127.0.0.1:58100/v2/health/ready
    curl -fsS http://127.0.0.1:58100/v2/models/tts_orchestrator/versions/2/ready
    models_mount_state="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.RW}}{{end}}{{end}}' qwen3tts-0818-triton-eval)"
    frozen_mount_state="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/frozen"}}{{.RW}}{{end}}{{end}}' qwen3tts-0818-triton-eval)"
    if [[ "$models_mount_state" != "false" || "$frozen_mount_state" != "false" ]]; then
      echo "Triton staging and frozen package mounts must both be read-only" >&2
      exit 1
    fi
    repository_index="$output_dir/runtime/triton_repository_index.json"
    curl -fsS -X POST http://127.0.0.1:58100/v2/repository/index \
      > "$repository_index"
    /home/rime/miniforge3/envs/qwen3-tts/bin/python -c \
      'import json,sys; rows=json.load(open(sys.argv[1],encoding="utf-8")); assert {(row["name"],row["version"]) for row in rows} == {("tts_orchestrator","2")}, rows' \
      "$repository_index"
    echo "/models RW=false; /frozen RW=false; isolated model=tts_orchestrator:2"
    ;;

  *)
    usage
    exit 2
    ;;
esac
