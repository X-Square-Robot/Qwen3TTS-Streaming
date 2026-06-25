#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
    cat <<'EOF'
Usage:
  scripts/bash/run_engine_story_full_dump.sh [options] [config]

Options:
  --config PATH                Engine config path (default: engine.yaml)
  --dump-tag TAG               Prefix for auto dump dir name (default: full_dump)
  --dump-dir DIR               Explicit dump dir; overrides --dump-tag
  --session ID[,ID...]         Dump only matching session ids/base session ids
  --do-sample true|false       Sampling switch (default: false)
  --temperature FLOAT          Sampling temperature (default: 1.0)
  --repetition-penalty FLOAT   Repetition penalty (default: 1.05)
  --top-k INT                  Sampling top-k (default: 50)
  --include-wav 0|1            Include wav tensor in dumps (default: 0)
  --dump-limit INT             Max dump calls; 0 means unlimited (default: 0)
  --help                       Show this message

Examples:
  scripts/bash/run_engine_story_full_dump.sh \
    --dump-tag 4a_greedy \
    --session longtext-4a-greedy-dump

  scripts/bash/run_engine_story_full_dump.sh \
    --session longtext-story-greedy \
    --do-sample true \
    --temperature 0.9
EOF
}

CONFIG_PATH="engine.yaml"
DUMP_TAG="full_dump"
DUMP_DIR=""
SESSION_FILTER="${ENGINE_DUMP_SESSIONS:-}"
DO_SAMPLE="${ENGINE_SAMPLING_DO_SAMPLE:-false}"
TEMPERATURE="${ENGINE_SAMPLING_TEMPERATURE:-1.0}"
REPETITION_PENALTY="${ENGINE_SAMPLING_REPETITION_PENALTY:-1.05}"
TOP_K="${ENGINE_SAMPLING_TOP_K:-50}"
INCLUDE_WAV="${ENGINE_DUMP_INCLUDE_WAV:-0}"
DUMP_LIMIT="${ENGINE_DUMP_LIMIT:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dump-tag)
            DUMP_TAG="$2"
            shift 2
            ;;
        --dump-dir)
            DUMP_DIR="$2"
            shift 2
            ;;
        --session)
            SESSION_FILTER="$2"
            shift 2
            ;;
        --do-sample)
            DO_SAMPLE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --repetition-penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --include-wav)
            INCLUDE_WAV="$2"
            shift 2
            ;;
        --dump-limit)
            DUMP_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            CONFIG_PATH="$1"
            shift
            ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${DUMP_DIR}" ]]; then
    DUMP_DIR="${ROOT_DIR}/workspace/engine_dumps/${DUMP_TAG}_${TIMESTAMP}"
fi

export ENGINE_SAMPLING_DO_SAMPLE="${DO_SAMPLE}"
export ENGINE_SAMPLING_TEMPERATURE="${TEMPERATURE}"
export ENGINE_SAMPLING_REPETITION_PENALTY="${REPETITION_PENALTY}"
export ENGINE_SAMPLING_TOP_K="${TOP_K}"
export ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS="${ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS:-8}"
export ENGINE_SERVER_REQUEST_TIMEOUT_SEC="${ENGINE_SERVER_REQUEST_TIMEOUT_SEC:-1800}"
export ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC="${ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC:-1800}"
export ENGINE_SESSION_RESULT_QUEUE_MAXSIZE="${ENGINE_SESSION_RESULT_QUEUE_MAXSIZE:-4096}"
export ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE="${ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE:-4096}"
export ENGINE_DUMP_DIR="${ENGINE_DUMP_DIR:-${DUMP_DIR}}"
export ENGINE_DUMP_SESSIONS="${SESSION_FILTER}"
export ENGINE_DUMP_LIMIT="${DUMP_LIMIT}"
export ENGINE_DUMP_TEXT="${ENGINE_DUMP_TEXT:-1}"
export ENGINE_DUMP_SUMMARY="${ENGINE_DUMP_SUMMARY:-1}"
export ENGINE_DUMP_INCLUDE_WAV="${INCLUDE_WAV}"
export ENGINE_DUMP_TEXT_MAX_ELEMENTS="${ENGINE_DUMP_TEXT_MAX_ELEMENTS:-0}"
export ENGINE_DUMP_INPUT_KEYS="${ENGINE_DUMP_INPUT_KEYS:-input_embeds,position_ids,token_counts,gumbel_noise,cp_gumbel_noise,temperature,penalty,attention_bias,talker_past_kv,cache_position,c2w_attention_bias,c2w_past_kv,c2w_conv_state_*,c2w_transconv_overlap_*}"
export ENGINE_DUMP_OUTPUT_KEYS="${ENGINE_DUMP_OUTPUT_KEYS:-full_codec,codec_sum,updated_token_counts,talker_new_kv,c2w_new_kv,c2w_new_conv_state_*,c2w_new_transconv_overlap_*,wav}"
export ENGINE_DUMP_EXCLUDE_KEYS="${ENGINE_DUMP_EXCLUDE_KEYS:-hidden,logits}"

mkdir -p "${ENGINE_DUMP_DIR}"

echo "Config: ${CONFIG_PATH}"
echo "ENGINE_SAMPLING_DO_SAMPLE=${ENGINE_SAMPLING_DO_SAMPLE}"
echo "ENGINE_SAMPLING_TEMPERATURE=${ENGINE_SAMPLING_TEMPERATURE}"
echo "ENGINE_SAMPLING_REPETITION_PENALTY=${ENGINE_SAMPLING_REPETITION_PENALTY}"
echo "ENGINE_SAMPLING_TOP_K=${ENGINE_SAMPLING_TOP_K}"
echo "ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS=${ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS}"
echo "ENGINE_SERVER_REQUEST_TIMEOUT_SEC=${ENGINE_SERVER_REQUEST_TIMEOUT_SEC}"
echo "ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC=${ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC}"
echo "ENGINE_SESSION_RESULT_QUEUE_MAXSIZE=${ENGINE_SESSION_RESULT_QUEUE_MAXSIZE}"
echo "ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE=${ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE}"
echo "ENGINE_DUMP_DIR=${ENGINE_DUMP_DIR}"
echo "ENGINE_DUMP_SESSIONS=${ENGINE_DUMP_SESSIONS:-all}"
echo "ENGINE_DUMP_LIMIT=${ENGINE_DUMP_LIMIT}"
echo "ENGINE_DUMP_TEXT=${ENGINE_DUMP_TEXT}"
echo "ENGINE_DUMP_SUMMARY=${ENGINE_DUMP_SUMMARY}"
echo "ENGINE_DUMP_INCLUDE_WAV=${ENGINE_DUMP_INCLUDE_WAV}"
echo "ENGINE_DUMP_TEXT_MAX_ELEMENTS=${ENGINE_DUMP_TEXT_MAX_ELEMENTS}"
echo "ENGINE_DUMP_INPUT_KEYS=${ENGINE_DUMP_INPUT_KEYS}"
echo "ENGINE_DUMP_OUTPUT_KEYS=${ENGINE_DUMP_OUTPUT_KEYS}"
echo "ENGINE_DUMP_EXCLUDE_KEYS=${ENGINE_DUMP_EXCLUDE_KEYS}"

exec mamba run -n qwen3-tts python -m engine.server --config "${CONFIG_PATH}"
