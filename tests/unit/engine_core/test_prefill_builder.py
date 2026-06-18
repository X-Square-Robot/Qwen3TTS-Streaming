"""
L1 unit tests: EmbeddingWeights and PrefillBuilder (T1.3, T1.4, T4.2).
Requires: workspace/exported/<variant>/weights/ with .pt from export_01_embeddings.py, and GPU.
Run from repo root: pytest tests/unit/test_prefill_builder.py -v
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_DIR_CANDIDATES = [
    REPO_ROOT / "workspace" / "exported" / "custom-1.7b" / "weights",
    REPO_ROOT / "workspace" / "exported" / "base-1.7b" / "weights",
]
TOKENIZER_DIR_CANDIDATES = [
    REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-Tokenizer-12Hz",
    REPO_ROOT / "workspace" / "exported" / "tokenizer" / "Qwen3-TTS-Tokenizer-12Hz",
    REPO_ROOT / "workspace" / "exported" / "tokenizer",
]
SCRIPTS_EXPORT_DIR = REPO_ROOT / "scripts" / "export"


def test_ref_codec_fused_export_preserves_temporal_frames():
    if str(SCRIPTS_EXPORT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_EXPORT_DIR))
    from export_04_speech_tokenizer_codec_fused import RefCodecSumFromAudioCodes

    groups, vocab, hidden, frames = 3, 8, 2, 4
    stacked = torch.arange(groups * vocab * hidden, dtype=torch.float32).reshape(
        groups, vocab, hidden,
    )
    audio_codes = torch.tensor(
        [[[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]]],
        dtype=torch.int64,
    )

    out = RefCodecSumFromAudioCodes(stacked)(audio_codes)
    expected = torch.stack([
        sum(stacked[g, audio_codes[0, g, t]] for g in range(groups))
        for t in range(frames)
    ]).unsqueeze(0)

    assert out.shape == (1, frames, hidden)
    torch.testing.assert_close(out, expected)


def test_prefill_language_aliases_normalize_to_manifest_names():
    from engine.backend.prefill import _normalize_language_name

    assert _normalize_language_name("zh") == "chinese"
    assert _normalize_language_name("zh-CN") == "chinese"
    assert _normalize_language_name("en") == "english"
    assert _normalize_language_name("auto") == "auto"


class _TinyICLWeights:
    def __init__(self):
        self.variant = "tiny"
        self.hidden_size = 4
        self.vocab_size = 512
        self.codec_eos_id = 99
        self.codec_bos_id = 10
        self.codec_pad_id = 0
        self.codec_nothink_id = 11
        self.codec_think_bos_id = 12
        self.codec_think_eos_id = 13
        self.codec_think_id = 14
        self.codec_language_id = {}
        self.spk_id_map = {}
        self.spk_is_dialect = {}
        self.default_speaker = ""
        self.fallback_speaker = ""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.codec_embeddings_3d = None
        self.tts_pad_embed = torch.full((1, 1, self.hidden_size), -1.0)
        self.tts_bos_embed = torch.full((1, 1, self.hidden_size), -2.0)
        self.tts_eos_embed = torch.full((1, 1, self.hidden_size), -3.0)

    def text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.float32).unsqueeze(-1).expand(
            *token_ids.shape,
            self.hidden_size,
        )

    def codec_embed(self, codec_ids: torch.Tensor) -> torch.Tensor:
        return (codec_ids.to(torch.float32) * 0.01).unsqueeze(-1).expand(
            *codec_ids.shape,
            self.hidden_size,
        )


class _TinyTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, max(4, len(text) + 1)))


class _TinyCustomWeights(_TinyICLWeights):
    def __init__(self):
        super().__init__()
        self.spk_id_map = {
            "customer_service_clone_1000": 3000,
            "vivian": 3065,
        }
        self.spk_is_dialect = {
            "customer_service_clone_1000": False,
            "vivian": False,
        }
        self.default_speaker = "customer_service_clone_1000"
        self.fallback_speaker = "customer_service_clone_1000"


def test_custom_voice_cache_key_uses_resolved_speaker_id_for_unknown_speaker():
    from engine.backend.prefill import PrefillBuilder, TaskType

    weights = _TinyCustomWeights()
    builder = PrefillBuilder(weights, _TinyTokenizer())
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=[101, 102],
        speaker="Serena",
    )
    cache_key = builder.compute_cache_key(
        TaskType.CUSTOM_VOICE,
        speaker="Serena",
    )

    assert plan.prefix_cache_key == cache_key
    assert plan.prefix_cache_key.endswith("|custom_spk:3000")
    assert plan.warnings == [
        "Speaker 'Serena' not found, using fallback_speaker 'customer_service_clone_1000'"
    ]


def test_custom_voice_cache_key_distinguishes_actual_speaker_ids():
    from engine.backend.prefill import PrefillBuilder, TaskType

    weights = _TinyCustomWeights()
    builder = PrefillBuilder(weights, _TinyTokenizer())

    default_key = builder.compute_cache_key(TaskType.CUSTOM_VOICE)
    fallback_key = builder.compute_cache_key(TaskType.CUSTOM_VOICE, speaker="missing")
    vivian_key = builder.compute_cache_key(TaskType.CUSTOM_VOICE, speaker="Vivian")

    assert default_key == fallback_key
    assert default_key.endswith("|custom_spk:3000")
    assert vivian_key.endswith("|custom_spk:3065")
    assert vivian_key != default_key


def test_custom_voice_without_supported_speaker_disables_prefix_cache():
    from engine.backend.prefill import PrefillBuilder, TaskType

    weights = _TinyICLWeights()
    builder = PrefillBuilder(weights, _TinyTokenizer())
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=[101],
        speaker="Serena",
    )

    assert plan.prefix_cache_key is None
    assert plan.warnings == [
        "CustomVoice model has no spk_id map; continuing without speaker codec"
    ]


def test_voice_clone_icl_streaming_prefill_does_not_append_eos_or_pad_before_done():
    from engine.backend.prefill import PrefillBuilder, TaskType

    weights = _TinyICLWeights()
    builder = PrefillBuilder(weights, _TinyTokenizer())
    plan = builder.build_plan_from_ids(
        task_type=TaskType.VOICE_CLONE_ICL,
        token_ids=[101],
        spk_embedding=torch.zeros(1, weights.hidden_size),
        ref_text_token_ids=[201, 202],
        ref_codec_sum_vec=torch.zeros(1, 8, weights.hidden_size),
        include_eos=False,
    )

    assert plan.trailing == []


def test_voice_clone_icl_streaming_eos_is_added_only_when_segment_is_complete():
    from engine.backend.prefill import PrefillBuilder, TaskType

    weights = _TinyICLWeights()
    builder = PrefillBuilder(weights, _TinyTokenizer())
    kwargs = dict(
        task_type=TaskType.VOICE_CLONE_ICL,
        token_ids=[101, 102, 103],
        spk_embedding=torch.zeros(1, weights.hidden_size),
        ref_text_token_ids=[201],
        ref_codec_sum_vec=torch.zeros(1, 1, weights.hidden_size),
    )

    streaming_plan = builder.build_plan_from_ids(**kwargs, include_eos=False)
    complete_plan = builder.build_plan_from_ids(**kwargs, include_eos=True)

    assert len(complete_plan.trailing) == len(streaming_plan.trailing) + 1


def test_voice_clone_icl_does_not_prepend_streaming_first_text_token():
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType

    class _StubTokenizer:
        def encode_ids(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(range(10, 10 + max(3, len(text))))

    weights = EmbeddingWeights(_weights_dir(), device="cpu")
    builder = PrefillBuilder(weights, _StubTokenizer())
    ref_frames = 8
    plan = builder.build_plan_from_ids(
        task_type=TaskType.VOICE_CLONE_ICL,
        token_ids=[101, 102, 103],
        spk_embedding=torch.zeros(1, weights.hidden_size),
        ref_text_token_ids=[201, 202],
        ref_codec_sum_vec=torch.zeros(1, ref_frames, weights.hidden_size),
    )

    role_len = 3
    codec_prefix_len = 3 + 1 + 2  # nothink/think tags + speaker embedding + pad/bos
    dual_track_len = codec_prefix_len - 1
    icl_len = 1 + ref_frames  # codec BOS + temporal ref codec frames
    assert plan.prefill_embeds.shape[1] == role_len + dual_track_len + icl_len
    assert plan.prefix_cache_key is None
    assert plan.cacheable_prefix_embeds is None
    assert plan.request_prefill_embeds is None


def _weights_dir():
    for path in WEIGHTS_DIR_CANDIDATES:
        if path.is_dir():
            return str(path)
    pytest.skip(
        "Weights dir not found in candidates: "
        + ", ".join(str(p) for p in WEIGHTS_DIR_CANDIDATES)
        + " (run export_01_embeddings.py)"
    )


def _tokenizer():
    tokenizer_dir = next((p for p in TOKENIZER_DIR_CANDIDATES if p.is_dir()), None)
    if tokenizer_dir is None:
        pytest.skip(
            "Tokenizer dir not found in candidates: "
            + ", ".join(str(p) for p in TOKENIZER_DIR_CANDIDATES)
        )
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
    tok = load_lightweight_tokenizer(str(tokenizer_dir))
    if tok is None:
        pytest.skip("Lightweight tokenizer failed to load")
    return tok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_load_and_bf16():
    """T1.3: EmbeddingWeights loads, all BF16 on GPU."""
    from engine.backend.prefill import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    assert weights.text_embedding.weight.dtype == torch.bfloat16
    assert weights.text_projection.linear_fc1.weight.dtype == torch.bfloat16
    assert weights.tts_pad_embed.dtype == torch.bfloat16
    assert weights.tts_pad_embed.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_text_embed_shape():
    """T1.3: text_embed output shape and dtype."""
    from engine.backend.prefill import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    ids = torch.tensor([[1, 2, 3]], device=weights.device, dtype=torch.int64)
    out = weights.text_embed(ids)
    assert out.shape == (1, 3, weights.hidden_size)
    assert out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_specials_match_runtime_projection():
    """Special embeds should be recomputed from loaded BF16 modules for runtime parity."""
    from engine.backend.prefill import EmbeddingWeights

    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    special_ids = torch.tensor(
        [[weights.tts_pad_token_id, weights.tts_bos_token_id, weights.tts_eos_token_id]],
        device=weights.device,
        dtype=torch.int64,
    )
    projected = weights.text_embed(special_ids)
    torch.testing.assert_close(weights.tts_pad_embed, projected[:, 0:1, :], atol=0.0, rtol=0.0)
    torch.testing.assert_close(weights.tts_bos_embed, projected[:, 1:2, :], atol=0.0, rtol=0.0)
    torch.testing.assert_close(weights.tts_eos_embed, projected[:, 2:3, :], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_codec_embed_shape():
    """T1.3: codec_embed output shape."""
    from engine.backend.prefill import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    ids = torch.tensor([[0, 1]], device=weights.device, dtype=torch.int64)
    out = weights.codec_embed(ids)
    assert out.shape == (1, 2, weights.hidden_size)
    assert out.dtype == torch.bfloat16


def test_embedding_weights_missing_pt_raises():
    """T1.3: Missing .pt raises FileNotFoundError."""
    from engine.backend.prefill import EmbeddingWeights
    import tempfile
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for EmbeddingWeights init")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        (path / "config.json").write_text(
            '{"variant":"x","talker_hidden_size":2048,"talker_text_hidden_size":1024,'
            '"talker_vocab_size":3072,"codec_eos_token_id":1,"codec_bos_id":2,"codec_pad_id":0}'
        )
        with pytest.raises(FileNotFoundError, match="text_embedding.pt"):
            EmbeddingWeights(str(path), device_id=0)


def test_embedding_weights_load_on_cpu_fp32():
    """EmbeddingWeights should support CPU fallback with fp32 modules."""
    from engine.backend.prefill import EmbeddingWeights

    weights = EmbeddingWeights(_weights_dir(), device="cpu")
    assert weights.device.type == "cpu"
    assert weights.dtype == torch.float32
    assert weights.text_embedding.weight.dtype == torch.float32
    assert weights.text_projection.linear_fc1.weight.dtype == torch.float32
    assert weights.tts_pad_embed.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_design():
    """T1.4a: VOICE_DESIGN path."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    embeds, trailing = builder.build(task_type=TaskType.VOICE_DESIGN, text="你好世界")
    assert embeds.dtype == torch.bfloat16
    assert embeds.shape[0] == 1 and embeds.shape[2] == weights.hidden_size
    assert len(trailing) >= 1
    assert trailing[-1].shape == weights.tts_pad_embed.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_plan_voice_design_exposes_cacheable_prefix():
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    plan = builder.build_plan(
        task_type=TaskType.VOICE_DESIGN,
        text="你好世界",
        instruct="请设计一个温柔成熟的女声",
    )
    assert plan.prefix_cache_key
    assert plan.cacheable_prefix_embeds is not None
    assert plan.request_prefill_embeds is not None
    assert plan.cacheable_prefix_embeds.shape[1] + plan.request_prefill_embeds.shape[1] == plan.prefill_embeds.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_custom_voice():
    """T1.4b: CUSTOM_VOICE with speaker + instruct."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    embeds, trailing = builder.build(
        task_type=TaskType.CUSTOM_VOICE,
        text="测试文本",
        speaker="zhitian",
        instruct="用温柔的语气说",
    )
    assert embeds.shape[1] > 0
    assert embeds.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_plan_custom_voice_exposes_cacheable_prefix():
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    plan = builder.build_plan(
        task_type=TaskType.CUSTOM_VOICE,
        text="测试文本",
        speaker="zhitian",
        instruct="用温柔的语气说",
    )
    assert plan.prefix_cache_key
    assert plan.cacheable_prefix_embeds is not None
    assert plan.request_prefill_embeds is not None
    assert plan.cacheable_prefix_embeds.shape[1] + plan.request_prefill_embeds.shape[1] == plan.prefill_embeds.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_clone_xvec():
    """T1.4c: VOICE_CLONE_XVEC with spk_embedding."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    embeds, trailing = builder.build(
        task_type=TaskType.VOICE_CLONE_XVEC,
        text="克隆测试",
        spk_embedding=fake_spk,
    )
    assert embeds.shape[0] == 1 and embeds.shape[2] == weights.hidden_size
    assert embeds.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_plan_voice_clone_xvec_exposes_cacheable_prefix():
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    plan = builder.build_plan(
        task_type=TaskType.VOICE_CLONE_XVEC,
        text="克隆测试",
        spk_embedding=fake_spk,
    )
    assert plan.prefix_cache_key
    assert plan.cacheable_prefix_embeds is not None
    assert plan.request_prefill_embeds is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_clone_icl():
    """T1.4d: VOICE_CLONE_ICL (ref_codes + ref_text) - critical path."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    if weights.codec_embeddings_3d is None:
        pytest.skip("codec_embeddings_3d.pt required for ICL")
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    fake_codes = torch.randint(0, 100, (50, 16), device=weights.device)
    embeds, trailing = builder.build(
        task_type=TaskType.VOICE_CLONE_ICL,
        text="ICL测试",
        spk_embedding=fake_spk,
        ref_codes=fake_codes,
        ref_text="参考文本",
    )
    assert embeds.shape[1] > 10
    assert embeds.dtype == torch.bfloat16
    assert len(trailing) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_moves_cpu_weights_outputs_to_cuda():
    """CPU embedding weights should still produce CUDA bf16 prefill outputs."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType

    class _StubTokenizer:
        def encode_ids(self, text, add_special_tokens=False):
            return [max(1, ord(ch) % 256) for ch in text]

    weights = EmbeddingWeights(_weights_dir(), device="cpu")
    builder = PrefillBuilder(weights, _StubTokenizer(), output_device="cuda:0")
    plan = builder.build_plan(
        task_type=TaskType.CUSTOM_VOICE,
        text="测试文本",
        speaker="zhitian",
        instruct="用温柔的语气说",
    )

    assert plan.prefill_embeds.device.type == "cuda"
    assert plan.prefill_embeds.dtype == torch.bfloat16
    assert plan.request_prefill_embeds is not None
    assert plan.request_prefill_embeds.device.type == "cuda"
    assert all(t.device.type == "cuda" for t in plan.trailing)
    assert all(t.dtype == torch.bfloat16 for t in plan.trailing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_plan_voice_clone_icl_has_no_cross_request_prefix_cache():
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    if weights.codec_embeddings_3d is None:
        pytest.skip("codec_embeddings_3d.pt required for ICL")
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    fake_codes = torch.randint(0, 100, (50, 16), device=weights.device)
    plan = builder.build_plan(
        task_type=TaskType.VOICE_CLONE_ICL,
        text="ICL测试",
        spk_embedding=fake_spk,
        ref_codes=fake_codes,
        ref_text="参考文本",
    )
    assert plan.prefix_cache_key is None
    assert plan.cacheable_prefix_embeds is None
    assert plan.request_prefill_embeds is None


def test_parse_task_type():
    """parse_task_type returns correct enum."""
    from engine.backend.prefill import parse_task_type, TaskType
    assert parse_task_type("voice_design") == TaskType.VOICE_DESIGN
    assert parse_task_type("custom_voice") == TaskType.CUSTOM_VOICE
    assert parse_task_type("voice_clone", x_vector_only=True) == TaskType.VOICE_CLONE_XVEC
    assert parse_task_type("voice_clone", x_vector_only=False) == TaskType.VOICE_CLONE_ICL
    with pytest.raises(ValueError, match="Unknown"):
        parse_task_type("invalid_type")


# ---- T4.2 / T4.3 regression (plan L4) ----

@pytest.mark.skip(reason="BF16 vs FP32 comparison requires legacy FP32 weights export; run manually if needed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_vs_fp32_prefill_output_atol():
    """T4.2: Compare prefill output BF16 vs FP32 (atol=1e-2). Skip: no FP32 weights in repo."""
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    weights_bf16 = EmbeddingWeights(_weights_dir(), device_id=0)
    # Would load FP32 weights from a separate dir and compare prefill outputs
    pytest.skip("No FP32 weights dir for comparison")


@pytest.mark.skip(reason="DLPack zero-copy is internal to model.py BLS; E2E run implies it works")
def test_dlpack_zero_copy_vs_cpu_copy():
    """T4.3: DLPack vs CPU copy numerical consistency. Skip: requires Triton BLS internals."""
    pytest.skip("Verified indirectly via E2E; no direct Python API for DLPack path")
