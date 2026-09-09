"""Contract tests for the fused native-cursor streaming step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))

from native_cursor_modules import (  # noqa: E402
    CursorHead,
    CursorStreamingStep,
    build_cursor_head_from_checkpoint,
)
from utils import (  # noqa: E402
    NATIVE_CURSOR_HEAD_FILENAME,
    prepare_native_cursor_head,
)


class _TinyTalker(nn.Module):
    num_layers = 1

    def forward(self, input_embeds, position_ids, token_counts, gumbel_noise,
                cp_gumbel_noise, temperature, penalty, attention_bias, *past_kv):
        b, _, h = input_embeds.shape
        v = token_counts.shape[1]
        codec0 = torch.full((b, 1), 7, dtype=torch.long, device=input_embeds.device)
        full_codec = torch.cat([codec0, torch.zeros(b, 15, dtype=torch.long, device=input_embeds.device)], dim=1)
        codec_sum = torch.zeros(b, 1, h, device=input_embeds.device)
        hidden = torch.zeros(b, 1, h, device=input_embeds.device)
        logits = torch.zeros(b, 1, v, device=input_embeds.device)
        updated = token_counts + torch.nn.functional.one_hot(codec0[:, 0], v).to(token_counts.dtype)
        kv = torch.zeros(b, 1, 1, 1, device=input_embeds.device)
        return codec_sum, full_codec, hidden, logits, updated, kv, kv


class _TinyCode2Wav(nn.Module):
    def forward(self, codes, cache_position, attention_bias, *states):
        b = codes.shape[0]
        device = codes.device
        kv = torch.zeros(b, 1, 1, 1, device=device)
        return torch.zeros(b, 1, device=device), kv, kv, states[0] + 1


def _initial_state(batch: int, d: int):
    return (
        torch.zeros(batch, 30, d),
        torch.zeros(batch, d),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch),
        torch.zeros(batch, 8),
    )


def test_streaming_trunk_and_matcher_match_offline_lookahead() -> None:
    torch.manual_seed(7)
    head = CursorHead(n_labels_plus_blank=9, d=16)
    head.eval()
    step = CursorStreamingStep(head).eval()

    codes = torch.tensor([3, 17, 4, 21, 8, 9, 2], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    label_count = torch.tensor([5], dtype=torch.long)
    active = torch.ones(1, dtype=torch.bool)
    text_start = torch.zeros(1, dtype=torch.long)
    override_valid = torch.zeros(1, dtype=torch.bool)
    override_mu = torch.zeros(1)

    conv_history, last_input, seen, frames_since, delta_history = _initial_state(
        1, head.d
    )
    streamed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for code in codes:
            out = step(
                code.view(1),
                labels,
                label_count,
                active,
                torch.zeros(1) if not streamed else mu,
                frames_since,
                delta_history,
                conv_history,
                last_input,
                seen,
                text_start,
                override_valid,
                override_mu,
            )
            valid, mu, delta, confidence = out[:4]
            conv_history, last_input, seen = out[7], out[8], out[9]
            frames_since, delta_history = out[5], out[6]
            if bool(valid.item()):
                streamed.append((mu.clone(), delta.clone(), confidence.clone()))

        offline_h = head.forward_trunk(codes.view(1, -1))
        etab = head.encode_text(labels)
        mu_ref = torch.zeros(1)
        fs_ref = torch.zeros(1)
        dh_ref = torch.zeros(1, 8)
        reference: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for frame_idx in range(codes.numel() - 1):
            base = torch.floor(mu_ref).long().view(1, 1)
            loc = head.location_features(mu_ref, fs_ref, dh_ref.mean(1)).view(1, 1, 3)
            known = label_count.view(1, 1)
            logits = head.window_logits(
                offline_h[:, frame_idx : frame_idx + 1], etab, base, loc, known
            )
            prob = logits.softmax(-1).view(1, -1)
            delta_ref = (prob * head.offset_values).sum(-1)
            new_mu = torch.minimum(
                (mu_ref + delta_ref).clamp_min(0), label_count.float()
            )
            crossed = torch.floor(new_mu) > torch.floor(mu_ref)
            fs_ref = torch.where(crossed, torch.zeros_like(fs_ref), fs_ref + 1.0)
            dh_ref = torch.cat([dh_ref[:, 1:], delta_ref[:, None]], dim=1)
            mu_ref = new_mu
            reference.append((mu_ref.clone(), delta_ref.clone(), prob.max(-1).values))

    assert len(streamed) == len(reference)
    for got, want in zip(streamed, reference):
        torch.testing.assert_close(got[0], want[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got[2], want[2], rtol=1e-5, atol=1e-5)


def test_streaming_cursor_matches_offline_reference_for_many_frames() -> None:
    """Exercise recurrent state for a long enough run to expose drift."""
    torch.manual_seed(20260904)
    head = CursorHead(n_labels_plus_blank=17, d=16).eval()
    step = CursorStreamingStep(head).eval()
    b, frames, labels_count = 2, 96, 32
    codec0 = torch.randint(0, head.emb.num_embeddings, (b, frames))
    labels = torch.randint(1, head.n_labels_plus_blank, (b, labels_count))
    known = torch.tensor([labels_count, labels_count - 5], dtype=torch.long)
    active = torch.ones(b, dtype=torch.int64)
    text_start = torch.tensor([0, 3], dtype=torch.long)

    mu = torch.zeros(b)
    frames_since = torch.zeros(b)
    delta_history = torch.zeros(b, 8)
    conv_history = torch.zeros(b, head.history_width, head.d)
    last_trunk_input = torch.zeros(b, head.d)
    seen_frames = torch.zeros(b, dtype=torch.long)
    override_valid = torch.zeros(b, dtype=torch.int64)
    override_mu = torch.zeros(b)

    with torch.no_grad():
        offline_h = head.forward_trunk(codec0)
        text_h = head.encode_text(labels)
        for frame_idx in range(frames):
            state_before = (
                mu.clone(),
                frames_since.clone(),
                delta_history.clone(),
            )
            actual = step(
                codec0[:, frame_idx],
                labels,
                known,
                active,
                mu,
                frames_since,
                delta_history,
                conv_history,
                last_trunk_input,
                seen_frames,
                text_start,
                override_valid,
                override_mu,
            )

            if frame_idx > 0:
                mu_before, frames_before, deltas_before = state_before
                base = torch.floor(mu_before).long().view(b, 1)
                loc = head.location_features(
                    mu_before,
                    frames_before,
                    deltas_before.mean(dim=1),
                ).view(b, 1, 3)
                logits = head.window_logits(
                    offline_h[:, frame_idx - 1 : frame_idx],
                    text_h,
                    base,
                    loc,
                    known.view(b, 1),
                )
                probability = logits.softmax(-1).view(b, -1)
                delta_ref = (probability * head.offset_values).sum(-1)
                valid_ref = active.bool() & (frame_idx - 1 >= text_start) & (known > 0)
                delta_ref = torch.where(valid_ref, delta_ref, torch.zeros_like(delta_ref))
                candidate = (mu_before + delta_ref).clamp_min(0.0)
                candidate = torch.minimum(candidate, known.float())
                mu_ref = torch.where(valid_ref, candidate, mu_before)
                confidence_ref = torch.where(
                    valid_ref,
                    probability.max(-1).values,
                    torch.zeros_like(delta_ref),
                )
                torch.testing.assert_close(actual[0], valid_ref.to(torch.int64))
                torch.testing.assert_close(actual[1], mu_ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(actual[2], delta_ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    actual[3], confidence_ref, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(actual[4], torch.floor(mu_ref).long())

            mu, frames_since, delta_history = actual[1], actual[5], actual[6]
            conv_history, last_trunk_input, seen_frames = actual[7], actual[8], actual[9]


def test_streaming_cursor_onnx_state_recurrence_matches_pytorch(tmp_path) -> None:
    """Run the exported streaming step repeatedly, feeding its state back."""
    head = CursorHead(n_labels_plus_blank=9, d=8).eval()
    step = CursorStreamingStep(head).eval()
    b, max_labels = 1, 6
    initial = (
        torch.tensor([3], dtype=torch.long),
        torch.ones(b, max_labels, dtype=torch.long),
        torch.tensor([max_labels], dtype=torch.long),
        torch.ones(b, dtype=torch.int64),
        torch.zeros(b),
        torch.zeros(b),
        torch.zeros(b, 8),
        torch.zeros(b, head.history_width, head.d),
        torch.zeros(b, head.d),
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.int64),
        torch.zeros(b),
    )
    input_names = [
        "codec0",
        "label_ids",
        "label_count",
        "cursor_active",
        "mu",
        "frames_since_advance",
        "delta_history",
        "conv_history",
        "last_trunk_input",
        "seen_frames",
        "text_start_frame",
        "override_valid",
        "override_mu",
    ]
    output_names = [f"output_{idx}" for idx in range(10)]
    onnx_path = tmp_path / "cursor_step.onnx"
    torch.onnx.export(
        step,
        initial,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    torch_state = list(initial)
    ort_state = list(initial)
    session_input_names = {item.name for item in session.get_inputs()}
    with torch.no_grad():
        for frame_idx in range(20):
            code = torch.tensor([frame_idx + 1], dtype=torch.long)
            torch_args = [code, *torch_state[1:]]
            expected = step(*torch_args)
            feed = {
                name: value.detach().cpu().numpy()
                for name, value in zip(input_names, [code, *ort_state[1:]])
                if name in session_input_names
            }
            actual = session.run(output_names, feed)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(
                    torch.from_numpy(got), want.cpu(), rtol=1e-4, atol=1e-4
                )
            torch_state[4:7] = [expected[1], expected[5], expected[6]]
            torch_state[7:10] = [expected[7], expected[8], expected[9]]
            ort_outputs = [torch.from_numpy(value) for value in actual]
            ort_state[4:7] = [ort_outputs[1], ort_outputs[5], ort_outputs[6]]
            ort_state[7:10] = [ort_outputs[7], ort_outputs[8], ort_outputs[9]]


_RELEASED_CURSOR_HEAD = (
    REPO_ROOT
    / "workspace"
    / "models"
    / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
).resolve() / NATIVE_CURSOR_HEAD_FILENAME


@pytest.mark.skipif(
    not _RELEASED_CURSOR_HEAD.is_file(),
    reason="released custom-1.7B cursor head is not mounted",
)
def test_released_cursor_head_streaming_matches_offline_for_many_frames() -> None:
    """Regression against the model-owned 0818 head, not only a tiny random head."""
    head, meta = build_cursor_head_from_checkpoint(_RELEASED_CURSOR_HEAD)
    assert meta["left_context"] == 29
    assert meta["right_context"] == 1
    assert meta["parameter_count"] == 2_036_991
    head.eval()
    step = CursorStreamingStep(head).eval()
    torch.manual_seed(20260905)
    b, frames, max_labels = 2, 96, 48
    codec0 = torch.randint(0, head.emb.num_embeddings, (b, frames))
    labels = torch.randint(1, head.n_labels_plus_blank, (b, max_labels))
    known = torch.tensor([max_labels, max_labels - 7], dtype=torch.long)
    active = torch.ones(b, dtype=torch.int64)
    text_start = torch.tensor([0, 4], dtype=torch.long)
    mu = torch.zeros(b)
    frames_since = torch.zeros(b)
    delta_history = torch.zeros(b, 8)
    conv_history = torch.zeros(b, head.history_width, head.d)
    last_trunk_input = torch.zeros(b, head.d)
    seen_frames = torch.zeros(b, dtype=torch.long)
    override_valid = torch.zeros(b, dtype=torch.int64)
    override_mu = torch.zeros(b)

    with torch.no_grad():
        offline_h = head.forward_trunk(codec0)
        text_h = head.encode_text(labels)
        for frame_idx in range(frames):
            mu_before = mu.clone()
            frames_before = frames_since.clone()
            deltas_before = delta_history.clone()
            actual = step(
                codec0[:, frame_idx],
                labels,
                known,
                active,
                mu,
                frames_since,
                delta_history,
                conv_history,
                last_trunk_input,
                seen_frames,
                text_start,
                override_valid,
                override_mu,
            )
            if frame_idx > 0:
                base = torch.floor(mu_before).long().view(b, 1)
                loc = head.location_features(
                    mu_before,
                    frames_before,
                    deltas_before.mean(dim=1),
                ).view(b, 1, 3)
                logits = head.window_logits(
                    offline_h[:, frame_idx - 1 : frame_idx],
                    text_h,
                    base,
                    loc,
                    known.view(b, 1),
                )
                probability = logits.softmax(-1).view(b, -1)
                delta_ref = (probability * head.offset_values).sum(-1)
                valid_ref = active.bool() & (frame_idx - 1 >= text_start) & (known > 0)
                delta_ref = torch.where(valid_ref, delta_ref, torch.zeros_like(delta_ref))
                candidate = torch.minimum(
                    (mu_before + delta_ref).clamp_min(0.0), known.float()
                )
                mu_ref = torch.where(valid_ref, candidate, mu_before)
                torch.testing.assert_close(actual[0], valid_ref.to(torch.int64))
                torch.testing.assert_close(actual[1], mu_ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(actual[2], delta_ref, rtol=1e-5, atol=1e-5)
            mu, frames_since, delta_history = actual[1], actual[5], actual[6]
            conv_history, last_trunk_input, seen_frames = actual[7], actual[8], actual[9]


def test_cursor_step_keeps_trunk_state_when_text_is_not_visible() -> None:
    torch.manual_seed(11)
    head = CursorHead(n_labels_plus_blank=6, d=8)
    step = CursorStreamingStep(head).eval()
    state = _initial_state(1, head.d)
    conv_history, last_input, seen, frames_since, delta_history = state
    common = dict(
        label_ids=torch.ones(1, 3, dtype=torch.long),
        label_count=torch.zeros(1, dtype=torch.long),
        cursor_active=torch.ones(1, dtype=torch.bool),
        text_start_frame=torch.full((1,), 3, dtype=torch.long),
        override_valid=torch.zeros(1, dtype=torch.bool),
        override_mu=torch.zeros(1),
    )
    with torch.no_grad():
        for code in (1, 2, 3):
            out = step(
                torch.tensor([code]),
                common["label_ids"],
                common["label_count"],
                common["cursor_active"],
                torch.zeros(1),
                frames_since,
                delta_history,
                conv_history,
                last_input,
                seen,
                common["text_start_frame"],
                common["override_valid"],
                common["override_mu"],
            )
            assert not bool(out[0].item())
            conv_history, last_input, seen = out[7], out[8], out[9]
            frames_since, delta_history = out[5], out[6]
    assert int(seen.item()) == 3
    torch.testing.assert_close(frames_since, torch.zeros_like(frames_since))
    torch.testing.assert_close(delta_history, torch.zeros_like(delta_history))


def test_cursor_step_is_a_full_state_passthrough_when_inactive() -> None:
    """Prefill must not consume a decoder cursor frame or mutate its state."""
    torch.manual_seed(17)
    head = CursorHead(n_labels_plus_blank=6, d=8).eval()
    step = CursorStreamingStep(head).eval()
    b = 2
    labels = torch.ones(b, 3, dtype=torch.long)
    label_count = torch.tensor([3, 2], dtype=torch.long)
    active = torch.zeros(b, dtype=torch.int64)
    mu = torch.tensor([1.25, 0.5])
    frames_since = torch.tensor([2.0, 4.0])
    delta_history = torch.randn(b, 8)
    conv_history = torch.randn(b, head.history_width, head.d)
    last_trunk_input = torch.randn(b, head.d)
    seen_frames = torch.tensor([7, 11], dtype=torch.long)
    text_start = torch.zeros(b, dtype=torch.long)
    override_valid = torch.ones(b, dtype=torch.int64)
    override_mu = torch.tensor([4.0, 3.0])

    with torch.no_grad():
        out = step(
            torch.tensor([13, 29], dtype=torch.long),
            labels,
            label_count,
            active,
            mu,
            frames_since,
            delta_history,
            conv_history,
            last_trunk_input,
            seen_frames,
            text_start,
            override_valid,
            override_mu,
        )

    assert not bool(out[0].any())
    torch.testing.assert_close(out[1], mu)
    torch.testing.assert_close(out[2], torch.zeros_like(mu))
    torch.testing.assert_close(out[3], torch.zeros_like(mu))
    torch.testing.assert_close(out[5], frames_since)
    torch.testing.assert_close(out[6], delta_history)
    torch.testing.assert_close(out[7], conv_history)
    torch.testing.assert_close(out[8], last_trunk_input)
    torch.testing.assert_close(out[9], seen_frames)


def _tiny_fused_inputs():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
    from export_09_talker_code2wav_fused import (  # noqa: E402
        TalkerCode2WavCursorFusedONNX,
    )

    head = CursorHead(n_labels_plus_blank=6, d=8)
    fused = TalkerCode2WavCursorFusedONNX(
        _TinyTalker(), _TinyCode2Wav(), 1, CursorStreamingStep(head), 4
    ).eval()
    b, h, v = 1, 8, 32
    base = (
        torch.zeros(b, 1, h),
        torch.zeros(b, 3, 1, 1, dtype=torch.long),
        torch.zeros(b, 1, 1, 1),
        torch.zeros(b, v, dtype=torch.long),
        torch.zeros(b, 50),
        torch.zeros(b, 15, 50),
        torch.ones(b, 1),
        torch.ones(b, 1),
        torch.zeros(b, 1),
        torch.zeros(b, 1, 1, 2),
        torch.zeros(b, 2, 1, 1, 1),
        torch.zeros(b, 2, 1, 1, 1),
    )
    cursor = (
        torch.ones(b, 4, dtype=torch.long), torch.tensor([3]),
        torch.ones(b, dtype=torch.long), torch.zeros(b), torch.zeros(b),
        torch.zeros(b, 8), torch.zeros(b, 30, h), torch.zeros(b, h),
        torch.zeros(b, dtype=torch.long), torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.long), torch.zeros(b),
    )
    return fused, base + cursor + (torch.zeros(b, 1, 1),)


def test_fused_export_prefers_model_owned_speech_tokenizer(tmp_path) -> None:
    from export_09_talker_code2wav_fused import resolve_fused_tokenizer_path

    model_dir = tmp_path / "x2-model"
    bundled = model_dir / "speech_tokenizer"
    bundled.mkdir(parents=True)

    assert resolve_fused_tokenizer_path(model_dir) == bundled


def test_fused_export_keeps_split_tokenizer_fallback(tmp_path, monkeypatch) -> None:
    from export_09_talker_code2wav_fused import resolve_fused_tokenizer_path

    model_dir = tmp_path / "legacy-model"
    models_dir = tmp_path / "models"
    fallback = models_dir / "Qwen3-TTS-Tokenizer-12Hz"
    fallback.mkdir(parents=True)

    monkeypatch.setattr(
        "export_09_talker_code2wav_fused.resolve_tokenizer_path",
        lambda value: fallback,
    )

    assert resolve_fused_tokenizer_path(model_dir, str(models_dir)) == fallback


def test_export_cursor_vocab_fingerprint_is_order_independent() -> None:
    from engine.core.native_cursor_labelizer import vocab_fingerprint
    from export_09_talker_code2wav_fused import vocab_fingerprint as export_fingerprint

    vocab = {"zhi": 3, "bai": 1, "fen": 2}

    assert export_fingerprint(vocab) == vocab_fingerprint(vocab)
    assert export_fingerprint(dict(reversed(list(vocab.items())))) == export_fingerprint(vocab)


def test_cursor_fused_wrapper_connects_sampled_codec0_once() -> None:
    fused, inputs = _tiny_fused_inputs()
    out = fused(*inputs)
    # 8 legacy outputs + one C2W state + 10 cursor outputs + codec0 alias.
    assert len(out) == 20
    assert int(out[-1].item()) == int(out[2][0, 0].item()) == 7


def test_cursor_fused_wrapper_exports_to_onnx(tmp_path) -> None:
    fused, inputs = _tiny_fused_inputs()
    output_names = [f"output_{i}" for i in range(20)]
    torch.onnx.export(
        fused,
        inputs,
        str(tmp_path / "cursor_fused.onnx"),
        input_names=[f"input_{i}" for i in range(len(inputs))],
        output_names=output_names,
        opset_version=18,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(tmp_path / "cursor_fused.onnx"), providers=["CPUExecutionProvider"]
    )
    reference = fused(*inputs)
    feed = {
        item.name: inputs[int(item.name.rsplit("_", 1)[1])].detach().cpu().numpy()
        for item in session.get_inputs()
    }
    actual = session.run(None, feed)
    assert len(actual) == len(reference)
    for got, want in zip(actual, reference):
        torch.testing.assert_close(torch.from_numpy(got), want.detach().cpu(), rtol=1e-4, atol=1e-4)


def test_cursor_profile_keeps_fixed_label_and_state_shapes() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))
    from trt_fused_talk_c2w_profiles import compute_fused_decode_profiles  # noqa: E402

    minimum, optimum, maximum = compute_fused_decode_profiles(
        2048,
        8,
        128,
        28,
        8,
        cursor_enabled=True,
        cursor_max_labels=512,
        cursor_d=256,
        cursor_history=30,
    )
    assert "cursor_label_ids:1x512" in minimum
    assert "cursor_label_ids:8x512" in optimum
    assert "cursor_conv_history_in:8x30x256" in maximum


def test_executor_cursor_plan_pads_without_using_bpe_ids() -> None:
    from engine.backend.executor import Executor  # noqa: E402
    from engine.backend.kv_cache_pool import ModelConfig, SlotKVState  # noqa: E402

    executor = Executor.__new__(Executor)
    executor._cursor_enabled = True
    executor._cursor_max_labels = 8
    executor._cursor_d = 4
    executor._cursor_history = 6
    executor._device = torch.device("cpu")
    executor._config = ModelConfig(dtype=torch.float32)
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(
        slot, torch.tensor([4, 5, 6], dtype=torch.long), text_start_frame=2
    )
    assert tuple(slot.cursor_label_ids.shape) == (1, 8)
    assert slot.cursor_label_ids[0, :3].tolist() == [4, 5, 6]
    assert int(slot.cursor_label_count.item()) == 3
    assert int(slot.cursor_text_start_frame.item()) == 2
    assert int(slot.cursor_active.item()) == 1
    assert tuple(slot.cursor_conv_history.shape) == (1, 6, 4)


def test_model_owned_cursor_head_is_staged_with_exported_weights(tmp_path) -> None:
    model_dir = tmp_path / "0818-trained"
    model_dir.mkdir()
    source_head = model_dir / NATIVE_CURSOR_HEAD_FILENAME
    source_head.write_bytes(b"native-cursor-head")
    exported_weights = tmp_path / "exported" / "custom-1.7b" / "weights"

    resolved = prepare_native_cursor_head(model_dir, exported_weights)

    assert resolved == exported_weights / NATIVE_CURSOR_HEAD_FILENAME
    assert resolved.read_bytes() == source_head.read_bytes()


def test_exported_cursor_head_can_enable_standalone_export(tmp_path) -> None:
    model_dir = tmp_path / "official-model"
    model_dir.mkdir()
    exported_weights = tmp_path / "exported" / "weights"
    exported_weights.mkdir(parents=True)
    package_head = exported_weights / NATIVE_CURSOR_HEAD_FILENAME
    package_head.write_bytes(b"packaged-head")

    resolved = prepare_native_cursor_head(model_dir, exported_weights)

    assert resolved == package_head


def test_conflicting_model_and_exported_cursor_heads_fail_closed(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / NATIVE_CURSOR_HEAD_FILENAME).write_bytes(b"source-head")
    exported_weights = tmp_path / "exported" / "weights"
    exported_weights.mkdir(parents=True)
    (exported_weights / NATIVE_CURSOR_HEAD_FILENAME).write_bytes(b"stale-head")

    with pytest.raises(ValueError, match="conflicts"):
        prepare_native_cursor_head(model_dir, exported_weights)
