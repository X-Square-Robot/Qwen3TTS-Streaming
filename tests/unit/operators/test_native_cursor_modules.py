"""Contract tests for the fused native-cursor streaming step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))

from native_cursor_modules import CursorHead, CursorStreamingStep  # noqa: E402
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
