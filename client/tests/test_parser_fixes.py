"""Regression tests for client-side parsing bugs.

- segment_id == 0 must survive decode (was corrupted to -1 by an `or -1` fallback)
- gRPC capabilities enum fields must decode to names, not their integer values
- pcm_s16le must decode into the valid [-1, 1] range (divisor 32768, not 32767)
"""

from __future__ import annotations

import pytest

from qwen3tts._internal.utils import decode_stream_event


class TestSegmentIdDecode:
    def test_segment_id_zero_is_preserved(self):
        # 0 is a legitimate first-segment id, not the "no segment" sentinel.
        assert (
            decode_stream_event({"type": "segment_start", "segment_id": 0}).segment_id
            == 0
        )

    def test_segment_id_nonzero_is_preserved(self):
        assert (
            decode_stream_event({"type": "segment_start", "segment_id": 3}).segment_id
            == 3
        )

    def test_missing_segment_id_defaults_to_sentinel(self):
        assert decode_stream_event({"type": "done"}).segment_id == -1


class TestGrpcCapabilitiesEnumDecode:
    def test_enum_fields_decode_to_names(self):
        pytest.importorskip("grpc")
        from qwen3tts._proto import tts_pb2
        from qwen3tts._adapters.engine_grpc import _capabilities_message_to_dict

        resp = tts_pb2.GetCapabilitiesResponse()
        resp.supported_input_modes.append(tts_pb2.INPUT_MODE_FULL_TEXT)
        resp.supported_input_modes.append(tts_pb2.INPUT_MODE_TOKEN)
        resp.supported_group_policies.append(tts_pb2.GROUP_POLICY_AUTO)

        decoded = _capabilities_message_to_dict(resp)
        assert decoded["supported_input_modes"] == ["full_text", "token"]
        assert decoded["supported_group_policies"] == ["auto"]


class TestPcmS16leDecode:
    def test_full_scale_negative_stays_in_range(self):
        np = pytest.importorskip("numpy")
        from qwen3tts.audio import decode_audio_bytes_to_array

        pcm = np.array([-32768, 32767, 0], dtype=np.int16).tobytes()
        arr = decode_audio_bytes_to_array(pcm, encoding="pcm_s16le")

        assert float(arr.min()) >= -1.0
        assert float(arr.max()) <= 1.0
        assert float(arr.min()) == -1.0  # -32768 / 32768
