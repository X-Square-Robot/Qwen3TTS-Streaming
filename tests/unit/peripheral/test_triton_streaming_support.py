from __future__ import annotations

import json

import numpy as np

from tests.support.triton_streaming import (
    build_variant_request_payload,
    build_text_stream_requests,
    infer_stream_sequence,
)


def test_build_text_stream_requests_expands_append_and_complete():
    requests = build_text_stream_requests(
        {
            "action": "init",
            "session_id": "sid-123",
            "task_type": "custom_voice",
            "speaker": "Serena",
            "text": "",
        },
        ["你好，", "世界"],
    )

    assert requests == [
        {
            "action": "init",
            "session_id": "sid-123",
            "task_type": "custom_voice",
            "speaker": "Serena",
            "text": "",
        },
        {
            "action": "append_text",
            "session_id": "sid-123",
            "text": "你好，",
        },
        {
            "action": "append_text",
            "session_id": "sid-123",
            "text": "世界",
        },
        {
            "action": "text_complete",
            "session_id": "sid-123",
        },
    ]


def test_build_variant_request_payload_maps_custom_and_design_variants():
    custom = build_variant_request_payload(
        variant="custom-1.7b",
        text="你好",
        language="Chinese",
        speaker="Serena",
        instruct="温柔",
    )
    design = build_variant_request_payload(
        variant="design-1.7b",
        text="Hello",
        language="English",
        instruct="Calm",
    )

    assert custom == {
        "text": "你好",
        "language": "Chinese",
        "task_type": "custom_voice",
        "speaker": "Serena",
        "instruct": "温柔",
    }
    assert design == {
        "text": "Hello",
        "language": "English",
        "task_type": "voice_design",
        "instruct": "Calm",
    }


def test_infer_stream_sequence_accumulates_audio_across_requests():
    sent_payloads: list[dict[str, object]] = []

    class _Result:
        def __init__(self, event_type="", payload=None, audio=b"", is_final=False):
            self._values = {
                "event_type": np.array([event_type], dtype=object),
                "event_json": np.array(
                    [
                        json.dumps(payload or {}, ensure_ascii=False)
                        if payload is not None
                        else ""
                    ],
                    dtype=object,
                ),
                "audio_chunk": np.array([audio], dtype=object),
                "is_final": np.array([is_final], dtype=bool),
            }

        def as_numpy(self, name):
            return self._values.get(name)

    class _InferInput:
        def __init__(self, name, shape, datatype):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.data = None

        def set_data_from_numpy(self, data):
            self.data = data

    class _RequestedOutput:
        def __init__(self, name):
            self.name = name

    class _Client:
        def __init__(self):
            self.callback = None

        def start_stream(self, callback):
            self.callback = callback

        def stop_stream(self):
            pass

        def async_stream_infer(self, model_name, inputs, outputs):
            raw = inputs[0].data.reshape(-1)[0]
            payload = json.loads(
                raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            )
            sent_payloads.append(payload)
            action = payload.get("action", "synthesize")
            if action == "init":
                self.callback(
                    _Result(
                        "start",
                        {
                            "audio_format": {
                                "encoding": "pcm_f32",
                                "sample_rate": 24000,
                                "channels": 1,
                            }
                        },
                    ),
                    None,
                )
            elif action == "append_text":
                self.callback(
                    _Result(
                        "audio",
                        {"meta": {"phase": "token"}},
                        np.array([0.5], dtype=np.float32).tobytes(),
                    ),
                    None,
                )
            elif action == "text_complete":
                self.callback(_Result("end", {"meta": {}}, is_final=True), None)

    class _GrpcModule:
        InferInput = _InferInput
        InferRequestedOutput = _RequestedOutput

    client = _Client()
    stream = infer_stream_sequence(
        client,
        _GrpcModule,
        build_text_stream_requests(
            {
                "action": "init",
                "session_id": "sid-456",
                "task_type": "custom_voice",
                "speaker": "Serena",
                "text": "",
            },
            ["片段一", "片段二"],
        ),
        timeout=1.0,
        result_text="片段一 片段二",
        session_id="sid-456",
    )

    assert [payload["action"] for payload in sent_payloads] == [
        "init",
        "append_text",
        "append_text",
        "text_complete",
    ]
    assert stream.session_id == "sid-456"
    assert stream.text == "片段一 片段二"
    assert stream.error is None
    assert stream.num_chunks == 2
    assert stream.total_samples == 2
    assert stream.audio is not None
    assert stream.audio.tolist() == [0.5, 0.5]
