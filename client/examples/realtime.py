"""Realtime audio — wall-clock aligned frames for playback / WebRTC.

The engine emits audio in an irregular rhythm. ``RealtimeAudioStream`` wraps a
streaming session and yields fixed-size frames on a wall-clock cadence,
inserting silence frames to cover gaps so a playback device / WebRTC track
never underruns.

    python realtime.py [endpoint]

Default endpoint: ws://localhost:50052/v1/ws
"""

from __future__ import annotations

import sys

from qwen3tts import (
    RealtimeAudioStream,
    SessionStartRequest,
    SynthesisConfig,
    TTSClient,
)

ENDPOINT = (
    sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:50052/v1/ws"
)


def main() -> None:
    client = TTSClient.connect(ENDPOINT)

    config = SynthesisConfig(task_type="custom_voice", speaker="serena")
    session = client.open_stream(
        SessionStartRequest(session_id="realtime-demo", config=config)
    )
    session.send_text("你好，欢迎使用实时语音合成。")
    session.end()

    audio_frames = silence_frames = 0
    # 20 ms frames (WebRTC Opus frame size); silence-filled for steady cadence.
    for frame in RealtimeAudioStream(session, chunk_s=0.02, fill_silence=True):
        if frame.is_silence:
            silence_frames += 1
            continue
        audio_frames += 1
        # In a real app: webrtc_track.write(frame.data) / audio_device.play(frame.data)

    print(
        f"audio frames={audio_frames}  silence frames={silence_frames}  "
        f"usage={session.usage}"
    )


if __name__ == "__main__":
    main()
