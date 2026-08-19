# Qwen3-TTS Browser SDK

Framework-neutral TypeScript client for the Qwen3TTS-Streaming OpenAI Realtime
endpoint. It discovers the deployed instance before connecting and fails closed
when a task, audio format, VAD strategy, output policy, reference feature, or
required protocol extension is not advertised.

```ts
import {
  AudioEncoding,
  BrowserAudioPlayer,
  RealtimeTTSClient,
  SynthesisTask,
  VadStrategy,
} from "@xmultimodalinteraction/qwen3tts-browser";

const client = new RealtimeTTSClient({
  capabilitiesUrl: new URL("../v1/capabilities", location.href),
  websocketUrl: new URL("../v1/realtime", location.href),
});
const player = new BrowserAudioPlayer();
await player.start();
client.onEvent((event) => {
  if (event.type === "audio") {
    player.enqueue(event.pcm, 24_000, event.startSample, event.endSample);
  }
});
await client.connect();
const run = await client.synthesize("你好，欢迎使用 Qwen3-TTS。", {
  task: SynthesisTask.CustomVoice,
  speaker: "Serena",
  audio: {encoding: AudioEncoding.PcmS16Le, sample_rate: 24_000, channels: 1},
  vad: {enabled: false, strategy: VadStrategy.Disabled},
});
await run.done;
```

The player uses an AudioWorklet, continuous explicit resampling, a bounded
queue, and playback sample cursors. Feed its progress callback to
`run.acknowledgePlayback()` when guarded delivery is enabled. `WavCollector`
provides independent bounded WAV capture; reaching its limit does not stop live
playback.

The SDK requires a modern browser with WebSocket, Web Audio, AudioWorklet,
BigInt, and ES2022 module support. See the same-release `/demo/#/docs/` portal
for the protocol, deployment, and limitations documentation.
