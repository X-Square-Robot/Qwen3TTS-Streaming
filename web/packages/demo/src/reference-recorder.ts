import {WavCollector} from "@xmultimodalinteraction/qwen3tts-browser";

export interface ReferenceRecorder {
  stop(): Promise<File>;
}

/** Record microphone PCM into the RIFF/WAV format accepted by the engine. */
export async function startReferenceRecorder(
  maxBytes: number,
  onLimit: () => void,
): Promise<ReferenceRecorder> {
  if (!navigator.mediaDevices || !("getUserMedia" in navigator.mediaDevices)) {
    throw new Error("当前浏览器或安全上下文不支持麦克风录音");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true},
  });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const silent = context.createGain();
  silent.gain.value = 0;
  const collector = new WavCollector({sampleRate: context.sampleRate, maxBytes});
  let limitNotified = false;
  let stopped: Promise<File> | null = null;

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index] ?? 0));
      pcm[index] = sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
    }
    if (!collector.append(pcm) && !limitNotified) {
      limitNotified = true;
      queueMicrotask(onLimit);
    }
  };
  source.connect(processor);
  processor.connect(silent);
  silent.connect(context.destination);
  await context.resume();

  return {
    stop(): Promise<File> {
      if (stopped) return stopped;
      stopped = (async () => {
        processor.onaudioprocess = null;
        source.disconnect();
        processor.disconnect();
        silent.disconnect();
        stream.getTracks().forEach((track) => track.stop());
        await context.close();
        if (collector.snapshot().samples === 0) throw new Error("没有采集到麦克风音频");
        return new File([collector.toArrayBuffer()], "reference-recording.wav", {
          type: "audio/wav",
        });
      })();
      return stopped;
    },
  };
}
