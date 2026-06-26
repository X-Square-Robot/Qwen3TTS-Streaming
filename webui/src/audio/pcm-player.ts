import type { AudioFormat } from "../types";

export class PcmStreamPlayer {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private fallbackTime = 0;

  async start(): Promise<void> {
    if (this.context) {
      await this.context.resume();
      return;
    }
    this.context = new AudioContext({ sampleRate: 24000 });
    try {
      await this.context.audioWorklet.addModule(new URL("./pcm-worklet.js", import.meta.url));
      this.node = new AudioWorkletNode(this.context, "pcm-stream-player", {
        outputChannelCount: [1]
      });
      this.node.connect(this.context.destination);
    } catch {
      this.node = null;
      this.fallbackTime = this.context.currentTime + 0.04;
    }
    await this.context.resume();
  }

  enqueue(bytes: ArrayBuffer, format: AudioFormat): void {
    if (!this.context) {
      return;
    }
    const samples = decodeSamples(bytes, format);
    if (this.node) {
      const copy = new Float32Array(samples.length);
      copy.set(samples);
      const transferable = copy.buffer as ArrayBuffer;
      this.node.port.postMessage({ type: "push", samples: transferable }, [transferable]);
      return;
    }
    const buffer = this.context.createBuffer(1, samples.length, format.sample_rate || this.context.sampleRate);
    const channel = new Float32Array(samples.length);
    channel.set(samples);
    buffer.copyToChannel(channel, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    const startAt = Math.max(this.context.currentTime + 0.015, this.fallbackTime);
    source.start(startAt);
    this.fallbackTime = startAt + buffer.duration;
  }

  clear(): void {
    this.node?.port.postMessage({ type: "clear" });
  }

  async stop(): Promise<void> {
    this.clear();
    this.node?.disconnect();
    this.node = null;
    if (this.context) {
      await this.context.close();
      this.context = null;
    }
  }
}

function decodeSamples(bytes: ArrayBuffer, format: AudioFormat): Float32Array {
  if (format.encoding === "pcm_s16le") {
    const view = new Int16Array(bytes);
    const out = new Float32Array(view.length);
    for (let i = 0; i < view.length; i += 1) {
      out[i] = Math.max(-1, Math.min(1, view[i] / 32768));
    }
    return out;
  }
  return new Float32Array(bytes);
}
