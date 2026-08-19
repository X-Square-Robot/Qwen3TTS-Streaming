declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  abstract process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
}
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;

const MAX_QUEUED_FRAMES = 48_000 * 10;

class QwenPcmPlayerProcessor extends AudioWorkletProcessor {
  private chunks: Float32Array[] = [];
  private chunkOffset = 0;
  private queuedFrames = 0;
  private consumedSinceReport = 0;
  private paused = false;
  private underrunReported = false;

  constructor() {
    super();
    this.port.onmessage = (message: MessageEvent) => {
      const payload = message.data as {type?: string; samples?: ArrayBuffer};
      if (payload.type === "push" && payload.samples instanceof ArrayBuffer) {
        const samples = new Float32Array(payload.samples);
        if (this.queuedFrames + samples.length > MAX_QUEUED_FRAMES) {
          this.port.postMessage({type: "overflow", queuedFrames: this.queuedFrames});
          return;
        }
        this.chunks.push(samples);
        this.queuedFrames += samples.length;
        this.underrunReported = false;
      } else if (payload.type === "clear") {
        this.chunks = [];
        this.chunkOffset = 0;
        this.queuedFrames = 0;
      } else if (payload.type === "pause") {
        this.paused = true;
      } else if (payload.type === "resume") {
        this.paused = false;
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const output = outputs[0]?.[0];
    if (!output || this.paused) return true;
    let outputOffset = 0;
    while (outputOffset < output.length && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      if (!chunk) break;
      const count = Math.min(output.length - outputOffset, chunk.length - this.chunkOffset);
      output.set(chunk.subarray(this.chunkOffset, this.chunkOffset + count), outputOffset);
      outputOffset += count;
      this.chunkOffset += count;
      this.queuedFrames -= count;
      this.consumedSinceReport += count;
      if (this.chunkOffset === chunk.length) {
        this.chunks.shift();
        this.chunkOffset = 0;
      }
    }
    if (this.consumedSinceReport >= 1024 || (this.queuedFrames === 0 && this.consumedSinceReport > 0)) {
      this.port.postMessage({type: "consumed", frames: this.consumedSinceReport});
      this.consumedSinceReport = 0;
    }
    if (outputOffset === 0 && !this.underrunReported) {
      this.underrunReported = true;
      this.port.postMessage({type: "underrun"});
    }
    return true;
  }
}

registerProcessor("qwen3tts-pcm-player", QwenPcmPlayerProcessor);
