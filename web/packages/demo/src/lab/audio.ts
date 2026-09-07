/** Audio helpers used by the optional engineering trace panels.
 *
 * The public Demo playback path uses BrowserAudioPlayer from the Browser SDK.
 * This module is intentionally smaller: it only turns the diagnostic WebSocket
 * byte stream into a bounded downloadable WAV and a best-effort live preview.
 */

export interface PcmFormat {
  readonly encoding: "pcm_f32" | "pcm_s16le";
  readonly sampleRate: number;
  readonly channels: number;
}

export const DEFAULT_PCM_FORMAT: PcmFormat = {
  encoding: "pcm_f32",
  sampleRate: 24_000,
  channels: 1,
};

export const WAV_HEADER_BYTES = 44;

/** Convert one or more arbitrarily split PCM frames into a mono PCM16 WAV. */
export function wavBlobFromPcm(
  parts: readonly ArrayBuffer[],
  format: PcmFormat = DEFAULT_PCM_FORMAT,
): Blob {
  const bytesPerInputSample = format.encoding === "pcm_s16le" ? 2 : 4;
  const channels = safeChannels(format.channels);
  const frameBytes = bytesPerInputSample * channels;
  const inputBytes = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const sampleCount = Math.floor(inputBytes / frameBytes);
  const output = new ArrayBuffer(WAV_HEADER_BYTES + sampleCount * 2);
  const view = new DataView(output);
  const sampleRate = safeSampleRate(format.sampleRate);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  // Diagnostic downloads are deliberately mono, matching the Browser SDK's
  // normal TTS output. For a legacy multi-channel trace we retain channel 0.
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let outputOffset = WAV_HEADER_BYTES;
  let pending = new Uint8Array(0);
  for (const part of parts) {
    const incoming = new Uint8Array(part);
    const merged = new Uint8Array(pending.length + incoming.length);
    merged.set(pending);
    merged.set(incoming, pending.length);
    let offset = 0;
    while (offset + frameBytes <= merged.byteLength && outputOffset < output.byteLength) {
      const sample = format.encoding === "pcm_s16le"
        ? new DataView(merged.buffer, merged.byteOffset + offset, 2).getInt16(0, true)
        : floatToInt16(new DataView(merged.buffer, merged.byteOffset + offset, 4).getFloat32(0, true));
      view.setInt16(outputOffset, sample, true);
      outputOffset += 2;
      offset += frameBytes;
    }
    // Keep a copy: `merged` is short-lived and the next frame may arrive after
    // the browser has recycled the WebSocket event buffer.
    pending = merged.subarray(offset).slice();
  }
  return new Blob([output], {type: "audio/wav"});
}

/** Normalize WebSocket binary payloads without losing typed-array offsets. */
export async function asArrayBuffer(value: unknown): Promise<ArrayBuffer | null> {
  if (value instanceof ArrayBuffer) return value.slice(0);
  if (typeof Blob !== "undefined" && value instanceof Blob) return await value.arrayBuffer();
  if (ArrayBuffer.isView(value)) {
    const view = value as ArrayBufferView;
    const copy = new Uint8Array(view.byteLength);
    copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
    return copy.buffer;
  }
  return null;
}

/** Best-effort scheduled-buffer preview for a diagnostic PCM stream. */
export class PcmScheduler {
  private context: AudioContext | null = null;
  private nextTime = 0;
  private pending = new Uint8Array(0);
  private formatKey = "";

  async start(): Promise<void> {
    if (this.context) {
      await this.context.resume();
      return;
    }
    const AudioContextCtor = window.AudioContext;
    if (typeof AudioContextCtor !== "function") throw new Error("AudioContext unavailable");
    this.context = new AudioContextCtor();
    await this.context.resume();
    this.nextTime = this.context.currentTime + 0.04;
  }

  enqueue(bytes: ArrayBuffer, format: PcmFormat): void {
    const context = this.context;
    if (!context) return;
    const channels = safeChannels(format.channels);
    const bytesPerSample = format.encoding === "pcm_s16le" ? 2 : 4;
    const frameBytes = bytesPerSample * channels;
    const key = `${format.encoding}:${safeSampleRate(format.sampleRate)}:${channels}`;
    if (key !== this.formatKey) {
      this.pending = new Uint8Array(0);
      this.formatKey = key;
    }
    const incoming = new Uint8Array(bytes);
    const merged = new Uint8Array(this.pending.length + incoming.length);
    merged.set(this.pending);
    merged.set(incoming, this.pending.length);
    const frames = Math.floor(merged.byteLength / frameBytes);
    const consumedBytes = frames * frameBytes;
    this.pending = merged.subarray(consumedBytes).slice();
    if (frames <= 0) return;

    const buffer = context.createBuffer(channels, frames, safeSampleRate(format.sampleRate));
    const view = new DataView(merged.buffer, merged.byteOffset, consumedBytes);
    for (let channel = 0; channel < channels; channel += 1) {
      const output = buffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        const offset = (frame * channels + channel) * bytesPerSample;
        output[frame] = format.encoding === "pcm_s16le"
          ? view.getInt16(offset, true) / 32768
          : finiteFloat(view.getFloat32(offset, true));
      }
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const start = Math.max(this.nextTime, context.currentTime + 0.02);
    source.start(start);
    this.nextTime = start + frames / safeSampleRate(format.sampleRate);
  }

  close(): void {
    const context = this.context;
    this.context = null;
    this.pending = new Uint8Array(0);
    this.formatKey = "";
    if (context) void context.close().catch(() => undefined);
  }
}

export async function closeAudioContext(
  contextRef: {current: AudioContext | null},
): Promise<void> {
  const context = contextRef.current;
  contextRef.current = null;
  if (context) await context.close().catch(() => undefined);
}

function safeChannels(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.max(1, Math.floor(value)) : 1;
}

function safeSampleRate(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.max(1, Math.round(value)) : DEFAULT_PCM_FORMAT.sampleRate;
}

function finiteFloat(value: number): number {
  return Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
}

function floatToInt16(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-32768, Math.min(32767, Math.round(value * (value < 0 ? 32768 : 32767))));
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}
