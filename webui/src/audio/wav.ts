interface PcmFormat {
  encoding: string;
  sample_rate: number;
}

// Convert streamed PCM chunks (pcm_f32 or pcm_s16le) into a single Int16Array.
function pcmChunksToInt16(chunks: ArrayBuffer[], encoding: string): Int16Array {
  if (encoding === "pcm_s16le") {
    // Already 16-bit PCM — concatenate the samples directly.
    const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0) / 2;
    const out = new Int16Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      const part = new Int16Array(chunk);
      out.set(part, offset);
      offset += part.length;
    }
    return out;
  }
  // pcm_f32: clip to [-1, 1] and scale to int16.
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0) / 4;
  const out = new Int16Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    const part = new Float32Array(chunk);
    for (let i = 0; i < part.length; i += 1) {
      const clipped = Math.max(-1, Math.min(1, part[i]));
      out[offset + i] = Math.round(clipped * 32767);
    }
    offset += part.length;
  }
  return out;
}

export function wavBlobFromPcm(chunks: ArrayBuffer[], format: PcmFormat): Blob {
  const pcm16 = pcmChunksToInt16(chunks, format.encoding);
  const sampleRate = format.sample_rate || 24000;

  const dataSize = pcm16.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let cursor = 44;
  for (const sample of pcm16) {
    view.setInt16(cursor, sample, true);
    cursor += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
