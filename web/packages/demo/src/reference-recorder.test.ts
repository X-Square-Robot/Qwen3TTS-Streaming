import {afterEach, describe, expect, it} from "vitest";

import {startReferenceRecorder} from "./reference-recorder";

const originalAudioContext = globalThis.AudioContext;
const originalNavigator = globalThis.navigator;

afterEach(() => {
  Object.defineProperty(globalThis, "AudioContext", {configurable: true, value: originalAudioContext});
  Object.defineProperty(globalThis, "navigator", {configurable: true, value: originalNavigator});
});

describe("reference WAV recorder", () => {
  it("captures microphone float PCM as an engine-compatible RIFF/WAV file", async () => {
    let processor: {onaudioprocess: ((event: AudioProcessingEvent) => void) | null} | null = null;
    let trackStopped = false;
    const connectable = {connect() { return connectable; }, disconnect() {}};
    class FakeAudioContext {
      sampleRate = 48_000;
      destination = {};
      createMediaStreamSource() { return connectable; }
      createScriptProcessor() {
        processor = {...connectable, onaudioprocess: null};
        return processor;
      }
      createGain() { return {...connectable, gain: {value: 1}}; }
      async resume() {}
      async close() {}
    }
    Object.defineProperty(globalThis, "AudioContext", {configurable: true, value: FakeAudioContext});
    Object.defineProperty(globalThis, "navigator", {configurable: true, value: {
      mediaDevices: {getUserMedia: async () => ({getTracks: () => [{stop: () => { trackStopped = true; }}]})},
    }});

    const recorder = await startReferenceRecorder(4096, () => undefined);
    const active = processor as {onaudioprocess: ((event: AudioProcessingEvent) => void) | null} | null;
    active?.onaudioprocess?.({
      inputBuffer: {getChannelData: () => Float32Array.of(-1, 0, 1)},
    } as unknown as AudioProcessingEvent);
    const file = await recorder.stop();
    const bytes = new Uint8Array(await file.arrayBuffer());
    expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe("RIFF");
    expect(new TextDecoder().decode(bytes.subarray(8, 12))).toBe("WAVE");
    expect(new DataView(bytes.buffer).getUint32(24, true)).toBe(48_000);
    expect(file.type).toBe("audio/wav");
    expect(trackStopped).toBe(true);
  });
});
