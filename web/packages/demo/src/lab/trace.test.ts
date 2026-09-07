import {describe, expect, it} from "vitest";

import {buildDecodeTrace, eventAudioDurationMs, latestTextProgress} from "./trace";
import type {LabTraceEvent} from "./types";

describe("Lab trace model", () => {
  it("reads nested decode metadata and separates token and PAD steps", () => {
    const events: LabTraceEvent[] = [
      {type: "text_token", t_ms: 1, text: "你", meta: {segment_id: 0, token_idx: 0}},
      {type: "first_audio_chunk", t_ms: 4, meta: {
        meta: {phase: "token", token_idx: 0, chunk_ms: 20, bytes: 1920,
          audio_format: {encoding: "pcm_f32", sample_rate: 24_000, channels: 1}},
      }},
      {type: "audio_chunk", t_ms: 24, meta: {
        phase: "pad", decode_step: 1, chunk_ms: 20, bytes: 1920,
        audio_format: {encoding: "pcm_f32", sample_rate: 24_000, channels: 1},
      }},
    ];
    const model = buildDecodeTrace(events, "你");
    expect(model.source).toBe("engine_trace");
    expect(model.steps.map((step) => step.phase)).toEqual(["token", "pad"]);
    expect(model.steps[0]?.token?.text).toBe("你");
    expect(model.padStepCount).toBe(1);
    expect(eventAudioDurationMs(events[1]!)).toBe(20);
  });

  it("provides deterministic synthetic steps when a trace has no text tokens", () => {
    const model = buildDecodeTrace([], "好！");
    expect(model.source).toBe("synthetic");
    expect(model.tokens.map((token) => token.text)).toEqual(["好", "！"]);
    expect(model.steps).toHaveLength(2);
    expect(model.steps.every((step) => step.token?.synthetic)).toBe(true);
  });

  it("normalizes structured progress events", () => {
    const progress = latestTextProgress([{
      type: "text_progress", t_ms: 12, meta: {
        meta: {text_progress: 0.5, text_token_end: 3, text_token_count: 6,
          progress_basis: "source_frames", progress_quality: "exact", progress_final: true},
      },
    }]);
    expect(progress).toMatchObject({progress: 0.5, textTokenEnd: 3, textTokenCount: 6, final: true});
  });
});
