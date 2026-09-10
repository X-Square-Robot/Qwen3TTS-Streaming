import {describe, expect, it} from "vitest";
import {concurrencyStats, MAX_CONCURRENCY, percentile, safeConcurrency} from "./experiment-model";

describe("experiment model", () => {
  it("accepts only positive safe concurrency integers", () => {
    expect(safeConcurrency(256)).toBe(256);
    expect(safeConcurrency(1.5)).toBe(1);
    expect(safeConcurrency(0)).toBe(1);
    expect(safeConcurrency(Number.MAX_SAFE_INTEGER + 1)).toBe(1);
    expect(safeConcurrency(512)).toBe(512);
    expect(safeConcurrency(Number.MAX_SAFE_INTEGER)).toBe(MAX_CONCURRENCY);
  });
  it("calculates percentile and excludes failed lanes", () => {
    expect(percentile([20, 10, 40, 30], .9)).toBe(40);
    expect(concurrencyStats([
      {id: 0, status: "done", firstAudioMs: 10},
      {id: 1, status: "done", firstAudioMs: 30},
      {id: 2, status: "failed"},
      {id: 3, status: "queued"},
      {id: 4, status: "connecting"},
    ])).toMatchObject({
      completed: 2,
      failed: 1,
      queued: 1,
      active: 1,
      started: 4,
      averageClientFirstAudioMs: 20,
      p90ClientFirstAudioMs: 30,
      clientFirstAudioSamples: 2,
    });
  });
  it("uses server TTFT for benchmark statistics when it is available", () => {
    expect(concurrencyStats([
      {id: 0, status: "done", firstAudioMs: 900, clientFirstAudioMs: 900, serverTtftMs: 40},
      {id: 1, status: "done", firstAudioMs: 1100, clientFirstAudioMs: 600, serverTtftMs: 60},
    ])).toMatchObject({
      averageServerTtftMs: 50,
      p90ServerTtftMs: 60,
      serverTtftSamples: 2,
      averageClientFirstAudioMs: 750,
      p90ClientFirstAudioMs: 900,
      clientFirstAudioSamples: 2,
      averageBurstFirstAudioMs: 1000,
      p90BurstFirstAudioMs: 1100,
    });
  });
});
