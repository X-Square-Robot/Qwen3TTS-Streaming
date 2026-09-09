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
    ])).toMatchObject({completed: 2, failed: 1, averageFirstAudioMs: 20, p90FirstAudioMs: 30});
  });
});
