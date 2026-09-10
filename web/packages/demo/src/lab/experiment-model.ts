export type LaneStatus = "idle" | "queued" | "connecting" | "streaming" | "done" | "failed" | "cancelled";
export const MAX_CONCURRENCY = 512;
/** Maximum number of browser WebSockets admitted at once by the demo. */
export const BROWSER_ACTIVE_CONCURRENCY = 16;

export interface LaneSnapshot { id: number; status: LaneStatus; firstAudioMs?: number; totalMs?: number; error?: string; audioUrl?: string; }

export function safeConcurrency(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) return 1;
  return Math.min(value, MAX_CONCURRENCY);
}

export function percentile(values: readonly number[], p: number): number {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1)] ?? 0;
}

export function concurrencyStats(lanes: readonly LaneSnapshot[]) {
  const completed = lanes.filter((lane) => lane.status === "done");
  const firstAudio = completed.flatMap((lane) => lane.firstAudioMs === undefined ? [] : [lane.firstAudioMs]);
  return {completed: completed.length, failed: lanes.filter((lane) => lane.status === "failed").length,
    queued: lanes.filter((lane) => lane.status === "queued").length,
    active: lanes.filter((lane) => lane.status === "connecting" || lane.status === "streaming").length,
    averageFirstAudioMs: firstAudio.length ? firstAudio.reduce((a, b) => a + b, 0) / firstAudio.length : 0,
    p90FirstAudioMs: percentile(firstAudio, .9)};
}
