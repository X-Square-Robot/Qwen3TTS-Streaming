export type LaneStatus = "idle" | "queued" | "connecting" | "streaming" | "done" | "failed" | "cancelled";
export const MAX_CONCURRENCY = 512;

export interface LaneSnapshot {
  id: number;
  status: LaneStatus;
  /** Client-observed time from the burst start to the first audio delta. */
  firstAudioMs?: number;
  /** Client request start to the first audio delta. */
  clientFirstAudioMs?: number;
  /** Server-reported response-create to first raw audio. */
  serverTtftMs?: number;
  totalMs?: number;
  error?: string;
  audioUrl?: string;
}

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
  const clientFirstAudio = completed.flatMap((lane) => {
    const value = lane.clientFirstAudioMs ?? lane.firstAudioMs;
    return value !== undefined && Number.isFinite(value) && value > 0 ? [value] : [];
  });
  const burstFirstAudio = completed.flatMap((lane) => (
    lane.firstAudioMs !== undefined && Number.isFinite(lane.firstAudioMs) && lane.firstAudioMs > 0
      ? [lane.firstAudioMs]
      : []
  ));
  const serverTtft = completed.flatMap((lane) => (
    lane.serverTtftMs !== undefined && Number.isFinite(lane.serverTtftMs) && lane.serverTtftMs > 0
      ? [lane.serverTtftMs]
      : []
  ));
  return {completed: completed.length, failed: lanes.filter((lane) => lane.status === "failed").length,
    queued: lanes.filter((lane) => lane.status === "queued").length,
    active: lanes.filter((lane) => lane.status === "connecting" || lane.status === "streaming").length,
    started: lanes.filter((lane) => !["idle", "queued"].includes(lane.status)).length,
    averageClientFirstAudioMs: clientFirstAudio.length ? clientFirstAudio.reduce((a, b) => a + b, 0) / clientFirstAudio.length : 0,
    p90ClientFirstAudioMs: percentile(clientFirstAudio, .9),
    clientFirstAudioSamples: clientFirstAudio.length,
    averageBurstFirstAudioMs: burstFirstAudio.length ? burstFirstAudio.reduce((a, b) => a + b, 0) / burstFirstAudio.length : 0,
    p90BurstFirstAudioMs: percentile(burstFirstAudio, .9),
    averageServerTtftMs: serverTtft.length ? serverTtft.reduce((a, b) => a + b, 0) / serverTtft.length : 0,
    p90ServerTtftMs: percentile(serverTtft, .9),
    serverTtftSamples: serverTtft.length,
  };
}
