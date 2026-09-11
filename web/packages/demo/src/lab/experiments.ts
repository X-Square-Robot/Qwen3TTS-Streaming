import {
  AudioEncoding, discoverCapabilities, InputMode, RealtimeTTSClient, resolveRelativeUrl,
  SynthesisTask, VadStrategy, WavCollector,
  type Capabilities, type IncrementalSynthesisRun, type SynthesisOptions, type SynthesisRun, type TTSEvent,
} from "@xmultimodalinteraction/qwen3tts-browser";
import type {LoadedDemoConfig} from "../config";

export interface ExperimentOptions {
  loaded: LoadedDemoConfig;
  text: string;
  chunkSize: number;
  chunkDelayMs: number;
  speaker?: string;
  language?: string;
  task: SynthesisTask;
  maxAudioBytes?: number;
  capabilities?: Capabilities;
}
export interface RunOutput {
  firstResponseMs: number;
  /** Time from the test/burst origin to the first audio delta. */
  firstAudioMs: number;
  /** Client request start to the first audio delta. */
  clientFirstAudioMs: number;
  /** Server-reported response-create to first raw audio. */
  serverTtftMs: number;
  totalMs: number;
  audioDurationMs?: number;
  audioUrl?: string;
  trace: Array<Record<string, unknown>>;
  error?: string;
  audioTruncated?: boolean;
}
export interface RunUpdate {
  phase: "connecting" | "streaming" | "done" | "failed" | "cancelled";
  firstAudioMs?: number;
  clientFirstAudioMs?: number;
  serverTtftMs?: number;
  totalMs?: number;
  error?: string;
}
export interface ActiveExperiment { readonly cancel: () => void; readonly done: Promise<RunOutput>; }

export interface ExperimentStartBarrier {
  readonly wait: () => Promise<void>;
  readonly ready: (laneId: number) => void;
  readonly fail: (laneId: number, error: unknown) => void;
  readonly cancel: (laneId: number) => void;
}

export class ExperimentConnectionError extends Error {
  readonly laneId: number;
  readonly cause: unknown;
  constructor(laneId: number, cause: unknown) {
    const message = cause instanceof Error ? cause.message : String(cause);
    super(`第 ${laneId + 1} 路连接失败：${message}`);
    this.name = "ExperimentConnectionError";
    this.laneId = laneId;
    this.cause = cause;
  }
}

export class ExperimentStartAbortedError extends Error {
  readonly failedLaneId: number;
  readonly cause: ExperimentConnectionError;
  constructor(failure: ExperimentConnectionError) {
    super(`连接屏障已中止（第 ${failure.laneId + 1} 路失败）：${failure.message}`);
    this.name = "ExperimentStartAbortedError";
    this.failedLaneId = failure.laneId;
    this.cause = failure;
  }
}

/** Coordinates the common response/text start without owning any client. */
export function createExperimentStartBarrier(count: number): ExperimentStartBarrier {
  if (!Number.isInteger(count) || count < 1) throw new RangeError("连接屏障至少需要一路");
  let readyCount = 0;
  const readyLanes = new Set<number>();
  let failure: ExperimentConnectionError | undefined;
  let settled = false;
  let resolveWait: (() => void) | undefined;
  let rejectWait: ((error: unknown) => void) | undefined;
  const waitPromise = new Promise<void>((resolve, reject) => { resolveWait = resolve; rejectWait = reject; });
  return {
    wait: () => waitPromise,
    ready: (laneId) => {
      if (settled) return;
      if (readyLanes.has(laneId)) return;
      readyLanes.add(laneId);
      readyCount += 1;
      if (readyCount === count) { settled = true; resolveWait?.(); }
    },
    fail: (laneId, error) => {
      if (settled) return;
      settled = true;
      failure = error instanceof ExperimentConnectionError ? error : new ExperimentConnectionError(laneId, error);
      rejectWait?.(new ExperimentStartAbortedError(failure));
    },
    cancel: (laneId) => {
      if (settled) return;
      settled = true;
      const cancellation = new ExperimentConnectionError(laneId, new Error("实验已取消"));
      rejectWait?.(new ExperimentStartAbortedError(cancellation));
    },
  };
}

interface ExperimentCoordination { barrier: ExperimentStartBarrier; laneId: number; }

function endpoints(options: ExperimentOptions) { const capabilitiesUrl = resolveRelativeUrl(options.loaded.config.endpoints.capabilities_url, options.loaded.responseUrl); const websocketUrl = new URL(resolveRelativeUrl(options.loaded.config.endpoints.openai_realtime_url, options.loaded.responseUrl)); websocketUrl.protocol = websocketUrl.protocol === "https:" ? "wss:" : "ws:"; return {capabilitiesUrl, websocketUrl}; }
export async function discoverExperimentCapabilities(loaded: LoadedDemoConfig): Promise<Capabilities> { return discoverCapabilities(resolveRelativeUrl(loaded.config.endpoints.capabilities_url, loaded.responseUrl)); }
function delay(ms: number, cancelled: () => boolean): Promise<void> { return new Promise((resolve, reject) => { const timer = setTimeout(resolve, Math.max(0, ms)); const poll = setInterval(() => { if (cancelled()) { clearTimeout(timer); clearInterval(poll); reject(new Error("实验已取消")); } }, 10); setTimeout(() => clearInterval(poll), Math.max(0, ms) + 20); }); }

export function startExperiment(options: ExperimentOptions, mode: "streaming" | "offline", startAt = performance.now(), onUpdate?: (update: RunUpdate) => void, coordination?: ExperimentCoordination): ActiveExperiment {
  let client: RealtimeTTSClient | undefined; let synthesis: SynthesisRun | undefined; let cancelled = false;
  const done = (async (): Promise<RunOutput> => {
    const {capabilitiesUrl, websocketUrl} = endpoints(options);
    client = new RealtimeTTSClient({
      capabilitiesUrl,
      websocketUrl,
      ...(options.capabilities ? {capabilities: options.capabilities} : {}),
    });
    const trace: Array<Record<string, unknown>> = [];
    let firstResponseMs = 0;
    let firstAudioMs = 0;
    let clientFirstAudioMs = 0;
    let serverTtftMs = 0;
    let requestStartedAt = startAt;
    const collectorRef: {current?: WavCollector} = {};
    const at = () => performance.now() - startAt;
    const clientAt = () => performance.now() - requestStartedAt;
    client.onRawEvent((event) => { trace.push({at_ms: Number(at().toFixed(2)), type: event.type, sample_start: event.qwen_output_sample_start, sample_end: event.qwen_output_sample_end, text: event.text}); if (trace.length > 300) trace.shift(); });
    client.onEvent((event: TTSEvent) => {
      if (event.type === "response_started" && !firstResponseMs) firstResponseMs = at();
      if (event.type === "completed") {
        serverTtftMs = event.server?.ttft_ms ?? 0;
      }
      if (event.type === "progress") trace.push({at_ms: Number(at().toFixed(2)), type: event.type, sample_end: event.sample.toString(), meta: event.meta});
      const collector = collectorRef.current;
      if (event.type === "audio" && collector) {
        if (!firstAudioMs) {
          firstAudioMs = at();
          clientFirstAudioMs = clientAt();
          onUpdate?.({phase: "streaming", firstAudioMs, clientFirstAudioMs});
        }
        collector.append(event.pcm);
      }
    });
    onUpdate?.({phase: "connecting"});
    try {
      await client.connect();
    } catch (error) {
      if (cancelled) {
        coordination?.barrier.cancel(coordination.laneId);
        throw new Error("实验已取消");
      }
      if (coordination) coordination.barrier.fail(coordination.laneId, error);
      throw coordination ? new ExperimentConnectionError(coordination.laneId, error) : error;
    }
    if (cancelled) {
      coordination?.barrier.cancel(coordination.laneId);
      throw new Error("实验已取消");
    }
    coordination?.barrier.ready(coordination.laneId);
    await coordination?.barrier.wait();
    if (cancelled) throw new Error("实验已取消");
    const capabilities = client.capabilities;
    if (!capabilities) throw new Error("实例未返回 capabilities");
    const audio = capabilities.audio_formats.find((value) => value.encoding === AudioEncoding.PcmS16Le) ?? capabilities.audio_formats[0];
    if (!audio) throw new Error("实例没有可用音频格式");
    if (!capabilities.tasks.includes(options.task)) throw new Error(`task ${options.task} 不在实例能力中`);
    collectorRef.current = new WavCollector({sampleRate: audio.sample_rate, maxBytes: options.maxAudioBytes ?? 32 * 1024 * 1024});
    const inputMode = mode === "streaming" ? InputMode.Token : InputMode.FullText; const speaker = options.speaker ?? capabilities.speakers?.[0] ?? "Serena"; const language = options.language ?? capabilities.languages?.[0] ?? "auto";
    const request: SynthesisOptions = {task: options.task, speaker, language, inputMode, audio, vad: {enabled: false, strategy: VadStrategy.Disabled}};
    const chunks = options.text.match(new RegExp(`.{1,${Math.max(1, options.chunkSize)}}`, "gu")) ?? [options.text];
    if (mode === "offline") for (let index = 0; index < chunks.length; index += 1) await delay(options.chunkDelayMs, () => cancelled);
    requestStartedAt = performance.now();
    synthesis = mode === "streaming" ? await client.startIncremental(request) : await client.synthesize(options.text, request);
    if (mode === "streaming") { const incremental = synthesis as IncrementalSynthesisRun; for (const chunk of chunks) { if (cancelled) throw new Error("实验已取消"); incremental.append(chunk); await delay(options.chunkDelayMs, () => cancelled); } incremental.commit(); }
    const terminal = await synthesis.done;
    if (terminal.type === "error") throw new Error(`${terminal.code}: ${terminal.message}`);
    if (terminal.type === "cancelled") throw new Error("实验已取消");
    const collector = collectorRef.current;
    if (!collector) throw new Error("音频收集器未初始化");
    const snapshot = collector.snapshot();
    const audioUrl = snapshot.samples > 0 ? URL.createObjectURL(collector.toBlob()) : undefined;
    const output = {
      firstResponseMs,
      firstAudioMs,
      clientFirstAudioMs,
      serverTtftMs,
      totalMs: at(),
      audioDurationMs: snapshot.samples / audio.sample_rate * 1000,
      trace,
      ...(audioUrl ? {audioUrl} : {}),
      ...(snapshot.limitReached ? {audioTruncated: true} : {}),
    };
    onUpdate?.({phase: "done", firstAudioMs, clientFirstAudioMs, serverTtftMs, totalMs: output.totalMs});
    return output;
  })().catch((error): RunOutput => { const message = String(error instanceof Error ? error.message : error); onUpdate?.({phase: cancelled ? "cancelled" : "failed", error: message}); return {firstResponseMs: 0, firstAudioMs: 0, clientFirstAudioMs: 0, serverTtftMs: 0, totalMs: performance.now() - startAt, trace: [], error: message}; }).finally(() => client?.close());
  return {
    done,
    cancel: () => {
      cancelled = true;
      coordination?.barrier.cancel(coordination.laneId);
      try { synthesis?.cancel(); } catch { /* the socket may already be closed */ }
      client?.close();
    },
  };
}

export function startPkExperiment(options: ExperimentOptions, onUpdate?: (lane: "streaming" | "offline", update: RunUpdate) => void): {streaming: ActiveExperiment; offline: ActiveExperiment} {
  const start = performance.now(); return {streaming: startExperiment(options, "streaming", start, (update) => onUpdate?.("streaming", update)), offline: startExperiment(options, "offline", start, (update) => onUpdate?.("offline", update))};
}
