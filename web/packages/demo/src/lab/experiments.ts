import {
  AudioEncoding, discoverCapabilities, InputMode, RealtimeTTSClient, resolveRelativeUrl,
  SynthesisTask, VadStrategy, WavCollector,
  type Capabilities, type IncrementalSynthesisRun, type SynthesisOptions, type SynthesisRun, type TTSEvent,
} from "@xmultimodalinteraction/qwen3tts-browser";
import type {LoadedDemoConfig} from "../config";

export interface ExperimentOptions { loaded: LoadedDemoConfig; text: string; chunkSize: number; chunkDelayMs: number; speaker?: string; language?: string; task: SynthesisTask; maxAudioBytes?: number; }
export interface RunOutput { firstResponseMs: number; firstAudioMs: number; totalMs: number; audioUrl?: string; trace: Array<Record<string, unknown>>; error?: string; audioTruncated?: boolean; }
export interface RunUpdate { phase: "connecting" | "streaming" | "done" | "failed" | "cancelled"; firstAudioMs?: number; totalMs?: number; error?: string; }
export interface ActiveExperiment { readonly cancel: () => void; readonly done: Promise<RunOutput>; }

function endpoints(options: ExperimentOptions) { const capabilitiesUrl = resolveRelativeUrl(options.loaded.config.endpoints.capabilities_url, options.loaded.responseUrl); const websocketUrl = new URL(resolveRelativeUrl(options.loaded.config.endpoints.openai_realtime_url, options.loaded.responseUrl)); websocketUrl.protocol = websocketUrl.protocol === "https:" ? "wss:" : "ws:"; return {capabilitiesUrl, websocketUrl}; }
export async function discoverExperimentCapabilities(loaded: LoadedDemoConfig): Promise<Capabilities> { return discoverCapabilities(resolveRelativeUrl(loaded.config.endpoints.capabilities_url, loaded.responseUrl)); }
function delay(ms: number, cancelled: () => boolean): Promise<void> { return new Promise((resolve, reject) => { const timer = setTimeout(resolve, Math.max(0, ms)); const poll = setInterval(() => { if (cancelled()) { clearTimeout(timer); clearInterval(poll); reject(new Error("实验已取消")); } }, 10); setTimeout(() => clearInterval(poll), Math.max(0, ms) + 20); }); }

export function startExperiment(options: ExperimentOptions, mode: "streaming" | "offline", startAt = performance.now(), onUpdate?: (update: RunUpdate) => void): ActiveExperiment {
  let client: RealtimeTTSClient | undefined; let synthesis: SynthesisRun | undefined; let cancelled = false;
  const done = (async (): Promise<RunOutput> => {
    const {capabilitiesUrl, websocketUrl} = endpoints(options); const capabilities = await discoverCapabilities(capabilitiesUrl);
    const audio = capabilities.audio_formats.find((value) => value.encoding === AudioEncoding.PcmS16Le) ?? capabilities.audio_formats[0]; if (!audio) throw new Error("实例没有可用音频格式");
    if (!capabilities.tasks.includes(options.task)) throw new Error(`task ${options.task} 不在实例能力中`);
    client = new RealtimeTTSClient({capabilitiesUrl, websocketUrl}); const collector = new WavCollector({sampleRate: audio.sample_rate, maxBytes: options.maxAudioBytes ?? 32 * 1024 * 1024});
    const trace: Array<Record<string, unknown>> = []; let firstResponseMs = 0; let firstAudioMs = 0; const at = () => performance.now() - startAt;
    client.onRawEvent((event) => { trace.push({at_ms: Number(at().toFixed(2)), type: event.type, sample_start: event.qwen_output_sample_start, sample_end: event.qwen_output_sample_end, text: event.text}); if (trace.length > 300) trace.shift(); });
    client.onEvent((event: TTSEvent) => { if (event.type === "response_started" && !firstResponseMs) firstResponseMs = at(); if (event.type === "progress") trace.push({at_ms: Number(at().toFixed(2)), type: event.type, sample_end: event.sample.toString(), meta: event.meta}); if (event.type === "audio") { if (!firstAudioMs) { firstAudioMs = at(); onUpdate?.({phase: "streaming", firstAudioMs}); } collector.append(event.pcm); } });
    onUpdate?.({phase: "connecting"}); await client.connect(); if (cancelled) throw new Error("实验已取消");
    const inputMode = mode === "streaming" ? InputMode.Token : InputMode.FullText; const speaker = options.speaker ?? capabilities.speakers?.[0] ?? "Serena"; const language = options.language ?? capabilities.languages?.[0] ?? "auto";
    const request: SynthesisOptions = {task: options.task, speaker, language, inputMode, audio, vad: {enabled: false, strategy: VadStrategy.Disabled}};
    const chunks = options.text.match(new RegExp(`.{1,${Math.max(1, options.chunkSize)}}`, "gu")) ?? [options.text];
    if (mode === "offline") for (let index = 0; index < chunks.length; index += 1) await delay(options.chunkDelayMs, () => cancelled);
    synthesis = mode === "streaming" ? await client.startIncremental(request) : await client.synthesize(options.text, request);
    if (mode === "streaming") { const incremental = synthesis as IncrementalSynthesisRun; for (const chunk of chunks) { if (cancelled) throw new Error("实验已取消"); incremental.append(chunk); await delay(options.chunkDelayMs, () => cancelled); } incremental.commit(); }
    await synthesis.done;
    const snapshot = collector.snapshot();
    const audioUrl = snapshot.samples > 0 ? URL.createObjectURL(collector.toBlob()) : undefined;
    const output = {
      firstResponseMs,
      firstAudioMs,
      totalMs: at(),
      trace,
      ...(audioUrl ? {audioUrl} : {}),
      ...(snapshot.limitReached ? {audioTruncated: true} : {}),
    };
    onUpdate?.({phase: "done", firstAudioMs, totalMs: output.totalMs});
    return output;
  })().catch((error): RunOutput => { const message = String(error instanceof Error ? error.message : error); onUpdate?.({phase: cancelled ? "cancelled" : "failed", error: message}); return {firstResponseMs: 0, firstAudioMs: 0, totalMs: performance.now() - startAt, trace: [], error: message}; }).finally(() => client?.close());
  return {done, cancel: () => { cancelled = true; synthesis?.cancel(); client?.close(); }};
}

export function startPkExperiment(options: ExperimentOptions, onUpdate?: (lane: "streaming" | "offline", update: RunUpdate) => void): {streaming: ActiveExperiment; offline: ActiveExperiment} {
  const start = performance.now(); return {streaming: startExperiment(options, "streaming", start, (update) => onUpdate?.("streaming", update)), offline: startExperiment(options, "offline", start, (update) => onUpdate?.("offline", update))};
}
