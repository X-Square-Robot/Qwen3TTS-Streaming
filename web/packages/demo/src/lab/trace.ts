import type {LabTraceEvent} from "./types";

export interface TextToken {
  key: string;
  text: string;
  tMs: number;
  segmentIdx: number;
  tokenIdx: number;
  punctLevel: number;
  synthetic: boolean;
}

export interface DecodeStep {
  key: string;
  index: number;
  segmentIdx: number;
  phase: "token" | "pad" | "unknown";
  label: string;
  startMs: number;
  endMs: number;
  token?: TextToken;
}

export interface DecodeTraceModel {
  tokens: TextToken[];
  steps: DecodeStep[];
  stepMs: number;
  audioDurationMs: number;
  chunkCount: number;
  textTokenCount: number;
  padStepCount: number;
  source: "engine_trace" | "synthetic";
}

export interface TextProgressEstimate {
  segmentIdx: number;
  sourceFrameEnd: number;
  textTokenEnd: number;
  textTokenCount: number;
  progress: number;
  basis: string;
  quality: string;
  final: boolean;
}

/** Return the most recent structured text-progress event, if one exists. */
export function latestTextProgress(events: LabTraceEvent[]): TextProgressEstimate | undefined {
  const event = [...events].reverse().find((item) => {
    if (item.type === "text_progress") return true;
    if (item.type !== "segment_end") return false;
    const meta = nestedMeta(item.meta);
    return numberFrom(meta.text_progress ?? item.meta?.text_progress) !== undefined;
  });
  if (!event) return undefined;
  const meta = nestedMeta(event.meta);
  const progress = numberFrom(meta.text_progress ?? event.meta?.text_progress);
  const tokenEnd = numberFrom(meta.text_token_end ?? event.meta?.text_token_end);
  const tokenCount = numberFrom(meta.text_token_count ?? event.meta?.text_token_count);
  if (progress === undefined || tokenEnd === undefined || tokenCount === undefined) return undefined;
  return {
    segmentIdx: segmentIdx(event),
    sourceFrameEnd: numberFrom(meta.source_frame_end ?? event.meta?.source_frame_end) ?? 0,
    textTokenEnd: Math.max(0, Math.round(tokenEnd)),
    textTokenCount: Math.max(0, Math.round(tokenCount)),
    progress: clamp(progress, 0, 1),
    basis: String(meta.progress_basis ?? event.meta?.progress_basis ?? "unknown"),
    quality: String(meta.progress_quality ?? event.meta?.progress_quality ?? "unknown"),
    final: String(meta.progress_final ?? event.meta?.progress_final ?? "false") === "true",
  };
}

/**
 * Build a display model from the trace contract.  Newer engines include
 * decode-step metadata on every audio chunk; older traces are still rendered
 * with a deterministic estimate so the panel remains useful across releases.
 */
export function buildDecodeTrace(events: LabTraceEvent[], fallbackText: string): DecodeTraceModel {
  const tokens = textTokens(events, fallbackText);
  const realTokens = tokens.filter((token) => !token.synthetic);
  const chunks = audioChunkEvents(events);
  const durations = chunks.map(eventAudioDurationMs).filter((value) => value > 0);
  const stepMs = median(durations) ?? 80;
  const directSteps = metadataSteps(chunks, tokens, stepMs);
  if (directSteps.length > 0) return modelFromSteps(tokens, directSteps, chunks, durations, stepMs, "engine_trace");

  const steps: DecodeStep[] = [];
  let cursorMs = 0;
  const segmentEnds = events
    .filter((event) => event.type === "segment_end")
    .sort((a, b) => segmentIdx(a) - segmentIdx(b) || a.t_ms - b.t_ms);
  if (segmentEnds.length > 0) {
    for (const event of segmentEnds) {
      const seg = segmentIdx(event);
      const segmentTokens = realTokens.filter((token) => token.segmentIdx === seg);
      const meta = nestedMeta(event.meta);
      const tokenCount = numberFrom(meta.text_tokens ?? event.meta?.text_tokens) ?? segmentTokens.length;
      const audioSteps = numberFrom(meta.audio_steps ?? event.meta?.audio_steps) ?? Math.max(tokenCount, segmentTokens.length);
      cursorMs = appendSteps(steps, segmentTokens, seg, audioSteps, tokenCount, stepMs, cursorMs);
    }
  } else {
    const count = Math.max(chunks.length, realTokens.length, tokens.length);
    cursorMs = appendSteps(steps, realTokens.length > 0 ? realTokens : tokens, 0, count, realTokens.length || tokens.length, stepMs, cursorMs);
  }
  if (steps.length === 0 && tokens.length > 0) {
    appendSteps(steps, tokens, 0, tokens.length, tokens.length, stepMs, cursorMs);
  }
  return modelFromSteps(tokens, steps, chunks, durations, stepMs, realTokens.length > 0 ? "engine_trace" : "synthetic");
}

export function eventAudioDurationMs(event: LabTraceEvent): number {
  const meta = nestedMeta(event.meta);
  const bytes = numberFrom(meta.bytes ?? event.meta?.bytes) ?? 0;
  const rawFormat = meta.audio_format ?? event.meta?.audio_format;
  if (!bytes || !rawFormat || typeof rawFormat !== "object" || Array.isArray(rawFormat)) return 0;
  const format = rawFormat as Record<string, unknown>;
  const sampleRate = numberFrom(format.sample_rate) ?? 24_000;
  const channels = numberFrom(format.channels) ?? 1;
  const bytesPerSample = String(format.encoding ?? "pcm_f32") === "pcm_s16le" ? 2 : 4;
  return sampleRate > 0 && channels > 0 ? bytes / (sampleRate * channels * bytesPerSample) * 1000 : 0;
}

export function firstEventMs(events: LabTraceEvent[], type: string): number | undefined {
  return events.find((event) => event.type === type)?.t_ms;
}

function modelFromSteps(
  tokens: TextToken[],
  steps: DecodeStep[],
  chunks: LabTraceEvent[],
  durations: number[],
  stepMs: number,
  source: DecodeTraceModel["source"],
): DecodeTraceModel {
  const audioDurationMs = Math.max(
    steps[steps.length - 1]?.endMs ?? 0,
    durations.reduce((sum, value) => sum + value, 0),
    1,
  );
  return {
    tokens,
    steps,
    stepMs,
    audioDurationMs,
    chunkCount: chunks.length,
    textTokenCount: steps.filter((step) => step.phase === "token").length,
    padStepCount: steps.filter((step) => step.phase === "pad").length,
    source,
  };
}

function metadataSteps(chunks: LabTraceEvent[], tokens: TextToken[], fallbackStepMs: number): DecodeStep[] {
  const tokenCursor = new Map<number, number>();
  const steps: DecodeStep[] = [];
  let cursorMs = 0;
  let hasMetadata = false;
  chunks.forEach((event, index) => {
    const meta = nestedMeta(event.meta);
    const phase = phaseFrom(meta.phase ?? event.meta?.phase);
    const decodeStep = numberFrom(meta.decode_step ?? event.meta?.decode_step);
    const tokenIdx = numberFrom(meta.token_idx ?? event.meta?.token_idx);
    const chunkMs = numberFrom(meta.chunk_ms ?? event.meta?.chunk_ms);
    const measured = eventAudioDurationMs(event);
    const duration = Math.max(1, chunkMs ?? (measured > 0 ? measured : fallbackStepMs));
    const seg = segmentIdx(event);
    hasMetadata = hasMetadata || phase !== undefined || decodeStep !== undefined || tokenIdx !== undefined || chunkMs !== undefined;
    let resolved = phase ?? "unknown";
    let token: TextToken | undefined;
    if (resolved === "token" || (resolved === "unknown" && tokenIdx !== undefined)) {
      const idx = tokenIdx ?? tokenCursor.get(seg) ?? 0;
      token = tokens.find((item) => item.segmentIdx === seg && item.tokenIdx === idx);
      tokenCursor.set(seg, idx + 1);
      resolved = "token";
    }
    steps.push({
      key: `chunk-${seg}-${decodeStep ?? index}`,
      index,
      segmentIdx: seg,
      phase: resolved,
      label: token?.text ?? (resolved === "pad" ? "PAD" : `step ${index + 1}`),
      startMs: cursorMs,
      endMs: cursorMs + duration,
      ...(token ? {token} : {}),
    });
    cursorMs += duration;
  });
  return hasMetadata ? steps : [];
}

function appendSteps(
  steps: DecodeStep[],
  segmentTokens: TextToken[],
  segment: number,
  audioSteps: number,
  textTokenCount: number,
  stepMs: number,
  cursorMs: number,
): number {
  const count = Math.max(0, Math.round(audioSteps));
  const tokenCount = Math.max(0, Math.round(textTokenCount));
  for (let index = 0; index < count; index += 1) {
    const token = index < tokenCount ? segmentTokens[index] : undefined;
    const phase: DecodeStep["phase"] = token ? "token" : index >= tokenCount ? "pad" : "unknown";
    steps.push({
      key: `seg-${segment}-step-${index}`,
      index: steps.length,
      segmentIdx: segment,
      phase,
      label: token?.text ?? (phase === "pad" ? "PAD" : `step ${index + 1}`),
      startMs: cursorMs,
      endMs: cursorMs + stepMs,
      ...(token ? {token} : {}),
    });
    cursorMs += stepMs;
  }
  return cursorMs;
}

function textTokens(events: LabTraceEvent[], fallbackText: string): TextToken[] {
  const real = events
    .filter((event) => event.type === "text_token" && Boolean(event.text))
    .map((event, index) => {
      const meta = nestedMeta(event.meta);
      const seg = numberFrom(event.meta?.segment_id ?? event.meta?.segment_idx ?? meta.segment_idx) ?? 0;
      const tokenIdx = numberFrom(meta.token_idx ?? event.meta?.token_idx) ?? index;
      return {
        key: `${event.run_id ?? "run"}-${seg}-${tokenIdx}-${index}`,
        text: event.text ?? "",
        tMs: event.t_ms,
        segmentIdx: seg,
        tokenIdx,
        punctLevel: numberFrom(meta.punct_level ?? event.meta?.punct_level) ?? 0,
        synthetic: false,
      } satisfies TextToken;
    })
    .sort((a, b) => a.segmentIdx - b.segmentIdx || a.tokenIdx - b.tokenIdx || a.tMs - b.tMs);
  if (real.length > 0) return real;
  return Array.from(fallbackText.trim()).map((text, index) => ({
    key: `synthetic-${index}-${text}`,
    text,
    tMs: index * 80,
    segmentIdx: 0,
    tokenIdx: index,
    punctLevel: /[，。！？,.!?]/.test(text) ? 1 : 0,
    synthetic: true,
  }));
}

function audioChunkEvents(events: LabTraceEvent[]): LabTraceEvent[] {
  return events.filter((event) => ["first_audio_chunk", "audio_chunk", "audio"].includes(event.type));
}

function segmentIdx(event: LabTraceEvent): number {
  const meta = nestedMeta(event.meta);
  return numberFrom(event.meta?.segment_id ?? event.meta?.segment_idx ?? meta.segment_idx) ?? 0;
}

function nestedMeta(meta: Record<string, unknown> | undefined): Record<string, unknown> {
  const nested = meta?.meta;
  return nested && typeof nested === "object" && !Array.isArray(nested) ? nested as Record<string, unknown> : meta ?? {};
}

function numberFrom(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function phaseFrom(value: unknown): DecodeStep["phase"] | undefined {
  const phase = String(value ?? "").trim().toLowerCase();
  if (phase === "token" || phase === "text_token") return "token";
  if (phase === "pad" || phase === "flush") return "pad";
  if (phase === "unknown") return "unknown";
  return undefined;
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
