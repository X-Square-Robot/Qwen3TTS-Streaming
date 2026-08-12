import type { TraceEvent } from "./types";

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

export function latestTextProgress(events: TraceEvent[]): TextProgressEstimate | undefined {
  const event = [...events].reverse().find((item) => {
    if (item.type === "text_progress") {
      return true;
    }
    if (item.type !== "segment_end") {
      return false;
    }
    const meta = nestedMeta(item.meta);
    return numberFrom(meta.text_progress ?? item.meta?.text_progress) !== undefined;
  });
  if (!event) {
    return undefined;
  }
  const meta = nestedMeta(event.meta);
  const progress = numberFrom(meta.text_progress ?? event.meta?.text_progress);
  const tokenEnd = numberFrom(meta.text_token_end ?? event.meta?.text_token_end);
  const tokenCount = numberFrom(meta.text_token_count ?? event.meta?.text_token_count);
  if (progress === undefined || tokenEnd === undefined || tokenCount === undefined) {
    return undefined;
  }
  return {
    segmentIdx: segmentIdx(event),
    sourceFrameEnd: numberFrom(meta.source_frame_end ?? event.meta?.source_frame_end) ?? 0,
    textTokenEnd: Math.max(0, Math.round(tokenEnd)),
    textTokenCount: Math.max(0, Math.round(tokenCount)),
    progress: Math.max(0, Math.min(1, progress)),
    basis: String(meta.progress_basis ?? event.meta?.progress_basis ?? "unknown"),
    quality: String(meta.progress_quality ?? event.meta?.progress_quality ?? "unknown"),
    final: String(meta.progress_final ?? event.meta?.progress_final ?? "false") === "true",
  };
}

export function buildDecodeTrace(events: TraceEvent[], fallbackText: string): DecodeTraceModel {
  const tokens = textTokens(events, fallbackText);
  const realTokens = tokens.filter((token) => !token.synthetic);
  const chunks = audioChunkEvents(events);
  const audioDurations = chunks.map(eventAudioDurationMs).filter((value) => value > 0);
  const stepMs = median(audioDurations) ?? 80;
  const chunkCount = chunks.length;
  const directSteps = audioChunkMetadataSteps(chunks, realTokens, stepMs);
  if (directSteps.length > 0) {
    const audioDurationMs = Math.max(
      directSteps[directSteps.length - 1]?.endMs ?? 0,
      audioDurations.reduce((sum, value) => sum + value, 0),
      1
    );
    return {
      tokens,
      steps: directSteps,
      stepMs,
      audioDurationMs,
      chunkCount,
      textTokenCount: directSteps.filter((step) => step.phase === "token").length,
      padStepCount: directSteps.filter((step) => step.phase === "pad").length,
      source: "engine_trace",
    };
  }
  const segmentEnds = events
    .filter((event) => event.type === "segment_end")
    .sort((a, b) => segmentIdx(a) - segmentIdx(b) || a.t_ms - b.t_ms);

  const steps: DecodeStep[] = [];
  let cursorMs = 0;

  if (segmentEnds.length > 0) {
    for (const event of segmentEnds) {
      const segIdx = segmentIdx(event);
      const segmentTokens = realTokens.filter((token) => token.segmentIdx === segIdx);
      const meta = nestedMeta(event.meta);
      const textTokenCount = numberFrom(meta.text_tokens ?? event.meta?.text_tokens) ?? segmentTokens.length;
      const audioSteps = numberFrom(meta.audio_steps ?? event.meta?.audio_steps) ?? Math.max(textTokenCount, segmentTokens.length);
      appendSegmentSteps(steps, segmentTokens, segIdx, audioSteps, textTokenCount, stepMs, () => cursorMs, (value) => {
        cursorMs = value;
      });
    }
  } else {
    const audioSteps = Math.max(chunkCount, realTokens.length, tokens.length);
    appendSegmentSteps(steps, realTokens.length > 0 ? realTokens : tokens, 0, audioSteps, realTokens.length || tokens.length, stepMs, () => cursorMs, (value) => {
      cursorMs = value;
    });
  }

  if (steps.length === 0 && tokens.length > 0) {
    appendSegmentSteps(steps, tokens, 0, tokens.length, tokens.length, stepMs, () => cursorMs, (value) => {
      cursorMs = value;
    });
  }

  const audioDurationMs = Math.max(
    steps.length * stepMs,
    audioDurations.reduce((sum, value) => sum + value, 0),
    1
  );
  return {
    tokens,
    steps,
    stepMs,
    audioDurationMs,
    chunkCount,
    textTokenCount: steps.filter((step) => step.phase === "token").length,
    padStepCount: steps.filter((step) => step.phase === "pad").length,
    source: realTokens.length > 0 ? "engine_trace" : "synthetic",
  };
}

function audioChunkMetadataSteps(chunks: TraceEvent[], tokens: TextToken[], fallbackStepMs: number): DecodeStep[] {
  const tokenCursorBySegment = new Map<number, number>();
  const steps: DecodeStep[] = [];
  let cursorMs = 0;
  let sawStepMetadata = false;

  chunks.forEach((event, index) => {
    const meta = nestedMeta(event.meta);
    const phase = phaseFrom(meta.phase ?? event.meta?.phase);
    const decodeStep = numberFrom(meta.decode_step ?? event.meta?.decode_step);
    const tokenIdx = numberFrom(meta.token_idx ?? event.meta?.token_idx);
    const chunkMs = numberFrom(meta.chunk_ms ?? event.meta?.chunk_ms);
    const measuredMs = eventAudioDurationMs(event);
    const durationMs = Math.max(1, chunkMs ?? (measuredMs > 0 ? measuredMs : fallbackStepMs));
    const segIdx = segmentIdx(event);
    sawStepMetadata = sawStepMetadata || phase !== undefined || decodeStep !== undefined || tokenIdx !== undefined || chunkMs !== undefined;

    let token: TextToken | undefined;
    let resolvedPhase: DecodeStep["phase"] = phase ?? "unknown";
    if (resolvedPhase === "token" || (resolvedPhase === "unknown" && tokenIdx !== undefined)) {
      const idx = tokenIdx ?? tokenCursorBySegment.get(segIdx) ?? 0;
      token = tokens.find((item) => item.segmentIdx === segIdx && item.tokenIdx === idx);
      tokenCursorBySegment.set(segIdx, idx + 1);
      resolvedPhase = "token";
    }
    if (resolvedPhase === "pad") {
      tokenCursorBySegment.set(segIdx, tokenCursorBySegment.get(segIdx) ?? 0);
    }

    steps.push({
      key: `chunk-${segIdx}-${decodeStep ?? index}`,
      index,
      segmentIdx: segIdx,
      phase: resolvedPhase,
      label: token?.text ?? (resolvedPhase === "pad" ? "PAD" : `step ${index + 1}`),
      startMs: cursorMs,
      endMs: cursorMs + durationMs,
      token,
    });
    cursorMs += durationMs;
  });

  return sawStepMetadata ? steps : [];
}

export function eventAudioDurationMs(event: TraceEvent): number {
  const bytes = Number(event.meta?.bytes ?? 0);
  const format = event.meta?.audio_format;
  if (!bytes || !format || typeof format !== "object" || Array.isArray(format)) {
    return 0;
  }
  const record = format as Record<string, unknown>;
  const sampleRate = Number(record.sample_rate ?? 24000);
  const channels = Number(record.channels ?? 1);
  const encoding = String(record.encoding ?? "pcm_f32");
  const bytesPerSample = encoding === "pcm_s16le" ? 2 : 4;
  if (!sampleRate || !channels || !bytesPerSample) {
    return 0;
  }
  return (bytes / (sampleRate * channels * bytesPerSample)) * 1000;
}

export function firstEventMs(events: TraceEvent[], type: string): number | undefined {
  return events.find((event) => event.type === type)?.t_ms;
}

export function metricNumber(value: unknown): number | undefined {
  return numberFrom(value);
}

function appendSegmentSteps(
  steps: DecodeStep[],
  segmentTokens: TextToken[],
  segmentIdx: number,
  audioSteps: number,
  textTokenCount: number,
  stepMs: number,
  getCursor: () => number,
  setCursor: (value: number) => void
) {
  const safeAudioSteps = Math.max(0, Math.round(audioSteps));
  const safeTextTokens = Math.max(0, Math.round(textTokenCount));
  for (let localStep = 0; localStep < safeAudioSteps; localStep += 1) {
    const token = localStep < safeTextTokens ? segmentTokens[localStep] : undefined;
    const phase = token ? "token" : localStep >= safeTextTokens ? "pad" : "unknown";
    const startMs = getCursor();
    const endMs = startMs + stepMs;
    steps.push({
      key: `seg-${segmentIdx}-step-${localStep}`,
      index: steps.length,
      segmentIdx,
      phase,
      label: token?.text ?? (phase === "pad" ? "PAD" : `step ${localStep + 1}`),
      startMs,
      endMs,
      token,
    });
    setCursor(endMs);
  }
}

function textTokens(events: TraceEvent[], fallbackText: string): TextToken[] {
  const real = events
    .filter((event) => event.type === "text_token" && event.text)
    .map((event, index) => {
      const meta = nestedMeta(event.meta);
      const segIdx = numberFrom(event.meta?.segment_id ?? event.meta?.segment_idx ?? meta.segment_idx) ?? 0;
      const tokenIdx = numberFrom(meta.token_idx ?? event.meta?.token_idx) ?? index;
      return {
        key: `${event.run_id}-${segIdx}-${tokenIdx}-${index}`,
        text: event.text ?? "",
        tMs: event.t_ms,
        segmentIdx: segIdx,
        tokenIdx,
        punctLevel: numberFrom(meta.punct_level ?? event.meta?.punct_level) ?? 0,
        synthetic: false,
      };
    })
    .sort((a, b) => a.segmentIdx - b.segmentIdx || a.tokenIdx - b.tokenIdx || a.tMs - b.tMs);
  if (real.length > 0) {
    return real;
  }

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

function audioChunkEvents(events: TraceEvent[]): TraceEvent[] {
  return events.filter((event) => event.type === "first_audio_chunk" || event.type === "audio_chunk");
}

function segmentIdx(event: TraceEvent): number {
  const meta = nestedMeta(event.meta);
  return numberFrom(event.meta?.segment_id ?? event.meta?.segment_idx ?? meta.segment_idx) ?? 0;
}

function nestedMeta(meta: Record<string, unknown> | undefined): Record<string, unknown> {
  const nested = meta?.meta;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }
  return meta ?? {};
}

function numberFrom(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function phaseFrom(value: unknown): DecodeStep["phase"] | undefined {
  const phase = String(value ?? "").trim().toLowerCase();
  if (phase === "token" || phase === "text_token") {
    return "token";
  }
  if (phase === "pad" || phase === "flush") {
    return "pad";
  }
  if (phase === "unknown") {
    return "unknown";
  }
  return undefined;
}

function median(values: number[]): number | undefined {
  if (values.length === 0) {
    return undefined;
  }
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
