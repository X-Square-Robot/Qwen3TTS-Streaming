import {Download, Play, RadioTower, Square} from "lucide-react";
import {useCallback, useEffect, useRef, useState} from "react";

import {WavCollector} from "@xmultimodalinteraction/qwen3tts-browser";

import type {LoadedDemoConfig} from "../config";
import type {LabApi} from "./api";
import {createLabApi} from "./api";
import {ConcurrencyPanel} from "./components/ConcurrencyPanel";
import {PerformanceRace} from "./components/PerformanceRace";
import {TextPlayer} from "./components/TextPlayer";
import type {LabRequest, LabRunResult, LabTraceEvent} from "./types";

/**
 * Props for the feature showcase embedded in the unified Demo portal.
 *
 * `api` is deliberately nullable: the product portal must remain useful when
 * the optional engineering backend is not deployed.  In that case the
 * component renders an explanatory card instead of attempting cross-origin
 * requests or inventing fixture data.
 */
export type FeatureShowcaseProps =
  | {
      /** Preferred integration used by the current ExperimentLab wrapper. */
      readonly loaded: LoadedDemoConfig;
      /** Whether the optional backend health probe succeeded. */
      readonly reachable: boolean;
      readonly api?: never;
      readonly request?: never;
    }
  | {
      /** Direct composition form, useful for tests and other portal hosts. */
      readonly api: LabApi | null;
      readonly request: LabRequest;
      readonly loaded?: never;
      readonly reachable?: never;
    };

/**
 * Reuses the former feature panels from the single `/demo/#/lab` entry.
 *
 * The optional backend remains the source of detailed engine traces and lane
 * metrics.  Live trace capture and playback are kept in this composition
 * layer so the migrated panels stay presentational and only depend on the
 * narrow `LabApi` contract.
 */
export function FeatureShowcase(props: FeatureShowcaseProps) {
  const {api, request} = "loaded" in props
    ? {
        api: props.reachable ? safeCreateLabApi(props.loaded) : null,
        request: DEFAULT_LAB_REQUEST,
      }
    : props;
  if (!api) {
    return (
      <section className="panel lab-feature-unavailable" data-testid="feature-showcase-unavailable">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">OPTIONAL ENGINEERING LAB</p>
            <h2>特性实验暂不可用</h2>
          </div>
          <RadioTower size={18} aria-hidden="true" />
        </div>
        <p className="hint">
          当前实例没有配置可访问的 <code>demo_api</code>。基础体验和公共 Realtime
          实验仍可使用；启用 <code>DEMO_LAB_URL</code> 后可在这里查看详细 trace、
          LLM PK 和多路并发结果。
        </p>
      </section>
    );
  }

  return <FeatureShowcaseConnected api={api} request={request} />;
}

const DEFAULT_LAB_REQUEST: LabRequest = {
  text: "你好，这是千问3 TTS 多路合成验证。",
  speaker: "Serena",
  language: "auto",
  ms_per_token: 30,
};

function safeCreateLabApi(loaded: LoadedDemoConfig): LabApi | null {
  try {
    return createLabApi(loaded);
  } catch {
    return null;
  }
}

function FeatureShowcaseConnected({api, request}: {api: LabApi; request: LabRequest}) {
  const [trace, setTrace] = useState<LabTraceEvent[]>([]);
  const [traceText, setTraceText] = useState(request.text);
  const [traceAudioUrl, setTraceAudioUrl] = useState("");
  const [traceLiveMs, setTraceLiveMs] = useState(0);
  const [traceLive, setTraceLive] = useState(false);
  const [traceError, setTraceError] = useState("");
  const [raceError, setRaceError] = useState("");
  const traceSocketRef = useRef<WebSocket | null>(null);
  const traceGenerationRef = useRef(0);
  const traceStartedAtRef = useRef(0);
  const tracePartsRef = useRef<ArrayBuffer[]>([]);
  const traceBytesRef = useRef(0);
  const traceFormatRef = useRef<PcmFormat>(DEFAULT_PCM_FORMAT);
  const traceCollectorRef = useRef<WavCollector | null>(null);
  const traceAudioUrlRef = useRef("");
  const livePlayerRef = useRef<PcmScheduler | null>(null);
  const raceContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    setTraceText(request.text);
  }, [request.text]);

  const revokeTraceAudio = useCallback(() => {
    const current = traceAudioUrlRef.current;
    if (current) {
      URL.revokeObjectURL(current);
      traceAudioUrlRef.current = "";
    }
    setTraceAudioUrl("");
  }, []);

  const stopLiveTrace = useCallback(() => {
    traceGenerationRef.current += 1;
    traceSocketRef.current?.close(1000, "trace stopped");
    traceSocketRef.current = null;
    livePlayerRef.current?.close();
    livePlayerRef.current = null;
    traceCollectorRef.current?.stop();
    traceCollectorRef.current = null;
    setTraceLive(false);
  }, []);

  useEffect(() => {
    return () => {
      stopLiveTrace();
      revokeTraceAudio();
      void closeAudioContext(raceContextRef);
    };
  }, [revokeTraceAudio, stopLiveTrace]);

  async function startLiveTrace(): Promise<void> {
    stopLiveTrace();
    revokeTraceAudio();
    setTrace([]);
    setTraceError("");
    setTraceLiveMs(0);
    setTraceLive(true);
    const generation = traceGenerationRef.current;
    traceStartedAtRef.current = 0;
    tracePartsRef.current = [];
    traceBytesRef.current = 0;
    traceFormatRef.current = {...DEFAULT_PCM_FORMAT};

    // Keep the capture bounded.  The collector emits a valid WAV even when it
    // reaches the cap; live playback continues independently through the
    // scheduler below.
    traceCollectorRef.current = new WavCollector({sampleRate: DEFAULT_PCM_FORMAT.sampleRate});
    let socket: WebSocket;
    try {
      socket = api.openTrtLive();
    } catch (cause) {
      setTraceError(`无法打开实时 trace：${String(cause)}`);
      setTraceLive(false);
      return;
    }
    socket.binaryType = "arraybuffer";
    traceSocketRef.current = socket;

    socket.onopen = () => {
      if (generation !== traceGenerationRef.current) {
        socket.close();
        return;
      }
      traceStartedAtRef.current = performance.now();
      socket.send(JSON.stringify({
        type: "speak",
        text: traceText.trim() || request.text,
        speaker: request.speaker,
        language: request.language,
      }));
    };
    socket.onmessage = (message) => {
      if (generation !== traceGenerationRef.current) return;
      if (typeof message.data === "string") {
        handleTraceMessage(message.data, generation, socket);
        return;
      }
      void handleTraceAudio(message.data, generation);
    };
    socket.onerror = () => {
      if (generation !== traceGenerationRef.current) return;
      setTraceError("实时 trace WebSocket 连接失败");
      setTraceLive(false);
    };
    socket.onclose = () => {
      if (generation !== traceGenerationRef.current) return;
      traceSocketRef.current = null;
      setTraceLive(false);
    };

    try {
      livePlayerRef.current = new PcmScheduler();
      await livePlayerRef.current.start();
    } catch {
      // Browsers can reject AudioContext startup (autoplay policy, an older
      // WebView, or a test double).  Keep capturing the real bytes and let the
      // completed WAV remain available.
      livePlayerRef.current?.close();
      livePlayerRef.current = null;
    }
  }

  function handleTraceMessage(raw: string, generation: number, socket: WebSocket): void {
    let message: unknown;
    try {
      message = JSON.parse(raw);
    } catch {
      setTraceError("实时 trace 返回了无法解析的 JSON");
      return;
    }
    if (!isRecord(message)) return;
    if (message.type === "error") {
      setTraceError(String(message.message ?? "实时 trace 失败"));
      setTraceLive(false);
      socket.close();
      return;
    }
    if (message.type !== "event" || !isRecord(message.event)) return;
    const event = normalizeTraceEvent(message.event);
    if (!event || generation !== traceGenerationRef.current) return;
    const nextMeta = event.meta;
    const eventFormat = nextMeta?.audio_format;
    if (isRecord(eventFormat)) {
      traceFormatRef.current = normalizePcmFormat(eventFormat);
      if (!traceCollectorRef.current && traceFormatRef.current.encoding === "pcm_s16le") {
        traceCollectorRef.current = new WavCollector({sampleRate: traceFormatRef.current.sampleRate});
      }
    }
    setTrace((current) => appendBounded(current, event, MAX_TRACE_EVENTS));
    const elapsed = traceStartedAtRef.current > 0
      ? performance.now() - traceStartedAtRef.current
      : event.t_ms;
    setTraceLiveMs(Math.max(0, event.t_ms || elapsed));
    if (event.type === "done") {
      finishTraceCapture();
      setTraceLive(false);
      socket.close(1000, "trace complete");
    }
  }

  async function handleTraceAudio(data: unknown, generation: number): Promise<void> {
    const bytes = await asArrayBuffer(data);
    if (!bytes || generation !== traceGenerationRef.current) return;
    const format = traceFormatRef.current;
    const accepted = appendBoundedAudio(bytes, tracePartsRef, traceBytesRef, MAX_CAPTURE_BYTES);
    if (accepted.byteLength === 0) return;
    const samples = pcmToInt16(accepted, format.encoding);
    traceCollectorRef.current?.append(samples);
    try {
      livePlayerRef.current?.enqueue(accepted, format);
    } catch {
      // Capturing and the eventual WAV download must not fail just because a
      // browser cannot schedule one malformed/unsupported live chunk.
    }
    const elapsed = traceStartedAtRef.current > 0 ? performance.now() - traceStartedAtRef.current : 0;
    setTraceLiveMs((current) => Math.max(current, elapsed));
  }

  function finishTraceCapture(): void {
    const collector = traceCollectorRef.current;
    if (collector && collector.snapshot().samples > 0) {
      const url = URL.createObjectURL(collector.toBlob());
      traceAudioUrlRef.current = url;
      setTraceAudioUrl(url);
    } else if (tracePartsRef.current.length > 0) {
      const blob = wavBlobFromPcm(tracePartsRef.current, traceFormatRef.current);
      if (blob.size > WAV_HEADER_BYTES) {
        const url = URL.createObjectURL(blob);
        traceAudioUrlRef.current = url;
        setTraceAudioUrl(url);
      }
    }
    traceCollectorRef.current?.stop();
    traceCollectorRef.current = null;
  }

  async function playRaceRows(rows: LabRunResult[], aligned: boolean): Promise<void> {
    setRaceError("");
    await closeAudioContext(raceContextRef);
    const playable = rows.filter((row) => Boolean(row.audio?.url));
    if (playable.length === 0) return;
    try {
      const context = new AudioContext();
      raceContextRef.current = context;
      await context.resume();
      const decoded = await Promise.all(playable.map(async (row) => {
        const response = await fetch(api.audioUrl(row.audio!.url));
        if (!response.ok) throw new Error(`${row.label}: audio ${response.status}`);
        const buffer = await response.arrayBuffer();
        return {row, audio: await context.decodeAudioData(buffer.slice(0))};
      }));
      const baseTime = context.currentTime + 0.08;
      decoded.forEach(({row, audio}) => {
        const source = context.createBufferSource();
        const gain = context.createGain();
        source.buffer = audio;
        gain.gain.value = decoded.length > 1 ? 0.42 : 0.9;
        source.connect(gain);
        gain.connect(context.destination);
        const delay = aligned ? Number(row.audio?.scheduled_start_ms ?? 0) / 1000 : 0;
        source.start(baseTime + Math.max(0, delay));
      });
    } catch (cause) {
      await closeAudioContext(raceContextRef);
      setRaceError(`对齐播放失败：${String(cause)}`);
    }
  }

  const traceRequest = {
    ...request,
    text: traceText.trim() || request.text,
  } satisfies LabRequest;

  return <section className="lab-feature-showcase" data-testid="feature-showcase">
    <div className="panel lab-feature-intro">
      <div className="panel-heading"><div><p className="panel-kicker">MIGRATED FEATURE SHOWCASE</p><h2>工程特性实验</h2></div><p>原独立 WebUI 的时间轴、LLM PK 和并发面板已汇入当前门户。</p></div>
      <p className="hint">以下指标来自可选 <code>demo_api</code> 或当前实时连接，不代表预置 benchmark；无真实音频时不会生成占位声音。</p>
    </div>

    <LiveTracePanel
      text={traceText}
      onTextChange={setTraceText}
      events={trace}
      audioUrl={traceAudioUrl || undefined}
      liveMs={traceLiveMs}
      live={traceLive}
      source="live TRT stream"
      error={traceError}
      onStart={() => void startLiveTrace()}
      onStop={stopLiveTrace}
    />
    <PerformanceRace
      api={api}
      defaults={traceRequest}
      onPlayAligned={(rows) => void playRaceRows(rows, true)}
      onPlayOne={(row) => void playRaceRows([row], false)}
    />
    <ConcurrencyPanel api={api} request={traceRequest} />
    {traceAudioUrl && <div className="panel lab-feature-download"><a className="button" href={traceAudioUrl} download="qwen3tts-live-trace.wav"><Download size={15}/>下载实时 trace WAV</a></div>}
    {raceError && <p className="alert" role="alert">{raceError}</p>}
  </section>;
}

interface LiveTracePanelProps {
  readonly text: string;
  readonly onTextChange: (value: string) => void;
  readonly events: LabTraceEvent[];
  readonly audioUrl: string | undefined;
  readonly liveMs: number;
  readonly live: boolean;
  readonly source: string;
  readonly error: string;
  readonly onStart: () => void;
  readonly onStop: () => void;
}

/** Live protocol adapter plus the migrated TextPlayer presentation. */
export function LiveTracePanel({
  text,
  onTextChange,
  events,
  audioUrl,
  liveMs,
  live,
  source,
  error,
  onStart,
  onStop,
}: LiveTracePanelProps) {
  return <section className="panel lab-live-trace-panel">
    <div className="panel-heading"><div><p className="panel-kicker">TEXT PLAYER · LIVE TRACE</p><h2>文本与解码时间轴</h2></div><p>通过可选工程后端捕获真实 PCM 和 engine trace。</p></div>
    <textarea value={text} onChange={(event) => onTextChange(event.target.value)} rows={2} aria-label="实时 trace 文本" />
    <div className="actions">
      <button className="primary" disabled={live || !text.trim()} onClick={onStart}><Play size={15}/>开始实时 trace</button>
      <button disabled={!live} onClick={onStop}><Square size={15}/>停止</button>
      {audioUrl && <a className="button" href={audioUrl} download="qwen3tts-live-trace.wav"><Download size={15}/>下载 WAV</a>}
      {live && <span className="hint"><RadioTower size={14}/> {liveMs.toFixed(0)} ms</span>}
    </div>
    {error && <p className="alert" role="alert">{error}</p>}
    <TextPlayer
      events={events}
      text={text}
      {...(audioUrl ? {audioUrl} : {})}
      liveMs={liveMs}
      live={live}
      source={source}
    />
  </section>;
}

interface PcmFormat {
  readonly encoding: "pcm_f32" | "pcm_s16le";
  readonly sampleRate: number;
  readonly channels: number;
}

const DEFAULT_PCM_FORMAT: PcmFormat = {encoding: "pcm_f32", sampleRate: 24_000, channels: 1};
const MAX_TRACE_EVENTS = 500;
const MAX_CAPTURE_BYTES = 32 * 1024 * 1024;
const WAV_HEADER_BYTES = 44;

/** Exported for focused unit tests and for other migrated panels. */
export function wavBlobFromPcm(parts: readonly ArrayBuffer[], format: PcmFormat = DEFAULT_PCM_FORMAT): Blob {
  const bytesPerInputSample = format.encoding === "pcm_s16le" ? 2 : 4;
  const inputBytes = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const sampleCount = Math.floor(inputBytes / (bytesPerInputSample * Math.max(1, format.channels)));
  const output = new ArrayBuffer(WAV_HEADER_BYTES + sampleCount * 2);
  const view = new DataView(output);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, Math.max(1, Math.round(format.sampleRate)), true);
  view.setUint32(28, Math.max(1, Math.round(format.sampleRate)) * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let outputOffset = WAV_HEADER_BYTES;
  let pending = new Uint8Array(0);
  for (const part of parts) {
    const incoming = new Uint8Array(part);
    const merged = new Uint8Array(pending.length + incoming.length);
    merged.set(pending);
    merged.set(incoming, pending.length);
    let offset = 0;
    const frameBytes = bytesPerInputSample * Math.max(1, format.channels);
    while (offset + frameBytes <= merged.byteLength && outputOffset < output.byteLength) {
      // The public TTS contract is mono.  If an older trace reports multiple
      // channels, retain the first channel and skip the remaining samples.
      const sample = format.encoding === "pcm_s16le"
        ? new DataView(merged.buffer, merged.byteOffset + offset, 2).getInt16(0, true)
        : floatToInt16(new DataView(merged.buffer, merged.byteOffset + offset, 4).getFloat32(0, true));
      view.setInt16(outputOffset, sample, true);
      outputOffset += 2;
      offset += frameBytes;
    }
    pending = merged.subarray(offset).slice();
  }
  return new Blob([output], {type: "audio/wav"});
}

function pcmToInt16(bytes: ArrayBuffer, encoding: PcmFormat["encoding"]): Int16Array {
  if (encoding === "pcm_s16le") {
    const count = Math.floor(bytes.byteLength / 2);
    const result = new Int16Array(count);
    const view = new DataView(bytes);
    for (let index = 0; index < count; index += 1) result[index] = view.getInt16(index * 2, true);
    return result;
  }
  const count = Math.floor(bytes.byteLength / 4);
  const result = new Int16Array(count);
  const view = new DataView(bytes);
  for (let index = 0; index < count; index += 1) result[index] = floatToInt16(view.getFloat32(index * 4, true));
  return result;
}

function floatToInt16(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-32768, Math.min(32767, Math.round(value * (value < 0 ? 32768 : 32767))));
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function normalizePcmFormat(value: Record<string, unknown>): PcmFormat {
  const encoding = String(value.encoding ?? "pcm_f32") === "pcm_s16le" ? "pcm_s16le" : "pcm_f32";
  const sampleRate = Number(value.sample_rate ?? value.sampleRate ?? DEFAULT_PCM_FORMAT.sampleRate);
  const channels = Number(value.channels ?? 1);
  return {
    encoding,
    sampleRate: Number.isFinite(sampleRate) && sampleRate > 0 ? Math.round(sampleRate) : DEFAULT_PCM_FORMAT.sampleRate,
    channels: Number.isFinite(channels) && channels > 0 ? Math.round(channels) : 1,
  };
}

function normalizeTraceEvent(value: Record<string, unknown>): LabTraceEvent | null {
  if (typeof value.type !== "string" || !value.type) return null;
  const tMs = Number(value.t_ms ?? 0);
  const event: LabTraceEvent = {
    type: value.type,
    t_ms: Number.isFinite(tMs) ? tMs : 0,
  };
  if (typeof value.run_id === "string") event.run_id = value.run_id;
  if (typeof value.backend === "string") event.backend = value.backend;
  if (typeof value.server_t_ms === "number") event.server_t_ms = value.server_t_ms;
  if (typeof value.stream_id === "string") event.stream_id = value.stream_id;
  if (typeof value.text === "string") event.text = value.text;
  if (isRecord(value.meta)) event.meta = value.meta;
  return event;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function appendBounded<T>(items: readonly T[], item: T, limit: number): T[] {
  const next = [...items, item];
  return next.length > limit ? next.slice(next.length - limit) : next;
}

function appendBoundedAudio(
  bytes: ArrayBuffer,
  partsRef: {current: ArrayBuffer[]},
  totalRef: {current: number},
  limit: number,
): ArrayBuffer {
  const remaining = Math.max(0, limit - totalRef.current);
  if (remaining <= 0) return new ArrayBuffer(0);
  const accepted = bytes.byteLength <= remaining ? bytes.slice(0) : bytes.slice(0, remaining);
  partsRef.current.push(accepted);
  totalRef.current += accepted.byteLength;
  return accepted;
}

async function asArrayBuffer(value: unknown): Promise<ArrayBuffer | null> {
  if (value instanceof ArrayBuffer) return value.slice(0);
  if (typeof Blob !== "undefined" && value instanceof Blob) return await value.arrayBuffer();
  if (ArrayBuffer.isView(value)) {
    const view = value as ArrayBufferView;
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer;
  }
  return null;
}

async function closeAudioContext(contextRef: {current: AudioContext | null}): Promise<void> {
  const context = contextRef.current;
  contextRef.current = null;
  if (context) await context.close().catch(() => undefined);
}

/** Small bounded scheduler used only for live trace preview. */
class PcmScheduler {
  private context: AudioContext | null = null;
  private nextTime = 0;

  async start(): Promise<void> {
    const AudioContextCtor = window.AudioContext;
    if (!AudioContextCtor) throw new Error("AudioContext unavailable");
    this.context = new AudioContextCtor();
    await this.context.resume();
    this.nextTime = this.context.currentTime + 0.04;
  }

  enqueue(bytes: ArrayBuffer, format: PcmFormat): void {
    const context = this.context;
    if (!context) return;
    const channels = Math.max(1, format.channels);
    const bytesPerSample = format.encoding === "pcm_s16le" ? 2 : 4;
    const frames = Math.floor(bytes.byteLength / (bytesPerSample * channels));
    if (frames <= 0) return;
    const buffer = context.createBuffer(channels, frames, format.sampleRate);
    const view = new DataView(bytes);
    for (let channel = 0; channel < channels; channel += 1) {
      const output = buffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        const offset = (frame * channels + channel) * bytesPerSample;
        output[frame] = format.encoding === "pcm_s16le"
          ? view.getInt16(offset, true) / 32768
          : view.getFloat32(offset, true);
      }
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const start = Math.max(this.nextTime, context.currentTime + 0.02);
    source.start(start);
    this.nextTime = start + frames / format.sampleRate;
  }

  close(): void {
    const context = this.context;
    this.context = null;
    if (context) void context.close().catch(() => undefined);
  }
}
