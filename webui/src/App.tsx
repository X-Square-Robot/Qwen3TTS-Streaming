import { Activity, AlertTriangle, Download, RadioTower, RotateCcw, Zap } from "lucide-react";
import type { MutableRefObject } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getCapabilities, mediaUrl, runLlmPk, wsUrl } from "./api";
import { PcmStreamPlayer } from "./audio/pcm-player";
import { wavBlobFromPcm } from "./audio/wav";
import { ConcurrencyPanel } from "./components/ConcurrencyPanel";
import { PerformanceRace } from "./components/PerformanceRace";
import { TextPlayer } from "./components/TextPlayer";
import { formatMs } from "./components/Timeline";
import type {
  AudioFormat,
  Capabilities,
  DemoRequest,
  LlmPkRequest,
  LlmPkResult,
  TraceEvent,
} from "./types";
import "./styles.css";

const DEFAULT_AUDIO: AudioFormat = { encoding: "pcm_f32", sample_rate: 24000, channels: 1 };

const FALLBACK_LLM_PK: LlmPkRequest = {
  text: "你好，今天天气不错，我们来聊聊最近你看过的书，有没有什么推荐的？",
  speaker: "Serena",
  language: "auto",
  ms_per_token: 30,
};

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [llmPkDefaults, setLlmPkDefaults] = useState<LlmPkRequest>(FALLBACK_LLM_PK);
  const [trtRequest, setTrtRequest] = useState<DemoRequest>({
    text: FALLBACK_LLM_PK.text,
    speaker: FALLBACK_LLM_PK.speaker,
    language: FALLBACK_LLM_PK.language,
  });
  const [llmPk, setLlmPk] = useState<LlmPkResult | null>(null);
  const [llmPkBusy, setLlmPkBusy] = useState(false);
  const [liveEvents, setLiveEvents] = useState<TraceEvent[]>([]);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveClockRunning, setLiveClockRunning] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [liveClockMs, setLiveClockMs] = useState(0);
  const [liveAudioUrl, setLiveAudioUrl] = useState<string | undefined>(undefined);
  const playerRef = useRef<PcmStreamPlayer | null>(null);
  const raceAudioRef = useRef<AudioContext | null>(null);
  const liveSocketRef = useRef<WebSocket | null>(null);
  const liveRunSeqRef = useRef(0);
  const liveAudioPartsRef = useRef<ArrayBuffer[]>([]);
  const liveAudioFormatRef = useRef<AudioFormat>(DEFAULT_AUDIO);
  const liveStartedAtRef = useRef<number | null>(null);
  const liveFirstAudioMsRef = useRef<number | null>(null);
  const liveAudioDurationMsRef = useRef(0);
  const livePlaybackEndMsRef = useRef<number | null>(null);

  useEffect(() => {
    getCapabilities()
      .then((caps) => {
        setCapabilities(caps);
        setLlmPkDefaults(caps.default_request);
        setTrtRequest({
          text: caps.default_request.text,
          speaker: caps.default_request.speaker,
          language: caps.default_request.language,
        });
      })
      .catch((error) => setWarnings([String(error)]));
  }, []);

  useEffect(() => {
    if (!liveClockRunning || liveStartedAtRef.current === null) {
      return;
    }
    let frame = 0;
    const tick = () => {
      if (liveStartedAtRef.current !== null) {
        const elapsedMs = performance.now() - liveStartedAtRef.current;
        const playbackEndMs = livePlaybackEndMsRef.current;
        if (playbackEndMs !== null && elapsedMs >= playbackEndMs) {
          setLiveClockMs(playbackEndMs);
          liveStartedAtRef.current = null;
          livePlaybackEndMsRef.current = null;
          setLiveClockRunning(false);
          setLiveBusy(false);
          return;
        }
        setLiveClockMs(elapsedMs);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [liveClockRunning]);

  function resetLiveProgress() {
    liveStartedAtRef.current = null;
    liveFirstAudioMsRef.current = null;
    liveAudioDurationMsRef.current = 0;
    livePlaybackEndMsRef.current = null;
    setLiveClockRunning(false);
  }

  function clearLiveAudioUrl() {
    if (liveAudioUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(liveAudioUrl);
    }
    setLiveAudioUrl(undefined);
    liveAudioPartsRef.current = [];
    liveAudioFormatRef.current = DEFAULT_AUDIO;
  }

  async function onRunLlmPk(request: LlmPkRequest) {
    setLlmPkBusy(true);
    setWarnings([]);
    try {
      const result = await runLlmPk(request);
      setLlmPk(result);
      setWarnings(result.warnings ?? []);
    } catch (error) {
      setWarnings([String(error)]);
    } finally {
      setLlmPkBusy(false);
    }
  }

  async function speakTrt() {
    const runSeq = liveRunSeqRef.current + 1;
    liveRunSeqRef.current = runSeq;
    liveSocketRef.current?.close();
    liveSocketRef.current = null;
    resetLiveProgress();
    setLiveBusy(true);
    setLiveEvents([]);
    setLiveClockMs(0);
    clearLiveAudioUrl();
    setWarnings([]);
    await playerRef.current?.stop();
    const player = new PcmStreamPlayer();
    playerRef.current = player;
    await player.start();

    const socket = new WebSocket(wsUrl("/api/v1/trt-live"));
    liveSocketRef.current = socket;
    socket.binaryType = "arraybuffer";
    let audioFormat = DEFAULT_AUDIO;
    socket.onopen = () => {
      if (runSeq !== liveRunSeqRef.current) {
        socket.close();
        return;
      }
      liveStartedAtRef.current = performance.now();
      setLiveClockRunning(true);
      socket.send(JSON.stringify({ type: "speak", ...trtRequest }));
    };
    socket.onmessage = (event) => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      if (typeof event.data !== "string") {
        liveAudioPartsRef.current.push(event.data.slice(0));
        liveAudioDurationMsRef.current += audioDurationMs(event.data.byteLength, audioFormat);
        player.enqueue(event.data, audioFormat);
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === "event") {
        const trace = message.event as TraceEvent;
        const maybeFormat = trace.meta?.audio_format as AudioFormat | undefined;
        if (maybeFormat) {
          audioFormat = maybeFormat;
          liveAudioFormatRef.current = maybeFormat;
        }
        if (trace.type === "first_audio_chunk") {
          liveFirstAudioMsRef.current = trace.t_ms;
        }
        setLiveEvents((current) => [...current, trace]);
        if (trace.type === "warning" && trace.meta?.message) {
          setWarnings((current) => [...current, String(trace.meta?.message)]);
        }
        if (trace.type === "done") {
          const playbackEndMs =
            liveFirstAudioMsRef.current === null
              ? trace.t_ms
              : Math.max(trace.t_ms, liveFirstAudioMsRef.current + liveAudioDurationMsRef.current);
          setLiveClockMs(trace.t_ms);
          livePlaybackEndMsRef.current = playbackEndMs;
          if (liveAudioPartsRef.current.length > 0) {
            const blob = wavBlobFromPcm(liveAudioPartsRef.current, liveAudioFormatRef.current);
            setLiveAudioUrl(URL.createObjectURL(blob));
          }
          if (playbackEndMs <= trace.t_ms + 16) {
            liveStartedAtRef.current = null;
            livePlaybackEndMsRef.current = null;
            setLiveClockRunning(false);
            setLiveBusy(false);
          }
          socket.close();
        }
      }
      if (message.type === "error") {
        setWarnings((current) => [...current, message.message]);
        resetLiveProgress();
        setLiveBusy(false);
      }
    };
    socket.onerror = () => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      setWarnings((current) => [...current, "TRT live websocket failed"]);
      resetLiveProgress();
      setLiveBusy(false);
    };
    socket.onclose = () => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      if (liveSocketRef.current === socket) {
        liveSocketRef.current = null;
      }
      if (livePlaybackEndMsRef.current !== null) {
        return;
      }
      resetLiveProgress();
      setLiveBusy(false);
    };
  }

  const liveFirstAudio = liveEvents.find((event) => event.type === "first_audio_chunk")?.t_ms;
  const tritonReady = capabilities?.backends.find((backend) => backend.id === "triton_trt_streaming")?.live_available;
  const llmPkReady = capabilities?.backends.some((backend) => (backend.id === "triton_streaming" || backend.id === "triton_offline") && backend.live_available);
  const release = llmPk?.release ?? capabilities?.release;
  const limitations = llmPk?.limitations ?? capabilities?.limitations ?? [];
  const showingLiveTrace = liveBusy || liveEvents.length > 0;
  const tokenEvents = useMemo(
    () => (showingLiveTrace ? liveEvents : preferredTokenEvents(llmPk)),
    [llmPk, liveEvents, showingLiveTrace],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Qwen3TTS-Streaming</h1>
          <p>token streaming demo · LLM upstream simulation</p>
        </div>
        <div className="headline-metrics">
          <div>
            <span>Cache-hit TTFT</span>
            <strong>{capabilities?.headline.single_stream_cache_hit_ttft_ms ?? 13}ms</strong>
          </div>
          <div>
            <span>128-stream avg</span>
            <strong>{capabilities?.headline.concurrent_128_avg_ttft_ms ?? 180}ms</strong>
          </div>
        </div>
      </header>

      <section className="release-notice">
        <div>
          <AlertTriangle size={17} />
          <strong>{release?.stage === "engineering_preview" ? "Engineering Preview" : "Preview"}</strong>
          <span>{release?.positioning ?? "This demo shows the TensorRT token-streaming path and does not represent production stability."}</span>
        </div>
        {limitations.length > 0 && (
          <ul>
            {limitations.slice(0, 4).map((item) => <li key={item}>{item}</li>)}
          </ul>
        )}
      </section>

      <section className="control-surface">
        <textarea value={trtRequest.text} onChange={(event) => setTrtRequest({ ...trtRequest, text: event.target.value })} />
        <div className="controls-row">
          <input value={trtRequest.speaker} onChange={(event) => setTrtRequest({ ...trtRequest, speaker: event.target.value })} aria-label="speaker" />
          <input value={trtRequest.language} onChange={(event) => setTrtRequest({ ...trtRequest, language: event.target.value })} aria-label="language" />
          <button className="button primary" onClick={speakTrt} disabled={liveBusy}>
            <Zap size={16} />
            {liveBusy ? "Speaking" : "Speak TRT"}
          </button>
          <button className="button ghost" onClick={() => clearLiveAudioUrl()}>
            <RotateCcw size={16} />
            Clear
          </button>
        </div>
      </section>

      <section className="status-strip">
        <div className={`status-pill ${tritonReady ? "ready" : "muted"}`}>
          <RadioTower size={15} />
          Triton {tritonReady ? "live" : "fixture fallback"}
        </div>
        <div className={`status-pill ${llmPkReady ? "ready" : "muted"}`}>
          <RadioTower size={15} />
          LLM PK {llmPkReady ? "live" : "offline"}
        </div>
        <div className="status-pill">
          <Activity size={15} />
          Live first audio {formatMs(liveFirstAudio)}
        </div>
        <button className="button tiny" onClick={() => downloadJson(llmPk)}>
          <Download size={14} />
          JSON
        </button>
      </section>

      {warnings.length > 0 && (
        <section className="warning-band">
          {warnings.map((warning, index) => <span key={`${warning}-${index}`}>{warning}</span>)}
        </section>
      )}

      <TextPlayer
        events={tokenEvents}
        text={trtRequest.text}
        audioUrl={liveAudioUrl}
        liveMs={showingLiveTrace ? liveClockMs : 0}
        live={liveBusy}
        source={showingLiveTrace ? "live TRT stream" : "trace"}
      />

      <PerformanceRace
        result={llmPk}
        busy={llmPkBusy}
        defaults={llmPkDefaults}
        onRun={onRunLlmPk}
        onPlayAligned={(items) => playAudioRace(items, true, raceAudioRef)}
        onPlayOne={(item) => playAudioRace([item], false, raceAudioRef)}
      />

      <ConcurrencyPanel
        request={{
          text: trtRequest.text,
          speaker: trtRequest.speaker,
          language: trtRequest.language,
        }}
      />
    </main>
  );
}

function preferredTokenEvents(result: LlmPkResult | null): TraceEvent[] {
  if (!result) return [];
  const streaming = result.results.find((row) => row.backend === "triton_streaming");
  if (streaming?.events.length) {
    return streaming.events;
  }
  const offline = result.results.find((row) => row.backend === "triton_offline");
  return offline?.events ?? [];
}

async function playAudioRace(
  results: Array<{ audio?: { url: string; scheduled_start_ms?: number }; label: string }>,
  measuredWait: boolean,
  contextRef: MutableRefObject<AudioContext | null>,
): Promise<void> {
  if (contextRef.current) {
    await contextRef.current.close();
  }
  const playable = results.filter((result) => result.audio?.url);
  if (playable.length === 0) {
    return;
  }
  const context = new AudioContext();
  contextRef.current = context;
  await context.resume();
  const decoded = await Promise.all(
    playable.map(async (result) => {
      const response = await fetch(mediaUrl(result.audio!.url));
      if (!response.ok) {
        throw new Error(`audio fetch failed for ${result.label}: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      return {
        result,
        audio: await context.decodeAudioData(buffer.slice(0)),
      };
    }),
  );
  const baseTime = context.currentTime + 0.08;
  decoded.forEach(({ result, audio }) => {
    const source = context.createBufferSource();
    const gain = context.createGain();
    source.buffer = audio;
    gain.gain.value = playable.length > 1 ? 0.42 : 0.9;
    source.connect(gain);
    gain.connect(context.destination);
    const delaySec = measuredWait ? Math.max(0, result.audio?.scheduled_start_ms ?? 0) / 1000 : 0;
    source.start(baseTime + delaySec);
  });
}

function downloadJson(result: LlmPkResult | null): void {
  if (!result) return;
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "qwen3-tts-llm-pk-result.json";
  link.click();
  URL.revokeObjectURL(url);
}

function audioDurationMs(byteLength: number, format: AudioFormat): number {
  const sampleRate = format.sample_rate || 24000;
  const channels = format.channels || 1;
  const bytesPerSample = format.encoding === "pcm_s16le" ? 2 : 4;
  return (byteLength / (sampleRate * channels * bytesPerSample)) * 1000;
}
