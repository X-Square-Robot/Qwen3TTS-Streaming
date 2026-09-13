import {
  Activity,
  BadgeCheck,
  BookOpen,
  Cpu,
  Download,
  Headphones,
  Mic,
  Package,
  Pause,
  Play,
  Square,
} from "lucide-react";
import {useEffect, useRef, useState} from "react";
import {
  AudioEncoding,
  BrowserAudioPlayer,
  DeliveryPolicy,
  discoverCapabilities,
  InputMode,
  RealtimeTTSClient,
  resolveRelativeUrl,
  SynthesisTask,
  VadStrategy,
  WavCollector,
  type Capabilities,
  type IncrementalSynthesisRun,
  type SynthesisRun,
  type TTSEvent,
} from "@xmultimodalinteraction/qwen3tts-browser";

import {loadDemoConfig, type LoadedDemoConfig} from "./config";
import {DocsPage} from "./DocsPage";
import {ExperimentLab} from "./ExperimentLab";
import {PlaybackWaveform} from "./components/PlaybackWaveform";
import {MediaPlayer} from "./components/MediaPlayer";
import {ProgressTrack} from "./components/ProgressTrack";
import {appendAudioEnvelope, type AudioEnvelope} from "./components/audio-envelope";
import {
  latestProgressEvent,
  monotonicSampleRatio,
  monotonicRawEnd,
  progressRawEnd,
  progressSample,
  progressSampleBigInt,
  sortProgressEvents,
  type ProgressEvent,
} from "./progress";
import {SdkPage} from "./SdkPage";
import {
  DEFAULT_DEMO_SETTINGS,
  defaultVadTuning,
  type DemoSynthesisSettings,
} from "./demo-settings";
import {startReferenceRecorder, type ReferenceRecorder} from "./reference-recorder";

type Route = "experience" | "sdk" | "docs";

export function App() {
  const [loaded, setLoaded] = useState<LoadedDemoConfig | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [route, setRoute] = useState<Route>(routeFromHash());
  const [docsOnly, setDocsOnly] = useState(false);
  const [exampleSettings, setExampleSettings] = useState<DemoSynthesisSettings>(DEFAULT_DEMO_SETTINGS);

  useEffect(() => {
    loadDemoConfig().then(setLoaded).catch(() => setDocsOnly(true));
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const nav = (target: Route, label: string, icon: React.ReactNode) => (
    <a className={route === target ? "active" : ""} href={`#/${target}`}>{icon}{label}</a>
  );

  return <>
    <header className="site-header">
      <div className="header-inner">
        <a className="brand" href="#/experience" aria-label="Qwen3 TTS 首页">
          <span className="brand-mark" aria-hidden="true"><i/><i/><i/><i/></span>
          <span className="brand-copy"><strong>QWEN3 TTS</strong><small>STREAMING STUDIO</small></span>
        </a>
        <nav aria-label="主导航">
          {nav("experience", "体验", <Headphones size={16} />)}
          {nav("sdk", "SDK", <Package size={16} />)}
          {nav("docs", "文档", <BookOpen size={16} />)}
        </nav>
        <span className="release"><i/>{loaded?.config.engine_version || "DEV"}</span>
      </div>
    </header>
    <main className="app-main">
      {docsOnly && <div className="alert">文档只读模式：此 Pages 站点未连接具体 TTS 实例。请在部署实例的 /demo/ 页面进行合成、试听和 SDK 下载。</div>}
      {route === "experience" && <Experience loaded={loaded} onCapabilities={setCapabilities} onSettings={setExampleSettings} settings={exampleSettings} />}
      {route === "sdk" && <SdkPage loaded={loaded} capabilities={capabilities} settings={exampleSettings} docsOnly={docsOnly} />}
      {route === "docs" && <DocsPage />}
    </main>
  </>;
}

function Experience({loaded, onCapabilities, onSettings, settings}: {
  loaded: LoadedDemoConfig | null;
  onCapabilities: (value: Capabilities) => void;
  onSettings: (value: DemoSynthesisSettings) => void;
  settings: DemoSynthesisSettings;
}) {
  const [text, setText] = useState("你好，欢迎体验 Qwen3 TTS 流式语音合成服务。");
  const [synthesisText, setSynthesisText] = useState("");
  const [synthesisSampleRate, setSynthesisSampleRate] = useState(24_000);
  const synthesisRateRef = useRef(24_000);
  const [receivedSamples, setReceivedSamples] = useState(0);
  const [speaker, setSpeaker] = useState("Serena");
  const [language, setLanguage] = useState("auto");
  const [task, setTask] = useState(SynthesisTask.CustomVoice);
  const [inputMode, setInputMode] = useState<"full" | "long" | "incremental">("full");
  const [vad, setVad] = useState(VadStrategy.Disabled);
  const [delivery, setDelivery] = useState(DeliveryPolicy.Guarded);
  const [sampleRate, setSampleRate] = useState(24_000);
  const [vadBeginThreshold, setVadBeginThreshold] = useState(DEFAULT_DEMO_SETTINGS.vadBeginThreshold);
  const [vadEndThreshold, setVadEndThreshold] = useState(DEFAULT_DEMO_SETTINGS.vadEndThreshold);
  const [vadBeginCount, setVadBeginCount] = useState(DEFAULT_DEMO_SETTINGS.vadBeginCount);
  const [vadEndCount, setVadEndCount] = useState(DEFAULT_DEMO_SETTINGS.vadEndCount);
  const [vadChunkMs, setVadChunkMs] = useState(DEFAULT_DEMO_SETTINGS.vadChunkMs);
  const [vadStartMarginMs, setVadStartMarginMs] = useState(DEFAULT_DEMO_SETTINGS.vadStartMarginMs);
  const [deliveryWindowMs, setDeliveryWindowMs] = useState(160);
  const [outputChunkMs, setOutputChunkMs] = useState(0);
  const [emitTextEvents, setEmitTextEvents] = useState(true);
  const [volume, setVolume] = useState(1);
  const [outputDevice, setOutputDevice] = useState("");
  const [outputDevices, setOutputDevices] = useState<MediaDeviceInfo[]>([]);
  const [instruct, setInstruct] = useState("");
  const [reference, setReference] = useState<{audioBase64: string; name: string} | null>(null);
  const [referenceText, setReferenceText] = useState("");
  const [recording, setRecording] = useState(false);
  const [xVectorOnly, setXVectorOnly] = useState(false);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [events, setEvents] = useState<TTSEvent[]>([]);
  // Keep progress history separate from the bounded diagnostic event log.
  const [progressHistory, setProgressHistory] = useState<TTSEvent[]>([]);
  const [audioEnvelope, setAudioEnvelope] = useState<AudioEnvelope[]>([]);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [outputIssue, setOutputIssue] = useState("");
  const downloadUrlRef = useRef("");
  const clientRef = useRef<RealtimeTTSClient | null>(null);
  const playerRef = useRef<BrowserAudioPlayer | null>(null);
  const runRef = useRef<SynthesisRun | null>(null);
  const collectorRef = useRef<WavCollector | null>(null);
  const recorderRef = useRef<ReferenceRecorder | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  const wavLimitWarned = useRef(false);
  const playbackFailed = useRef(false);
  const startedAt = useRef(0);
  const requestStartedAt = useRef(0);
  const streamGeneration = useRef(0);
  const [usage, setUsage] = useState<Record<string, number>>({});
  const [timing, setTiming] = useState({setup: 0, ttfb: 0, firstAudio: 0, firstAudible: 0, total: 0});
  const [serverTiming, setServerTiming] = useState({
    ttft: 0,
    total: 0,
    queueWait: 0,
    prefill: 0,
    dequeueToRaw: 0,
    rawToEffective: 0,
    prefixTrimmed: 0,
    prefixApplied: false,
    vad: "",
  });
  const [playback, setPlayback] = useState({played: 0n, buffered: 0n, queuedFrames: 0, underruns: 0});

  useEffect(() => onSettings({
    task, speaker, language, sampleRate, inputMode, vad, vadChunkMs,
    vadBeginThreshold, vadBeginCount, vadEndThreshold, vadEndCount,
    vadStartMarginMs, delivery, deliveryWindowMs, outputChunkMs, emitTextEvents,
  }), [
    task, speaker, language, sampleRate, inputMode, vad, vadChunkMs,
    vadBeginThreshold, vadBeginCount, vadEndThreshold, vadEndCount,
    vadStartMarginMs, delivery, deliveryWindowMs, outputChunkMs, emitTextEvents,
    onSettings,
  ]);

  useEffect(() => () => {
    streamGeneration.current += 1;
    clientRef.current?.close();
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    void recorderRef.current?.stop().catch(() => undefined);
    void playerRef.current?.close();
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const endpoint = resolveRelativeUrl(loaded.config.endpoints.capabilities_url, loaded.responseUrl);
    discoverCapabilities(endpoint).then((value) => {
      setCaps(value);
      onCapabilities(value);
      const firstTask = value.tasks.find((candidate) => Object.values(SynthesisTask).includes(candidate as SynthesisTask));
      if (firstTask) setTask(firstTask as SynthesisTask);
      const firstFormat = value.audio_formats.find((format) => format.encoding === AudioEncoding.PcmS16Le);
      if (firstFormat) setSampleRate(firstFormat.sample_rate);
      if (value.speakers?.length) setSpeaker((current) => value.speakers?.includes(current) ? current : value.speakers?.[0] ?? current);
      if (value.languages?.length) setLanguage((current) => value.languages?.includes(current) ? current : value.languages?.[0] ?? current);
      if (value.input_modes?.includes("full_text")) setInputMode("full");
      else if (value.input_modes?.includes("long_segment")) setInputMode("long");
      else if (value.input_modes?.includes("token")) setInputMode("incremental");
      setXVectorOnly(!value.reference.icl_available && Boolean(value.reference.speaker_encoder_available));
    }).catch((cause) => setEvents((current) => [...current, {type: "error", code: "capabilities", message: String(cause)}]));
  }, [loaded, onCapabilities]);

  async function connectAndPlay() {
    if (!loaded) return;
    setEvents([]);
    setProgressHistory([]);
    setAudioEnvelope([]);
    setSynthesisText(text);
    setSynthesisSampleRate(sampleRate);
    synthesisRateRef.current = sampleRate;
    setReceivedSamples(0);
    setPlayback({played: 0n, buffered: 0n, queuedFrames: 0, underruns: 0});
    setUsage({});
    wavLimitWarned.current = false;
    playbackFailed.current = false;
    setBusy(true);
    setTiming({setup: 0, ttfb: 0, firstAudio: 0, firstAudible: 0, total: 0});
    setServerTiming({
      ttft: 0,
      total: 0,
      queueWait: 0,
      prefill: 0,
      dequeueToRaw: 0,
      rawToEffective: 0,
      prefixTrimmed: 0,
      prefixApplied: false,
      vad: "",
    });
    setOutputIssue("");
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current);
    downloadUrlRef.current = "";
    setDownloadUrl("");
    streamGeneration.current += 1;
    const generation = streamGeneration.current;
    // Keep the click-to-finish wall clock separate from request-to-audio
    // timings. Connection/player setup is useful client-side telemetry, but it
    // must not be mislabeled as TTFT.
    startedAt.current = performance.now();
    requestStartedAt.current = startedAt.current;
    try {
      const configUrl = loaded.responseUrl;
      const capabilitiesUrl = resolveRelativeUrl(
        loaded.config.endpoints.capabilities_url,
        configUrl,
      );
      const websocketUrl = resolveRelativeUrl(
        loaded.config.endpoints.openai_realtime_url,
        configUrl,
      );
      let client = clientRef.current;
      if (!client) {
        client = new RealtimeTTSClient({capabilitiesUrl, websocketUrl});
        client.onEvent(handleEvent);
        await client.connect();
        clientRef.current = client;
        if (!client.capabilities) throw new Error("Capabilities unavailable after connect");
        setCaps(client.capabilities);
        onCapabilities(client.capabilities);
      }
      const collector = new WavCollector({sampleRate});
      collectorRef.current = collector;
      await playerRef.current?.close();
      const player = new BrowserAudioPlayer({
        onPlaybackProgress: (played, buffered) => {
          runRef.current?.acknowledgePlayback(played, buffered);
          const snapshot = playerRef.current?.snapshot();
          setPlayback({played, buffered, queuedFrames: snapshot?.queuedFrames ?? 0, underruns: snapshot?.underruns ?? 0});
          if (played > 0n) setTiming((current) => current.firstAudible ? current : {...current, firstAudible: performance.now() - requestStartedAt.current});
        },
        onUnderrun: () => setEvents((current) => [
          ...current,
          {type: "warning", message: "浏览器播放缓冲发生 underrun"},
        ]),
        onFallback: () => setEvents((current) => [
          ...current,
          {type: "warning", message: "当前 HTTP 页面不支持 AudioWorklet，已切换到兼容播放模式"},
        ]),
        onError: (error) => setEvents((current) => [...current, {type: "error", code: "audio_player", message: error.message}]),
      });
      await player.start();
      player.setVolume(volume);
      if (outputDevice) await player.setOutputDevice(outputDevice);
      playerRef.current = player;
      if (player.supportsOutputDeviceSelection() && navigator.mediaDevices?.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setOutputDevices(devices.filter((device) => device.kind === "audiooutput"));
      } else {
        setOutputDevices([]);
      }
      requestStartedAt.current = performance.now();
      setTiming((current) => ({...current, setup: requestStartedAt.current - startedAt.current}));
      const options = {
        task,
        speaker,
        language,
        ...(instruct ? {instruct} : {}),
        ...(reference ? {reference: {audioBase64: reference.audioBase64, text: referenceText, xVectorOnly}} : {}),
        inputMode: inputMode === "full" ? InputMode.FullText : inputMode === "long" ? InputMode.LongSegment : InputMode.Token,
        audio: {encoding: AudioEncoding.PcmS16Le, sample_rate: sampleRate, channels: 1 as const},
        vad: {
          enabled: vad !== VadStrategy.Disabled,
          strategy: vad,
          chunk_ms: vadChunkMs,
          begin_threshold: vadBeginThreshold,
          begin_count: vadBeginCount,
          end_threshold: vadEndThreshold,
          end_count: vadEndCount,
          start_margin_ms: vadStartMarginMs,
        },
        outputPolicy: {delivery, delivery_window_ms: deliveryWindowMs, chunk_ms: outputChunkMs, emit_text_events: emitTextEvents},
      };
      const run = inputMode !== "incremental"
        ? await client.synthesize(text, options)
        : await client.startIncremental(options);
      runRef.current = run;
      if (inputMode === "incremental") void streamText(run as IncrementalSynthesisRun, text, generation, streamGeneration);
      await run.done;
    } catch (cause) {
      setEvents((current) => [...current, {type: "error", code: "demo_error", message: String(cause)}]);
      setBusy(false);
      setPaused(false);
      runRef.current = null;
    }
  }

  function handleEvent(event: TTSEvent) {
    const maxSessions =
      (event.type === "warning" && isMaxSessionsMessage(event.message))
      || (event.type === "error" && (event.code === "max_sessions" || isMaxSessionsMessage(event.message)));
    const displayEvent = event.type === "warning" && isMaxSessionsMessage(event.message)
      ? {...event, message: "本次请求未启动：推理槽位已满，请稍后重试。"}
      : event.type === "error" && (event.code === "max_sessions" || isMaxSessionsMessage(event.message))
        ? {...event, code: "max_sessions", message: "本次请求未启动：推理槽位已满，请稍后重试。"}
        : event;
    if (event.type === "progress") setProgressHistory((current) => [...current.slice(-4_999), event]);
    setEvents((current) => [...current.slice(-199), displayEvent]);
    if (maxSessions) {
      setOutputIssue("本次请求未启动：引擎当前推理槽位已满，请稍后重试。");
      setBusy(false);
      setPaused(false);
      runRef.current = null;
      // Drop a stale response/socket after the capacity rejection. The SDK
      // settles its active run before closing, so connectAndPlay cannot hang.
      if (clientRef.current?.snapshot().state === "responding") {
        clientRef.current.close();
        clientRef.current = null;
      }
      return;
    }
    if (event.type === "audio") {
      setReceivedSamples(Number(event.endSample));
      setAudioEnvelope((current) => appendAudioEnvelope(current, event.pcm, event.startSample, event.endSample));
      setTiming((current) => current.firstAudio ? current : {...current, firstAudio: performance.now() - requestStartedAt.current});
      if (event.server?.ttft_ms !== undefined) {
        setServerTiming((current) => ({...current, ttft: event.server?.ttft_ms ?? current.ttft}));
      }
      const collector = collectorRef.current;
      collector?.append(event.pcm);
      if (collector?.snapshot().limitReached && !wavLimitWarned.current) {
        wavLimitWarned.current = true;
        setEvents((current) => [...current.slice(-199), {
          type: "warning",
          message: "WAV 收集达到 32 MiB 上限；实时播放不受影响",
        }]);
      }
      if (!playbackFailed.current) {
        try {
          playerRef.current?.enqueue(event.pcm, synthesisRateRef.current, event.startSample, event.endSample);
        } catch (cause) {
          playbackFailed.current = true;
          const message = `浏览器实时播放缓冲失败：${String(cause)}；完整音频仍会保留为 WAV。`;
          setOutputIssue(message);
          setEvents((current) => [...current.slice(-199), {
            type: "error", code: "audio_playback_buffer", message,
          }]);
        }
      }
    } else if (event.type === "response_started") {
      setTiming((current) => ({...current, ttfb: performance.now() - requestStartedAt.current}));
    } else if (event.type === "completed" || event.type === "cancelled" || event.type === "error") {
      if (event.type === "completed") {
        setUsage(event.usage ?? {});
        setServerTiming({
          ttft: event.server?.ttft_ms ?? 0,
          total: event.server?.total_ms ?? 0,
          queueWait: event.server?.engine_queue_wait_ms ?? 0,
          prefill: event.server?.engine_prefill_ms ?? 0,
          dequeueToRaw: event.server?.first_text_dequeue_to_first_raw_audio_ms ?? 0,
          rawToEffective: event.server?.first_raw_to_first_effective_audio_ms ?? 0,
          prefixTrimmed: event.server?.prefix_trimmed_ms ?? 0,
          prefixApplied: event.server?.prefix_trim_applied ?? false,
          vad: event.server?.vad_strategy ?? "",
        });
        const collector = collectorRef.current;
        if (
          collector?.snapshot().samples === 0
          && event.server?.prefix_trim_applied
          && event.server.vad_strategy
          && event.server.vad_strategy !== VadStrategy.Disabled
        ) {
          const message = `输出 VAD（${event.server.vad_strategy}）过滤了整段音频；请降低 Begin threshold 或关闭 VAD 后重试。`;
          setOutputIssue(message);
          setEvents((current) => [...current.slice(-199), {type: "warning", message}]);
        }
      }
      streamGeneration.current += 1;
      setTiming((current) => ({...current, total: performance.now() - startedAt.current}));
      if (!playbackFailed.current) {
        try {
          playerRef.current?.flush();
        } catch (cause) {
          playbackFailed.current = true;
          const message = `浏览器播放尾帧提交失败：${String(cause)}；完整音频仍会保留为 WAV。`;
          setOutputIssue(message);
          setEvents((current) => [...current.slice(-199), {
            type: "error", code: "audio_playback_flush", message,
          }]);
        }
      }
      const collector = collectorRef.current;
      if (collector && collector.snapshot().samples > 0) {
        const url = URL.createObjectURL(collector.toBlob());
        downloadUrlRef.current = url;
        setDownloadUrl(url);
      }
      setBusy(false);
      setPaused(false);
      runRef.current = null;
    }
  }

  async function togglePause() {
    if (paused) await playerRef.current?.resume();
    else await playerRef.current?.pause();
    setPaused(!paused);
  }

  const availableTasks = (caps?.tasks ?? [])
    .filter((value): value is SynthesisTask => Object.values(SynthesisTask).includes(value as SynthesisTask));
  const vadStrategies = (caps?.output_policy.vad_strategies ?? [])
    .filter((value): value is VadStrategy => Object.values(VadStrategy).includes(value as VadStrategy));
  const pcmFormats = (caps?.audio_formats ?? [])
    .filter((format) => format.encoding === AudioEncoding.PcmS16Le);
  const fullTextAvailable = Boolean(caps?.input_modes?.includes("full_text"));
  const longTextAvailable = Boolean(caps?.input_modes?.includes("long_segment"));
  const incrementalAvailable = Boolean(caps?.input_modes?.includes("token"));
  const canSynthesize = Boolean(
    loaded
    && caps
    && availableTasks.includes(task)
    && pcmFormats.some((format) => format.sample_rate === sampleRate)
    && vadStrategies.includes(vad)
    && (inputMode === "full" ? fullTextAvailable : inputMode === "long" ? longTextAvailable : incrementalAvailable),
  );
  const audioDuration = receivedSamples / synthesisSampleRate;
  const textProgress = latestProgressEvent(
    progressHistory.filter((event): event is ProgressEvent => event.type === "progress"),
  );
  const selectedTaskStatus = caps?.task_status.find((status) => status.task === task);

  function selectVadStrategy(strategy: VadStrategy) {
    setVad(strategy);
    if (strategy === VadStrategy.Disabled) return;
    const tuning = defaultVadTuning(strategy);
    setVadChunkMs(tuning.vadChunkMs);
    setVadBeginThreshold(tuning.vadBeginThreshold);
    setVadBeginCount(tuning.vadBeginCount);
    setVadEndThreshold(tuning.vadEndThreshold);
    setVadEndCount(tuning.vadEndCount);
    setVadStartMarginMs(tuning.vadStartMarginMs);
  }

  async function selectReference(file: File | undefined) {
    if (!file) return setReference(null);
    const maxBytes = caps?.reference.max_bytes ?? 0;
    if (maxBytes && file.size > maxBytes) throw new RangeError(`参考音频超过实例公布的 ${(maxBytes / 1024 / 1024).toFixed(1)} MiB 上限`);
    const signature = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    if (!isRiffWave(signature)) throw new TypeError("当前实例仅接受未压缩 RIFF/WAV 参考音频");
    const duration = await referenceDuration(file);
    if (caps?.reference.max_duration_sec && duration > caps.reference.max_duration_sec) {
      throw new RangeError(`参考音频 ${duration.toFixed(1)}s 超过实例公布的 ${caps.reference.max_duration_sec}s 上限`);
    }
    setReference({name: file.name, audioBase64: await fileToBase64(file)});
  }

  async function toggleRecording() {
    if (recording) {
      await finishRecording();
      return;
    }
    if (!caps?.reference.available) return;
    const reportRecordingError = (cause: unknown) => setEvents((current) => [...current, {
      type: "error", code: "reference_recording", message: String(cause),
    }]);
    const recorder = await startReferenceRecorder(
      caps.reference.max_bytes,
      () => void finishRecording().catch(reportRecordingError),
    );
    recorderRef.current = recorder;
    if (caps.reference.max_duration_sec > 0) {
      recordingTimerRef.current = window.setTimeout(
        () => void finishRecording().catch(reportRecordingError),
        caps.reference.max_duration_sec * 1000,
      );
    }
    setRecording(true);
  }

  async function finishRecording() {
    const recorder = recorderRef.current;
    if (!recorder) return;
    recorderRef.current = null;
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    recordingTimerRef.current = null;
    setRecording(false);
    await selectReference(await recorder.stop());
  }

  return <>
    <section className={`hero experience-hero${busy ? " is-live" : ""}`}>
      <div className="hero-copy">
        <p className="eyebrow"><span/>CURRENT INSTANCE · PUBLIC REALTIME</p>
        <h1 aria-label="让文字，即刻成为声音。">让文字，<br/><em>即刻成为声音。</em></h1>
        <p className="hero-intro">写下一句话，选择声音，然后直接听见当前实例的真实输出。参数、播放与诊断都留在同一张工作台上。</p>
        <div className="instance-facts" aria-label="当前实例信息">
          <span><small>MODEL</small>{caps?.model || "正在读取"}</span>
          <span><small>ENGINE</small>{caps?.engine_version || "—"}</span>
          <span><small>OUTPUT</small>{pcmFormats.length ? `${sampleRate / 1000} kHz · PCM16` : "等待能力"}</span>
        </div>
      </div>
      <div className="signal-stage">
        <div className="signal-stage-head"><span>SIGNAL PATH</span><strong>{busy ? "SYNTHESIZING" : caps ? "READY" : "CONNECTING"}</strong></div>
        <div className="signal-copy" aria-hidden="true"><span>TEXT</span><i/><span>VOICE</span><i/><span>PCM</span></div>
        <SignalRibbon active={busy}/>
        <div className="signal-stage-foot"><span>{task.replaceAll("_", " ")}</span><span>{caps?.protocols.openai_realtime.base || "openai realtime"}</span></div>
      </div>
    </section>

    <EngineStatus capabilities={caps} events={events} busy={busy} />

    <CursorProgressPanel text={synthesisText || text} events={[...progressHistory, ...events.filter((event) => event.type !== "progress")]} capabilities={caps} busy={busy} playback={playback} />

    <div className="experience-workbench">
      <div className="control-stack">
        <section className="panel input-panel">
          <div className="panel-heading"><div><p className="panel-kicker">VOICE SOURCE</p><h2>输入与声音</h2></div><p>决定要说什么，以及由谁来表达。</p></div>
          <label className="script-field"><span>合成文本</span><textarea className="script-input" value={text} onChange={(event) => setText(event.target.value)} maxLength={10_000} /></label>
          <div className="grid controls">
            <label>任务<select disabled={!caps || availableTasks.length === 0} value={task} onChange={(event) => setTask(event.target.value as SynthesisTask)}>
              {availableTasks.map((value) => <option key={value}>{value}</option>)}</select></label>
            <label>说话人{caps?.speakers?.length
              ? <select value={speaker} onChange={(event) => setSpeaker(event.target.value)}>{caps.speakers.map((value) => <option key={value}>{value}</option>)}</select>
              : <input disabled value="" placeholder={caps ? "当前实例未公布说话人" : "等待 capabilities"} />}</label>
            <label>语言{caps?.languages?.length
              ? <select value={language} onChange={(event) => setLanguage(event.target.value)}>{caps.languages.map((value) => <option key={value}>{value}</option>)}</select>
              : <input disabled value="" placeholder={caps ? "当前实例未公布语言" : "等待 capabilities"} />}</label>
            <label>输入方式<select disabled={!fullTextAvailable && !longTextAvailable && !incrementalAvailable} value={inputMode} onChange={(event) => setInputMode(event.target.value as "full" | "long" | "incremental")}>
              {fullTextAvailable && <option value="full">完整文本</option>}{longTextAvailable && <option value="long">长文本</option>}{incrementalAvailable && <option value="incremental">模拟 LLM 增量</option>}</select></label>
            <label>输出格式<select disabled={pcmFormats.length === 0} value={sampleRate} onChange={(event) => setSampleRate(Number(event.target.value))}>
              {pcmFormats.map((format) => <option key={format.sample_rate} value={format.sample_rate}>{format.sample_rate} Hz · PCM16</option>)}</select></label>
          </div>
          {(task === SynthesisTask.VoiceDesign || task === SynthesisTask.CustomVoice) && <label className="wide-field">声音指令<input value={instruct} onChange={(event) => setInstruct(event.target.value)} placeholder="例如：温暖、沉稳、语速适中" /></label>}
          {caps?.reference.available && <div className="reference-tools"><label>参考音频<input type="file" accept={caps.reference.mime_types.join(",")} onChange={(event) => void selectReference(event.target.files?.[0]).catch((cause) => setEvents((current) => [...current, {type: "error", code: "reference", message: String(cause)}]))}/><small>{reference?.name || `WAV · 最长 ${caps.reference.max_duration_sec}s · ${(caps.reference.max_bytes / 1024 / 1024).toFixed(1)} MiB`}</small></label>
            {typeof AudioContext !== "undefined" && navigator.mediaDevices && "getUserMedia" in navigator.mediaDevices && <button type="button" onClick={() => void toggleRecording().catch((cause) => setEvents((current) => [...current, {type: "error", code: "microphone", message: String(cause)}]))}><Mic size={15}/>{recording ? "停止并使用录音" : "录制 WAV 参考音频"}</button>}</div>}
          {caps?.reference.available && reference && <div className="grid controls advanced">
            <label>参考文本<input value={referenceText} onChange={(event) => setReferenceText(event.target.value)} placeholder="与参考音频一致的文本（ICL）"/></label>
            <label>克隆方式<select value={String(xVectorOnly)} onChange={(event) => setXVectorOnly(event.target.value === "true")}>
              <option value="false" disabled={!caps.reference.icl_available}>ICL clone{!caps.reference.icl_available ? "（不可用）" : ""}</option>
              <option value="true" disabled={!caps.reference.speaker_encoder_available}>x-vector only{!caps.reference.speaker_encoder_available ? "（不可用）" : ""}</option>
            </select></label>
          </div>}
          {caps && !caps.reference.available && <p className="hint inline-note">参考音频未开放 · {caps.reference.reason}</p>}
          {selectedTaskStatus?.stability === "experimental" && <p className="alert">当前任务属于工程预览能力；可用性与限制以本实例 capabilities 和“已知限制”文档为准。</p>}
        </section>

        <section className="panel policy-panel">
          <div className="panel-heading"><div><p className="panel-kicker">OUTPUT SHAPING</p><h2>输出策略</h2></div><p>控制交付节奏与输出端静音裁剪。</p></div>
          <div className="grid controls">
            <label>输出 VAD<select disabled={vadStrategies.length === 0} value={vad} onChange={(event) => selectVadStrategy(event.target.value as VadStrategy)}>
              {vadStrategies.map((value) => <option key={value}>{value}</option>)}</select></label>
            <label>交付模式<select value={delivery} onChange={(event) => setDelivery(event.target.value as DeliveryPolicy)}>
              <option value={DeliveryPolicy.Guarded}>guarded</option><option value={DeliveryPolicy.Firehose}>firehose</option></select></label>
            <NumberInput label="交付窗口 (ms)" value={deliveryWindowMs} min={100} max={10_000} onChange={setDeliveryWindowMs} disabled={delivery !== DeliveryPolicy.Guarded}/>
            <NumberInput label="输出分块 (ms)" value={outputChunkMs} min={0} max={10_000} onChange={setOutputChunkMs}/>
            <label>文本进度<select value={String(emitTextEvents)} onChange={(event) => setEmitTextEvents(event.target.value === "true")}><option value="true">开启</option><option value="false">关闭</option></select></label>
          </div>
          {vad !== VadStrategy.Disabled && <div className="grid controls advanced">
            <NumberInput label="VAD chunk (ms)" value={vadChunkMs} min={1} max={1000} onChange={setVadChunkMs}/>
            <NumberInput label="Begin threshold" value={vadBeginThreshold} min={0} max={1} step={0.05} onChange={setVadBeginThreshold}/>
            <NumberInput label="Begin count" value={vadBeginCount} min={1} max={1000} onChange={setVadBeginCount}/>
            <NumberInput label="End threshold" value={vadEndThreshold} min={0} max={1} step={0.05} onChange={setVadEndThreshold}/>
            <NumberInput label="End count" value={vadEndCount} min={1} max={1000} onChange={setVadEndCount}/>
            <NumberInput label="Start margin (ms)" value={vadStartMarginMs} min={0} max={10_000} onChange={setVadStartMarginMs}/>
          </div>}
          <p className="hint inline-note">VAD 只过滤 TTS 输出静音，不参与麦克风 endpointing；切换算法会载入对应推荐阈值，仍可继续微调。</p>
        </section>
      </div>

      <aside className="playback-stack">
        <section className={`panel action-panel${busy ? " is-live" : ""}`}>
          <div className="panel-heading stage-heading"><div><p className="panel-kicker">LISTENING STAGE</p><h2>监听台</h2></div><span className={`monitor-status${outputIssue ? " has-warning" : ""}`}><i/>{busy ? paused ? "已暂停" : "正在合成" : downloadUrl ? "可重放" : outputIssue ? "无有效音频" : "等待输入"}</span></div>
          <div className="monitor-display"><div className="monitor-head"><span>OUTPUT MONITOR · FRAME / TEXT ALIGNMENT</span><span>{(receivedSamples ? synthesisSampleRate : sampleRate) / 1000} kHz</span></div><PlaybackWaveform envelope={audioEnvelope} playedSample={playback.played}/><AudioTextCursor text={synthesisText || text} events={[...progressHistory, ...events.filter((event) => event.type !== "progress")]} playback={playback} sampleRate={synthesisSampleRate} showTrack={!downloadUrl}/>
            {outputIssue
              ? <p className="monitor-empty monitor-warning" role="status">{outputIssue}</p>
              : textProgress?.type === "progress" ? <p className="text-progress">{textProgress.text || "游标已对齐文本与音频"}<small>sample {textProgress.sample.toString()} · {String(textProgress.meta?.progress_basis ?? "progress")}</small></p> : <p className="monitor-empty">合成后，音频波形与文本进度会出现在这里。</p>}</div>
          <div className="actions stage-actions">
            <button className="primary" disabled={!canSynthesize || busy || !text.trim()} onClick={() => void connectAndPlay()}><Play size={17}/>合成并播放</button>
            <button disabled={!busy} onClick={() => void togglePause()}><Pause size={17}/>{paused ? "继续" : "暂停"}</button>
            <button className="icon-action" aria-label="取消合成" disabled={!busy} onClick={() => runRef.current?.cancel()}><Square size={16}/></button>
          </div>
          <div className="monitor-controls">
            <label>监听音量<input type="range" min="0" max="1" step="0.05" value={volume} onChange={(event) => {
              const next = Number(event.target.value); setVolume(next); playerRef.current?.setVolume(next);
            }}/></label>
            {outputDevices.length > 0 && <label>输出设备<select value={outputDevice} onChange={(event) => {
              setOutputDevice(event.target.value); void playerRef.current?.setOutputDevice(event.target.value);
            }}><option value="">系统默认扬声器</option>{outputDevices.map((device) => <option value={device.deviceId} key={device.deviceId}>{device.label || `扬声器 ${device.deviceId.slice(0, 6)}`}</option>)}</select></label>}
          </div>
          {downloadUrl && <div className="recording-result"><MediaPlayer className="media-player--dark" src={downloadUrl} label="合成音频" volume={volume} onPlay={() => playerRef.current?.pause()} onPositionChange={(seconds) => {
            const sample = BigInt(Math.max(0, Math.floor(seconds * synthesisSampleRate)));
            setPlayback((current) => ({...current, played: sample}));
          }}/><a className="button download-action" href={downloadUrl} download="qwen3tts.wav"><Download size={17}/>下载 WAV</a></div>}
        </section>
      </aside>
    </div>

    <section className="panel diagnostics-panel">
      <div className="panel-heading"><div><p className="panel-kicker">SIGNAL TELEMETRY</p><h2>实时诊断</h2></div><p>从请求发出到扬声器消费，按同一条时间线观察。</p></div>
      <p className="diagnostics-note">同一条请求时间线：先统计客户端连接/播放器准备，再从请求发出观察响应开始、首个音频和首个可听；服务端 TTFT 单独按 response.create → 首个原始音频计算。</p>
      <div className="metrics">
        <Metric label="客户端准备（连接 / 播放器）" value={formatMs(timing.setup)} />
        <Metric label="客户端响应开始（请求→响应）" value={formatMs(timing.ttfb)} />
        <Metric label="服务端 TTFT（响应→原始音频）" value={formatMs(serverTiming.ttft)} />
        <Metric label="客户端 TTFT（请求→首音频）" value={formatMs(timing.firstAudio)} />
        <Metric label="客户端首可听（请求→扬声器）" value={formatMs(timing.firstAudible)} />
        <Metric label="服务端拆解（队列 / Prefill）" value={`${formatMs(serverTiming.queueWait)} / ${formatMs(serverTiming.prefill)}`} />
        <Metric label="总耗时" value={formatMs(timing.total)} />
        <Metric label="音频时长" value={`${audioDuration.toFixed(2)} s`} />
        <Metric label="客户端 RTF" value={audioDuration > 0 && timing.total > 0 ? (timing.total / 1000 / audioDuration).toFixed(3) : "—"} />
        <Metric label="播放 sample" value={playback.played.toString()} />
        <Metric label="Buffer lead" value={`${Number(playback.buffered - playback.played) / (receivedSamples ? synthesisSampleRate : sampleRate) * 1000 | 0} ms`} />
        <Metric label="播放队列" value={playback.queuedFrames} />
        <Metric label="Underrun" value={playback.underruns} />
        <Metric label="Usage" value={Object.keys(usage).length > 0
          ? Object.entries(usage).map(([key, value]) => `${key}=${value}`).join(" · ")
          : "—"} />
        <Metric label="VAD trimming" value={serverTiming.prefixApplied
          ? `${serverTiming.prefixTrimmed.toFixed(1)} ms · ${serverTiming.vad || "active"}`
          : serverTiming.vad ? `0 ms · ${serverTiming.vad}` : "—"} />
      </div>
      <div className="event-log">{events.length === 0
        ? <p>尚无事件。点击“合成并播放”后，这里会记录连接、音频、恢复与告警。</p>
        : events.slice(-12).map((event, index) => <code key={index}>{event.type}{event.type === "warning" || event.type === "error" ? ` · ${event.message}` : ""}</code>)}</div>
    </section>
    {loaded && <section className="experience-lab-dock" aria-label="实验室">
      <div className="experience-lab-intro">
        <p className="eyebrow">ENGINEERING LAB · 02 / 03</p>
        <h2>把实时体验继续拆开看</h2>
        <p>文本进度、LLM 模拟 PK 与并发压测都使用上方同一个实例。</p>
      </div>
      <ExperimentLab loaded={loaded} embedded capabilities={caps} settings={settings} />
    </section>}
  </>;
}

function EngineStatus({capabilities, events, busy}: {
  capabilities: Capabilities | null;
  events: TTSEvent[];
  busy: boolean;
}) {
  const native = capabilities?.native_cursor;
  const speechState = capabilities?.speech_state;
  const progressEvents = events.filter((event): event is ProgressEvent => event.type === "progress");
  // Native is the session route once admitted. A conservative EMA sample can
  // appear before lookahead is valid or during final flush; it must not make
  // the UI misreport an otherwise native session as EMA.
  const orderedProgress = sortProgressEvents(progressEvents);
  const progress = [...orderedProgress].reverse().find(
    (event) => String(event.meta?.progress_basis ?? "") === "native_cursor_v1",
  ) ?? orderedProgress.at(-1);
  const basis = progress?.type === "progress" ? String(progress.meta?.progress_basis ?? "") : "";
  const sessionMode = basis === "native_cursor_v1" ? "native" : basis === "ema_frame_ratio_v1" ? "ema" : "unknown";
  const modeLabel = sessionMode === "native" ? "原生游标" : sessionMode === "ema" ? "EMA" : busy ? "等待进度" : "未开始";
  const modeClass = sessionMode === "native" ? "is-good" : sessionMode === "ema" ? "is-warn" : "";

  return <section className="engine-status" aria-label="引擎能力状态" data-testid="engine-status">
    <div className="engine-status-heading"><span className="panel-kicker">ENGINE ROUTE</span><span>{capabilities?.engine_version || "能力等待中"}</span></div>
    <div className="engine-status-grid">
      <StatusItem icon={<Cpu size={15}/>} label="游标图" value={native?.graph_enabled ? "已加载" : "标准图"} detail={cursorCapabilityDetail(native)} />
      <StatusItem icon={<Activity size={15}/>} label="本次进度" value={modeLabel} detail={progress?.type === "progress" ? String(progress.meta?.progress_quality ?? "已收到事件") : "以事件为准"} className={modeClass} />
      <StatusItem icon={<BadgeCheck size={15}/>} label="状态继承" value={speechState?.supported ? "可用" : "未启用"} detail={speechState?.supported ? "runtime admitted" : speechState?.reason || "未配置"} className={speechState?.supported ? "is-good" : "is-muted"} />
    </div>
  </section>;
}

function cursorCapabilityDetail(native: Capabilities["native_cursor"]): string {
  if (!native?.graph_enabled) return "未加载，使用 EMA / disabled";
  if (native.progress_available) return "native 可用";
  const reason = native.reason;
  const labels: Record<string, string> = {
    cursor_graph_disabled: "游标图未启用",
    cursor_head_missing: "游标 head 缺失",
    cursor_labelizer_unavailable: "TN / Label Plan 不可用",
    malformed_native_cursor_capability: "能力声明格式错误",
    native_cursor_release_evidence_incomplete: "发布验收证据未完成",
  };
  return `未准入 · ${labels[reason ?? ""] ?? reason ?? "TN / Label Plan 桥接未连接"}`;
}

function CursorProgressPanel({text, events, capabilities, busy, playback}: {
  text: string;
  events: TTSEvent[];
  capabilities: Capabilities | null;
  busy: boolean;
  playback: {played: bigint};
}) {
  const progressEvents = events.filter((event): event is ProgressEvent => event.type === "progress");
  const orderedProgress = sortProgressEvents(progressEvents);
  const textLength = Array.from(text).length;
  const native = orderedProgress.filter((event) => String(event.meta?.progress_basis ?? "") === "native_cursor_v1");
  const ema = orderedProgress.filter((event) => String(event.meta?.progress_basis ?? "") === "ema_frame_ratio_v1");
  const generated = orderedProgress.at(-1);
  const heard = [...orderedProgress].reverse().find((event) => progressSampleBigInt(event) <= playback.played);
  const latest = playback.played > 0n ? heard : undefined;
  const latestIndex = latest ? orderedProgress.lastIndexOf(latest) : -1;
  const generatedIndex = generated ? orderedProgress.length - 1 : -1;
  const rawEnd = latestIndex >= 0
    ? monotonicRawEnd(orderedProgress, latestIndex, textLength)
    : 0;
  const progress = textLength ? rawEnd / textLength : 0;
  const nativeReady = Boolean(capabilities?.native_cursor?.progress_available);
  return <section className="cursor-panel" aria-label="文本游标进度">
    <div className="cursor-panel-head"><div><p className="panel-kicker">TEXT CURSOR · 01 / 03</p><h2>看见每个字何时被说出来</h2></div>
      <span className={`cursor-route ${nativeReady ? "is-native" : ""}`}><i/>{nativeReady ? "NATIVE CURSOR READY" : "EMA FALLBACK"}</span></div>
    <p className="cursor-panel-copy">高亮跟随正在播放的声音。实色表示已听到的位置，浅色表示已生成的文本。</p>
    <div className="cursor-text-stage"><div className="cursor-text-meta"><span>RAW TEXT</span><span>{rawEnd.toFixed(1)}/{textLength} codepoints</span></div>
      <div className="cursor-text" aria-live="polite">{Array.from(text).map((character, index) => <span key={`${index}-${character}`} className={index < Math.floor(rawEnd) ? "is-read" : index === Math.floor(rawEnd) ? "is-current" : ""}>{character === " " ? " " : character}</span>)}</div>
      <ProgressTrack className="cursor-progress-track" label="文本已播放进度" value={progress} buffered={generatedIndex >= 0 ? monotonicRawEnd(orderedProgress, generatedIndex, textLength) / Math.max(1, textLength) : 0}/>
    </div>
    <div className="cursor-trajectory"><div className="cursor-trajectory-head"><span>TRAJECTORY</span><span>{busy ? "LIVE" : progressEvents.length ? "CAPTURED" : "WAITING FOR AUDIO"}</span></div>
      <CursorChart native={native} ema={ema} textLength={textLength}/>
      <div className="cursor-legend"><span className="native-key"><i/>原生游标 {native.length ? `${native.length} points` : "等待"}</span><span className="ema-key"><i/>EMA {ema.length ? `${ema.length} points` : "降级时显示"}</span><span>已听 {latestIndex >= 0 ? `${rawEnd.toFixed(1)}/${textLength}` : "—"} · 生成 {generatedIndex >= 0 ? `${monotonicRawEnd(orderedProgress, generatedIndex, textLength).toFixed(1)}/${textLength}` : "—"}</span></div>
    </div>
  </section>;
}

function CursorChart({native, ema, textLength}: {native: ProgressEvent[]; ema: ProgressEvent[]; textLength: number}) {
  const points = [...native, ...ema];
  const maxSample = Math.max(1, ...points.map((event) => progressSample(event)));
  const timeline = sortProgressEvents(points);
  const path = (series: ProgressEvent[]) => series.map((event, index) => {
    const x = 8 + progressSample(event) / maxSample * 484;
    const timelineIndex = timeline.indexOf(event);
    const highWater = timelineIndex < 0
      ? progressRawEnd(event, textLength)
      : monotonicRawEnd(timeline, timelineIndex, textLength);
    const y = 72 - highWater / Math.max(1, textLength) * 58;
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return <svg className="cursor-chart" viewBox="0 0 500 84" role="img" aria-label="原生游标和 EMA 文本进度轨迹"><path className="chart-grid" d="M8 14H492M8 43H492M8 72H492"/>{native.length > 0 && <path className="chart-native" d={path(native)}/>} {ema.length > 0 && <path className="chart-ema" d={path(ema)}/>}<text x="8" y="82">0</text><text x="476" y="82">audio samples</text></svg>;
}

function StatusItem({icon, label, value, detail, className = ""}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  className?: string;
}) {
  return <div className={`engine-status-item ${className}`}><span className="engine-status-icon">{icon}</span><span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span></div>;
}

const SIGNAL_PROFILE = [18, 30, 46, 26, 62, 84, 44, 34, 72, 96, 58, 38, 76, 52, 30, 68, 90, 48, 28, 58, 78, 40, 66, 88, 50, 32, 70, 54, 36, 62, 44];

function SignalRibbon({active}: {active: boolean}) {
  return <div className={`signal-ribbon${active ? " active" : ""}`} aria-hidden="true">
    <div className="signal-axis"><span>0</span><span>TEXT → AUDIO</span><span>1.0</span></div>
    <div className="signal-bars">{SIGNAL_PROFILE.map((level, index) =>
      <i key={index} style={{"--signal-level": `${level}%`, "--signal-delay": `${index * -37}ms`} as React.CSSProperties}/>)}</div>
    <div className="signal-baseline"/>
  </div>;
}

function Metric({label, value}: {label: string; value: string | number}) { return <div><small>{label}</small><strong>{value}</strong></div>; }

function NumberInput({label, value, min, max, step = 1, disabled = false, onChange}: {
  label: string; value: number; min: number; max: number; step?: number; disabled?: boolean; onChange: (value: number) => void;
}) {
  return <label>{label}<input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
    onChange={(event) => onChange(Number(event.target.value))}/></label>;
}

function AudioTextCursor({text, events, playback, sampleRate, showTrack}: {text: string; events: TTSEvent[]; playback: {played: bigint; buffered: bigint}; sampleRate: number; showTrack: boolean}) {
  const displayedPlayedRatio = useRef(0);
  const displayedBufferedRatio = useRef(0);
  const textLength = Array.from(text).length;
  const anchors = sortProgressEvents(events.filter((event): event is ProgressEvent => event.type === "progress"));
  let progressIndex = -1;
  for (let index = anchors.length - 1; index >= 0; index -= 1) {
    const anchor = anchors[index];
    if (anchor && progressSampleBigInt(anchor) <= playback.played) {
      progressIndex = index;
      break;
    }
  }
  const progress = progressIndex >= 0 ? anchors[progressIndex] : undefined;
  const audio = events.filter((event): event is Extract<TTSEvent, {type: "audio"}> => event.type === "audio");
  const generatedEnd = audio.reduce(
    (highWater, event) => event.endSample > highWater ? event.endSample : highWater,
    0n,
  );
  const sourceSample = playback.played;
  if (audio.length === 0 && anchors.length === 0) {
    // A new live request starts with an empty timeline. Reset the display
    // high-water while retaining monotonicity within the active request.
    displayedPlayedRatio.current = 0;
    displayedBufferedRatio.current = 0;
  }
  // During live synthesis the denominator grows as more PCM arrives. A raw
  // fraction would therefore move backwards even though playback is
  // monotonic. Keep the rendered high-water conservative until the request
  // completes; the text cursor still exposes the exact sample-to-text route.
  displayedPlayedRatio.current = monotonicSampleRatio(
    displayedPlayedRatio.current,
    playback.played,
    generatedEnd,
  );
  const bufferedRatio = monotonicSampleRatio(
    displayedBufferedRatio.current,
    playback.buffered,
    generatedEnd,
  );
  displayedBufferedRatio.current = Math.max(
    displayedBufferedRatio.current,
    bufferedRatio,
    displayedPlayedRatio.current,
  );
  const rawEnd = progress ? monotonicRawEnd(anchors, progressIndex, textLength) : 0;
  const route = progress ? String(progress.meta?.progress_basis ?? "").replace("_v1", "") : "waiting";
  return <div className="audio-text-cursor"><div className="audio-cursor-labels"><span>TEXT POSITION <b>{rawEnd.toFixed(1)}/{textLength}</b></span><span>PCM SAMPLE <b>{sourceSample.toString()}</b></span></div>
    {showTrack && <ProgressTrack className="audio-text-line progress-track--dark" label="监听音频播放进度" value={displayedPlayedRatio.current} buffered={displayedBufferedRatio.current}/>}
    <div className="audio-text-preview">{Array.from(text).map((character, index) => <span key={`${index}-${character}`} className={index < Math.floor(rawEnd) ? "is-read" : index === Math.floor(rawEnd) ? "is-current" : ""}>{character === " " ? " " : character}</span>)}</div>
    <div className="audio-cursor-foot"><span>{route} · codec frame {String(progress?.meta?.source_frame_end ?? "—")}</span><span>{sampleRate ? `${(Number(sourceSample) / sampleRate).toFixed(2)}s` : "—"} played</span></div>
  </div>;
}

function formatMs(value: number): string { return value > 0 ? `${Math.round(value)} ms` : "—"; }

function isMaxSessionsMessage(message: string): boolean {
  return /max[\s_-]+sessions/i.test(message) || /推理槽位已满/.test(message);
}

function isRiffWave(bytes: Uint8Array): boolean {
  return bytes.length >= 12
    && String.fromCharCode(...bytes.subarray(0, 4)) === "RIFF"
    && String.fromCharCode(...bytes.subarray(8, 12)) === "WAVE";
}

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

async function referenceDuration(file: File): Promise<number> {
  const context = new AudioContext();
  try {
    return (await context.decodeAudioData(await file.arrayBuffer())).duration;
  } finally {
    await context.close();
  }
}

async function streamText(
  run: IncrementalSynthesisRun,
  text: string,
  generation: number,
  generationRef: React.MutableRefObject<number>,
) {
  const pieces = text.match(/.{1,6}/gu) ?? [text];
  for (const piece of pieces) {
    if (generation !== generationRef.current) return;
    run.append(piece);
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  if (generation === generationRef.current) run.commit();
}

function routeFromHash(): Route {
  const route = window.location.hash.replace(/^#\/?/, "") as Route;
  if (String(route).startsWith("docs/")) return "docs";
  return ["experience", "sdk", "docs"].includes(route) ? route : "experience";
}
