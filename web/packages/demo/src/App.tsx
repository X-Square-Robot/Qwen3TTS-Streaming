import {
  BookOpen,
  Download,
  FlaskConical,
  Headphones,
  Mic,
  Package,
  Pause,
  Play,
  Square,
} from "lucide-react";
import {useEffect, useMemo, useRef, useState} from "react";
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
import {SdkPage} from "./SdkPage";
import {DEFAULT_DEMO_SETTINGS, type DemoSynthesisSettings} from "./demo-settings";
import {startReferenceRecorder, type ReferenceRecorder} from "./reference-recorder";

type Route = "experience" | "sdk" | "docs" | "lab";

export function App() {
  const [loaded, setLoaded] = useState<LoadedDemoConfig | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [route, setRoute] = useState<Route>(routeFromHash());
  const [docsOnly, setDocsOnly] = useState(false);
  const [labReachable, setLabReachable] = useState(false);
  const [exampleSettings, setExampleSettings] = useState<DemoSynthesisSettings>(DEFAULT_DEMO_SETTINGS);

  useEffect(() => {
    loadDemoConfig().then(setLoaded).catch(() => setDocsOnly(true));
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!loaded?.config.lab.available || !loaded.config.lab.url) {
      setLabReachable(false);
      return;
    }
    const controller = new AbortController();
    const base = new URL(loaded.config.lab.url, loaded.responseUrl);
    if (!base.pathname.endsWith("/")) base.pathname += "/";
    const timer = window.setTimeout(() => controller.abort(), 2_000);
    fetch(new URL("healthz", base), {signal: controller.signal, cache: "no-store"})
      .then((response) => setLabReachable(response.ok))
      .catch(() => setLabReachable(false))
      .finally(() => window.clearTimeout(timer));
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [loaded]);

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
          {labReachable && nav("lab", "实验", <FlaskConical size={16} />)}
        </nav>
        <span className="release"><i/>{loaded?.config.engine_version || "DEV"}</span>
      </div>
    </header>
    <main className="app-main">
      {docsOnly && <div className="alert">文档只读模式：此 Pages 站点未连接具体 TTS 实例。请在部署实例的 /demo/ 页面进行合成、试听和 SDK 下载。</div>}
      {route === "experience" && <Experience loaded={loaded} onCapabilities={setCapabilities} onSettings={setExampleSettings} />}
      {route === "sdk" && <SdkPage loaded={loaded} capabilities={capabilities} settings={exampleSettings} docsOnly={docsOnly} />}
      {route === "docs" && <DocsPage />}
      {route === "lab" && labReachable && <ExperimentLab loaded={loaded} />}
      {route === "lab" && !labReachable && <p className="alert">工程实验后端未启用或当前不可达。</p>}
    </main>
  </>;
}

function Experience({loaded, onCapabilities, onSettings}: {
  loaded: LoadedDemoConfig | null;
  onCapabilities: (value: Capabilities) => void;
  onSettings: (value: DemoSynthesisSettings) => void;
}) {
  const [text, setText] = useState("你好，欢迎体验 Qwen3 TTS 流式语音合成服务。");
  const [speaker, setSpeaker] = useState("Serena");
  const [language, setLanguage] = useState("auto");
  const [task, setTask] = useState(SynthesisTask.CustomVoice);
  const [inputMode, setInputMode] = useState<"full" | "incremental">("full");
  const [vad, setVad] = useState(VadStrategy.Disabled);
  const [delivery, setDelivery] = useState(DeliveryPolicy.Guarded);
  const [sampleRate, setSampleRate] = useState(24_000);
  const [vadBeginThreshold, setVadBeginThreshold] = useState(0.6);
  const [vadEndThreshold, setVadEndThreshold] = useState(0.35);
  const [vadBeginCount, setVadBeginCount] = useState(5);
  const [vadEndCount, setVadEndCount] = useState(31);
  const [vadChunkMs, setVadChunkMs] = useState(16);
  const [vadStartMarginMs, setVadStartMarginMs] = useState(20);
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
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState("");
  const downloadUrlRef = useRef("");
  const clientRef = useRef<RealtimeTTSClient | null>(null);
  const playerRef = useRef<BrowserAudioPlayer | null>(null);
  const runRef = useRef<SynthesisRun | null>(null);
  const collectorRef = useRef<WavCollector | null>(null);
  const recorderRef = useRef<ReferenceRecorder | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  const wavLimitWarned = useRef(false);
  const startedAt = useRef(0);
  const streamGeneration = useRef(0);
  const [usage, setUsage] = useState<Record<string, number>>({});
  const [timing, setTiming] = useState({ttfb: 0, firstAudio: 0, firstAudible: 0, total: 0});
  const [serverTiming, setServerTiming] = useState({ttft: 0, total: 0, prefixTrimmed: 0, prefixApplied: false, vad: ""});
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
      if (!value.input_modes?.includes("full_text") && value.input_modes?.includes("token")) setInputMode("incremental");
      setXVectorOnly(!value.reference.icl_available && Boolean(value.reference.speaker_encoder_available));
    }).catch((cause) => setEvents((current) => [...current, {type: "error", code: "capabilities", message: String(cause)}]));
  }, [loaded, onCapabilities]);

  async function connectAndPlay() {
    if (!loaded) return;
    setEvents([]);
    setUsage({});
    wavLimitWarned.current = false;
    setBusy(true);
    setTiming({ttfb: 0, firstAudio: 0, firstAudible: 0, total: 0});
    setServerTiming({ttft: 0, total: 0, prefixTrimmed: 0, prefixApplied: false, vad: ""});
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current);
    downloadUrlRef.current = "";
    setDownloadUrl("");
    streamGeneration.current += 1;
    const generation = streamGeneration.current;
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
        maxBufferMs: 5_000,
        onPlaybackProgress: (played, buffered) => {
          runRef.current?.acknowledgePlayback(played, buffered);
          const snapshot = playerRef.current?.snapshot();
          setPlayback({played, buffered, queuedFrames: snapshot?.queuedFrames ?? 0, underruns: snapshot?.underruns ?? 0});
          if (played > 0n) setTiming((current) => current.firstAudible ? current : {...current, firstAudible: performance.now() - startedAt.current});
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
      startedAt.current = performance.now();
      const options = {
        task,
        speaker,
        language,
        ...(instruct ? {instruct} : {}),
        ...(reference ? {reference: {audioBase64: reference.audioBase64, text: referenceText, xVectorOnly}} : {}),
        inputMode: inputMode === "full" ? InputMode.FullText : InputMode.Token,
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
      const run = inputMode === "full"
        ? await client.synthesize(text, options)
        : await client.startIncremental(options);
      runRef.current = run;
      if (inputMode === "incremental") void streamText(run as IncrementalSynthesisRun, text, generation, streamGeneration);
      await run.done;
    } catch (cause) {
      setEvents((current) => [...current, {type: "error", code: "demo_error", message: String(cause)}]);
      setBusy(false);
    }
  }

  function handleEvent(event: TTSEvent) {
    setEvents((current) => [...current.slice(-199), event]);
    if (event.type === "audio") {
      setTiming((current) => current.firstAudio ? current : {...current, firstAudio: performance.now() - startedAt.current});
      const collector = collectorRef.current;
      collector?.append(event.pcm);
      if (collector?.snapshot().limitReached && !wavLimitWarned.current) {
        wavLimitWarned.current = true;
        setEvents((current) => [...current.slice(-199), {
          type: "warning",
          message: "WAV 收集达到 32 MiB 上限；实时播放不受影响",
        }]);
      }
      playerRef.current?.enqueue(event.pcm, sampleRate, event.startSample, event.endSample);
    } else if (event.type === "response_started") {
      setTiming((current) => ({...current, ttfb: performance.now() - startedAt.current}));
    } else if (event.type === "completed" || event.type === "cancelled" || event.type === "error") {
      if (event.type === "completed") {
        setUsage(event.usage ?? {});
        setServerTiming({
          ttft: event.server?.ttft_ms ?? 0,
          total: event.server?.total_ms ?? 0,
          prefixTrimmed: event.server?.prefix_trimmed_ms ?? 0,
          prefixApplied: event.server?.prefix_trim_applied ?? false,
          vad: event.server?.vad_strategy ?? "",
        });
      }
      streamGeneration.current += 1;
      setTiming((current) => ({...current, total: performance.now() - startedAt.current}));
      playerRef.current?.flush();
      const collector = collectorRef.current;
      if (collector && collector.snapshot().samples > 0) {
        const url = URL.createObjectURL(collector.toBlob());
        downloadUrlRef.current = url;
        setDownloadUrl(url);
      }
      setBusy(false);
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
  const incrementalAvailable = Boolean(caps?.input_modes?.includes("token"));
  const canSynthesize = Boolean(
    loaded
    && caps
    && availableTasks.includes(task)
    && pcmFormats.some((format) => format.sample_rate === sampleRate)
    && vadStrategies.includes(vad)
    && (inputMode === "full" ? fullTextAvailable : incrementalAvailable),
  );
  const receivedSamples = useMemo(() => events.reduce((sum, event) => event.type === "audio" ? sum + event.pcm.length : sum, 0), [events]);
  const audioDuration = receivedSamples / sampleRate;
  const textProgress = [...events].reverse().find((event) => event.type === "progress");
  const selectedTaskStatus = caps?.task_status.find((status) => status.task === task);

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
            <label>输入方式<select disabled={!fullTextAvailable && !incrementalAvailable} value={inputMode} onChange={(event) => setInputMode(event.target.value as "full" | "incremental")}>
              {fullTextAvailable && <option value="full">完整文本</option>}{incrementalAvailable && <option value="incremental">模拟 LLM 增量</option>}</select></label>
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
            <label>输出 VAD<select disabled={vadStrategies.length === 0} value={vad} onChange={(event) => setVad(event.target.value as VadStrategy)}>
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
          <p className="hint inline-note">VAD 只过滤 TTS 输出静音，不参与麦克风 endpointing；可选项来自当前实例。</p>
        </section>
      </div>

      <aside className="playback-stack">
        <section className={`panel action-panel${busy ? " is-live" : ""}`}>
          <div className="panel-heading stage-heading"><div><p className="panel-kicker">LISTENING STAGE</p><h2>监听台</h2></div><span className="monitor-status"><i/>{busy ? paused ? "已暂停" : "正在合成" : downloadUrl ? "可重放" : "等待输入"}</span></div>
          <div className="monitor-display"><div className="monitor-head"><span>OUTPUT MONITOR</span><span>{sampleRate / 1000} kHz</span></div><Waveform events={events}/>
            {textProgress?.type === "progress" ? <p className="text-progress">{textProgress.text}<small>sample {textProgress.sample.toString()}</small></p> : <p className="monitor-empty">合成后，音频波形与文本进度会出现在这里。</p>}</div>
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
          {downloadUrl && <div className="recording-result"><audio className="replay" controls src={downloadUrl}/><a className="button download-action" href={downloadUrl} download="qwen3tts.wav"><Download size={17}/>下载 WAV</a></div>}
        </section>
      </aside>
    </div>

    <section className="panel diagnostics-panel">
      <div className="panel-heading"><div><p className="panel-kicker">SIGNAL TELEMETRY</p><h2>实时诊断</h2></div><p>从请求发出到扬声器消费，按同一条时间线观察。</p></div>
      <div className="metrics">
        <Metric label="Client TTFB" value={formatMs(timing.ttfb)} />
        <Metric label="Server TTFT" value={formatMs(serverTiming.ttft)} />
        <Metric label="首个音频（收到）" value={formatMs(timing.firstAudio)} />
        <Metric label="First audible" value={formatMs(timing.firstAudible)} />
        <Metric label="总耗时" value={formatMs(timing.total)} />
        <Metric label="音频时长" value={`${audioDuration.toFixed(2)} s`} />
        <Metric label="客户端 RTF" value={audioDuration > 0 && timing.total > 0 ? (timing.total / 1000 / audioDuration).toFixed(3) : "—"} />
        <Metric label="播放 sample" value={playback.played.toString()} />
        <Metric label="Buffer lead" value={`${Number(playback.buffered - playback.played) / sampleRate * 1000 | 0} ms`} />
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
  </>;
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

function Waveform({events}: {events: TTSEvent[]}) {
  const bars = useMemo(() => {
    const audio = events.filter((event): event is Extract<TTSEvent, {type: "audio"}> => event.type === "audio").slice(-24);
    return audio.map((event) => {
      let peak = 0;
      for (let index = 0; index < event.pcm.length; index += Math.max(1, Math.floor(event.pcm.length / 128))) {
        peak = Math.max(peak, Math.abs(event.pcm[index] ?? 0));
      }
      return Math.max(4, Math.round(peak / 32768 * 64));
    });
  }, [events]);
  return <div className="waveform" aria-label="实时音频波形">{(bars.length ? bars : [4, 4, 4, 4]).map((height, index) => <i key={index} style={{height}}/>)}</div>;
}

function formatMs(value: number): string { return value > 0 ? `${Math.round(value)} ms` : "—"; }

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
  return ["experience", "sdk", "docs", "lab"].includes(route) ? route : "experience";
}
