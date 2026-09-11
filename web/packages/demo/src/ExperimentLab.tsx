import {useEffect, useMemo, useRef, useState} from "react";
import {Headphones, Play, RotateCcw} from "lucide-react";
import {type Capabilities} from "@xmultimodalinteraction/qwen3tts-browser";
import {DEFAULT_DEMO_SETTINGS, type DemoSynthesisSettings} from "./demo-settings";
import type {LoadedDemoConfig} from "./config";
import {concurrencyStats, MAX_CONCURRENCY, safeConcurrency, type LaneSnapshot} from "./lab/experiment-model";
import {createExperimentStartBarrier, discoverExperimentCapabilities, startExperiment, startPkExperiment, type ActiveExperiment, type RunOutput} from "./lab/experiments";
import {MediaPlayer, type MediaPlayerHandle} from "./components/MediaPlayer";
import "./lab/experiments.css";

export interface ExperimentLabProps { loaded: LoadedDemoConfig | null; embedded?: boolean; capabilities?: Capabilities | null; settings?: DemoSynthesisSettings; }

export function ExperimentLab({loaded, embedded = false, capabilities: providedCapabilities, settings = DEFAULT_DEMO_SETTINGS}: ExperimentLabProps) {
  const [text, setText] = useState("你好，这是从上游大模型逐步到达的流式文本。");
  const [chunkDelayMs, setChunkDelayMs] = useState(45); const [chunkSize, setChunkSize] = useState(4);
  const [concurrency, setConcurrency] = useState(128); const [running, setRunning] = useState(false);
  const [pk, setPk] = useState<{streaming?: RunOutput; offline?: RunOutput}>({});
  const [pkPlayedMs, setPkPlayedMs] = useState<Record<"streaming" | "offline", number>>({streaming: 0, offline: 0});
  const [pkActive, setPkActive] = useState<Record<"streaming" | "offline", boolean>>({streaming: false, offline: false});
  const [lanes, setLanes] = useState<LaneSnapshot[]>([]); const [error, setError] = useState("");
  const [selectedAudio, setSelectedAudio] = useState<string | null>(null);
  const streamingAudioRef = useRef<MediaPlayerHandle | null>(null);
  const offlineAudioRef = useRef<MediaPlayerHandle | null>(null);
  const alignedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const active = useRef<ActiveExperiment[]>([]); const objectUrls = useRef<string[]>([]);
  const concurrencyGeneration = useRef(0);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(providedCapabilities ?? null);
  useEffect(() => { if (providedCapabilities) setCapabilities(providedCapabilities); else if (loaded) void discoverExperimentCapabilities(loaded).then(setCapabilities).catch((e) => setError(String(e))); }, [loaded, providedCapabilities]);
  useEffect(() => () => {
    concurrencyGeneration.current += 1;
    active.current.forEach((run) => run.cancel());
    active.current = [];
    if (alignedTimer.current) clearTimeout(alignedTimer.current);
    objectUrls.current.forEach(URL.revokeObjectURL);
  }, []);
  const task = useMemo(() => capabilities?.tasks.includes(settings.task) ? settings.task : capabilities?.tasks[0], [capabilities, settings.task]);
  const canRun = Boolean(loaded && text.trim() && task);
  const opts = loaded && task
    ? {loaded, text, chunkSize, chunkDelayMs, task, speaker: settings.speaker, language: settings.language, ...(capabilities ? {capabilities} : {})}
    : null;
  function remember(output: RunOutput) { if (output.audioUrl) objectUrls.current.push(output.audioUrl); }
  async function runPk() {
    if (!opts) return;
    concurrencyGeneration.current += 1;
    active.current.forEach((run) => run.cancel());
    active.current = [];
    setRunning(true); setError(""); setPk({}); setPkPlayedMs({streaming: 0, offline: 0}); setPkActive({streaming: false, offline: false});
    const pair = startPkExperiment(opts);
    const stream = pair.streaming; const offline = pair.offline;
    active.current = [stream, offline];
    const [a, b] = await Promise.all([stream.done, offline.done]);
    remember(a); remember(b); setPk({streaming: a, offline: b});
    active.current = [];
    setRunning(false);
  }
  async function runConcurrency() {
    if (!opts) return;
    const generation = concurrencyGeneration.current + 1;
    concurrencyGeneration.current = generation;
    active.current.forEach((run) => run.cancel());
    active.current = [];
    setRunning(true); setError(""); setSelectedAudio(null);
    const count = safeConcurrency(concurrency);
    // Launch the selected set as one burst. A worker pool here would turn a
    // 128-way pressure test into repeated 16-way waves and contaminate TTFT
    // with client-side queue time.
    const initial = Array.from({length: count}, (_, id) => ({id, status: "connecting" as const}));
    setLanes(initial);
    const commonStart = performance.now();
    const startBarrier = createExperimentStartBarrier(count);
    const launched = Array.from({length: count}, (_, laneId) => {
      const run = startExperiment(opts, "streaming", commonStart, (update) => {
        if (generation !== concurrencyGeneration.current) return;
        setLanes((current) => current.map((item) => item.id === laneId ? {
          ...item,
          status: update.phase === "streaming" ? "streaming" : update.phase === "failed" ? "failed" : item.status,
          ...(update.firstAudioMs === undefined ? {} : {firstAudioMs: update.firstAudioMs}),
          ...(update.clientFirstAudioMs === undefined ? {} : {clientFirstAudioMs: update.clientFirstAudioMs}),
          ...(update.serverTtftMs === undefined ? {} : {serverTtftMs: update.serverTtftMs}),
          ...(update.error ? {error: update.error} : {}),
        } : item));
      }, {barrier: startBarrier, laneId});
      active.current.push(run);
      return {laneId, run};
    });
    await Promise.all(launched.map(async ({laneId, run}) => {
      const output = await run.done;
      active.current = active.current.filter((item) => item !== run);
      if (generation !== concurrencyGeneration.current) return;
      remember(output);
      setLanes((current) => current.map((item) => item.id === laneId ? {
        ...item,
        status: output.error ? "failed" : "done",
        firstAudioMs: output.firstAudioMs,
        clientFirstAudioMs: output.clientFirstAudioMs,
        serverTtftMs: output.serverTtftMs,
        totalMs: output.totalMs,
        ...(output.error ? {error: output.error} : {}),
        ...(output.audioUrl ? {audioUrl: output.audioUrl} : {}),
      } : item));
    }));
    if (generation === concurrencyGeneration.current) {
      setRunning(false);
      active.current = [];
    }
  }
  function cancel() {
    concurrencyGeneration.current += 1;
    active.current.forEach((run) => run.cancel());
    active.current = [];
    setLanes((current) => current.map((lane) => ["queued", "connecting", "streaming"].includes(lane.status) ? {...lane, status: "cancelled"} : lane));
    setRunning(false);
  }
  function playAligned() {
    const stream = streamingAudioRef.current; const offline = offlineAudioRef.current; const streamOutput = pk.streaming; const offlineOutput = pk.offline;
    if (!stream || !offline || !streamOutput?.audioUrl || !offlineOutput?.audioUrl) return;
    if (alignedTimer.current) clearTimeout(alignedTimer.current);
    stream.reset(); offline.reset();
    const start = (player: MediaPlayerHandle, label: string) => {
      void player.play().catch(() => setError(`${label} 音频加载失败，请重新运行 PK。`));
    };
    const offset = (offlineOutput.firstAudioMs || 0) - (streamOutput.firstAudioMs || 0);
    const delayed = offset >= 0 ? offline : stream;
    const immediate = offset >= 0 ? stream : offline;
    start(immediate, offset >= 0 ? "Streaming" : "Offline");
    alignedTimer.current = setTimeout(() => {
      start(delayed, offset >= 0 ? "Offline" : "Streaming");
      alignedTimer.current = null;
    }, Math.abs(offset));
  }
  function playSingle(key: "streaming" | "offline") {
    const player = key === "streaming" ? streamingAudioRef.current : offlineAudioRef.current;
    const output = pk[key];
    if (!player || !output?.audioUrl) return;
    stopAligned();
    setPkPlayedMs((current) => ({...current, [key]: 0}));
    player.reset();
    void player.play().catch(() => setError(`${key === "streaming" ? "Streaming" : "Offline"} 音频加载失败，请重新运行 PK。`));
  }
  function stopAligned() { if (alignedTimer.current) clearTimeout(alignedTimer.current); alignedTimer.current = null; setPkActive({streaming: false, offline: false}); streamingAudioRef.current?.pause(); offlineAudioRef.current?.pause(); }
  function resetAligned() { stopAligned(); setPkPlayedMs({streaming: 0, offline: 0}); streamingAudioRef.current?.reset(); offlineAudioRef.current?.reset(); }
  function reportPkPosition(key: "streaming" | "offline", seconds: number) {
    setPkPlayedMs((current) => ({...current, [key]: Math.max(0, seconds * 1000)}));
  }
  function reportPkPlaybackState(key: "streaming" | "offline", playing: boolean) {
    setPkActive((current) => ({...current, [key]: playing}));
  }
  const stats = concurrencyStats(lanes);
  const firstLaneError = lanes.find((lane) => lane.status === "failed" && lane.error)?.error;
  const pkScaleMs = Math.max(
    1,
    ...[pk.streaming, pk.offline].flatMap((output) => output ? [
      output.totalMs,
      output.firstAudioMs + (output.audioDurationMs ?? 0),
    ] : []),
  );
  const pkFirstTimes = [pk.streaming, pk.offline].flatMap((output) => output ? [output.firstAudioMs] : []);
  const pkOriginMs = pkFirstTimes.length ? Math.min(...pkFirstTimes) : 0;
  const pkPlaybackMs = Math.max(0, pkScaleMs - pkOriginMs);
  const pkPlayedCandidates = (["streaming", "offline"] as const)
    .filter((key) => pkActive[key] || pkPlayedMs[key] > 0)
    .map((key) => (pk[key]?.firstAudioMs ?? 0) + pkPlayedMs[key] - pkOriginMs);
  const pkCursorMs = pkPlayedCandidates.length ? Math.max(0, ...pkPlayedCandidates) : 0;
  const pkHasResult = Boolean(pk.streaming || pk.offline);
  const pkReady = Boolean(pk.streaming?.audioUrl && pk.offline?.audioUrl);
  const pkLead = pk.streaming && pk.offline
    ? Math.abs(pk.streaming.firstAudioMs - pk.offline.firstAudioMs)
    : 0;
  const pkWinner = pk.streaming && pk.offline && pk.streaming.firstAudioMs <= pk.offline.firstAudioMs
    ? "Streaming"
    : "Offline";
  const displayLanes: LaneSnapshot[] = lanes.length > 0
    ? lanes
    : Array.from({length: Math.min(concurrency, MAX_CONCURRENCY)}, (_, id) => ({id, status: "idle" as const}));
  return <section className={embedded ? "embedded-lab experiment-lab" : "page experiment-lab"}>
    <p className="eyebrow">ENGINEERING LAB · REALTIME FIELD TEST</p><h1>{embedded ? "让上游文本，和声音一起到达。" : "同一入口，观察不同负载。"}</h1><p className="lab-intro">同一个 Browser SDK、同一个计时起点。流式路按字块到达；完整文本路等上游结束后再发送。</p>
    <div className="lab-grid">
      <section className="panel lab-card lab-pk-card"><div className="panel-heading"><div><p className="panel-kicker">01 · LLM PK</p><h2>流式输入 vs 完整输入</h2></div><span className="capability-chip">{task ? `task · ${task}` : "未发现可用 task"}</span></div>
        <label className="field-label">模拟上游文本<textarea value={text} onChange={(e) => setText(e.target.value)} /></label>
        <div className="control-row"><label>每块字数<input type="number" min="1" max="32" value={chunkSize} onChange={(e) => setChunkSize(Math.min(32, Math.max(1, Number(e.target.value) || 1)))}/></label><label>块间隔（毫秒）<input type="number" min="0" max="10000" value={chunkDelayMs} onChange={(e) => setChunkDelayMs(Math.min(10000, Math.max(0, Number(e.target.value) || 0)))}/></label></div>
        <div className="preset-row"><span>速度预设</span>{[[25,"快"],[45,"标准"],[90,"慢"]].map(([value, label]) => <button key={value} className={chunkDelayMs === value ? "selected" : ""} onClick={() => setChunkDelayMs(Number(value))}>{label} · {value}ms</button>)}</div>
        <div className="actions"><button className="primary" disabled={!canRun || running} onClick={() => void runPk()}>{embedded ? "开始 LLM PK" : "运行 LLM PK"}</button><button disabled={!running} onClick={cancel}>取消当前实验</button></div>
        {pkHasResult && <div className="pk-overview">
          <div className="pk-overview-actions"><button className="primary" disabled={!pkReady} onClick={playAligned}><Play size={16}/> 对齐播放两路</button><button className="icon-button" onClick={resetAligned} aria-label="重置对齐播放" title="重置播放"><RotateCcw size={16}/></button></div>
          <span className="pk-overview-time">{formatDuration(pkCursorMs)} / {formatDuration(pkPlaybackMs)}</span>
        </div>}
        {pk.streaming && pk.offline && <p className="pk-lead"><strong>{pkWinner} 提前 {Math.round(pkLead)}ms 出声</strong><span>从同一时刻回放，保留首音频等待。</span></p>}
        {pkHasResult && <p className="pk-metric-note">服务端 TTFT = response.create → 首个原始音频；客户端 TTFT = 本路请求发出 → 浏览器收到首个音频；轨道上的 burst→首音频用于比较两路从同一实验起点的实际到达。</p>}
        {pkHasResult && !pkReady && <p className="pk-audio-warning">两路都没有可播放音频；重新运行 PK 后可在这里对齐试听。</p>}
        <div className="pk-results">{(["streaming", "offline"] as const).map((key) => <LaneCard key={key} name={key === "streaming" ? "Streaming" : "Offline"} kind={key === "streaming" ? "增量文本" : "完整文本"} audioRef={key === "streaming" ? streamingAudioRef : offlineAudioRef} scaleMs={pkScaleMs} playedMs={pkPlayedMs[key]} onPlay={() => playSingle(key)} onPositionChange={(seconds) => reportPkPosition(key, seconds)} onPlaybackStateChange={(playing) => reportPkPlaybackState(key, playing)} {...(pk[key] ? {output: pk[key]} : {})} tone={key} />)}</div>
        {pk.streaming && pk.offline && <><div className="pk-axis"><span>0 s · 同一起点</span><span>{formatSeconds(pkScaleMs / 2)}</span><span>{formatSeconds(pkScaleMs)}</span></div><div className="pk-legend"><span><i className="is-waiting"/>等待首音频</span><span><i className="is-available"/>可播放音频</span><span><i className="is-played"/>已播放</span><span>播放时标记实时移动</span></div></>}
      </section>
      <section className="panel lab-card lab-concurrency-card"><div className="panel-heading"><div><p className="panel-kicker">02 · CONCURRENCY</p><h2>并发压力面板</h2></div><span className="capability-chip">{capabilities?.native_cursor?.graph_enabled ? "native cursor" : "Browser SDK"}</span></div>
        <label className="field-label">并发路数<input type="number" min="1" max={MAX_CONCURRENCY} value={concurrency} onChange={(e) => setConcurrency(safeConcurrency(Number(e.target.value)))}/></label>
        <div className="preset-row concurrency-presets"><span>常用基准</span>{[16, 32, 64, 128, 256, 512].map((value) => <button key={value} className={concurrency === value ? "selected" : ""} onClick={() => setConcurrency(value)}>{value} 路</button>)}</div>
        <p className="hint">选择的 1–{MAX_CONCURRENCY} 路会在同一 burst 中全部发起，不在浏览器端按 16 路排队。服务端 active session 上限之外的请求会直接返回 max_sessions。服务端 TTFT = response.create → 首个原始音频；客户端 TTFT = 本路请求发出 → 首个音频；另列 burst → 首音频来观察连接、准入和浏览器调度。</p>
        <div className="actions"><button className="primary" disabled={!canRun || running} onClick={() => void runConcurrency()}>开始并发测试</button></div>
        <div className="concurrency-stats">
          <div><strong>{stats.averageServerTtftMs ? `${stats.averageServerTtftMs.toFixed(0)}ms` : "—"}</strong><span>服务端 TTFT 平均 · n={stats.serverTtftSamples}</span></div>
          <div><strong>{stats.p90ServerTtftMs ? `${stats.p90ServerTtftMs.toFixed(0)}ms` : "—"}</strong><span>服务端 TTFT p90 · n={stats.serverTtftSamples}</span></div>
          <div><strong>{stats.averageClientFirstAudioMs ? `${stats.averageClientFirstAudioMs.toFixed(0)}ms` : "—"}</strong><span>客户端 TTFT 平均 · n={stats.clientFirstAudioSamples}</span></div>
          <div><strong>{stats.p90ClientFirstAudioMs ? `${stats.p90ClientFirstAudioMs.toFixed(0)}ms` : "—"}</strong><span>客户端 TTFT p90 · n={stats.clientFirstAudioSamples}</span></div>
          <div><strong>{stats.completed} / {stats.failed}</strong><span>完成 / 失败</span></div>
        </div>
        {lanes.length > 0 && <p className="lane-progress">统计分母：完成 {stats.completed} 条 · 客户端首音频样本 {stats.clientFirstAudioSamples} 条 · 服务端 TTFT 样本 {stats.serverTtftSamples} 条</p>}
        {stats.averageBurstFirstAudioMs > 0 && <p className="lane-progress">burst → 首音频：平均 {stats.averageBurstFirstAudioMs.toFixed(0)}ms · p90 {stats.p90BurstFirstAudioMs.toFixed(0)}ms（包含连接、准入等待与浏览器调度）</p>}
        {(stats.active > 0 || stats.queued > 0) && <p className="lane-progress">已发起 {stats.started} 条 · 当前连接 {stats.active} 条 · 浏览器排队 {stats.queued} 条</p>}
        {firstLaneError && <p className="lane-error-summary">最近失败原因：{firstLaneError}</p>}
        <div className="lane-grid">{displayLanes.map((lane) => <button className={`lane lane-${lane.status} ${selectedAudio === lane.audioUrl ? "selected" : ""}`} key={lane.id} title={lane.error ?? `并发 ${lane.id + 1}`} onClick={() => lane.audioUrl && setSelectedAudio(lane.audioUrl)}><span>{String(lane.id + 1).padStart(3, "0")}</span><small>{lane.status === "done" ? "试听" : lane.status === "idle" ? "待启动" : lane.status === "queued" ? "排队" : lane.status === "connecting" ? "连接中" : lane.status === "streaming" ? "生成中" : lane.status === "failed" ? "失败" : "已取消"}</small></button>)}</div>
        {selectedAudio && <MediaPlayer className="selected-audio" src={selectedAudio} label="并发音轨试听" autoPlay />}
      </section>
    </div>
    {(pk.streaming || pk.offline) && <button className="trace-button" onClick={() => downloadTrace({text, chunkDelayMs, chunkSize, pk})}>下载 JSON trace</button>}
    {error && <p className="alert">{error}</p>}
  </section>;
}
function LaneCard({name, kind, output, tone, audioRef, scaleMs, playedMs, onPlay, onPositionChange, onPlaybackStateChange}: {name: string; kind: string; output?: RunOutput; tone: string; audioRef: React.MutableRefObject<MediaPlayerHandle | null>; scaleMs: number; playedMs: number; onPlay: () => void; onPositionChange: (seconds: number) => void; onPlaybackStateChange: (playing: boolean) => void}) {
  const first = output ? Math.min(output.firstAudioMs, scaleMs) : 0;
  const total = output
    ? Math.min(Math.max(output.audioUrl ? first + (output.audioDurationMs ?? 0) : output.totalMs, first), scaleMs)
    : 0;
  const played = output?.audioUrl
    ? Math.min(Math.max(0, playedMs), output.audioDurationMs ?? Math.max(0, total - first))
    : 0;
  const cursor = Math.min(scaleMs, first + played);
  return <article className={`pk-lane ${tone}`}>
    <div className="pk-lane-head"><div><strong>{name}</strong><span>{kind}</span></div><button className="pk-lane-audio" disabled={!output?.audioUrl} onClick={onPlay} title={output?.audioUrl ? `试听 ${name}` : "等待音频"}><Headphones size={16}/><span>{output?.audioUrl ? "可试听" : "等待结果"}</span></button></div>
    <div className="pk-track"><i className="pk-track-wait" style={{width: `${first / scaleMs * 100}%`}}/><i className={`pk-track-audio ${tone}`} style={{left: `${first / scaleMs * 100}%`, width: `${Math.max(0, (total - first) / scaleMs * 100)}%`}}/><i className={`pk-track-played ${tone}`} style={{left: `${first / scaleMs * 100}%`, width: `${played / scaleMs * 100}%`}}/><b className="pk-track-marker" style={{left: `${cursor / scaleMs * 100}%`}}/></div>
    <div className="pk-lane-meta"><span>burst→首音频 <strong>{output?.firstAudioMs ? `${output.firstAudioMs.toFixed(0)}ms` : "—"}</strong></span><span>客户端 TTFT <strong>{output?.clientFirstAudioMs ? `${output.clientFirstAudioMs.toFixed(0)}ms` : "—"}</strong></span><span>服务端 TTFT <strong>{output?.serverTtftMs ? `${output.serverTtftMs.toFixed(0)}ms` : "—"}</strong></span><span>合成耗时 <strong>{output?.totalMs ? `${(output.totalMs / 1000).toFixed(2)}s` : "—"}</strong></span><span>音频 <strong>{output?.audioDurationMs ? formatDuration(output.audioDurationMs) : "—"}</strong></span></div>
    {output?.error && <p className="pk-lane-error">{output.error}</p>}
    {output?.audioUrl && <MediaPlayer ref={audioRef} src={output.audioUrl} label={`${name} 音频`} showControls={false} onPositionChange={onPositionChange} onPlaybackStateChange={onPlaybackStateChange} />}
  </article>;
}
function downloadTrace(value: unknown) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"})); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "qwen3tts-experiment-trace.json"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 0); }

function formatSeconds(milliseconds: number): string {
  const seconds = Math.max(0, milliseconds) / 1000;
  return `${seconds === 0 ? "0" : seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
}

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes}:${String(totalSeconds % 60).padStart(2, "0")}`;
}
