import {useEffect, useMemo, useRef, useState} from "react";
import {type Capabilities} from "@xmultimodalinteraction/qwen3tts-browser";
import {DEFAULT_DEMO_SETTINGS, type DemoSynthesisSettings} from "./demo-settings";
import type {LoadedDemoConfig} from "./config";
import {concurrencyStats, MAX_CONCURRENCY, safeConcurrency, type LaneSnapshot} from "./lab/experiment-model";
import {discoverExperimentCapabilities, startExperiment, startPkExperiment, type ActiveExperiment, type RunOutput} from "./lab/experiments";
import {MediaPlayer, type MediaPlayerHandle} from "./components/MediaPlayer";
import "./lab/experiments.css";

export interface ExperimentLabProps { loaded: LoadedDemoConfig | null; embedded?: boolean; capabilities?: Capabilities | null; settings?: DemoSynthesisSettings; }

export function ExperimentLab({loaded, embedded = false, capabilities: providedCapabilities, settings = DEFAULT_DEMO_SETTINGS}: ExperimentLabProps) {
  const [text, setText] = useState("你好，这是从上游大模型逐步到达的流式文本。");
  const [chunkDelayMs, setChunkDelayMs] = useState(45); const [chunkSize, setChunkSize] = useState(4);
  const [concurrency, setConcurrency] = useState(4); const [running, setRunning] = useState(false);
  const [pk, setPk] = useState<{streaming?: RunOutput; offline?: RunOutput}>({});
  const [lanes, setLanes] = useState<LaneSnapshot[]>([]); const [error, setError] = useState("");
  const [selectedAudio, setSelectedAudio] = useState<string | null>(null);
  const streamingAudioRef = useRef<MediaPlayerHandle | null>(null);
  const offlineAudioRef = useRef<MediaPlayerHandle | null>(null);
  const alignedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const active = useRef<ActiveExperiment[]>([]); const objectUrls = useRef<string[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(providedCapabilities ?? null);
  useEffect(() => { if (providedCapabilities) setCapabilities(providedCapabilities); else if (loaded) void discoverExperimentCapabilities(loaded).then(setCapabilities).catch((e) => setError(String(e))); }, [loaded, providedCapabilities]);
  useEffect(() => () => { active.current.forEach((run) => run.cancel()); if (alignedTimer.current) clearTimeout(alignedTimer.current); objectUrls.current.forEach(URL.revokeObjectURL); }, []);
  const task = useMemo(() => capabilities?.tasks.includes(settings.task) ? settings.task : capabilities?.tasks[0], [capabilities, settings.task]);
  const canRun = Boolean(loaded && text.trim() && task);
  const opts = loaded && task ? {loaded, text, chunkSize, chunkDelayMs, task, speaker: settings.speaker, language: settings.language} : null;
  function remember(output: RunOutput) { if (output.audioUrl) objectUrls.current.push(output.audioUrl); }
  async function runPk() { if (!opts) return; setRunning(true); setError(""); setPk({}); const pair = startPkExperiment(opts); const stream = pair.streaming; const offline = pair.offline; active.current = [stream, offline]; const [a, b] = await Promise.all([stream.done, offline.done]); remember(a); remember(b); setPk({streaming: a, offline: b}); setRunning(false); }
  async function runConcurrency() { if (!opts) return; setRunning(true); setError(""); setSelectedAudio(null); const count = safeConcurrency(concurrency); const initial = Array.from({length: count}, (_, id) => ({id, status: "connecting" as const})); setLanes(initial); const runs = initial.map((lane) => { const run = startExperiment(opts, "streaming", performance.now(), (update) => { setLanes((current) => current.map((item) => item.id === lane.id ? {...item, status: update.phase === "streaming" ? "streaming" : update.phase === "failed" ? "failed" : item.status, ...(update.firstAudioMs === undefined ? {} : {firstAudioMs: update.firstAudioMs}), ...(update.error ? {error: update.error} : {})} : item)); }); active.current.push(run); void run.done.then((out) => { remember(out); setLanes((current) => current.map((item) => { if (item.id !== lane.id) return item; return {...item, status: out.error ? "failed" : "done", firstAudioMs: out.firstAudioMs, totalMs: out.totalMs, ...(out.error ? {error: out.error} : {}), ...(out.audioUrl ? {audioUrl: out.audioUrl} : {})}; })); }); return run; }); await Promise.all(runs.map((run) => run.done)); setRunning(false); }
  function cancel() { active.current.forEach((run) => run.cancel()); active.current = []; setRunning(false); }
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
  function stopAligned() { if (alignedTimer.current) clearTimeout(alignedTimer.current); alignedTimer.current = null; streamingAudioRef.current?.pause(); offlineAudioRef.current?.pause(); }
  const stats = concurrencyStats(lanes);
  const pkScaleMs = Math.max(1, pk.streaming?.totalMs ?? 0, pk.offline?.totalMs ?? 0);
  return <section className={embedded ? "embedded-lab experiment-lab" : "page experiment-lab"}>
    <p className="eyebrow">ENGINEERING LAB · REALTIME FIELD TEST</p><h1>{embedded ? "让上游文本，和声音一起到达。" : "同一入口，观察不同负载。"}</h1><p className="lab-intro">同一个 Browser SDK、同一个计时起点。流式路按字块到达；完整文本路等上游结束后再发送，时间轴因此能看出等待成本。</p>
    <div className="lab-grid">
      <section className="panel lab-card"><div className="panel-heading"><div><p className="panel-kicker">01 · LLM PK</p><h2>流式输入 vs 完整输入</h2></div><span className="capability-chip">{task ? `task · ${task}` : "未发现可用 task"}</span></div>
        <label className="field-label">模拟上游文本<textarea value={text} onChange={(e) => setText(e.target.value)} /></label>
        <div className="control-row"><label>每块字数<input type="number" min="1" max="32" value={chunkSize} onChange={(e) => setChunkSize(Math.min(32, Math.max(1, Number(e.target.value) || 1)))}/></label><label>块间隔（毫秒）<input type="number" min="0" max="10000" value={chunkDelayMs} onChange={(e) => setChunkDelayMs(Math.min(10000, Math.max(0, Number(e.target.value) || 0)))}/></label></div>
        <div className="preset-row"><span>速度预设</span>{[[25,"快"],[45,"标准"],[90,"慢"]].map(([value, label]) => <button key={value} className={chunkDelayMs === value ? "selected" : ""} onClick={() => setChunkDelayMs(Number(value))}>{label} · {value}ms</button>)}</div>
        <div className="actions"><button className="primary" disabled={!canRun || running} onClick={() => void runPk()}>{embedded ? "开始 LLM PK" : "运行 LLM PK"}</button><button disabled={!running} onClick={cancel}>取消当前实验</button></div>
        <div className="pk-results">{(["streaming", "offline"] as const).map((key) => <LaneCard key={key} name={embedded ? (key === "streaming" ? "Streaming" : "Offline") : (key === "streaming" ? "增量文本" : "完整文本")} audioRef={key === "streaming" ? streamingAudioRef : offlineAudioRef} scaleMs={pkScaleMs} {...(pk[key] ? {output: pk[key]} : {})} tone={key} />)}</div>
        {pk.streaming?.audioUrl && pk.offline?.audioUrl && <div className="pk-transport"><button className="primary" onClick={playAligned}>▶ 对齐播放两路</button><button onClick={stopAligned}>暂停对比</button><span>按两路首音频时间差错开播放，直接听出 TTFT 差异。</span></div>}
      </section>
      <section className="panel lab-card"><div className="panel-heading"><div><p className="panel-kicker">02 · CONCURRENCY</p><h2>并发压力面板</h2></div><span className="capability-chip">{capabilities?.native_cursor?.graph_enabled ? "native cursor" : "Browser SDK"}</span></div>
        <label className="field-label">并发路数<input type="number" min="1" max={MAX_CONCURRENCY} value={concurrency} onChange={(e) => setConcurrency(safeConcurrency(Number(e.target.value)))}/></label>
        <div className="preset-row concurrency-presets"><span>常用基准</span>{[16, 32, 64, 128, 256, 512].map((value) => <button key={value} className={concurrency === value ? "selected" : ""} onClick={() => setConcurrency(value)}>{value} 路</button>)}</div>
        <p className="hint">可测试 1–{MAX_CONCURRENCY} 路；128 路是常用基准，实际容量仍由实例和浏览器资源决定。</p>
        <div className="actions"><button className="primary" disabled={!canRun || running} onClick={() => void runConcurrency()}>开始并发测试</button></div>
        <div className="concurrency-stats"><strong>{stats.averageFirstAudioMs ? `${stats.averageFirstAudioMs.toFixed(0)}ms` : "—"}</strong><span>平均首音频</span><strong>{stats.p90FirstAudioMs ? `${stats.p90FirstAudioMs.toFixed(0)}ms` : "—"}</strong><span>p90 首音频</span><strong>{stats.completed} / {stats.failed}</strong><span>完成 / 失败</span></div>
        <div className="lane-grid">{lanes.map((lane) => <button className={`lane lane-${lane.status} ${selectedAudio === lane.audioUrl ? "selected" : ""}`} key={lane.id} title={lane.error ?? `并发 ${lane.id + 1}`} onClick={() => lane.audioUrl && setSelectedAudio(lane.audioUrl)}><span>{lane.id + 1}</span><small>{lane.status === "done" ? "试听" : lane.status}</small></button>)}</div>
        {selectedAudio && <MediaPlayer className="selected-audio" src={selectedAudio} label="并发音轨试听" autoPlay />}
      </section>
    </div>
    {(pk.streaming || pk.offline) && <button className="trace-button" onClick={() => downloadTrace({text, chunkDelayMs, chunkSize, pk})}>下载 JSON trace</button>}
    {error && <p className="alert">{error}</p>}
  </section>;
}
function LaneCard({name, output, tone, audioRef, scaleMs}: {name: string; output?: RunOutput; tone: string; audioRef: React.MutableRefObject<MediaPlayerHandle | null>; scaleMs: number}) {
  return <article className={`pk-lane ${tone}`}>
    <div><strong>{name}</strong><small>{output?.error ?? "等待结果"}</small></div>
    <div className="timeline"><i style={{width: output ? `${Math.min(100, output.totalMs / scaleMs * 100)}%` : "0%"}}/><b style={{left: output ? `${Math.min(100, output.firstAudioMs / scaleMs * 100)}%` : "0%"}}/><span>首音频 {output?.firstAudioMs ? `${output.firstAudioMs.toFixed(0)}ms` : "—"} · 完成 {output?.totalMs ? `${output.totalMs.toFixed(0)}ms` : "—"}</span></div>
    {output?.audioUrl && <MediaPlayer ref={audioRef} src={output.audioUrl} label={`${name} 音频`} showControls={false} />}
  </article>;
}
function downloadTrace(value: unknown) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"})); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "qwen3tts-experiment-trace.json"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 0); }
