import {Play, RadioTower, Volume2} from "lucide-react";
import {useEffect, useMemo, useRef, useState} from "react";

import type {LabApi} from "../api";
import {eventAudioDurationMs} from "../trace";
import type {LabRequest, LabResult, LabRunResult} from "../types";
import {formatMs} from "./Timeline";

const ROW_ORDER = ["triton_streaming", "triton_offline"];
const PRESETS = [
  {label: "Local 7B", ms: 7, hint: "≈140 tok/s"},
  {label: "Claude Sonnet", ms: 12, hint: "≈80 tok/s"},
  {label: "GPT-4o", ms: 30, hint: "≈33 tok/s"},
  {label: "Reasoning", ms: 50, hint: "≈20 tok/s"},
];

interface PerformanceRaceProps {
  api: LabApi;
  defaults: LabRequest;
  onResult?: (result: LabResult) => void;
  onPlayAligned?: (rows: LabRunResult[]) => void;
  onPlayOne?: (row: LabRunResult) => void;
}

/** Rich streaming-vs-offline comparison from the former standalone WebUI. */
export function PerformanceRace({api, defaults, onResult, onPlayAligned, onPlayOne}: PerformanceRaceProps) {
  const [text, setText] = useState(defaults.text);
  const [speaker, setSpeaker] = useState(defaults.speaker);
  const [language, setLanguage] = useState(defaults.language);
  const [msPerToken, setMsPerToken] = useState(defaults.ms_per_token);
  const [result, setResult] = useState<LabResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [clockMs, setClockMs] = useState(0);
  const animationRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (animationRef.current !== null) window.cancelAnimationFrame(animationRef.current);
  }, []);

  useEffect(() => {
    setText(defaults.text); setSpeaker(defaults.speaker); setLanguage(defaults.language); setMsPerToken(defaults.ms_per_token);
  }, [defaults.text, defaults.speaker, defaults.language, defaults.ms_per_token]);

  const rows = useMemo(() => [...(result?.results ?? [])].sort((a, b) => {
    const ai = ROW_ORDER.indexOf(a.backend); const bi = ROW_ORDER.indexOf(b.backend);
    return (ai < 0 ? ROW_ORDER.length : ai) - (bi < 0 ? ROW_ORDER.length : bi);
  }), [result]);
  const maxMs = useMemo(() => Math.max(1000, ...rows.map((row) => scheduledStartMs(row) + audioDurationMs(row)), ...rows.map((row) => row.metrics.total_ms ?? 0), ...rows.map((row) => row.metrics.simulated_llm_complete_ms ?? 0)) * 1.05, [rows]);
  const hasAudio = rows.some((row) => Boolean(row.audio?.url));

  async function run() {
    const request: LabRequest = {
      text: text.trim() || defaults.text,
      speaker: speaker.trim() || defaults.speaker,
      language: language.trim() || defaults.language,
      ms_per_token: Math.max(1, msPerToken),
    };
    setBusy(true); setError(""); setResult(null);
    try {
      const value = await api.runLlmPk(request);
      setResult(value); onResult?.(value);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setBusy(false);
    }
  }

  function playAligned() {
    if (!hasAudio) return;
    // The parent owns the audio context so it can stop a previous comparison.
    onPlayAligned?.(rows.map(withPlaybackSchedule));
    setClockMs(0);
    const started = performance.now();
    const tick = () => {
      const elapsed = performance.now() - started;
      setClockMs(Math.min(elapsed, maxMs));
      if (elapsed < maxMs) {
        animationRef.current = window.requestAnimationFrame(tick);
      } else {
        animationRef.current = null;
      }
    };
    if (animationRef.current !== null) window.cancelAnimationFrame(animationRef.current);
    animationRef.current = window.requestAnimationFrame(tick);
  }

  return <section className="panel lab-performance-race">
    <div className="panel-heading"><div><p className="panel-kicker">LLM PK</p><h2>流式 vs 离线 TTS</h2></div>
      <p>模拟上游 LLM 按指定速率吐出 token，观察流式路径在文本结束前开始发声。</p></div>
    <div className="lab-race-controls">
      <textarea value={text} onChange={(event) => setText(event.target.value)} rows={2} aria-label="LLM PK text" />
      <div className="lab-race-row">
        <input value={speaker} onChange={(event) => setSpeaker(event.target.value)} aria-label="LLM PK speaker" placeholder="speaker" />
        <input value={language} onChange={(event) => setLanguage(event.target.value)} aria-label="LLM PK language" placeholder="language" />
        <label className="lab-token-rate">ms / token<input type="range" min={5} max={100} step={1} value={msPerToken} onChange={(event) => setMsPerToken(Number(event.target.value))}/><strong>{msPerToken}ms</strong></label>
        <button className="primary" onClick={() => void run()} disabled={busy}><RadioTower size={16}/>{busy ? "运行中" : "运行 PK"}</button>
        <button onClick={playAligned} disabled={!hasAudio}><Play size={16}/>对齐播放</button>
      </div>
      <div className="lab-presets"><span>Presets</span>{PRESETS.map((preset) => <button className={msPerToken === preset.ms ? "active" : ""} key={preset.label} onClick={() => setMsPerToken(preset.ms)} type="button">{preset.label} {preset.ms}ms · {preset.hint}</button>)}</div>
    </div>
    {error && <p className="alert">{error}</p>}
    {result?.warnings.map((warning) => <p className="lab-warning" key={warning}>{warning}</p>)}
    {rows.length > 0 && <p className="lab-race-summary">{buildSummary(rows) ?? "两条路径已完成；请查看时间轴和指标。"}</p>}
    <div className="lab-race-player">
      {rows.map((row) => {
        const start = scheduledStartMs(row); const duration = audioDurationMs(row);
        const active = Math.max(0, Math.min(duration, clockMs - start));
        const llmDone = row.metrics.simulated_llm_complete_ms;
        return <div className="lab-race-row-result" key={`${row.backend}-${row.run_id}`}>
          <div className="lab-race-label"><strong>{row.label}</strong><span>{row.mode}</span><small>{timingLabel(row)}</small>{row.warnings.slice(0, 2).map((warning) => <em key={warning}>{warning}</em>)}</div>
          <div className="lab-race-track" aria-label={`${row.label} timeline`}>
            {row.events.filter((event) => event.type === "llm_token").map((event, index) => <i className="lab-token-tick" key={`${index}-${event.t_ms}`} style={{left: `${event.t_ms / maxMs * 100}%`}} title={`${event.text ?? "token"} @ ${formatMs(event.t_ms)}`}/>)}
            {llmDone !== undefined && <i className="lab-llm-done" style={{left: `${llmDone / maxMs * 100}%`}} title={`LLM done @ ${formatMs(llmDone)}`}/>}<div className="lab-audio-window" style={{left: `${start / maxMs * 100}%`, width: `${Math.max(1, duration / maxMs * 100)}%`}}><i style={{width: `${duration > 0 ? active / duration * 100 : 0}%`}}/></div>
          </div>
          <button className="tiny" onClick={() => onPlayOne?.(withPlaybackSchedule(row))} disabled={!row.audio?.url}><Volume2 size={14}/>{row.audio?.url ? "音频" : "无音频"}</button>
        </div>;
      })}
    </div>
    {rows.length === 0 && !busy && <p className="lab-player-note">点击“运行 PK”后，结果会从可选工程 API 返回；不会用 fixture 或历史数字填充。</p>}
    {rows.length > 0 && <p className="lab-player-note">细线表示模拟 token 到达，虚线表示上游结束；音频窗口来自本次真实返回。</p>}
  </section>;
}

function buildSummary(rows: LabRunResult[]): string | null {
  const streaming = rows.find((row) => row.backend === "triton_streaming");
  const offline = rows.find((row) => row.backend === "triton_offline");
  if (!streaming || !offline) return null;
  const start = streaming.metrics.first_playable_ms ?? streaming.metrics.client_ttfb_ms;
  const end = offline.metrics.first_playable_ms ?? offline.metrics.client_ttfb_ms;
  if (start === undefined || end === undefined || end <= start) return null;
  return `流式路径提前 ${formatMs(end - start)} 开始发声（${formatMs(start)} vs ${formatMs(end)}）。`;
}

function scheduledStartMs(row: LabRunResult): number { return Number(row.audio?.scheduled_start_ms ?? row.metrics.first_playable_ms ?? row.metrics.client_ttfb_ms ?? 0); }
function audioDurationMs(row: LabRunResult): number {
  const measured = Number(row.metrics.audio_duration_ms);
  if (Number.isFinite(measured) && measured > 0) return measured;
  const traced = row.events.map(eventAudioDurationMs).filter((value) => value > 0);
  return traced.reduce((sum, value) => sum + value, 0);
}
function timingLabel(row: LabRunResult): string { return `first audio ${formatMs(row.metrics.client_ttfb_ms ?? row.metrics.first_playable_ms)} · LLM done ${formatMs(row.metrics.simulated_llm_complete_ms)} · total ${formatMs(row.metrics.total_ms)}`; }
function withPlaybackSchedule(row: LabRunResult): LabRunResult {
  if (!row.audio?.url) return row;
  const scheduled = scheduledStartMs(row);
  return {...row, audio: {...row.audio, scheduled_start_ms: scheduled}};
}
