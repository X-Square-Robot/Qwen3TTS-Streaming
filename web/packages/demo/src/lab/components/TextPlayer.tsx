import {Pause, Play, RadioTower, RotateCcw} from "lucide-react";
import {useEffect, useMemo, useRef, useState} from "react";

import {buildDecodeTrace, latestTextProgress, type DecodeStep} from "../trace";
import type {LabTraceEvent} from "../types";
import {formatMs, Timeline} from "./Timeline";

interface TextPlayerProps {
  events: LabTraceEvent[];
  text: string;
  audioUrl?: string;
  liveMs: number;
  live: boolean;
  source: string;
}

/** Token/decode timeline player used by both live trace and saved PK results. */
export function TextPlayer({events, text, audioUrl, liveMs, live, source}: TextPlayerProps) {
  const model = useMemo(() => buildDecodeTrace(events, text), [events, text]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [cursorMs, setCursorMs] = useState(0);
  const [manualPlayback, setManualPlayback] = useState(false);
  const hasSeekableAudio = Boolean(audioUrl);
  const displayMs = hasSeekableAudio && (!live || manualPlayback) ? cursorMs : liveMs;
  const maxMs = Math.max(model.audioDurationMs, 1);
  const activeStep = stepAt(model.steps, displayMs);
  const progress = latestTextProgress(events);
  const sourceTokens = model.tokens.filter((token) => !token.synthetic);
  const segmentOffset = progress ? sourceTokens.filter((token) => token.segmentIdx < progress.segmentIdx).length : 0;
  const globalCount = progress ? Math.max(sourceTokens.length, segmentOffset + progress.textTokenCount) : 0;
  const globalEnd = progress ? Math.min(globalCount, segmentOffset + progress.textTokenEnd) : 0;
  const progressPercent = live && progress && globalCount > 0
    ? globalEnd / globalCount * 100
    : Math.max(0, Math.min(100, displayMs / maxMs * 100));
  const excerpt = live && progress
    ? sourceTokens.slice(Math.max(0, segmentOffset + progress.textTokenEnd - 18), segmentOffset + progress.textTokenEnd).map((token) => token.text).join("")
    : "";

  useEffect(() => {
    setPlaying(false);
    setCursorMs(0);
    setManualPlayback(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [audioUrl, events]);

  function togglePlayback() {
    const audio = audioRef.current;
    if (!audio || !hasSeekableAudio) return;
    if (audio.paused) {
      if (audio.ended || (Number.isFinite(audio.duration) && audio.currentTime >= audio.duration - 0.02)) {
        audio.currentTime = 0;
        setCursorMs(0);
      }
      setManualPlayback(true);
      void audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      audio.pause();
      setPlaying(false);
    }
  }

  function seek(value: number) {
    const next = Math.max(0, Math.min(value, maxMs));
    setManualPlayback(true);
    setCursorMs(next);
    if (audioRef.current && hasSeekableAudio) audioRef.current.currentTime = next / 1000;
  }

  return <section className="panel lab-text-player">
    <div className="panel-heading">
      <div><p className="panel-kicker">TEXT PLAYER</p><h2>文本与解码时间轴</h2></div>
      <div className="lab-player-summary">
        <span><RadioTower size={14}/> {source}</span>
        <span>{model.textTokenCount} token steps</span>
        <span>{model.padStepCount} PAD flush steps</span>
        <span>{formatMs(model.stepMs)} / step</span>
      </div>
    </div>

    <audio ref={audioRef} src={audioUrl} onTimeUpdate={(event) => setCursorMs(event.currentTarget.currentTime * 1000)}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={(event) => {
        event.currentTarget.currentTime = 0; setCursorMs(0); setPlaying(false);
      }}/>

    <div className="lab-player-transport">
      <button className="primary" onClick={togglePlayback} disabled={!hasSeekableAudio}>
        {playing ? <Pause size={16}/> : <Play size={16}/>} {playing ? "暂停" : "播放"}
      </button>
      <input type="range" min={0} max={Math.round(maxMs)} value={Math.max(0, Math.min(Math.round(displayMs), Math.round(maxMs)))}
        onChange={(event) => seek(Number(event.target.value))} disabled={!hasSeekableAudio} aria-label="text player seek"/>
      <span>{formatMs(displayMs)} / {formatMs(maxMs)}</span>
      <button className="tiny" onClick={() => seek(0)} disabled={!hasSeekableAudio}><RotateCcw size={14}/> 0</button>
    </div>

    <div className="lab-text-progress" aria-label="estimated text progress">
      <div><span>Estimated text progress</span><strong>{progressPercent.toFixed(0)}%</strong></div>
      <div className="lab-progress-track"><i style={{width: `${progressPercent}%`}}/></div>
      <p>{live && progress ? `约 ${progress.textTokenEnd}/${progress.textTokenCount} tokens · ${progress.quality}` : "根据音频播放位置推算"}
        {excerpt && <em>“{excerpt}”</em>}</p>
    </div>

    <Timeline events={events} maxMs={Math.max(maxMs, liveMs)}/>
    {!hasSeekableAudio && <p className="lab-player-note">{live ? "实时流仍在采集；收到 WAV 后可拖动播放。" : "此 trace 没有可播放音频，但仍展示事件与解码结构。"}</p>}
    <div className="lab-token-transcript" aria-label="clickable token transcript">
      {model.steps.map((step) => <button type="button" key={`step-${step.key}`} className={[
        "lab-token", `lab-token-${step.phase}`, step.endMs <= displayMs ? "played" : "", activeStep?.index === step.index ? "active" : "",
      ].join(" ")} onClick={() => seekStep(step, seek)} disabled={!hasSeekableAudio}
      title={`#${step.index + 1} ${step.phase} ${formatMs(step.startMs)}-${formatMs(step.endMs)}`}>{step.label}</button>)}
    </div>
    <p className="lab-player-note">PAD 单元表示文本 token 消费后引擎仍在冲刷音频，并非隐藏延迟。Trace source: {model.source}。</p>
  </section>;
}

function seekStep(step: DecodeStep, seek: (value: number) => void): void {
  seek(step.startMs);
}

function stepAt(steps: DecodeStep[], ms: number): DecodeStep | undefined {
  return steps.find((step) => ms >= step.startMs && ms < step.endMs) ?? steps[steps.length - 1];
}
