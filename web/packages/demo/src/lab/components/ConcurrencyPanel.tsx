import {Activity, Gauge, Grid3X3, Play, RadioTower} from "lucide-react";
import {useEffect, useMemo, useRef, useState, type ReactNode} from "react";

import type {LabApi} from "../api";
import type {LabConcurrencySummary, LabLaneUpdate, LabRequest, LabSocketEvent} from "../types";
import {formatMs} from "./Timeline";

interface ConcurrencyPanelProps {
  api: LabApi;
  request: LabRequest;
}

/** Multi-stream TTFT distribution panel from the former feature WebUI. */
export function ConcurrencyPanel({api, request}: ConcurrencyPanelProps) {
  const [concurrency, setConcurrency] = useState(32);
  const [laneText, setLaneText] = useState("你好，这是 Qwen3 TTS 多路合成验证。");
  const [lanes, setLanes] = useState<Record<string, LabLaneUpdate>>({});
  const [summary, setSummary] = useState<LabConcurrencySummary | null>(null);
  const [running, setRunning] = useState(false);
  const [source, setSource] = useState("");
  const [error, setError] = useState("");
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const autoplaySelectedRef = useRef(false);
  const generationRef = useRef(0);

  useEffect(() => () => {
    generationRef.current += 1;
    socketRef.current?.close();
    socketRef.current = null;
    audioRef.current?.pause();
  }, []);

  const laneList = useMemo(() => Array.from({length: concurrency}, (_, index) => {
    const id = `${index + 1}`.padStart(3, "0");
    return lanes[id] ?? ({type: "lane_update", job_id: "", stream_id: id, status: "pending"} satisfies LabLaneUpdate);
  }), [concurrency, lanes]);
  const selectedLane = selectedLaneId ? lanes[selectedLaneId] : undefined;
  const selectedAudioUrl = selectedLane?.audio?.url ? api.audioUrl(selectedLane.audio.url).toString() : "";

  useEffect(() => {
    if (!selectedAudioUrl || !autoplaySelectedRef.current) return;
    autoplaySelectedRef.current = false;
    void audioRef.current?.play().catch(() => undefined);
  }, [selectedAudioUrl]);

  async function run() {
    socketRef.current?.close();
    audioRef.current?.pause();
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    setRunning(true); setSummary(null); setLanes({}); setSource(""); setError(""); setSelectedLaneId(null);
    try {
      const jobId = await api.startConcurrency({
        ...request,
        text: laneText.trim() || request.text,
        concurrency,
        live: true,
      });
      if (generation !== generationRef.current) return;
      const jobSocket = new WebSocket(api.concurrencyWsUrl(jobId));
      socketRef.current = jobSocket;
      jobSocket.onmessage = (event) => {
        if (generation !== generationRef.current) return;
        let message: LabSocketEvent;
        try { message = JSON.parse(String(event.data)) as LabSocketEvent; } catch { return; }
        if (message.type === "job_started") setSource(message.source ?? "");
        else if (message.type === "lane_update") {
          setLanes((current) => ({...current, [message.stream_id]: message}));
          if (message.audio?.url) setSelectedLaneId((current) => current ?? message.stream_id);
        } else if (message.type === "summary") {
          setSummary(message); setRunning(false); jobSocket.close();
        } else if (message.type === "error") setError(message.message ?? "Lab concurrency failed");
      };
      jobSocket.onerror = () => {
        if (generation === generationRef.current) { setError("Lab concurrency WebSocket failed"); setRunning(false); }
      };
      jobSocket.onclose = () => {
        if (generation === generationRef.current) setRunning(false);
      };
    } catch (cause) {
      if (generation === generationRef.current) { setError(String(cause)); setRunning(false); }
    }
  }

  function playLane(lane: LabLaneUpdate) {
    if (!lane.audio?.url) {
      autoplaySelectedRef.current = false;
      setSelectedLaneId(lane.stream_id);
      return;
    }
    if (lane.stream_id === selectedLaneId) {
      void audioRef.current?.play().catch(() => undefined);
      return;
    }
    setSelectedLaneId(lane.stream_id);
    autoplaySelectedRef.current = true;
  }

  return <section className="panel lab-concurrency-panel">
    <div className="panel-heading"><div><p className="panel-kicker">CONCURRENCY</p><h2>多路合成与 TTFT 分布</h2></div>
      <div className="lab-concurrency-controls"><select value={concurrency} onChange={(event) => setConcurrency(Number(event.target.value))} aria-label="concurrency"><option value={8}>8 streams</option><option value={32}>32 streams</option><option value={64}>64 streams</option><option value={128}>128 streams</option></select><button className="primary" onClick={() => void run()} disabled={running}><Play size={16}/>{running ? "运行中" : "运行并发测试"}</button></div></div>
    <div className="lab-concurrency-input"><span>Text</span><input value={laneText} onChange={(event) => setLaneText(event.target.value)} aria-label="multi-stream synthesis text"/></div>
    {error && <p className="alert">{error}</p>}
    <div className="lab-summary-row"><SummaryMetric icon={<RadioTower size={17}/>} label="All avg TTFT" value={formatMs(summary?.avg_ttft_ms)}/><SummaryMetric icon={<Gauge size={17}/>} label="Active-slot avg" value={formatMs(summary?.active_avg_ttft_ms ?? summary?.avg_ttft_ms)}/><SummaryMetric icon={<Activity size={17}/>} label="p90 TTFT" value={formatMs(summary?.p90_ttft_ms)}/><SummaryMetric icon={<Grid3X3 size={17}/>} label="Slots / queued" value={slotLabel(summary)}/></div>
    <div className="lab-lane-grid" style={{gridTemplateColumns: `repeat(${concurrency === 8 ? 8 : 16}, minmax(0, 1fr))`}}>
      {laneList.map((lane) => <button type="button" className={`lab-lane lane-${lane.status} ${ttftClass(lane.ttft_ms)} ${lane.audio?.url ? "has-audio" : ""}`} key={lane.stream_id} onClick={() => playLane(lane)} data-selected={lane.stream_id === selectedLaneId ? "true" : "false"} data-queued={lane.queued_by_slot_limit ? "true" : "false"} title={`stream ${lane.stream_id}: ${lane.ttft_ms ? formatMs(lane.ttft_ms) : lane.status}`}>{lane.stream_id}</button>)}
    </div>
    <div className="lab-lane-player"><div><span>Selected lane</span><strong>{selectedLane ? `${selectedLane.stream_id} · ${selectedLane.status} · ${formatMs(selectedLane.ttft_ms)}` : "-"}</strong></div><div><span>Audio</span><strong>{selectedAudioUrl ? selectedLane?.audio?.source ?? "captured" : source === "simulated" ? "simulated: no audio" : "waiting for captured audio"}</strong></div>{selectedAudioUrl ? <audio ref={audioRef} key={selectedAudioUrl} controls src={selectedAudioUrl}/> : <p>{selectedLane?.error ?? "Live lane audio is shown only when real waveform bytes are captured."}</p>}</div>
    {summary && <div className="lab-benchmark-footer"><span>Completed {summary.count}/{summary.concurrency}</span><span>Failed {summary.failed_streams}</span><span>Max {formatMs(summary.max_ttft_ms)}</span><span>{summary.source}</span>{summary.queued_streams ? <span>{summary.queued_streams} queued beyond {summary.active_slot_limit} slots</span> : null}</div>}
  </section>;
}

function SummaryMetric({icon, label, value}: {icon: ReactNode; label: string; value: string}) { return <div className="lab-summary-metric">{icon}<span>{label}</span><strong>{value}</strong></div>; }
function ttftClass(value?: number): string { if (value === undefined) return "ttft-pending"; if (value < 150) return "ttft-fast"; if (value < 250) return "ttft-good"; if (value < 500) return "ttft-warm"; return "ttft-hot"; }
function slotLabel(summary: LabConcurrencySummary | null): string { return summary?.active_slot_limit ? `${summary.active_slot_limit} / ${summary.queued_streams ?? 0}` : summary?.source ?? "-"; }
