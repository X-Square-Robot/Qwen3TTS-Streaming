import { Activity, Gauge, Grid3X3, Play, RadioTower } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { mediaUrl, startConcurrency, wsUrl } from "../api";
import type { ConcurrencySummary, DemoRequest, LaneUpdate } from "../types";
import { formatMs } from "./Timeline";

interface ConcurrencyPanelProps {
  request: DemoRequest;
}

export function ConcurrencyPanel({ request }: ConcurrencyPanelProps) {
  const [concurrency, setConcurrency] = useState(128);
  const [laneText, setLaneText] = useState("你好，这是千问3 TTS多路合成验证。");
  const [lanes, setLanes] = useState<Record<string, LaneUpdate>>({});
  const [summary, setSummary] = useState<ConcurrencySummary | null>(null);
  const [running, setRunning] = useState(false);
  const [source, setSource] = useState("");
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const autoplaySelectedRef = useRef(false);

  async function run() {
    socketRef.current?.close();
    audioRef.current?.pause();
    setRunning(true);
    setSummary(null);
    setLanes({});
    setSource("");
    setSelectedLaneId(null);
    const jobId = await startConcurrency({ ...request, text: laneText.trim() || request.text, concurrency, live: true });
    const socket = new WebSocket(wsUrl(`/api/v1/concurrency/${jobId}`));
    socketRef.current = socket;
    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return; // ignore malformed frames instead of killing the run
      }
      if (message.type === "job_started") {
        setSource(message.source);
      }
      if (message.type === "lane_update") {
        setLanes((current) => ({ ...current, [message.stream_id]: message }));
        if (message.audio?.url) {
          setSelectedLaneId((current) => current ?? message.stream_id);
        }
      }
      if (message.type === "summary") {
        setSummary(message);
        setRunning(false);
        socket.close();
      }
    };
    socket.onerror = () => setRunning(false);
    // Re-enable the Run button if the socket drops without a summary frame,
    // otherwise it stays stuck on "Running" forever.
    socket.onclose = () => setRunning(false);
  }

  // Close the live socket if the panel unmounts mid-run (avoids a leaked
  // connection whose handlers keep setting state on an unmounted component).
  useEffect(() => {
    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  const laneList = useMemo(() => {
    return Array.from({ length: concurrency }, (_, index) => {
      const id = `${index + 1}`.padStart(3, "0");
      return lanes[id] ?? { type: "lane_update", job_id: "", stream_id: id, status: "pending" };
    });
  }, [concurrency, lanes]);
  const selectedLane = selectedLaneId ? lanes[selectedLaneId] : undefined;
  const selectedAudioUrl = selectedLane?.audio?.url ? mediaUrl(selectedLane.audio.url) : "";

  useEffect(() => {
    if (!selectedAudioUrl || !autoplaySelectedRef.current) {
      return;
    }
    autoplaySelectedRef.current = false;
    void audioRef.current?.play();
  }, [selectedAudioUrl]);

  function playLane(lane: LaneUpdate) {
    if (lane.audio?.url && lane.stream_id === selectedLaneId) {
      void audioRef.current?.play();
      return;
    }
    setSelectedLaneId(lane.stream_id);
    autoplaySelectedRef.current = Boolean(lane.audio?.url);
  }

  return (
    <section className="panel concurrency-panel">
      <div className="panel-head">
        <div>
          <h2>Multi-Stream Synthesis</h2>
          <p>parallel synthesis, TTFT distribution, real lane audio when live</p>
        </div>
        <div className="concurrency-controls">
          <select value={concurrency} onChange={(event) => setConcurrency(Number(event.target.value))}>
            {[8, 32, 64, 128].map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
          <button className="button primary" onClick={run} disabled={running}>
            <Play size={16} />
            {running ? "Running" : "Run"}
          </button>
        </div>
      </div>

      <div className="concurrency-text-row">
        <span>Text</span>
        <input
          value={laneText}
          onChange={(event) => setLaneText(event.target.value)}
          aria-label="multi-stream synthesis text"
        />
      </div>

      <div className="summary-row">
        <SummaryMetric icon={<RadioTower size={18} />} label="All avg TTFT" value={formatMs(summary?.avg_ttft_ms)} />
        <SummaryMetric icon={<Gauge size={18} />} label="Active-slot avg" value={formatMs(summary?.active_avg_ttft_ms ?? summary?.avg_ttft_ms)} />
        <SummaryMetric icon={<Activity size={18} />} label="p90 TTFT" value={formatMs(summary?.p90_ttft_ms)} />
        <SummaryMetric icon={<Grid3X3 size={18} />} label="Slots / queued" value={slotLabel(summary)} />
      </div>

      <div className="lane-grid" style={{ gridTemplateColumns: `repeat(${concurrency === 8 ? 8 : 16}, minmax(0, 1fr))` }}>
        {laneList.map((lane) => (
          <div
            className={`lane lane-${lane.status} ${ttftClass(lane.ttft_ms)} ${lane.audio?.url ? "lane-audio" : ""}`}
            key={lane.stream_id}
            title={`stream ${lane.stream_id}: ${lane.ttft_ms ? formatMs(lane.ttft_ms) : lane.status}${lane.queued_by_slot_limit ? " · queued beyond active slot limit" : ""}${lane.error ? ` · ${lane.error}` : ""}${lane.audio?.url ? " · click to play" : ""}`}
            onClick={() => playLane(lane)}
            data-selected={lane.stream_id === selectedLaneId ? "true" : "false"}
            data-queued={lane.queued_by_slot_limit ? "true" : "false"}
          >
            <span>{lane.stream_id}</span>
          </div>
        ))}
      </div>

      <div className="lane-player">
        <div>
          <span>Selected lane</span>
          <strong>{selectedLane ? `${selectedLane.stream_id} · ${selectedLane.status} · ${formatMs(selectedLane.ttft_ms)}` : "-"}</strong>
        </div>
        <div>
          <span>Audio</span>
          <strong>{selectedAudioUrl ? selectedLane?.audio?.source ?? "captured" : source === "simulated" ? "simulated: no audio" : "waiting for captured audio"}</strong>
        </div>
        {selectedAudioUrl ? (
          <audio ref={audioRef} key={selectedAudioUrl} controls src={selectedAudioUrl} />
        ) : (
          <p>{selectedLane?.error ?? "Live lanes expose playback only after real waveform bytes are captured."}</p>
        )}
      </div>

      {summary && (
        <div className="benchmark-footer">
          <span>Completed {summary.count}/{summary.concurrency}</span>
          <span>Failed {summary.failed_streams}</span>
          <span>Max {formatMs(summary.max_ttft_ms)}</span>
          <span>{summary.source}</span>
          {summary.queued_streams ? <span>{summary.queued_streams} streams include queue wait beyond {summary.active_slot_limit} active slots</span> : null}
          <span>{summary.source === "live_triton" ? "Click lanes with audio to verify real samples" : "Simulated run: no fake audio attached"}</span>
        </div>
      )}
    </section>
  );
}

function SummaryMetric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="summary-metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ttftClass(value?: number): string {
  if (value === undefined) return "ttft-pending";
  if (value < 150) return "ttft-fast";
  if (value < 250) return "ttft-good";
  if (value < 500) return "ttft-warm";
  return "ttft-hot";
}

function slotLabel(summary: ConcurrencySummary | null): string {
  if (!summary) {
    return "-";
  }
  if (summary.active_slot_limit) {
    return `${summary.active_slot_limit} / ${summary.queued_streams ?? 0}`;
  }
  return summary.source || "-";
}
