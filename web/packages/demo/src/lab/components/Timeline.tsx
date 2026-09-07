import type {LabTraceEvent} from "../types";

const EVENT_LABELS: Record<string, string> = {
  request_started: "request",
  official_approx_ttft: "ttft approx",
  first_audio_chunk: "first audio",
  first_audible: "audible",
  full_audio_ready: "full wav",
  done: "done",
};

export function Timeline({events, maxMs}: {events: LabTraceEvent[]; maxMs: number}) {
  const visible = events.filter((event) => [
    "request_started", "official_approx_ttft", "first_audio_chunk", "first_audible",
    "full_audio_ready", "done",
  ].includes(event.type));
  return <div className="lab-timeline" aria-label="event timeline">
    <div className="lab-timeline-track" />
    {visible.map((event, index) => {
      const left = Math.max(0, Math.min(100, event.t_ms / Math.max(maxMs, 1) * 100));
      return <div className="lab-timeline-marker" style={{left: `${left}%`}} key={`${event.type}-${index}`}>
        <span className={`lab-marker-dot marker-${event.type}`} />
        <span className="lab-marker-label">{EVENT_LABELS[event.type] ?? event.type}</span>
        <span className="lab-marker-time">{formatMs(event.t_ms)}</span>
      </div>;
    })}
  </div>;
}

export function formatMs(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(value < 100 ? 1 : 0)}ms`;
}
