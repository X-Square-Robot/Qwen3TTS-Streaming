import {Gauge, RadioTower, ShieldCheck, TriangleAlert} from "lucide-react";

import type {LabCapabilities} from "../types";

interface CapabilitiesStripProps {
  readonly capabilities: LabCapabilities;
  readonly error?: string;
}

/** Compact, factual status surface for the optional engineering backend. */
export function CapabilitiesStrip({capabilities, error}: CapabilitiesStripProps) {
  const backends = capabilities.backends ?? [];
  const releaseStage = capabilities.release?.stage?.replaceAll("_", " ");
  return <section className="lab-capabilities" aria-label="工程后端能力" data-testid="lab-capabilities">
    <div className="lab-capabilities-heading">
      <span className="panel-kicker">BACKEND CAPABILITIES</span>
      <span className="lab-capabilities-release"><ShieldCheck size={14}/>{releaseStage || "configured"}</span>
    </div>
    <div className="lab-capabilities-grid">
      {backends.map((backend) => <span className={`lab-capability-pill ${backend.live_available ? "is-live" : "is-muted"}`} key={backend.id}>
        <RadioTower size={14}/><strong>{backend.label || backend.id}</strong><small>{backend.live_available ? "live" : backend.error ? "error" : "offline"}</small>
      </span>)}
      {capabilities.concurrency && <span className="lab-capability-pill">
        <Gauge size={14}/><strong>Concurrency</strong><small>{capabilities.concurrency.live_enabled ? "live enabled" : "simulated"}</small>
      </span>}
      {capabilities.headline?.single_stream_cache_hit_ttft_ms !== undefined && <span className="lab-capability-pill is-metric">
        <Gauge size={14}/><strong>Cache-hit TTFT</strong><small>{formatMetric(capabilities.headline.single_stream_cache_hit_ttft_ms)}</small>
      </span>}
      {capabilities.headline?.concurrent_128_avg_ttft_ms !== undefined && <span className="lab-capability-pill is-metric">
        <Gauge size={14}/><strong>128-stream avg</strong><small>{formatMetric(capabilities.headline.concurrent_128_avg_ttft_ms)}</small>
      </span>}
    </div>
    {error && <p className="lab-capabilities-error"><TriangleAlert size={14}/>{error}</p>}
    {!error && capabilities.limitations && capabilities.limitations.length > 0 && <p className="lab-capabilities-note">
      {capabilities.limitations.slice(0, 2).join(" · ")}
    </p>}
  </section>;
}

function formatMetric(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(value < 100 ? 1 : 0)} ms` : "—";
}
