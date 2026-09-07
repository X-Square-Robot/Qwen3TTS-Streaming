/**
 * Data contracts for the optional engineering-lab API.
 *
 * These types intentionally live outside the Browser SDK contract.  The lab
 * API is a diagnostic surface and may expose fields that are not part of the
 * public synthesis protocol (for example queueing and simulated-LLM timing).
 */

export type LabBackendId =
  | "triton_streaming"
  | "triton_offline"
  | "triton_trt_streaming"
  | (string & {});

export interface LabAudioFormat {
  encoding: string;
  sample_rate: number;
  channels: number;
}

export interface LabAudioAsset {
  id?: string;
  url: string;
  encoding?: string;
  sample_rate?: number;
  scheduled_start_ms?: number;
  source?: string;
}

export interface LabTraceEvent {
  run_id?: string;
  backend?: LabBackendId;
  type: string;
  t_ms: number;
  server_t_ms?: number;
  stream_id?: string;
  text?: string;
  meta?: Record<string, unknown>;
}

export interface LabRunMetrics {
  first_playable_ms?: number;
  total_ms?: number;
  server_ttft_ms?: number;
  triton_adapter_ttft_ms?: number;
  engine_internal_ttft_ms?: number;
  client_ttfb_ms?: number;
  first_audible_ms?: number;
  full_audio_ready_ms?: number;
  audio_duration_ms?: number;
  simulated_llm_complete_ms?: number;
  chunks?: number;
  cache_hit?: boolean;
}

export interface LabRunResult {
  run_id: string;
  backend: LabBackendId;
  label: string;
  mode: string;
  source: string;
  metrics: LabRunMetrics;
  events: LabTraceEvent[];
  warnings: string[];
  audio_format: LabAudioFormat;
  audio?: LabAudioAsset;
}

export interface LabRequest {
  text: string;
  speaker: string;
  language: string;
  ms_per_token: number;
}

export interface LabResult {
  type: "llm_pk_result";
  request: LabRequest;
  results: LabRunResult[];
  warnings: string[];
  release?: {
    stage?: string;
    positioning?: string;
    recommended_variant?: string;
    stable_paths?: string[];
    experimental_paths?: string[];
    planned_paths?: string[];
  };
  limitations?: string[];
}

export interface LabCapabilities {
  default_request?: LabRequest;
  backends?: Array<{
    id: LabBackendId;
    label?: string;
    streaming?: boolean;
    live_available?: boolean;
    error?: string;
    runtime?: {max_batch_slots?: number; max_sessions?: number};
  }>;
  concurrency?: {
    live_enabled?: boolean;
    triton_active_slot_limit?: number;
    triton_max_sessions?: number;
  };
  release?: LabResult["release"];
  limitations?: string[];
  headline?: {
    single_stream_cache_hit_ttft_ms?: number;
    concurrent_128_avg_ttft_ms?: number;
  };
}

export interface LabLaneUpdate {
  type: "lane_update";
  job_id: string;
  stream_id: string;
  status: string;
  ttft_ms?: number;
  queued_by_slot_limit?: boolean;
  elapsed_ms?: number;
  error?: string;
  audio?: LabAudioAsset;
}

export interface LabConcurrencySummary {
  type: "summary";
  job_id: string;
  source: string;
  concurrency: number;
  failed_streams: number;
  elapsed_ms: number;
  active_slot_limit?: number;
  queued_streams?: number;
  count: number;
  avg_ttft_ms?: number;
  active_avg_ttft_ms?: number;
  queued_avg_ttft_ms?: number;
  p50_ttft_ms?: number;
  p90_ttft_ms?: number;
  p99_ttft_ms?: number;
  max_ttft_ms?: number;
  throughput_audio_sec_per_sec?: number;
  error?: string;
}

export type LabSocketEvent =
  | {type: "job_started"; job_id: string; concurrency?: number; source?: string}
  | LabLaneUpdate
  | LabConcurrencySummary
  | {type: "heartbeat"}
  | {type: "error"; message?: string};
