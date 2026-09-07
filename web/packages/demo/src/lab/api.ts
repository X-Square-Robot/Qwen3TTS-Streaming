import type {LoadedDemoConfig} from "../config";
import type {
  LabCapabilities,
  LabRequest,
  LabResult,
} from "./types";

/** A prefix-safe client for the optional demo_api engineering backend. */
export interface LabApi {
  readonly baseUrl: URL;
  readonly capabilitiesUrl: URL;
  readonly llmPkUrl: URL;
  readonly concurrencyUrl: URL;
  readonly trtLiveUrl: URL;
  readonly concurrencyWsUrl: (jobId: string) => URL;
  readonly audioUrl: (path: string) => URL;
  readonly getCapabilities: (signal?: AbortSignal) => Promise<LabCapabilities>;
  readonly runLlmPk: (request: LabRequest, signal?: AbortSignal) => Promise<LabResult>;
  readonly startConcurrency: (
    request: LabRequest & {concurrency: number; live: boolean},
    signal?: AbortSignal,
  ) => Promise<string>;
  readonly openTrtLive: () => WebSocket;
}

/**
 * Resolve the lab URL from the instance config.  `config.json` can itself be
 * served below an ingress prefix, so all relative URLs must use its actual
 * response URL as the base rather than `window.location.origin`.
 */
export function createLabApi(loaded: LoadedDemoConfig | null): LabApi | null {
  const raw = loaded?.config.lab?.url?.trim() ?? "";
  if (!loaded || !raw || !loaded.config.lab?.available) return null;

  let resolved: URL;
  try {
    resolved = new URL(raw, loaded.responseUrl);
  } catch {
    return null;
  }
  // The runtime config builder already filters this value, but keep the
  // browser boundary fail-closed when a hand-written config is deployed.
  if (resolved.protocol !== "http:" && resolved.protocol !== "https:") return null;
  const baseUrl = withTrailingSlash(resolved);
  const api = (path: string) => new URL(path.replace(/^\//, ""), baseUrl);
  const capabilitiesUrl = api("api/v1/capabilities");
  const llmPkUrl = api("api/v1/llm-pk");
  const concurrencyUrl = api("api/v1/concurrency");
  const trtLiveUrl = toWebSocketUrl(api("api/v1/trt-live"));

  return {
    baseUrl,
    capabilitiesUrl,
    llmPkUrl,
    concurrencyUrl,
    trtLiveUrl,
    concurrencyWsUrl: (jobId) => toWebSocketUrl(api(`api/v1/concurrency/${encodeURIComponent(jobId)}`)),
    audioUrl: (path) => new URL(path.replace(/^\//, ""), baseUrl),
    getCapabilities: (signal) => requestJson<LabCapabilities>(capabilitiesUrl, signal ? {signal} : {}),
    runLlmPk: (request, signal) => requestJson<LabResult>(llmPkUrl, {
      method: "POST",
      body: JSON.stringify(request),
      ...(signal ? {signal} : {}),
    }),
    startConcurrency: async (request, signal) => {
      const result = await requestJson<{job_id: string}>(concurrencyUrl, {
        method: "POST",
        body: JSON.stringify(request),
      ...(signal ? {signal} : {}),
      });
      if (!result.job_id) throw new Error("Lab API returned no job_id");
      return result.job_id;
    },
    openTrtLive: () => new WebSocket(trtLiveUrl),
  };
}

export function toWebSocketUrl(value: URL): URL {
  const result = new URL(value);
  if (result.protocol === "http:") result.protocol = "ws:";
  else if (result.protocol === "https:") result.protocol = "wss:";
  return result;
}

async function requestJson<T>(url: URL, init: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init.body ? {"content-type": "application/json"} : {}),
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Lab API ${response.status}${detail ? `: ${detail.slice(0, 240)}` : ""}`);
  }
  return await response.json() as T;
}

function withTrailingSlash(value: URL): URL {
  const result = new URL(value);
  if (!result.pathname.endsWith("/")) result.pathname += "/";
  return result;
}
