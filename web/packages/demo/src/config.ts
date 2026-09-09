export interface PythonSdkConfig {
  available: boolean;
  project: string;
  version?: string;
  filename?: string;
  sha256?: string;
  index_url: string;
  download_url?: string;
  reason?: string;
}

export interface DemoConfig {
  schema_version: "qwen.tts.demo-config.v1";
  engine_version: string;
  runtime_type: "standalone" | "triton";
  endpoints: {
    capabilities_url: string;
    openai_realtime_url: string;
    native_websocket_url: string;
  };
  python_sdk: PythonSdkConfig;
  browser_sdk: {
    available: boolean;
    package: string;
    version: string;
    registry_url: string;
    tarball_url: string;
  };
  docs: {version: string; route: string};
}

export interface LoadedDemoConfig {
  config: DemoConfig;
  responseUrl: URL;
}

export async function loadDemoConfig(): Promise<LoadedDemoConfig> {
  const response = await fetch("./config.json", {headers: {Accept: "application/json"}});
  if (!response.ok) throw new Error(`Demo config failed with HTTP ${response.status}`);
  const config = await response.json() as DemoConfig;
  if (config.schema_version !== "qwen.tts.demo-config.v1") {
    throw new Error("Unsupported Demo config schema");
  }
  return {config, responseUrl: new URL(response.url)};
}
