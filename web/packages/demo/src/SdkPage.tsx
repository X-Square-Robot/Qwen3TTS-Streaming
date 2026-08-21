import {Download} from "lucide-react";
import {useState} from "react";
import {resolveRelativeUrl, type Capabilities} from "@xmultimodalinteraction/qwen3tts-browser";

import type {LoadedDemoConfig} from "./config";
import type {DemoSynthesisSettings} from "./demo-settings";

interface SdkPageProps {
  readonly loaded: LoadedDemoConfig | null;
  readonly capabilities: Capabilities | null;
  readonly settings: DemoSynthesisSettings;
  readonly docsOnly: boolean;
}

export function SdkPage({loaded, capabilities, settings, docsOnly}: SdkPageProps) {
  const sdk = loaded?.config.python_sdk;
  const download = !docsOnly && sdk?.download_url && loaded
    ? resolveRelativeUrl(sdk.download_url, loaded.responseUrl).toString()
    : "";
  const browserTarball = !docsOnly && loaded?.config.browser_sdk.tarball_url
    ? resolveRelativeUrl(loaded.config.browser_sdk.tarball_url, loaded.responseUrl).toString()
    : "";
  const browserPackage = loaded?.config.browser_sdk.package ?? "@xmultimodalinteraction/qwen3tts-browser";
  const browserVersion = loaded?.config.browser_sdk.version ?? "";
  const browserRegistry = loaded?.config.browser_sdk.registry_url ?? "";
  const browserInstall = browserTarball
    ? `npm install ${JSON.stringify(browserTarball)}`
    : browserRegistry && browserVersion
      ? `npm install ${browserPackage}@${browserVersion} --registry ${JSON.stringify(browserRegistry)}`
      : "当前实例未内置 Browser SDK 安装包";
  const command = download
    ? `pip install "${sdk?.project ?? "qwen3-tts-client"}[all] @ ${download}"`
    : "请在部署实例的 /demo/#/sdk 获取匹配 wheel 的安装命令";
  const capabilitiesUrl = loaded ? resolveRelativeUrl(loaded.config.endpoints.capabilities_url, loaded.responseUrl).toString() : "";
  const nativeUrl = loaded ? websocketEndpoint(resolveRelativeUrl(loaded.config.endpoints.native_websocket_url, loaded.responseUrl)).toString() : "";
  const realtimeUrl = loaded ? websocketEndpoint(resolveRelativeUrl(loaded.config.endpoints.openai_realtime_url, loaded.responseUrl)).toString() : "";
  const pythonExample = buildPythonExample(nativeUrl, settings);
  const browserExample = buildBrowserExample(capabilitiesUrl, realtimeUrl, settings);
  const versions = {
    engine: loaded?.config.engine_version ?? "",
    python: sdk?.version ?? "",
    browser: loaded?.config.browser_sdk.version ?? "",
    docs: loaded?.config.docs.version ?? "",
  };
  const coherent = loaded ? sameRelease(Object.values(versions).filter(Boolean)) : false;

  return <section className="page"><p className="eyebrow">SDK DISTRIBUTION</p><h1>与当前实例精确匹配。</h1>
    {docsOnly && <p className="alert">Pages 当前处于只读文档模式；未连接实例，因此下载和实例代码生成已禁用。</p>}
    <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">PYTHON CLIENT</p><h2>Python SDK</h2></div><p>从当前实例下载严格匹配的 wheel。</p></div><pre>{command}</pre>
      {download && <a className="button primary" href={download}><Download size={17}/>下载 {sdk?.filename}</a>}
      {!docsOnly && !sdk?.available && <p className="alert">{sdk?.reason}</p>}
      {sdk?.sha256 && !docsOnly && <p className="hash">SHA256 {sdk.sha256}</p>}
    </div>
    <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">WEB CLIENT</p><h2>Browser SDK</h2></div><p>用于网页实时合成与播放。</p></div>
      <pre>{browserInstall}</pre>
      {browserTarball && <a className="button" href={browserTarball} download><Download size={17}/>下载 npm tarball</a>}
      {browserRegistry && browserVersion && browserTarball && <><p className="hint">也可以从已配置 Registry 安装：</p>
        <pre>npm install {browserPackage}@{browserVersion} --registry {JSON.stringify(browserRegistry)}</pre></>}
      {!docsOnly && !loaded?.config.browser_sdk.available && <p className="alert">Browser SDK 尚未由当前实例或 Registry 提供。</p>}
      <p className="hint">协议 {capabilities?.schema_version ?? "连接实例后显示"}</p>
    </div>
    {!docsOnly && loaded && <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">RELEASE PAIRING</p><h2>Release 一致性</h2></div></div>
      <p className={coherent ? "hint" : "alert"}>{coherent ? "Engine、Python SDK、Browser SDK 与文档来自同一 release。" : "版本不一致，请勿混用当前产物。"}</p>
      <pre>{Object.entries(versions).map(([name, version]) => `${name.padEnd(8)} ${version || "unavailable"}`).join("\n")}</pre>
    </div>}
    {!docsOnly && loaded && capabilities && <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">LIVE CONFIGURATION</p><h2>当前参数代码</h2></div><p>复制即可连接当前公共入口。</p></div>
      <p className="hint">代码随“体验”页当前 task、声音、语言、采样率、VAD 与交付策略更新，只使用公共入口。</p>
      <CodeExample title="Python" code={pythonExample}/><CodeExample title="Browser" code={browserExample}/>
    </div>}
    {!docsOnly && loaded && !capabilities && <p className="alert">尚未取得当前实例 capabilities；参数代码生成保持禁用。</p>}
  </section>;
}

export function buildPythonExample(nativeUrl: string, value: DemoSynthesisSettings): string {
  const vadEnabled = value.vad !== "disabled";
  const imports = value.inputMode === "full"
    ? "AudioFormat, OutputPolicy, TTSClient, SynthesisConfig, VADPolicy"
    : "AudioChunk, AudioFormat, OutputPolicy, SessionStartRequest, TTSClient, SynthesisConfig, VADPolicy";
  const execute = value.inputMode === "full"
    ? `result = client.synthesize_bytes("你好，欢迎使用 Qwen3-TTS。", request=config)\nopen("qwen3tts.pcm", "wb").write(result.audio_bytes)`
    : `session = client.open_stream(SessionStartRequest(session_id="demo", config=config))\nsession.send_text("你好，")\nsession.send_text("欢迎使用 Qwen3-TTS。")\nsession.end()\nwith open("qwen3tts.pcm", "wb") as output:\n    for message in session.iter_messages():\n        if isinstance(message, AudioChunk):\n            output.write(message.pcm_bytes)`;
  return `from qwen3tts import (\n    ${imports},\n)\n\nclient = TTSClient.connect(${JSON.stringify(nativeUrl)})\nconfig = SynthesisConfig(\n    task_type=${JSON.stringify(value.task)},\n    speaker=${JSON.stringify(value.speaker)},\n    language=${JSON.stringify(value.language)},\n    input_mode=${JSON.stringify(value.inputMode === "full" ? "full_text" : "token")},\n    audio=AudioFormat(encoding="pcm_s16le", sample_rate=${value.sampleRate}, channels=1),\n    output_policy=OutputPolicy(\n        vad=VADPolicy(\n            enabled=${vadEnabled ? "True" : "False"}, strategy=${JSON.stringify(value.vad)},\n            chunk_ms=${value.vadChunkMs}, begin_threshold=${value.vadBeginThreshold},\n            begin_count=${value.vadBeginCount}, end_threshold=${value.vadEndThreshold},\n            end_count=${value.vadEndCount}, start_margin_ms=${value.vadStartMarginMs},\n        ),\n        chunk_ms=${value.outputChunkMs}, emit_text_events=${value.emitTextEvents ? "True" : "False"},\n        config={"delivery": ${JSON.stringify(value.delivery)}, "delivery_window_ms": ${value.deliveryWindowMs}},\n    ),\n)\n${execute}`;
}

export function buildBrowserExample(capabilitiesUrl: string, realtimeUrl: string, value: DemoSynthesisSettings): string {
  const task = enumMember({base: "Base", voice_clone: "VoiceClone", custom_voice: "CustomVoice", voice_design: "VoiceDesign"}, value.task);
  const vad = enumMember({disabled: "Disabled", energy: "Energy", tenvad: "TenVad"}, value.vad);
  const delivery = enumMember({guarded: "Guarded", firehose: "Firehose"}, value.delivery);
  const input = value.inputMode === "full" ? "FullText" : "Token";
  const start = value.inputMode === "full"
    ? `const run = await client.synthesize("你好，欢迎使用 Qwen3-TTS。", options);`
    : `const run = await client.startIncremental(options);\nrun.append("你好，");\nrun.append("欢迎使用 Qwen3-TTS。");\nrun.commit();`;
  return `import {\n  AudioEncoding, DeliveryPolicy, InputMode, RealtimeTTSClient,\n  SynthesisTask, VadStrategy,\n} from "@xmultimodalinteraction/qwen3tts-browser";\n\nconst client = new RealtimeTTSClient({\n  capabilitiesUrl: ${JSON.stringify(capabilitiesUrl)},\n  websocketUrl: ${JSON.stringify(realtimeUrl)},\n});\nawait client.connect();\nconst options = {\n  task: SynthesisTask.${task}, speaker: ${JSON.stringify(value.speaker)}, language: ${JSON.stringify(value.language)},\n  inputMode: InputMode.${input},\n  audio: {encoding: AudioEncoding.PcmS16Le, sample_rate: ${value.sampleRate}, channels: 1},\n  vad: {enabled: ${value.vad !== "disabled"}, strategy: VadStrategy.${vad}, chunk_ms: ${value.vadChunkMs},\n    begin_threshold: ${value.vadBeginThreshold}, begin_count: ${value.vadBeginCount},\n    end_threshold: ${value.vadEndThreshold}, end_count: ${value.vadEndCount}, start_margin_ms: ${value.vadStartMarginMs}},\n  outputPolicy: {delivery: DeliveryPolicy.${delivery}, delivery_window_ms: ${value.deliveryWindowMs},\n    chunk_ms: ${value.outputChunkMs}, emit_text_events: ${value.emitTextEvents}},\n};\n${start}\nawait run.done;`;
}

function enumMember(mapping: Record<string, string>, value: string): string { return mapping[value] ?? Object.values(mapping)[0] ?? ""; }

function sameRelease(values: string[]): boolean {
  return values.length > 0 && new Set(values.map(releaseIdentity)).size === 1;
}

function releaseIdentity(value: string): string {
  return value.replace(/^v/, "").replace(
    /-(alpha|beta|rc|dev)\.(\d+)$/,
    (_match, stage: string, number: string) => `${stage === "alpha" ? "a" : stage === "beta" ? "b" : stage === "dev" ? ".dev" : "rc"}${number}`,
  );
}

function CodeExample({title, code}: {title: string; code: string}) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");
  async function copy() {
    try { await navigator.clipboard.writeText(code); setStatus("copied"); setTimeout(() => setStatus("idle"), 1200); }
    catch { setStatus("failed"); }
  }
  return <div><div className="actions"><strong>{title}</strong><button onClick={() => void copy()}>
    {status === "copied" ? "已复制" : status === "failed" ? "复制失败" : "复制"}
  </button></div><pre>{code}</pre></div>;
}

function websocketEndpoint(url: URL): URL {
  const result = new URL(url);
  result.protocol = result.protocol === "https:" ? "wss:" : "ws:";
  return result;
}
