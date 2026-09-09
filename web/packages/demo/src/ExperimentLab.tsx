import {useRef, useState} from "react";
import {
  AudioEncoding, discoverCapabilities, InputMode, RealtimeTTSClient, resolveRelativeUrl,
  SynthesisTask, VadStrategy, type IncrementalSynthesisRun, type SynthesisRun,
} from "@xmultimodalinteraction/qwen3tts-browser";

import type {LoadedDemoConfig} from "./config";

interface ExperimentResult {
  readonly name: string;
  readonly firstResponseMs: number;
  readonly firstAudioMs: number;
  readonly totalMs: number;
  readonly audioSeconds: number;
  readonly trace: ReadonlyArray<Record<string, unknown>>;
}

export function ExperimentLab({loaded}: {
  loaded: LoadedDemoConfig | null;
}) {
  const [text, setText] = useState("你好，这是从上游大模型逐步到达的流式文本。");
  const [concurrency, setConcurrency] = useState(4);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [error, setError] = useState("");
  const activeRuns = useRef<SynthesisRun[]>([]);
  const activeClients = useRef<RealtimeTTSClient[]>([]);

  async function run(mode: "full" | "incremental", name: string): Promise<ExperimentResult> {
    if (!loaded) throw new Error("当前页面未连接 TTS 实例");
    const capabilitiesUrl = resolveRelativeUrl(loaded.config.endpoints.capabilities_url, loaded.responseUrl);
    const realtimeUrl = toWebSocketUrl(resolveRelativeUrl(loaded.config.endpoints.openai_realtime_url, loaded.responseUrl));
    const capabilities = await discoverCapabilities(capabilitiesUrl);
    const task = capabilities.tasks.find(isSynthesisTask);
    const audio = capabilities.audio_formats.find((entry) => entry.encoding === AudioEncoding.PcmS16Le);
    if (!task || !audio) throw new Error("实例没有 Browser SDK 支持的 task 或 PCM16 格式");

    const client = new RealtimeTTSClient({capabilitiesUrl, websocketUrl: realtimeUrl});
    activeClients.current.push(client);
    let firstResponseMs = 0;
    let firstAudioMs = 0;
    let receivedSamples = 0;
    let startedAt = 0;
    const trace: Array<Record<string, unknown>> = [];
    client.onRawEvent((event) => {
      if (trace.length >= 200) trace.shift();
      trace.push({
        at_ms: startedAt > 0 ? Number((performance.now() - startedAt).toFixed(3)) : 0,
        type: event.type,
        response_id: event.response_id ?? (event.response as Record<string, unknown> | undefined)?.id,
        delivery_seq: event.qwen_delivery_seq,
        output_sample_start: event.qwen_output_sample_start,
        output_sample_end: event.qwen_output_sample_end,
        segment_id: event.segment_id,
        text: event.type === "qwen.text_progress" ? event.text : undefined,
      });
    });
    client.onEvent((event) => {
      if (event.type === "response_started" && firstResponseMs === 0) firstResponseMs = performance.now() - startedAt;
      if (event.type === "audio") {
        if (firstAudioMs === 0) firstAudioMs = performance.now() - startedAt;
        receivedSamples += event.pcm.length;
      }
    });
    try {
      await client.connect();
      startedAt = performance.now();
      const options = {
        task,
        speaker: capabilities.speakers?.[0] ?? "Serena",
        inputMode: mode === "full" ? InputMode.FullText : InputMode.Token,
        audio,
        vad: {enabled: false, strategy: VadStrategy.Disabled},
      };
      const synthesis = mode === "full" ? await client.synthesize(text, options) : await client.startIncremental(options);
      activeRuns.current.push(synthesis);
      if (mode === "incremental") await sendIncrementally(synthesis as IncrementalSynthesisRun, text);
      await synthesis.done;
      return {name, firstResponseMs, firstAudioMs, totalMs: performance.now() - startedAt, audioSeconds: receivedSamples / audio.sample_rate, trace};
    } finally {
      client.close();
    }
  }

  async function execute(work: () => Promise<ExperimentResult[]>) {
    setRunning(true); setError(""); setResults([]);
    activeRuns.current = []; activeClients.current = [];
    try { setResults(await work()); }
    catch (cause) { setError(String(cause)); }
    finally { activeRuns.current = []; activeClients.current = []; setRunning(false); }
  }

  function cancel() {
    for (const synthesis of activeRuns.current) synthesis.cancel();
    for (const client of activeClients.current) client.close();
    setRunning(false);
  }

  return <section className="page">
    <p className="eyebrow">ENGINEERING LAB · ONE PORTAL</p><h1>同一入口，观察不同负载。</h1>
    <p>所有实验直接使用当前实例的 Browser SDK 与公共 Realtime。事件 trace、LLM PK
      和多路并发都来自同一 Gateway 会话，不依赖独立实验后端。</p>
    <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">PUBLIC REALTIME</p><h2>基础实验输入</h2></div><p>浏览器直连当前实例；结果只代表本次请求。</p></div>
      <textarea value={text} onChange={(event) => setText(event.target.value)} />
      <div className="grid controls"><label>并发数<input type="number" min="1" max="16" value={concurrency}
        onChange={(event) => setConcurrency(clamp(Number(event.target.value), 1, 16))}/></label></div>
      <div className="actions">
        <button className="primary" disabled={running || !loaded || !text.trim()} onClick={() => void execute(() => Promise.all([
          run("incremental", "增量文本"), run("full", "完整文本"),
        ]))}>运行 LLM PK</button>
        <button disabled={running || !loaded || !text.trim()} onClick={() => void execute(() => Promise.all(
          Array.from({length: concurrency}, (_, index) => run("full", `并发 ${index + 1}`)),
        ))}>运行并发测试</button>
        <button disabled={!running} onClick={cancel}>取消全部</button>
      </div>{error && <p className="alert">{error}</p>}
    </div>
    <div className="panel"><div className="panel-heading"><div><p className="panel-kicker">LIVE RESULT</p><h2>本次实时结果</h2></div></div>
      <div className="metrics">{results.map((result) => <div key={result.name}><small>{result.name}</small>
        <strong>{formatMs(result.firstAudioMs)}</strong><small>response {formatMs(result.firstResponseMs)}</small>
        <small>total {formatMs(result.totalMs)} · {result.audioSeconds.toFixed(2)}s audio</small></div>)}</div>
      {results.length === 0 && <p className="hint">结果只代表当前浏览器到当前实例的本次请求，不是预置 benchmark。</p>}
      {results.length > 0 && <div className="actions"><button onClick={() => downloadTrace({
        schema_version: "qwen.tts.demo-trace.v1",
        generated_at: new Date().toISOString(),
        engine_version: loaded?.config.engine_version,
        text,
        results,
      })}>下载 JSON trace</button></div>}
      {results.length > 0 && <div className="event-log">{results.flatMap((result) => result.trace.slice(-4).map((event, index) =>
        <code key={`${result.name}-${index}`}>{result.name} · {String(event.type)} · {String(event.at_ms)}ms</code>))}</div>}
    </div>
  </section>;
}

async function sendIncrementally(run: IncrementalSynthesisRun, text: string): Promise<void> {
  for (const chunk of text.match(/.{1,4}/gu) ?? [text]) {
    run.append(chunk);
    await new Promise((resolve) => setTimeout(resolve, 35));
  }
  run.commit();
}

function isSynthesisTask(value: string): value is SynthesisTask {
  return Object.values(SynthesisTask).includes(value as SynthesisTask);
}

function toWebSocketUrl(url: URL): URL {
  const result = new URL(url);
  result.protocol = result.protocol === "https:" ? "wss:" : "ws:";
  return result;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, Math.round(value))) : minimum;
}

function formatMs(value: number): string { return value > 0 ? `${value.toFixed(1)} ms` : "—"; }

function downloadTrace(value: Record<string, unknown>): void {
  const url = URL.createObjectURL(new Blob([`${JSON.stringify(value, null, 2)}\n`], {type: "application/json"}));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "qwen3tts-demo-trace.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
