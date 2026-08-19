import {createServer} from "node:http";
import {readFile, stat} from "node:fs/promises";
import {extname, resolve, sep} from "node:path";

const root = resolve(import.meta.dirname, "../packages/demo/dist");
const prefix = "/infer/instance/demo/";
const browserTarball = "xmultimodalinteraction-qwen3tts-browser-1.2.3.tgz";
const capabilities = {
  schema_version: "qwen.tts.capabilities.v1", engine_version: "v1.2.3", model: "custom-1.7b",
  tasks: ["custom_voice", "voice_design"], speakers: ["Serena", "Ryan"], languages: ["auto", "Chinese", "English"],
  task_status: [{task: "custom_voice", available: true, stability: "stable"}, {task: "voice_design", available: true, stability: "experimental"}],
  input_modes: ["full_text", "token"],
  audio_formats: [{encoding: "pcm_s16le", sample_rate: 24000, channels: 1}],
  output_policy: {features: ["vad_policy", "chunk_ms", "emit_text_events", "guarded_delivery"], vad_strategies: ["disabled", "energy"]},
  limits: {max_input_tokens: 128, max_realtime_message_bytes: 8_388_608},
  reference: {available: false, max_duration_sec: 0, max_bytes: 4_194_304, mime_types: ["audio/wav", "audio/x-wav"], reason: "not loaded"},
  protocols: {openai_realtime: {
    path: "/v1/realtime", base: "openai-realtime-v1", extension_protocol: "qwen-realtime-v1",
    supported_extensions: ["qwen.input_text_buffer.v1", "qwen.text_progress.v1", "qwen.playback_ack.v1", "qwen.response_resume.v1"],
    features: ["active_response_resume", "playback_ack"], audio_formats: ["pcm_s16le"],
  }},
};

const server = createServer(async (request, response) => {
  try {
    const path = new URL(request.url ?? "/", "http://localhost").pathname;
    if (path === "/health") return send(response, 200, "text/plain", "ok");
    if (path === "/legacy-lab/healthz") return json(response, {ok: true});
    if (path === "/infer/instance/v1/capabilities") return json(response, capabilities);
    if (path === `${prefix}config.json`) return json(response, {
      schema_version: "qwen.tts.demo-config.v1", engine_version: "v1.2.3", runtime_type: "standalone",
      endpoints: {capabilities_url: "../v1/capabilities", openai_realtime_url: "../v1/realtime", native_websocket_url: "../v1/ws"},
      python_sdk: {available: true, project: "qwen3-tts-client", version: "1.2.3", filename: "qwen3_tts_client-1.2.3-py3-none-any.whl", sha256: "abcd", index_url: "../sdk/", download_url: "../sdk/qwen3_tts_client-1.2.3-py3-none-any.whl"},
      browser_sdk: {available: true, package: "@xmultimodalinteraction/qwen3tts-browser", version: "1.2.3", registry_url: "", tarball_url: `./downloads/${browserTarball}`},
      docs: {version: "v1.2.3", route: "./#/docs/"}, lab: {available: true, url: "/legacy-lab"},
    });
    if (path === "/infer/instance/sdk/qwen3_tts_client-1.2.3-py3-none-any.whl") return send(response, 200, "application/octet-stream", "wheel");
    if (path === `${prefix}downloads/${browserTarball}`) return send(response, 200, "application/gzip", "browser-sdk");
    const relative = path.startsWith(prefix) ? path.slice(prefix.length) : path.startsWith("/pages/") ? path.slice(7) : null;
    if (relative === null || relative === "config.json") return send(response, 404, "text/plain", "not found");
    const file = resolve(root, relative || "index.html");
    if (file !== root && !file.startsWith(`${root}${sep}`)) return send(response, 404, "text/plain", "not found");
    const target = (await stat(file)).isDirectory() ? resolve(file, "index.html") : file;
    send(response, 200, mime(target), await readFile(target));
  } catch {
    send(response, 404, "text/plain", "not found");
  }
});

server.listen(4173, "127.0.0.1");

function json(response, value) { send(response, 200, "application/json", JSON.stringify(value)); }
function send(response, status, type, body) { response.writeHead(status, {"Content-Type": type}); response.end(body); }
function mime(path) { return ({".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".gif": "image/gif"})[extname(path)] ?? "application/octet-stream"; }
