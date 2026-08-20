import {expect, test, type Page} from "@playwright/test";

async function installFakeAudio(page: Page) {
  await page.addInitScript(() => {
    Object.defineProperty(globalThis.crypto, "randomUUID", {
      configurable: true,
      value: undefined,
    });
    class FakeSource {
      buffer: AudioBuffer | null = null;
      onended: (() => void) | null = null;
      connect() {}
      disconnect() {}
      start() { queueMicrotask(() => this.onended?.()); }
      stop() {}
    }
    class FakeContext {
      sampleRate = 48_000;
      currentTime = 0;
      destination = {};
      createGain() { return {gain: {setValueAtTime() {}}, connect: () => this.destination, disconnect() {}}; }
      createBuffer(_channels: number, frames: number) {
        return {getChannelData: () => new Float32Array(frames)};
      }
      createBufferSource() { return new FakeSource(); }
      async resume() {}
      async suspend() {}
      async close() {}
    }
    Object.assign(globalThis, {AudioContext: FakeContext, AudioWorkletNode: undefined});
  });
}

test("synthesizes mock PCM through the browser playback contract and exports WAV", async ({page}) => {
  await installFakeAudio(page);
  let route: Parameters<Parameters<typeof page.routeWebSocket>[1]>[0] | undefined;
  const messages: Array<{type: string}> = [];
  await page.routeWebSocket(/\/infer\/instance\/v1\/realtime$/, (socket) => {
    route = socket;
    socket.onMessage((message) => {
      messages.push(JSON.parse(String(message)) as {type: string});
    });
  });
  await page.goto("/infer/instance/demo/");
  await expect(page.getByRole("button", {name: "合成并播放"})).toBeEnabled();
  await page.getByRole("button", {name: "合成并播放"}).click();
  await expect.poll(() => route !== undefined).toBe(true);
  route?.send(JSON.stringify({type: "session.created", session: {id: "sess_e2e"}}));
  await expect.poll(() => messages.some((event) => event.type === "session.update")).toBe(true);
  route?.send(JSON.stringify({type: "session.updated", session: {id: "sess_e2e"}}));
  await expect.poll(() => messages.some((event) => event.type === "response.create")).toBe(true);
  route?.send(JSON.stringify({type: "response.created", response: {id: "resp_e2e"}}));
  const pcm = Buffer.alloc(2400 * 2);
  for (let index = 0; index < 2400; index += 1) pcm.writeInt16LE(Math.round(Math.sin(index / 10) * 3000), index * 2);
  route?.send(JSON.stringify({
    type: "response.output_audio.delta", response_id: "resp_e2e", delta: pcm.toString("base64"),
    qwen_delivery_seq: 1, qwen_output_sample_start: 0, qwen_output_sample_end: 2400,
  }));
  route?.send(JSON.stringify({type: "response.done", qwen_delivery_seq: 2, response: {
    id: "resp_e2e", status: "completed", usage: {audio_tokens: 10},
    metadata: {qwen_server_ttft_ms: "8.5", qwen_server_total_ms: "40", server_prefix_trimmed_ms: "12", server_prefix_trim_applied: "true", vad_strategy: "energy"},
  }}));
  await expect(page.getByRole("link", {name: "下载 WAV"})).toBeVisible();
  await expect(page.getByText(/已切换到兼容播放模式/)).toBeVisible();
  await expect(page.getByText("0.10 s")).toBeVisible();
  await expect(page.getByText("9 ms", {exact: true})).toBeVisible();
  await expect(page.getByText("12.0 ms · energy")).toBeVisible();
});

test("keeps a guarded-delivery tail longer than five seconds in automatic playback", async ({page}) => {
  await installFakeAudio(page);
  await page.routeWebSocket(/\/infer\/instance\/v1\/realtime$/, (socket) => {
    socket.send(JSON.stringify({type: "session.created", session: {id: "sess_long_tail"}}));
    socket.onMessage((message) => {
      const event = JSON.parse(String(message)) as {type: string};
      if (event.type === "session.update") {
        socket.send(JSON.stringify({type: "session.updated", session: {id: "sess_long_tail"}}));
      }
      if (event.type === "response.create") {
        socket.send(JSON.stringify({type: "response.created", response: {id: "resp_long_tail"}}));
        const samples = 24_000 * 6;
        socket.send(JSON.stringify({
          type: "response.output_audio.delta",
          response_id: "resp_long_tail",
          delta: Buffer.alloc(samples * 2).toString("base64"),
          qwen_delivery_seq: 1,
          qwen_output_sample_start: 0,
          qwen_output_sample_end: samples,
        }));
        socket.send(JSON.stringify({
          type: "response.done",
          qwen_delivery_seq: 2,
          response: {id: "resp_long_tail", status: "completed"},
        }));
      }
    });
  });
  await page.goto("/infer/instance/demo/");
  await page.getByLabel("合成文本").fill("不说话，只吃菜。一说话就紧张。一个好人，真不错。不错啊，真不错。");
  await page.getByRole("button", {name: "合成并播放"}).click();
  await expect(page.getByRole("link", {name: "下载 WAV"})).toBeVisible();
  await expect(page.getByText("6.00 s")).toBeVisible();
  await expect(page.getByText(/播放缓冲失败|播放尾帧提交失败/)).toHaveCount(0);
});

test("explains when output VAD filters the complete result", async ({page}) => {
  await installFakeAudio(page);
  await page.routeWebSocket(/\/infer\/instance\/v1\/realtime$/, (socket) => {
    socket.send(JSON.stringify({type: "session.created", session: {id: "sess_vad"}}));
    socket.onMessage((message) => {
      const event = JSON.parse(String(message)) as {type: string};
      if (event.type === "session.update") {
        socket.send(JSON.stringify({type: "session.updated", session: {id: "sess_vad"}}));
      }
      if (event.type === "response.create") {
        socket.send(JSON.stringify({type: "response.created", response: {id: "resp_vad"}}));
        socket.send(JSON.stringify({
          type: "response.done",
          qwen_delivery_seq: 1,
          response: {
            id: "resp_vad",
            status: "completed",
            metadata: {
              server_prefix_trimmed_ms: "4000",
              server_prefix_trim_applied: "true",
              vad_strategy: "energy",
            },
          },
        }));
      }
    });
  });
  await page.goto("/infer/instance/demo/");
  await page.getByLabel("输出 VAD").selectOption("energy");
  await expect(page.getByLabel("Begin threshold")).toHaveValue("0.3");
  await expect(page.getByLabel("End threshold")).toHaveValue("0.2");
  await page.getByRole("button", {name: "合成并播放"}).click();
  await expect(page.getByText("无有效音频", {exact: true})).toBeVisible();
  await expect(page.getByRole("status")).toContainText("输出 VAD（energy）过滤了整段音频");
  await expect(page.getByRole("link", {name: "下载 WAV"})).toHaveCount(0);
});

test("keeps an instance prefix and gates controls from capabilities", async ({page}) => {
  await page.goto("/infer/instance/demo/");
  await expect(page.getByRole("heading", {name: "让文字，即刻成为声音。"})).toBeVisible();
  await expect(page.getByLabel("任务").locator("option")).toHaveCount(2);
  await expect(page.getByLabel("输出 VAD").locator("option")).toHaveCount(2);
  await page.getByLabel("输出 VAD").selectOption("energy");
  await expect(page.getByLabel("Begin threshold")).toBeVisible();
  await expect(page.getByLabel("Begin threshold")).toHaveValue("0.3");
  await expect(page.getByLabel("End threshold")).toHaveValue("0.2");
  await page.getByLabel("任务").selectOption("voice_design");
  await page.getByLabel("输入方式").selectOption("incremental");
  await expect(page.getByText(/工程预览能力/)).toBeVisible();
  await page.getByRole("link", {name: "SDK"}).click();
  await expect(page.getByRole("link", {name: /下载 qwen3_tts_client/})).toHaveAttribute(
    "href", "http://127.0.0.1:4173/infer/instance/sdk/qwen3_tts_client-1.2.3-py3-none-any.whl",
  );
  await expect(page.getByText('npm install "http://127.0.0.1:4173/infer/instance/demo/downloads/xmultimodalinteraction-qwen3tts-browser-1.2.3.tgz"')).toBeVisible();
  await expect(page.getByRole("link", {name: "下载 npm tarball"})).toHaveAttribute(
    "href", "http://127.0.0.1:4173/infer/instance/demo/downloads/xmultimodalinteraction-qwen3tts-browser-1.2.3.tgz",
  );
  await expect(page.getByText("Engine、Python SDK、Browser SDK 与文档来自同一 release。")).toBeVisible();
  await expect(page.getByText(/SynthesisTask\.VoiceDesign/)).toBeVisible();
  await expect(page.getByText(/strategy: VadStrategy\.Energy/)).toBeVisible();
  await expect(page.getByText(/client\.startIncremental\(options\)/)).toBeVisible();
  await expect(page.getByText(/client\.open_stream\(SessionStartRequest/)).toBeVisible();
});

test("runs the built-in LLM comparison through public Realtime", async ({page}) => {
  let sequence = 0;
  await page.routeWebSocket(/\/infer\/instance\/v1\/realtime$/, (socket) => {
    const id = ++sequence;
    socket.send(JSON.stringify({type: "session.created", session: {id: `lab_${id}`}}));
    socket.onMessage((message) => {
      const event = JSON.parse(String(message)) as {type: string};
      if (event.type === "session.update") {
        socket.send(JSON.stringify({type: "session.updated", session: {id: `lab_${id}`}}));
      }
      if (event.type === "response.create") {
        socket.send(JSON.stringify({type: "response.created", response: {id: `resp_lab_${id}`}}));
        setTimeout(() => {
          socket.send(JSON.stringify({
            type: "response.output_audio.delta", response_id: `resp_lab_${id}`,
            delta: Buffer.alloc(480).toString("base64"), qwen_delivery_seq: 1,
            qwen_output_sample_start: 0, qwen_output_sample_end: 240,
          }));
          socket.send(JSON.stringify({
            type: "response.done", qwen_delivery_seq: 2,
            response: {id: `resp_lab_${id}`, status: "completed", usage: {audio_tokens: 1}},
          }));
        }, 300);
      }
    });
  });
  await page.goto("/infer/instance/demo/#/lab");
  await expect(page.getByRole("heading", {name: "同一入口，观察不同负载。"})).toBeVisible();
  await page.getByRole("button", {name: "运行 LLM PK"}).click();
  await expect(page.getByText("增量文本", {exact: true})).toBeVisible();
  await expect(page.getByText("完整文本", {exact: true})).toBeVisible();
  await expect(page.getByRole("button", {name: "下载 JSON trace"})).toBeVisible();
  expect(sequence).toBe(2);
});

test("Pages stays useful in docs-only mode", async ({page}) => {
  await page.goto("/pages/#/docs/quickstart-zh");
  await expect(page.getByRole("heading", {name: "先让声音出来，再按需深入。"})).toBeVisible();
  await expect(page.getByRole("heading", {name: "5 分钟接入", exact: true})).toBeVisible();
  const docsNav = page.getByLabel("文档层级");
  await expect(docsNav.locator(".docs-nav-group > header strong")).toHaveText([
    "快速接入", "高级配置", "更多细节",
  ]);
  await expect(page.locator("article.markdown .hljs-keyword").first()).toBeVisible();
  await page.getByRole("button", {name: "EN"}).click();
  await expect(page.getByRole("heading", {name: "5-minute setup", exact: true})).toBeVisible();
  await page.getByRole("link", {name: "体验"}).click();
  await expect(page.getByRole("button", {name: "合成并播放"})).toBeDisabled();
  await page.getByRole("link", {name: "SDK"}).click();
  await expect(page.getByText(/只读文档模式/)).toBeVisible();
  await expect(page.getByRole("heading", {name: "当前参数代码"})).toHaveCount(0);
});
