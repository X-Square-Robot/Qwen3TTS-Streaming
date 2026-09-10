import {expect, test, type Page} from "@playwright/test";

async function installFakeAudio(page: Page) {
  await page.addInitScript(() => {
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

async function mockAudio(page: Page, seconds = 103) {
  let sequence = 0;
  await page.routeWebSocket(/\/infer\/instance\/v1\/realtime$/, (socket) => {
    const id = `playback_${++sequence}`;
    socket.send(JSON.stringify({type: "session.created", session: {id}}));
    socket.onMessage((message) => {
      const event = JSON.parse(String(message)) as {type: string};
      if (event.type === "session.update") socket.send(JSON.stringify({type: "session.updated", session: {id}}));
      if (event.type === "response.create") {
        const responseId = `${id}_response`;
        socket.send(JSON.stringify({type: "response.created", response: {id: responseId}}));
        setTimeout(() => {
          const samples = 24000 * seconds;
          socket.send(JSON.stringify({type: "response.output_audio.delta", response_id: responseId,
            delta: Buffer.alloc(samples * 2).toString("base64"), qwen_delivery_seq: 1,
            qwen_output_sample_start: 0, qwen_output_sample_end: samples}));
          socket.send(JSON.stringify({type: "response.done", qwen_delivery_seq: 2, response: {id: responseId, status: "completed"}}));
        }, 120);
      }
    });
  });
}

test("listener replay shares compact controls and updates the playback cursor on seek", async ({page}, testInfo) => {
  await installFakeAudio(page);
  await mockAudio(page, 6);
  await page.goto("/infer/instance/demo/");
  await page.getByRole("button", {name: "合成并播放"}).click();
  const player = page.getByRole("region", {name: "合成音频", exact: true});
  await expect(player).toBeVisible();
  const seek = player.getByRole("slider");
  await expect(seek).toBeEnabled();
  await seek.focus(); await seek.press("End");
  await expect(player.locator(".media-player__time")).toContainText("0:06 / 0:06");
  await expect(page.locator(".audio-cursor-labels")).toContainText("144000");
  await expect(page.getByRole("progressbar", {name: "监听音频播放进度"})).toHaveCount(0);
  await expect(page.locator("audio[controls]")).toHaveCount(0);
  await page.locator(".action-panel").screenshot({path: testInfo.outputPath("listener-playback.png")});
});
