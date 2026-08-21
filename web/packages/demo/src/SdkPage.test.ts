import {describe, expect, it} from "vitest";

import {DEFAULT_DEMO_SETTINGS} from "./demo-settings";
import {buildBrowserExample, buildPythonExample} from "./SdkPage";

describe("SDK code examples", () => {
  it("uses native WebSocket for Python and Realtime for Browser", () => {
    const python = buildPythonExample(
      "wss://tts.example/infer/instance/v1/ws",
      DEFAULT_DEMO_SETTINGS,
    );
    const browser = buildBrowserExample(
      "https://tts.example/infer/instance/v1/capabilities",
      "wss://tts.example/infer/instance/v1/realtime",
      DEFAULT_DEMO_SETTINGS,
    );

    expect(python).toContain(
      'TTSClient.connect("wss://tts.example/infer/instance/v1/ws")',
    );
    expect(python).not.toContain("/v1/realtime");
    expect(browser).toContain(
      'websocketUrl: "wss://tts.example/infer/instance/v1/realtime"',
    );
  });
});
