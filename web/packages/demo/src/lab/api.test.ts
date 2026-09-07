import {describe, expect, it} from "vitest";

import type {DemoConfig, LoadedDemoConfig} from "../config";
import {createLabApi} from "./api";

function loaded(lab: DemoConfig["lab"]): LoadedDemoConfig {
  return {
    responseUrl: new URL("https://tts.example/infer/instance/demo/config.json"),
    config: {
      schema_version: "qwen.tts.demo-config.v1",
      engine_version: "test",
      runtime_type: "triton",
      endpoints: {
        capabilities_url: "../v1/capabilities",
        openai_realtime_url: "../v1/realtime",
        native_websocket_url: "../v1/ws",
      },
      python_sdk: {available: false, project: "qwen3-tts-client", index_url: "../sdk/"},
      browser_sdk: {
        available: false,
        package: "@xmultimodalinteraction/qwen3tts-browser",
        version: "",
        registry_url: "",
        tarball_url: "",
      },
      docs: {version: "test", route: "./#/docs/"},
      lab,
    },
  };
}

describe("optional Lab API URL resolution", () => {
  it("keeps an instance prefix for relative backend URLs and upgrades WebSockets", () => {
    const api = createLabApi(loaded({available: true, url: "./legacy-lab"}));
    expect(api).not.toBeNull();
    expect(api?.baseUrl.toString()).toBe("https://tts.example/infer/instance/demo/legacy-lab/");
    expect(api?.capabilitiesUrl.toString()).toBe(
      "https://tts.example/infer/instance/demo/legacy-lab/api/v1/capabilities",
    );
    expect(api?.trtLiveUrl.toString()).toBe(
      "wss://tts.example/infer/instance/demo/legacy-lab/api/v1/trt-live",
    );
    expect(api?.concurrencyWsUrl("job/a").toString()).toBe(
      "wss://tts.example/infer/instance/demo/legacy-lab/api/v1/concurrency/job%2Fa",
    );
  });

  it("accepts an explicitly absolute backend URL", () => {
    const api = createLabApi(loaded({available: true, url: "http://lab.example:7860"}));
    expect(api?.baseUrl.toString()).toBe("http://lab.example:7860/");
    expect(api?.audioUrl("/api/v1/audio/sample.wav").toString()).toBe(
      "http://lab.example:7860/api/v1/audio/sample.wav",
    );
  });

  it("does not construct a client when the optional backend is disabled", () => {
    expect(createLabApi(null)).toBeNull();
    expect(createLabApi(loaded({available: false, url: "./legacy-lab"}))).toBeNull();
    expect(createLabApi(loaded({available: true, url: ""}))).toBeNull();
  });

  it("rejects non-HTTP backend schemes from hand-written configs", () => {
    expect(createLabApi(loaded({available: true, url: "javascript:alert(1)"}))).toBeNull();
  });
});
