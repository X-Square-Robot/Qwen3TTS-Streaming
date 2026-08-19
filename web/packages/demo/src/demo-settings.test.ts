import {describe, expect, it} from "vitest";

import {VadStrategy} from "@xmultimodalinteraction/qwen3tts-browser";

import {DEFAULT_DEMO_SETTINGS, defaultVadTuning} from "./demo-settings";

describe("Demo VAD presets", () => {
  it("uses the energy scale for the built-in energy detector", () => {
    expect(defaultVadTuning(VadStrategy.Energy)).toMatchObject({
      vadBeginThreshold: 0.3,
      vadEndThreshold: 0.2,
    });
    expect(DEFAULT_DEMO_SETTINGS).toMatchObject({
      vadBeginThreshold: 0.3,
      vadEndThreshold: 0.2,
    });
  });

  it("keeps TenVAD probability defaults separate", () => {
    expect(defaultVadTuning(VadStrategy.TenVad)).toMatchObject({
      vadBeginThreshold: 0.6,
      vadEndThreshold: 0.35,
    });
  });
});
