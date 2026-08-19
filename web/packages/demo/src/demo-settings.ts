import {DeliveryPolicy, SynthesisTask, VadStrategy} from "@xmultimodalinteraction/qwen3tts-browser";

export interface DemoSynthesisSettings {
  readonly task: SynthesisTask;
  readonly speaker: string;
  readonly language: string;
  readonly sampleRate: number;
  readonly inputMode: "full" | "incremental";
  readonly vad: VadStrategy;
  readonly vadChunkMs: number;
  readonly vadBeginThreshold: number;
  readonly vadBeginCount: number;
  readonly vadEndThreshold: number;
  readonly vadEndCount: number;
  readonly vadStartMarginMs: number;
  readonly delivery: DeliveryPolicy;
  readonly deliveryWindowMs: number;
  readonly outputChunkMs: number;
  readonly emitTextEvents: boolean;
}

export const DEFAULT_DEMO_SETTINGS: DemoSynthesisSettings = {
  task: SynthesisTask.CustomVoice,
  speaker: "Serena",
  language: "auto",
  sampleRate: 24_000,
  inputMode: "full",
  vad: VadStrategy.Disabled,
  vadChunkMs: 16,
  vadBeginThreshold: 0.6,
  vadBeginCount: 5,
  vadEndThreshold: 0.35,
  vadEndCount: 31,
  vadStartMarginMs: 20,
  delivery: DeliveryPolicy.Guarded,
  deliveryWindowMs: 160,
  outputChunkMs: 0,
  emitTextEvents: true,
};
