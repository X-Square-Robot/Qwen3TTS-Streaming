import {DeliveryPolicy, SynthesisTask, VadStrategy} from "@xmultimodalinteraction/qwen3tts-browser";

export interface DemoSynthesisSettings {
  readonly task: SynthesisTask;
  readonly speaker: string;
  readonly language: string;
  readonly sampleRate: number;
  readonly inputMode: "full" | "long" | "incremental";
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

export interface DemoVadTuning {
  readonly vadChunkMs: number;
  readonly vadBeginThreshold: number;
  readonly vadBeginCount: number;
  readonly vadEndThreshold: number;
  readonly vadEndCount: number;
  readonly vadStartMarginMs: number;
}

const COMMON_VAD_TUNING = {
  vadChunkMs: 16,
  vadBeginCount: 5,
  vadEndCount: 31,
  vadStartMarginMs: 20,
} as const;

export function defaultVadTuning(strategy: VadStrategy): DemoVadTuning {
  return strategy === VadStrategy.TenVad
    ? {...COMMON_VAD_TUNING, vadBeginThreshold: 0.6, vadEndThreshold: 0.35}
    : {...COMMON_VAD_TUNING, vadBeginThreshold: 0.3, vadEndThreshold: 0.2};
}

const DEFAULT_VAD_TUNING = defaultVadTuning(VadStrategy.Energy);

export const DEFAULT_DEMO_SETTINGS: DemoSynthesisSettings = {
  task: SynthesisTask.CustomVoice,
  speaker: "Serena",
  language: "auto",
  sampleRate: 24_000,
  inputMode: "full",
  vad: VadStrategy.Disabled,
  ...DEFAULT_VAD_TUNING,
  delivery: DeliveryPolicy.Guarded,
  deliveryWindowMs: 160,
  outputChunkMs: 0,
  emitTextEvents: true,
};
