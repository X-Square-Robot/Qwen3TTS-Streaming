# Decode Session FSM Design

## Overview

The decode FSM governs per-session decode states. For **offline** (non-streaming)
requests, the full text is available upfront: the request is conceptually
`s-t1-t2-...-tN-e` (start → text tokens → EOS). For **streaming** requests, text
arrives incrementally — the FSM **IDLEs** whenever available text is exhausted
and resumes once new text (or `text_complete`) arrives.

### Offline vs Streaming Example

Assume text tokens `t1, t2, t3, t4`.

**Offline** — text pre-segmented into `[t1,t2]` and `[t3,t4]`:

```
Segment 1:  s → t1 → t2 → e → [Phase B] → done
Segment 2:  s → t3 → t4 → e → [Phase B] → done
```

**Streaming** — same tokens arriving incrementally:

```
init(t1,t2):   s → t1 → t2 → IDLE (no e yet)
append(t3,t4):           resume → t3 → t4 → IDLE
text_complete:           resume → e → [Phase B] → done
```

The key rule: **before `e` (EOS) arrives, if there are no more text tokens,
the session goes IDLE.** Phase B only runs after EOS is consumed.

## State Diagram

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> IDLE : 未知事件 / ()
    IDLE --> IDLE : 文本开始信号 / reset_context()
    IDLE --> HALT : 文本结束信号 / ()
    IDLE --> PREFILL : 开始token / prefill()
    IDLE --> IDLE : 结束token / ()
    IDLE --> PREFILL : 正常token / prefill()
    IDLE --> PREFILL : 标点token / prefill()

    PREFILL --> TEXT_INPUTING : Always / ()

    TEXT_INPUTING --> TEXT_INPUTING : 未知事件 / ()
    TEXT_INPUTING --> TEXT_INPUTING : 文本开始信号 / ()
    TEXT_INPUTING --> PAD_TEXT_EOS : 文本结束信号 / set_final()
    TEXT_INPUTING --> TEXT_INPUTING : 开始token / ()
    TEXT_INPUTING --> PAD_TEXT_NOP : 结束token / decode()
    TEXT_INPUTING --> TEXT_INPUTING : 正常token [未超长] / decode()
    TEXT_INPUTING --> PAD_TEXT_EOS : 正常token [超长] / decode()
    TEXT_INPUTING --> TEXT_INPUTING : 标点token [不满足切分] / decode()
    TEXT_INPUTING --> PAD_TEXT_EOS : 标点token [满足切分] / decode()

    PAD_TEXT_EOS --> HALT : [is_final] / flush_eos()
    PAD_TEXT_EOS --> IDLE : [else] / flush_eos()

    PAD_TEXT_NOP --> HALT : [is_final] / flush_nop()
    PAD_TEXT_NOP --> IDLE : [else] / flush_nop()
```
'''

## State Descriptions

| State | Description |
|-------|-------------|
| **IDLE** | Session paused — no text to process. Entry: initial start, SB1/SB2 complete, Prefill with no text. Transitions: text arrives → Prefill; text_complete with no text → HALT. |
| **Prefill** | Run prefill to populate KV cache with prefix + first text token. On completion: if trailing text exists → SA; if only E (empty text) → IDLE (defensive). |
| **SA** | Phase A: consume one trailing text token per decode step. Emit audio. Check threshold each step. Transitions: threshold met + mid-cut → SB0; E consumed → SB1; no more text + streaming → SAIdle; otherwise → loop SA. |
| **SAIdle** | Streaming sub-state of SA — all text consumed but `text_complete` not set. Orchestrator preserves KV cache. On resume: new text → SA; E (text_complete) → SA → SB1. |
| **SB0** | Mid-cut only: inject `tts_eos_embed` so the model gets an end-of-segment signal. → SB1. |
| **SB1** | Pad phase: decode with pad embeddings. Each step: check for natural codec EOS or silence. EOS/silence step's audio is **not emitted** (meaningless). Transitions: EOS/silence → IDLE; KV overflow → SB2. |
| **SB2** | KV overflow handler. Update EMA ratio with overflow penalty. → IDLE. |
| **HALT** | Session synthesis complete. |

### Key Design Decisions

1. **SA → SB1 direct** (text_complete, non mid-cut): E was already consumed
   as a trailing token in SA. The model has seen EOS in its KV cache. No need
   for SB0 to inject a redundant EOS — go directly to PAD phase (SB1).

2. **SA → SB0** (mid-cut only): Text still remains after the cut point.
   The model hasn't seen EOS yet, so SB0 injects `tts_eos_embed` to signal
   end-of-segment before padding begins.

3. **SB1 EOS audio not emitted**: When SB1 detects natural codec EOS or
   silence threshold, that frame's audio is not meaningful speech.
   `emit_wav=False` for the terminating step.

4. **SAIdle vs IDLE**: Two separate "waiting" states with different resume
   semantics:
   - **SAIdle → SA**: KV-continuous resume, no re-prefill needed.
   - **IDLE → Prefill**: New segment, requires full prefill.

## Transition Table

| From | To | Condition |
|------|----|-----------|
| IDLE | Prefill | Text (or S) arrives |
| IDLE | HALT | `text_complete` with no remaining text |
| Prefill | SA | Trailing text exists after prefill |
| Prefill | IDLE | No text (defensive: empty segment) |
| SA | SA | Threshold not met → consume next token |
| SA | SAIdle | All trailing consumed, `!text_complete` |
| SA | SB0 | Threshold met, `text_idx < trailing_len` (mid-cut) |
| SA | SB1 | All trailing consumed (E consumed), `text_complete` |
| SAIdle | SA | New text arrives (KV-continuous resume) |
| SAIdle | SA | `text_complete` arrives (inject E as trailing → SA → SB1) |
| SB0 | SB1 | EOS injected → begin PAD phase |
| SB1 | IDLE | Natural codec EOS (emit_wav=False) |
| SB1 | IDLE | Consecutive silence ≥ dynamic N (emit_wav=False) |
| SB1 | SB2 | KV overflow |
| SB2 | IDLE | Overflow handled |

## IDLE vs SAIdle Resume

### SAIdle → SA (KV-continuous)

1. Build new trailing embeddings from appended text.
2. Create a **new FSM** and call `enter_phase_a()`.
3. Set `next_embed = last_codec_sum + new_trailing[0]`.
4. Set `flow_state = ACTIVE` → session re-enters decode loop.

If `text_complete` with no new text: inject `tts_eos_embed` as single
trailing → SA consumes it → SA → SB1 (direct, skip SB0).

### IDLE → Prefill (new segment)

After SB1/SB2 → IDLE:
1. If remaining text in current trailing (mid-cut) → Prefill with
   checkpoint restore → SA.
2. If no remaining text + more segments → next segment → Prefill → SA.
3. If no remaining text + `text_complete` → HALT.

**Priority**: remaining text > next segment > HALT.

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `steps_in_phase_a` | int | Steps accumulated in current Phase A round. |
| `phase_b_start_frame` | int | `frame_idx` when entering SB0 or SB1. |
| `thresholds.a/b/c/d` | int | Phase A step thresholds for 3-tier + forced cut. |
| `pad_consecutive_silence` | int | Consecutive silent frames in SB1. |

## Threshold Computation

```python
remaining_kv = engine_max_decode_len - past_len
remaining_usable = remaining_kv - safety_margin
phase_a_cap = remaining_usable / (ema_ratio + 1)

a = int(phase_a_cap * 0.70)   # L1 punct only
b = int(phase_a_cap * 0.80)   # L1 + L2 punct
c = int(phase_a_cap * 0.90)   # L1 + L2 + L3 punct
d = phase_a_cap                # forced cut
```

## Dynamic Silence Threshold N (SB1)

```python
remaining_kv = engine_max_decode_len - past_len
if remaining_kv > 100:   N = 12
elif remaining_kv > 50:  N = 6
elif remaining_kv > 20:  N = 3
else:                    N = 1
```

## Streaming Flow Example

Full text: "你好，这是流式文本输入测试。我们正在验证。"

```
                        init("你好，这是流式文本输入测试。")
                        ┌──────────────────────────────────┐
Timeline    prefill     │  t1   t2   t3  ...  t8           │
            ━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━ SAIdle
                        │           Phase A                 │
                        └──────────────────────────────────┘

                        append("我们正在验证。")
                        ┌──────────────────────────────┐
                        │  t9   t10  ...  tN           │
            ━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━ SAIdle
                        │         Phase A               │
                        └──────────────────────────────┘

                        text_complete
                        ┌─────────────────────────────────┐
                        │  E  │  PAD  PAD ... silence      │
            ━━━━━━━━━━━━┿━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━ HALT
                        │ SA  │   SB1 (direct, no SB0)     │
                        └─────────────────────────────────┘
```
