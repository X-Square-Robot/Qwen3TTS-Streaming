**English** | [中文](mixed_precision_plan.zh-CN.md)

# Mixed Precision Implementation Plan

> Date: 2026-06-22
> Status: Landed, but **the implementation differs from this draft** — the final version was implemented in the bash lifecycle rather than the Python CLI approach described below.
> Scope: Per-submodule mixed precision control for the single `talker_code2wav_fused` TensorRT engine
>
> **2026-07-06 update — the cp=fp32 motivation is obsolete**: the hallucination that motivated this plan was root-caused to the damaged 0601 checkpoint, not CP precision (see `../investigation/streaming_hallucination.md`). On the 0701 retrain, a full-bf16 engine (cp=bf16) measures 0/100 on the same deterministic probe set as the cp=fp32 baseline, so `CP_PRECISION` now defaults to follow `ENGINE_DTYPE` (bf16). The per-submodule precision *mechanism* built by this plan remains in place and useful. Separately, `CODE2WAV_PRECISION` now **defaults to fp16** (the fast c2w conv path on sm120 — see `../architecture/engine_overview.md` §1.5), and the mixed build auto-selects `--precisionConstraints=prefer` via `trt_fused_io_formats.py --emit constraints`. The best-performance default is now **cp=bf16 (follow base) + code2wav=fp16**; `--cp-precision fp32` / `--code2wav-precision bf16` remain for numerical-parity debugging.
>
> **Actual usage**: `bash scripts/bash/build_engines.sh --variant custom-1.7b --cp-precision fp32` (defaults to follow `ENGINE_DTYPE`).
> The precision is written into the manifest and translated by `scripts/python/trt_fused_io_formats.py --emit layer-precisions` into trtexec
> `--layerPrecisions` wildcards. The original Python CLI design is preserved below as a historical record (the `qwen3tts_*` modules it references have been removed).

## Background

The current production main path uses a single fused TensorRT engine:

- `backbone`: the Talker backbone network
- `cp`: the Code Predictor
- `code2wav`: the vocoder decode path

Recent troubleshooting has shown that under `bf16` inference, the `cp` path exhibits higher numerical sensitivity and can induce generation hallucinations. At the same time, TensorRT supports mixed precision execution, so the following capabilities need to be evaluated and pushed forward:

- The compute precision of `backbone`, `cp`, and `code2wav` can be configured independently
- The external engine I/O types can be adapted stably, without requiring a one-to-one correspondence with each internal segment's compute precision
- The existing fused engine architecture is preserved as much as possible, without immediately splitting into multiple engines just for precision control

### Chain of Evidence

The core motivation for this plan comes from the systematic investigation in `docs/dev/investigation/streaming_hallucination.md`. The key findings are as follows:

1. **CP BF16 numerical instability has been independently verified** (Finding #15):
   - Standalone `code_predictor_unrolled` BF16 TRT vs ORT: `7/10` mismatches on random inputs
   - Standalone `code_predictor_unrolled` FP32 TRT vs ORT: `0/10` mismatches
   - On the real bad-dump state `000003`, FP32 TRT matches ORT exactly

2. **TRT BF16 vs official PyTorch BF16 still mismatches** (Finding #16):
   - On the same CP inputs, TRT BF16 cannot match the official cached BF16
   - `7/8` mismatches across random trials
   - For the 4a bad state, the first divergence occurs at stages 2–5

3. **The mismatch is concentrated in the CP tail tokens, not in the backbone or penalty** (Finding #17):
   - `updated_token_counts` matches the official BF16
   - `full_codec` mismatches (divergence begins at CP stage 2+)
   - The problem is localized to the TRT BF16 execution of the CP branch itself

4. **A debug engine forcing `hidden/logits` to FP32 changes the TRT builder numerics** (Finding #18):
   - Even changing only the output format is enough to flip token decisions
   - This indicates that TRT's internal precision choices are extremely sensitive on the CP path

Taken together, the strongest explanation is that **the TensorRT BF16 execution of `code_predictor_unrolled` is itself numerically/semantically unstable**, and promoting CP to FP32 is the most promising mitigation available at present.

## Current State Summary

### 1. The current production main path is a single-engine fused path

The production main graph is exported by `export_09_talker_code2wav_fused.py`, with the compute path:

- `talker(backbone + logits processing + cp + codec_sum) -> code2wav -> wav`

This means the current problem is not about coordinating precision across multiple sub-engines orchestrated by Triton, but rather **precision constraints across different subgraphs inside a single TRT engine**.

### 2. The project already distinguishes "compute precision" from "I/O precision"

Two layers of concepts already exist in the current manifest and build chain:

- `engine_dtype`: the TRT builder compute precision
- `triton_io_float_dtype`: the floating-point I/O type that TRT/Triton exposes externally

This shows that the engineering design already accepts that:

- The internal compute dtype and the external I/O dtype can differ
- TRT is allowed to automatically insert cast / reformat operations inside the graph

### 3. The current implementation still supports only "one compute precision for the whole graph"

The existing build scripts pass `dtype` to `trtexec` as a global parameter, and do not yet support:

- `backbone_precision`
- `cp_precision`
- `code2wav_precision`

Therefore the core gap for mixed precision capability lies at the **build/export layer**, not the runtime layer.

### 4. The existing ONNX graph is amenable to per-subgraph identification

In the ready-made `talker_code2wav_fused.onnx`, node names preserve stable prefixes:

- `/talker_fused/talker_unified/...`
- `/talker_fused/cp/...`
- `/talker_fused/codec_sum/...`
- `/code2wav/...`

This means TensorRT layers can be classified by prefix, enabling "setting compute precision per submodule."

### 5. Key files in the current build chain

| File | Responsibility | Mixed precision impact |
|------|------|-------------|
| `scripts/python/qwen3tts_tools/trtexec.py` | trtexec Python wrapper, `run_trtexec()` / `build_talker_code2wav_fused()` | Needs extension to pass layer precision parameters |
| `scripts/python/trt_fused_io_formats.py` | Generates `--inputIOFormats` / `--outputIOFormats` | I/O format does not change with internal mixed precision; keeps a unified `triton_io_float_dtype` |
| `scripts/python/triton_manifest_io.py` | Reads/writes `triton_manifest.json` | Needs extension to record the three per-segment precision fields |
| `scripts/python/schemas/triton_manifest.schema.json` | manifest JSON schema | Needs to add `backbone_precision` / `cp_precision` / `code2wav_precision` |
| `scripts/python/update_triton_manifest_profile.py` | Writes back the build profile | Needs to record the mixed precision configuration |
| `scripts/python/generate_triton_configs.py` | Generates Triton config.pbtxt | Unaffected (I/O dtype is unified) |
| `engine/config.py` | `ModelArchConfig` / `EngineProfileConfig` | `EngineProfileConfig` needs the mixed precision fields added |
| `scripts/export/export_09_talker_code2wav_fused.py` | Exports the fused ONNX | Unaffected (ONNX is exported uniformly in FP32) |

## Goals

The direct goals of this plan are as follows:

1. Keep the `talker_code2wav_fused` single-engine production path unchanged
2. Support independent configuration of the three compute precisions: `backbone`, `cp`, `code2wav`
3. Keep the external engine floating-point I/O at a unified dtype for now
4. Continue to automatically adapt input/output tensors at runtime according to the engine's actual I/O dtype
5. Prioritize solving the `cp` numerical stability problem under `bf16`

The recommended first-phase target configuration:

- `backbone = bf16`
- `cp = fp32`
- `code2wav = bf16`
- `io_float = bf16`

## Design Decisions

### Decision 1: The external I/O adopts a unified floating-point dtype, rather than drifting segment by segment with the internal subgraphs

The recommendation is to keep:

- Floating-point main I/O: uniformly use a single `triton_io_float_dtype`
- Control tensors: continue to keep fixed-type rules
  - `position_ids` / `token_counts` / `full_codec` and so on stay integer types
  - `gumbel_noise` / `cp_gumbel_noise` / `temperature` / `penalty` / `cache_position` stay `fp32`

Reasons:

- The external boundary of the fused engine does not need to expose the precision differences of each internal segment
- A unified I/O dtype reduces runtime branching and debugging complexity
- TRT can automatically insert conversions at the `backbone -> cp -> code2wav` boundaries internally
- The current executor already supports organizing inputs and outputs according to the engine's actual tensor dtype

Not recommended:

- Designing the external I/O dtype as a "direct mirror of the internal module precision"
- Exposing a separate heterogeneous dtype policy for each external float tensor of the fused engine

Such approaches yield limited benefit but significantly increase the complexity of the manifest, build parameters, runtime adaptation, and validation.

### Decision 2: Preserve the single-engine architecture first, rather than immediately splitting into multiple engines

Splitting into `backbone.engine + cp.engine + code2wav.engine` is theoretically the most intuitive, but is not recommended in the short term, for reasons including:

- The current production chain has already been optimized for scheduling and cache layout around the fused engine
- Multiple engines increase boundary synchronization, context switching, and engineering complexity
- The existing fused ONNX graph is already amenable to layer precision constraints by prefix

Splitting engines should only be considered when the following arise:

- Single-engine mixed precision cannot stably reproduce the target behavior in TRT
- A given segment must be exposed as an independent runtime boundary so that the dtype can be controlled explicitly from outside
- A submodule needs to be independently deployed, cached, or reused in the future

### Decision 3: In implementation, distinguish the "short-term validation route" from the "long-term stable route"

#### Short-term validation route

Use the layer precision constraint capability available in TensorRT 10.x to quickly validate:

- Whether `cp=fp32` significantly mitigates the `bf16` hallucination
- Whether the performance loss is acceptable when `backbone/code2wav` stay at low precision

The goal is to first prove the approach works, without pursuing a final-form builder wrapper up front.

#### Long-term stable route

Gradually migrate the build capability from purely stacking `trtexec` parameters to the TensorRT Python API builder.

Reasons:

- Artifacts in the current workspace indicate TensorRT `10.13.3.9` is in use
- TensorRT 10.x still supports layer precision / output type related capabilities
- TensorRT 11.x makes significant changes to the relevant `trtexec` options, so the long-term plan should not be entirely bet on CLI capabilities

Therefore the recommendation is:

- Short term: use TRT 10.x to quickly validate mixed precision
- Medium term: fill in the Python builder version to reduce the risk of future TRT upgrades

### Decision 4: codec_sum belongs to the backbone precision domain

In the ONNX graph, the `codec_sum` subgraph has the prefix `/talker_fused/codec_sum/...`, and logically it is the bridging segment from the backbone's output logits to the CP input. The reasons for assigning it to the backbone precision domain:

- `codec_sum` has a small compute footprint (a single sum operation), so independent precision control yields limited benefit
- Keeping `codec_sum` at low precision (bf16) adds no cast overhead
- If it later turns out that `codec_sum` itself also needs fp32, it can be configured separately through the same mechanism

## trtexec Mixed Precision Mechanism

### The `--layerPrecisions` option in TRT 10.x

trtexec in TensorRT 10.x supports the `--layerPrecisions` option, with the syntax:

```bash
--layerPrecisions=layer_name:precision[,layer_name:precision...]
```

where:
- `layer_name`: the ONNX node name or the layer name auto-generated by TRT
- `precision`: `fp32`, `fp16`, `bf16`, `fp8`, etc.

**Key limitations**:
- Must be used together with the global `--bf16` or `--fp16` flag (the global flag sets the default precision, `--layerPrecisions` overrides specific layers)
- Layers not specified in `--layerPrecisions` use the global precision
- The layer name must exactly match the layer name at TRT build time, not the ONNX node name

### Layer precision constraints via the TRT Python API

```python
import tensorrt as trt

# Build the network
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("talker_code2wav_fused.onnx")

# Iterate over layers and set precision by prefix
for i in range(network.num_layers):
    layer = network.get_layer(i)
    name = layer.name

    if name.startswith("/talker_fused/cp/"):
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)
    elif name.startswith("/talker_fused/talker_unified/"):
        layer.precision = trt.bfloat16
    elif name.startswith("/code2wav/"):
        layer.precision = trt.bfloat16
    # Other layers keep the global default precision

# Set the global builder precision
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.BF16)  # global bf16
```

**Advantages of the Python API**:
- Can use prefix matching rather than exact layer names
- Can set both `layer.precision` and `set_output_type()` (trtexec `--layerPrecisions` only sets precision)
- Makes pre-build auditing easier (print all layer names and classification results)
- Makes post-build validation easier (inspect the actual precision distribution in the engine)

### The trtexec path for the short-term PoC

The current `run_trtexec()` supports an `extra_args` parameter, into which any trtexec argument can be passed:

```python
# The existing interface in trtexec.py
result = run_trtexec(
    onnx=str(onnx_path),
    engine=str(engine_path),
    dtype="bf16",              # global bf16
    extra_args=[
        # Promote the cp subgraph to fp32 via --layerPrecisions
        "--layerPrecisions="
        + ",".join(f"{name}:fp32" for name in cp_layer_names),
    ],
    ...
)
```

**But there are practical difficulties**:
1. The exact list of layer names must first be obtained by parsing the ONNX
2. The layer names at TRT build time may not fully match the ONNX node names
3. Layer names may change after every export change, making maintenance costly

Therefore, for the short-term PoC, the recommendation is to use a **Python builder + prefix matching**, rather than trtexec `--layerPrecisions`.

## Configuration Plan

The recommendation is to introduce new precision fields into the manifest / build configuration:

```yaml
precision:
  backbone: bf16
  cp: fp32
  code2wav: bf16
  io_float: bf16
```

Flat fields can also be used:

```yaml
backbone_precision: bf16
cp_precision: fp32
code2wav_precision: bf16
triton_io_float_dtype: bf16
```

The recommendation is to prefer flat fields, because they are more compatible with the existing manifest structure.

The suggested meanings are as follows:

- `backbone_precision`
  - Controls the compute precision of the `/talker_fused/talker_unified/` and `/talker_fused/codec_sum/` backbone
- `cp_precision`
  - Controls the compute precision of the `/talker_fused/cp/` subgraph
- `code2wav_precision`
  - Controls the compute precision of the `/code2wav/` subgraph
- `triton_io_float_dtype`
  - Controls the dtype of the vast majority of external floating-point I/O tensors

### Manifest Schema Changes

Add to `scripts/python/schemas/triton_manifest.schema.json`:

```json
{
  "backbone_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Backbone (talker_unified + codec_sum) compute precision; defaults to follow engine_dtype"
  },
  "cp_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Code Predictor compute precision; setting fp32 mitigates BF16 numerical sensitivity"
  },
  "code2wav_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Code2Wav decoder compute precision; defaults to follow engine_dtype"
  }
}
```

Backward-compatibility rules:
- If none of the three fields are specified, fall back to a unified `engine_dtype` precision (consistent with current behavior)
- If some are specified, the unspecified fields default to `engine_dtype`
- `engine_dtype` is still retained as the fallback for the global default precision

### CLI Argument Design

```bash
# Global precision (backward compatible)
bash scripts/bash/build_engines.sh --variant custom-1.7b --dtype bf16

# Mixed precision
bash scripts/bash/build_engines.sh --variant custom-1.7b --dtype bf16 --cp-precision fp32

# Explicitly specify each segment's precision
bash scripts/bash/build_engines.sh --variant custom-1.7b \
    --backbone-precision bf16 \
    --cp-precision fp32 \
    --code2wav-precision bf16 \
    --io-float-dtype bf16
```

Environment variable mapping:

```bash
ENGINE_DTYPE=bf16                  # global precision
BACKBONE_PRECISION=bf16            # backbone precision
CP_PRECISION=fp32                  # cp precision
CODE2WAV_PRECISION=bf16            # code2wav precision
TRITON_IO_FLOAT_DTYPE=bf16        # I/O precision
```

## Implementation Steps

### Phase 0: Solidify the plan and converge naming

Goals:

- Confirm the precision field naming
- Confirm the first-phase default values
- Confirm the "unified I/O float dtype" design principle

Outputs:

- manifest field definitions
- CLI argument definitions
- documentation updates

Recommended default values:

- `backbone_precision = bf16`
- `cp_precision = fp32`
- `code2wav_precision = bf16`
- `triton_io_float_dtype = bf16`

### Phase 1: Extend the manifest and build parameters

#### 1.1 Schema update

**File**: `scripts/python/schemas/triton_manifest.schema.json`

- Add the `backbone_precision`, `cp_precision`, `code2wav_precision` fields
- Keep `engine_dtype` as a compatibility field

#### 1.2 Manifest read/write

**File**: `scripts/python/triton_manifest_io.py`

- `build_manifest_for_export()` adds writing of the three per-segment precision fields
- Default-value rule for the precision fields: take `engine_dtype` when unspecified

**File**: `scripts/python/update_triton_manifest_profile.py`

- The `engine_profile` section adds records for the three per-segment precisions

#### 1.3 I/O format unchanged

**File**: `scripts/python/trt_fused_io_formats.py`

- **No changes needed.** The I/O format is uniformly controlled by `triton_io_float_dtype` and does not change with internal mixed precision
- Control tensors (`gumbel_noise`, `cp_gumbel_noise`, `temperature`, `penalty`, `cache_position`) stay at `fp32:chw`
- The floating-point main I/O stays at `{triton_io_float_dtype}:chw`

#### 1.4 Engine config

**File**: `engine/config.py`

- `EngineProfileConfig` adds the `backbone_precision`, `cp_precision`, `code2wav_precision` fields
- The I/O dtype logic in `to_model_config()` is unchanged (still uses `triton_io_float_dtype`)

Phase goal:

- Modify only the configuration and metadata flow, without changing runtime logic

### Phase 2: Implement single-engine mixed precision build

The recommendation is to proceed in two steps:

#### 2.1 Fast validation version (Python builder PoC)

Add `scripts/python/qwen3tts_tools/mixed_precision_builder.py`, implementing:

```python
def build_fused_mixed_precision(
    onnx_path: Path,
    engine_path: Path,
    *,
    backbone_precision: str = "bf16",
    cp_precision: str = "fp32",
    code2wav_precision: str = "bf16",
    triton_io_float_dtype: str = "bf16",
    min_shapes: str,
    opt_shapes: str,
    max_shapes: str,
    gpu_device: str = "auto",
) -> int:
    """Build talker_code2wav_fused.engine with per-submodule precision."""
```

Core logic:

1. **Parse the ONNX**: `trt.OnnxParser` parses the fused ONNX into a TRT network
2. **Iterate over layers and classify**: classify by `layer.name` prefix into `backbone` / `cp` / `code2wav` / `other`
3. **Audit stage**: print the layer count and representative names for each category, for pre-build inspection
4. **Apply precision constraints**:
   - Global flag: `config.set_flag(trt.BuilderFlag.BF16)` or `FP16`
   - CP-prefixed layers: `layer.precision = trt.float32` + `layer.set_output_type(0, trt.float32)`
   - Other prefixed layers: set the corresponding precision according to configuration
5. **Set the I/O dtype**: via `config.set_flag()` and the Python API equivalent of `--inputIOFormats`
6. **Build the engine**: `builder.build_serialized_network(network, config)`
7. **Save the engine**: write out the `.engine` file
8. **Post-validation**: optionally deserialize with `trt.Runtime` and inspect the layer info

Key design points:

- **Prefix classification table**:

| Prefix | Category | Default precision |
|------|------|---------|
| `/talker_fused/talker_unified/` | backbone | bf16 |
| `/talker_fused/codec_sum/` | backbone | bf16 |
| `/talker_fused/cp/` | cp | fp32 |
| `/code2wav/` | code2wav | bf16 |
| others | backbone | bf16 |

- **Handling of unmatched layers**: layers whose prefix does not match any known pattern are assigned to the backbone precision domain and emit a `logger.warning`
- **Pre-build audit**: before setting precision, print the complete layer classification statistics:

```
Layer classification summary:
  backbone: 342 layers (prefixes: /talker_fused/talker_unified/, /talker_fused/codec_sum/)
  cp:       45 layers  (prefix: /talker_fused/cp/)
  code2wav: 89 layers  (prefix: /code2wav/)
  unclassified: 3 layers (warning: check prefix table)
```

PoC validation focus:

- Whether TensorRT accepts this combination
- Whether the engine builds successfully
- Whether the precision distribution in the post-build layer info matches expectations
- Whether the inference output improves

#### 2.2 Integrate into `build_talker_code2wav_fused()`

**File**: `scripts/python/qwen3tts_tools/trtexec.py`

Add branching logic to `build_talker_code2wav_fused()`:

```python
def build_talker_code2wav_fused(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    backbone_precision: str = "",    # new, defaults to follow dtype
    cp_precision: str = "",          # new, defaults to follow dtype
    code2wav_precision: str = "",    # new, defaults to follow dtype
    triton_io_float_dtype: str = "",
    ...
) -> int:
    # Resolve default values
    if not backbone_precision:
        backbone_precision = dtype
    if not cp_precision:
        cp_precision = dtype
    if not code2wav_precision:
        code2wav_precision = dtype

    # Determine whether it is mixed precision
    is_mixed = len({backbone_precision, cp_precision, code2wav_precision}) > 1

    if is_mixed:
        # Use the Python builder path
        from qwen3tts_tools.mixed_precision_builder import build_fused_mixed_precision
        return build_fused_mixed_precision(...)
    else:
        # Use the existing trtexec path (backward compatible)
        return _build_fused_trtexec(...)
```

#### 2.3 CLI integration

**File**: the build subcommand in `scripts/python/qwen3tts_cli/`

- Add the `--backbone-precision`, `--cp-precision`, `--code2wav-precision` arguments
- Pass them through to `build_talker_code2wav_fused()`

### Phase 3: ONNX node prefix validation

To prevent prefix drift caused by changes to the export code, add pre-build validation:

**New**: `scripts/python/qwen3tts_tools/layer_audit.py`

```python
def audit_onnx_prefixes(onnx_path: Path) -> dict[str, list[str]]:
    """Scan the ONNX graph, classify all nodes by prefix, and return a classification report.

    Returns:
        A {category: [node_names]} dict

    Raises:
        LayerAuditError: if the number of unclassified nodes exceeds the threshold
    """
```

Validation rules:
- Every node must match one of the known prefixes
- More than 5% unmatched nodes triggers `LayerAuditError`
- The audit runs automatically before every build; the build aborts on failure

Integrate into `build_fused_mixed_precision()`:

```python
# Pre-build audit
audit = audit_onnx_prefixes(onnx_path)
if audit.get("unclassified_ratio", 0) > 0.05:
    raise LayerAuditError(
        f"More than 5% of nodes could not be classified; prefix drift may exist: {audit}"
    )
```

### Phase 4: Runtime consistency checks

The runtime already has the basic capability to organize inputs and outputs according to the engine's actual tensor dtype, but the following checks are still recommended:

- The `triton_io_float_dtype` declared in the manifest is consistent with the engine's actual I/O dtype
- All expected input and output names actually exist in the engine
- The control tensor types still satisfy the convention
- The `backbone_precision` / `cp_precision` / `code2wav_precision` in the manifest are consistent with the engine's actual layer precision distribution (post-build check)

If an inconsistency is found, it should raise a clear error during the engine load stage, rather than silently degrading.

### Phase 5: Validation and regression testing

At least add the following validation dimensions:

#### Functional correctness

- The engine can build / load / infer normally when `cp=fp32`
- External I/O can still be fed normally at the unified dtype
- The output audio tensors and state tensor types are as expected

#### Numerical stability

- Compare against the existing full-graph `bf16` version
- Verify whether the hallucination repro samples improve
- Compare the emission behavior of `full_codec`, `codec_sum`, `wav`, and `EOS`
- Focus on validating the bad-dump steps `000003`, `000004`, `000008` recorded in `docs/dev/investigation/streaming_hallucination.md`

#### Performance and resources

- First-packet latency
- Per-step decode latency
- GPU memory usage
- Engine build time
- **Internal cast overhead within the CP region**: observe the number of reformat layers via `--dumpLayerInfo` or post-build validation of the Python builder

#### Regression compatibility

- Full-graph `fp32`
- Full-graph `bf16`
- Full-graph `fp16`
- Mixed precision combinations

#### Test code

The recommendation is to add:

| Test file | Coverage |
|---------|---------|
| `tests/unit/test_mixed_precision_config.py` | manifest field parsing, default values, backward compatibility |
| `tests/unit/test_layer_audit.py` | prefix classification, unmatched node detection |
| `tests/integration/test_mixed_precision_build.py` | end-to-end build (requires GPU + ONNX file) |

## Suggested Validation Matrix

The first round should cover at least the following combinations:

| Backbone | CP | Code2Wav | I/O | Purpose |
|----------|----|----------|-----|------|
| bf16 | bf16 | bf16 | bf16 | Current baseline |
| bf16 | fp32 | bf16 | bf16 | **Preferred candidate** |
| bf16 | fp32 | fp32 | bf16 | Determine whether code2wav is also sensitive |
| fp32 | fp32 | fp32 | fp32 | Upper precision reference |
| fp16 | fp32 | fp16 | fp16 | Observe fp16 platform behavior |

The highest priority is the second row.

### Numerical validation method

For each configuration combination, the recommendation is to run the following validation flow:

1. **Standalone CP parity**: use `tests/tools/verify_code_predictor_trt.py` to validate 10 random trials
2. **Bad-state replay**: use `scripts/python/analyze_engine_dump.py` to compare `full_codec` on the 4a dump steps `000003` / `000004` / `000008`
3. **End-to-end audio**: use `tests/tools/run_engine_long_case.py` to run on the 4a / story cases, comparing audio duration and content

## Risks

### 1. TensorRT version evolution risk

The current workflow can use TRT 10.x capabilities to complete the PoC, but after a future upgrade to TRT 11.x, the maintainability of the pure `trtexec` route will decline.

Mitigation:

- Do not tie the long-term plan to `trtexec --layerPrecisions`
- Fill in the Python builder implementation as early as possible

### 2. Node naming drift risk

If mixed precision classification depends on ONNX node prefixes, future adjustments to the export code may cause the prefixes to change.

Mitigation:

- Explicitly constrain module naming at the export stage (the `_prefix` parameter of `torch.onnx.export` in `export_09_talker_code2wav_fused.py`)
- Do a layer classification audit before the build (Phase 3)
- Fail immediately if a key prefix does not exist

### 3. Performance loss from internal automatic casts

Precision switching between `backbone -> cp -> code2wav` introduces reformat/cast.

Mitigation:

- Prioritize promoting only `cp` to `fp32`
- Observe the number of conversions via profile / dumpLayerInfo / post-build validation of the Python builder
- If there are too many conversions, then evaluate whether local graph boundary restructuring is needed

Expected cast overhead estimate:
- `backbone(bf16) -> cp(fp32)`: 1 cast (the hidden tensor from bf16 to fp32)
- `cp(fp32) -> codec_sum(bf16)`: 1 cast (logits from fp32 to bf16)
- Inside `code2wav(bf16)`: no additional cast
- About 2 explicit casts in total, plus any implicit reformat that TRT may insert

### 4. The problem may not stem solely from CP precision

Although the current localization points to `cp`'s `bf16` sensitivity, it is still necessary to guard against the case where "after changing CP precision, the problem only partially improves."

Mitigation:

- Keep the `code2wav=fp32` control experiment
- Compare `hidden/logits/full_codec/wav` level by level
- Refer to `streaming_hallucination.md` Finding #18: forcing `hidden/logits` to fp32 also changed the TRT numerics, so watch for this interaction effect

### 5. Risk of divergent build results between the Python builder and trtexec

The Python builder and trtexec may produce different engine optimization results, even with the same input ONNX.

Mitigation:

- In non-mixed-precision scenarios, compare the engine output of the Python builder and trtexec using the same ONNX
- Ensure the full-graph `bf16` baseline behaves consistently across the two paths
- Retain the trtexec path in CI as a fallback validation

## Rollback Strategy

If the mixed precision plan encounters an unacceptable regression during the validation stage, the rollback plan is as follows:

1. **Build-layer rollback**: restore `cp_precision` to the same value as `engine_dtype`, falling back to full-graph unified precision
2. **Manifest rollback**: remove the `backbone_precision` / `cp_precision` / `code2wav_precision` fields; `engine_dtype` remains the effective configuration
3. **No runtime impact**: the runtime does not consume the mixed precision fields, so the rollback does not require runtime code changes
4. **Engine rebuild**: after rollback, the engine must be rebuilt (full-graph unified precision)

Key point: the mixed precision fields are **incrementally added** and do not modify the semantics of the existing fields, so rollback only involves "not using the new fields."

## Recommended Order of Progress

The recommendation is to proceed in the following order:

1. Extend the manifest / CLI / schema to add the three per-segment precision configurations
2. Keep a unified `triton_io_float_dtype`
3. Implement the Python builder PoC + ONNX prefix audit
4. Prioritize validating `bf16 + fp32(cp) + bf16`
5. If the effect holds, integrate into `build_talker_code2wav_fused()`
6. Complete the full test matrix and documentation
7. Medium term: evaluate whether the Python builder should replace trtexec as the unified build path

## Final Recommendation

The most recommended approach at the current stage is:

- Keep a single `talker_code2wav_fused` engine
- Use a unified floating-point dtype for the external I/O
- Internally promote `cp` to `fp32`
- Keep `backbone` and `code2wav` at low precision for now

This route is the least intrusive to the existing engineering, has the best chance of validating within a short cycle "whether the hallucination is indeed triggered by `cp`'s `bf16` numerical problem," and also lays the groundwork for a more general TensorRT mixed precision build capability in the future.
