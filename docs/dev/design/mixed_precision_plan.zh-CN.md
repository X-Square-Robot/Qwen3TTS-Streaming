[English](mixed_precision_plan.md) | **中文**

# 混合精度方案落地计划

> 日期：2026-06-22
> 状态：已落地，但**实现方式与本草案不同**——最终在 bash 生命周期实现，而非下文的 Python CLI 方案。
> 范围：`talker_code2wav_fused` TensorRT 单引擎的子模块混合精度控制
>
> **2026-07-06 更新——cp=fp32 的动机已过时**：促成本方案的幻觉已定位为 0601 checkpoint 权重训坏,与 CP 精度无关（见 `../investigation/streaming_hallucination.zh-CN.md`）。0701 重训后,全 bf16 引擎（cp=bf16）在与 cp=fp32 基线相同的确定性探测集上实测 0/100,因此 `CP_PRECISION` 现默认跟随 `ENGINE_DTYPE`（bf16）。本方案建立的子模块精度控制*机制*保留且仍然有用。另外，`CODE2WAV_PRECISION` 现**默认 fp16**（sm120 上更快的 c2w 卷积路径，见 `../architecture/engine_overview.zh-CN.md` §1.5），混合构建经 `trt_fused_io_formats.py --emit constraints` 自动选用 `--precisionConstraints=prefer`。当前最佳性能默认为 **cp=bf16（跟随基础）+ code2wav=fp16**；`--cp-precision fp32` / `--code2wav-precision bf16` 保留用于数值对齐调试。
>
> **实际用法**：`bash scripts/bash/build_engines.sh --variant custom-1.7b --cp-precision fp32`（默认跟随 `ENGINE_DTYPE`）。
> 精度写入 manifest，由 `scripts/python/trt_fused_io_formats.py --emit layer-precisions` 翻译成 trtexec
> `--layerPrecisions` 通配符。下文保留原始 Python CLI 设计作为历史记录（其中的 `qwen3tts_*` 模块已移除）。

## 背景

当前工程的生产主链路使用单个 fused TensorRT 引擎：

- `backbone`：Talker 主干网络
- `cp`：Code Predictor
- `code2wav`：声码器解码路径

近期问题排查表明，在使用 `bf16` 推理时，`cp` 路径存在更高的数值敏感性，并可能诱发生成幻觉。与此同时，TensorRT 支持混合精度执行，因此需要评估并推进以下能力：

- `backbone`、`cp`、`code2wav` 的计算精度可分别配置
- 外部 engine I/O 类型可以稳定适配，不要求与内部每段计算精度一一对应
- 尽量保持现有 fused engine 架构，不为了精度控制而立即拆分多引擎

### 证据链

本方案的核心动机来自 `docs/dev/investigation/streaming_hallucination.zh-CN.md` 的系统化调查。关键发现如下：

1. **CP BF16 数值不稳定性已独立验证**（发现 #15）：
   - 独立 `code_predictor_unrolled` BF16 TRT vs ORT：随机输入 `7/10` 不匹配
   - 独立 `code_predictor_unrolled` FP32 TRT vs ORT：`0/10` 不匹配
   - 在真实不良转储状态 `000003` 上，FP32 TRT 精确匹配 ORT

2. **TRT BF16 vs 官方 PyTorch BF16 仍不匹配**（发现 #16）：
   - 在相同 CP 输入上，TRT BF16 无法匹配官方缓存 BF16
   - 随机试验 `7/8` 不匹配
   - 4a 不良状态首次分歧在 stage 2~5

3. **不匹配集中在 CP 尾部 token，而非 backbone 或 penalty**（发现 #17）：
   - `updated_token_counts` 匹配官方 BF16
   - `full_codec` 不匹配（CP stage2+ 开始分歧）
   - 问题定位在 CP 分支的 TRT BF16 执行本身

4. **强制 `hidden/logits` 为 FP32 的调试引擎改变了 TRT 构建器数值**（发现 #18）：
   - 即使只改输出格式也足以翻转 token 决策
   - 说明 TRT 内部精度选择对 CP 路径极其敏感

综上，最强解释是 **`code_predictor_unrolled` 的 TensorRT BF16 执行本身在数值/语义上不稳定**，将 CP 提升到 FP32 是当前最有前景的缓解手段。

## 现状总结

### 1. 当前生产主链是单引擎 fused 路径

生产主图由 `export_09_talker_code2wav_fused.py` 导出，计算路径为：

- `talker(backbone + logits处理 + cp + codec_sum) -> code2wav -> wav`

这意味着当前问题不是 Triton 编排多个子引擎的精度协同，而是 **单个 TRT engine 内部不同子图的精度约束**。

### 2. 当前工程已经区分了"计算精度"和"I/O 精度"

当前 manifest 和构建链路中已经存在两层概念：

- `engine_dtype`：TRT builder 计算精度
- `triton_io_float_dtype`：TRT/Triton 暴露给外部的浮点 I/O 类型

这说明工程设计上已经接受：

- 内部计算 dtype 与外部 I/O dtype 可以不同
- TRT 允许在图内部自动插入 cast / reformat

### 3. 当前实现仍然只支持"整张图一个计算精度"

现有构建脚本会把 `dtype` 作为全局参数传给 `trtexec`，尚未支持：

- `backbone_precision`
- `cp_precision`
- `code2wav_precision`

因此混合精度能力的核心缺口在 **build/export 层**，不是 runtime 层。

### 4. 现有 ONNX 图具备按子图识别的条件

现成的 `talker_code2wav_fused.onnx` 中，节点名保留了稳定前缀：

- `/talker_fused/talker_unified/...`
- `/talker_fused/cp/...`
- `/talker_fused/codec_sum/...`
- `/code2wav/...`

这意味着可以基于前缀对 TensorRT layer 进行分类，从而实现"按子模块设定计算精度"。

### 5. 当前构建链路关键文件

| 文件 | 职责 | 混合精度影响 |
|------|------|-------------|
| `scripts/python/qwen3tts_tools/trtexec.py` | trtexec Python 封装，`run_trtexec()` / `build_talker_code2wav_fused()` | 需扩展以传递 layer precision 参数 |
| `scripts/python/trt_fused_io_formats.py` | 生成 `--inputIOFormats` / `--outputIOFormats` | I/O 格式不随内部混合精度变化，保持统一 `triton_io_float_dtype` |
| `scripts/python/triton_manifest_io.py` | 读写 `triton_manifest.json` | 需扩展以记录三段精度字段 |
| `scripts/python/schemas/triton_manifest.schema.json` | manifest JSON schema | 需增加 `backbone_precision` / `cp_precision` / `code2wav_precision` |
| `scripts/python/update_triton_manifest_profile.py` | 构建 profile 回写 | 需记录混合精度配置 |
| `scripts/python/generate_triton_configs.py` | 生成 Triton config.pbtxt | 不受影响（I/O dtype 统一） |
| `engine/config.py` | `ModelArchConfig` / `EngineProfileConfig` | `EngineProfileConfig` 需增加混合精度字段 |
| `scripts/export/export_09_talker_code2wav_fused.py` | 导出 fused ONNX | 不受影响（ONNX 统一 FP32 导出） |

## 目标

本方案的直接目标如下：

1. 保持 `talker_code2wav_fused` 单引擎生产路径不变
2. 支持 `backbone`、`cp`、`code2wav` 三段计算精度独立配置
3. 外部 engine 浮点 I/O 先保持统一 dtype
4. 运行时继续按 engine 实际 I/O dtype 自动适配输入输出张量
5. 优先解决 `cp` 在 `bf16` 下的数值稳定性问题

建议的第一阶段目标配置：

- `backbone = bf16`
- `cp = fp32`
- `code2wav = bf16`
- `io_float = bf16`

## 设计决策

### 决策 1：外部 I/O 采用统一浮点 dtype，而不是随内部子图逐段漂移

推荐保持：

- 浮点主 I/O：统一使用一个 `triton_io_float_dtype`
- 控制张量：继续保留固定类型规则
  - `position_ids` / `token_counts` / `full_codec` 等保持整数类型
  - `gumbel_noise` / `cp_gumbel_noise` / `temperature` / `penalty` / `cache_position` 保持 `fp32`

原因：

- fused engine 的外部边界不需要暴露内部每一段的精度差异
- 统一 I/O dtype 能减少 runtime 分支与调试复杂度
- TRT 内部可以自动在 `backbone -> cp -> code2wav` 边界插入转换
- 当前 executor 已经支持依据 engine 实际 tensor dtype 组织输入输出

不推荐的方案：

- 把外部 I/O dtype 设计成"内部模块精度的直接镜像"
- 为 fused engine 的每个外部 float tensor 单独暴露异构 dtype 策略

该类方案收益有限，但会显著增加 manifest、构建参数、运行时适配和验证复杂度。

### 决策 2：优先保留单引擎架构，不立即拆分多引擎

拆分为 `backbone.engine + cp.engine + code2wav.engine` 理论上最直观，但短期不推荐，原因包括：

- 当前生产链已经针对 fused engine 做了调度和缓存布局优化
- 多引擎会增加边界同步、上下文切换和工程复杂度
- 现有 fused ONNX 图已经具备按前缀做 layer precision 约束的条件

只有在以下情况出现时，才考虑拆引擎：

- 单引擎混合精度在 TRT 中无法稳定复现目标行为
- 某一段必须暴露为独立 runtime 边界，便于外部显式控制 dtype
- 未来需要对子模块做独立部署、缓存或复用

### 决策 3：实现上区分"短期验证路线"和"长期稳定路线"

#### 短期验证路线

使用 TensorRT 10.x 可用的 layer precision 约束能力，快速验证：

- `cp=fp32` 是否能显著缓解 `bf16` 幻觉
- `backbone/code2wav` 保持低精度时性能损失是否可接受

目标是先证明方案有效，不先追求最终形态的构建器封装。

#### 长期稳定路线

逐步将 build 能力从纯 `trtexec` 参数堆叠，迁移到 TensorRT Python API builder。

原因：

- 当前工作区中产物显示使用的是 TensorRT `10.13.3.9`
- TensorRT 10.x 仍支持 layer precision / output type 相关能力
- TensorRT 11.x 对相关 `trtexec` 选项有较大调整，不宜把长期方案完全押在 CLI 能力上

因此推荐：

- 短期：用 TRT 10.x 快速验证混合精度
- 中期：补齐 Python builder 版本，降低未来 TRT 升级风险

### 决策 4：codec_sum 归属 backbone 精度域

ONNX 图中 `codec_sum` 子图前缀为 `/talker_fused/codec_sum/...`，逻辑上它是 backbone 输出 logits 到 CP 输入的桥接段。将其归入 backbone 精度域的原因：

- `codec_sum` 计算量小（一次 sum 操作），独立精度控制收益有限
- 将 `codec_sum` 留在低精度（bf16）不增加 cast 开销
- 如果未来发现 `codec_sum` 本身也需要 fp32，可通过同样机制单独配置

## trtexec 混合精度机制

### TRT 10.x 的 `--layerPrecisions` 选项

TensorRT 10.x 的 trtexec 支持 `--layerPrecisions` 选项，语法为：

```bash
--layerPrecisions=layer_name:precision[,layer_name:precision...]
```

其中：
- `layer_name`：ONNX 节点名或 TRT 自动生成的层名
- `precision`：`fp32`、`fp16`、`bf16`、`fp8` 等

**关键限制**：
- 必须与 `--bf16` 或 `--fp16` 全局标志配合使用（全局标志设置默认精度，`--layerPrecisions` 覆盖指定层）
- 不指定 `--layerPrecisions` 的层使用全局精度
- 层名必须精确匹配 TRT 构建时的 layer name，不是 ONNX 节点名

### TRT Python API 的 layer precision 约束

```python
import tensorrt as trt

# 构建 network
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("talker_code2wav_fused.onnx")

# 遍历层并按前缀设置精度
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
    # 其他层保持全局默认精度

# 设置全局 builder 精度
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.BF16)  # 全局 bf16
```

**Python API 的优势**：
- 可以用前缀匹配而非精确层名
- 可以同时设置 `layer.precision` 和 `set_output_type()`（trtexec `--layerPrecisions` 只设置 precision）
- 更容易做构建前审计（打印所有层名和分类结果）
- 更容易做构建后验证（检查 engine 中的实际精度分布）

### 短期 PoC 的 trtexec 路径

当前 `run_trtexec()` 支持 `extra_args` 参数，可以传入任意 trtexec 参数：

```python
# trtexec.py 中的现有接口
result = run_trtexec(
    onnx=str(onnx_path),
    engine=str(engine_path),
    dtype="bf16",              # 全局 bf16
    extra_args=[
        # 通过 --layerPrecisions 将 cp 子图提升到 fp32
        "--layerPrecisions="
        + ",".join(f"{name}:fp32" for name in cp_layer_names),
    ],
    ...
)
```

**但存在实际困难**：
1. 需要先解析 ONNX 获取精确层名列表
2. TRT 构建时层名可能与 ONNX 节点名不完全一致
3. 每次导出变更后层名可能变化，维护成本高

因此短期 PoC 推荐使用 **Python builder + 前缀匹配**，而非 trtexec `--layerPrecisions`。

## 配置方案

建议在 manifest / build 配置中引入新的精度字段：

```yaml
precision:
  backbone: bf16
  cp: fp32
  code2wav: bf16
  io_float: bf16
```

也可使用扁平字段：

```yaml
backbone_precision: bf16
cp_precision: fp32
code2wav_precision: bf16
triton_io_float_dtype: bf16
```

推荐优先使用扁平字段，原因是更容易兼容现有 manifest 结构。

建议含义如下：

- `backbone_precision`
  - 控制 `/talker_fused/talker_unified/` 和 `/talker_fused/codec_sum/` 主干计算精度
- `cp_precision`
  - 控制 `/talker_fused/cp/` 子图计算精度
- `code2wav_precision`
  - 控制 `/code2wav/` 子图计算精度
- `triton_io_float_dtype`
  - 控制绝大多数外部浮点 I/O 张量的 dtype

### Manifest Schema 变更

在 `scripts/python/schemas/triton_manifest.schema.json` 中增加：

```json
{
  "backbone_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Backbone (talker_unified + codec_sum) 计算精度；默认跟随 engine_dtype"
  },
  "cp_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Code Predictor 计算精度；设为 fp32 可缓解 BF16 数值敏感性"
  },
  "code2wav_precision": {
    "type": "string",
    "enum": ["fp32", "bf16", "fp16", "fp8"],
    "default": "bf16",
    "description": "Code2Wav 解码器计算精度；默认跟随 engine_dtype"
  }
}
```

向后兼容规则：
- 如果三个字段均未指定，退化为 `engine_dtype` 统一精度（与当前行为一致）
- 如果部分指定，未指定的字段默认为 `engine_dtype`
- `engine_dtype` 仍然保留，作为全局默认精度的 fallback

### CLI 参数设计

```bash
# 全局精度（向后兼容）
bash scripts/bash/build_engines.sh --variant custom-1.7b --dtype bf16

# 混合精度
bash scripts/bash/build_engines.sh --variant custom-1.7b --dtype bf16 --cp-precision fp32

# 显式指定每段精度
bash scripts/bash/build_engines.sh --variant custom-1.7b \
    --backbone-precision bf16 \
    --cp-precision fp32 \
    --code2wav-precision bf16 \
    --io-float-dtype bf16
```

环境变量映射：

```bash
ENGINE_DTYPE=bf16                  # 全局精度
BACKBONE_PRECISION=bf16            # backbone 精度
CP_PRECISION=fp32                  # cp 精度
CODE2WAV_PRECISION=bf16            # code2wav 精度
TRITON_IO_FLOAT_DTYPE=bf16        # I/O 精度
```

## 实施步骤

### 阶段 0：方案固化与命名收敛

目标：

- 确认精度字段命名
- 确认第一阶段默认值
- 确认"统一 I/O float dtype"的设计原则

输出：

- manifest 字段定义
- CLI 参数定义
- 文档更新

建议默认值：

- `backbone_precision = bf16`
- `cp_precision = fp32`
- `code2wav_precision = bf16`
- `triton_io_float_dtype = bf16`

### 阶段 1：扩展 manifest 与构建参数

#### 1.1 Schema 更新

**文件**：`scripts/python/schemas/triton_manifest.schema.json`

- 增加 `backbone_precision`、`cp_precision`、`code2wav_precision` 字段
- 保持 `engine_dtype` 作为兼容字段

#### 1.2 Manifest 读写

**文件**：`scripts/python/triton_manifest_io.py`

- `build_manifest_for_export()` 增加三段精度字段写入
- 精度字段默认值规则：未指定时取 `engine_dtype`

**文件**：`scripts/python/update_triton_manifest_profile.py`

- `engine_profile` 部分增加三段精度记录

#### 1.3 I/O 格式不变

**文件**：`scripts/python/trt_fused_io_formats.py`

- **不需要修改**。I/O 格式由 `triton_io_float_dtype` 统一控制，不随内部混合精度变化
- 控制张量（`gumbel_noise`、`cp_gumbel_noise`、`temperature`、`penalty`、`cache_position`）保持 `fp32:chw`
- 浮点主 I/O 保持 `{triton_io_float_dtype}:chw`

#### 1.4 Engine config

**文件**：`engine/config.py`

- `EngineProfileConfig` 增加 `backbone_precision`、`cp_precision`、`code2wav_precision` 字段
- `to_model_config()` 中 I/O dtype 逻辑不变（仍然使用 `triton_io_float_dtype`）

阶段目标：

- 仅修改配置与 metadata 流，不改运行时逻辑

### 阶段 2：实现单引擎混合精度 build

推荐分两步：

#### 2.1 快速验证版（Python builder PoC）

新增 `scripts/python/qwen3tts_tools/mixed_precision_builder.py`，实现：

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

核心逻辑：

1. **解析 ONNX**：`trt.OnnxParser` 解析 fused ONNX 到 TRT network
2. **遍历 layer 分类**：按 `layer.name` 前缀归类到 `backbone` / `cp` / `code2wav` / `other`
3. **审计阶段**：打印每个分类的 layer 数量和代表性名称，供构建前检查
4. **施加精度约束**：
   - 全局 flag：`config.set_flag(trt.BuilderFlag.BF16)` 或 `FP16`
   - CP 前缀层：`layer.precision = trt.float32` + `layer.set_output_type(0, trt.float32)`
   - 其他前缀层：按配置设置对应精度
5. **设置 I/O dtype**：通过 `config.set_flag()` 和 `--inputIOFormats` 等效的 Python API
6. **构建 engine**：`builder.build_serialized_network(network, config)`
7. **保存 engine**：写入 `.engine` 文件
8. **后验证**：可选地用 `trt.Runtime` 反序列化并检查 layer info

关键设计点：

- **前缀分类表**：

| 前缀 | 归类 | 默认精度 |
|------|------|---------|
| `/talker_fused/talker_unified/` | backbone | bf16 |
| `/talker_fused/codec_sum/` | backbone | bf16 |
| `/talker_fused/cp/` | cp | fp32 |
| `/code2wav/` | code2wav | bf16 |
| 其他 | backbone | bf16 |

- **未匹配层处理**：前缀不匹配任何已知模式的 layer 归入 backbone 精度域，并发出 `logger.warning`
- **构建前审计**：在设置精度前，打印完整的 layer 分类统计：

```
Layer classification summary:
  backbone: 342 layers (prefixes: /talker_fused/talker_unified/, /talker_fused/codec_sum/)
  cp:       45 layers  (prefix: /talker_fused/cp/)
  code2wav: 89 layers  (prefix: /code2wav/)
  unclassified: 3 layers (warning: check prefix table)
```

PoC 验证重点：

- TensorRT 是否接受该组合
- engine 是否可成功 build
- build 后的 layer info 中精度分布是否符合预期
- 推理输出是否改善

#### 2.2 集成到 `build_talker_code2wav_fused()`

**文件**：`scripts/python/qwen3tts_tools/trtexec.py`

在 `build_talker_code2wav_fused()` 中增加分支逻辑：

```python
def build_talker_code2wav_fused(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    backbone_precision: str = "",    # 新增，默认跟随 dtype
    cp_precision: str = "",          # 新增，默认跟随 dtype
    code2wav_precision: str = "",    # 新增，默认跟随 dtype
    triton_io_float_dtype: str = "",
    ...
) -> int:
    # 解析默认值
    if not backbone_precision:
        backbone_precision = dtype
    if not cp_precision:
        cp_precision = dtype
    if not code2wav_precision:
        code2wav_precision = dtype

    # 判断是否为混合精度
    is_mixed = len({backbone_precision, cp_precision, code2wav_precision}) > 1

    if is_mixed:
        # 使用 Python builder 路径
        from qwen3tts_tools.mixed_precision_builder import build_fused_mixed_precision
        return build_fused_mixed_precision(...)
    else:
        # 使用现有 trtexec 路径（向后兼容）
        return _build_fused_trtexec(...)
```

#### 2.3 CLI 集成

**文件**：`scripts/python/qwen3tts_cli/` 中的 build 子命令

- 增加 `--backbone-precision`、`--cp-precision`、`--code2wav-precision` 参数
- 传递到 `build_talker_code2wav_fused()`

### 阶段 3：ONNX 节点前缀验证

为防止导出代码变更导致前缀漂移，增加构建前验证：

**新增**：`scripts/python/qwen3tts_tools/layer_audit.py`

```python
def audit_onnx_prefixes(onnx_path: Path) -> dict[str, list[str]]:
    """扫描 ONNX 图，按前缀分类所有节点，返回分类报告。

    Returns:
        {category: [node_names]} 字典

    Raises:
        LayerAuditError: 如果发现未分类节点超过阈值
    """
```

验证规则：
- 所有节点必须匹配已知前缀之一
- 未匹配节点数超过 5% 触发 `LayerAuditError`
- 每次构建前自动运行审计，失败则中止构建

在 `build_fused_mixed_precision()` 中集成：

```python
# 构建前审计
audit = audit_onnx_prefixes(onnx_path)
if audit.get("unclassified_ratio", 0) > 0.05:
    raise LayerAuditError(
        f"超过 5% 的节点无法分类，可能存在前缀漂移: {audit}"
    )
```

### 阶段 4：运行时一致性校验

当前 runtime 已具备按实际 engine tensor dtype 组织输入输出的基础能力，但仍建议补充以下校验：

- manifest 声明的 `triton_io_float_dtype` 与 engine 实际 I/O dtype 一致
- engine 实际存在所有预期输入输出名
- 控制张量类型仍满足约定
- manifest 中 `backbone_precision` / `cp_precision` / `code2wav_precision` 与 engine 实际 layer 精度分布一致（构建后校验）

若发现不一致，应在 engine load 阶段明确报错，而不是静默退化。

### 阶段 5：验证与回归测试

至少补充以下验证维度：

#### 功能正确性

- `cp=fp32` 时 engine 能正常 build / load / infer
- 外部 I/O 仍能按统一 dtype 正常喂入
- 输出音频张量与状态张量类型符合预期

#### 数值稳定性

- 与现有 `bf16` 全图版本对比
- 验证幻觉复现样本是否改善
- 比较 `full_codec`、`codec_sum`、`wav`、`EOS` 发射行为
- 重点验证 `docs/dev/investigation/streaming_hallucination.zh-CN.md` 中已记录的不良转储步骤 `000003`、`000004`、`000008`

#### 性能与资源

- 首包延迟
- 单步 decode 延迟
- 显存占用
- engine build 时间
- **CP 区域内部 cast 开销**：通过 `--dumpLayerInfo` 或 Python builder 后验证观察 reformat 层数量

#### 回归兼容性

- `fp32` 全图
- `bf16` 全图
- `fp16` 全图
- 混合精度组合

#### 测试代码

建议新增：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/unit/test_mixed_precision_config.py` | manifest 字段解析、默认值、向后兼容 |
| `tests/unit/test_layer_audit.py` | 前缀分类、未匹配节点检测 |
| `tests/integration/test_mixed_precision_build.py` | 端到端构建（需要 GPU + ONNX 文件） |

## 验证矩阵建议

第一轮建议至少覆盖以下组合：

| Backbone | CP | Code2Wav | I/O | 用途 |
|----------|----|----------|-----|------|
| bf16 | bf16 | bf16 | bf16 | 当前基线 |
| bf16 | fp32 | bf16 | bf16 | **首选候选** |
| bf16 | fp32 | fp32 | bf16 | 判断 code2wav 是否也敏感 |
| fp32 | fp32 | fp32 | fp32 | 精度上界参考 |
| fp16 | fp32 | fp16 | fp16 | 观察 fp16 平台表现 |

优先级最高的是第二行。

### 数值验证方法

对于每种配置组合，建议执行以下验证流程：

1. **独立 CP 奇偶性**：使用 `tests/tools/verify_code_predictor_trt.py` 验证 10 次随机试验
2. **不良状态回放**：使用 `scripts/python/analyze_engine_dump.py` 在 4a 转储步骤 `000003` / `000004` / `000008` 上比较 `full_codec`
3. **端到端音频**：使用 `tests/tools/run_engine_long_case.py` 在 4a / story 用例上运行，比较音频时长和内容

## 风险

### 1. TensorRT 版本演进风险

当前工作流可利用 TRT 10.x 的能力完成 PoC，但未来升级到 TRT 11.x 后，纯 `trtexec` 路线的可维护性会下降。

缓解方式：

- 不把长期方案绑定在 `trtexec --layerPrecisions` 上
- 尽早补齐 Python builder 实现

### 2. 节点命名漂移风险

混合精度分类如果依赖 ONNX 节点前缀，未来导出代码调整后可能导致前缀变化。

缓解方式：

- 在导出阶段显式约束模块命名（`export_09_talker_code2wav_fused.py` 中 `torch.onnx.export` 的 `_prefix` 参数）
- 在 build 前做 layer 分类审计（阶段 3）
- 若关键前缀不存在则直接失败

### 3. 内部自动 cast 带来的性能损失

`backbone -> cp -> code2wav` 间的精度切换会引入 reformat/cast。

缓解方式：

- 优先只提升 `cp` 到 `fp32`
- 通过 profile / dumpLayerInfo / Python builder 后验证观察转换数量
- 若转换过多，再评估是否需要局部重构图边界

预期 cast 开销评估：
- `backbone(bf16) -> cp(fp32)`：1 次 cast（hidden 张量从 bf16 到 fp32）
- `cp(fp32) -> codec_sum(bf16)`：1 次 cast（logits 从 fp32 到 bf16）
- `code2wav(bf16)` 内部：无额外 cast
- 总计约 2 次显式 cast，加上 TRT 可能插入的隐式 reformat

### 4. 问题可能不只来自 CP 精度

虽然当前定位指向 `cp` 的 `bf16` 敏感性，但仍需防止"改了 CP 精度后问题只部分改善"的情况。

缓解方式：

- 保留 `code2wav=fp32` 的对照实验
- 对 `hidden/logits/full_codec/wav` 逐级比对
- 参考 `streaming_hallucination.zh-CN.md` 发现 #18：强制 `hidden/logits` 为 fp32 也改变了 TRT 数值，需关注这种交互效应

### 5. Python builder 与 trtexec 构建结果差异风险

Python builder 和 trtexec 可能产生不同的 engine 优化结果，即使输入 ONNX 相同。

缓解方式：

- 在非混合精度场景下，用相同 ONNX 对比 Python builder 和 trtexec 的 engine 输出
- 确保 `bf16` 全图基线在两个路径下行为一致
- 在 CI 中保留 trtexec 路径作为 fallback 验证

## 回滚策略

如果混合精度方案在验证阶段遇到不可接受的回归，回滚方案如下：

1. **构建层回滚**：将 `cp_precision` 恢复为与 `engine_dtype` 相同，回退到全图统一精度
2. **manifest 回滚**：删除 `backbone_precision` / `cp_precision` / `code2wav_precision` 字段，`engine_dtype` 仍为有效配置
3. **运行时无影响**：运行时不消费混合精度字段，回滚不需要改运行时代码
4. **引擎重建**：回滚后需重新构建 engine（全图统一精度）

关键点：混合精度字段是**增量添加**的，不修改现有字段的语义，因此回滚仅涉及"不使用新字段"。

## 推荐推进顺序

建议按以下顺序推进：

1. 扩展 manifest / CLI / schema，增加三段精度配置
2. 保持统一 `triton_io_float_dtype`
3. 实现 Python builder PoC + ONNX 前缀审计
4. 优先验证 `bf16 + fp32(cp) + bf16`
5. 若效果成立，集成到 `build_talker_code2wav_fused()`
6. 补完整测试矩阵与文档
7. 中期：评估是否需要 Python builder 替代 trtexec 作为统一构建路径

## 最终建议

当前阶段最推荐的方案是：

- 保持单个 `talker_code2wav_fused` 引擎
- 外部 I/O 使用统一浮点 dtype
- 内部让 `cp` 提升到 `fp32`
- `backbone` 与 `code2wav` 先保持低精度

这条路线对现有工程侵入最小，最有机会在较短周期内验证"是否确实由 `cp` 的 `bf16` 数值问题触发幻觉"，同时也为未来更通用的 TensorRT 混合精度构建能力打基础。
