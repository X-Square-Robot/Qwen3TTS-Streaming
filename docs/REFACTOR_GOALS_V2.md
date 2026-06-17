# 项目精简重构目标（第二轮）

> 分支：`refact`
> 编写日期：2026-06-16
> 前置：第一轮重构已完成（协议层建立、demo_api 改造、scripts/python 清理、README 精简）
> 状态：已实施（Phase 1-4 完成，Phase 3/5 调整后跳过）

---

## 0. 问题诊断

### 0.1 tests/ 太多太杂

当前 tests/ 包含 **86 个 Python 文件**，其中 `tests/tools/` 独占 **42 个手动脚本**。用户看完目录后完全不知道该用哪个。

**具体问题**：

| 问题 | 例子 |
|------|------|
| 同类工具重复 9 套 | 音频对比工具：gen_audio / gen_engine_audio / gen_reference_audio / compare_official_vs_triton_audio / compare_official_vs_fused_onnx / full_chain_audio_listen / generate_audio_compare / fused_onnx_audio / long_streaming_listen_ab |
| 验证脚本碎片化 14 个 | verify_e2e / verify_e2e_trt / verify_e2e_trt_ref / verify_fused_triton_backend / verify_code2wav_streaming / verify_code_predictor_trt / verify_onnx_autoregressive / verify_precision_ort / verify_prototype_parity / verify_speech_tokenizer_encoder / verify_trt_talker / verify_multi_variant / verify_code_predictor_trt |
| 过时工具未清理 | greedy_punish_mode_matrix / greedy_punish_parity / greedy_punish_stagewise_compare — 调试特定 commit 的逻辑 |
| 根目录散落文件 | tests/paths.py、tests/test_engine_integration.py 不在任何规范子目录 |
| 集成测试冗余 | test_triton_assembly_packaging.py 和 test_triton_manifest_generator.py 重叠 |

### 0.2 scripts/bash/ 太裸太冗长

当前 scripts/bash/ 包含 **30 个脚本 + 15 个 lib 模块 ≈ 13,400 行**。大量 JSON 解析、环境检测、pip 安装、Docker 编排都用裸 Bash 实现，可维护性差。

**具体问题**：

| 问题 | 例子 |
|------|------|
| JSON 解析用 Bash | cross_host.sh 中 `cross_host_json_value()` 用 grep/sed 解析 JSON，脆弱且不可维护 |
| pip 安装散在 Bash | pip.sh 中 100+ 行 pip install 封装，应该用 requirements.txt / pyproject.toml |
| Docker 编排手写 | compose.sh 1017 行手写 docker compose 编排，应该用声明式 compose.yaml |
| 环境检测冗余 | env_plan.sh 771 行做 Python 版本兼容性检测，pip/uv 已有原生能力 |
| 重复封装 | probe_endpoints.sh / test_serving_endpoints.sh 只有 13 行，是 Python 脚本的一行包装 |
| 版本矩阵硬编码 | ngc_matrix.conf 手写 NGC 镜像兼容性矩阵 |

### 0.3 缺乏现代 Python 工具链利用

| 该用的工具 | 当前做法 |
|-----------|---------|
| `pyproject.toml` + `pip` | pip.sh 手动安装每个包 |
| `pytest` 插件 + fixture | 手写 Bash 探测脚本（probe_target.sh、verify_e2e_trt.sh） |
| `docker compose` 声明式 YAML | compose.sh 手写 1017 行命令式编排 |
| `pydantic` / `dataclass` | JSON 解析用 grep/sed |
| `click` / `argparse` CLI | autorun.sh 1373 行 Bash 参数解析 |

---

## 1. 目标

### G1：tests/tools 从 42 个精简到 ≤10 个

**原则**：同一类功能只保留一个入口，通过参数/子命令区分模式。

**目标工具清单**（10 个）：

| 工具 | 用途 | 合并来源 |
|------|------|---------|
| `serving_endpoints.py` | 服务验收 + TTFT benchmark | 保留 |
| `compare_audio.py` | 音频对比（official vs ORT vs TRT vs Triton，--mode 区分） | 合并：gen_audio / gen_engine_audio / gen_reference_audio / compare_official_vs_triton_audio / compare_official_vs_fused_onnx / full_chain_audio_listen / generate_audio_compare / fused_onnx_audio / long_streaming_listen_ab / gen_reference_audio |
| `verify_engine.py` | 引擎验证（ORT/TRT/fused，--phase 区分） | 合并：verify_e2e / verify_e2e_trt / verify_e2e_trt_ref / verify_fused_triton_backend / verify_onnx_autoregressive / verify_code_predictor_trt / verify_trt_talker / verify_multi_variant |
| `verify_components.py` | 组件级验证（code2wav / speech_tokenizer / precision） | 合并：verify_code2wav_streaming / verify_speech_tokenizer_encoder / verify_precision_ort / verify_prototype_parity |
| `benchmark.py` | 性能基准测试 | 合并：engine_standalone_benchmark / triton_concurrent_tts |
| `prefill_compare.py` | prefill 对比（--mode live/exported/manual） | 合并：official_prefill / compare_live_vs_exported_prefill / compare_prefill_paths / official_vs_manual_rollout / cp_sampled_parity |
| `gen_audio.py` | 通用音频生成 | 保留（已有，更新为使用 compare_audio 的公共层） |
| `suggest_engine_profile.py` | TRT profile 建议 | 保留 |
| `pad_tolerance_experiment.py` | pad 容忍度实验 | 保留（独立用途） |
| `vad_verification.py` | VAD 验证 | 保留（如存在） |

**删除**（过时/无价值）：
- `greedy_punish_mode_matrix.py`
- `greedy_punish_parity.py`
- `greedy_punish_stagewise_compare.py`
- `greedy_baseline.py`（与 compare_audio 重叠）

### G2：tests/unit + integration + e2e 精简

**目标**：清理冗余，根目录文件归位。

1. **`tests/test_engine_integration.py` → `tests/integration/test_engine_pipeline.py`**
2. **`tests/paths.py` → 合并入 `tests/conftest.py`**（paths.py 只有 30 行有效代码，conftest.py 已经引用它）
3. **`tests/integration/test_triton_assembly_packaging.py` + `test_triton_manifest_generator.py` → 合并为 `tests/integration/test_triton_packaging.py`**
4. **`tests/integration/test_export_consistency.py` → 删除**（逻辑已被 verify_engine.py 覆盖）

### G3：scripts/bash 从 30 个精简到 ≤15 个

**原则**：用 Python/pip/pytest/docker compose 替代裸 Bash，保留系统级操作的 Bash。

**保留的 Bash 脚本**（≤15 个，系统级操作）：

| 脚本 | 行数 | 保留原因 |
|------|------|---------|
| `autorun.sh` | 1373 | 核心入口，交互式 UX，但需精简（见 G4） |
| `setup_env.sh` | 379 | 子模块初始化 + 模型下载 |
| `build_engines.sh` | 904 | trtexec 调用（系统级） |
| `deploy.sh` | 746 | 部署编排 |
| `compose.sh` | 1017 | Docker compose 管理（但需重构为声明式） |
| `download_models.sh` | 94 | 模型下载 |
| `probe_target.sh` | 203 | 系统信息采集 |
| `verify_e2e_trt.sh` | 161 | 跨容器验证编排 |
| `model_versions.conf` | 26 | 版本锁定 |
| `ngc_matrix.conf` | 56 | NGC 兼容性矩阵 |
| `lib/build_pipeline.sh` | 577 | 构建流水线核心 |
| `lib/trtexec_runner.sh` | 274 | trtexec 抽象层 |
| `lib/cross_host.sh` | 358 | 跨机构建 |
| `lib/logging.sh` | 21 | 日志工具 |
| `lib/utils.sh` | 248 | 系统工具 |

**删除的 Bash 脚本**（被 Python/pip/pytest/docker 替代）：

| 脚本 | 替代方案 |
|------|---------|
| `pip.sh` (100+ 行) | `pyproject.toml` + `pip install -e .` |
| `venv.sh` (394 行) | `uv` / `pip` + `pyproject.toml` |
| `env_plan.sh` (771 行) | `pip check` + `pyproject.toml` 依赖声明 |
| `prerequisites.sh` (~100 行) | 合并入 `autorun.sh` 的 pre-check 或 Python 脚本 |
| `mirrors.sh` (~100 行) | `pip.conf` / 环境变量 |
| `network.sh` (~100 行) | Python `requests` + mirror fallback |
| `docker.sh` (779 行) | Python `docker` SDK + 声明式配置 |
| `triton.sh` (1101 行) | Python Triton 管理模块 |
| `engine.sh` (688 行) | Python `subprocess` 管理 |
| `status.sh` (~100 行) | Python 状态检查脚本 |
| `tools.sh` (30 行) | 直接 source lib/ 或 Python import |
| `build_triton.sh` (579 行) | 合并入 `deploy.sh` 的 Triton 子命令 |
| `build_on_target.sh` (141 行) | 合并入 `build_engines.sh` |
| `probe_endpoints.sh` (13 行) | `pytest tests/e2e/` 或 `python tests/tools/serving_endpoints.py` |
| `test_serving_endpoints.sh` (13 行) | 同上 |
| `run_engine_greedy_dump.sh` (34 行) | Python 脚本 + 环境变量 |
| `run_engine_story_full_dump.sh` (155 行) | Python 脚本 + argparse |
| `_probe_standalone.py` | 迁入 tests/tools/ |

### G4：创建项目级 Python CLI（替代 autorun.sh 的大部分逻辑）

**目标**：将 autorun.sh 中 1373 行的参数解析、环境检测、Phase 调度用 Python 重写，暴露为 `qwen3tts` CLI。

```
# 目标用法（替代 bash scripts/bash/autorun.sh all -m custom-1.7b）
qwen3tts all -m custom-1.7b
qwen3tts setup -m custom-1.7b
qwen3tts build -m custom-1.7b --max-batch-size 64
qwen3tts deploy -m custom-1.7b --gateway standalone
qwen3tts status
qwen3tts probe
```

**实现**：
- 使用 `click` 或 `argparse` 构建多级子命令
- 内部调用 subprocess 调度 Bash 脚本（渐进迁移，不一次性重写）
- pyproject.toml 中注册 `[project.scripts]` 入口

**渐进迁移路径**：
1. **Phase A**：创建 `scripts/python/qwen3tts_cli/` 包，仅做参数解析 + 转发到 Bash
2. **Phase B**：将 Bash 中的 JSON 解析、环境检测逻辑迁移到 Python
3. **Phase C**：将 pip 安装、Docker 管理等迁移到 Python SDK

### G5：创建项目级 pyproject.toml（统一依赖管理）

**目标**：项目根目录有 `pyproject.toml`，统一管理所有 Python 依赖。

```toml
[project]
name = "qwen3-tts-triton"
version = "0.1.0"

[project.optional-dependencies]
# 最小安装：纯 Python 测试
test = ["pytest>=8.0", "pytest-asyncio"]

# 引擎运行时
engine = ["torch>=2.1", "tensorrt>=10.0", "onnxruntime>=1.17"]

# 客户端
client = ["qwen3-tts-client[all]"]

# Demo API
demo = ["aiohttp>=3.9", "qwen3-tts-client[all]"]

# 导出脚本
export = ["torch>=2.1", "safetensors", "transformers"]

# 开发全量
dev = ["qwen3-tts-triton[test,engine,client,demo,export]"]

[project.scripts]
qwen3tts = "qwen3tts_cli:main"
```

**替代**：`pip.sh`、`venv.sh`、`env_plan.sh`、`requirements.txt` 散落各处。

---

## 2. 实施顺序

### Phase 1：tests/tools 合并（G1）

> 优先级最高，效果最显著

1. 创建 `tests/tools/compare_audio.py`（合并 9+1 个音频工具）
2. 创建 `tests/tools/verify_engine.py`（合并 8 个验证工具）
3. 创建 `tests/tools/verify_components.py`（合并 4 个组件验证工具）
4. 创建 `tests/tools/benchmark.py`（合并 2 个 benchmark 工具）
5. 创建 `tests/tools/prefill_compare.py`（合并 5 个 prefill 工具）
6. 删除 3 个 greedy_punish_* 和 greedy_baseline
7. 更新 tests/tools/README.md

**验收**：tests/tools/ 从 42 个文件精简到 ≤10 个

### Phase 2：tests 结构精简（G2）

> 依赖 Phase 1

1. `tests/test_engine_integration.py` → `tests/integration/`
2. `tests/paths.py` → 合并入 `tests/conftest.py`
3. 合并 integration/ 中的冗余测试
4. 删除 `tests/integration/test_export_consistency.py`

**验收**：tests/ 根目录只有 conftest.py 和 __init__.py

### Phase 3：项目级 pyproject.toml（G5）

> 前置：为后续 Python CLI 和依赖统一做基础

1. 创建根目录 `pyproject.toml`
2. 将 `demo_api/requirements.txt`、`scripts/python/qwen3tts_tools/` 的依赖声明统一到 pyproject.toml
3. 将 `pip.sh` 中列出的包版本约束迁移到 pyproject.toml
4. 验证 `pip install -e ".[dev]"` 可用

**验收**：`pip install -e ".[dev]"` 安装所有依赖

### Phase 4：scripts/bash 精简（G3）

> 依赖 Phase 3（pyproject.toml 就绪后才能删除 pip.sh）

1. 删除被 pyproject.toml 替代的脚本（pip.sh、venv.sh、env_plan.sh、mirrors.sh）
2. 合并 build_triton.sh → deploy.sh（Triton 子命令）
3. 合并 build_on_target.sh → build_engines.sh
4. 删除 13 行包装脚本（probe_endpoints.sh、test_serving_endpoints.sh）
5. 将 run_engine_*_dump.sh 替换为 Python 脚本
6. 将 _probe_standalone.py 迁入 tests/tools/
7. 将 network.sh 迁移为 Python 模块
8. 删除 tools.sh（直接 source lib/）

**验收**：scripts/bash/ 从 30 个脚本精简到 ≤15 个

### Phase 5：Python CLI 原型（G4）

> 依赖 Phase 3 + 4

1. 创建 `scripts/python/qwen3tts_cli/` 包
2. 实现子命令：`qwen3tts all/setup/build/deploy/status/probe`
3. 参数解析用 click/argparse
4. 内部 subprocess 调用剩余 Bash 脚本
5. 在 pyproject.toml 注册 `[project.scripts]`
6. 保留 autorun.sh 作为兼容入口，打印 deprecation 提示

**验收**：`qwen3tts all -m custom-1.7b` 可用

### Phase 6：最终验证

1. 全量 pytest
2. `pip install -e ".[dev]"` 验证
3. `qwen3tts status` 验证
4. README 更新（反映新工具名和 CLI 命令）
5. 删除 REFACTOR_GOALS.md 和本轮目标文档

---

## 3. 不变范围

- `engine/` 目录不变
- `client/` 包不变（第一轮重构已完成）
- `webui/` 不变
- `scripts/export/` 不变
- `scripts/bash/autorun.sh`、`setup_env.sh`、`build_engines.sh`、`deploy.sh` 核心流程不变（Phase 5 只加 CLI 入口，不重写逻辑）

---

## 4. 预期效果

| 指标 | 当前 | 目标 | 改善 |
|------|------|------|------|
| tests/tools/ 文件数 | 42 | ≤10 | -76% |
| tests/ 总文件数 | 86 | ≤35 | -59% |
| scripts/bash/ 脚本数 | 30 | ≤15 | -50% |
| scripts/bash/ 总行数 | ~13,400 | ~7,000 | -48% |
| 依赖管理 | pip.sh + 散落 requirements.txt | pyproject.toml | 统一 |
| 用户入口 | `bash scripts/bash/autorun.sh` | `qwen3tts` CLI | 现代 |
| 音频对比工具 | 9 个独立脚本 | 1 个 compare_audio.py | -89% |
| 验证工具 | 14 个 verify_* | 2 个（verify_engine + verify_components） | -86% |
