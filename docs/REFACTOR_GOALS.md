# 项目治理重构目标

> 分支：`refact`
> 编写日期：2026-06-16
> 状态：待实施

---

## 0. 问题诊断

当前 `scripts/`、`tests/`、`client/`、`demo_api/` 存在以下结构性问题：

### 0.1 无单一真相源

| 概念 | 重复实现 | 位置 |
|------|---------|------|
| Triton 请求构建 | `TtsRequest` + `build_payload` | `demo_api/triton_client.py` |
| | `build_request_payload` | `tests/support/triton_streaming.py` |
| | `build_variant_request_payload` | `tests/support/triton_streaming.py` |
| Triton 流式推理 | `stream_once` (async) | `demo_api/triton_client.py:205-352` |
| | `infer_stream` / `infer_stream_sequence` (sync) | `tests/support/triton_streaming.py:175-260` |
| 事件/结果 Schema | `TraceEvent` / `RunMetrics` / `RunResult` | `demo_api/schemas.py` |
| | 同名但不同结构 | `tests/schemas.py`（如存在） |
| WAV 写入 | `save_wav` | `tests/support/triton_streaming.py:55-65` |
| | `save_wav` | `scripts/python/qwen3tts_tools/audio.py:15-32` |
| | `make_wav` | `tests/gen_audio.py:58-69` |
| _decode_obj | `_decode_obj` | `demo_api/triton_client.py:37-39` |
| | `decode_obj` | `tests/support/triton_streaming.py:43-46` |
| 服务端点常量 | `DEFAULT_TRITON_GRPC` 等 | `scripts/python/qwen3tts_tools/common.py` |
| | `DEFAULT_TRITON_GRPC` 等 | `demo_api/triton_client.py:16-17` |
| REPO_ROOT | `REPO_ROOT` | `tests/paths.py`、`tests/token_streaming.py`、`qwen3tts_tools/common.py` |

### 0.2 目录职责混乱

- `tests/save.py`：音频播放模拟工具，放在 tests/ 根目录，不是测试也不是工具
- `tests/gen_audio.py`：Triton 音频生成脚本，依赖 `qwen3tts_tools` + `tests/support`，属于 tools 而非 tests
- `tests/paths.py`：路径常量，与 `qwen3tts_tools/common.py` 重叠
- `tests/token_streaming.py`：token 分块辅助，应归入 `tests/support/`
- `scripts/python/` 混合了 30+ 入口脚本，其中部分属于用户工具（应在 `tests/tools/`），部分属于内部工具

### 0.3 客户端未成为依赖中心

- `client/` 包本身是干净的（不依赖 engine/）
- 但 `demo_api/` 和 `tests/tools/` 完全不使用 client 包，各自重新实现了 Triton 客户端逻辑
- 结果：修一个协议字段要改 3 处

### 0.4 治理规则形同虚设

`scripts/README.md` 定义了清晰的规则，但实际状态：
- 67 个 Python 入口点散落各处
- 8 处 `_run` 函数重复、9 处 `StubExecutor` 类重复、7 处 `request_gen` 重复
- 用户工具混在 `scripts/python/`（如 `probe_serving_endpoints.py`、`test_serving_endpoints.py`）
- `scripts/export/deprecated/` 废弃代码未清理

### 0.5 README 过长且有过时引用

- README.md 662 行，包含大量参数细节，不适合首次阅读
- 引用 `tests/e2e/test_engine_standalone.py` 作为主要 E2E 入口，但规则已改为 `tests/tools/`
- 项目结构部分缺少 `client/` 子项目描述

---

## 1. 目标

### G1：目录清晰分层

**目标状态**：每个目录有唯一职责，目录间依赖方向明确，与 engine/ 的分层风格一致。

**目标结构**：

```
Qwen3-TTS-Triton/
├── engine/                        # 核心推理引擎（不变）
│   ├── core/                      #   执行器、调度原语
│   ├── backend/                   #   EngineLoop、Scheduler
│   ├── frontend/                  #   请求处理、分词
│   ├── gateway/                   #   gRPC / WebSocket 协议
│   ├── interface/                 #   输出协议
│   ├── runtime/                   #   运行时工具
│   ├── config.py
│   └── server.py
│
├── client/                        # 独立客户端 SDK 包（G2）
│   ├── src/qwen3_tts_client/      #   客户端实现
│   │   ├── _adapters/             #     传输适配器
│   │   ├── _internal/             #     内部工具
│   │   ├── _proto/                #     protobuf 生成代码
│   │   ├── client.py              #     同步客户端
│   │   ├── async_client.py        #     异步客户端
│   │   ├── audio.py               #     音频工具
│   │   ├── constants.py
│   │   ├── detect.py
│   │   └── exceptions.py
│   ├── src/qwen3_tts_protocol/    #   共享协议层（单一真相源）
│   │   ├── schemas.py             #     TraceEvent / RunMetrics / RunResult（来自 demo_api/schemas.py）
│   │   ├── triton_types.py        #     TtsRequest / build_payload（来自 demo_api/triton_client.py）
│   │   └── ...
│   ├── pyproject.toml
│   └── README.md
│
├── demo_api/                      # Web 演示应用（G3）
│   ├── app.py                     #   依赖 client 包，不再内嵌客户端逻辑
│   ├── llm_pk.py                  #   LLM PK 比较逻辑（纯业务，不含 Triton 协议细节）
│   ├── audio_assets.py
│   ├── audio_store.py
│   ├── jobs.py
│   ├── trace_store.py
│   ├── requirements.txt           #   新增 qwen3-tts-client 依赖
│   └── __main__.py
│
├── scripts/                       # 构建与部署生命周期（G4）
│   ├── bash/                      #   Shell 生命周期脚本
│   │   ├── autorun.sh
│   │   ├── setup_env.sh
│   │   ├── build_engines.sh
│   │   ├── deploy.sh
│   │   ├── compose.sh
│   │   ├── lib/                   #     共享 shell 库
│   │   └── ...
│   ├── export/                    #   模型导出实现（不变）
│   │   ├── export_all.py
│   │   ├── export_01_embeddings.py
│   │   └── ...
│   ├── compose/                   #   容器入口
│   └── demo/                      #   Demo 启动器
│
├── tests/                         # 测试与验证工具（G5）
│   ├── conftest.py
│   ├── unit/                      #   pytest 单元测试
│   ├── integration/               #   pytest 集成测试
│   ├── e2e/                       #   pytest E2E 测试
│   ├── repro/                     #   问题复现
│   ├── support/                   #   测试共享代码
│   │   ├── triton_streaming.py    #     删除，改用 client 包
│   │   ├── engine_standalone.py
│   │   └── token_streaming.py     #     ← 来自 tests/token_streaming.py
│   ├── tools/                     #   手动验证与 benchmark 工具
│   │   ├── serving_endpoints.py   #     依赖 client 包
│   │   └── ...
│   └── data/                      #   测试数据
│
├── docs/                          # 文档（G6）
│   ├── zh/
│   └── ...
├── webui/                         # 前端（不变）
├── third_party/                   # 子模块（不变）
└── workspace/                     # 运行时产物（gitignored）
```

**依赖方向**（不允许反向依赖）：

```
client  ←  demo_api
client  ←  tests/support  ←  tests/tools
client  ←  tests/unit
scripts/python/qwen3tts_tools  ←  tests/tools  （仅共享路径常量等轻量工具）
engine  ←  scripts/export  （导出脚本依赖 engine 结构）
```

### G2：客户端独立成包 + 成为依赖中心

**目标**：`client/` 是唯一客户端实现，其他模块通过 `pip install` 或 `sys.path` 引用。

**具体变更**：

1. **共享协议层** `qwen3_tts_protocol` 扩展：
   - 将 `demo_api/schemas.py` 中的 `TraceEvent` / `RunMetrics` / `RunResult` / `percentile` / `summarize_ttft` / `normalize_backend_result` 迁入 `client/src/qwen3_tts_protocol/schemas.py`
   - 将 `demo_api/triton_client.py` 中的 `TtsRequest` / `build_payload` / `build_action_payload` 迁入 `client/src/qwen3_tts_protocol/triton_types.py`
   - 将 `demo_api/triton_client.py` 中的 `probe_ready` 迁入 `client/src/qwen3_tts_client/_adapters/triton_grpc.py`（已有文件）

2. **Triton 流式推理** 统一：
   - `demo_api/triton_client.py` 的 `stream_once` → 迁入 `client/src/qwen3_tts_client/_adapters/triton_grpc.py`，作为 client 包的异步流式接口
   - `tests/support/triton_streaming.py` 的 `infer_stream` / `infer_stream_sequence` → 删除，tests/tools 改用 client 包的同步/异步接口
   - 保留 `build_text_stream_requests` 等纯数据构建函数在 protocol 层

3. **WAV 写入** 统一：
   - `scripts/python/qwen3tts_tools/audio.py` 的 `save_wav` → 迁入 `client/src/qwen3_tts_client/audio.py`（已有，扩展）
   - 删除 `tests/support/triton_streaming.py` 中的 `save_wav`
   - 删除 `tests/gen_audio.py` 中的 `make_wav`

4. **常量统一**：
   - `REPO_ROOT`、`DEFAULT_TRITON_GRPC` 等 → 保留在 `qwen3tts_tools/common.py`（这些是仓库内部常量，不属于客户端）
   - `tests/paths.py` → 删除，改用 `qwen3tts_tools.common` 或 conftest.py

5. **client 包发布**：
   - `client/pyproject.toml` 保持独立，不依赖 engine/
   - 内部组件（demo_api、tests）通过 `pip install -e ./client[all]` 引用
   - demo_api/requirements.txt 添加 `qwen3-tts-client`

### G3：demo_api 清理

**目标**：demo_api 只保留 Web 演示业务逻辑，所有客户端/协议逻辑依赖 client 包。

**具体变更**：

1. **删除 `demo_api/triton_client.py`**：
   - `TtsRequest` / `build_payload` / `build_action_payload` → `from qwen3_tts_protocol.triton_types import ...`
   - `probe_ready` → `from qwen3_tts_client import ...` 或内联极简探测
   - `stream_once` → `from qwen3_tts_client.adapters.triton_grpc import ...`
   - `measure_once` → 保留在 demo_api 中，但改用 client 包的流式接口

2. **删除 `demo_api/schemas.py`**：
   - `TraceEvent` / `RunMetrics` / `RunResult` / `percentile` / `summarize_ttft` → `from qwen3_tts_protocol.schemas import ...`
   - `BACKENDS` / `normalize_backend_result` → 迁入 `qwen3_tts_protocol`

3. **demo_api/triton_client.py 中 `TritonUnavailable` 异常**：
   - 迁入 `client/src/qwen3_tts_client/exceptions.py`

4. **demo_api/requirements.txt 更新**：
   ```
   aiohttp>=3.9
   numpy>=1.24
   qwen3-tts-client[all]
   ```

### G4：scripts/ 清理

**目标**：scripts/ 只保留构建/部署/导出生命周期，删除用户工具和重复入口。

**具体变更**：

1. **`scripts/python/` 清理**（当前 30+ 文件，目标 ≤15）：

   **迁移至 tests/tools/**（用户验证/验收工具）：
   - `probe_serving_endpoints.py` → 已有 `tests/tools/serving_endpoints.py`，**删除**
   - `test_serving_endpoints.py` → 同上，**删除**
   - `suggest_engine_profile.py` → 迁移至 `tests/tools/`
   - `compare_live_vs_exported_prefill.py` → 迁移至 `tests/tools/`
   - `compare_prefill_paths.py` → 迁移至 `tests/tools/`
   - `official_prefill.py` → 迁移至 `tests/tools/`
   - `official_vs_manual_rollout.py` → 迁移至 `tests/tools/`
   - `pytorch_streaming_baseline.py` → 迁移至 `tests/tools/`
   - `cp_sampled_parity.py` → 迁移至 `tests/tools/`
   - `greedy_punish_*.py`（3 个） → 迁移至 `tests/tools/`

   **保留在 scripts/python/**（构建/部署/配置工具）：
   - `generate_triton_configs.py`
   - `triton_manifest_io.py`
   - `update_triton_manifest_profile.py`
   - `trt_fused_io_formats.py`
   - `trt_fused_talk_c2w_profiles.py`
   - `build_talker_code2wav_fused_trt_host.py`
   - `codec_embedding_sum.py`
   - `audit_tooling_surface.py`

   **保留但归入 qwen3tts_tools/**（共享代码）：
   - `qwen3tts_tools/` 现有文件不变
   - `triton_manifest_io.py` 考虑迁入 `qwen3tts_tools/`

   **删除**（功能已被 tests/tools/ 覆盖或已废弃）：
   - `probe_serving_endpoints.py`（与 tests/tools/serving_endpoints.py 重复）
   - `test_serving_endpoints.py`（同上）
   - `trace_official_streaming.py`（旧版 trace 工具）
   - `analyze_code_predictor_dump.py`（一次性调试脚本）
   - `analyze_engine_dump.py`（同上）
   - `analyze_engine_dump_loop.py`（同上）
   - `repeat_engine_weather_case.py`（调试用例）

2. **`scripts/export/deprecated/` 清理**：
   - 删除 `deprecated/export_04_talker_context.py`
   - 删除 `deprecated/export_05_talker_decode_fused.py`
   - 删除 `deprecated/` 目录

3. **`scripts/bash/` 清理**：
   - 删除 `ngc_matrix.conf.bak`
   - 评估 `_probe_standalone.py` 是否应迁移至 tests/tools/

### G5：tests/ 清理

**目标**：tests/ 根目录只保留 conftest.py 和 __init__.py，其余归入子目录。

**具体变更**：

1. **根目录散落文件归位**：
   - `tests/save.py` → 核心逻辑迁入 `client/src/qwen3_tts_client/realtime.py`（详见 `docs/realtime_audio_stream_goal.md`），原文件删除
   - `tests/gen_audio.py` → 迁移至 `tests/tools/`（这是 Triton 音频生成工具）
   - `tests/paths.py` → 删除，改用 `qwen3tts_tools.common` 或 `conftest.py`
   - `tests/token_streaming.py` → 迁移至 `tests/support/`

2. **`tests/support/triton_streaming.py` 清理**：
   - `build_request_payload` / `build_variant_request_payload` → 迁入 `qwen3_tts_protocol`，本文件改为 `from qwen3_tts_protocol import ...` 的薄转发层，最终删除
   - `build_stream_request` / `build_stream_outputs` → 迁入 `qwen3_tts_protocol`
   - `build_text_stream_requests` → 迁入 `qwen3_tts_protocol`
   - `infer_stream` / `infer_stream_sequence` / `infer_text_stream` → 删除，改用 client 包
   - `StreamResult` → 评估与 `RunResult` 合并，统一到 `qwen3_tts_protocol`
   - `save_wav` → 删除，改用 `qwen3_tts_client.audio.save_wav` 或 `qwen3tts_tools.audio.save_wav`
   - `decode_obj` / `decode_audio_bytes` → 迁入 `qwen3_tts_protocol`

3. **`tests/tools/` 依赖更新**：
   - 所有 `from tests.support.triton_streaming import ...` → `from qwen3_tts_protocol import ...` 或 `from qwen3_tts_client import ...`
   - 所有手动构建 Triton 请求的代码 → 改用 `qwen3_tts_protocol.triton_types.build_payload`

4. **tests/unit/ 重复代码清理**：
   - 9 处 `StubExecutor` → 提取到 `tests/support/stubs.py`
   - 8 处 `_run` → 提取到 `tests/support/`
   - 7 处 `request_gen` → 提取到 `tests/support/`
   - 8 处 `encode_ids` → 提取到 `tests/support/`

### G6：README 与文档清晰化

**目标**：新用户读完 README 后能独立上手，文档不过时。

**具体变更**：

1. **README.md 重构**（当前 662 行 → 目标 ~300 行）：

   **保留**：
   - 引言与定位（精简）
   - 能力状态表
   - 快速开始（3 条命令即可运行）
   - 部署方式（standalone / engine-docker / triton，各 3-5 行）
   - 项目结构（更新后的结构）
   - 中文文档链接
   - 许可证

   **精简/迁移**：
   - "统一入口与控制参数"详细参数表 → 迁入 `docs/zh/deployment.md`，README 只保留最常用 3 个示例
   - "Engine Profile" 详细计算逻辑 → 迁入 `docs/zh/deployment.md`
   - "GPU 选择" 详细参数说明 → 迁入 `docs/zh/deployment.md`
   - "模型版本号" → 迁入 `docs/zh/deployment.md`
   - "Text Player 口径" → 迁入 `docs/zh/` 专门文档或 WebUI 自身 README
   - "流式协议" → 保留 WebSocket 示例，详细协议说明迁入 `docs/zh/`
   - "性能声明" → 保留核心声明，详细口径引用 benchmark_methodology.md

   **更新**：
   - 项目结构图：补充 `client/` 描述
   - 测试部分：更新入口命令，去掉对 `tests/e2e/test_engine_standalone.py` 的主入口引用
   - WebUI Demo 部分：精简，详细配置迁入 demo_api/ 或 webui/ 自身 README

2. **docs/zh/ 更新**：
   - `client_sdk.md`：更新为重构后的 client 包 API
   - `deployment.md`：吸收 README 迁出的详细参数
   - 新增 `docs/zh/project_structure.md`：详细目录职责与依赖关系图
   - 删除 `docs/todolist_2026-03-26.md`（过时）

3. **子项目 README**：
   - `client/README.md`：更新 API 示例
   - `demo_api/` 缺少 README → 新增
   - `scripts/README.md`：更新目录角色描述
   - `tests/README.md`：更新，补充 tools/ 与 support/ 的使用说明

---

## 2. 实施顺序

按依赖关系，必须按以下顺序执行：

### Phase 1：协议层建立（G2 的核心）

> 所有后续步骤的前提

1. 扩展 `client/src/qwen3_tts_protocol/`：
   - 新建 `schemas.py`（迁入 `demo_api/schemas.py` 的 TraceEvent/RunMetrics/RunResult 等）
   - 新建 `triton_types.py`（迁入 TtsRequest/build_payload/build_action_payload）
   - 新建 `audio.py`（统一 WAV 写入、音频解码函数）
2. 更新 `client/pyproject.toml`：确保 qwen3_tts_protocol 包含新模块
3. 确保 `pip install -e ./client[all]` 后所有新模块可导入
4. 运行现有 client 单测确保不破坏

**验收**：`from qwen3_tts_protocol.schemas import TraceEvent, RunMetrics, RunResult` 可用

### Phase 2：demo_api 改造（G3）

> 依赖 Phase 1

1. demo_api/requirements.txt 添加 qwen3-tts-client
2. demo_api/schemas.py → `from qwen3_tts_protocol.schemas import ...`
3. demo_api/triton_client.py 中的类型和请求构建 → `from qwen3_tts_protocol.triton_types import ...`
4. demo_api/triton_client.py 中的 stream_once → 改用 client 包的流式接口或保留为薄适配层
5. 最终 demo_api/triton_client.py 缩减为仅包含 demo 特有的 `measure_once` 和 WebSocket 适配逻辑
6. 运行 demo_api 的单元测试（tests/unit/test_demo_api.py）

**验收**：`demo_api/` 不再定义 TraceEvent/RunMetrics/RunResult/TtsRequest/build_payload

### Phase 3：tests/support 清理（G5 的核心）

> 依赖 Phase 1

1. `tests/support/triton_streaming.py` 的类型和请求构建 → 迁入 qwen3_tts_protocol
2. `tests/support/triton_streaming.py` 的推理函数 → 改用 client 包
3. `tests/token_streaming.py` → 迁入 `tests/support/`
4. `tests/paths.py` → 删除，改用 conftest.py 或 qwen3tts_tools.common
5. `tests/save.py` → 迁入 `tests/support/` 或删除
6. `tests/gen_audio.py` → 迁入 `tests/tools/`
7. 更新所有 `from tests.support.triton_streaming import ...` 的导入路径
8. 提取 tests/unit/ 中的重复代码到 `tests/support/stubs.py`
9. 运行全部 pytest

**验收**：`tests/` 根目录只有 conftest.py 和 __init__.py；`pytest tests/ -q` 全部通过

### Phase 4：scripts/python/ 清理（G4）

> 依赖 Phase 3（因为迁移的工具可能引用 support）

1. 迁移用户工具到 tests/tools/
2. 删除重复和废弃脚本
3. 删除 scripts/export/deprecated/
4. 删除 scripts/bash/ngc_matrix.conf.bak
5. 运行 audit_tooling_surface.py 确认入口数量减少

**验收**：`scripts/python/` 入口 ≤15；audit 报告重复 ≤2

### Phase 5：README 与文档（G6）

> 依赖 Phase 1-4（结构确定后才能写文档）

1. 重构 README.md
2. 更新 docs/zh/
3. 新增 demo_api/README.md
4. 更新 client/README.md、scripts/README.md、tests/README.md
5. 删除过时文档

**验收**：README ≤300 行；所有文档引用的路径与代码一致

### Phase 6：最终验证

1. 全量 pytest
2. demo_api 启动验证
3. client 包 pip install 验证
4. README 中的命令全部可执行
5. 运行 audit_tooling_surface.py 生成最终报告

---

## 3. 不变范围

以下内容本次**不改动**：

- `engine/` 目录结构不变
- `webui/` 目录不变
- `scripts/bash/` 核心脚本（autorun.sh/setup_env.sh/build_engines.sh/deploy.sh/compose.sh）不变
- `scripts/export/` 核心导出脚本（export_01 ~ export_09, export_all.py）不变
- `model_repository/` 不变
- `workspace/` 不变
- Git 历史和分支策略不变

---

## 4. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 导入路径变更导致测试大面积失败 | Phase 3 先更新所有导入，再逐个运行测试确认 |
| client 包新增协议层后包体积增大 | protocol 层保持纯 dataclass，不引入重依赖 |
| demo_api 改造后 WebUI 不可用 | Phase 2 完成后立即启动 demo_api + WebUI 做冒烟验证 |
| 迁移遗漏导致循环导入 | 依赖方向严格遵守 G1 中的方向图 |
| 用户习惯旧脚本路径 | 保留 1 个版本的兼容性 thin wrapper，指向新位置并打印 deprecation 警告 |

---

## 5. 验收标准

1. ✅ 项目目录清晰分层，每目录有唯一职责，与 engine/ 分层风格一致
2. ✅ `client/` 是唯一客户端实现，demo_api 和 tests/tools 通过 client 包引用，不重复实现
3. ✅ `client/` 可独立 `pip install`，不依赖 engine/
4. ✅ `tests/` 根目录只有 conftest.py 和 __init__.py
5. ✅ `scripts/python/` 入口 ≤15
6. ✅ audit_tooling_surface.py 报告中重复函数/类 ≤2
7. ✅ README ≤300 行，所有引用路径与代码一致
8. ✅ 全量 pytest 通过
9. ✅ `demo_api` 启动正常，WebUI 可访问
10. ✅ `pip install -e ./client[all]` 后 client SDK 可正常使用
