# 项目治理重构目标（第三轮：系统化治理）

> 分支：`refact`
> 编写日期：2026-06-17
> 前置：第二轮重构已完成（tests/tools 精简、scripts/bash 精简、pyproject.toml 建立、Python CLI 原型）
> 状态：待实施

---

## 0. 问题诊断：为什么这个项目仍然感觉像「散装代码」

前两轮重构聚焦于「量」的精简——减少文件数、合并同类工具、建立 pyproject.toml。这些工作有价值，但没有解决一个更根本的问题：**项目缺乏统一的层次感和归属感**。一个新用户克隆仓库后，面对的是：

### 0.1 目录扁平，没有归属层级

```
Qwen3-TTS-Triton/
├── engine/           ← 核心产品代码
├── client/           ← 独立 SDK 包
├── demo_api/         ← WebUI 后端
├── webui/            ← WebUI 前端
├── model_repository/ ← Triton 模型（只有 2 个 model.py）
├── scripts/          ← 一切运维脚本
├── tests/            ← 一切测试
├── docs/             ← 一切文档
├── resources/        ← 2 个参考音频文件
├── third_party/      ← git 子模块
└── workspace/        ← 运行时产物（gitignored）
```

这 11 个顶级目录没有分类、没有优先级。用户无法一眼看出「这个项目的核心是什么，辅助是什么，临时是什么」。`model_repository/` 只有 2 个 Python 文件却占了一个顶级目录；`resources/` 只有 2 个文件也是一个顶级目录。

### 0.2 文档语言混乱，没有统一规范

| 位置 | 语言 | 问题 |
|------|------|------|
| `README.md` | 中文 | 主 README |
| `client/README.md` | 英文 | SDK 文档 |
| `demo_api/README.md` | 中文 | Demo 文档 |
| `scripts/README.md` | 英文 | 脚本指南 |
| `tests/README.md` | 英文 | 测试指南 |
| `docs/architecture.md` | 中文（2333 行） | 架构文档 |
| `docs/*.md`（13 个） | 英文 | 技术设计文档 |
| `docs/zh/*.md`（5 个） | 中文 | 用户文档 |
| `docs/tooling_governance.md` | 英文 | 治理规范 |

**问题**：文档语言完全没有规律。同一个 `docs/` 目录下，有的是中文有的是英文，`docs/zh/` 只收了 5 篇但大量中文文档散在 `docs/` 根目录。用户不知道该看哪篇、先看哪篇、哪篇是给谁看的。

### 0.3 文档职责不清，读者无法导航

`docs/` 下 20 篇文档，没有分类：

- `architecture.md`（2333 行）既是架构文档又包含实现细节，既是给新用户的概览又包含开发者才需要的内部协议
- `progress_2026-03-25.md` 是历史记录，和设计文档混在一起
- `REFACTOR_GOALS_V2.md` 是流程文档，和技术文档混在一起
- `streaming_hallucination_investigation.md` 是调查报告，和目标设计文档混在一起
- `e2e_test_summary.md` 是测试报告，放在 docs 根目录

**没有索引页**，没有「新用户从这里开始」的路径。用户需要把 20 篇文档标题看一遍才能判断哪些和自己相关。

### 0.4 重复的协议定义散落

- `engine/gateway/tts.proto` 和 `client/src/qwen3_tts_client/_proto/tts.proto` 是同一份源文件的拷贝
- 生成的 `tts_pb2.py` / `tts_pb2_grpc.py` 在 `engine/gateway/` 和 `client/.../_proto/` 各有一份
- `workspace/model_repository/` 里还有第三份拷贝（deploy 时复制进去的）
- 没有单一的 proto 源头和生成流程

### 0.5 tests/ 仍然杂乱

虽然 V2 精简了 `tests/tools/`，但 tests/ 的整体结构仍然缺乏清晰的心智模型：

- `tests/support/` 和 `tests/conftest.py` 的职责边界模糊
- `tests/tools/` 的定位——手动验证工具——和 tests/ 的核心职能（自动化测试）放在一起，概念上冲突
- `tests/repro/` 是冻结的复现案例，不是测试
- `tests/data/` 是测试数据
- 四种东西（测试、工具、复现、数据）混在一个 `tests/` 里

### 0.6 配置散落，缺乏单一配置源

| 配置 | 位置 | 问题 |
|------|------|------|
| 引擎运行配置 | `engine.yaml` | 根目录 |
| 依赖管理 | `pyproject.toml` | 根目录 |
| Demo 依赖 | `demo_api/requirements.txt` | 独立 |
| Client 依赖 | `client/pyproject.toml` | 独立 |
| Pytest 配置 | `pytest.ini` | 根目录，和 `pyproject.toml` 中的 `[tool.pytest.ini_options]` 重复 |
| Docker Compose | `compose.yaml` + `compose.dev.yaml` | 根目录 |
| 模型版本 | `scripts/bash/model_versions.conf` | Bash 脚本目录 |
| NGC 镜像矩阵 | `scripts/bash/ngc_matrix.conf` | Bash 脚本目录 |
| 环境变量模板 | `.env.example` | 根目录 |
| Dockerfile ×3 | 根目录 | 根目录 |

根目录有 **13 个配置/构建文件**，新用户无法区分哪些是「需要关心的」、哪些是「构建基础设施」。

### 0.7 缺少 CLAUDE.md

没有项目级的 CLAUDE.md 文件，AI 助手（包括 Claude Code）无法快速理解项目约定、目录意图和常见操作。

### 0.8 残留的 .cursor/ 和 .vscode/ 配置

`.cursor/plans/` 包含 3 个中文 plan 文件，`.cursor/rules/` 包含项目规则。这些是 IDE 特有的配置，和项目本身混在一起。`.vscode/` 同理。

### 0.9 __pycache__ 散落

虽然 `.gitignore` 已覆盖 `__pycache__`，但磁盘上仍有 **27 个** `__pycache__` 目录存在，说明 `.gitignore` 是后来加的，之前提交过的可能还在 git 历史里。

---

## 1. 总体目标

**将 Qwen3-TTS-Triton 从「一堆相关代码的集合」治理为「一个有清晰层次的产品项目」。**

核心原则：

1. **一眼可懂**：新用户看根目录就能理解项目由哪几个部分组成
2. **统一语言**：项目有明确的语言策略，不混用
3. **单一入口**：每个关注点有且只有一个入口文件/目录
4. **分层归属**：顶级目录按角色分组，不是按实现技术分组
5. **文档有路**：文档按读者角色和阅读路径组织，不是扁平堆放
6. **配置收敛**：根目录只保留用户必须关心的配置，其余收归子系统

---

## 2. 目标

### G1：目录层级治理——建立清晰的顶级结构

**现状问题**：11 个顶级目录没有分类，`model_repository/` 和 `resources/` 各只有 2 个文件却占顶级目录。

**目标结构**：

```
Qwen3-TTS-Triton/
├── src/                        # 所有产品代码
│   ├── engine/                 #   推理引擎（从根目录迁入）
│   ├── client/                 #   Python SDK（保持独立 pyproject.toml）
│   ├── demo_api/               #   WebUI 后端（从根目录迁入）
│   └── model_repository/       #   Triton 模型定义（从根目录迁入）
│
├── infra/                      # 所有基础设施
│   ├── scripts/                #   生命周期脚本（保持现有结构）
│   ├── docker/                 #   Dockerfile + compose（从根目录迁入）
│   │   ├── Dockerfile.engine
│   │   ├── Dockerfile.triton
│   │   ├── Dockerfile.demo-api
│   │   ├── compose.yaml
│   │   └── compose.dev.yaml
│   └── export/                 #   模型导出脚本（从 scripts/ 迁入）
│
├── tests/                      # 所有测试（仅 pytest 可发现的内容）
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── tools/                      # 所有手动工具（从 tests/ 剥离）
│   ├── validation/             #   验证工具（原 tests/tools/）
│   ├── repro/                  #   复现案例（原 tests/repro/）
│   └── data/                   #   工具数据（原 tests/data/）
│
├── docs/                       # 所有文档（重新组织）
│   ├── zh/                     #   中文文档（面向用户）
│   └── en/                     #   英文文档（面向贡献者）
│
├── resources/                  # 静态资源（保留）
│
├── third_party/                # 第三方子模块（保留）
│
├── engine.yaml                 # 引擎配置（用户必须关心）
├── pyproject.toml              # 项目依赖（用户必须关心）
├── .env.example                # 环境变量模板（用户必须关心）
├── README.md                   # 项目入口文档
└── CLAUDE.md                   # AI 助手指引
```

**根目录文件数从 13 个收敛到 4 个**（+ README + CLAUDE.md）。

**关键变更**：

| 变更 | 原因 |
|------|------|
| `engine/` → `src/engine/` | 产品代码归入 src/，与基础设施分离 |
| `demo_api/` → `src/demo_api/` | 同上 |
| `model_repository/` → `src/model_repository/` | 同上 |
| `client/` → `src/client/` | 同上 |
| `Dockerfile.*` + `compose.*` → `infra/docker/` | 基础设施归入 infra/ |
| `scripts/export/` → `infra/export/` | 导出是基础设施，不是脚本 |
| `tests/tools/` → `tools/validation/` | 手动工具不是测试，概念上分离 |
| `tests/repro/` → `tools/repro/` | 复现案例不是测试 |
| `tests/data/` → `tools/data/` | 工具数据跟随工具 |

**不变**：

| 目录 | 原因 |
|------|------|
| `resources/` | 只有 2 个文件但语义独立 |
| `third_party/` | git 子模块，位置约定 |
| `webui/` | 前端项目有独立的 npm 生态，保持顶级 |
| `workspace/` | 运行时产物（gitignored），保持顶级 |

### G2：文档治理——统一语言、建立索引、按角色分层

**2A. 语言策略**：

| 文档类型 | 语言 | 目录 |
|----------|------|------|
| README（主入口） | 中文 | 根目录 |
| 用户文档（部署、使用、限制） | 中文 | `docs/zh/` |
| 开发者/贡献者文档 | 英文 | `docs/en/` |
| API/SDK 文档 | 英文 | `docs/en/` 或 `src/client/README.md` |
| 架构设计文档 | 英文 | `docs/en/` |

**规则**：`docs/` 根目录不再放文档。所有文档进入 `docs/zh/` 或 `docs/en/`。

**2B. 文档分类重组**：

当前 20 篇文档 → 按角色和阶段重新组织：

```
docs/
├── zh/                              # 面向用户（中文）
│   ├── README.md                    #   文档索引（新用户入口）
│   ├── quickstart.md                #   快速开始（从 README.md 提取）
│   ├── deployment.md                #   部署指南（已有）
│   ├── client_sdk.md                #   Client SDK（已有）
│   ├── benchmark_methodology.md     #   Benchmark 方法（已有）
│   ├── known_limitations.md         #   已知限制（已有）
│   └── roadmap.md                   #   路线图（已有）
│
└── en/                              # 面向贡献者/开发者（英文）
    ├── README.md                    #   文档索引（贡献者入口）
    ├── architecture.md              #   架构概览（从 2333 行精简为概览 + 拆分子篇）
    ├── architecture/                #   架构子篇（从 architecture.md 拆出）
    │   ├── decode_fsm.md            #     解码 FSM 设计
    │   ├── scheduler.md             #     调度器设计
    │   ├── prefix_cache.md          #     Prefix Cache 设计
    │   └── streaming_protocol.md    #     流式协议设计
    ├── design/                      #   设计目标文档
    │   ├── vad_design_goals.md      #     VAD 设计目标
    │   ├── observability_goals.md   #     可观测性目标
    │   ├── realtime_audio.md        #     实时音频目标
    │   └── engine_decisions.md      #     引擎架构决策
    ├── investigation/               #   调查报告
    │   ├── streaming_hallucination.md
    │   └── code2wav_state_size.md
    ├── operations/                  #   运维文档
    │   ├── cross_host_build.md      #     跨机构建
    │   ├── e2e_test_summary.md      #     E2E 测试总结
    │   ├── timing_metrics.md        #     计时指标参考
    │   └── tooling_governance.md    #     工具治理规范
    └── process/                     #   流程文档
        ├── refactor_goals_v1.md     #     V1 重构目标（归档）
        ├── refactor_goals_v2.md     #     V2 重构目标（归档）
        └── progress_2026-03-25.md   #     历史进度记录
```

**2C. architecture.md 精简**：

当前 `architecture.md` 有 2333 行，是全项目最大的单文件。它同时承载了：
- 新用户需要的概览信息
- 开发者需要的内部协议细节
- 调试需要的时序图

**目标**：精简为 200 行以内的架构概览 + 4-5 个子篇。子篇按需阅读，不需要全量消化。

**2D. 每个子目录的 README 索引**：

每个有意义的子目录必须有 README.md 作为该目录的入口说明。当前缺失的：
- `src/engine/README.md`
- `infra/README.md`
- `tools/README.md`
- `docs/zh/README.md`
- `docs/en/README.md`

### G3：协议定义单一源——消除 proto 拷贝

**现状**：`tts.proto` 有 3 份拷贝，生成的 `_pb2.py` / `_pb2_grpc.py` 有 3 份。

**目标**：

```
src/
├── _proto/                      # 协议定义单一源
│   ├── tts.proto                #   唯一的 .proto 源文件
│   ├── tts_pb2.py               #   生成的 Python 代码
│   └── tts_pb2_grpc.py
├── engine/
│   └── gateway/
│       └── ...                  #   import from src._proto
└── client/
    └── src/qwen3_tts_client/
        └── _proto/              #   import from src._proto（或发布时拷贝）
```

**实施策略**：

1. 将 `engine/gateway/tts.proto` 移至 `src/_proto/tts.proto`（单一源）
2. 生成代码也放在 `src/_proto/`
3. `engine/gateway/` 改为 `from src._proto import tts_pb2, tts_pb2_grpc`
4. `client/` 包作为独立发布包，build 时从 `src/_proto/` 拷贝生成的代码
5. 添加 Makefile / justfile 目标 `make proto` 一键重新生成
6. 删除 `engine/gateway/tts.proto` 和 `client/.../_proto/tts.proto`

### G4：tests/ 职责纯化——测试归测试，工具归工具

**现状**：`tests/` 混合了 4 种不同性质的东西。

**目标**：

```
tests/                           # 仅包含 pytest 可发现的内容
├── conftest.py
├── unit/
├── integration/
├── e2e/
└── support/                     # 测试共享代码

tools/                           # 手动工具（不是测试）
├── validation/                  # 原	tests/tools/
│   ├── serving_endpoints.py
│   ├── compare_audio.py
│   ├── verify_engine.py
│   └── ...
├── repro/                       # 原 tests/repro/
└── data/                        # 原 tests/data/
```

**规则**：
- `tests/` 里只有 `test_*.py`、`conftest.py`、`__init__.py` 和 `support/` 中的共享代码
- 任何 `python some_tool.py` 手动运行的脚本不在 `tests/` 里
- `tools/` 中的验证工具可以 import `tests/support/` 的共享代码

### G5：配置收敛——根目录只留用户必须关心的

**现状**：根目录 13 个配置/构建文件。

**目标**：根目录只保留 **4 个** 用户直接交互的文件：

| 保留 | 原因 |
|------|------|
| `engine.yaml` | 用户必须编辑的引擎配置 |
| `pyproject.toml` | 用户必须知道的依赖管理 |
| `.env.example` | 用户必须关心的环境变量 |
| `README.md` | 用户必须先看的入口 |

**迁移**：

| 文件 | 迁移到 | 原因 |
|------|--------|------|
| `pytest.ini` | 合并入 `pyproject.toml [tool.pytest.ini_options]` | 已有重复定义，统一 |
| `Dockerfile.engine` | `infra/docker/` | 基础设施 |
| `Dockerfile.triton` | `infra/docker/` | 基础设施 |
| `Dockerfile.demo-api` | `infra/docker/` | 基础设施 |
| `compose.yaml` | `infra/docker/` | 基础设施 |
| `compose.dev.yaml` | `infra/docker/` | 基础设施 |
| `.dockerignore` | `infra/docker/` | 基础设施 |

`scripts/bash/model_versions.conf` 和 `ngc_matrix.conf` 暂时保留在 `infra/scripts/bash/`。

### G6：新增 CLAUDE.md

**目标**：创建项目根目录的 `CLAUDE.md`，包含：

1. 项目一句话描述
2. 目录结构意图说明
3. 关键入口命令（setup/build/deploy/test）
4. 代码风格约定
5. 文档语言策略
6. 常见操作速查

### G7：清理 IDE 特有配置和历史残留

| 清理项 | 操作 |
|--------|------|
| `.cursor/plans/` | 删除（3 个历史 plan 文件，不应入库） |
| `.cursor/rules/` | 考虑合并入 `CLAUDE.md` 后删除 |
| `.vscode/` | 保留（开发者普遍使用） |
| `__pycache__` 目录 | `git clean -fdX` 清理，确认 `.gitignore` 覆盖 |
| `pytest.ini` | 合并入 `pyproject.toml` 后删除 |
| `docs/REFACTOR_GOALS_V2.md` | 迁移至 `docs/en/process/` 后删除原位 |

### G8：webui/ 资源清理

| 清理项 | 操作 |
|--------|------|
| `webui/node_modules/` | 确认 `.gitignore` 覆盖，`git clean -fdX` |
| `webui/README.md` 缺失 | 创建，说明前端开发流程 |

---

## 3. 实施顺序

### Phase 1：配置收敛与清理（G5 + G7）

> 风险最低，效果立竿见影——根目录立刻清爽

1. 合并 `pytest.ini` → `pyproject.toml [tool.pytest.ini_options]`
2. 创建 `infra/docker/`，迁移 3 个 Dockerfile + 2 个 compose + `.dockerignore`
3. 更新所有引用这些文件的路径（Docker build context、compose 文件、脚本）
4. 删除 `.cursor/plans/`
5. 清理 `__pycache__`
6. 创建 `CLAUDE.md`（G6）

**验收**：根目录非隐藏文件只有 `README.md`、`CLAUDE.md`、`engine.yaml`、`pyproject.toml`、`.env.example`

### Phase 2：tests/ 职责纯化（G4）

> 概念分离，让 tests/ 回归纯粹

1. 创建顶级 `tools/` 目录
2. 将 `tests/tools/` → `tools/validation/`
3. 将 `tests/repro/` → `tools/repro/`
4. 将 `tests/data/` → `tools/data/`
5. 更新所有 import 路径和脚本引用
6. 更新 `tests/conftest.py` 和 `tests/support/` 中的路径引用
7. 更新 README 中的测试命令

**验收**：`tests/` 只包含 `conftest.py`、`__init__.py`、`unit/`、`integration/`、`e2e/`、`support/`

### Phase 3：目录层级治理（G1）

> 最大的结构变更，需要最仔细的路径更新

1. 创建 `src/` 目录
2. 迁移 `engine/` → `src/engine/`
3. 迁移 `demo_api/` → `src/demo_api/`
4. 迁移 `model_repository/` → `src/model_repository/`
5. 迁移 `client/` → `src/client/`
6. 创建 `infra/` 目录
7. 迁移 `scripts/` → `infra/scripts/`
8. 迁移 `infra/docker/`（Phase 1 已创建）
9. 全量更新 Python import 路径（`engine.xxx` → `src.engine.xxx` 或调整 `sys.path` / `pyproject.toml`）
10. 全量更新脚本中的路径引用
11. 更新所有 Dockerfile 中的 COPY 路径
12. 更新 `engine.yaml` 中的 `model_package_dir` 路径
13. 更新所有文档中的路径引用

**验收**：
- 根目录顶级目录为：`src/`、`infra/`、`tests/`、`tools/`、`docs/`、`resources/`、`third_party/`、`webui/`、`workspace/`
- `python -m engine.server` 仍然可用（或更新为新路径）
- `pytest tests/` 通过
- `qwen3tts` CLI 可用

**⚠️ 这是破坏性最大的变更，建议在独立分支上完成，全量验证后再合并。**

### Phase 4：协议定义单一源（G3）

> 依赖 Phase 3（proto 位置已随 engine 迁移）

1. 创建 `src/_proto/` 目录
2. 将 `src/engine/gateway/tts.proto` 移至 `src/_proto/tts.proto`
3. 生成 `_pb2.py` / `_pb2_grpc.py` 放在 `src/_proto/`
4. 更新 `src/engine/gateway/` 的 import
5. 更新 `src/client/` 的 build 流程（从 `src/_proto/` 拷贝生成代码）
6. 添加 `make proto` / `just proto` 目标
7. 删除冗余的 proto 文件

**验收**：`find . -name 'tts.proto'` 只返回 `src/_proto/tts.proto`

### Phase 5：文档治理（G2）

> 依赖 Phase 3（路径已更新），可部分并行

1. 创建 `docs/en/` 和 `docs/zh/` 目录结构
2. 按分类迁移文档：
   - 中文用户文档 → `docs/zh/`
   - 英文开发者文档 → `docs/en/`，按 `architecture/`、`design/`、`investigation/`、`operations/`、`process/` 分组
3. 精简 `architecture.md`：提取 200 行概览 + 拆分子篇
4. 创建 `docs/zh/README.md` 和 `docs/en/README.md` 索引页
5. 统一所有子目录 README 的语言：
   - `src/client/README.md` → 英文（SDK 是国际化的）
   - `src/demo_api/README.md` → 英文（跟随 developer docs 策略）
   - `infra/scripts/README.md` → 英文
   - `tools/README.md` → 英文
6. 创建 `docs/zh/quickstart.md`（从根 README 提取快速开始部分）
7. 精简根 `README.md` 为概览 + 链接索引（不超过 100 行）

**验收**：
- `docs/` 根目录无 .md 文件
- 每篇文档在正确的语言目录和分类下
- `docs/zh/README.md` 和 `docs/en/README.md` 作为导航入口
- 根 README 简洁，指向 docs/ 索引

### Phase 6：最终验证

1. 全量 pytest
2. `qwen3tts` CLI 全子命令验证
3. Docker 构建验证（`infra/docker/` 路径正确）
4. 链接检查：所有 README 和文档中的相对链接有效
5. Import 路径检查：无断裂的 Python import
6. `git ls-files` 检查：无残留的旧路径文件
7. 删除 `docs/en/process/refactor_goals_v1.md` 和 V2（归档至 git history 即可）
8. 更新 `CLAUDE.md` 反映最终结构

---

## 4. 不变范围

| 项目 | 原因 |
|------|------|
| `webui/` | 前端项目有独立 npm 生态，不迁入 src/ |
| `workspace/` | 运行时产物（gitignored），保持顶级 |
| `third_party/` | git 子模块位置约定 |
| `resources/` | 静态资源，语义独立 |
| `engine/` 内部代码结构 | 仅迁移位置，不重组内部 |
| `client/` 内部代码结构 | 仅迁移位置，不重组内部 |
| `scripts/bash/` 内部逻辑 | 仅迁移位置，继续由 V2 目标精简 |
| Python CLI (`qwen3tts`) | 继续由 V2 目标迭代 |

---

## 5. 预期效果

### 5.1 根目录

| 指标 | 当前 | 目标 | 改善 |
|------|------|------|------|
| 顶级目录数 | 11 | 9 | 语义更清晰 |
| 根目录文件数 | 13 | 4 + README + CLAUDE.md | -69% |
| 用户必须看的根文件 | 13 | 2（README + engine.yaml） | -85% |

### 5.2 文档

| 指标 | 当前 | 目标 | 改善 |
|------|------|------|------|
| 文档语言混用 | 是 | 按目录严格区分 | 消除 |
| docs/ 根目录文件数 | 15 | 0 | 全部分类 |
| 新用户导航入口 | 无 | `docs/zh/README.md` | 新增 |
| architecture.md 行数 | 2333 | ≤200（概览）+ 子篇 | 可导航 |

### 5.3 代码组织

| 指标 | 当前 | 目标 | 改善 |
|------|------|------|------|
| 产品代码位置 | 根目录散落 | `src/` 统一 | 一眼可懂 |
| 基础设施位置 | 根目录散落 | `infra/` 统一 | 一眼可懂 |
| proto 拷贝数 | 3 | 1 | -67% |
| tests/ 中非测试内容 | tools/ + repro/ + data/ | 0 | 概念纯净 |
| 配置文件根目录数 | 13 | 4 | -69% |

### 5.4 新用户体验

| 维度 | 当前 | 目标 |
|------|------|------|
| 克隆后第一眼 | 11 个目录 + 13 个文件 = 看不出重点 | 9 个有明确命名的目录 + 4 个文件 = 清晰 |
| 找文档 | 在 20 个 .md 文件中翻找 | `docs/zh/README.md` 索引导航 |
| 找核心代码 | 不确定 engine/ 和 model_repository/ 的关系 | `src/` 下是所有产品代码 |
| 找基础设施 | Dockerfile 和 scripts 散在根目录 | `infra/` 下是所有基础设施 |
| 找验证工具 | 在 tests/tools/ 里找（和测试混在一起） | `tools/validation/` |
| 语言困惑 | README 中文，子目录英文，混搭 | 中文用户看 `docs/zh/`，贡献者看 `docs/en/` |

---

## 6. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Phase 3 目录迁移破坏 import 路径 | 在独立分支上完成；用 `grep -r` 全量搜索旧路径；CI 全量测试 |
| Docker build context 路径变更 | 迁移后立即本地构建验证 |
| 用户习惯了旧路径 | 保留一段兼容期，旧路径放 deprecation wrapper |
| architecture.md 拆分可能丢失上下文 | 拆分前确保每篇子篇自含必要背景 |
| webui/ 不迁入 src/ 造成不一致 | webui/ 有独立的 npm/node 生态，硬迁入反而破坏标准前端工作流 |
