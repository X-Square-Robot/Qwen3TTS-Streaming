# CLAUDE.md — Qwen3-TTS-Triton 项目指引

## 项目概述

Qwen3-TTS-Triton 将官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，围绕 Triton Inference Server / standalone engine 实现流式 TTS 推理，包含 prefix cache、连续批处理、前端分词和 WebUI 演示。

## 目录结构

```
Qwen3-TTS-Triton/
├── engine/              # 推理引擎核心（frontend/backend/gateway/core/interface）
├── client/              # 独立 Python SDK 包 (pip install qwen3-tts-client)
│   ├── src/qwen3_tts_client/     # 客户端实现与传输适配器
│   └── src/qwen3_tts_protocol/   # 共享协议层（单一真相源）
├── demo_api/            # WebUI Demo API 后端（aiohttp）
├── webui/               # Vite/React 前端
├── model_repository/    # Triton Python BLS 模型定义
├── scripts/
│   ├── bash/            # 生命周期脚本（autorun/setup/build/deploy）
│   │   └── lib/         # 可复用 shell 模块
│   ├── export/          # PyTorch → ONNX/manifest 导出
│   ├── compose/         # 容器入口点脚本
│   ├── demo/            # Demo 启动脚本
│   └── python/          # Python CLI + 工具库
│       └── qwen3tts_tools/  # 共享 Python 工具函数
├── tests/
│   ├── unit/            # pytest 单元测试
│   ├── integration/     # pytest 集成测试
│   ├── e2e/             # pytest 端到端测试
│   └── support/         # 测试共享代码
├── tools/
│   ├── validation/      # 手动验证与 benchmark 工具（非 pytest）
│   ├── repro/           # 冻结的 bug 复现案例
│   └── data/            # 工具数据
├── docs/
│   ├── user/              # 用户文档（部署、SDK、Benchmark、限制）
│   ├── dev/               # 开发者文档（架构、设计、调查、运维）
│   │   ├── architecture/  #   架构子篇
│   │   ├── design/        #   设计目标
│   │   ├── investigation/ #   调查报告
│   │   └── operations/    #   运维文档
│   └── process/           # 流程/历史文档
├── infra/
│   └── docker/          # Dockerfile + compose 配置
├── proto/               # 协议定义单一源（tts.proto + 生成代码）
├── resources/           # 静态资源（参考音频等）
├── third_party/         # git 子模块（Qwen3-TTS 上游）
└── workspace/           # 运行时产物（gitignored）
```

## 关键入口命令

```bash
# 交互式全流程
bash scripts/bash/autorun.sh

# 分阶段执行
bash scripts/bash/autorun.sh setup   -m custom-1.7b          # Phase A: 下载+导出
bash scripts/bash/autorun.sh build   -m custom-1.7b          # Phase B: 编译 TRT
bash scripts/bash/autorun.sh deploy  -m custom-1.7b --gateway standalone --engine-mode trt  # Phase C

# Python CLI（替代 autorun.sh 的现代入口）
qwen3tts all -m custom-1.7b
qwen3tts setup -m custom-1.7b
qwen3tts build -m custom-1.7b
qwen3tts deploy -m custom-1.7b --gateway standalone

# 测试
pytest tests/unit tests/integration -q
python tools/validation/serving_endpoints.py --targets engine-grpc

# 引擎直接运行
python -m engine.server --config engine.yaml
```

## 代码风格约定

- Python：类型注解、docstring、PEP 8
- Bash：shellcheck 兼容，lib/ 模块用 `snake_case` 函数名
- Protobuf：`proto/tts.proto` 是协议定义唯一源，生成代码不要手动编辑；`make proto` 重新生成，`make proto-sync` 同步到 consumer
- 配置：`engine.yaml` 是引擎运行时配置，`pyproject.toml` 是依赖管理

## 文档语言策略

- 所有文档统一使用中文
- 代码块、变量名、文件路径、命令示例保持英文
- `docs/user/`：面向用户的文档（部署、使用、限制）
- `docs/dev/`：面向开发者/贡献者的文档（架构、设计、调查、运维）
- `docs/process/`：流程/历史文档（归档）

## 常见操作速查

| 操作 | 命令 |
|------|------|
| 安装依赖 | `pip install -e ".[dev]"` |
| 运行单元测试 | `pytest tests/unit -q` |
| 运行集成测试 | `pytest tests/integration -q` |
| Serving 验收 | `python tools/validation/serving_endpoints.py --targets engine-grpc` |
| 启动 WebUI Demo | `bash scripts/demo/start_webui_demo.sh --variant custom-1.7b` |
| Docker Compose | `bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b` |
| 导出 ONNX | `bash scripts/bash/autorun.sh setup -m custom-1.7b` |
| 编译 TRT | `bash scripts/bash/autorun.sh build -m custom-1.7b` |

## 重要约束

- `custom-1.7b` / `custom_voice` 是 v0.1 推荐路径，其他变体为实验状态
- 流式模式仍可能出现幻觉/重复/漏读，不用于生产
- `workspace/` 是运行时产物目录，不提交到 git
- `third_party/Qwen3-TTS/` 是 git 子模块，不要修改其中的代码
