# CLAUDE.md — Qwen3TTS-Streaming 项目指引

## 项目概述

Qwen3TTS-Streaming 将官方 Qwen3-TTS PyTorch 权重导出为 ONNX/TensorRT 运行时，围绕 Triton Inference Server / standalone engine 实现流式 TTS 推理，包含 prefix cache、连续批处理、前端分词和 WebUI 演示。

## 目录结构

```
Qwen3TTS-Streaming/
├── engine/              # 推理引擎核心（frontend/backend/gateway/core/interface）
├── client/              # 独立 Python SDK 包 (pip install qwen3-tts-client)
│   ├── src/qwen3tts/     # 客户端实现与传输适配器
│   └── src/qwen3tts_protocol/   # 共享协议层（单一真相源）
├── demo_api/            # WebUI Demo API 后端（aiohttp）
├── webui/               # Vite/React 前端
├── model_repository/    # Triton Python BLS 模型定义
├── scripts/
│   ├── bash/            # 生命周期脚本（autorun/setup/build/deploy）
│   │   └── lib/         # 可复用 shell 模块
│   ├── export/          # PyTorch → ONNX/manifest 导出
│   ├── compose/         # 容器入口点脚本
│   ├── demo/            # Demo 启动脚本
│   └── python/          # bash 调用的 Python 辅助（manifest/profile/NGC/triton 配置生成）
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

> **生命周期入口：bash 脚本（`scripts/bash/`）是唯一受支持的真相源。**
> bash 负责编排 docker / trtexec，Python 仅作为被 bash 调用的辅助，处理 JSON / profile /
> 网页解析等 bash 不擅长的部分。（早期的 `qwen3tts` Python CLI 已移除。）

```bash
# Phase B: 编译 TRT 引擎（docker 内 trtexec；支持分子模块混合精度）
bash scripts/bash/build_engines.sh --variant custom-1.7b              # 全 bf16
bash scripts/bash/build_engines.sh --variant custom-1.7b --cp-precision fp32  # 混合：cp 用 fp32

# Phase C: 组装模型包 + 启动 engine-docker 服务
bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt   # 组装 model_repository
bash scripts/bash/compose.sh up --gateway engine --engine-mode trt        # 构建 engine 镜像并启动
bash scripts/bash/compose.sh down --gateway engine                        # 停止

# 全流程 / 分阶段（autorun 子命令，交互模式直接 `bash scripts/bash/autorun.sh`）
bash scripts/bash/autorun.sh all     -m custom-1.7b                          # A→B→package→deploy
bash scripts/bash/autorun.sh setup   -m custom-1.7b                          # Phase A: 下载+导出
bash scripts/bash/autorun.sh build   -m custom-1.7b                          # Phase B: 编译 TRT
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway standalone     # Phase C1: 组装产物
bash scripts/bash/autorun.sh deploy  -m custom-1.7b --gateway standalone     # Phase C2: 启动服务

# 跨机编译
bash scripts/bash/autorun.sh probe-target --out target_profile.json                                  # 采集目标机指纹
bash scripts/bash/autorun.sh make-bundle  -m custom-1.7b --target-profile target_profile.json        # 创建构建包
bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst                # 导入编译产物
bash scripts/bash/autorun.sh remote-build -m custom-1.7b --target-profile target_profile.json --remote-host user@host  # SSH 远程编译

# 状态 / 停止
bash scripts/bash/autorun.sh status
bash scripts/bash/autorun.sh stop

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
| 编译 TRT | `bash scripts/bash/build_engines.sh --variant custom-1.7b` |
| 跨机编译 | `bash scripts/bash/autorun.sh make-bundle -m custom-1.7b --target-profile target_profile.json` |
| 导入编译产物 | `bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst` |
| 采集目标机信息 | `bash scripts/bash/autorun.sh probe-target --out target_profile.json` |

## 重要约束

- `custom-1.7b` / `custom_voice` 是 v0.1 推荐路径，其他变体为实验状态
- 流式模式仍可能出现幻觉/重复/漏读，不用于生产
- `workspace/` 是运行时产物目录，不提交到 git
- `third_party/Qwen3-TTS/` 是 git 子模块，不要修改其中的代码
