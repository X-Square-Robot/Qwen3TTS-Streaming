[English](README.md) | **中文**

# 测试指南

本仓库刻意将测试入口分离开来：

| Location | Purpose | How to run |
| --- | --- | --- |
| `tests/unit/` | 快速 pytest 单元测试。无需外部 serving 进程。 | `pytest tests/unit -q` |
| `tests/integration/` | 针对导出产物、manifest、ONNX/TRT 配置生成以及本地构建输出的 pytest 集成检查。 | `pytest tests/integration -q` |
| `tests/e2e/` | 针对运行中的 standalone 引擎或 Triton 服务的 pytest 端到端检查。当服务不可达时测试会自动跳过。 | `pytest tests/e2e -v -s` |
| `tests/support/` | 供 pytest 套件与手动工具使用的共享支持代码。不作为测试收集。 | 由 tests/tools 导入 |
| `tools/validation/` | 手动验证、基准、音频生成与调查工具。这些不是 pytest 测试。 | `mamba run -n qwen3-tts python tools/validation/<tool>.py --help` |
| `tools/repro/` | 已知底层问题的冻结复现（reproduction）案例。 | 见复现 README |
| `tools/data/` | 供测试与工具使用的小型 fixture。 | 由测试导入 |

关于横跨脚本/测试的思维模型，也请阅读 [scripts/README.zh-CN.md](../scripts/README.zh-CN.md) 与 [工具治理](../docs/dev/operations/tooling_governance.zh-CN.md)。

## 推荐入口

运行常规开发者套件：

```bash
pytest tests/unit tests/integration -q
```

运行 Triton E2E pytest 检查：

```bash
bash scripts/bash/deploy.sh run --gateway triton
pytest tests/e2e/test_e2e.py -v -s
```

运行 standalone 引擎 E2E pytest 检查：

```bash
python -m engine.server --config engine.yaml
pytest tests/e2e/test_engine_standalone.py -v -s
```

在 standalone 引擎与 Triton 端点上运行完整的 serving 验收工具：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py
```

仅运行裸 standalone 引擎的 gRPC 验收路径：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-long --skip-badcase
```

运行带 warmup 的裸引擎 TTFT 分布基准，包含均值、方差、标准差、百分位以及波动条：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-single --skip-streaming --skip-custom-instruct \
  --skip-concurrent --skip-long --skip-badcase \
  --ttft-warmup 3 \
  --ttft-samples 30
```

## 命名规则

- `tests/unit/`、`tests/integration/` 与 `tests/e2e/` 下命名为 `test_*.py` 的文件是 pytest 测试。
- 手动脚本必须放在 `tools/validation/` 下，且不得使用 `test_*.py` 前缀。
- 由 pytest 测试与手动工具共享的可复用辅助应放在 `tests/support/` 下，这样工具无需导入 pytest 文件。
- `scripts/` 可以包含启动器以及构建/部署辅助，但不应包含测试的规范实现。
- 当已存在旧的命令路径时，兼容性包装器（wrapper）可以保留在 `scripts/` 中；该包装器应委托给 `tools/validation/`。

这些规则旨在让测试面对新贡献者一目了然，并对 CI 可预测。

## 治理检查

在添加或重构手动工具时，运行：

```bash
python scripts/python/audit_tooling_surface.py
```

如果某个辅助被多个工具共享，应优先将其提取到 `scripts/python/`（或对仅测试使用的辅助提取到 `tests/support/`），而不是复制粘贴。
