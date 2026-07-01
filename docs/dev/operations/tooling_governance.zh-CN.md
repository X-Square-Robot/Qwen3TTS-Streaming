[English](tooling_governance.md) | **中文**

# 工具治理

本项目现已足够庞大，`scripts/` 和 `tests/` 需要有自己的架构，而不仅仅是临时添加。本文档的目标是让新贡献者能够理解工具层面，并为开源发布工作保持可持续性。

## 目标

- 保持默认贡献者路径清晰可见。
- 区分自动化测试、手动工具和一次性调查。
- 优先使用共享辅助层而非复制粘贴工具函数。
- 尽早暴露重复行为，避免每个脚本都自己实现一套。

## 心智模型

### 1. 产品代码

- `engine/`：运行时代码，我们希望社区将其作为产品行为来依赖和审查。

### 2. 生命周期脚本

- `scripts/bash/`：setup、build、package、deploy、探测和环境编排。
- `scripts/bash/lib/`：可复用的 shell 原语。新的 shell 逻辑应先放在这里，避免在各个入口点重复。

### 3. 导出实现

- `scripts/export/`：模型导出内部实现。这些文件是实现模块，不是新人应该直接运行的东西，除非他们在做导出内部相关工作。

### 4. 共享工具库

- `scripts/python/`：被 bash 调用的小型、依赖轻量的 Python 辅助工具（JSON/profile/网页解析等）。
- 这是可复用的仓库路径、端点、目标列表和 WAV 辅助逻辑的首选位置。
- 保持这一层小型和通用。它应该支持工具，而不是成为第二个应用运行时。

### 5. Python 维护 CLI

- `scripts/python/`：轻量 CLI、封装器、分析脚本和维护者工具。
- 如果这里的某个文件成为用户的规范验证工作流，通常应该移到 `tools/validation/`，并可选地留下一个兼容性封装器。

### 6. 自动化测试

- `tests/unit/`：快速逻辑测试。
- `tests/integration/`：导出产物和打包检查。
- `tests/e2e/`：针对运行系统的自动化服务级检查。

### 6a. 共享测试支持

- `tests/support/`：测试套件和手动工具的共享支持代码。
- 当 pytest 文件和 `tools/validation/` 都需要时，将可复用的独立引擎辅助工具放在这里。
- 当支持模块可以拥有某个行为时，`tools/validation/` 不应直接导入 `tests/e2e/test_*.py`。

### 7. 手动验证工具

- `tools/validation/`：验收、基准测试、音频检查和调试工具，这些故意不是 pytest 测试。
- 这是我们期望开源用户直接运行的手动验证入口的规范位置。

### 8. 冻结复现

- `tools/repro/`：历史 bug 复现，应与正常验证层面保持隔离。

## 放置规则

添加新工具时使用以下决策规则：

1. 如果行为应该在 CI 中运行并自动断言正确性，添加 pytest 测试。
2. 如果行为是探索性的、面向基准测试的，或依赖于人工听音/检查，放在 `tools/validation/`。
3. 如果行为支持 setup/export/build/deploy 而非验证，放在 `scripts/`。
4. 如果两个或更多文件需要相同的 Python 工具，提取到 `scripts/python/`（测试专用的提取到 `tests/support/`）。
5. 如果两个或更多 shell 入口点需要相同逻辑，提取到 `scripts/bash/lib/`。

## 命名规则

- `tests/unit/`、`tests/integration/` 和 `tests/e2e/` 使用 `test_*.py` 用于 pytest 发现。
- `tools/validation/` 不得使用 `test_*.py` 前缀。
- `scripts/python/` 中的调查脚本应使用描述性前缀，如 `analyze_`、`compare_`、`probe_`、`replay_` 或 `verify_`。
- 兼容性封装器应在模块文档字符串中说明，并委托给 `tools/validation/`。

## 轻量入口点模式

本仓库中一个健康的 Python CLI 通常应该长这样：

1. 最小化 import-path 引导
2. 参数解析
3. 调用共享辅助/模块代码
4. 渲染摘要 / 产物

它不应该反复做：

- 重新定义仓库根目录常量
- 重新定义默认 serving 端点
- 重新定义 WAV 序列化辅助工具
- 重复相同的 CSV 目标解析

## 发布检查清单

在开源发布或大型工具合并之前：

```bash
python scripts/python/audit_tooling_surface.py
pytest tests/unit/test_tooling_helpers.py -q
```

使用审计报告检查：

- 总脚本/测试层面大小
- 独立 Python 入口点数量
- 值得提取的重复符号
- 剩余的 `sys.path` 引导
- 意外的工具导入 pytest 模块
- `scripts/` 或 `tests/` 下意外的缓存目录

## 近期迁移队列

当前首批治理工作专注于低风险的共享需求：

- serving 端点默认值
- 仓库/workspace 路径辅助工具
- 目标解析辅助工具
- WAV 写入辅助工具
- 可重复的审计报告

之后，下一个有价值的提取可能是：

- 变体/模型路径解析
- 重复的 serving 客户端工具
- 重复的 TRT/ONNX manifest 加载辅助工具
- 重复的实验输出/报告格式化

这种分阶段方法避免了破坏性的大规模重写，同时让项目对新贡献者来说逐步更容易导航。
