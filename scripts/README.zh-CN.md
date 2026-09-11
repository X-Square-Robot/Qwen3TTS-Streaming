[English](README.md) | **中文**

# 脚本指南

## 从哪里开始

- **`scripts/bash/autorun.sh`**：主要的操作者入口（setup → build → package → deploy）
- **`scripts/bash/prepare_release_checkout.sh`**：解析发布候选 commit，并创建仅存在于 CI 工作区的版本 tag
- **`scripts/bash/promote_container_image.sh`**：晋级已验证候选镜像，禁止覆盖不同的正式 digest
- **`scripts/python/audit_tooling_surface.py`**：针对 scripts/ + tests/ 面的治理报告
- **`tools/validation/serving_endpoints.py`**：规范的 serving 验收与基准工具
- **`tests/README.zh-CN.md`**：pytest 套件与手动验证工具的主索引

## 目录职责

| Location | Role | Audience |
| --- | --- | --- |
| `scripts/bash/` | 生命周期与操作者工作流 | 部署或打包本项目的用户 |
| `scripts/bash/lib/` | 共享的 shell 库代码 | 扩展 shell 流程的维护者 |
| `scripts/export/` | 模型导出实现（01–09） | 从事 ONNX / TRT 导出的维护者 |
| `scripts/python/` | 被 bash 调用的 Python 辅助（JSON/profile/NGC/Triton-config） | 维护者 |
| `scripts/python/` | 构建/配置/manifest 工具（9 个脚本） | 高级开发者与维护者 |
| `scripts/compose/` | 容器入口点 | 部署维护者 |
| `scripts/demo/` | 演示启动器 | 演示用户 |

runtime 自带统一的 `/demo/#/lab` 入口，Lab 通过公共 Realtime 工作，不再启动独立的
实验后端或第二套前端。

## scripts/python/ 清单

| Script | Purpose |
|--------|---------|
| `audit_tooling_surface.py` | scripts/ + tests/ 面的治理审计 |
| `build_talker_code2wav_fused_trt_host.py` | 在主机上构建 fused TRT 引擎 |
| `codec_embedding_sum.py` | Codec embedding 求和工具 |
| `generate_triton_configs.py` | 生成 Triton 模型仓库配置 |
| `triton_manifest_io.py` | 读/写 triton_manifest.json |
| `trt_fused_io_formats.py` | Fused TRT 引擎 I/O 格式工具 |
| `trt_fused_talk_c2w_profiles.py` | Fused Talker + Code2Wav TRT profiles |
| `update_triton_manifest_profile.py` | 用引擎 profile 更新 manifest |

## 治理规则

- 新的面向用户的验证或基准工具应放在 `tools/validation/`，而非 `scripts/python/`。
- 共享的类型与 schema 属于 `qwen3tts_protocol`；仓库内部辅助属于 `scripts/python/`。
- 保持 `scripts/python/` 的入口精简——解析参数、调用共享代码、渲染结果。
- 在添加新工具之前，先检查是否已有统一工具（如 `compare_audio.py`、`benchmark.py`）覆盖了该用例。

## 审计工作流

```bash
python scripts/python/audit_tooling_surface.py
```
