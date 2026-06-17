# 跨机 TensorRT Engine 编译

TensorRT `.engine` 绑定目标 GPU 架构、TensorRT 版本和构建 profile。导图机或打包机不应假设自己和生产机一致；推荐流程是：当前机器导出 ONNX，在目标生产同构 GPU 上编译 engine，再把 engine artifact 带回当前机器组装模型包和运行镜像。启动服务是独立步骤，只应在当前机器就是生产服务机或本机验证机时执行。

## 离线 Bundle 流程

1. 在目标生产同构机器上采集指纹：

```bash
bash scripts/bash/probe_target.sh --out target_profile.json
```

2. 把 `target_profile.json` 拷回导图/打包机，生成构建包：

```bash
bash scripts/bash/autorun.sh make-bundle -m custom-1.7b \
  --target-profile target_profile.json \
  --out workspace/engine_build_bundle.tar.zst
```

3. 把 `engine_build_bundle.tar.zst` 拷到目标机器并执行：

```bash
mkdir -p /tmp/qwen3-engine-build
tar --zstd -xf engine_build_bundle.tar.zst -C /tmp/qwen3-engine-build
cd /tmp/qwen3-engine-build
bash build_on_target.sh
```

4. 把 `engine_artifact_bundle.tar.zst` 拷回打包机并导入：

```bash
bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst
```

5. 回到打包机组装部署产物，不启动服务：

```bash
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build
```

`package --gateway engine-docker` 会组装 `workspace/model_repository/tts_orchestrator/<version>`，并用当前 checkout 重建 engine 镜像。镜像 tag 默认从 Phase B manifest 的 NGC tag 推导，例如 `qwen3-engine:25.03`。

6. 只有当前机器就是要提供服务的机器时，才启动服务：

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

## Remote SSH 流程

当打包机可以 SSH 到目标机器时，可以把 bundle 流程自动化：

```bash
bash scripts/bash/autorun.sh remote-build -m custom-1.7b \
  --target-profile target_profile.json \
  --remote-host user@prod-gpu-host \
  --remote-workdir /tmp/qwen3-engine-build
```

该命令会在本机打 bundle、上传到远端、远端执行 `build_on_target.sh`、拉回 artifact 并导入 `workspace/exported/`。

## NGC Tag 的来源

跨机场景下，`target_profile.json` 是 NGC tag 的唯一事实来源：

- `probe_target.sh` 在目标机器根据生产驱动选择 `recommended_ngc_tag`
- `make-bundle` 使用该 tag 写入 `build_manifest.json`
- 目标机器用同一 NGC 镜像编译 engine
- Phase C package/run 从 manifest 推导运行镜像，不再用打包机本机 driver 回退猜测

## 严格指纹校验

导入 artifact 后会写入 `workspace/exported/artifact_manifest.json`。`build_triton.sh assemble/run/build` 在 TRT 模式下会校验：

- `ngc_tag`
- `tensorrt_version` major.minor
- `gpu_sm`
- `engine_dtype`
- `max_batch_size`
- `max_input_len`
- `max_seq_len`

没有 artifact manifest 的旧本地构建会保留兼容，只打印警告。跨机流程导入的 artifact 若不匹配会直接失败；开发调试可显式设置 `ALLOW_FINGERPRINT_MISMATCH=1`。

## 和 Phase C 的关系

`autorun.sh package` 和 `autorun.sh deploy` 的职责不同：

```bash
# 打包机/发布流水线：只产出模型包和运行镜像
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build

# 服务机/本机验证：用已有模型包和运行镜像启动服务
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

本地部署可以直接执行 `autorun.sh all -m custom-1.7b --gateway engine-docker`，它会按 `setup → build → package → deploy` 跑完；跨机部署通常不要在打包机执行最后的 `deploy`。
