# Demo、Browser SDK 与统一文档验收矩阵

本文是
[建设计划](demo_browser_sdk_docs_plan.zh-CN.md) 的验收记录。它区分“代码和可重复自动
测试已通过”与“必须在真实发布环境人工执行”，避免把 mock、源码构建或历史数据写成
线上验收结论。

## 已完成的自动验收

| 范围 | 证据 | 当前结果 |
|---|---|---|
| Python 引擎、两种网关、分发组件 | `pytest -q tests/unit -m 'not gpu and not docker'` | 607 passed，21 个 GPU/Docker 用例按标记跳过 |
| Python SDK 与共享 Realtime 合同 | `pytest -q client/tests` | 219 passed |
| Browser SDK、音频、Demo 与文档构建 | `cd web && npm run lint && npm run typecheck && npm test && npm run build` | lint/typecheck/build 通过；9 个版本测试、22 个 Browser SDK 测试、1 个录音测试通过 |
| 产品入口浏览器链路 | `cd web && npm run test:e2e -- --project=chromium --project=mobile-chromium` | 8 passed；覆盖前缀、能力门控、合成到 AudioWorklet、WAV、SDK 代码生成、docs-only 和 Lab |
| 协议生成一致性 | `make proto-check` | 通过 |
| Shell、格式与补丁健全性 | `bash -n ...`、`ruff check`、`ruff format --check`、`git diff --check` | 本次改动范围通过 |

自动测试进一步覆盖：

- `/sdk` 相对 308、pip-compatible 索引、GET/HEAD、查询参数、路径穿越、下载文件名、
  MIME、长度与 SHA256；
- standalone、Triton sidecar 与兼容 health 端口的 SDK 合同，以及任意实例路径前缀；
- capabilities 与 Demo config JSON Schema、Python/Browser golden vectors、Realtime 业务
  Zod 校验；
- 完整/增量文本、取消、串行响应、usage、播放 ACK、同一 response 恢复、有界事件、
  WAV 和播放队列；
- PCM16/float、16/24 kHz 到浏览器实际采样率的连续重采样、sample cursor、underrun
  诊断和输出设备能力降级；
- Demo 关闭 404、开启后的页面/assets/config/SDK，以及 standalone/Triton 容器内 mock
  Realtime smoke。容器 smoke 是 GitLab/GitHub 正式镜像发布前的强制 job。

## 发布流水线门禁

- GitLab 和 GitHub 都使用 Node 22 只构建一次站点与 Browser SDK，后续 npm/Pages、
  Release 和两种运行时镜像复用同一 artifact 并检查 SHA256。
- 两种镜像只复制预构建站点、其中内置且经过 SHA256 校验的 Browser SDK tarball，
  以及经过发布端校验的唯一 Python wheel；镜像内不含 Node/npm，也不重新构建 SDK。
- 本地 Compose 使用 `infra/docker/Dockerfile.web-builder` 的 Node 22 BuildKit stage，
  将开发版 Demo 与 Browser SDK 写入 gitignored staging 目录后再构建运行镜像。
- GitLab Pages 的 `documentation` environment 指向
  `$CI_PAGES_URL/demo/#/docs/overview-zh`；Pages 根路径跳转到同一文档入口。

## 必须在真实环境完成的晋级门禁

以下项目不能由当前源码工作区或 mock server 代替，未执行前不得声称 beta/main 发布
验收完成：

1. 在真实 `/infer/<instance>` 前缀网关访问 `/demo/`，下载页面列出的 wheel，在全新
   虚拟环境安装并连接同一实例。
2. 核对页面展示的 engine、Python wheel、Browser SDK、协议与文档 release 全部来自
   同一 tag；记录 wheel、Browser tarball 和站点归档 SHA256。
3. 在 Chrome、Edge、Firefox、Safari 当前及前两个大版本执行合成、播放、取消、重放、
   WAV 下载和断线恢复；CI 的 Chromium/Firefox/WebKit 不能替代 Edge/Safari 真机。
4. 在桌面、Android 和 iOS 各进行一次物理扬声器 smoke；支持 `setSinkId` 的浏览器还要
   验证输出设备切换，不支持的浏览器确认不显示该控件。
5. 分别部署 custom voice、voice design 和 clone 能力实例，确认页面只展示
   capabilities 明确声明且运行组件实际可用的任务、speaker、language 与 reference
   控件。
6. 在真实输出上调节 energy/tenvad 参数，核对发送的 output policy 与服务端裁剪诊断；
   对长文本观察 buffer lead、underrun、播放 cursor 和恢复事件。
7. 确认产品体验的网络流量只经过公共 `/v1/realtime`，不经过 `demo_api` 或浏览器直连
   Triton gRPC；`demo_api` 不可达时基本体验仍可用且实验入口隐藏。

执行时应保存：实例 URL（可脱敏）、release tag、浏览器/系统/设备版本、能力 JSON、
下载制品 SHA256、每项通过/失败、失败 trace 与音频问题的人工描述。真实 GitLab Pages
基础 URL 由部署后的 `$CI_PAGES_URL` 决定；在该环境可访问之前，不应在根 README
猜测或硬编码域名。
