# Qwen3-TTS 内置 Demo、浏览器 SDK 与统一文档建设计划

> 实施状态（2026-08-19）：代码与本地自动验收已完成；真实 release 网关、完整
> 跨浏览器矩阵和物理扬声器晋级门禁仍需在部署环境执行。逐项证据与执行入口见
> [验收矩阵](demo_browser_sdk_docs_acceptance.zh-CN.md)。

> 当前结构（2026-09-04）：原独立顶层 `webui/` 已并入 `web/packages/demo`，运行时只
> 发布一个 `/demo/` 浏览器门户；`/demo/#/lab` 是统一的实验入口。`demo_api/` 仅是
> 可选的后端工程实验 API（详细 decode trace 与能力查询），不再提供第二套前端。
> 下文的“分阶段交付”保留为实施记录，涉及迁移的条目已完成。

## 目标与总体方案

参考 FunASR Nano 的实例门户、SDK 分发和单源文档方案，将 Qwen3TTS-Streaming
建设为“可自描述、可直接试听、可调参数、可下载匹配 SDK”的完整产品入口：

```text
/demo/                  实例内置 Demo、SDK 页面与同版本文档
/demo/config.json       当前实例、协议、构建产物和 SDK 元数据
/sdk/                   pip-compatible Python wheel 索引
/v1/capabilities        协议、模型、任务、输出策略与限制的能力发现
/v1/ws                  原生主协议（Python SDK 使用）
/v1/realtime            OpenAI Realtime 兼容协议（Browser SDK 使用）
```

Browser Demo 首版以 `/v1/realtime` 为体验链路；产品文档同时说明 Python SDK 使用
原生 `/v1/ws`。Demo 默认开启，通过 `DEMO_ENABLED=false`
显式关闭。页面只连接当前实例，不接受跨域 API Key，不把凭据写入浏览器存储。
认证、WebSocket Origin、租户配额和公网限流由同源部署网关统一负责。

稳定文档站发布到 GitLab Pages；实例内文档与 Pages 使用同一份构建产物，正文直接
来自仓库 README 和精选 Markdown，不在前端源码中维护第二份文档。

## 与 FunASR Nano 保持一致的产品风格

两套语音服务采用相同的信息架构和视觉语言：

- 顶部固定品牌栏，统一提供“体验 / SDK / 文档 / 实验”入口和当前版本标识；
- 深色背景、薄荷绿强调色、等宽协议标签、编号面板、1200px 内容宽度；
- 首页 Hero 文案、主操作区、基础参数、诊断结果、代码生成保持相同层级；
- SDK 页均展示精确版本、复制安装命令、下载文件和当前参数示例；
- 文档页均使用构建期 Markdown 渲染，Pages 缺少实例配置时自动进入只读文档模式；
- 移动端保持相同的响应式断点、触控尺寸、错误提示和禁用态语义。

Qwen3TTS 保留现有 React/Vite 实现，不为视觉统一改写为 Vue。现有 Text Player、
Timeline、LLM PK 和 Concurrency 组件继续复用；视觉 token 与页面结构向 FunASR
Demo 对齐。两边样式稳定后，再评估是否提取共享 design-token 包，首版不引入跨仓库
运行时依赖。

品牌文案建议使用：

```text
QWEN3 TTS · STREAMING
让文字，即刻成为声音。
```

## 代码组织

已将原单包 `webui/` 迁入 npm workspace，当前不保留两套前端实现：

```text
web/
├── package.json
├── packages/
│   ├── browser-sdk/       # 无 React 依赖的 TypeScript SDK
│   └── demo/              # React/Vite 产品门户，复用现有 WebUI 组件
├── scripts/               # 版本映射、文档 manifest、产物校验
└── e2e/                   # Playwright

engine/distribution/
├── sdk.py                 # wheel 发现、索引、下载与响应合同
└── site.py                # Demo 静态资源、config.json 与安全响应头

protocol/
├── tts.proto              # gRPC 合同唯一源
└── contracts/             # Realtime JSON Schema 与跨 SDK 合同向量
```

依赖保持单向：Demo 依赖 Browser SDK；Browser SDK 只依赖协议合同；运行时静态分发
只读取已构建产物，不依赖 React、Node 或 demo_api。基本 LLM PK/并发实验通过门户的
公共 `/v1/realtime` 执行；`demo_api/` 作为可选后端提供详细 decode trace 与能力查询，
不进入基本合成链路。

## 核心实现

### 1. 修复并统一 SDK 下载路径

- `/sdk` 返回相对 `Location: ./sdk/` 的 308，保证 `/infer/<instance>/sdk` 等代理
  前缀不丢失。
- `/sdk/` 只生成相对 wheel 文件名，不生成 `/sdk/...` 根路径绝对链接。
- wheel 响应提供安全文件名、`Content-Disposition: attachment`、正确 MIME、长度和
  SHA256 元数据。
- 同一个 `engine.distribution.sdk` 组件挂载到 standalone 主服务端口、Triton
  Realtime sidecar 和独立 health 端口；health 端口仅为向后兼容，文档默认展示公共
  服务地址。
- `Dockerfile.engine` 与 `Dockerfile.triton` 都嵌入当前 release 已发布且经过 SHA256
  校验的同一个 Python wheel。运行时不重新构建 wheel。
- `/sdk/` 保持 pip `--find-links` 可消费的简洁索引；人工安装命令、版本说明和下载
  按钮放在 Demo 的 SDK 页。
- 正式镜像必须恰好包含一个匹配 wheel；源码开发环境没有 wheel 时，
  `/demo/config.json` 返回 `available=false` 和明确原因，页面禁用按钮而不是生成坏链。
- 保留 GET/HEAD、查询参数、路径穿越防护和不存在文件的 404 语义。

`/demo/config.json` 中的 Python SDK 元数据使用结构化对象，而不是让页面解析目录：

```json
{
  "python_sdk": {
    "available": true,
    "project": "qwen3-tts-client",
    "version": "0.2.0",
    "filename": "qwen3_tts_client-0.2.0-py3-none-any.whl",
    "sha256": "...",
    "index_url": "../sdk/",
    "download_url": "../sdk/qwen3_tts_client-0.2.0-py3-none-any.whl"
  }
}
```

所有 URL 都以 `config.json` 的真实响应 URL 为基准解析，禁止使用
`window.location.origin + "/sdk"` 一类会丢失实例前缀的拼接。

### 2. 建立正式浏览器 SDK

新增 TypeScript 包 `@xmultimodalinteraction/qwen3tts-browser`，发布到 GitLab npm
Package Registry。SDK 与 UI 解耦，任何 React/Vue/原生网页都能调用。

公开接口包括：

- `RealtimeTTSClient`：能力发现、连接、完整文本合成、增量 append/commit、取消、
  串行复用和活动 response 恢复；
- `SynthesisTask`、`InputMode`、`AudioEncoding`、`VadStrategy`、
  `DeliveryPolicy` 等枚举，不以任意字符串作为公开动作接口；
- `SynthesisOptions`：task、speaker、language、instruct、reference、audio、VAD、
  output policy 和 timing；
- `TTSEvent` discriminated union：connected、response_started、audio、progress、
  warning、completed、cancelled、reconnecting、error；
- `BrowserAudioPlayer`：PCM16/PCM float 解码、采样率转换、AudioWorklet 有界队列、
  播放/暂停/清空、音量和可选输出设备；
- `WavCollector`：在独立可配置上限内收集合成音频并导出 WAV，达到上限只停止收集，
  不影响实时播放；
- 原始事件订阅与诊断快照，便于用户定位 underrun、buffer lead、TTFT、首响和恢复。

协议约束：

- 产品页明确区分原生 `/v1/ws` 主协议与 OpenAI Realtime `/v1/realtime` 兼容协议；
  Browser SDK 当前使用兼容入口，Python SDK 默认使用原生入口。
- `session.update`、`conversation.item.create`、`response.create` 和
  `qwen.input_text_buffer.append/commit` 使用现有服务合同，不另造动作协议。
- 使用版本化 JSON Schema 和 Zod 校验 Realtime 业务字段；与 Python SDK 共用
  `protocol/contracts` 下的 golden vectors，保证事件、错误和配置不会漂移。
- 连接前读取 `/v1/capabilities`，对协议版本、扩展、task、音频格式、VAD 和 reference
  能力 fail-closed；未声明的能力不得仅凭 UI 猜测发送。
- 播放器按实际消费的 sample cursor 发送 `qwen.playback.ack`；服务端 guarded delivery
  与浏览器真实播放头使用同一坐标，禁止按“已接收字节”冒充“已播放”。
- 支持 `qwen.response_resume.v1` 时，从最后完整 delivery/sample 恢复同一次执行，不能
  重新合成后猜测音频去重。
- AudioWorklet 运行在浏览器实际 `AudioContext.sampleRate`；服务输出为 16/24 kHz 时
  显式连续重采样，不能假设浏览器会接受请求的 context sample rate。
- 播放队列、事件历史和 WAV 收集全部有界；发生 underrun、恢复失败或内存上限时返回
  明确诊断，不允许无界缓存。
- npm 版本从 PEP 440 tag 映射：`.devN -> -dev.N`、`aN -> -alpha.N`、
  `bN -> -beta.N`、`rcN -> -rc.N`，稳定版保持 `X.Y.Z`，分别发布到对应 dist-tag。

### 3. 建设产品 Demo 页面

Demo 沿用 React、Vite、TypeScript；使用 hash history 和相对 asset base，使
`/infer/<instance>/demo/` 等任意代理前缀无需额外构建即可工作。

#### “体验”页

- 文本输入、task、speaker、language 和当前模型状态；
- “完整文本”与“模拟 LLM 增量输入”两种输入方式；
- 合成、暂停、继续、取消、重放和下载 WAV；
- 实时 waveform、播放缓冲、Text Player、文本进度和音频时间轴；
- 输出设备选择仅在浏览器能力可用时展示，默认使用系统扬声器；
- custom voice、voice design、x-vector clone、ICL clone 严格按 capabilities 展示；
  实验能力必须带状态与已知限制，不能伪装成稳定能力；
- reference audio 上传、录制和 ref text 只在后端真实可用时出现，并遵守实例公布的
  时长、格式和大小上限。

#### “参数”区

- 基础：task、speaker、language、sample rate、audio encoding、input mode；
- 输出 VAD：disabled / energy / tenvad，以及 begin/end threshold、count、chunk、
  start margin；明确说明这是 TTS 输出过滤，不是麦克风 endpointing；
- Output Policy：guarded delivery、delivery window、chunk size、text progress events；
- 高级参数由 capabilities 和版本化合同门控。当前尚未形成端到端逐请求合同的 sampling
  参数不进入首版 UI，先完成协议、SDK、引擎和 Triton parity 后再开放；
- tenvad 只有在运行时依赖与许可条件均满足、服务明确声明时才显示。

#### “结果与诊断”区

- client TTFB、server TTFT、first audible、total latency、audio duration、RTF；
- 当前 buffer lead、underrun 次数、播放 sample cursor、response usage；
- text progress、segment、warning、guarded delivery、VAD trimming 和恢复事件；
- 所有指标标明测量边界，页面不再显示硬编码 headline 性能数字；
- 音频或 live backend 不可用时 fail closed，不用 fixture 或模拟数据冒充真实结果。

#### “SDK”页

- 从 `config.json` 读取精确 wheel、版本和 SHA256，生成带实例前缀的 pip 命令；
- 提供 wheel 下载、命令复制、浏览器 SDK 包名/npm 版本和 tarball 信息；
- 根据当前页面参数生成 Python SDK 与 Browser SDK 示例；
- 生成的示例默认使用当前实例相对公共入口，不暴露内部 Triton gRPC 地址；
- 展示 engine、Python SDK、Browser SDK、协议和文档是否来自同一 release tag。

#### “实验”页

- 统一从 `/demo/#/lab` 提供基础 LLM PK/Concurrency，并在可选后端可达时提供已迁移的
  实时 TRT/Text Player、服务端 PK、多路并发和 JSON/WAV 下载能力；
- 明确标记为工程实验区，与普通试听入口隔离；
- 基础 LLM PK/Concurrency 走公共 `/v1/realtime`；详细 trace 工具按 `demo_api` 能力
  与 `lab.available` 门控，不让基本 Demo 依赖 demo_api；
- fixture 只能作为离线 trace 查看，必须明显标记，不参与实时体验和公开性能口径。

### 4. 服务端静态站点与能力元数据

- 新增固定 `/demo` 路由；未启用时 `/demo`、`/demo/`、assets 和 config 全部返回 404。
- `/demo` 使用相对 308 进入 `./demo/`；HTML/config 禁止长期缓存，带 hash 的 assets
  使用 immutable cache。
- Demo 只挂在 standalone 主服务端口和 Triton Realtime sidecar，不挂载到独立 health
  端口；`/sdk` 为兼容性可同时挂载到 health 端口。
- standalone 与 Triton sidecar 复用同一个 `engine.distribution.site`，避免两种部署
  出现不同的静态路由和安全头。
- `GET /demo/config.json` 返回：engine 版本、runtime 类型、两个相对 WebSocket 路径、
  capabilities 相对路径、Python wheel 元数据、Browser SDK 包名/npm 版本、文档构建
  版本和可选 lab 地址。
- `/v1/capabilities` 统一 standalone 与 Triton sidecar 的字段，补充 loaded model、
  task 状态、speaker/language、音频格式、reference 支持与上限、输出策略、VAD、
  progress/recovery 扩展和输入限制；旧客户端继续忽略未知字段。
- task、VAD 和格式使用枚举集合；复杂参数的类型与边界引用版本化 JSON Schema，避免
  capabilities 内出现任意字符串配置协议。
- 静态响应增加 CSP、`X-Content-Type-Options`、`Referrer-Policy`、
  `Permissions-Policy` 和 `frame-ancestors 'none'`。首版不主动申请麦克风；reference
  recording 启用后仅允许 `microphone=(self)`。
- 应用不实现第二套登录系统。部署文档要求网关同时保护 `/demo`、`/sdk` 和 `/v1/*`，
  保留路径前缀并实施 Origin 校验、租户配额、文本/reference 大小上限和 WebSocket
  超时。

## 单源文档与发布

精选渲染源首版固定为：

- `README.zh-CN.md` / `README.md`；
- `client/README.zh-CN.md` / `client/README.md`；
- `docs/user/deployment.zh-CN.md` / `deployment.md`；
- `docs/user/known_limitations.zh-CN.md` / `known_limitations.md`；
- `docs/dev/architecture/openai_realtime.zh-CN.md` / `openai_realtime.md`；
- `docs/user/benchmark_methodology.zh-CN.md` / `benchmark_methodology.md`。

构建规则：

- 使用 Markdown-it 在构建期渲染，禁止前端源码复制 Markdown 正文；
- 按 manifest 生成中英文导航、标题锚点和页面 slug；
- 精选文档之间的相对链接重写为站内文档路由；未收录源码/文档链接重写到当前 tag
  的 GitHub/GitLab source base；
- 图片、GIF 和演示视频作为站点 asset 打包，并检查大小、MIME 和坏链；
- README 中不再硬编码某个已过期 wheel 文件名。实例 SDK 页根据 config 动态显示精确
  安装命令，静态文档只说明通用安装渠道；
- 根 README 继续作为代码托管首页内容源，对外“完整文档”链接统一指向 Pages 的
  `/demo/#/docs/...`，不再维护第二套站点正文；
- Pages 没有 `/demo/config.json` 时进入 docs-only 模式，禁用体验、下载和实例代码
  生成；交互体验通过用户自己的实例 `/demo/` 完成。

同一 tag 的发布产物遵守“只构建一次、后续复用”：

- CI 使用固定 Node 22 builder，依次执行 typecheck、lint、Vitest、站点构建、
  Playwright 和 `npm pack`；运行时镜像不包含 Node/npm；
- 在同一 forge pipeline 中只构建一次 Browser SDK tarball 和站点压缩包；GitLab npm
  发布、Pages、Release 与镜像 job 复用同一 artifact，并验证 SHA256；
- GitLab 发布 npm 包和 Pages；GitHub Release 附加 Browser SDK tarball 与站点归档，
  不向 npmjs 发布；
- 两个正式 runtime 镜像分别嵌入对应 forge 已验证的 Python wheel 和同一静态站点
  artifact，不在 Docker build 内重新执行前端构建；
- 本地 build/compose 使用相同 Node builder stage 产生开发版本 artifact，并将其放入
  gitignored staging 目录。

## 测试与验收

### 自动测试

- SDK 路径：standalone、Triton sidecar、health 端口以及 `/infer/<id>` 前缀下的
  `/sdk`、`/sdk/`、相对下载、GET/HEAD、查询参数、Content-Disposition、SHA256 和
  路径穿越；
- Browser SDK：capabilities fail-closed、Realtime 状态机、完整/增量文本、串行响应、
  取消、错误、usage、text progress、playback ACK、断线恢复和 int64/sample cursor；
- 音频：PCM16/float、多采样率、连续重采样、分块边界、AudioWorklet 队列、暂停恢复、
  underrun、输出设备降级和 WAV 上限；
- UI：task 能力门控、speaker/language、VAD 联动、实验能力警告、播放控制、诊断、
  SDK 命令与代码生成；
- 文档：所有精选 Markdown、站内相对链接、标题锚点、图片、视频和代码块构建检查；
- Playwright：Chromium、Firefox、WebKit 及移动设备模拟，使用固定 mock Realtime
  server 验证从点击合成到 AudioWorklet 消费样本的完整链路；
- 容器 smoke：两种 runtime 下 Demo 关闭为 404，开启后页面/assets/config/SDK 可用，
  并完成一次 `/v1/realtime` mock 或 live 合成；
- 合同 parity：同一 golden request 经过 Python SDK 和 Browser SDK 后，核心
  `session.update`/response 事件与音频 sample 边界一致。

### 发布验收

- 在带真实实例路径前缀的网关中打开 `/demo/`，点击 Python wheel 能成功下载并安装；
- 页面显示的 engine、Python wheel、Browser SDK、协议和文档版本来自同一 release；
- Chrome、Edge、Firefox、Safari 当前及前两个大版本能完成合成、实时播放、取消、
  重放和 WAV 下载；
- 桌面与 Android/iOS 浏览器能通过系统默认扬声器播放；支持输出设备选择的浏览器可
  切换设备，不支持时不显示无效控件；
- custom voice 默认路径可用；voice design/clone 只在对应模型和 reference 能力真实
  可用时出现，并携带工程预览警告；
- VAD 调节改变的是服务端实际 output policy，页面诊断能显示裁剪结果；
- Product Demo 不经过 demo_api 或直接 Triton gRPC，确实验证公共 `/v1/realtime`；
- beta 在桌面、Android 和 iOS 各执行一次人工扬声器 smoke，确认首响、连续播放、取消
  和长文本无明显缓冲失控后方可晋级 main。

## 分阶段交付

1. **SDK 相对路径修复**：抽取共享分发组件，修复代理前缀，并让 standalone、Triton
   sidecar 与 health 端口合同一致；可独立优先发布。
2. **能力合同统一**：补齐两种 runtime 的 `/v1/capabilities` parity、Demo config、
   JSON Schema 和跨 SDK golden vectors。
3. **npm workspace 与 Browser SDK**（已完成）：迁移现有 React WebUI，完成公开类型、Realtime
   客户端、版本映射和 npm pack。
4. **浏览器播放链路**：AudioWorklet、连续重采样、sample cursor、playback ACK、
   guarded delivery、恢复和 WAV 收集。
5. **产品 Demo**（已完成）：体验、参数、诊断、SDK、代码生成和 FunASR 同风格响应式 UI；
   LLM PK/Concurrency 与 Text Player trace 统一进入 `/demo/#/lab`，详细 trace 后端保持可选。
6. **Markdown 单源文档**：精选 manifest、站内链接、资产打包、docs-only 模式和
   GitLab Pages。
7. **CI 与镜像集成**：一次构建多处复用、GitLab npm、GitHub assets、两种 runtime
   镜像与 SHA256 校验。
8. **跨浏览器与真实网关验收**：路径前缀、桌面/移动端扬声器、长文本、取消/恢复和
   beta 人工 smoke。

预计熟悉仓库的单人完整实施约 4–6 周。第一阶段 SDK 下载路径修复约 1–2 天，可在
不等待 Browser SDK 和新 UI 的情况下先行发布。

## 明确不做

- 不把当前 React WebUI 重写为 Vue；
- 不让产品 Demo 继续通过 `demo_api -> Triton gRPC` 合成；
- 不在网页中输入或持久化长期 API Key；
- 不把原生 WebSocket 主协议与 Realtime 兼容协议包装成看似等价的用户选择；
- 不展示服务没有通过 capabilities 明确声明的 task、VAD 或 sampling 参数；
- 不复制 README/用户文档正文到前端组件；
- 不在运行时镜像安装 Node/npm，也不在多个发布 job 重复构建前端产物；
- 不用 fixture、嘟声或历史 benchmark 数字冒充当前实例的实时结果。
