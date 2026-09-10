**中文** | [English](README.md)

# Qwen3-TTS 内置 Demo

本包是 Qwen3TTS-Streaming 唯一的 React/Vite 浏览器门户。原来位于根目录的独立
`webui/` 特性展示前端已合并到这里；runtime 将构建产物从 `/demo/` 提供。

## 运行时合同

门户只保留一个浏览器入口，并用 hash 路由承载各个页面：

| URL | 用途 |
| --- | --- |
| `/demo/` | 发现实例、合成、播放与诊断 |
| `/demo/#/sdk` | 匹配的 Python/Browser SDK 元数据与下载 |
| `/demo/#/docs/` | 同版本 Markdown 文档 |

设置 `DEMO_ENABLED=false` 会关闭整个门户。docs-only 构建没有 live runtime，也没有可交互
合成。

## 本地开发

workspace 要求 Node.js 22 或更高版本及 npm：

```bash
cd web
npm ci

# 检查所有 workspace 并构建门户与 Browser SDK
npm run typecheck
npm run build

# 修改布局或文档时运行门户
npm run dev --workspace @xmultimodalinteraction/qwen3tts-demo
```

Vite 服务适合只做 UI/文档迭代。应用会读取相对路径 `./config.json`，并使用相对的
`/v1/*` URL；要验证真实合成，请使用构建后的 runtime（或配置本地反向代理）。
不要启动或重新创建已经退出的独立 `webui/` 开发服务器。

## 测试与打包

在 `web/` 目录执行：

```bash
npm run lint
npm test
npm run test:e2e -- --project=chromium
```

`npm run build` 会运行文档构建器、检查类型，并生成供 engine 与 Triton runtime 镜像消费的
静态产物。runtime 镜像不安装 Node/npm，也不会在镜像内重新构建本包。UI 代码放在
`src/`，Browser SDK 依赖保持在独立 workspace 包中。
