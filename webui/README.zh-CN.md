[English](README.md) | **中文**

# WebUI — Qwen3-TTS 演示前端

Qwen3-TTS 演示应用的 Vite + React 前端。

## 开发

```bash
npm install
npm run dev
```

打开 `http://localhost:5173`。前端期望 Demo API 位于 `http://localhost:7860`（可通过 `VITE_DEMO_API_URL` 配置）。

## 生产构建

```bash
npm run build
```

产物输出到 `dist/`。

## 特性

- **Text Player**：跟随引擎解码步骤逐 token 播放文本，随后可对合成的 WAV 音频进行拖动定位
- **LLM PK**：并排对比流式与非流式合成
- **Concurrency**：多路流的 TTFT 分布与吞吐量可视化

## 快速启动

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b
```

或手动启动：

```bash
# Terminal 1: Start Demo API
python -m demo_api --host 0.0.0.0 --port 7860

# Terminal 2: Start WebUI
cd webui && npm install && npm run dev
```
