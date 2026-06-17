# WebUI — Qwen3-TTS Demo Frontend

Vite + React frontend for the Qwen3-TTS demo application.

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend expects the Demo API at `http://localhost:7860` (configurable via `VITE_DEMO_API_URL`).

## Production Build

```bash
npm run build
```

Output goes to `dist/`.

## Features

- **Text Player**: Play text token-by-token following engine decode steps, then seek the synthesized WAV audio
- **LLM PK**: Compare streaming vs non-streaming synthesis side by side
- **Concurrency**: Multi-stream TTFT distribution and throughput visualization

## Quick Launch

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b
```

Or manually:

```bash
# Terminal 1: Start Demo API
python -m demo_api --host 0.0.0.0 --port 7860

# Terminal 2: Start WebUI
cd webui && npm install && npm run dev
```
