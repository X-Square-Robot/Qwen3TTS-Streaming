[中文](README.zh-CN.md) | **English**

# Qwen3-TTS built-in Demo

This package is the single React/Vite browser portal for Qwen3TTS-Streaming.
The former standalone root `webui/` feature showcase was consolidated here;
the runtime serves the built artifact at `/demo/`.

## Runtime contract

The portal keeps one browser entry point and uses hash routes for its surfaces:

| URL | Purpose |
| --- | --- |
| `/demo/` | Instance discovery, synthesis, playback, and diagnostics |
| `/demo/#/sdk` | Matching Python and Browser SDK metadata/downloads |
| `/demo/#/docs/` | Same-release Markdown documentation |
| `/demo/#/lab` | Engineering Lab: public-Realtime LLM PK/concurrency and optional trace tools |

`DEMO_ENABLED=false` disables the whole portal. The Lab uses the current
instance's public `/v1/realtime` endpoint. Trace capture, LLM PK, and
concurrency experiments all use the same Gateway; there is no separate backend
or direct Triton connection. A docs-only build has no live runtime or
interactive Lab.

## Local development

The workspace requires Node.js 22 or newer and npm:

```bash
cd web
npm ci

# Check every workspace and build the portal plus Browser SDK
npm run typecheck
npm run build

# Run the portal while iterating on layout or docs
npm run dev --workspace @xmultimodalinteraction/qwen3tts-demo
```

The Vite server is useful for UI-only work. The app fetches `./config.json` and
uses relative `/v1/*` URLs, so use a built runtime (or a local reverse proxy)
for live synthesis and Lab checks. Do not start or recreate the retired
standalone `webui/` development server.

## Tests and packaging

From `web/`:

```bash
npm run lint
npm test
npm run test:e2e -- --project=chromium
```

`npm run build` runs the documentation builder, type-checks the package, and
produces the static artifact consumed by the engine and Triton runtime images.
The runtime image does not install Node/npm or rebuild this package. Keep UI
code in `src/`, with the Browser SDK dependency in its own workspace package.
