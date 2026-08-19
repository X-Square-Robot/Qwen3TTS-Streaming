import { mkdir, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const destination = resolve(root, "dist");
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
execFileSync("npm", ["pack", "--workspace", "@xmultimodalinteraction/qwen3tts-browser", "--pack-destination", destination], {
  cwd: root,
  env: {...process.env, npm_config_cache: resolve(root, ".cache", "npm")},
  stdio: "inherit",
});
