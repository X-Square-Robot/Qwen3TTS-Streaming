import {createHash} from "node:crypto";
import {readFile, writeFile} from "node:fs/promises";
import {basename, resolve} from "node:path";

const required = ["CI_API_V4_URL", "CI_PROJECT_ID", "CI_JOB_TOKEN", "BROWSER_SDK_SEMVER", "BROWSER_SDK_TARBALL", "DEMO_ARCHIVE"];
for (const name of required) if (!process.env[name]) throw new Error(`${name} is required`);

const root = resolve(import.meta.dirname, "..");
const packageName = "qwen3tts-web";
const version = process.env.BROWSER_SDK_SEMVER;
const headers = {"JOB-TOKEN": process.env.CI_JOB_TOKEN};
const published = {};

for (const [key, filename] of [["BROWSER_SDK_GENERIC_URL", process.env.BROWSER_SDK_TARBALL], ["DEMO_ARCHIVE_URL", process.env.DEMO_ARCHIVE]]) {
  const path = resolve(root, "dist", basename(filename));
  const bytes = await readFile(path);
  const url = `${process.env.CI_API_V4_URL}/projects/${process.env.CI_PROJECT_ID}/packages/generic/${packageName}/${version}/${basename(path)}`;
  const existing = await fetch(url, {headers});
  if (existing.ok) {
    const remote = Buffer.from(await existing.arrayBuffer());
    if (sha256(remote) !== sha256(bytes)) throw new Error(`GitLab already contains different bytes for ${basename(path)}`);
  } else if (existing.status === 404) {
    const uploaded = await fetch(url, {method: "PUT", headers, body: bytes});
    if (!uploaded.ok) throw new Error(`GitLab upload failed for ${basename(path)}: HTTP ${uploaded.status} ${await uploaded.text()}`);
  } else {
    throw new Error(`GitLab lookup failed for ${basename(path)}: HTTP ${existing.status}`);
  }
  published[key] = url;
}

await writeFile(resolve(root, "dist", "published-web.env"), Object.entries(published).map(([key, value]) => `${key}=${value}\n`).join(""));

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
