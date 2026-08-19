import {createHash} from "node:crypto";
import {copyFile, mkdir, readFile, rm, stat, writeFile} from "node:fs/promises";
import {statSync} from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import MarkdownIt from "markdown-it";
import anchor from "markdown-it-anchor";

import {configureCodeRendering} from "./markdown-code.mjs";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(webRoot, "..");
const publicRoot = path.join(webRoot, "packages/demo/public");
const assetsRoot = path.join(publicRoot, "docs-assets");
const manifest = JSON.parse(await readFile(path.join(webRoot, "docs-manifest.json"), "utf8"));
const bySource = new Map(manifest.map((entry) => [normalize(entry.source), entry.slug]));
const sourceBase = (process.env.DOCS_SOURCE_BASE ?? "https://github.com/X-Square-Robot/Qwen3TTS-Streaming/blob/main").replace(/\/$/, "");

await mkdir(publicRoot, {recursive: true});
await rm(assetsRoot, {recursive: true, force: true});
await mkdir(assetsRoot, {recursive: true});

const documents = [];
for (const entry of manifest) {
  const sourcePath = path.resolve(repoRoot, entry.source);
  ensureInsideRepo(sourcePath);
  const markdown = normalizePresentationalHtml(await readFile(sourcePath, "utf8"));
  const md = new MarkdownIt({html: false, linkify: true, typographer: false})
    .use(anchor, {slugify: slugifyHeading});
  configureCodeRendering(md, entry.slug);
  const defaultLinkOpen = md.renderer.rules.link_open
    ?? ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const hrefIndex = token?.attrIndex("href") ?? -1;
    if (token && hrefIndex >= 0) {
      const href = token.attrs?.[hrefIndex]?.[1] ?? "";
      token.attrSet("href", rewriteLink(href, sourcePath));
    }
    return defaultLinkOpen(tokens, index, options, env, self);
  };
  const tokens = md.parse(markdown, {});
  for (const token of tokens) {
    if (token.type !== "inline" || !token.children) continue;
    for (const child of token.children) {
      if (child.type === "image") {
        const source = child.attrGet("src") ?? "";
        if (!isExternal(source)) child.attrSet("src", await copyAsset(source, sourcePath));
      }
      if (child.type === "link_open") {
        const href = child.attrGet("href") ?? "";
        if (!isExternal(href) && isMedia(href)) child.attrSet("href", await copyAsset(href, sourcePath));
      }
    }
  }
  documents.push({...entry, html: md.renderer.render(tokens, md.options, {})});
}

await writeFile(
  path.join(publicRoot, "docs.json"),
  `${JSON.stringify({schema_version: "qwen.tts.docs.v1", documents})}\n`,
  "utf8",
);
console.log(`Built ${documents.length} documents from repository Markdown`);

function rewriteLink(href, sourcePath) {
  if (!href || isExternal(href) || href.startsWith("#") || href.startsWith("./docs-assets/")) return href;
  const [pathname, anchorPart = ""] = href.split("#", 2);
  const absolute = path.resolve(path.dirname(sourcePath), decodeURIComponent(stripQueryAndHash(pathname)));
  ensureInsideRepo(absolute);
  try {
    statSync(absolute);
  } catch {
    throw new Error(`Broken documentation link: ${path.relative(repoRoot, sourcePath)} -> ${href}`);
  }
  const target = normalize(path.relative(repoRoot, absolute));
  const slug = bySource.get(target);
  if (slug) return `#/docs/${slug}${anchorPart ? `?anchor=${encodeURIComponent(anchorPart)}` : ""}`;
  const encoded = target.split("/").map(encodeURIComponent).join("/");
  return `${sourceBase}/${encoded}${anchorPart ? `#${anchorPart}` : ""}`;
}

function normalizePresentationalHtml(markdown) {
  return markdown
    .replace(
      /^[\t ]*<div[\t ]+align=(?:"center"|'center'|center)[\t ]*>[\t ]*$([\s\S]*?)^[\t ]*<\/div>[\t ]*$/gmi,
      "$1",
    )
    .replace(/<img\b([^>]*)>/gi, (_tag, attributes) => {
      const source = htmlAttribute(attributes, "src");
      if (!source) throw new Error("Documentation image is missing a src attribute");
      const alt = htmlAttribute(attributes, "alt")
        .replace(/\\/g, "\\\\")
        .replace(/([\[\]])/g, "\\$1");
      return `![${alt}](${source})`;
    })
    .replace(/<br[\t ]*\/?[\t ]*>/gi, "  \n");
}

function htmlAttribute(attributes, name) {
  const match = attributes.match(new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\x60]+))`, "i"));
  return match?.[1] ?? match?.[2] ?? match?.[3] ?? "";
}

async function copyAsset(source, sourcePath) {
  const absolute = path.resolve(path.dirname(sourcePath), decodeURIComponent(stripQueryAndHash(source)));
  ensureInsideRepo(absolute);
  const info = await stat(absolute);
  if (!info.isFile() || info.size > 20 * 1024 * 1024) throw new Error(`Unsupported documentation asset: ${absolute}`);
  const extension = path.extname(absolute).toLowerCase();
  if (!new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm"]).has(extension)) {
    throw new Error(`Unsupported documentation asset type: ${absolute}`);
  }
  const bytes = await readFile(absolute);
  const mime = detectMime(bytes);
  const expected = expectedMime(extension);
  if (!expected.includes(mime)) throw new Error(`Documentation asset MIME mismatch: ${absolute} (${mime})`);
  const filename = `${createHash("sha256").update(bytes).digest("hex").slice(0, 16)}${extension}`;
  await copyFile(absolute, path.join(assetsRoot, filename));
  return `./docs-assets/${filename}`;
}

function expectedMime(extension) {
  return ({
    ".png": ["image/png"], ".jpg": ["image/jpeg"], ".jpeg": ["image/jpeg"],
    ".gif": ["image/gif"], ".webp": ["image/webp"], ".svg": ["image/svg+xml"],
    ".mp4": ["video/mp4"], ".webm": ["video/webm"],
  })[extension] ?? [];
}

function detectMime(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "image/png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "image/jpeg";
  if (bytes.subarray(0, 6).toString("ascii") === "GIF87a" || bytes.subarray(0, 6).toString("ascii") === "GIF89a") return "image/gif";
  if (bytes.subarray(0, 4).toString("ascii") === "RIFF" && bytes.subarray(8, 12).toString("ascii") === "WEBP") return "image/webp";
  const prefix = bytes.subarray(0, 512).toString("utf8").replace(/^\uFEFF/, "").trimStart();
  if (/^(?:<\?xml[^>]*>\s*)?<svg[\s>]/i.test(prefix)) return "image/svg+xml";
  if (bytes.subarray(4, 8).toString("ascii") === "ftyp") return "video/mp4";
  if (bytes.subarray(0, 4).equals(Buffer.from([0x1a, 0x45, 0xdf, 0xa3]))) return "video/webm";
  return "application/octet-stream";
}

function isMedia(value) {
  return /\.(?:png|jpe?g|gif|webp|svg|mp4|webm)(?:[?#]|$)/i.test(value);
}

function slugifyHeading(value) {
  return value.trim().toLowerCase().replace(/[^\p{Letter}\p{Number}]+/gu, "-").replace(/^-|-$/g, "");
}

function isExternal(value) {
  return /^(?:[a-z]+:|\/\/|data:)/i.test(value);
}

function stripQueryAndHash(value) {
  return value.split(/[?#]/, 1)[0] ?? value;
}

function normalize(value) {
  return value.split(path.sep).join("/");
}

function ensureInsideRepo(value) {
  const relative = path.relative(repoRoot, value);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Documentation path escapes repository: ${value}`);
  }
}
