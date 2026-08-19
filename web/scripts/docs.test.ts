import {readFileSync} from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import MarkdownIt from "markdown-it";
import {describe, expect, it} from "vitest";

import {configureCodeRendering} from "./markdown-code.mjs";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(webRoot, "..");
const manifest = JSON.parse(readFileSync(path.join(webRoot, "docs-manifest.json"), "utf8")) as Array<{
  slug: string;
  topic: string;
  locale: "zh-CN" | "en";
  layer: "start" | "advanced" | "details";
  source: string;
}>;

describe("caller documentation", () => {
  it("keeps a complete three-layer navigation in both languages", () => {
    expect(new Set(manifest.map((entry) => entry.slug)).size).toBe(manifest.length);
    for (const locale of ["zh-CN", "en"] as const) {
      const localized = manifest.filter((entry) => entry.locale === locale);
      expect(localized[0]?.topic).toBe("quickstart");
      expect(new Set(localized.map((entry) => entry.layer))).toEqual(
        new Set(["start", "advanced", "details"]),
      );
    }
    for (const topic of new Set(manifest.map((entry) => entry.topic))) {
      expect(manifest.filter((entry) => entry.topic === topic).map((entry) => entry.locale).sort())
        .toEqual(["en", "zh-CN"]);
    }
  });

  it("does not publish engine-maintainer documents in the caller portal", () => {
    for (const entry of manifest) {
      expect(entry.source).not.toMatch(/^(?:README(?:\.zh-CN)?\.md|docs\/dev\/)/);
      expect(readFileSync(path.join(repoRoot, entry.source), "utf8").length).toBeGreaterThan(100);
    }
  });

  it("builds highlighted, copyable code without a browser highlighter", () => {
    const markdown = configureCodeRendering(new MarkdownIt(), "quickstart-zh");
    const html = markdown.render("```python\nname = \"Qwen3-TTS\"\nif name:\n    print(name)\n```\n");
    expect(html).toContain("class=\"code-frame\"");
    expect(html).toContain("data-copy-code");
    expect(html).toContain("hljs-keyword");
    expect(html).toContain("hljs-string");
  });
});
