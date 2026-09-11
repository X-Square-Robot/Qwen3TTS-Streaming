import {chmod, mkdir, mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {execFile} from "node:child_process";
import {promisify} from "node:util";

import {describe, expect, it} from "vitest";

const exec = promisify(execFile);
const webRoot = resolve(import.meta.dirname, "..");
const script = resolve(import.meta.dirname, "publish-gitlab-npm.mjs");

describe("publish-gitlab-npm", () => {
  it("waits for a newly published version to become visible", async () => {
    const tempRoot = await mkdtemp(join(tmpdir(), "qwen3tts-npm-test-"));
    const archiveName = "xmultimodalinteraction-qwen3tts-browser-0.2.2-alpha.0.tgz";
    const archivePath = resolve(webRoot, "dist", archiveName);
    const originalArchive = await readOptional(archivePath);
    const fakeNpm = join(tempRoot, "npm");
    const npmLog = join(tempRoot, "npm.log");
    const packAttempts = join(tempRoot, "pack-attempts");

    try {
      await mkdir(resolve(webRoot, "dist"), {recursive: true});
      await writeFile(archivePath, "candidate archive bytes\n");
      await writeFile(
        fakeNpm,
        `#!/usr/bin/env node
import {copyFile, readFile, writeFile} from "node:fs/promises";
const args = process.argv.slice(2);
const log = process.env.NPM_LOG;
await writeFile(log, (await readFile(log, "utf8").catch(() => "")) + args.join(" ") + "\\n");
if (args[0] === "view") {
  console.error("npm error code ETARGET");
  console.error("npm error notarget No matching version found");
  process.exit(1);
}
if (args[0] === "publish") process.exit(0);
if (args[0] === "pack") {
  const attempts = Number(await readFile(process.env.PACK_ATTEMPTS, "utf8").catch(() => "0")) + 1;
  await writeFile(process.env.PACK_ATTEMPTS, String(attempts));
  if (attempts < 3) {
    console.error("npm error code ETARGET");
    console.error("npm error notarget No matching version found");
    process.exit(1);
  }
  const destination = args[args.indexOf("--pack-destination") + 1];
  await copyFile(process.env.SOURCE_ARCHIVE, destination + "/downloaded.tgz");
  process.exit(0);
}
if (args[0] === "dist-tag" && args[1] === "ls") process.exit(0);
if (args[0] === "dist-tag" && args[1] === "add") process.exit(0);
console.error("unexpected npm command", args.join(" "));
process.exit(2);
`,
        "utf8",
      );
      await chmod(fakeNpm, 0o755);
      await writeFile(npmLog, "");

      const {stdout} = await exec(process.execPath, [script], {
        cwd: webRoot,
        env: {
          ...process.env,
          PATH: `${tempRoot}:${process.env.PATH ?? ""}`,
          NPM_LOG: npmLog,
          PACK_ATTEMPTS: packAttempts,
          SOURCE_ARCHIVE: archivePath,
          CI_API_V4_URL: "https://gitlab.example.test/api/v4",
          CI_PROJECT_ID: "844",
          CI_JOB_TOKEN: "job-token",
          CI_SERVER_HOST: "gitlab.example.test",
          BROWSER_SDK_SEMVER: "0.2.2-alpha.0",
          BROWSER_SDK_TARBALL: archiveName,
          BROWSER_SDK_DIST_TAG: "alpha",
          NPM_PUBLISH_VERIFY_ATTEMPTS: "3",
          NPM_PUBLISH_VERIFY_INITIAL_DELAY_MS: "1",
          NPM_PUBLISH_VERIFY_MAX_DELAY_MS: "1",
        },
      });

      const log = await readFile(npmLog, "utf8");
      expect(stdout).toContain("retrying");
      expect(log.match(/^pack /gm)).toHaveLength(3);
      expect(log).toContain("publish");
      expect(log).toContain("dist-tag add");
    } finally {
      if (originalArchive === undefined) {
        await rm(archivePath, {force: true});
      } else {
        await writeFile(archivePath, originalArchive);
      }
      await rm(tempRoot, {recursive: true, force: true});
    }
  });
});

async function readOptional(path: string) {
  try {
    return await readFile(path);
  } catch {
    return undefined;
  }
}
