import {createHash} from "node:crypto";
import {mkdtemp, readFile, readdir, rm} from "node:fs/promises";
import {basename, dirname, resolve} from "node:path";
import {execFile} from "node:child_process";
import {setTimeout as sleep} from "node:timers/promises";
import {promisify} from "node:util";

const exec = promisify(execFile);
const VERIFY_ATTEMPTS = readInteger("NPM_PUBLISH_VERIFY_ATTEMPTS", 8, 1);
const VERIFY_INITIAL_DELAY_MS = readInteger("NPM_PUBLISH_VERIFY_INITIAL_DELAY_MS", 2000, 0);
const VERIFY_MAX_DELAY_MS = readInteger("NPM_PUBLISH_VERIFY_MAX_DELAY_MS", 15000, 0);
const required = [
  "CI_API_V4_URL",
  "CI_PROJECT_ID",
  "CI_JOB_TOKEN",
  "CI_SERVER_HOST",
  "BROWSER_SDK_SEMVER",
  "BROWSER_SDK_TARBALL",
  "BROWSER_SDK_DIST_TAG",
];
for (const name of required) {
  if (!process.env[name]) throw new Error(`${name} is required`);
}

const root = resolve(import.meta.dirname, "..");
const packageJson = JSON.parse(
  await readFile(resolve(root, "packages/browser-sdk/package.json"), "utf8"),
);
const packageName = packageJson.name;
const version = process.env.BROWSER_SDK_SEMVER;
const distTag = process.env.BROWSER_SDK_DIST_TAG;
const archive = resolve(root, "dist", basename(process.env.BROWSER_SDK_TARBALL));
const registry = `${process.env.CI_API_V4_URL}/projects/${process.env.CI_PROJECT_ID}/packages/npm/`;

const localSha256 = await sha256File(archive);
const existing = await runNpm(
  ["view", `${packageName}@${version}`, "version", "--registry", registry],
  true,
);

if (existing.status === 0) {
  if (existing.stdout.trim() !== version) {
    throw new Error(`GitLab npm registry returned an unexpected version for ${packageName}`);
  }
  const downloaded = await downloadPublishedArchive();
  if (downloaded.sha256 !== localSha256) {
    throw new Error(`GitLab npm registry already contains different bytes for ${basename(archive)}`);
  }
  console.log(`Browser SDK npm package is already present: ${packageName}@${version}`);
} else if (!/(E404|ETARGET|404|no matching version found|not found)/i.test(`${existing.stdout}\n${existing.stderr}`)) {
  throw new Error(
    `GitLab npm registry lookup failed: ${existing.stderr || existing.stdout}`.trim(),
  );
} else {
  const published = await runNpm(
    ["publish", archive, "--registry", registry, "--tag", distTag],
    true,
  );
  const publishOutput = `${published.stdout}\n${published.stderr}`;
  if (published.status !== 0 && !isAlreadyPublishedError(publishOutput)) {
    throw new Error(`GitLab npm publish failed: ${publishOutput}`.trim());
  }
  if (published.status === 0) {
    console.log(`Published ${packageName}@${version}; waiting for registry visibility`);
  } else {
    console.log(`Package publish was already acknowledged; verifying ${packageName}@${version}`);
  }
  const downloaded = await downloadPublishedArchive();
  if (downloaded.sha256 !== localSha256) {
    throw new Error(`Published npm package does not match ${basename(archive)}`);
  }
}
await verifyOrCreateDistTag();

async function downloadPublishedArchive() {
  let lastError = "unknown npm registry error";
  for (let attempt = 1; attempt <= VERIFY_ATTEMPTS; attempt += 1) {
    const result = await tryDownloadPublishedArchive();
    if (result.archive) return result.archive;
    lastError = result.error;
    if (attempt < VERIFY_ATTEMPTS) {
      const delayMs = Math.min(
        VERIFY_MAX_DELAY_MS,
        VERIFY_INITIAL_DELAY_MS * 2 ** (attempt - 1),
      );
      console.log(
        `Published npm package is not visible yet (attempt ${attempt}/${VERIFY_ATTEMPTS}); ` +
          `retrying in ${delayMs}ms`,
      );
      await sleep(delayMs);
    }
  }
  throw new Error(
    `Published npm package ${packageName}@${version} was not visible after ` +
      `${VERIFY_ATTEMPTS} attempts: ${lastError}`,
  );
}

async function tryDownloadPublishedArchive() {
  const tempDir = await mkdtemp("/tmp/qwen3tts-npm-");
  try {
    const packed = await runNpm(
      [
        "pack",
        `${packageName}@${version}`,
        "--registry",
        registry,
        "--pack-destination",
        tempDir,
      ],
      true,
    );
    if (packed.status !== 0) {
      return {
        error:
          `${packed.stderr || packed.stdout}`.trim() ||
          `npm pack exited with ${packed.status}`,
      };
    }
    const files = (await readdir(tempDir)).filter((name) => name.endsWith(".tgz"));
    if (files.length !== 1) {
      return {error: `Expected exactly one downloaded npm archive, found ${files.length}`};
    }
    const path = resolve(tempDir, files[0]);
    const archive = {path, sha256: await sha256File(path)};
    return {archive};
  } finally {
    await rm(tempDir, {recursive: true, force: true});
  }
}

async function runNpm(args, allowFailure = false) {
  try {
    const result = await exec("npm", args, {
      cwd: dirname(archive),
      env: {
        ...process.env,
        npm_config_always_auth: "true",
        npm_config_registry: registry,
      },
    });
    return {status: 0, stdout: result.stdout, stderr: result.stderr};
  } catch (error) {
    if (!allowFailure) throw error;
    return {
      status: error.code ?? 1,
      stdout: error.stdout ?? "",
      stderr: error.stderr ?? "",
    };
  }
}

function isAlreadyPublishedError(output) {
  return /(?:EPUBLISHCONFLICT|E409|409|already exists|cannot publish over|previously published)/i.test(
    output,
  );
}

function readInteger(name, fallback, minimum) {
  const value = Number.parseInt(process.env[name] ?? `${fallback}`, 10);
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return value;
}

async function verifyOrCreateDistTag() {
  const listing = await runNpm(
    ["dist-tag", "ls", packageName, "--registry", registry],
    true,
  );
  if (listing.status !== 0) {
    throw new Error(
      `Could not read npm dist-tags for ${packageName}: ${listing.stderr || listing.stdout}`.trim(),
    );
  }
  const current = listing.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.startsWith(`${distTag}: `))
    ?.slice(distTag.length + 2);
  if (current && current !== version) {
    throw new Error(`npm dist-tag ${distTag} already points to ${current}, not ${version}`);
  }
  if (!current) {
    await runNpm(
      ["dist-tag", "add", `${packageName}@${version}`, distTag, "--registry", registry],
      false,
    );
  }
}

async function sha256File(path) {
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}
