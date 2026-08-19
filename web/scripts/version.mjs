const PEP440_RELEASE = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)(0|[1-9]\d*)|\.dev(0|[1-9]\d*))?$/;

/** Map the repository's supported PEP 440 release versions to SemVer. */
export function pep440ToSemver(version) {
  const match = PEP440_RELEASE.exec(version);
  if (!match) {
    throw new Error(`Unsupported release version: ${version}`);
  }
  const [, major, minor, patch, prerelease, sequence, development] = match;
  if (development) return `${major}.${minor}.${patch}-dev.${development}`;
  if (!prerelease) return `${major}.${minor}.${patch}`;
  const label = prerelease === "a" ? "alpha" : prerelease === "b" ? "beta" : "rc";
  return `${major}.${minor}.${patch}-${label}.${sequence}`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const input = process.argv[2]?.replace(/^v/, "");
  if (!input) throw new Error("Usage: node version.mjs <PEP440 version or tag>");
  process.stdout.write(`${pep440ToSemver(input)}\n`);
}
