import { describe, expect, it } from "vitest";
import { pep440ToSemver } from "./version.mjs";

describe("pep440ToSemver", () => {
  it.each([
    ["1.2.3", "1.2.3"],
    ["1.2.3a1", "1.2.3-alpha.1"],
    ["1.2.3b2", "1.2.3-beta.2"],
    ["1.2.3rc4", "1.2.3-rc.4"],
    ["1.2.3.dev5", "1.2.3-dev.5"],
  ])("maps %s", (input, expected) => expect(pep440ToSemver(input)).toBe(expected));

  it.each(["1.2", "01.2.3", "1.2.3post1", "1.2.3.post1"])(
    "rejects unsupported %s",
    (input) => expect(() => pep440ToSemver(input)).toThrow("Unsupported release version"),
  );
});
