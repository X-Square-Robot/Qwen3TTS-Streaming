**English** | [中文](roadmap.zh-CN.md)

# Roadmap

Goal: evolve the current high-performance engineering prototype into a trustworthy, reproducible, collaboration-friendly, high-quality open-source project.

## v0.1: Engineering Preview

Scope:

- `custom-1.7b` / `custom_voice` as the only recommended stable path.
- README, WebUI, and demo API clearly state the engineering preview positioning.
- Only publish benchmark numbers that carry complete conditions.
- autorun/build/deploy wire up max batch, max input len, max seq len, dtype, and engine mode.
- The manifest records the engine profile, and the runtime validates the profile limits before startup.
- The WebUI supports fixture/live source differentiation and risk notices.
- Complete Chinese documentation.

Exit criteria:

- `custom-1.7b` passes minimal end-to-end acceptance.
- Key scripts pass `bash -n`.
- Python unit tests pass, or blocking reasons are recorded.
- The WebUI builds.
- The README no longer advertises untested paths as stable and ready to use.

## v0.2: Stability Focus

Focus:

- Systematically locate streaming hallucination, repetition, dropped reading, and inserted content issues.
- Build a text corpus: short sentences, long sentences, numbers, English, mixed Chinese-English, punctuation-dense text, and long paragraphs.
- Add audio quality regression tests and a manual acceptance sheet.
- Improve spliter, EOS/pad, cache, and sampling defaults.
- Distill failure cases into `docs/streaming_hallucination_investigation.md` or a new Chinese document.

Exit criteria:

- Publish a set of stability test cases.
- Every release can provide known issues and reproduction inputs.
- The probability of severe hallucination/repetition under default parameters is significantly reduced.

## v0.3: base / ICL Voice Cloning

Focus:

- Complete ref audio preprocessing.
- Wire up speaker embedding, ref codes, and ref codec sum vec into the prefill/build plan.
- Distinguish the request protocols for x-vector clone and ICL clone.
- Add base/ICL end-to-end tests.
- Add reference audio upload and ref text input to the WebUI, but still mark it experimental by default.

Exit criteria:

- base voice clone can run through with real ref audio.
- ICL voice clone can run through with ref audio + ref text.
- Error messages clearly explain missing fields or capabilities that are not enabled.

## v0.4: voice design

Focus:

- Fully validate `design-1.7b`.
- Sort out the mutual exclusivity between the instruct field, the speaker field, and custom voice.
- Build a voice design example set.
- Clarify its quality boundary relative to custom voice.

Exit criteria:

- voice design has its own demo and test inputs.
- The README can upgrade it from an "experimental path" to a "usable path".

## v0.5: Deployment and Maintainability

Focus:

- Optimize the environment-layer/code-layer experience of engine Docker.
- Add K8s/Helm or production compose examples.
- Add health checks, rate limiting, logging, metrics, and trace ids.
- Improve CI: Python unit tests, manifest schema validation, bash syntax, WebUI build.
- Publish versioned artifacts and release notes.

Exit criteria:

- Development no longer requires rebuilding the dependency image for ordinary code changes.
- CI can block README convention, manifest schema, WebUI type errors, and core unit test regressions.

## English Documentation

The English version is not a priority for the current phase. Once the Chinese README and docs stabilize, translate:

- README
- known limitations
- deployment
- benchmark methodology
- roadmap

When translating, do not weaken the risk notices, and do not present experimental paths as stable capabilities.
