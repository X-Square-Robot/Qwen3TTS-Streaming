**English** | [中文](roadmap.zh-CN.md)

# Roadmap

The v0.1 section below is historical. The current release line is v0.2, focused
on streaming stability, native text progress, and reproducible release evidence.
The former standalone WebUI has since been consolidated into the capability-gated built-in
Demo; current implementation and acceptance are tracked in
[`demo_browser_sdk_docs_plan.zh-CN.md`](dev/design/demo_browser_sdk_docs_plan.zh-CN.md).

Goal: evolve the current high-performance engineering prototype into a trustworthy, reproducible, collaboration-friendly, high-quality open-source project.

## v0.1: Engineering Preview (completed baseline)

Scope:

- `custom-1.7b` / `custom_voice` as the only recommended stable path.
- README, built-in Demo, and demo API clearly state the engineering preview positioning.
- Only publish benchmark numbers that carry complete conditions.
- autorun/build/deploy wire up max batch, max input len, max seq len, dtype, and engine mode.
- The manifest records the engine profile, and the runtime validates the profile limits before startup.
- The built-in Demo supports live capability gating and risk notices.
- Complete Chinese documentation.

Exit criteria:

- `custom-1.7b` passes minimal end-to-end acceptance.
- Key scripts pass `bash -n`.
- Python unit tests pass, or blocking reasons are recorded.
- The built-in Demo builds.
- The README no longer advertises untested paths as stable and ready to use.

## v0.2: Stability Release (current)

Focus:

- Ship the validated retrained checkpoint and runtime safeguards that substantially suppress
  the historical runaway-hallucination failure mode.
- Keep the 0/100 deterministic probe evidence and any future regression evidence tied to exact
  checkpoint, engine profile, sampling settings, and request corpus.
- Maintain text corpora across short sentences, long sentences, numbers, English, mixed
  Chinese-English, punctuation-dense text, and long paragraphs.
- Keep improving streaming TN, spliter behavior, EOS/pad handling, cache, and sampling defaults.
- Publish known limitations and reproduction inputs with every release.

Exit criteria:

- `custom-1.7b` / `custom_voice` remains the recommended stable path for v0.2.
- The validated full-bf16 engine has a recorded 0/100 deterministic runaway probe result.
- Every release can provide known issues, reproduction inputs, and benchmark conditions.
- Severe hallucination/repetition under default parameters is substantially reduced, while
  arbitrary checkpoints remain caller/operator validation scope.

## v0.3: base / ICL Voice Cloning

Focus:

- Complete ref audio preprocessing.
- Wire up speaker embedding, ref codes, and ref codec sum vec into the prefill/build plan.
- Distinguish the request protocols for x-vector clone and ICL clone.
- Add base/ICL end-to-end tests.
- Add reference audio upload and ref text input to the built-in Demo, but still mark it experimental by default.

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
- Improve CI: Python unit tests, manifest schema validation, bash syntax, built-in Demo build.
- Publish versioned artifacts and release notes.

Exit criteria:

- Development no longer requires rebuilding the dependency image for ordinary code changes.
- CI can block README convention, manifest schema, Demo type errors, and core unit test regressions.

## English Documentation

The English version is not a priority for the current phase. Once the Chinese README and docs stabilize, translate:

- README
- known limitations
- deployment
- benchmark methodology
- roadmap

When translating, do not weaken the risk notices, and do not present experimental paths as stable capabilities.
