**English** | [中文](known_limitations.zh-CN.md)

# Limits and launch checks

This page describes product boundaries an API caller must own. How the service is built, scheduled,
or deployed does not change these conclusions.

## Trust the running service's capabilities

Deployments may load different models and disable individual features. Read `/v1/capabilities`
before making requests. Do not infer availability from SDK types or another deployment's Demo.

`custom_voice` is the primary validated path today. Use voice design or voice clone in production
only when capabilities advertise it and the operator confirms that deployment has been validated.

## Synthesized content is not strongly consistent

Streaming TTS may repeat, omit, or insert content, produce abnormal silence, or degrade on long text.
Punctuation, numbers, English, and mixed-language input can change segmentation and prosody. Guarded
delivery, VAD, and length guards can reduce the chance that some bad tails reach a client, but they
cannot prove that speech matches the source text.

Customer service, medical, financial, legal, alerting, and other high-risk uses need application-level
text constraints, output sampling, cancellation, and human fallback. A completed synthesis state is
not a semantic correctness guarantee.

## VAD can change the beginning and end

Output VAD trims content classified as silence and adds speech-onset confirmation time. A high
threshold can reject an entire response; too little start margin can clip an initial consonant.
Threshold scales from different detectors are not interchangeable.

Test real voices with soft onset, plosives, long pauses, numbers, and mixed languages. If VAD removes
all audio, return an explicit error instead of waiting indefinitely.

## Break latency into stages

Raw server TTFT, post-VAD effective TTFT, first audio received by a browser, and first sample consumed
by a speaker are different metrics. Public networking, proxies, Base64 transport, browser scheduling,
and playback buffering can add substantial end-to-end latency.

Published benchmarks represent only their stated hardware, concurrency, cache, and network
conditions. Measure P50/P95/P99 with your region, text distribution, and target concurrency, together
with error rate and audio completeness.

## Browser constraints

- Microphone access, AudioWorklet, and output-device selection require a secure HTTPS context.
- Browsers usually require a user gesture before starting an `AudioContext`.
- Device enumeration, switching, and labels differ between browsers.
- Tab closure, system sleep, and mobile network changes can interrupt an active response; retry only
  when the service advertises recovery.
- Never bundle a long-lived API key into a public web application.

## Launch checklist

- Pin the service URL, SDK release, and protocol major version; validate capabilities at startup.
- Set a total timeout, cancellation path, concurrency limit, and retry budget.
- Log terminal state, usage, server timing, and business trace by `response_id`.
- Cover short, long, empty, mixed-language, unsupported-speaker, disconnection, and rate-limit cases.
- Validate PCM format, continuous sample cursors, player underruns, and final audio duration.
- Tune VAD within product tolerance and provide an explicit no-valid-audio fallback.
- Add content validation or human confirmation for high-risk messages.
