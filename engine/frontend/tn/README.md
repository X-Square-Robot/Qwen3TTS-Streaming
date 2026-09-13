# Streaming TN span layer

`tn` owns the control plane for incremental text normalization. `SpanFSM` is
the table-driven lifecycle (`SCAN → OPEN → READY → NORMALIZE → COMMIT` with
`WAIT`/`FALLBACK` and `DONE` terminal paths), while recognizers expose lexical
classification through a narrow protocol. The existing committer remains the
single owner of normalization rules during this migration; `SpanDriver` feeds
its decisions into the FSM so the frontend has one lifecycle boundary.

The frontend collects all `TextCommit` spoken deltas produced by one input
packet, appends their raw mappings to `CanonicalTextJournal`, and invokes the
tokenizer once for the concatenated spoken suffix. Per-commit normalized
offsets are retained as splitter boundaries, so batching does not merge
semantic slots.

Candidate generation is optional. `CandidateRecognizer` accepts the existing
WeText candidate contract (or a future WFST provider) and attaches n-best
evidence to a recognition; it never decides closure. `CommitPolicy` remains
the authority for `WAIT`, `COMMIT`, and deterministic `FALLBACK` at the commit
fence.
