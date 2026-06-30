"""L0 client-side self-analysis — answer the daily questions from the protocol
alone, without server logs.

This is the entry point of the escalation chain (see
``docs/dev/design/observability_tiers.md`` §7): a client can answer most daily
questions from the ``done`` event meta (and optional ``segment_end`` metas);
when it *cannot*, :meth:`SessionDiagnostics.explain` prints the concrete next
step — read the server L1 ``session.summary``, or replay with
``obs_level=debug`` to get the L2 ``split_decision`` / ``segment_synthesis``
records.

Usage::

    from qwen3tts.diagnostics import SessionDiagnostics

    diag = SessionDiagnostics.from_messages(session.iter_messages())
    print(diag.explain())          # human-readable + upgrade hints
    diag.summary()                 # structured five-question answers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .timing import ServerTimingReport


@dataclass
class SessionDiagnostics:
    """Self-analysis of one request from its protocol meta (L0)."""

    done_meta: dict[str, str] = field(default_factory=dict)
    segment_metas: list[dict[str, str]] = field(default_factory=list)
    timing: ServerTimingReport = field(default_factory=ServerTimingReport)

    @classmethod
    def from_done_meta(
        cls,
        done_meta: dict[str, str],
        segment_metas: Optional[list[dict[str, str]]] = None,
    ) -> "SessionDiagnostics":
        return cls(
            done_meta=dict(done_meta or {}),
            segment_metas=list(segment_metas or []),
            timing=ServerTimingReport.from_done_meta(done_meta or {}),
        )

    @classmethod
    def from_messages(cls, messages: Iterable[Any]) -> "SessionDiagnostics":
        """Build from an iterable of StreamEvent/AudioChunk messages, picking
        the ``done`` event meta and any ``segment_end`` event metas."""
        done_meta: dict[str, str] = {}
        segment_metas: list[dict[str, str]] = []
        for m in messages:
            mtype = getattr(m, "type", None)
            meta = getattr(m, "meta", None)
            if not isinstance(meta, dict):
                continue
            if mtype in ("done", "error"):
                done_meta = meta
            elif mtype == "segment_end":
                segment_metas.append(meta)
        return cls.from_done_meta(done_meta, segment_metas)

    # -- the five daily questions ------------------------------------------

    def _batch_summary(self) -> Optional[dict]:
        raw = self.done_meta.get("server_batch_summary")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def _segment_eos_reasons(self) -> list[str]:
        out = []
        for sm in self.segment_metas:
            r = sm.get("eos_reason") or sm.get("segment_eos_reason")
            if r:
                out.append(r)
        return out

    def summary(self) -> dict[str, Any]:
        """Structured answers to the five daily questions."""
        t = self.timing
        return {
            "ttft": {
                "create_to_first_raw_ms": t.server_session_create_to_first_raw_audio_ms,
                "dequeue_to_first_raw_ms": t.server_first_text_dequeue_to_first_raw_audio_ms,
                "engine_prefill_ms": t.server_engine_prefill_ms,
                "total_latency_ms": t.server_total_latency_ms,
            },
            "synthesized_text": self.done_meta.get("server_final_synthesized_text"),
            "batch": self._batch_summary(),
            "vad": {
                "prefix_trim_applied": t.server_prefix_trim_applied,
                "prefix_trimmed_ms": t.server_prefix_trimmed_ms,
                "gating_ms": t.server_first_raw_to_first_effective_audio_ms,
            },
            "cache": {
                "hit": t.server_cache_hit,
                "tokens_reused": t.server_cache_tokens_reused,
            },
            "eos_reasons": self._segment_eos_reasons(),
        }

    # -- explanation + escalation hints ------------------------------------

    def _session_ref(self) -> str:
        return (
            self.done_meta.get("request_id")
            or self.done_meta.get("session_id")
            or "<session_id>"
        )

    def explain(self) -> str:
        """Human-readable answer to "what did the engine do", plus the concrete
        next escalation step when the protocol can't settle the question."""
        lines = ["Session self-analysis (L0, from protocol meta):"]
        lines.append(self.timing.explain_latency())

        text = self.done_meta.get("server_final_synthesized_text")
        if text:
            lines.append(f'Synthesized: "{text}"')
        batch = self._batch_summary()
        if batch:
            lines.append(
                f"Batching: {batch.get('batched', 0)}/{batch.get('segments', 0)} "
                f"segments co-batched (max batch {batch.get('max_batch_size_seen', 0)})"
            )
        eos = self._segment_eos_reasons()
        if eos:
            lines.append(f"Segment endings: {eos}")

        # Escalation hints — what to do when L0 can't settle it.
        hints: list[str] = []
        ref = self._session_ref()
        abnormal = [r for r in eos if r and r != "codec_eos"]
        if abnormal:
            hints.append(
                f"Abnormal segment endings {abnormal} → replay with "
                f"obs_level=debug and read the L2 `segment_synthesis` record "
                f"(why synthesis went wrong)."
            )
        # If the dominant latency is inference, the cause is server-side.
        prefill = self.timing.server_engine_prefill_ms or 0.0
        dequeue_raw = self.timing.server_first_text_dequeue_to_first_raw_audio_ms or 0.0
        if dequeue_raw and (dequeue_raw - prefill) > max(prefill, 50.0):
            hints.append(
                f"Inference dominates TTFT → grep server L1 `session.summary` for "
                f"session={ref}; if the split looks wrong, replay with "
                f"obs_level=debug for the `split_decision` records."
            )
        if not self.done_meta:
            hints.append(
                "No done-event meta found → the request likely errored before "
                "completion; inspect the error event and server L1 logs."
            )

        if hints:
            lines.append("Next step if unresolved:")
            lines.extend(f"  • {h}" for h in hints)
        return "\n".join(lines)
