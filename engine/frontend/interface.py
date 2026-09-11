"""Gateway-facing frontend interface.

The interface owns external session lifecycle and input semantics:

- create/cancel sessions for transport adapters
- tokenize and route text according to declared input mode
- consume backend results and surface ordered audio callbacks

It delegates backend request emission to ``frontend.dispatcher.Dispatcher``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..core.session import Session, SegmentOrderMeta
from ..core.speech_state import (
    SpeechStateCapability,
    coerce_speech_state_capability,
)
from ..core.text_journal import CanonicalTextJournal
from ..core.text_coordinates import (
    CodecTokenProgress, SegmentTextCoordinates, TextProgressProjection,
)
from ..core.text_progress import (
    EmaTextProgressEstimator,
    NativeCursorProgressProjector,
    NATIVE_CURSOR_PROGRESS_BASIS,
    TEXT_PROGRESS_BASIS,
)
from ..core.types import (
    EngineResult,
    AttributedAudioChunk,
    GroupPolicy,
    InputMode,
    ResultType,
    SegmentToken,
    SessionConfig,
    SessionState,
    TokenizedText,
)

# At 12 Hz this is roughly 1.3 seconds of source audio.  A short period of
# unchanged native output is expected around lookahead and chunk boundaries,
# but a longer stall must not block the session's text progress forever.
_NATIVE_CURSOR_STALL_FRAME_LIMIT = 16
from .text_commitment import AudioCreditEstimator, IncrementalTextCommitter, SemanticStartGate
from .text_commitment.x2_adapter import X2CommitmentAdapter
from .text_commitment.types import (
    FallbackPolicy,
    TextInputMetadata,
    TextNormalizationConfig as CommitterConfig,
)
from ..core.lifecycle import LifecycleLogger
from ..core import observability as obs
from ..core.timing import ServerTimingAccumulator
from ..session import SessionCapacityError
from ..text_normalization import strip_emoji, split_pending_emoji
from ..interface.output import ENGINE_SAMPLE_RATE
from .diagnostic_text import (
    DEFAULT_ENGINE_MODEL_VERSION,
    DiagnosticTextRouter,
    VERSION_QUERY_TEXT,
    normalize_version_query_text,
    resolve_diagnostic_text,
)
from .dispatcher import Dispatcher
from .hold_window import DeliveryHoldWindow, PrefixGateGuardBypass
from .spliter import Spliter
from .spliter.driver import ActionType
from .spliter.ratio import RatioObservation, RatioOutcome
from .spliter.reorder import AudioReorder

if TYPE_CHECKING:
    from .spliter.tokenizer import LightQwen3TTSTokenizer

logger = logging.getLogger(__name__)

_WHITESPACE_TO_STRIP = str.maketrans(
    {
        "\n": "",
        "\r": "",
        "\t": " ",
        "\u3000": "",
    }
)


def _normalize_tts_text(text: str) -> str:
    """Remove formatting whitespace that harms tokenization/prosody."""
    text = strip_emoji((text or "").translate(_WHITESPACE_TO_STRIP))
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _spoken_projection(
    raw_text: str,
    commits: tuple[Any, ...] | list[Any],
) -> tuple[str, list[int]]:
    """Build the final spoken text and its coarse raw-coordinate map.

    A ``TextCommit`` can expand one source span into many spoken characters
    (for example ``99%`` into a Chinese verbalization).  The regular journal
    append path cannot represent that rewrite because it intentionally enforces
    prefix monotonicity.  We therefore construct a source map from the
    append-only commit records and normalize the *spoken* stream in a small
    temporary journal before installing the completed projection.

    The map is deliberately conservative: when the backend does not expose a
    character-level alignment, every spoken character in one commit is
    attributed to that commit's raw interval.  This is sufficient for progress
    anchors and, importantly, never invents a raw offset outside the commit
    fence.
    """

    source = str(raw_text or "")
    spoken_parts: list[str] = []
    boundaries: list[int] = []
    previous_raw_end = 0
    for commit in commits:
        value = str(getattr(commit, "tts_text", "") or "")
        if not value:
            continue
        raw_start = max(0, min(len(source), int(getattr(commit, "raw_start", 0))))
        raw_end = max(
            raw_start,
            min(len(source), int(getattr(commit, "raw_end", raw_start))),
        )
        # Commits are append-only; tolerate a legacy/test double with a stale
        # coordinate by clamping rather than allowing a backwards map.
        raw_start = max(previous_raw_end, raw_start)
        raw_end = max(raw_start, raw_end)
        if not boundaries:
            boundaries.append(raw_start)
        else:
            boundaries[-1] = max(boundaries[-1], raw_start)
        source_slice = source[raw_start:raw_end]
        if len(value) == len(source_slice) and value == source_slice:
            # Literal commits retain character-level provenance (including a
            # leading whitespace run that the temporary spoken journal may
            # trim later).
            boundaries.extend(range(raw_start + 1, raw_end + 1))
        else:
            if len(value) > 1:
                boundaries.extend([raw_start] * (len(value) - 1))
            boundaries.append(raw_end)
        previous_raw_end = raw_end
        spoken_parts.append(value)

    spoken_raw = "".join(spoken_parts)
    if not spoken_raw:
        return "", [len(source)]

    # Apply the same whitespace/emoji contract used by streaming journal
    # appends.  Compose the temporary journal's boundaries back into the
    # original raw coordinate space.
    temporary = CanonicalTextJournal(
        _normalize_tts_text,
        strip_leading_whitespace=True,
    )
    temporary.append(spoken_raw)
    temporary.finish()
    normalized = temporary.trim_normalized()
    temporary_boundaries = temporary.normalized_to_raw
    composed: list[int] = []
    for boundary in temporary_boundaries:
        index = max(0, min(len(boundaries) - 1, int(boundary)))
        composed.append(boundaries[index])
    return normalized, composed


def _metric_int(value: Any) -> Optional[int]:
    """Parse numeric engine metadata while tolerating legacy string values."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_bool(value: Any) -> bool:
    """Parse bool-like engine metadata without treating ``"False"`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _resolve_tn_language(tn_language: Any, session_language: Any) -> str:
    """Resolve the language mode used by the causal TN controller.

    ``SessionConfig.language`` is also consumed by the acoustic model and must
    not force the TN backend to treat every span as that language.  Chinese
    sessions commonly contain URLs, email addresses, identifiers, and English
    words; those spans need the mixed-language resolver so their local spoken
    forms are selected correctly.  A caller can still opt into a uniform TN
    language through ``TextNormalizationConfig.language`` or the
    ``tn_language`` output-policy override.

    ``session_language`` remains an argument for compatibility with embedders
    that call this helper directly.  It is intentionally not consulted for TN
    routing: acoustic language and TN language are separate contracts.
    """

    raw_tn = str(tn_language or "mixed_zh_en").strip().lower().replace("_", "-")
    if raw_tn in {"", "auto", "unknown", "mixed", "mixed-zh-en", "zh-en"}:
        return "mixed_zh_en"
    if raw_tn in {"zh", "zh-cn", "zh-hans", "chinese"}:
        # ``zh`` is an explicit TN override only when it is supplied through
        # the TN config.  The dataclass default is mixed_zh_en and is handled
        # above, so this branch preserves the opt-in uniform route.
        return "zh"
    if raw_tn in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return "mixed_zh_en"


def _commitment_splitter_overrides(
    adapter: Any,
    *,
    base_ratio: float = 4.5,
    base_min_ratio: float = 1.0,
    base_max_ratio: float = 10.0,
) -> dict[str, Any]:
    """Read optional X2 capacity knobs without accepting arbitrary config.

    The policy can tune the existing Splitter FSM, but it cannot replace that
    FSM or provide a second text boundary stream. Invalid optional values are
    ignored as a group so the server's calibrated defaults remain active.
    """

    if adapter is None:
        return {}
    try:
        values = adapter.splitter_config()
    except Exception:
        logger.exception("Optional commitment splitter config failed")
        return {}
    if not isinstance(values, dict):
        return {}
    accepted: dict[str, Any] = {}
    float_fields = {
        "ema_ratio": lambda value: value >= 1.0,
        "ema_alpha": lambda value: 0.0 < value <= 1.0,
        "ema_overflow_alpha": lambda value: 0.0 < value <= 1.0,
        "ema_min_ratio": lambda value: value >= 1.0,
        "ema_max_ratio": lambda value: value >= 1.0,
        "l1_split_cap_ratio": lambda value: 0.0 < value <= 1.0,
        "l2_split_cap_ratio": lambda value: 0.0 < value <= 1.0,
        "l3_split_cap_ratio": lambda value: 0.0 < value <= 1.0,
    }
    try:
        for key, predicate in float_fields.items():
            if key not in values:
                continue
            if isinstance(values[key], bool):
                raise ValueError(f"invalid commitment splitter value: {key}")
            value = float(values[key])
            if not math.isfinite(value) or not predicate(value):
                raise ValueError(f"invalid commitment splitter value: {key}")
            accepted[key] = value
        if "safety_margin" in values:
            raw_margin = values["safety_margin"]
            if isinstance(raw_margin, bool) or (
                isinstance(raw_margin, float) and not raw_margin.is_integer()
            ):
                raise ValueError("invalid commitment splitter value: safety_margin")
            margin = int(raw_margin)
            if margin < 0:
                raise ValueError("invalid commitment splitter value: safety_margin")
            accepted["safety_margin"] = margin
        minimum = accepted.get("ema_min_ratio")
        maximum = accepted.get("ema_max_ratio")
        ratio = accepted.get("ema_ratio")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("commitment splitter EMA bounds are reversed")
        if ratio is not None:
            effective_minimum = (
                minimum if minimum is not None else float(base_min_ratio)
            )
            effective_maximum = (
                maximum if maximum is not None else float(base_max_ratio)
            )
            if ratio < effective_minimum:
                raise ValueError("commitment splitter EMA ratio is below minimum")
            if ratio > effective_maximum:
                raise ValueError("commitment splitter EMA ratio is above maximum")
        else:
            effective_minimum = (
                minimum if minimum is not None else float(base_min_ratio)
            )
            effective_maximum = (
                maximum if maximum is not None else float(base_max_ratio)
            )
            if not effective_minimum <= float(base_ratio) <= effective_maximum:
                raise ValueError("commitment splitter EMA bounds exclude baseline")
        caps = [accepted.get(key) for key in (
            "l1_split_cap_ratio", "l2_split_cap_ratio", "l3_split_cap_ratio"
        ) if key in accepted]
        if caps != sorted(caps):
            raise ValueError("commitment splitter punctuation caps are unordered")
    except (TypeError, ValueError, OverflowError):
        logger.warning("Ignoring invalid optional commitment splitter config")
        return {}
    return accepted


def _observe_spliter_ratio(
    session: Session,
    *,
    segment_idx: int,
    audio_steps: Any,
    text_tokens: Any,
    outcome: RatioOutcome,
) -> Optional[RatioObservation]:
    """Route typed backend feedback while preserving legacy test doubles."""
    spliter = session.spliter
    if spliter is None:
        return None
    steps = _metric_int(audio_steps) or 0
    tokens = _metric_int(text_tokens) or 0
    commitment = getattr(session, "commitment_adapter", None)
    if commitment is not None:
        commitment.observe_segment(
            audio_steps=steps,
            text_tokens=tokens,
            overflow=outcome is RatioOutcome.KV_OVERFLOW,
        )
    observer = getattr(spliter, "observe_segment", None)
    if callable(observer):
        observation = observer(
            steps,
            tokens,
            outcome=outcome,
            segment_idx=segment_idx,
        )
        LifecycleLogger.emit(
            session_id=session.session_id,
            phase="ratio_observation",
            segment_idx=segment_idx,
            session_level=session.config.observability_level,
            min_level=obs.ObsLevel.DEBUG,
            outcome=observation.outcome.value,
            observed_ratio=round(observation.observed_ratio, 4),
            duration_before=round(observation.duration_before, 4),
            duration_after=round(observation.duration_after, 4),
            safety_before=round(observation.safety_before, 4),
            safety_after=round(observation.safety_after, 4),
            duration_sample_accepted=observation.duration_sample_accepted,
            safety_tightened=observation.safety_tightened,
            audio_steps=observation.audio_steps,
            text_tokens=observation.text_tokens,
        )
        return observation

    # Compatibility for compact/legacy spliter doubles: retain the old clean
    # and overflow callback, while abort/retry failures remain skipped because
    # those doubles do not own the new safety controller.
    updater = getattr(spliter, "update_ratio", None)
    if (
        callable(updater)
        and steps > 0
        and tokens > 0
        and outcome in (RatioOutcome.CODEC_EOS, RatioOutcome.KV_OVERFLOW)
    ):
        updater(
            steps,
            tokens,
            overflow=(outcome is RatioOutcome.KV_OVERFLOW),
        )
    return None


class FrontendInterface:
    """Gateway-facing session and text interface."""

    def __init__(
        self,
        engine_inbox: asyncio.Queue,
        tokenizer: LightQwen3TTSTokenizer,
        *,
        max_sessions: int = 128,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 4.5,
        safety_ratio_initial: Optional[float] = None,
        max_concurrent_segments: int = 2,
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
        ema_min_observation_tokens: int = 8,
        safety_failure_multiplier: float = 1.25,
        safety_margin: int = 8,
        l1_split_cap_ratio: float = 0.70,
        l2_split_cap_ratio: float = 0.80,
        l3_split_cap_ratio: float = 0.90,
        guarded_delivery_default: bool = True,
        guarded_delivery_window_ms: int = 100,
        reorder_stall_timeout_ms: int = 10_000,
        engine_model_version: str = DEFAULT_ENGINE_MODEL_VERSION,
        version_query_text: str = VERSION_QUERY_TEXT,
        speech_state_capability: Optional[SpeechStateCapability] = None,
        cursor_plan_adapter_factory: Optional[Callable[[], Any]] = None,
        commitment_factory: Optional[Callable[[str, Any], Any]] = None,
        extensions: Any = None,
    ):
        self._dispatcher = Dispatcher(engine_inbox)
        self._tokenizer = tokenizer
        self._max_sessions = max_sessions
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._safety_ratio_initial = (
            ema_ratio if safety_ratio_initial is None else safety_ratio_initial
        )
        self._max_concurrent = max_concurrent_segments
        self._ema_alpha = ema_alpha
        self._ema_overflow_alpha = ema_overflow_alpha
        self._ema_min_ratio = ema_min_ratio
        self._ema_max_ratio = ema_max_ratio
        if max(float(ema_ratio), float(self._safety_ratio_initial)) > float(
            ema_max_ratio
        ):
            raise ValueError(
                "ema_max_ratio must cover both duration and safety initial ratios"
            )
        self._ema_min_observation_tokens = ema_min_observation_tokens
        self._safety_failure_multiplier = safety_failure_multiplier
        self._safety_margin = safety_margin
        self._l1_split_cap_ratio = l1_split_cap_ratio
        self._l2_split_cap_ratio = l2_split_cap_ratio
        self._l3_split_cap_ratio = l3_split_cap_ratio
        self._guarded_delivery_default = bool(guarded_delivery_default)
        self._guarded_delivery_window_ms = min(
            max(float(guarded_delivery_window_ms), 100.0), 10_000.0
        )
        try:
            reorder_timeout_ms = float(reorder_stall_timeout_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "reorder_stall_timeout_ms must be a finite nonnegative number"
            ) from exc
        if (
            reorder_timeout_ms < 0
            or not math.isfinite(reorder_timeout_ms)
        ):
            raise ValueError(
                "reorder_stall_timeout_ms must be a finite nonnegative number"
            )
        self._reorder_stall_timeout_sec = reorder_timeout_ms / 1000.0
        self._engine_model_version = str(engine_model_version).strip()
        if not self._engine_model_version:
            raise ValueError("engine_model_version must not be empty")
        self._version_query_text = normalize_version_query_text(version_query_text)
        # Session is asyncio-owned and may expose this read-only descriptor,
        # but never owns the backend's opaque state handle or payload.
        self._speech_state_capability = coerce_speech_state_capability(
            speech_state_capability
        )
        self._cursor_plan_adapter_factory = cursor_plan_adapter_factory
        self._commitment_factory = commitment_factory or getattr(
            extensions, "commitment_factory", None
        )

        self._sessions: Dict[str, Session] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._tn_tasks: Dict[str, asyncio.Task] = {}
        self._tn_locks: Dict[str, asyncio.Lock] = {}
        self._diagnostic_text_routers: Dict[str, DiagnosticTextRouter] = {}
        self._cursor_plan_adapters: Dict[str, Any] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def speech_state_capability(self) -> SpeechStateCapability:
        """Static backend capability copied into newly created sessions."""

        return self._speech_state_capability

    def count_text_tokens(self, text: str) -> int:
        """Count model input tokens using the synthesis tokenizer.

        Billing adapters must use the same normalization and tokenizer as the
        actual frontend.  Keeping this operation here avoids protocol layers
        guessing from characters or depending on tokenizer implementation
        details.
        """
        normalized = _normalize_tts_text(text).strip()
        if not normalized:
            return 0
        return len(self._encode_ids(normalized))

    async def create_session(
        self,
        session_id: str,
        *,
        config: Optional[SessionConfig] = None,
        speaker_key: Optional[str] = None,
        task_type: str = "custom",
        ref_audio: Optional[bytes] = None,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
    ) -> Session:
        if session_id in self._sessions:
            await self.cancel_session(session_id)

        if len(self._sessions) >= self._max_sessions:
            LifecycleLogger.emit(
                session_id=session_id,
                phase="session.admission_rejected",
                request_id=(
                    config.timing.request_id
                    if config is not None and config.timing.request_id
                    else None
                ),
                reason="max_sessions",
                max_sessions=self._max_sessions,
                active_sessions=len(self._sessions),
            )
            raise SessionCapacityError(self._max_sessions)

        if config is None:
            config = SessionConfig(
                task_type=task_type,
                speaker=speaker_key,
                ref_audio=ref_audio,
            )
        else:
            # Resolve per-session TN overrides on a private dataclass copy.
            # Callers commonly reuse a ``SessionConfig`` template; mutating
            # its nested normalization object here would leak one session's
            # rollout/language choice into the next session.
            config = replace(
                config,
                text_normalization=replace(config.text_normalization),
            )
        # Resolve the free-form per-session TN overrides without mutating the
        # caller's nested config object.  ``text_commitment.types`` exposes a
        # frozen config for direct committer users; accepting that object via
        # ``SessionConfig`` must remain safe as well.  Building one replaced
        # value also keeps the effective settings visible in ``session.config``.
        tn_cfg = config.text_normalization
        opts = config.output_policy.config or {}
        tn_overrides: dict[str, Any] = {}
        if "tn_enabled" in opts:
            tn_overrides["enabled"] = _metric_bool(opts["tn_enabled"])
        if "tn_semantic_max_wait_ms" in opts:
            tn_overrides["semantic_max_wait_ms"] = float(
                opts["tn_semantic_max_wait_ms"]
            )
        if "tn_semantic_idle_wait_ms" in opts:
            tn_overrides["semantic_idle_wait_ms"] = float(
                opts["tn_semantic_idle_wait_ms"]
            )
        if "tn_fallback" in opts:
            tn_overrides["fallback"] = str(opts["tn_fallback"])
        if "tn_projection" in opts:
            tn_overrides["projection"] = str(opts["tn_projection"])
        if "tn_language" in opts:
            tn_overrides["language"] = str(opts["tn_language"])
        if "tn_commit_mode" in opts:
            tn_overrides["commit_mode"] = str(opts["tn_commit_mode"])
        if "tn_candidate_nbest" in opts:
            tn_overrides["candidate_nbest"] = int(opts["tn_candidate_nbest"])
        if "tn_margin_threshold" in opts:
            tn_overrides["margin_threshold"] = float(opts["tn_margin_threshold"])
        if "tn_calibration_profile" in opts:
            tn_overrides["calibration_profile"] = str(opts["tn_calibration_profile"])
        if "tn_ambiguity_policy" in opts:
            tn_overrides["ambiguity_policy"] = str(opts["tn_ambiguity_policy"])
        if tn_overrides:
            tn_cfg = replace(tn_cfg, **tn_overrides)
            config = replace(config, text_normalization=tn_cfg)

        self._prepare_session_config(config)

        session = Session(
            session_id=session_id,
            config=config,
            speech_state_capability=self._speech_state_capability,
        )
        if self._commitment_factory is not None:
            try:
                policy = self._commitment_factory(session_id, config)
                if policy is not None:
                    session.commitment_adapter = X2CommitmentAdapter(policy)
                    session.commitment_adapter.bind_engine_budget(
                        self._engine_max - self._prefill_len
                    )
            except Exception:
                # Optional method-layer policy must never make the main TN or
                # session lifecycle unavailable.
                logger.exception(
                    "Disabling optional commitment policy for session %s",
                    session_id,
                )
        # An explicit model/session language is useful evidence for TN too.
        # Keep ``auto``/unknown as mixed so a numeric-only island still waits
        # instead of inheriting an accidental default graph.
        effective_tn_language = _resolve_tn_language(
            tn_cfg.language,
            config.language,
        )
        try:
            fallback_policy = FallbackPolicy(tn_cfg.fallback)
        except ValueError:
            fallback_policy = FallbackPolicy.CARDINAL_OR_LITERAL
        session.text_committer = IncrementalTextCommitter(
            CommitterConfig(
                enabled=tn_cfg.enabled,
                language=effective_tn_language,
                semantic_max_wait_ms=tn_cfg.semantic_max_wait_ms,
                semantic_idle_wait_ms=tn_cfg.semantic_idle_wait_ms,
                fallback=fallback_policy,
                projection=tn_cfg.projection,
                max_pending_chars=tn_cfg.max_pending_chars,
                commit_mode=getattr(tn_cfg, "commit_mode", "closed_span"),
                candidate_nbest=getattr(tn_cfg, "candidate_nbest", 8),
                calibration_profile=getattr(tn_cfg, "calibration_profile", ""),
                ambiguity_policy=getattr(tn_cfg, "ambiguity_policy", "wait"),
                margin_threshold=getattr(tn_cfg, "margin_threshold", None),
                family_margin_thresholds=getattr(tn_cfg, "family_margin_thresholds", ()),
            )
        )
        session.audio_credit_estimator = AudioCreditEstimator(codec_frame_rate=12.5)
        session.text_journal = CanonicalTextJournal(
            _normalize_tts_text,
            strip_leading_whitespace=True,
        )
        if self._cursor_plan_adapter_factory is not None:
            adapter = self._cursor_plan_adapter_factory()
            if adapter is None:
                raise ValueError("cursor_plan_adapter_factory returned None")
            self._cursor_plan_adapters[session_id] = adapter
        commitment_overrides = _commitment_splitter_overrides(
            session.commitment_adapter,
            base_ratio=self._ema_ratio,
            base_min_ratio=self._ema_min_ratio,
            base_max_ratio=self._ema_max_ratio,
        )
        if "ema_ratio" in commitment_overrides:
            # X2's capacity ratio is the same conservative planning quantity
            # consumed by the existing Splitter. Keep duration and safety
            # baselines aligned for an explicitly injected method policy.
            ema_ratio = commitment_overrides["ema_ratio"]
            safety_ratio_initial = ema_ratio
        else:
            ema_ratio = self._ema_ratio
            safety_ratio_initial = self._safety_ratio_initial
        splitter_kwargs = dict(
            engine_max_decode_len=self._engine_max,
            prefill_len=self._prefill_len,
            ema_ratio=ema_ratio,
            safety_ratio_initial=safety_ratio_initial,
            max_concurrent=self._max_concurrent,
            ema_alpha=commitment_overrides.get("ema_alpha", self._ema_alpha),
            ema_overflow_alpha=commitment_overrides.get(
                "ema_overflow_alpha", self._ema_overflow_alpha
            ),
            ema_min_ratio=commitment_overrides.get(
                "ema_min_ratio", self._ema_min_ratio
            ),
            ema_max_ratio=commitment_overrides.get(
                "ema_max_ratio", self._ema_max_ratio
            ),
            ema_min_observation_tokens=self._ema_min_observation_tokens,
            safety_failure_multiplier=self._safety_failure_multiplier,
            safety_margin=commitment_overrides.get(
                "safety_margin", self._safety_margin
            ),
            l1_split_cap_ratio=commitment_overrides.get(
                "l1_split_cap_ratio", self._l1_split_cap_ratio
            ),
            l2_split_cap_ratio=commitment_overrides.get(
                "l2_split_cap_ratio", self._l2_split_cap_ratio
            ),
            l3_split_cap_ratio=commitment_overrides.get(
                "l3_split_cap_ratio", self._l3_split_cap_ratio
            ),
        )
        session.spliter = Spliter(
            **splitter_kwargs,
        )
        reorder_timeout_sec = self._reorder_stall_timeout_sec
        if "reorder_stall_timeout_ms" in opts:
            try:
                configured_timeout_ms = float(opts["reorder_stall_timeout_ms"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "reorder_stall_timeout_ms must be a finite nonnegative number"
                ) from exc
            if configured_timeout_ms < 0 or not math.isfinite(configured_timeout_ms):
                raise ValueError(
                    "reorder_stall_timeout_ms must be a finite nonnegative number"
                )
            reorder_timeout_sec = configured_timeout_ms / 1000.0
        session.reorder = AudioReorder(stall_timeout_sec=reorder_timeout_sec)
        session.event_callback = on_event
        self._sessions[session_id] = session
        self._tn_locks[session_id] = asyncio.Lock()
        diagnostic_router = DiagnosticTextRouter(
            version_query_text=self._version_query_text
        )
        self._diagnostic_text_routers[session_id] = diagnostic_router
        # Keep a read-only-by-convention reference on the session so the
        # static raw-source helper can include a held exact-query prefix
        # without coupling Session to the frontend router registry.
        session._diagnostic_text_router = diagnostic_router

        # Resolve per-session observability level (raise-only override of the
        # global floor, clamped to max_session_level — see observability_tiers §3).
        requested_level = config.output_policy.config.get(
            "obs_level"
        ) or config.timing.extra.get("obs_level")
        obs_level_clamped = False
        if requested_level is not None:
            effective, obs_level_clamped = obs.resolve_session_level(requested_level)
            config.observability_level = effective
        session_level = config.observability_level
        active_level = (
            session_level if session_level is not None else obs.global_level()
        )
        session.spliter.enable_decision_recording(active_level >= obs.ObsLevel.DEBUG)

        # Emit lifecycle events
        LifecycleLogger.emit(
            session_id=session_id,
            phase="session.config.validated",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            session_level=session_level,
            input_mode=config.input_mode.value,
            group_policy=config.group_policy.value,
            task_type=config.task_type or None,
            vad_strategy=config.output_policy.vad.strategy or "disabled",
            protocol_version=str(
                config.timing.extra.get("client_protocol_version", "")
            ),
            obs_level=active_level.name.lower(),
            **(
                {"obs_level_clamped": True, "obs_level_requested": str(requested_level)}
                if obs_level_clamped
                else {}
            ),
        )
        LifecycleLogger.emit(
            session_id=session_id,
            phase="session.created",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            monotonic_ts=session.created_at,
            speaker_id=config.speaker,
        )

        # Set session_created on ServerTimingAccumulator if present
        acc = config.timing.extra.get("_server_timing_accumulator")
        if isinstance(acc, ServerTimingAccumulator):
            acc.session_created_monotonic = session.created_at

        task = asyncio.create_task(
            self._consume_results(
                session, on_audio=on_audio, on_done=on_done, on_event=on_event
            )
        )
        self._consumer_tasks[session_id] = task
        if session.text_committer.config.enabled:
            self._tn_tasks[session_id] = asyncio.create_task(self._text_commit_ticker(session))

        await self._dispatcher.submit_new_session(session)
        logger.info("Session %s created (active: %d)", session_id, self.active_count)
        return session

    async def cancel_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.state = SessionState.DONE
        await self._dispatcher.submit_cancel(session_id)
        task = self._consumer_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._cleanup_session(session_id)

    async def push_text_input(
        self,
        session_id: str,
        text: str,
        *,
        metadata: TextInputMetadata | None = None,
    ) -> None:
        """Feed transport text according to the session's declared input mode."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE or session.input_complete:
            return
        lock = self._tn_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tn_locks[session_id] = lock
        async with lock:
            await self._push_text_input_locked(session, text, metadata=metadata)

    def observe_raw_token_arrival(self, session_id: str, count: int = 1) -> None:
        """Record upstream LLM-token arrivals for Phase-1 rate estimation.

        The transport payload is text and cannot reliably reconstruct the LLM's
        tokenizer, so adapters that know the upstream token count should call
        this explicitly.  It is intentionally separate from ``push_text_input``.
        """
        session = self._sessions.get(session_id)
        if session is not None and session.audio_credit_estimator is not None:
            session.audio_credit_estimator.observe_raw_tokens(count)

    def observe_played_audio(self, session_id: str, elapsed_ms: float) -> None:
        """Feed a gateway/playhead playback estimate into audio credit."""
        session = self._sessions.get(session_id)
        if session is not None and session.audio_credit_estimator is not None:
            session.audio_credit_estimator.observe_played_audio(elapsed_ms)

    def streaming_rate_metrics(self, session_id: str) -> Optional[dict[str, float | int]]:
        """Return a transport-neutral Phase-1 metrics snapshot."""
        session = self._sessions.get(session_id)
        if session is None or session.audio_credit_estimator is None:
            return None
        snapshot = session.audio_credit_estimator.snapshot()
        return {
            "raw_tokens": snapshot.raw_tokens,
            "normalized_tokens": snapshot.normalized_tokens,
            "codec_frames": snapshot.codec_frames,
            "lambda_raw": snapshot.lambda_raw,
            "lambda_norm": snapshot.lambda_norm,
            "t_service_ms": snapshot.t_service_ms,
            "audio_credit_ms": snapshot.audio_credit_ms,
            "safe_wait_ms": snapshot.safe_wait_ms,
        }

    @staticmethod
    def _tn_raw_source(session: "Session") -> str:
        """Return the complete raw source, including held transport suffixes.

        The diagnostic router may withhold an exact-query prefix for one or
        more packets.  It is still part of the source coordinate space even
        though it has not entered the committer yet; likewise an incomplete
        emoji grapheme is retained until the next packet.  Keeping both here
        makes raw offsets and operational logs lossless without releasing
        provisional text to TTS.
        """

        committer = session.text_committer
        router = getattr(session, "_diagnostic_text_router", None)
        if router is None:
            routers = getattr(session, "_diagnostic_text_routers", None)
            if isinstance(routers, dict):
                router = routers.get(session.session_id)
        pending_query = str(getattr(router, "pending_text", "") or "")
        return (
            str(getattr(committer, "raw_text", "") or "")
            + pending_query
            + str(getattr(session, "_emoji_carry", "") or "")
        )

    async def _push_text_input_locked(
        self,
        session: "Session",
        text: str,
        *,
        metadata: TextInputMetadata | None = None,
    ) -> None:
        mode = session.config.input_mode
        # A timeout is a semantic commit decision, not a tokenizer concern.
        timeout_decision = session.text_committer.poll()
        if timeout_decision.commits:
            self._log_tn_commits(session, timeout_decision.commits)
            await self._ingest_commits(session, timeout_decision.commits)
        if mode != InputMode.FULL_TEXT:
            session.text_journal.update_raw_source(self._tn_raw_source(session))
        self._emit_text_commit_events(session, timeout_decision.events)
        if mode == InputMode.FULL_TEXT:
            # Keep the journal canonical while deferring all tokenization until
            # the single final committer flush.
            session.text_journal.append(text or "")
            session.append_text(text or "")
            return

        # Streaming modes: hold back a trailing partial-emoji suffix so an emoji
        # split across packets (e.g. a keycap base) does not leak into speech.
        raw = session._emoji_carry + (text or "")
        body, session._emoji_carry = split_pending_emoji(raw)
        router = self._diagnostic_text_routers[session.session_id]
        for routed_text in router.push(body):
            # Keep the original source delta for the committer.  It performs
            # grapheme/emoji filtering itself so TextCommit raw offsets remain
            # coordinates in the transport input; stripping here would shift
            # every span that follows a deleted emoji.
            decision = session.text_committer.feed(
                routed_text,
                metadata=metadata,
            )
            self._log_tn_commits(session, decision.commits)
            self._log_tn_pending(session, decision)
            await self._ingest_commits(session, decision.commits)
            session.text_journal.update_raw_source(self._tn_raw_source(session))
            self._emit_text_commit_events(session, decision.events)

    async def _text_commit_ticker(self, session: "Session") -> None:
        """Wake on semantic deadlines even when the upstream stalls."""
        try:
            while session.state != SessionState.DONE and not session.input_complete:
                deadline = session.text_committer.next_deadline
                if deadline is None:
                    await asyncio.sleep(0.02)
                    continue
                await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))
                # The upstream may have finalized the session while this task
                # was asleep.  In that case the journal is intentionally
                # closed and must not receive a late raw-source refresh.
                if session.state == SessionState.DONE or session.input_complete:
                    return
                lock = self._tn_locks.get(session.session_id)
                if lock is None:
                    return
                async with lock:
                    if session.state == SessionState.DONE or session.input_complete:
                        return
                    decision = session.text_committer.poll()
                    if decision.commits:
                        self._log_tn_commits(session, decision.commits)
                        await self._ingest_commits(session, decision.commits)
                    session.text_journal.update_raw_source(self._tn_raw_source(session))
                    self._log_tn_pending(session, decision)
                    self._emit_text_commit_events(session, decision.events)
        except asyncio.CancelledError:
            return

    async def _ingest_commits(self, session: "Session", commits) -> None:
        # Build the cursor revision before dispatching any spoken tokens from
        # this commit batch.  The engine queue is the happens-before boundary:
        # UPDATE_CURSOR_PLAN must precede START/APPEND_TOKENS so the first
        # decode step cannot observe a stale label buffer.
        pending_spoken: list[tuple[str, int, Any]] = []
        for commit in commits:
            # Keep the journal's source coordinate space in the raw upstream
            # text.  Feeding ``commit.tts_text`` through the ordinary append
            # path would make an expansion such as ``99%`` → ``百分之九十九``
            # look like raw input and would invalidate progress anchors.
            spoken, normalized_base = session.text_journal.append_spoken(
                session.text_committer.raw_text,
                commit.raw_start,
                commit.raw_end,
                commit.tts_text,
                mapping=commit.mapping,
            )
            session.cursor_commits.append(commit)
            session.cursor_spoken_texts.append(spoken)
            observation = self._observe_commitment(session, commit, spoken)
            pending_spoken.append((spoken, normalized_base, observation))

        if commits:
            await self._refresh_cursor_plan(session)

        for spoken, normalized_base, observation in pending_spoken:
            await self._ingest_streaming_text(
                session,
                spoken,
                journaled=True,
                normalized_base=normalized_base,
                force_boundary=bool(
                    observation is not None
                    and getattr(observation, "force_boundary", False)
                ),
                force_boundary_before=bool(
                    observation is not None
                    and getattr(observation, "force_boundary_before", False)
                ),
            )

    def _observe_commitment(
        self,
        session: "Session",
        commit: Any,
        spoken_text: str,
    ) -> Any:
        """Forward main-TN evidence to the optional X2 policy bridge."""

        adapter = getattr(session, "commitment_adapter", None)
        if adapter is None or adapter.disabled:
            return None
        try:
            observation = adapter.consume_commit(
                commit,
                spoken_text=spoken_text,
                token_count=len(self._encode_ids(spoken_text)),
                boundary_level=Spliter.classify_punct_level(spoken_text),
            )
            if not observation.accepted:
                LifecycleLogger.emit(
                    session_id=session.session_id,
                    phase="x2.commitment_disabled",
                    reason=observation.reason,
                    commit_fence=observation.fence,
                )
            return observation
        except Exception:
            # The adapter itself is fail-closed; this guard protects legacy
            # custom adapters injected by embedders.
            logger.exception(
                "Optional commitment observation failed for session %s",
                session.session_id,
            )
            return None

    @staticmethod
    def _finish_commitment(session: "Session") -> None:
        adapter = getattr(session, "commitment_adapter", None)
        if adapter is None:
            return
        try:
            adapter.finish()
        except Exception:
            logger.exception(
                "Optional commitment finish failed for session %s",
                session.session_id,
            )

    async def _refresh_cursor_plan(
        self,
        session: "Session",
        *,
        final: bool = False,
        spoken_texts: Optional[list[str]] = None,
    ) -> None:
        """Build and publish a TN-derived plan when a labelizer is injected."""
        if getattr(session, "native_cursor_disabled", False):
            return
        adapter = self._cursor_plan_adapters.get(session.session_id)
        if adapter is None:
            return
        revision = session.cursor_plan_revision + 1
        previous = session.cursor_label_plan
        # TextCommit records are append-only once emitted by the primary TN.
        # Protect the complete plan already published to the engine: a rebuilt
        # plan may append labels, but must never rewrite its committed prefix.
        committed_label_count = 0 if previous is None else previous.label_count
        try:
            plan = adapter.build(
                session.cursor_commits,
                spoken_texts=(
                    session.cursor_spoken_texts
                    if spoken_texts is None
                    else spoken_texts
                ),
                revision=revision,
                final=final,
                previous=previous,
                committed_label_count=committed_label_count,
            )
        except Exception as exc:
            # Native cursor is an optional progress route.  A malformed
            # label/plan must never abort text ingestion or acoustic state;
            # freeze this route for the session and let the shared publisher
            # continue with its conservative EMA estimator.
            self._disable_native_cursor(session, type(exc).__name__)
            logger.warning(
                "Native cursor disabled for session %s; falling back to EMA: %s",
                session.session_id,
                exc,
            )
            self._cursor_plan_adapters.pop(session.session_id, None)
            return
        session.cursor_plan_revision = revision
        session.cursor_label_plan = plan
        await self._dispatcher.submit_cursor_plan(session)

    @staticmethod
    def _disable_native_cursor(session: "Session", reason: str) -> None:
        """Disable only the optional progress route and keep the session alive."""
        if getattr(session, "native_cursor_disabled", False):
            return
        session.native_cursor_disabled = True
        session.native_cursor_fallback_reason = reason
        plans = getattr(session, "cursor_segment_plans", None)
        if plans is not None:
            plans.clear()
        config = getattr(session, "config", None)
        timing = getattr(config, "timing", None)
        LifecycleLogger.emit(
            session_id=session.session_id,
            phase="progress.native_fallback",
            request_id=getattr(timing, "request_id", None) or None,
            turn_id=getattr(timing, "turn_id", None) or None,
            session_level=getattr(
                config,
                "observability_level",
                getattr(session, "observability_level", None),
            ),
            route="ema",
            reason=reason,
        )

    def _emit_text_commit_events(self, session: "Session", events: tuple[str, ...]) -> None:
        callback = getattr(session, "event_callback", None)
        if not callable(callback):
            callback = None
        for event in events:
            LifecycleLogger.emit(
                session_id=session.session_id,
                phase=event,
                request_id=session.config.timing.request_id or None,
                turn_id=session.config.timing.turn_id or None,
                session_level=session.config.observability_level,
            )
            if callback is None:
                continue
            try:
                result = callback(session.session_id, {"type": event})
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                logger.exception("text commit event callback failed")

    def _log_tn_commits(self, session: "Session", commits) -> None:
        """Record raw→spoken TN decisions under the text privacy policy."""
        for commit in commits:
            LifecycleLogger.emit(
                session_id=session.session_id,
                phase="text.tn_commit",
                request_id=session.config.timing.request_id or None,
                turn_id=session.config.timing.turn_id or None,
                session_level=session.config.observability_level,
                raw_start=commit.raw_start,
                raw_end=commit.raw_end,
                raw_text=obs.text_preview(commit.raw_text),
                spoken_text=obs.text_preview(commit.tts_text),
                span_kind=commit.span_kind.value,
                commit_kind=commit.commit_kind.value,
                language=commit.language.value,
                commit_fence=commit.fence,
                normalization_changed=commit.raw_text != commit.tts_text,
                semantic_family=commit.semantic_family.value,
                decision_source=commit.decision_source,
                candidate_count=commit.candidate_count,
                best_cost=commit.best_cost,
                cost_margin=commit.cost_margin,
                calibrated_confidence=commit.calibrated_confidence,
                closure_reason=commit.closure_reason,
                fallback_reason=commit.fallback_reason,
            )

    def _log_tn_pending(self, session: "Session", decision) -> None:
        if not decision.pending_raw:
            return
        LifecycleLogger.emit(
            session_id=session.session_id,
            phase="text.tn_pending",
            request_id=session.config.timing.request_id or None,
            turn_id=session.config.timing.turn_id or None,
            session_level=session.config.observability_level,
            min_level=obs.ObsLevel.DEBUG,
            pending_raw=obs.text_preview(decision.pending_raw),
            pending_kind=decision.pending_kind.value if decision.pending_kind else None,
            reason=decision.reason,
        )

    async def _ingest_streaming_text(
        self,
        session: "Session",
        body: str,
        *,
        journaled: bool = False,
        normalized_base: int | None = None,
        force_boundary: bool = False,
        force_boundary_before: bool = False,
    ) -> None:
        """Normalize a streaming text body, tokenize, route to the spliter per
        input mode, and dispatch. Shared by push_text_input and the end-of-input
        emoji-carry flush."""
        if journaled:
            text = body or ""
            normalized_base = (
                len(session.text_journal.normalized_text) - len(text)
                if normalized_base is None
                else normalized_base
            )
        else:
            text, normalized_base = session.text_journal.append(body or "")
        if not text:
            return
        mode = session.config.input_mode
        if mode != InputMode.TOKEN and not text.strip():
            return
        if not getattr(session, "_first_text_sent", False):
            session._first_text_sent = True
            LifecycleLogger.emit(
                session_id=session.session_id,
                phase="text.first_sent",
                request_id=session.config.timing.request_id or None,
                session_level=session.config.observability_level,
                text_length=len(text),
                normalized_preview=obs.text_preview(text),
            )

        tokens = self._tokenize_segment_text(
            text,
            normalized_offset=normalized_base,
            journal=session.text_journal,
        )
        if not tokens:
            return

        spliter: Spliter = session.spliter
        if mode == InputMode.AUTO:
            seg_actions = spliter.feed_auto(
                tokens,
                force_boundary=force_boundary,
                force_boundary_before=force_boundary_before,
            )
        elif (
            mode == InputMode.LONG_SEGMENT
            and session.config.group_policy != GroupPolicy.NONE
        ):
            seg_actions = spliter.push_group_tokens(
                tokens,
                force_boundary=force_boundary,
                force_boundary_before=force_boundary_before,
            )
        else:
            seg_actions = spliter.feed_tokens(
                tokens,
                force_boundary=force_boundary,
                force_boundary_before=force_boundary_before,
            )
        await self._dispatch_segment_actions(session, seg_actions)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Explicit offline mode: set complete text, pre-split, drive all segments."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE or session.input_complete:
            return

        lock = self._tn_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tn_locks[session_id] = lock
        async with lock:
            await self._feed_full_text_locked(session, text)

    async def _feed_full_text_locked(self, session: "Session", text: str) -> None:
        """Locked implementation shared by direct FULL_TEXT and EOS flush."""
        if session.state == SessionState.DONE:
            return

        # ``feed_full_text`` is also used by the FULL_TEXT transport path after
        # it has accumulated packets in ``Session._text_buffer``.  Prefer the
        # explicit complete argument when it already contains the buffered
        # prefix.  Otherwise retain both pieces instead of silently discarding
        # transport data.  Never feed ``normalized_text`` back into the
        # committer: that value is the pre-TN canonical projection and may have
        # already dropped markup/emoji or collapsed whitespace.
        explicit_text = text or ""
        buffered_text = session.drain_text()
        if explicit_text and buffered_text:
            if explicit_text.startswith(buffered_text):
                raw_text = explicit_text
            elif buffered_text.startswith(explicit_text):
                raw_text = buffered_text
            else:
                raw_text = buffered_text + explicit_text
        else:
            raw_text = explicit_text or buffered_text
        if not raw_text and session.text_journal is not None:
            raw_text = session.text_journal.raw_text

        # Resolve only the exact diagnostic trigger before the normal TN
        # committer.  The replacement is an input to TN, never a post-TN
        # literal journal or tokenizer shortcut.
        tn_input = resolve_diagnostic_text(
            raw_text,
            self._engine_model_version,
            self._version_query_text,
        )
        decision = session.text_committer.feed(tn_input, final=True)
        self._log_tn_commits(session, decision.commits)
        self._log_tn_pending(session, decision)
        self._emit_text_commit_events(session, decision.events)

        # Install the TN result as a completed projection while preserving the
        # original raw coordinate space.  ``CanonicalTextJournal.append`` is
        # intentionally unable to do this because a TN expansion is not an
        # old-result prefix.
        text, spoken_to_raw = _spoken_projection(tn_input, decision.commits)
        if tn_input != raw_text:
            # The exact trigger is a control phrase, not the source of the
            # diagnostic sentence. Keep the trigger as the public raw source
            # and conservatively anchor the generated spoken payload at its
            # end; never publish synthetic coordinates outside that source.
            spoken_to_raw = [min(boundary, len(raw_text)) for boundary in spoken_to_raw]
        if session.text_journal is None:
            session.text_journal = CanonicalTextJournal(
                _normalize_tts_text,
                strip_leading_whitespace=True,
            )
        session.text_journal.install_projection(
            raw_text,
            text,
            spoken_to_raw,
        )
        full_commitment_tokens: list[SegmentToken] = []
        full_spoken_parts: list[str] = []
        full_spoken_offset = 0
        full_boundary_indices: list[int] = []
        full_boundary_before_indices: list[int] = []
        if decision.commits:
            session.cursor_commits.extend(decision.commits)
            session.cursor_spoken_texts.extend(
                _normalize_tts_text(commit.tts_text) for commit in decision.commits
            )
            for commit, spoken in zip(
                decision.commits,
                session.cursor_spoken_texts[-len(decision.commits):],
            ):
                observation = self._observe_commitment(session, commit, spoken)
                if observation is None:
                    continue
                commit_tokens = self._tokenize_segment_text(
                    spoken,
                    normalized_offset=full_spoken_offset,
                    journal=session.text_journal,
                )
                full_token_offset = len(full_commitment_tokens)
                commit_token_count = len(commit_tokens)
                if observation.accepted and commit_token_count:
                    if observation.force_boundary_before:
                        full_boundary_before_indices.append(full_token_offset)
                    if observation.force_boundary:
                        full_boundary_indices.append(
                            full_token_offset + commit_token_count - 1
                        )
                full_commitment_tokens.extend(commit_tokens)
                full_spoken_parts.append(spoken)
                full_spoken_offset += len(spoken)
        session.mark_input_complete()
        self._finish_commitment(session)

        if session.cursor_commits:
            await self._refresh_cursor_plan(session, final=True)
        if not text:
            # Whitespace/format-only FULL_TEXT input still needs an explicit
            # session-level completion after the committer has flushed it.
            await self._dispatcher.maybe_send_session_tokens_done(session)
            return

        adapter = session.commitment_adapter
        if (
            adapter is not None
            and not adapter.disabled
            and "".join(full_spoken_parts) == text
        ):
            # Preserve exact commit boundaries even when whole-text BPE would
            # merge across them. Mapping offsets remain codepoint coordinates.
            tokens = full_commitment_tokens
        else:
            tokens = self._tokenize_segment_text(
                text, normalized_offset=0, journal=session.text_journal
            )
            full_boundary_indices = []
            full_boundary_before_indices = []
        if not tokens:
            await self._dispatcher.maybe_send_session_tokens_done(session)
            return

        seg_actions = session.spliter.set_full_text(
            tokens,
            force_boundary_indices=full_boundary_indices,
            force_boundary_before_indices=full_boundary_before_indices,
        )
        await self._dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.maybe_send_session_tokens_done(session)

    async def mark_input_complete(self, session_id: str) -> None:
        """Upstream signals no more transport input will arrive."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        lock = self._tn_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tn_locks[session_id] = lock
        async with lock:
            await self._mark_input_complete_locked(session)

    async def _mark_input_complete_locked(self, session: "Session") -> None:
        """Flush the causal committer and close the text stream under its lock."""
        if session.input_complete:
            return
        mode = session.config.input_mode
        if mode == InputMode.FULL_TEXT:
            full_text = session.drain_text()
            # The text buffer is the transport's raw source of truth.  If a
            # legacy caller populated only the journal, use its raw history as
            # a fallback; never use ``normalized_text`` here because it is
            # pre-TN and may have discarded syntax needed for final routing.
            if not full_text and session.text_journal is not None:
                full_text = session.text_journal.raw_text
            await self._feed_full_text_locked(session, full_text)
            return

        query_matched = False
        if mode != InputMode.FULL_TEXT:
            router = self._diagnostic_text_routers[session.session_id]
            resolution = router.finish(self._engine_model_version)
            query_matched = resolution.query_matched
            if resolution.query_matched:
                # The diagnostic payload is still ordinary spoken text.  Feed
                # it back through the main committer so version numbers and
                # punctuation receive the same TN treatment as user input.
                # The router only recognizes the raw trigger; it must never
                # become a second normalization path.
                session._emoji_carry = ""
                for routed_text in resolution.chunks:
                    decision = session.text_committer.feed(routed_text)
                    self._log_tn_commits(session, decision.commits)
                    self._log_tn_pending(session, decision)
                    await self._ingest_commits(session, decision.commits)
                    session.text_journal.update_raw_source(self._tn_raw_source(session))
                    self._emit_text_commit_events(session, decision.events)
            else:
                for routed_text in resolution.chunks:
                    raw = session._emoji_carry + routed_text
                    body, session._emoji_carry = split_pending_emoji(raw)
                    if not body:
                        continue
                    decision = session.text_committer.feed(body)
                    self._log_tn_commits(session, decision.commits)
                    self._log_tn_pending(session, decision)
                    await self._ingest_commits(session, decision.commits)
                    session.text_journal.update_raw_source(self._tn_raw_source(session))
                    self._emit_text_commit_events(session, decision.events)
        # Flush any held partial-emoji carry as final streaming text before
        # asking the committer for its final decision.  A held keycap base with
        # no modifier is a normal digit and must still pass through it.
        if session._emoji_carry:
            body = session._emoji_carry
            session._emoji_carry = ""
            decision = session.text_committer.feed(body)
            self._log_tn_commits(session, decision.commits)
            self._log_tn_pending(session, decision)
            await self._ingest_commits(session, decision.commits)
            session.text_journal.update_raw_source(self._tn_raw_source(session))
            self._emit_text_commit_events(session, decision.events)

        final_decision = session.text_committer.feed("", final=True)
        self._log_tn_commits(session, final_decision.commits)
        self._log_tn_pending(session, final_decision)
        await self._ingest_commits(session, final_decision.commits)
        if session.cursor_label_plan is not None:
            await self._refresh_cursor_plan(session, final=True)
        if not query_matched:
            session.text_journal.update_raw_source(self._tn_raw_source(session))
        self._emit_text_commit_events(session, final_decision.events)

        session.mark_input_complete()
        self._finish_commitment(session)
        if session.text_journal is not None:
            # Streaming modes have already installed their raw→canonical
            # projection incrementally.  FULL_TEXT is handled below by the
            # final TN flush, so finishing this pre-TN journal here would make
            # us accidentally feed the old canonical snapshot to TN.
            if mode != InputMode.FULL_TEXT:
                session.text_journal.finalize_projection()

        if (
            mode == InputMode.LONG_SEGMENT
            and session.config.group_policy != GroupPolicy.NONE
        ):
            await self._dispatcher.maybe_send_session_tokens_done(session)
            return

        seg_actions = session.spliter.input_done()
        await self._dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.maybe_send_session_tokens_done(session)

    def _guarded_hold_for(self, session: Session) -> Optional[DeliveryHoldWindow]:
        """Build the guarded-delivery hold window for this session.

        Guarded delivery follows the server default and can be overridden by
        the free-form ``output_policy.config.delivery`` value (``guarded`` or
        ``firehose``). ``delivery_window_ms`` controls allowed client lead.
        It does not add an initial holdback: the first chunk is released at
        once. As synthesis outruns playback, only the excess beyond the
        estimated playhead plus this lead remains retractable server-side.
        Engine audio is float32 mono at ENGINE_SAMPLE_RATE regardless of the
        client's output format (the gateway converts downstream of this
        hold)."""
        cfg = session.config.output_policy.config or {}
        default_mode = "guarded" if self._guarded_delivery_default else "firehose"
        if str(cfg.get("delivery", default_mode)).strip().lower() != "guarded":
            return None
        try:
            window_ms = float(
                cfg.get("delivery_window_ms", self._guarded_delivery_window_ms)
                or self._guarded_delivery_window_ms
            )
        except (TypeError, ValueError):
            window_ms = self._guarded_delivery_window_ms
        window_ms = min(max(window_ms, 100.0), 10_000.0)
        return DeliveryHoldWindow(
            window_ms / 1000.0,
            ENGINE_SAMPLE_RATE * 4,
        )

    def _semantic_start_gate_for(self, session: Session) -> Optional[SemanticStartGate]:
        """Resolve the optional Phase-1 initial semantic release gate."""
        # Prefix VAD must inspect its raw onset at synthesis speed; delaying it
        # behind a semantic gate would change the existing VAD contract.
        if isinstance(session.config.timing.extra.get("_prefix_gate_guard_bypass"), PrefixGateGuardBypass):
            return None
        cfg = session.config.output_policy.config or {}
        enabled = str(cfg.get("semantic_start_gate", "false")).strip().lower()
        if enabled not in ("1", "true", "yes", "on"):
            return None
        try:
            min_ms = float(cfg.get("semantic_start_min_audio_ms", 160.0))
            max_ms = float(cfg.get("semantic_start_max_hold_ms", 300.0))
        except (TypeError, ValueError):
            min_ms, max_ms = 160.0, 300.0
        return SemanticStartGate(
            bytes_per_sec=ENGINE_SAMPLE_RATE * 4,
            min_audio_ms=min_ms,
            max_hold_ms=max_ms,
        )

    async def _consume_results(
        self,
        session: Session,
        *,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
    ) -> None:
        # L1 session-level aggregation for the session.summary log (answers the
        # daily "拼没拼 batch / 合成了什么" questions).
        batch_agg = {"segments": 0, "batched": 0, "solo": 0, "max_batch_size_seen": 0}
        final_text_parts: list[str] = []

        # Guarded delivery (server-default, per-request override): post-reorder
        # chunks pass through a hold window so hallucinated tails can still be
        # discarded server-side.
        # One lock serializes every release→send path (queue consumer and
        # ticker), keeping chunk order intact.
        hold = self._guarded_hold_for(session)
        hold_lock = asyncio.Lock() if hold is not None else None
        hold_ticker: Optional[asyncio.Task] = None
        # Some integration/unit harnesses intentionally call this unbound
        # coroutine with a narrow frontend test double.  Keep the Phase-1
        # gate optional at that boundary instead of requiring every existing
        # consumer to grow the new helper method.
        semantic_gate_factory = getattr(self, "_semantic_start_gate_for", None)
        semantic_gate = (
            semantic_gate_factory(session)
            if callable(semantic_gate_factory)
            else None
        )
        semantic_gate_lock = asyncio.Lock() if semantic_gate is not None else None
        semantic_ticker: Optional[asyncio.Task] = None
        reorder_watchdog: Optional[asyncio.Task] = None
        prefix_gate_guard_bypass = session.config.timing.extra.get(
            "_prefix_gate_guard_bypass"
        )
        # Segment lifecycle/progress events use the same hierarchical order
        # as AudioReorder. A decoder may finish a lookahead segment first, but
        # its final marker must wait in this map until the predecessor's final
        # marker has crossed the public high-water boundary.
        pending_segment_end_events: dict[
            tuple[int, int], tuple[dict, Optional[dict]]
        ] = {}
        if not isinstance(prefix_gate_guard_bypass, PrefixGateGuardBypass):
            prefix_gate_guard_bypass = None
        # Guarded delivery bookkeeping: verdicts recorded at SEGMENT_END for
        # segments whose audio has NOT fully passed into the hold yet
        # (segments complete out of order; the verdict must be applied to the
        # segment's own audio when the reorder drain reaches it), and a
        # once-only guard for prefill_done events (a rerun re-prefills, but
        # the retry is not client-visible).
        hold_verdicts: dict = {}
        prefill_done_seen: set = set()

        async def _send_now(chunks: list) -> None:
            if on_audio:
                for chunk in chunks:
                    await on_audio(session.session_id, chunk)

        async def _send_chunks(chunks: list) -> None:
            """Apply the optional semantic start gate before client release."""
            if not chunks:
                return
            if semantic_gate is None:
                await _send_now(chunks)
                return
            async with semantic_gate_lock:
                if semantic_gate.released:
                    await _send_now(chunks)
                    return
                semantic_gate.push(chunks)
                safe_wait_ms = 0.0
                estimator = session.audio_credit_estimator
                if estimator is not None:
                    safe_wait_ms = estimator.snapshot().safe_wait_ms
                released = semantic_gate.release_due(
                    semantic_pending=bool(session.text_committer.pending_raw),
                    safe_wait_ms=safe_wait_ms,
                )
                await _send_now(released)

        async def _release_semantic_due(*, force: bool = False) -> None:
            if semantic_gate is None:
                return
            async with semantic_gate_lock:
                released = semantic_gate.flush() if force else semantic_gate.release_due(
                    semantic_pending=bool(session.text_committer.pending_raw),
                    safe_wait_ms=(
                        session.audio_credit_estimator.snapshot().safe_wait_ms
                        if session.audio_credit_estimator is not None
                        else 0.0
                    ),
                )
                await _send_now(released)

        async def _deliver(chunks: list) -> None:
            """Route in-order chunks to the client, via the hold if guarded."""
            if not chunks:
                return
            if hold is None:
                await _send_chunks(chunks)
                return
            # Prefix VAD lives downstream in the gateway.  Feeding its leading
            # raw chunks through the playhead-paced hold makes 400 ms of
            # silence cost roughly 400 ms wall time even when synthesis runs
            # many times faster than realtime.  Let the VAD inspect only that
            # prefix at engine speed; it drops those chunks, and the chunk that
            # opens the gate is the same immediate first chunk guarded
            # delivery has always promised.  Start normal pacing afterwards.
            for chunk in chunks:
                if (
                    prefix_gate_guard_bypass is not None
                    and prefix_gate_guard_bypass.should_bypass
                ):
                    prefix_gate_guard_bypass.record_bypass(len(chunk))
                    await _send_chunks([chunk])
                    if not prefix_gate_guard_bypass.should_bypass:
                        hold.record_external_release(
                            prefix_gate_guard_bypass.first_effective_audio_bytes
                        )
                    continue
                hold.push([chunk])
                async with hold_lock:
                    await _send_chunks(hold.release_due())

        async def _settle_hold(verdict: dict) -> None:
            """Apply a segment's verdict to its audio, now fully in the hold.

            codec EOS / kv_overflow validate the tail (flush; kv_overflow
            audio legitimately continues in the follow-up segment carrying
            the overflow tokens); loop/silence aborts drop the condemned
            trailing bytes and flush the older held audio — legitimate speech
            the playback window had not reached yet. The lock is taken even
            when there is nothing to send: it doubles as the barrier against
            a ticker mid-send, keeping event/audio order intact."""
            # Raw chunks already passed into the downstream VAD cannot be
            # removed by this upstream hold.  At an abort boundary, discard
            # any VAD margin/onset/input state before a chained reorder drain
            # feeds the next segment, otherwise candidates can straddle the
            # two segments and leak condemned audio.
            if prefix_gate_guard_bypass is not None and (
                verdict["discard_all"]
                or verdict["discard_bytes"] > 0
                or str(verdict["eos_reason"]).endswith("_abort")
            ):
                prefix_gate_guard_bypass.discard_pending()

            async with hold_lock:
                dropped = 0
                if semantic_gate is not None and (
                    verdict["discard_all"] or verdict["discard_bytes"] > 0
                ):
                    semantic_gate.discard_segment(
                        verdict["seg_idx"],
                        discard_all=verdict["discard_all"],
                        discard_bytes=verdict["discard_bytes"],
                    )
                if verdict["discard_all"]:
                    dropped = hold.discard()
                elif verdict["discard_bytes"] > 0:
                    dropped = hold.discard_tail(verdict["discard_bytes"])
                flushed = hold.flush()
                await _send_chunks(flushed)
            if dropped:
                logger.info(
                    "Guarded delivery: %s seg=%d %s kept %d frames, "
                    "discarded %d held chunks, flushed %d",
                    session.session_id,
                    verdict["seg_idx"],
                    verdict["eos_reason"],
                    verdict["keep_frames"],
                    dropped,
                    len(flushed),
                )
                LifecycleLogger.emit(
                    session_id=session.session_id,
                    phase="guarded_delivery_discard",
                    segment_idx=verdict["seg_idx"],
                    eos_reason=verdict["eos_reason"],
                    kept_frames=verdict["keep_frames"],
                    discarded_chunks=dropped,
                    flushed_chunks=len(flushed),
                )

        def _hold_verdict(seg_idx: int, metrics: dict) -> dict:
            eos_reason = str(metrics.get("eos_reason", ""))
            discard_bytes = 0
            discard_all = bool(metrics.get("discard_all_audio", False))
            keep = -1
            if eos_reason in ("loop_abort", "silence_abort", "length_abort"):
                frame_bytes = ENGINE_SAMPLE_RATE * 4 * 80 // 1000
                audio_steps = int(metrics.get("audio_steps", 0) or 0)
                tail = int(metrics.get("abort_tail_frames", 0) or 0)
                keep = max(0, audio_steps - tail)
                frozen_ratios = getattr(session.spliter, "_seg_ema_ratio", None)
                if isinstance(frozen_ratios, dict) and seg_idx in frozen_ratios:
                    ema = float(frozen_ratios[seg_idx] or 0.0)
                else:
                    ema = float(
                        getattr(
                            session.spliter,
                            "ema_ratio",
                            getattr(session.spliter, "_ema_ratio", 0.0),
                        )
                        or 0.0
                    )
                text_tokens = int(metrics.get("text_tokens", 0) or 0)
                if ema > 0 and text_tokens > 0:
                    expected = int(ema * text_tokens * 1.15) + 2
                    keep = min(keep, expected)
                discard_bytes = max(0, (audio_steps - keep) * frame_bytes)
            return {
                "eos_reason": eos_reason,
                "discard_bytes": discard_bytes,
                "discard_all": discard_all,
                "keep_frames": keep,
                "seg_idx": seg_idx,
            }

        def _build_segment_end_events(
            seg_idx: int,
            result: EngineResult,
            rm: dict,
        ) -> Optional[tuple[dict, Optional[dict]]]:
            """Build a segment's lifecycle/final-progress events.

            The prepared record is published only after this segment's own
            held audio has been settled and before ``AudioReorder`` drains a
            later segment. Otherwise a later segment's audio anchor can reach
            the transport before this segment's final anchor and make the
            global raw/text cursor appear to move backwards.
            """
            if on_event is None:
                return None

            metrics = {str(k): str(v) for k, v in rm.items()}
            segment_text = session.segment_texts.pop(seg_idx, "")
            metrics["segment_id"] = str(seg_idx)
            if segment_text:
                preview = (
                    segment_text[:64] + "..."
                    if len(segment_text) > 64
                    else segment_text
                )
                metrics["segment_text_preview"] = preview
            if "audio_steps" in rm:
                metrics["segment_decode_steps"] = str(rm["audio_steps"])
            if "text_tokens" in rm:
                metrics["segment_text_tokens"] = str(rm["text_tokens"])
            if "cache_hit" in rm:
                metrics["segment_cache_hit"] = str(rm["cache_hit"])
            if "prefill_duration_ms" in rm:
                metrics["segment_prefill_ms"] = str(rm["prefill_duration_ms"])
            if session.audio_credit_estimator is not None:
                rate = session.audio_credit_estimator.snapshot()
                metrics.update(
                    {
                        "lambda_raw": f"{rate.lambda_raw:.3f}",
                        "lambda_norm": f"{rate.lambda_norm:.3f}",
                        "audio_credit_ms": f"{rate.audio_credit_ms:.3f}",
                        "safe_wait_ms": f"{rate.safe_wait_ms:.3f}",
                    }
                )

            eos_reason = str(rm.get("eos_reason", ""))
            progress = self._make_text_progress_event(
                session,
                seg_idx,
                rm,
                final=not eos_reason.endswith("_abort"),
            )
            if progress is not None:
                # ``segment_end`` is a legacy diagnostic event.  The complete
                # progress record is sent separately so transports cannot
                # mistake a segment lifecycle notification for an output
                # sample anchor.
                metrics["text_progress"] = progress["meta"].get(
                    "text_progress", "0"
                )
                metrics["progress_final"] = progress["meta"].get(
                    "progress_final", "false"
                )
            session.segment_token_emitted_count.pop(seg_idx, None)
            session.text_boundary_emitted.discard(seg_idx)
            return (
                {
                    "type": "segment_end",
                    "segment_idx": seg_idx,
                    "text": segment_text,
                    "meta": metrics,
                },
                progress,
            )

        async def _emit_segment_end_events(
            prepared: Optional[tuple[dict, Optional[dict]]],
        ) -> None:
            if prepared is None or on_event is None:
                return
            segment_event, progress = prepared
            await on_event(session.session_id, segment_event)
            if progress is not None:
                await on_event(session.session_id, progress)

        async def _publish_segment_end_event(key: tuple[int, int]) -> None:
            pending = pending_segment_end_events.pop(key, None)
            if pending is None:
                return
            await _emit_segment_end_events(pending)

        if hold is not None:

            async def _hold_tick() -> None:
                while True:
                    await asyncio.sleep(0.1)
                    if hold.held_bytes:
                        async with hold_lock:
                            await _send_chunks(hold.release_due())

            hold_ticker = asyncio.create_task(_hold_tick())
        if semantic_gate is not None:

            async def _semantic_tick() -> None:
                while True:
                    await asyncio.sleep(0.02)
                    await _release_semantic_due()

            semantic_ticker = asyncio.create_task(_semantic_tick())

        reorder_timeout = float(
            getattr(session.reorder, "_stall_timeout_sec", 0.0) or 0.0
        )
        if reorder_timeout > 0:

            async def _reorder_watch() -> None:
                # A stalled predecessor must not be bypassed: doing so would
                # corrupt audio and text-progress order.  Recovery is an
                # internal cancel, which lets the engine publish its normal
                # terminal result and release every slot.
                interval = min(max(reorder_timeout / 4.0, 0.05), 0.5)
                while True:
                    await asyncio.sleep(interval)
                    stall = session.reorder.check_stall()
                    if stall is None:
                        continue
                    meta = {
                        "blocked_segment": f"{stall.blocked_segment[0]}:{stall.blocked_segment[1]}",
                        "stall_elapsed_ms": f"{stall.elapsed_sec * 1000.0:.1f}",
                        "buffered_keys": str(stall.buffered_keys),
                        "buffered_chunks": str(stall.buffered_chunks),
                        "reason": "audio_reorder_stall_timeout",
                    }
                    LifecycleLogger.emit(
                        session_id=session.session_id,
                        phase="audio.reorder_stall_timeout",
                        segment_idx=stall.blocked_segment[0],
                        **meta,
                    )
                    if on_event is not None:
                        try:
                            await on_event(
                                session.session_id,
                                {
                                    "type": "warning",
                                    "segment_idx": stall.blocked_segment[0],
                                    "message": "audio reorder stalled; cancelling session",
                                    "meta": meta,
                                },
                            )
                        except Exception:
                            # A diagnostic consumer must not prevent the
                            # terminal recovery request from reaching Engine.
                            logger.exception(
                                "Reorder-stall warning delivery failed for %s",
                                session.session_id,
                            )
                    # Stop accepting new text while the engine processes the
                    # cancellation request; the terminal result still owns
                    # final cleanup and on_done delivery.
                    session.state = SessionState.DONE
                    await self._dispatcher.submit_cancel(
                        session.session_id,
                        reason="audio_reorder_stall_timeout",
                    )
                    return

            reorder_watchdog = asyncio.create_task(_reorder_watch())

        try:
            while True:
                result: EngineResult = await session.result_queue.get()

                if result.type == ResultType.AUDIO_CHUNK:
                    audio = result.audio_bytes or b""
                    session.total_audio_bytes += len(audio)
                    if session.audio_credit_estimator is not None and audio:
                        frame_bytes = ENGINE_SAMPLE_RATE * 4 * 80 // 1000
                        session.audio_credit_estimator.observe_codec_frames(
                            max(1, round(len(audio) / frame_bytes))
                        )

                    # Propagate the first raw-audio timestamp from the engine
                    # thread.  A session can contain multiple segments; later
                    # segment metrics must not replace the session's first.
                    first_raw_audio_at = None
                    if result.metrics and "first_raw_audio_at" in result.metrics:
                        try:
                            first_raw_audio_at = float(
                                result.metrics["first_raw_audio_at"]
                            )
                        except (ValueError, TypeError):
                            pass
                    session.record_first_audio(first_raw_audio_at)

                    reorder = session.reorder
                    meta = session.segment_order.get(
                        result.segment_idx,
                        SegmentOrderMeta(result.segment_idx, 0, True),
                    )
                    progress = self._make_text_progress_event(
                        session,
                        result.segment_idx,
                        result.metrics or {},
                    )
                    attributed = AttributedAudioChunk(
                        pcm_bytes=audio,
                        progress_event=progress,
                        segment_idx=result.segment_idx,
                        source_frame_start=int(
                            (result.metrics or {}).get("source_frame_start", 0) or 0
                        ),
                        source_frame_end=int(
                            (result.metrics or {}).get("source_frame_end", 0) or 0
                        ),
                    )
                    ready = reorder.push(meta.group_idx, meta.local_idx, attributed)
                    await _deliver(ready)

                elif result.type == ResultType.PREFILL_DONE:
                    # Propagate prefill timing from engine thread
                    if result.metrics:
                        if (
                            "prefill_started_at" in result.metrics
                            and session.prefill_started_at is None
                        ):
                            try:
                                session.prefill_started_at = float(
                                    result.metrics["prefill_started_at"]
                                )
                            except (ValueError, TypeError):
                                pass
                        if (
                            "prefill_completed_at" in result.metrics
                            and session.prefill_completed_at is None
                        ):
                            try:
                                session.prefill_completed_at = float(
                                    result.metrics["prefill_completed_at"]
                                )
                            except (ValueError, TypeError):
                                pass
                        if (
                            "first_text_dequeued_at" in result.metrics
                            and session.first_text_dequeued_at is None
                        ):
                            try:
                                session.first_text_dequeued_at = float(
                                    result.metrics["first_text_dequeued_at"]
                                )
                            except (ValueError, TypeError):
                                pass

                    # Once per segment: a hallucination rerun re-prefills the
                    # same segment_idx, but the retry is not client-visible.
                    if on_event and result.segment_idx not in prefill_done_seen:
                        prefill_done_seen.add(result.segment_idx)
                        await on_event(
                            session.session_id,
                            {
                                "type": "prefill_done",
                                "segment_idx": result.segment_idx,
                                "text": session.segment_texts.get(
                                    result.segment_idx, ""
                                ),
                                "meta": {
                                    str(k): str(v)
                                    for k, v in (result.metrics or {}).items()
                                },
                            },
                        )

                elif result.type == ResultType.SEGMENT_RETRY:
                    # Engine is rerunning a hallucinated lookahead segment with
                    # a fresh seed: drop the buffered garbage attempt so the
                    # rerun's chunks land in a clean buffer. Not client-visible;
                    # no SEGMENT_END was (or will yet be) sent for this segment.
                    meta = session.segment_order.get(
                        result.segment_idx,
                        SegmentOrderMeta(result.segment_idx, 0, True),
                    )
                    dropped = session.reorder.discard(meta.group_idx, meta.local_idx)
                    session.text_progress_estimators.pop(result.segment_idx, None)
                    session.segment_progress_frames.pop(result.segment_idx, None)
                    session.native_cursor_projectors.pop(result.segment_idx, None)
                    session.text_coordinate_projectors.pop(result.segment_idx, None)
                    # Retry reuses the logical segment and its owner-safe
                    # plan; only the projector high-water is reset.
                    rm = result.metrics or {}
                    _observe_spliter_ratio(
                        session,
                        segment_idx=result.segment_idx,
                        audio_steps=rm.get("audio_steps", 0),
                        text_tokens=rm.get("text_tokens", 0),
                        outcome=RatioOutcome.from_retry_reason(
                            str(rm.get("retry_reason", ""))
                        ),
                    )
                    logger.info(
                        "Segment retry: %s seg=%d reason=%s attempt=%s "
                        "(discarded %d buffered chunks)",
                        session.session_id,
                        result.segment_idx,
                        rm.get("retry_reason", ""),
                        rm.get("retry_idx", ""),
                        dropped,
                    )
                    LifecycleLogger.emit(
                        session_id=session.session_id,
                        phase="segment_retry_discard",
                        segment_idx=result.segment_idx,
                        discarded_chunks=dropped,
                        retry_reason=str(rm.get("retry_reason", "")),
                        retry_idx=str(rm.get("retry_idx", "")),
                    )

                elif result.type == ResultType.SEGMENT_END:
                    seg_idx = result.segment_idx
                    session.segments_done += 1

                    # Aggregate batch facts for the session.summary line.
                    rm = result.metrics or {}
                    batch_agg["segments"] += 1
                    if rm.get("batched"):
                        batch_agg["batched"] += 1
                    else:
                        batch_agg["solo"] += 1
                    bseen = int(rm.get("batch_size_seen", 0) or 0)
                    if bseen > batch_agg["max_batch_size_seen"]:
                        batch_agg["max_batch_size_seen"] = bseen
                    seg_text_for_summary = session.segment_texts.get(seg_idx, "")
                    if seg_text_for_summary:
                        final_text_parts.append(seg_text_for_summary)

                    reorder = session.reorder
                    meta = session.segment_order.pop(
                        seg_idx,
                        SegmentOrderMeta(seg_idx, 0, True),
                    )

                    # Guarded delivery: a segment's verdict must be applied to
                    # that segment's own audio, and segments complete out of
                    # order. If this segment IS the playhead, all its audio is
                    # already in the hold (playhead chunks pass straight
                    # through the reorder) — settle now. Otherwise its audio
                    # is still buffered inside the reorder; record the verdict
                    # and settle when the drain below (or a later chained
                    # drain) moves it into the hold.
                    if hold is not None:
                        verdict = _hold_verdict(seg_idx, result.metrics or {})
                        if reorder.next_emit_segment == (
                            meta.group_idx,
                            meta.local_idx,
                        ):
                            await _settle_hold(verdict)
                        else:
                            hold_verdicts[(meta.group_idx, meta.local_idx)] = verdict

                    if hold is None:
                        if semantic_gate is not None:
                            verdict = _hold_verdict(seg_idx, result.metrics or {})
                            if verdict["discard_all"] or verdict["discard_bytes"] > 0:
                                semantic_gate.discard_segment(
                                    seg_idx,
                                    discard_all=verdict["discard_all"],
                                    discard_bytes=verdict["discard_bytes"],
                                )

                    if result.metrics:
                        audio_steps = result.metrics.get("audio_steps", 0)
                        text_tokens = result.metrics.get("text_tokens", 0)
                        overflow = _metric_bool(
                            result.metrics.get("overflow", False)
                        )
                        eos_reason = str(result.metrics.get("eos_reason", ""))
                        _observe_spliter_ratio(
                            session,
                            segment_idx=seg_idx,
                            audio_steps=audio_steps,
                            text_tokens=text_tokens,
                            outcome=RatioOutcome.from_eos_reason(
                                eos_reason,
                                overflow=overflow,
                            ),
                        )
                        if session.audio_credit_estimator is not None:
                            session.audio_credit_estimator.observe_normalized_tokens(int(text_tokens or 0))

                    # Queue the lifecycle/final marker using the same
                    # hierarchical key as AudioReorder. It is published now
                    # only when this segment is the current playhead; a
                    # lookahead segment waits until its predecessor has
                    # crossed the public text high-water boundary.
                    segment_key = (meta.group_idx, meta.local_idx)
                    prepared_end_events = _build_segment_end_events(
                        seg_idx, result, rm
                    )
                    if prepared_end_events is not None:
                        pending_segment_end_events[segment_key] = prepared_end_events
                    if (
                        prepared_end_events is not None
                        and reorder.next_emit_segment == segment_key
                    ):
                        await _publish_segment_end_event(segment_key)

                    if hold is None:
                        for key, chunks, _fully_passed in reorder.mark_done_ex(
                            meta.group_idx,
                            meta.local_idx,
                            group_final=meta.group_final,
                        ):
                            if key in pending_segment_end_events:
                                await _publish_segment_end_event(key)
                            if chunks:
                                await _deliver(chunks)
                    else:
                        # Segment-attributed drain: push each drained
                        # segment's chunks and settle it as soon as it has
                        # fully passed (its verdict is already recorded — a
                        # segment can only be fully drained after its own
                        # SEGMENT_END marked it done). The trailing partially
                        # drained segment is the new playhead: its audio just
                        # enters the window and waits for its own verdict.
                        for key, chunks, fully_passed in reorder.mark_done_ex(
                            meta.group_idx,
                            meta.local_idx,
                            group_final=meta.group_final,
                        ):
                            if key in pending_segment_end_events:
                                await _publish_segment_end_event(key)
                            if chunks:
                                await _deliver(chunks)
                            if fully_passed and key in hold_verdicts:
                                await _settle_hold(hold_verdicts.pop(key))

                    # L2 reorder_state: buffered audio waiting on an earlier
                    # segment at this boundary = reorder stall risk.
                    rstate = reorder.pending_state()
                    if rstate["buffered_chunks"] > 0 and obs.is_enabled(
                        obs.ObsLevel.DEBUG, session.config.observability_level
                    ):
                        LifecycleLogger.emit(
                            session_id=session.session_id,
                            phase="reorder_state",
                            segment_idx=seg_idx,
                            min_level=obs.ObsLevel.DEBUG,
                            session_level=session.config.observability_level,
                            **rstate,
                        )

                    new_actions = session.spliter.on_segment_done(seg_idx)
                    session.text_progress_estimators.pop(seg_idx, None)
                    session.segment_progress_frames.pop(seg_idx, None)
                    session.native_cursor_projectors.pop(seg_idx, None)
                    session.text_coordinate_projectors.pop(seg_idx, None)
                    session.cursor_segment_plans.pop(seg_idx, None)
                    session.cursor_segment_bounds.pop(seg_idx, None)
                    session.segment_token_spans.pop(seg_idx, None)
                    session.segment_token_keys.pop(seg_idx, None)
                    if new_actions:
                        await self._dispatch_segment_actions(session, new_actions)
                    await self._dispatcher.maybe_send_session_tokens_done(session)

                elif result.type == ResultType.RATIO_UPDATE:
                    if session.spliter and result.ema_ratio > 0:
                        setter = getattr(session.spliter, "set_duration_ratio", None)
                        if callable(setter):
                            setter(result.ema_ratio)
                        else:
                            # Legacy test doubles only; production Spliter
                            # routes this through the duration-only API so a
                            # stale external ratio cannot weaken safety.
                            session.spliter._ema_ratio = result.ema_ratio

                elif result.type == ResultType.WARNING:
                    if on_event:
                        await on_event(
                            session.session_id,
                            {
                                "type": "warning",
                                "segment_idx": result.segment_idx,
                                "message": result.warning_msg or "",
                                "text": session.segment_texts.get(
                                    result.segment_idx, ""
                                ),
                            },
                        )

                elif result.type == ResultType.SESSION_DONE:
                    # Lock unconditionally: it is the barrier that keeps the
                    # done event from overtaking ticker-in-flight audio
                    # (held_bytes hits 0 while popped chunks are still being
                    # awaited into the outbound queue).
                    if hold is not None:
                        async with hold_lock:
                            await _send_chunks(hold.flush())
                    await _release_semantic_due(force=True)
                    session.state = SessionState.DONE
                    # Surface the L1 batch / text facts to the client protocol
                    # (done_meta merges these), enabling L0 client self-analysis.
                    done_metrics = dict(result.metrics or {})
                    done_metrics["server_batch_summary"] = json.dumps(
                        batch_agg, ensure_ascii=False
                    )
                    done_metrics["server_final_synthesized_text"] = obs.text_preview(
                        "".join(final_text_parts)
                    )
                    done_metrics["server_total_segments"] = str(session.segments_done)
                    if session.audio_credit_estimator is not None:
                        rate = session.audio_credit_estimator.snapshot()
                        done_metrics.update(
                            {
                                "lambda_raw": rate.lambda_raw,
                                "lambda_norm": rate.lambda_norm,
                                "audio_credit_ms": rate.audio_credit_ms,
                                "safe_wait_ms": rate.safe_wait_ms,
                            }
                        )
                    try:
                        if on_done:
                            await on_done(session.session_id, done_metrics)
                    finally:
                        # Gateway on_done performs output-policy finalization
                        # (VAD flush and trim accounting). Emit afterwards so
                        # the summary sees final values, but retain it even if
                        # downstream delivery raises after partial finalization.
                        self._emit_session_summary(session, batch_agg, final_text_parts)
                        LifecycleLogger.emit(
                            session_id=session.session_id,
                            phase="session.completed",
                            request_id=session.config.timing.request_id or None,
                            turn_id=session.config.timing.turn_id or None,
                            session_level=session.config.observability_level,
                            total_segments=session.segments_done,
                            total_audio_bytes=session.total_audio_bytes,
                        )
                    break

                elif result.type == ResultType.ERROR:
                    if hold is not None:
                        async with hold_lock:
                            await _send_chunks(hold.flush())
                    await _release_semantic_due(force=True)
                    logger.error(
                        "Session %s error: %s", session.session_id, result.error_msg
                    )
                    session.state = SessionState.DONE
                    # Emit structured error lifecycle event
                    LifecycleLogger.emit(
                        session_id=session.session_id,
                        phase="session.error",
                        request_id=session.config.timing.request_id or None,
                        error_type="engine_error",
                        error_message=result.error_msg or "",
                        current_phase=session.state.value,
                        segments_completed=session.segments_done,
                    )
                    error_meta = {
                        "error_phase": session.state.value,
                        "error_type": "engine_error",
                        "error_message": str(result.error_msg or ""),
                        "segments_completed": str(session.segments_done),
                    }
                    if on_done:
                        await on_done(
                            session.session_id,
                            {"error": result.error_msg, **error_meta},
                        )
                    break

        except asyncio.CancelledError:
            pass
        finally:
            if hold_ticker is not None:
                hold_ticker.cancel()
            if semantic_ticker is not None:
                semantic_ticker.cancel()
            if reorder_watchdog is not None:
                reorder_watchdog.cancel()
            self._cleanup_session(session.session_id, expected=session)

    def _emit_session_summary(
        self,
        session: Session,
        batch_agg: dict,
        final_text_parts: list,
    ) -> None:
        """Emit the L1 ``session.summary`` line: one structured record + one
        human-readable line answering the five daily questions (TTFT / link
        timing / synthesized text / batch / VAD)."""
        acc = session.config.timing.extra.get("_server_timing_accumulator")
        summary: dict[str, Any] = {}
        if isinstance(acc, ServerTimingAccumulator):
            summary = acc.summary_dict()
        final_text = obs.text_preview("".join(final_text_parts))
        summary["batch_summary"] = batch_agg
        summary["text"] = {
            "final_synthesized_text": final_text,
            "total_segments": session.segments_done,
        }
        LifecycleLogger.emit(
            session_id=session.session_id,
            phase="session.summary",
            request_id=session.config.timing.request_id or None,
            turn_id=session.config.timing.turn_id or None,
            session_level=session.config.observability_level,
            **summary,
        )
        ttft = summary.get("ttft", {})
        raw_ttft_ms = ttft.get("create_to_first_raw_ms")
        effective_ttft_ms = ttft.get("create_to_first_effective_ms")
        infer_ms = summary.get("pipeline_ms", {}).get("inference_ms")
        gating_ms = summary.get("pipeline_ms", {}).get("gating_ms")
        cache = summary.get("cache", {})
        logger.info(
            "session=%s DONE ttft_raw=%sms ttft_effective=%sms infer=%sms "
            'gating=%sms batch=%d/%d vad_trim=%sms cache=%s segs=%d "%s"',
            session.session_id,
            raw_ttft_ms,
            effective_ttft_ms,
            infer_ms,
            gating_ms,
            batch_agg["batched"],
            batch_agg["segments"],
            summary.get("prefix_trimmed_ms", 0.0),
            "HIT" if cache.get("prefix_cache_hit") else "MISS",
            session.segments_done,
            final_text,
        )

    def _cleanup_session(
        self, session_id: str, *, expected: Optional[Session] = None
    ) -> None:
        # Identity guard: a cancelled consumer task unwinds and runs this
        # `finally` only later, after the event loop resumes it. If the same
        # session_id was re-created in the meantime (create_session cancels the
        # old session then registers a new one under the same id), the stale
        # task must not clobber the new session/task. Only clean up when the
        # registered session is still the one this call is for.
        if expected is not None and self._sessions.get(session_id) is not expected:
            return
        session = self._sessions.pop(session_id, None)
        self._consumer_tasks.pop(session_id, None)
        tn_tasks = getattr(self, "_tn_tasks", {})
        tn_task = tn_tasks.pop(session_id, None)
        if tn_task and not tn_task.done():
            tn_task.cancel()
        getattr(self, "_tn_locks", {}).pop(session_id, None)
        diagnostic_routers = getattr(self, "_diagnostic_text_routers", None)
        if diagnostic_routers is not None:
            diagnostic_routers.pop(session_id, None)
        cursor_adapters = getattr(self, "_cursor_plan_adapters", None)
        if cursor_adapters is not None:
            cursor_adapters.pop(session_id, None)
        if session:
            commitment = getattr(session, "commitment_adapter", None)
            if commitment is not None:
                try:
                    commitment.reset()
                except Exception:
                    logger.exception(
                        "Optional commitment reset failed for session %s",
                        session_id,
                    )
            # Compute structured summary metrics
            summary: dict[str, Any] = {
                "session_id": session_id,
                "segments_done": session.segments_done,
                "segments_submitted": session.segments_submitted,
                "total_audio_bytes": session.total_audio_bytes,
            }

            # Derived timing metrics
            if session.session_create_to_first_raw_audio_ms is not None:
                summary["session_create_to_first_raw_audio_ms"] = round(
                    session.session_create_to_first_raw_audio_ms, 3
                )
            if (
                session.first_raw_audio_at is not None
                and session.first_text_enqueued_at is not None
            ):
                summary["first_text_enqueue_to_first_raw_audio_ms"] = round(
                    (session.first_raw_audio_at - session.first_text_enqueued_at)
                    * 1000,
                    3,
                )
            if (
                session.first_raw_audio_at is not None
                and session.first_text_dequeued_at is not None
            ):
                summary["first_text_dequeue_to_first_raw_audio_ms"] = round(
                    (session.first_raw_audio_at - session.first_text_dequeued_at)
                    * 1000,
                    3,
                )
            if (
                session.prefill_completed_at is not None
                and session.prefill_started_at is not None
            ):
                summary["engine_prefill_ms"] = round(
                    (session.prefill_completed_at - session.prefill_started_at) * 1000,
                    3,
                )
            # Also log a human-readable summary
            latency = session.session_create_to_first_raw_audio_ms
            logger.info(
                "Session %s cleaned up (session_create_to_first_raw_audio_ms=%.1fms, segments=%d/%d, summary=%s)",
                session_id,
                latency or -1,
                session.segments_done,
                session.segments_submitted,
                json.dumps(summary, ensure_ascii=False),
            )

    async def _dispatch_segment_actions(
        self,
        session: Session,
        actions: list,
    ) -> None:
        if not actions:
            return
        self._emit_split_decisions(session)
        self._record_segment_text(actions, session)
        if session.cursor_label_plan is not None and any(
            session.cursor_plan_for_segment(index) != previous
            for index, previous in session.cursor_segment_plans.items()
            if index in session.segment_order
        ):
            # Bounds grow when tokens are assigned to an already active
            # segment. Publish its enlarged label window before these tokens,
            # even when no new TN commit was created (queued splitter work).
            session.cursor_plan_revision += 1
            session.cursor_label_plan = replace(
                session.cursor_label_plan, revision=session.cursor_plan_revision
            )
            await self._dispatcher.submit_cursor_plan(session)
        await self._dispatcher.dispatch_segment_actions(session, actions)
        await self._emit_text_token_events(session, actions)
        await self._emit_text_boundary_events(session, actions)

    def _emit_split_decisions(self, session: Session) -> None:
        """Drain the spliter's buffered L2 split-decision records and emit them
        as ``split_decision`` lifecycle events with session context (answers
        "为什么这么切"). No-op unless the session is at L2/DEBUG or above."""
        sp = session.spliter
        if sp is None:
            return
        for d in sp.drain_split_decisions():
            d = dict(d)
            # offline pre-split → split_decision; streaming FSM → driver_transition.
            phase = (
                "driver_transition"
                if d.get("obs") == "driver_transition"
                else "split_decision"
            )
            d.pop("obs", None)
            if "text_preview" in d:
                d["text_preview"] = obs.text_preview(d.get("text_preview", ""))
            LifecycleLogger.emit(
                session_id=session.session_id,
                phase=phase,
                request_id=session.config.timing.request_id or None,
                session_level=session.config.observability_level,
                min_level=obs.ObsLevel.DEBUG,
                **d,
            )

    def _record_segment_text(self, actions: list, session: Session) -> None:
        for sa in actions:
            if sa.token_text and sa.action.type in (
                ActionType.PREFILL,
                ActionType.DECODE,
            ):
                normalized_start = int(getattr(sa, "normalized_start", 0))
                normalized_end = int(getattr(sa, "normalized_end", 0))
                raw_start = int(getattr(sa, "raw_start", 0))
                raw_end = int(getattr(sa, "raw_end", 0))
                token_key = (
                    int(sa.segment_idx),
                    int(getattr(sa, "token_id", -1)),
                    str(sa.token_text),
                    normalized_start,
                    normalized_end,
                    raw_start,
                    raw_end,
                )
                keys = session.segment_token_keys.setdefault(
                    sa.segment_idx, set()
                )
                if token_key in keys:
                    continue
                keys.add(token_key)
                session.segment_texts[sa.segment_idx] = (
                    session.segment_texts.get(sa.segment_idx, "") + sa.token_text
                )
                session.segment_token_spans.setdefault(sa.segment_idx, []).append(
                    {
                        "normalized_start": normalized_start,
                        "normalized_end": normalized_end,
                        "raw_start": raw_start,
                        "raw_end": raw_end,
                    }
                )
                bounds = session.cursor_segment_bounds.get(sa.segment_idx)
                if bounds is None:
                    session.cursor_segment_bounds[sa.segment_idx] = (
                        normalized_start,
                        normalized_end,
                    )
                else:
                    session.cursor_segment_bounds[sa.segment_idx] = (
                        min(bounds[0], normalized_start),
                        max(bounds[1], normalized_end),
                    )

    def _make_text_progress_event(
        self,
        session: Session,
        segment_idx: int,
        metrics: dict,
        *,
        final: bool = False,
    ) -> Optional[dict]:
        """Estimate codec-to-token progress, then apply shared text coordinates."""

        if session.spliter is None:
            return None

        frame_end = _metric_int(metrics.get("source_frame_end"))
        if frame_end is None:
            frame_end = session.segment_progress_frames.get(segment_idx, 0) + (
                0 if final else 1
            )
        frame_start = _metric_int(metrics.get("source_frame_start"))
        if frame_start is None:
            frame_start = max(0, frame_end - (0 if final else 1))
        session.segment_progress_frames[segment_idx] = max(
            session.segment_progress_frames.get(segment_idx, 0), frame_end
        )

        text_token_count = _metric_int(metrics.get("text_tokens"))
        if text_token_count is None:
            text_token_count = session.segment_token_emitted_count.get(segment_idx, 0)
        text_token_count = max(0, text_token_count)
        spans = session.segment_token_spans.get(segment_idx, [])
        if not spans:
            return None
        coordinates = SegmentTextCoordinates(spans, session.text_journal)
        coordinate_projectors = getattr(session, "text_coordinate_projectors", None)
        if coordinate_projectors is None:
            coordinate_projectors = session.text_coordinate_projectors = {}
        projection = coordinate_projectors.setdefault(
            segment_idx, TextProgressProjection(segment_idx)
        )

        def project(token_end: int) -> dict[str, str]:
            return projection.project(
                CodecTokenProgress(segment_idx, frame_start, frame_end, token_end),
                coordinates,
            )

        # Native observations arrive attached to source audio. Lookahead holds
        # the precise route; unavailable/invalid estimates may fall back to
        # EMA without resetting shared coordinate high-water or audio state.
        plan = getattr(session, "cursor_segment_plans", {}).get(segment_idx)
        if plan is None:
            plan = getattr(session, "cursor_label_plan", None)
        cursor_revision = _metric_int(metrics.get("cursor_plan_revision"))
        has_cursor_observation = "cursor_valid" in metrics
        projectors = getattr(session, "native_cursor_projectors", None)
        if projectors is None:
            projectors = {}
        projector = projectors.get(segment_idx)
        native_revision_ready = cursor_revision == (plan.revision if plan else None)
        native_allowed = not getattr(session, "native_cursor_disabled", False)
        if native_allowed and plan is not None and plan.active and plan.label_normalized_spans:
            # The model estimates label progress. Only this adapter knows that
            # vocabulary; all protocol coordinates below use tokenizer spans.
            if (native_revision_ready and has_cursor_observation) or (
                final and projection.basis == NATIVE_CURSOR_PROGRESS_BASIS
            ):
                try:
                    valid = _metric_bool(metrics.get("cursor_valid"))
                    if valid and not final and "cursor_mu" not in metrics:
                        raise ValueError("valid native observation is missing mu")
                    mu = float(metrics.get("cursor_mu", plan.label_count if final else 0.0))
                    confidence = float(metrics.get("cursor_confidence", 0.0) or 0.0)
                    if not math.isfinite(mu) or not math.isfinite(confidence):
                        raise ValueError("nonfinite native cursor observation")
                    token_end = len(spans) if final else (
                        coordinates.token_end_from_labels(mu, plan.label_normalized_spans)
                        if valid else projection.token_end
                    )
                    mapped = project(token_end)
                    if valid and not final:
                        stall_frames = getattr(
                            session, "native_cursor_stall_frames", None
                        )
                        if stall_frames is None:
                            stall_frames = session.native_cursor_stall_frames = {}
                        last_token_end = getattr(
                            session, "native_cursor_last_token_end", None
                        )
                        if last_token_end is None:
                            last_token_end = session.native_cursor_last_token_end = {}
                        last_frame_end = getattr(
                            session, "native_cursor_last_frame_end", None
                        )
                        if last_frame_end is None:
                            last_frame_end = session.native_cursor_last_frame_end = {}
                        previous_frame_end = last_frame_end.get(segment_idx)
                        previous_token_end = last_token_end.get(segment_idx)
                        if (
                            previous_frame_end is not None
                            and frame_end > previous_frame_end
                            and previous_token_end is not None
                            and token_end <= previous_token_end
                        ):
                            stall_frames[segment_idx] = (
                                stall_frames.get(segment_idx, 0)
                                + frame_end
                                - previous_frame_end
                            )
                        elif previous_token_end is None or token_end > previous_token_end:
                            stall_frames[segment_idx] = 0
                        last_frame_end[segment_idx] = frame_end
                        last_token_end[segment_idx] = max(
                            token_end, previous_token_end or 0
                        )
                        if (
                            stall_frames.get(segment_idx, 0)
                            >= _NATIVE_CURSOR_STALL_FRAME_LIMIT
                        ):
                            logger.warning(
                                "Native cursor stalled for %s/%d; falling back to EMA",
                                session.session_id,
                                segment_idx,
                            )
                            FrontendInterface._disable_native_cursor(session, "stalled")
                            native_allowed = False
                    if not native_allowed:
                        # The watchdog may have disabled native progress after
                        # this event was decoded.  Fall through to EMA while
                        # retaining the shared projection high-water mark.
                        pass
                    else:
                        projection.basis = NATIVE_CURSOR_PROGRESS_BASIS
                        return {
                            "type": "text_progress", "segment_idx": segment_idx, "text": "",
                            "meta": {
                                "segment_id": str(segment_idx),
                                "progress_basis": NATIVE_CURSOR_PROGRESS_BASIS,
                                "progress_quality": "native",
                                "cursor_mu": f"{mu:.6f}",
                                "cursor_confidence": f"{max(0.0, min(1.0, confidence)):.6f}",
                                "text_progress": f"{projection.token_end / len(spans):.6f}",
                                "progress_final": "true" if final else "false",
                                **mapped,
                                "text_input_final": "true" if session.input_complete else "false",
                                "alignment_final": "true" if final else "false",
                            },
                        }
                except (TypeError, ValueError, OverflowError):
                    logger.warning("Invalid native token estimate for %s/%d",
                                   session.session_id, segment_idx)
        if final and projector is not None and plan is not None:
            native_revision_ready = projector.plan.revision == plan.revision
        if (
            native_allowed
            and
            plan is not None
            and plan.active
            and not plan.label_normalized_spans
            and native_revision_ready
            and (has_cursor_observation or final)
        ):
            if projector is None:
                projector = NativeCursorProgressProjector(
                    segment_idx=segment_idx,
                    plan=plan,
                )
                projectors[segment_idx] = projector
            else:
                try:
                    projector.update_plan(plan)
                except ValueError:
                    projectors.pop(segment_idx, None)
                    projector = None
            if projector is not None:
                try:
                    native = projector.update(
                        mu=metrics.get("cursor_mu") if has_cursor_observation else None,
                        valid=(
                            _metric_bool(metrics.get("cursor_valid"))
                            if has_cursor_observation
                            else True
                        ),
                        confidence=float(metrics.get("cursor_confidence", 0.0) or 0.0),
                        source_frame_start=frame_start,
                        source_frame_end=frame_end,
                        final=final,
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.warning(
                        "Native cursor output rejected; falling back to EMA: "
                        "session=%s segment=%d error=%s",
                        session.session_id,
                        segment_idx,
                        exc,
                    )
                    projectors.pop(segment_idx, None)
                    native = None
                if native is not None:
                    projection.basis = NATIVE_CURSOR_PROGRESS_BASIS
                    native_meta = native.to_meta()
                    native_meta.update(project(
                        len(spans) if final else coordinates.token_end_at(
                            native.normalized_codepoint_end
                        )
                    ))
                    return {
                        "type": "text_progress",
                        "segment_idx": segment_idx,
                        "text": "",
                        "meta": {
                            "segment_id": str(segment_idx),
                            **native_meta,
                            "text_input_final": "true"
                            if session.input_complete
                            else "false",
                            "alignment_final": "true" if final else "false",
                        },
                    }

        estimator = session.text_progress_estimators.get(segment_idx)
        if estimator is None:
            ratio_for_segment = getattr(
                session.spliter, "ema_ratio_for_segment", None
            )
            if callable(ratio_for_segment):
                ema_ratio = ratio_for_segment(segment_idx)
            else:
                # Keep custom/legacy spliter test doubles source-compatible;
                # production Spliter instances always expose the frozen API.
                ema_ratio = getattr(
                    session.spliter,
                    "ema_ratio",
                    getattr(session.spliter, "_ema_ratio", 4.5),
                )
            estimator = EmaTextProgressEstimator(
                segment_idx=segment_idx,
                ema_ratio=ema_ratio,
            )
            session.text_progress_estimators[segment_idx] = estimator

        estimate = estimator.update(
            source_frame_start=frame_start,
            source_frame_end=frame_end,
            text_token_count=text_token_count,
            final=final,
        )
        projection.basis = TEXT_PROGRESS_BASIS
        return {
            "type": "text_progress",
            "segment_idx": segment_idx,
            "text": "",
            "meta": {
                "segment_id": str(segment_idx),
                **estimate.to_meta(),
                **project(min(estimate.text_token_end, len(spans))),
                "text_progress": f"{projection.token_end / len(spans):.6f}",
                "text_input_final": "true" if session.input_complete else "false",
                "alignment_final": "true" if final else "false",
            },
        }

    @staticmethod
    def _segment_event_meta(sa) -> dict[str, str]:
        return {
            "group_idx": str(sa.group_idx),
            "local_idx": str(sa.local_idx),
            "group_final": "true" if sa.group_final else "false",
            "raw_codepoint_start": str(getattr(sa, "raw_start", 0)),
            "raw_codepoint_end": str(getattr(sa, "raw_end", 0)),
            "normalized_codepoint_start": str(
                getattr(sa, "normalized_start", 0)
            ),
            "normalized_codepoint_end": str(getattr(sa, "normalized_end", 0)),
        }

    async def _emit_text_token_events(self, session: Session, actions: list) -> None:
        on_event = session.event_callback
        if on_event is None:
            return
        for sa in actions:
            if sa.action.type not in (ActionType.PREFILL, ActionType.DECODE):
                continue
            if not sa.token_text:
                continue
            token_idx = session.segment_token_emitted_count.get(sa.segment_idx, 0)
            session.segment_token_emitted_count[sa.segment_idx] = token_idx + 1
            await on_event(
                session.session_id,
                {
                    "type": "text_token",
                    "segment_idx": sa.segment_idx,
                    "text": sa.token_text,
                    "meta": {
                        **self._segment_event_meta(sa),
                        "token_idx": str(token_idx),
                        "punct_level": str(Spliter.classify_punct_level(sa.token_text)),
                        "text_complete": "false",
                    },
                },
            )

    async def _emit_text_boundary_events(self, session: Session, actions: list) -> None:
        on_event = session.event_callback
        if on_event is None:
            return
        for sa in actions:
            if sa.action.type not in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                continue
            if sa.segment_idx in session.text_boundary_emitted:
                continue
            session.text_boundary_emitted.add(sa.segment_idx)
            await on_event(
                session.session_id,
                {
                    "type": "text_boundary_commit",
                    "segment_idx": sa.segment_idx,
                    "text": session.segment_texts.get(sa.segment_idx, ""),
                    "meta": {
                        **self._segment_event_meta(sa),
                        "boundary_reason": sa.action.type.value,
                        "text_complete": "true",
                    },
                },
            )

    def _prepare_session_config(self, config: SessionConfig) -> None:
        """Canonicalize session-level prompt text once at session creation."""
        config.instruct_spec = self._tokenize_prompt_text(
            config.instruct, field_name="instruct"
        )
        config.instruct = config.instruct_spec.text if config.instruct_spec else None
        config.ref_text_spec = self._tokenize_prompt_text(
            config.ref_text, field_name="ref_text"
        )
        config.ref_text = config.ref_text_spec.text if config.ref_text_spec else None

    def _tokenize_prompt_text(
        self, text: Optional[str], *, field_name: str
    ) -> Optional[TokenizedText]:
        raw_text = text or ""
        normalized = _normalize_tts_text(raw_text).strip()
        self._log_tokenization_debug(
            field_name,
            raw_text=raw_text,
            normalized_text=normalized,
        )
        if not normalized:
            return None
        return TokenizedText(
            text=normalized,
            token_ids=self._encode_ids(normalized),
        )

    def _tokenize_segment_text(
        self,
        text: str,
        *,
        normalized_offset: int = 0,
        journal: Optional[CanonicalTextJournal] = None,
    ) -> list[SegmentToken]:
        self._log_tokenization_debug(
            "segment_text",
            raw_text=text,
            normalized_text=text,
        )
        if hasattr(self._tokenizer, "encode_with_offsets"):
            ids, offsets = self._tokenizer.encode_with_offsets(
                text, add_special_tokens=False
            )
        else:
            ids, pieces = self._tokenizer.encode_with_text(
                text, add_special_tokens=False
            )
            offsets = []
            cursor = 0
            for piece in pieces:
                start = text.find(piece, cursor)
                if start < 0:
                    start = cursor
                end = min(len(text), start + len(piece))
                offsets.append((start, end))
                cursor = end
        # Keep the frontend safe even for legacy/test tokenizer adapters that
        # expose the raw tokenizers offsets directly instead of the stable
        # LightQwen3TTSTokenizer wrapper.
        stable_offsets = []
        previous_end = 0
        for index, (_start, end) in enumerate(offsets):
            start = min(len(text), previous_end)
            end = min(len(text), max(start, int(end)))
            if index == len(offsets) - 1:
                end = len(text)
            stable_offsets.append((start, end))
            previous_end = end
        return [
            SegmentToken(
                token_id=token_id,
                text=text[start:end],
                normalized_start=normalized_offset + int(start),
                normalized_end=normalized_offset + int(end),
                raw_start=(
                    journal.raw_span(normalized_offset + int(start), normalized_offset + int(end))[0]
                    if journal is not None
                    else normalized_offset + int(start)
                ),
                raw_end=(
                    journal.raw_span(normalized_offset + int(start), normalized_offset + int(end))[1]
                    if journal is not None
                    else normalized_offset + int(end)
                ),
                punct_level=Spliter.classify_punct_level(text[start:end]),
            )
            for token_id, (start, end) in zip(ids, stable_offsets)
        ]

    def _encode_ids(self, text: str) -> list[int]:
        if hasattr(self._tokenizer, "encode_ids"):
            return list(self._tokenizer.encode_ids(text, add_special_tokens=False))
        ids, _ = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        return list(ids)

    def _log_tokenization_debug(
        self,
        field_name: str,
        *,
        raw_text: str,
        normalized_text: str,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        payload = {
            "field": field_name,
            "raw_text": raw_text,
            "raw_text_repr": repr(raw_text),
            "normalized_text": normalized_text,
            "normalized_text_repr": repr(normalized_text),
        }
        if normalized_text:
            snapshot_fn = getattr(self._tokenizer, "debug_snapshot", None)
            if callable(snapshot_fn):
                payload["tokenizer"] = snapshot_fn(
                    normalized_text,
                    add_special_tokens=False,
                )
            else:
                ids, pieces = self._tokenizer.encode_with_text(
                    normalized_text,
                    add_special_tokens=False,
                )
                payload["tokenizer"] = {
                    "ids": [int(token_id) for token_id in ids],
                    "pieces": [
                        {
                            "index": idx,
                            "id": int(token_id),
                            "span": token_text,
                            "span_repr": repr(token_text),
                        }
                        for idx, (token_id, token_text) in enumerate(zip(ids, pieces))
                    ],
                }

        logger.debug(
            "Tokenizer observability: %s",
            json.dumps(payload, ensure_ascii=False),
        )
