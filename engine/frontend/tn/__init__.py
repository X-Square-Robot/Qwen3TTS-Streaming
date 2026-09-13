"""Narrow control-plane contract for streaming text normalization spans."""
from .state import SpanState
from .events import SpanEvent, SpanEventType
from .fsm import (
    SpanContext,
    SpanFSM,
    SpanStep,
    SpanTransition,
    default_span_transitions,
)
from .recognizer import (
    RecognitionKind,
    RecognizerContext,
    RecognizerRegistry,
    SpanRecognition,
    SpanRecognizer,
)
from .recognizers import CandidateRecognizer, ClassifierRecognizer
from .candidates import WfstCandidateProvider
from .driver import CommitterLike, SpanDriver, SpanDriverResult

__all__ = (
    "SpanState",
    "SpanEvent",
    "SpanEventType",
    "SpanContext",
    "SpanFSM",
    "SpanStep",
    "SpanTransition",
    "default_span_transitions",
    "RecognitionKind",
    "RecognizerContext",
    "RecognizerRegistry",
    "SpanRecognition",
    "SpanRecognizer",
    "ClassifierRecognizer",
    "CandidateRecognizer",
    "WfstCandidateProvider",
    "CommitterLike",
    "SpanDriver",
    "SpanDriverResult",
)
