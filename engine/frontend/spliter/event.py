from dataclasses import dataclass, field
from enum import Enum


class SpliterEventType(Enum):
    UNKNOWN = "unknown"
    START = "start"
    END = "end"

    START_TOKEN = "start_token"
    END_TOKEN = "end_token"
    NORMAL_TOKEN = "normal_token"
    PUNCTUATION_TOKEN = "punctuation_token"


@dataclass
class SpliterEvent:
    type: SpliterEventType = field(default=SpliterEventType.UNKNOWN)
    token: int = field(default=-1)
    text: str = field(default="")
    punct_level: int = field(default=0)  # 0=none, 1=L1(。！？), 2=L2(，；), 3=L3(\n——)


__all__ = ("SpliterEvent", "SpliterEventType")
