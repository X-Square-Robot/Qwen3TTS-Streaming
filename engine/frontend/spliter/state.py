from enum import Enum


class SpliterState(Enum):
    HALT = "halt"  # 关机
    IDLE = "idle"  # 空闲

    PREFILL = "prefill"  # 预填充

    TEXT_INPUTING = "text_inputing"  # 文本输入中
    WAITING_TEXT = "waiting_text"  # 等待文本

    PAD_TEXT_EOS = "pad_text_eos"  # 填充文本结束EOS
    PAD_TEXT_NOP = "pad_text_nop"  # 填充文本NOP(没有文本)

    KVCACHE_OVERFLOW = "kvcache_overflow"  # KV缓存溢出


__all__ = ("SpliterState",)
