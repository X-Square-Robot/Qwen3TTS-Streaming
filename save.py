import time
from queue import Queue, Empty

import numpy as np


class Audio:
    data: bytes
    time_s: float

    def __init__(self, data: bytes, time_s: float):
        self.data = data
        self.time_s = time_s

    def __repr__(self) -> str:
        return f"Audio(time_s={self.time_s:.4f}, bytes={len(self.data)})"


def _make_silence(duration_s: float, sample_rate: int, dtype: np.dtype = np.float32) -> Audio:
    """生成指定时长的静音帧。"""
    num_samples = int(sample_rate * duration_s)
    silence_bytes = np.zeros(num_samples, dtype=dtype).tobytes()
    return Audio(silence_bytes, duration_s)


def save(
    inp: Queue[Audio],
    out: Queue[Audio],
    chunk_s: float = 0.01,
    sample_rate: int = 24000,
) -> None:
    """从 inp 读取音频帧写入 out，模拟实时播放节奏。

    当音频帧之间出现空隙（等待超时或帧时长不足以覆盖已流逝的时间），
    自动填充静音帧，使 out 中的数据流与用户实际听到的音频一致。

    Args:
        inp: 输入音频队列，放入 None 表示结束。
        out: 输出音频队列（含静音填充）。
        chunk_s: 等待输入帧的超时时间（秒），也即静音填充的最小粒度。
        sample_rate: 采样率（Hz），项目默认 24000。
    """
    # 虚拟播放时钟：追踪"播放器"当前应处于的时间位置
    play_clock: float | None = None

    while True:
        wall_now = time.monotonic()

        try:
            audio = inp.get(timeout=chunk_s)
        except Empty:
            # 等待超时 → 没有新帧到达，填充静音
            elapsed = time.monotonic() - wall_now
            out.put(_make_silence(elapsed, sample_rate))
            if play_clock is not None:
                play_clock += elapsed
            continue

        # 收到哨兵值，退出
        if audio is None:
            break

        # 首帧：初始化播放时钟，直接输出
        if play_clock is None:
            out.put(audio)
            play_clock = audio.time_s
            # 模拟首帧播放耗时
            time.sleep(audio.time_s)
            continue

        # 计算从上次播放位置到当前实际流逝的墙钟时间
        elapsed = time.monotonic() - wall_now
        gap = elapsed - play_clock  # 墙钟流逝 - 已播放时长 = 空隙

        # 如果存在空隙（帧到达晚了），先填充静音
        if gap > 0:
            out.put(_make_silence(gap, sample_rate))
            play_clock += gap

        # 输出实际音频帧
        out.put(audio)

        # 如果帧时长 > 剩余等待时间，sleep 模拟播放
        remaining = audio.time_s - (time.monotonic() - wall_now - gap)
        if remaining > 0:
            time.sleep(remaining)
            play_clock = audio.time_s
        else:
            # 帧已"迟到"，播放时钟只推进帧时长
            play_clock += audio.time_s
