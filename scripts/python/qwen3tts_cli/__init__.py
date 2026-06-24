"""Qwen3-TTS CLI — Python lifecycle manager.

All subcommands call Python modules directly. No Bash forwarding.

Usage::

    qwen3tts all -m custom-1.7b
    qwen3tts setup -m custom-1.7b
    qwen3tts build -m custom-1.7b --max-batch-size 64
    qwen3tts package -m custom-1.7b
    qwen3tts run --gateway standalone
    qwen3tts stop
    qwen3tts status
    qwen3tts probe
    qwen3tts download
"""

from qwen3tts_cli.main import main

__all__ = ["main"]
