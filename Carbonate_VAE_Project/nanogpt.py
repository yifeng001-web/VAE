"""Compatibility for existing checkpoints that pickle nanogpt.GPTConfig."""

from thesis_code.models.nanogpt import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
