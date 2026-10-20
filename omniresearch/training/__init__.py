"""Self-training: turn the system's own verified work into a better local model.

    traces + experience + feedback + self-practice  ->  dataset (SFT + preference pairs)
    -> LoRA fine-tune of a local base model -> evaluate against the base -> export to Ollama
"""
from .dataset import DatasetBuilder, Example, PreferencePair
from .practice import PracticeLoop

__all__ = ["DatasetBuilder", "Example", "PreferencePair", "PracticeLoop"]
