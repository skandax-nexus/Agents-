"""VoiceCore: a from-scratch, self-trained, voice-to-voice engine.

No pretrained models anywhere.  Every weight starts random and is learned only from the
audio the engine hears (self-supervised), from its own exchanges (self-play with
self-scoring) and from optional thumbs-up/down feedback.  It models sound, not text, so
every language is handled the same way.
"""

__version__ = "0.1.0"

from .config import VoiceCoreConfig  # noqa: F401
from .engine import VoiceCore  # noqa: F401
