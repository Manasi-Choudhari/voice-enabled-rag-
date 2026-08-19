"""Speech-to-text: provider-agnostic interface."""

from stt.base import STTProvider
from stt.factory import getSTTProvider

__all__ = ["STTProvider", "getSTTProvider"]
