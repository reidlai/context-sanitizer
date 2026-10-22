"""Exception hierarchy for context-sanitizer (`Story 4.1`, `FR11`)."""

from __future__ import annotations

try:
    from context_sanitizer._core import (
        InjectionBlockedError,
        SanitizerError,
        UnauthorizedRehydrationError,
    )
except ImportError:  # pragma: no cover

    class SanitizerError(Exception):
        """Base exception for context-sanitizer."""

    class UnauthorizedRehydrationError(SanitizerError):
        """Raised when rehydration is attempted for an unwhitelisted tool destination."""

        def __init__(self, message: str, destination: str = "") -> None:
            super().__init__(message)
            self.destination = destination

    class InjectionBlockedError(SanitizerError):
        """Raised when a prompt injection or exfiltration threat is detected."""

        def __init__(
            self, message: str, score_bps: int = 10000, reason: str = ""
        ) -> None:
            super().__init__(message)
            self.score_bps = score_bps
            self.score = score_bps / 10000.0
            self.reason = reason or message


__all__ = [
    "InjectionBlockedError",
    "SanitizerError",
    "UnauthorizedRehydrationError",
]
