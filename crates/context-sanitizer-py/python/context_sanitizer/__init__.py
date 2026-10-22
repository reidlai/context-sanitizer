"""`context_sanitizer` Python API (`Story 4.1`, `FR11`, `AD-2`)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from context_sanitizer._core import (
    InjectionBlockedError,
    SanitizerError,
    UnauthorizedRehydrationError,
    VaultHandle,
    minify_tools,
)
from context_sanitizer._core import sanitize as _core_sanitize

__version__ = "0.1.0"


def sanitize(
    input_payload: Any,
    rate: float = 1.0,
    whitelist: Iterable[str] | None = None,
) -> tuple[Any, VaultHandle]:
    """Sanitizes raw string, messages, or chat completion request payload.

    Releases the Python GIL during execution (`FR11`, `AD-2`).

    Args:
        input_payload: Raw string, list of messages, or dict.
        rate: Token retention target rate (0.0 to 1.0).
        whitelist: List or iterable of authorized tool destinations.

    Returns:
        tuple[Any, VaultHandle]: Sanitized payload and session vault handle.
    """
    wl = list(whitelist) if whitelist is not None else None

    # Handle LangChain / custom message objects with .content
    if (
        isinstance(input_payload, (list, tuple))
        and input_payload
        and hasattr(input_payload[0], "content")
        and not isinstance(input_payload[0], (dict, str))
    ):
        msg_dicts = []
        for msg in input_payload:
            role = getattr(msg, "type", getattr(msg, "role", "user"))
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            msg_dicts.append(
                {
                    "role": role,
                    "content": str(msg.content),
                }
            )

        sanitized_dicts, vault = _core_sanitize(msg_dicts, rate=rate, whitelist=wl)

        result_messages = []
        for orig_msg, s_dict in zip(input_payload, sanitized_dicts):
            if hasattr(orig_msg, "copy"):
                new_msg = orig_msg.copy(update={"content": s_dict["content"]})
            elif hasattr(orig_msg, "model_copy"):
                new_msg = orig_msg.model_copy(update={"content": s_dict["content"]})
            else:
                try:
                    new_msg = type(orig_msg)(content=s_dict["content"])
                except (TypeError, ValueError, AttributeError):
                    orig_msg.content = s_dict["content"]
                    new_msg = orig_msg
            result_messages.append(new_msg)
        return result_messages, vault

    return _core_sanitize(input_payload, rate=rate, whitelist=wl)


__all__ = [
    "InjectionBlockedError",
    "SanitizerError",
    "UnauthorizedRehydrationError",
    "VaultHandle",
    "__version__",
    "minify_tools",
    "sanitize",
]
