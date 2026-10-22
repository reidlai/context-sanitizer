"""Turnkey drop-in middleware for LangChain agents and LLMs (`Story 4.2`, `FR11`)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import context_sanitizer as cs


class LangChainSanitizer:
    """Turnkey drop-in middleware for LangChain agents and LLMs (`Story 4.2`, `FR11`)."""

    def __init__(
        self,
        rate: float = 1.0,
        whitelist: Iterable[str] | None = None,
    ) -> None:
        self.rate = rate
        self.whitelist = list(whitelist) if whitelist is not None else []
        self.last_vault: cs.VaultHandle | None = None

    def wrap(self, model: Any) -> LangChainModelWrapper:
        """Wraps a LangChain ChatModel, pipeline, or LLM."""
        return LangChainModelWrapper(model, self)

    def sanitize_input(self, messages_or_prompt: Any) -> tuple[Any, cs.VaultHandle]:
        """Sanitizes outgoing messages, minifying tools and checking threats."""
        sanitized, vault = cs.sanitize(
            messages_or_prompt, rate=self.rate, whitelist=self.whitelist
        )
        if len(vault) > 0 or self.last_vault is None:
            self.last_vault = vault
        return sanitized, self.last_vault

    def rehydrate_tool_calls(
        self,
        tool_calls: list[Any],
        vault: cs.VaultHandle | None = None,
    ) -> list[Any]:
        """Rehydrates authorized destination tool calls."""
        active_vault = vault or self.last_vault
        if not active_vault:
            return tool_calls

        rehydrated = []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)

            if name and name in self.whitelist and args is not None:
                new_args = active_vault.rehydrate(args, destination=name)
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    tc_copy["args"] = new_args
                    rehydrated.append(tc_copy)
                else:
                    try:
                        tc.args = new_args
                    except (AttributeError, TypeError):
                        pass
                    rehydrated.append(tc)
            else:
                rehydrated.append(tc)
        return rehydrated


class LangChainModelWrapper:
    """Proxy wrapper around a LangChain model providing automated sanitization."""

    def __init__(self, model: Any, sanitizer: LangChainSanitizer) -> None:
        self._model = model
        self._sanitizer = sanitizer

    @property
    def sanitizer(self) -> LangChainSanitizer:
        return self._sanitizer

    @property
    def last_vault(self) -> cs.VaultHandle | None:
        return self._sanitizer.last_vault

    def invoke(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("tools"):
            try:
                kwargs["tools"] = cs.minify_tools(kwargs["tools"])
            except (ValueError, TypeError):
                pass

        sanitized_input, vault = self._sanitizer.sanitize_input(input_data)
        response = self._model.invoke(sanitized_input, *args, **kwargs)

        if hasattr(response, "tool_calls") and response.tool_calls:
            response.tool_calls = self._sanitizer.rehydrate_tool_calls(
                response.tool_calls, vault
            )
        elif isinstance(response, dict) and "tool_calls" in response:
            response["tool_calls"] = self._sanitizer.rehydrate_tool_calls(
                response["tool_calls"], vault
            )

        return response

    async def ainvoke(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("tools"):
            try:
                kwargs["tools"] = cs.minify_tools(kwargs["tools"])
            except (ValueError, TypeError):
                pass

        sanitized_input, vault = self._sanitizer.sanitize_input(input_data)
        if hasattr(self._model, "ainvoke"):
            response = await self._model.ainvoke(sanitized_input, *args, **kwargs)
        else:
            response = self._model.invoke(sanitized_input, *args, **kwargs)

        if hasattr(response, "tool_calls") and response.tool_calls:
            response.tool_calls = self._sanitizer.rehydrate_tool_calls(
                response.tool_calls, vault
            )
        elif isinstance(response, dict) and "tool_calls" in response:
            response["tool_calls"] = self._sanitizer.rehydrate_tool_calls(
                response["tool_calls"], vault
            )

        return response

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            return self.invoke(args[0], *args[1:], **kwargs)
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
