"""Turnkey drop-in middleware for AutoGen agents (`Story 4.2`, `FR11`)."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

import context_sanitizer as cs


class AutoGenSanitizer:
    """Turnkey drop-in middleware for AutoGen agents (`Story 4.2`, `FR11`)."""

    def __init__(
        self,
        rate: float = 1.0,
        whitelist: Iterable[str] | None = None,
    ) -> None:
        self.rate = rate
        self.whitelist = list(whitelist) if whitelist is not None else []
        self.last_vault: cs.VaultHandle | None = None

    def wrap_agent(self, agent: Any) -> Any:
        """Wraps an AutoGen agent, sanitizing incoming messages and protecting tool execution."""
        orig_generate_reply = getattr(agent, "generate_reply", None)
        if orig_generate_reply and callable(orig_generate_reply):
            agent.generate_reply = self._wrap_generate_reply(orig_generate_reply)

        orig_receive = getattr(agent, "receive", None)
        if orig_receive and callable(orig_receive):
            agent.receive = self._wrap_receive(orig_receive)

        orig_exec_fn = getattr(agent, "execute_function", None)
        if orig_exec_fn and callable(orig_exec_fn):
            agent.execute_function = self._wrap_execute_function(orig_exec_fn)

        return agent

    def wrap(self, target: Any) -> Any:
        """Convenience alias for wrap_agent."""
        return self.wrap_agent(target)

    def _wrap_generate_reply(self, orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(orig_fn)
        def wrapper(
            messages: list[dict[str, Any]] | None = None,
            sender: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active_messages = messages
            if active_messages is not None:
                sanitized_msgs, vault = cs.sanitize(
                    active_messages, rate=self.rate, whitelist=self.whitelist
                )
                if len(vault) > 0 or self.last_vault is None:
                    self.last_vault = vault
                active_messages = sanitized_msgs

            reply = orig_fn(*args, messages=active_messages, sender=sender, **kwargs)

            if isinstance(reply, dict) and self.last_vault:
                fn_call = reply.get("function_call")
                if isinstance(fn_call, dict):
                    fn_name = fn_call.get("name", "")
                    if fn_name in self.whitelist and "arguments" in fn_call:
                        fn_call["arguments"] = self.last_vault.rehydrate(
                            fn_call["arguments"], destination=fn_name
                        )

                tool_calls = reply.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn_obj = tc.get("function")
                            if isinstance(fn_obj, dict):
                                tc_name = fn_obj.get("name", "")
                                if tc_name in self.whitelist and "arguments" in fn_obj:
                                    fn_obj["arguments"] = self.last_vault.rehydrate(
                                        fn_obj["arguments"], destination=tc_name
                                    )
                            else:
                                tc_name = tc.get("name", "")
                                if tc_name in self.whitelist and "args" in tc:
                                    tc["args"] = self.last_vault.rehydrate(
                                        tc["args"], destination=tc_name
                                    )

            return reply

        return wrapper

    def _wrap_receive(self, orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(orig_fn)
        def wrapper(
            message: Any,
            sender: Any,
            request_reply: bool | None = None,
            silent: bool | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            sanitized_msg = message
            if isinstance(message, (str, dict, list)):
                clean_msg, vault = cs.sanitize(
                    message, rate=self.rate, whitelist=self.whitelist
                )
                self.last_vault = vault
                sanitized_msg = clean_msg

            return orig_fn(
                sanitized_msg,
                sender,
                *args,
                request_reply=request_reply,
                silent=silent,
                **kwargs,
            )

        return wrapper

    def _wrap_execute_function(self, orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(orig_fn)
        def wrapper(func_call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            active_vault = self.last_vault
            if active_vault and isinstance(func_call, dict):
                func_name = func_call.get("name", "")
                func_args = func_call.get("arguments", "")
                if func_name in self.whitelist and func_args:
                    func_call = dict(func_call)
                    func_call["arguments"] = active_vault.rehydrate(
                        func_args, destination=func_name
                    )
                elif func_name not in self.whitelist and ("<SHADOW:" in str(func_args)):
                    active_vault.rehydrate(func_args, destination=func_name)

            return orig_fn(func_call, *args, **kwargs)

        return wrapper
