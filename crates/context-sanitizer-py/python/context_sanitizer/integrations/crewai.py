"""Turnkey drop-in middleware for CrewAI tasks and agents (`Story 4.2`, `FR11`)."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

import context_sanitizer as cs


class CrewAISanitizer:
    """Turnkey drop-in middleware for CrewAI tasks and agents (`Story 4.2`, `FR11`)."""

    def __init__(
        self,
        rate: float = 1.0,
        whitelist: Iterable[str] | None = None,
    ) -> None:
        self.rate = rate
        self.whitelist = list(whitelist) if whitelist is not None else []
        self.last_vault: cs.VaultHandle | None = None

    def wrap_task(self, task: Any) -> Any:
        """Wraps a CrewAI Task, sanitizing descriptions and protecting tool arguments."""
        desc = getattr(task, "description", None)
        if desc and isinstance(desc, str):
            clean_desc, vault = cs.sanitize(
                desc, rate=self.rate, whitelist=self.whitelist
            )
            task.description = clean_desc
            if len(vault) > 0 or self.last_vault is None:
                self.last_vault = vault

        expected_output = getattr(task, "expected_output", None)
        if expected_output and isinstance(expected_output, str):
            clean_exp, vault = cs.sanitize(
                expected_output, rate=self.rate, whitelist=self.whitelist
            )
            task.expected_output = clean_exp
            if len(vault) > 0 or self.last_vault is None:
                self.last_vault = vault

        tools = getattr(task, "tools", None)
        if tools and isinstance(tools, (list, tuple)):
            task.tools = [self._wrap_tool(tool) for tool in tools]

        for exec_name in ("execute", "execute_sync"):
            orig_exec = getattr(task, exec_name, None)
            if orig_exec and callable(orig_exec):
                setattr(task, exec_name, self._wrap_execution(orig_exec))

        return task

    def wrap(self, target: Any) -> Any:
        """Convenience method wrapping tasks or agents."""
        if hasattr(target, "description"):
            return self.wrap_task(target)
        return self.wrap_agent(target)

    def wrap_agent(self, agent: Any) -> Any:
        """Wraps a CrewAI Agent, protecting tool calls and execution."""
        tools = getattr(agent, "tools", None)
        if tools and isinstance(tools, (list, tuple)):
            agent.tools = [self._wrap_tool(tool) for tool in tools]
        return agent

    def _wrap_tool(self, tool: Any) -> Any:
        tool_name = getattr(tool, "name", str(tool))
        is_authorized = tool_name in self.whitelist

        for attr in ("_run", "run", "func"):
            orig_func = getattr(tool, attr, None)
            if orig_func and callable(orig_func):
                setattr(
                    tool,
                    attr,
                    self._wrap_tool_callable(orig_func, tool_name, is_authorized),
                )

        if callable(tool) and not getattr(tool, "_cs_wrapped", False):
            orig_call = tool.__call__
            tool.__call__ = self._wrap_tool_callable(
                orig_call, tool_name, is_authorized
            )
            tool._cs_wrapped = True

        return tool

    def _wrap_tool_callable(
        self, func: Callable[..., Any], tool_name: str, is_authorized: bool
    ) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active_vault = self.last_vault
            if active_vault:
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, (str, dict, list)) and ("<SHADOW:" in str(v)):
                        new_kwargs[k] = active_vault.rehydrate(v, destination=tool_name)
                    else:
                        new_kwargs[k] = v
                new_args = []
                for a in args:
                    if isinstance(a, (str, dict, list)) and ("<SHADOW:" in str(a)):
                        new_args.append(
                            active_vault.rehydrate(a, destination=tool_name)
                        )
                    else:
                        new_args.append(a)
                return func(*new_args, **new_kwargs)
            return func(*args, **kwargs)

        return wrapper

    def _wrap_execution(self, orig_exec: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(orig_exec)
        def exec_wrapper(*args: Any, **kwargs: Any) -> Any:
            return orig_exec(*args, **kwargs)

        return exec_wrapper
