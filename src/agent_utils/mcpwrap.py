# Usage examples:
#
# 1. Keep one stdio MCP server open for multiple calls:
#
#     import asyncio
#     from agent_utils.mcpwrap import StdioMCPSession
#
#     async def main() -> None:
#         config = {"command": "python", "args": ["server.py"]}
#
#         async with StdioMCPSession(config, initialize_timeout=30) as mcp:
#             print(mcp.initialize_result.serverInfo)
#
#             tools = await mcp.list_tools(timeout=10)
#             print([tool.name for tool in tools.tools])
#
#             result = await mcp.call_tool(
#                 "tool_name",
#                 {"key": "value"},
#                 timeout=60,
#             )
#             print(result)
#
#     asyncio.run(main())
#
# 2. Manually open and close when async with is inconvenient:
#
#     import asyncio
#     from agent_utils.mcpwrap import StdioMCPSession
#
#     async def main() -> None:
#         mcp = StdioMCPSession({"command": "python", "args": ["server.py"]})
#         await mcp.open()
#
#         try:
#             tools = await mcp.list_tools(timeout=10)
#             print([tool.name for tool in tools.tools])
#
#             result = await mcp.call_tool(
#                 "tool_name",
#                 {"key": "value"},
#                 timeout=60,
#             )
#             print(result)
#         finally:
#             await mcp.aclose()
#
#     asyncio.run(main())
#
# 3. Open a server, call one tool, then close it automatically:
#
#     import asyncio
#     from agent_utils.mcpwrap import call_stdio_tool
#
#     async def main() -> None:
#         result = await call_stdio_tool(
#             {"command": "python", "args": ["server.py"]},
#             "tool_name",
#             {"key": "value"},
#             initialize_timeout=30,
#             timeout=60,
#         )
#         print(result)
#
#     asyncio.run(main())
#
# Config may be either a dict accepted by StdioServerParameters or an existing
# StdioServerParameters instance. Common dict keys are command, args, env, cwd,
# encoding, and encoding_error_handler. Timeout values are seconds; None means
# no timeout. Stdio MCP servers must write protocol messages only to stdout;
# logs and debug output should go to stderr.

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from datetime import timedelta
from math import isfinite
from types import TracebackType
from typing import Any, Self, TypeVar

from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.shared.session import ProgressFnT

import simpidlog


StdioMCPConfig = Mapping[str, Any] | StdioServerParameters

_T = TypeVar("_T")


class StdioMCPSession:
    """Async lifecycle wrapper for a stdio MCP server.

    Example:
        async with StdioMCPSession({"command": "python", "args": ["server.py"]}) as mcp:
            result = await mcp.call_tool("tool_name", {"key": "value"})
    """

    def __init__(
        self,
        config: StdioMCPConfig,
        *,
        initialize_timeout: float | None = 30.0,
    ) -> None:
        self.params = (
            config
            if isinstance(config, StdioServerParameters)
            else StdioServerParameters(**dict(config))
        )
        self.initialize_timeout = initialize_timeout

        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._initialize_result: mcp_types.InitializeResult | None = None

    @property
    def is_open(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("StdioMCPSession is not open; use 'async with' or await open().")
        return self._session

    @property
    def initialize_result(self) -> mcp_types.InitializeResult:
        if self._initialize_result is None:
            raise RuntimeError("StdioMCPSession has not been initialized; use 'async with' or await open().")
        return self._initialize_result

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self._close(exc_type, exc, tb)

    async def open(self) -> Self:
        if self._session is not None:
            raise RuntimeError("StdioMCPSession is already open.")
        if self.initialize_timeout is not None:
            self._validate_timeout(self.initialize_timeout)

        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(self.params))
            session = await stack.enter_async_context(ClientSession(read, write))
            initialize_result = await self._call_with_timeout(
                session.initialize,
                self.initialize_timeout,
            )
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session
        self._initialize_result = initialize_result
        return self

    async def aclose(self) -> None:
        await self._close(None, None, None)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        progress_callback: ProgressFnT | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> mcp_types.CallToolResult:
        return await self.session.call_tool(
            name=name,
            arguments=dict(arguments) if arguments is not None else None,
            read_timeout_seconds=self._timeout_delta(timeout),
            progress_callback=progress_callback,
            meta=dict(meta) if meta is not None else None,
        )

    async def list_tools(
        self,
        cursor: str | None = None,
        *,
        params: mcp_types.PaginatedRequestParams | None = None,
        timeout: float | None = None,
    ) -> mcp_types.ListToolsResult:
        session = self.session
        return await self._call_with_timeout(
            lambda: session.list_tools(cursor = cursor, params =params),
            timeout,
        )

    async def _close(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        stack = self._stack
        self._stack = None
        self._session = None
        self._initialize_result = None

        if stack is None:
            return None

        return await stack.__aexit__(exc_type, exc, tb)

    async def _call_with_timeout(
        self,
        make_awaitable: Callable[[], Awaitable[_T]],
        timeout: float | None,
    ) -> _T:
        if timeout is None:
            return await make_awaitable()
        self._validate_timeout(timeout)
        return await asyncio.wait_for(make_awaitable(), timeout = timeout)

    def _timeout_delta(self, timeout: float | None) -> timedelta | None:
        if timeout is None:
            return None
        self._validate_timeout(timeout)
        return timedelta(seconds = timeout)

    def _validate_timeout(self, timeout: float) -> None:
        if timeout < 0 or not isfinite(timeout):
            errmsg = "timeout must be a finite non-negative number or None."
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise ValueError(errmsg)


async def call_stdio_tool(
    config: StdioMCPConfig,
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    initialize_timeout: float | None = 30.0,
    timeout: float | None = None,
    progress_callback: ProgressFnT | None = None,
    meta: Mapping[str, Any] | None = None,
) -> mcp_types.CallToolResult:
    """Open a stdio MCP server, call one tool, then close it."""

    async with StdioMCPSession(config, initialize_timeout = initialize_timeout) as session:
        return await session.call_tool(
            name,
            arguments,
            timeout = timeout,
            progress_callback = progress_callback,
            meta = meta,
        )


__all__ = ["StdioMCPConfig", "StdioMCPSession", "call_stdio_tool"]
