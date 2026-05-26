# agent_utils

Utilities for AI agent development.

## Features

- `OnlineLLM`: async helpers for OpenAI-compatible chat and responses APIs.
- File attachment support for paths, byte dictionaries, and byte lists.
- Helpers for extracting JSON objects and fenced code blocks from LLM output.
- `StdioMCPSession`: async lifecycle wrapper for stdio MCP servers.
- `call_stdio_tool`: convenience helper for one-shot stdio MCP tool calls.

## Installation

```bash
pip install -e .
```

## Configuration

`OnlineLLM` reads configuration from explicit constructor arguments first, then
from environment variables.

Supported primary names:

- `BASEURL`
- `APIKEY`
- `MODEL`

Supported OpenAI-compatible fallback names:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Local `.env` files are intentionally ignored by git.

## Usage

```python
from agent_utils import OnlineLLM

llm = OnlineLLM()
text = await llm.call_compatible(
    system_prompt="You are a concise assistant.",
    user_prompt="Say hello.",
    temperature=0,
)
```

```python
from agent_utils.mcpwrap import StdioMCPSession

async with StdioMCPSession({"command": "python", "args": ["server.py"]}) as mcp:
    tools = await mcp.list_tools(timeout=10)
    result = await mcp.call_tool("tool_name", {"key": "value"}, timeout=60)
```

## Tests

```bash
python src/agent_utils/test.llmapi.py/main.py
```
