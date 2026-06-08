# agent_utils

`agent_utils` 是一组面向 AI Agent 开发的 Python 实用工具。它目前主要覆盖三类高频需求：调用 OpenAI / OpenAI-compatible LLM API、通过视觉模型把 PDF 分段转成结构化 Markdown、以及用更安全的生命周期包装 stdio MCP server。

这个项目不是一个大而全的 Agent 框架，而是把实际项目中经常重复写、容易写错的底层能力沉淀成可复用模块。

## 核心能力

- `OnlineLLM`：异步 LLM 调用封装，支持 Chat Completions 兼容接口与 OpenAI Responses API。
- 图片上传：支持从路径、`dict[name, bytes]`、`list[bytes]` 上传图片，并统一转为对应 API 的 data URL payload。
- 流式输出：兼容 Chat Completions stream chunk 与 Responses `response.output_text.delta` 事件。
- Responses Web Search：可在 Responses API 中启用 `web_search` 工具。
- 输出解析：提供 JSON 对象解析、Markdown fenced code block 提取工具。
- `StdioMCPSession`：stdio MCP server 的异步生命周期封装，避免手动管理初始化、超时和关闭。
- `pdf2markdown`：把 PDF 渲染成 PNG 后调用视觉模型，自动识别目录、分节提取内容，并导出 Markdown。

## 适用场景

- 你需要在自己的 Agent 项目里快速接入 OpenAI 或兼容 OpenAI 的模型网关。
- 你需要把图片输入传给视觉模型，但不想每次手写 base64、MIME 检测和 OpenAI payload。
- 你需要处理模型返回的 JSON 或 Markdown 代码块。
- 你需要在 Python 里调用 stdio MCP server，并希望调用结束后可靠释放资源。
- 你需要把教材、论文、书籍类 PDF 拆成适合 LLM 阅读的 Markdown 片段。

## 安装

本项目目前更适合以源码 editable 模式使用：

```bash
git clone https://github.com/Simpidbit/agent_utils.git
cd agent_utils
pip install -e .
```

Python 版本要求：`>=3.11`。

`pyproject.toml` 中声明的主要依赖包括：

```text
openai
python-dotenv
python-magic
markdown
beautifulsoup4
mcp
simpidlog
pymupdf
```

如果使用 `pdf2markdown`，当前源码还会用到 `typeguard`。如果你的环境里没有它，请额外安装：

```bash
pip install typeguard
```

`python-magic` 依赖系统的 `libmagic`。如果 MIME 检测报错，需要先安装系统包，例如：

```bash
# Debian / Ubuntu
sudo apt-get install libmagic1

# Fedora
sudo dnf install file-libs

# macOS
brew install libmagic
```

## LLM 配置

`OnlineLLM` 会优先读取构造函数显式传入的参数；未传入时，再从环境变量读取配置。

主要环境变量：

```bash
BASEURL="https://api.openai.com/v1"
APIKEY="sk-..."
MODEL="gpt-4.1"
```

也支持 OpenAI 风格的 fallback 名称：

```bash
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4.1"
```

源码会读取 `src/agent_utils/.env`，也可以直接使用当前 shell 的环境变量。`.env`、`.env.*` 已在 `.gitignore` 中忽略，不应提交密钥。

## 快速开始

```python
import asyncio

from agent_utils import OnlineLLM


async def main() -> None:
    async with OnlineLLM() as llm:
        text = await llm.call_compatible(
            system_prompt="你是一个简洁的助手。",
            user_prompt="用一句话介绍 agent_utils。",
            temperature=0.0,
        )
        print(text)


asyncio.run(main())
```

如果你不想使用环境变量，也可以显式传入配置：

```python
llm = OnlineLLM(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model_id="gpt-4.1",
)
```

推荐用 `async with OnlineLLM() as llm:`，这样退出作用域时会自动关闭底层 `AsyncOpenAI` client。也可以手动调用：

```python
llm = OnlineLLM()
try:
    result = await llm.call_responses(
        system_prompt=None,
        user_prompt="Hello",
        temperature=0.0,
    )
finally:
    await llm.close()
```

## OnlineLLM

`OnlineLLM` 是项目最核心的类，位于 `agent_utils.llmapi`，并在包根目录导出：

```python
from agent_utils import OnlineLLM
```

构造函数：

```python
OnlineLLM(
    base_url: str | None = None,
    api_key: str | None = None,
    model_id: str | None = None,
    client: AsyncOpenAI | None = None,
    timeout: httpx.Timeout | float | None = None,
)
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `base_url` | API base URL；不传时读取 `BASEURL` 或 `OPENAI_BASE_URL`。 |
| `api_key` | API key；不传时读取 `APIKEY` 或 `OPENAI_API_KEY`。 |
| `model_id` | 模型 ID；不传时读取 `MODEL` 或 `OPENAI_MODEL`。 |
| `client` | 可注入已有 `AsyncOpenAI` client，方便测试或复用连接。 |
| `timeout` | 传给 `AsyncOpenAI` 的 timeout；默认使用较长读写超时以适配长任务。 |

如果没有注入 `client`，则必须提供 `base_url`、`api_key`、`model_id`。如果注入了 `client`，仍然需要 `model_id`，因为调用 API 时必须指定模型。

### Chat Completions 兼容接口

`call_compatible` 调用 `client.chat.completions.create(...)`，适用于 OpenAI Chat Completions 或多数 OpenAI-compatible 网关。

```python
text = await llm.call_compatible(
    system_prompt="你是一个严谨的 Python 代码助手。",
    user_prompt="解释 pathlib.Path.read_text 的作用。",
    temperature=0.0,
    effort="medium",
    stream=False,
)
```

方法签名：

```python
await llm.call_compatible(
    system_prompt: str | None,
    user_prompt: str,
    temperature: float,
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium",
    file_paths: list[str | os.PathLike] | tuple[str | os.PathLike, ...] | None = None,
    file_bins: dict[str | os.PathLike, bytes] | list[bytes] | None = None,
    stream: bool = False,
) -> str
```

注意：`effort` 会作为 `reasoning_effort` 传给 Chat Completions API。不同供应商对该参数的支持不完全一致；如果某个兼容网关不支持，需要在调用层选择支持该参数的模型或网关。

### OpenAI Responses API

`call_responses` 调用 `client.responses.create(...)`，适用于 OpenAI Responses API，并支持内置 web search 工具。

```python
text = await llm.call_responses(
    system_prompt="你是一个研究助手。",
    user_prompt="总结最近一年 Python 包管理工具的发展趋势。",
    temperature=0.2,
    effort="medium",
    web_search="medium",
    tools_required=False,
)
```

方法签名：

```python
await llm.call_responses(
    system_prompt: str | None,
    user_prompt: str,
    temperature: float,
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium",
    web_search: Literal["none", "low", "medium", "high"] = "none",
    tools_required: bool = False,
    file_paths: list[str | os.PathLike] | tuple[str | os.PathLike, ...] | None = None,
    file_bins: dict[str | os.PathLike, bytes] | list[bytes] | None = None,
    stream: bool = False,
) -> str
```

当 `web_search != "none"` 时，会发送如下工具配置：

```python
tools = [{
    "type": "web_search",
    "search_context_size": web_search,
}]
```

`tools_required=True` 时会把 `tool_choice` 设为 `"required"`；否则为 `"auto"`。

### 流式调用

两个调用方法都支持 `stream=True`：

```python
text = await llm.call_responses(
    system_prompt=None,
    user_prompt="写一段 200 字的说明。",
    temperature=0.7,
    stream=True,
)
```

目前封装会在内部消费完整 stream，最后仍然返回一个完整 `str`。也就是说，它不是一个 async generator，而是“以流式方式接收、以完整字符串返回”。这样可以兼容较慢或较长的请求，同时保持调用方接口简单。

内部已兼容两类流式事件：

| API | 支持的增量格式 |
| --- | --- |
| Chat Completions | `choices[].delta.content` |
| Responses | `response.output_text.delta` |

## 图片上传

`agent_utils` 目前只支持图片上传，不支持通用文件上传。所有上传内容都会检查 MIME，只有 `image/*` 会被接受。

支持三种输入方式：

### 1. 从路径上传

```python
text = await llm.call_responses(
    system_prompt="你是一个图像理解助手。",
    user_prompt="请描述这张图。",
    temperature=0.0,
    file_paths=["/path/to/image.png"],
)
```

`file_paths` 必须是 list 或 tuple，元素必须是 `str` 或 `PathLike`。单独传一个字符串会被视为非法输入，因为它很容易误写。

### 2. 从 dict 上传 bytes

```python
image_bytes = Path("page.png").read_bytes()

text = await llm.call_responses(
    system_prompt=None,
    user_prompt="提取图片中的文字。",
    temperature=0.0,
    file_bins={"page.png": image_bytes},
)
```

dict 的 key 会作为图片名参与 MIME 判断。代码优先使用 binary MIME；如果 binary MIME 是 `application/octet-stream`，且扩展名能判断为 `image/*`，会用扩展名作为兜底。

### 3. 从 list 上传 bytes

```python
text = await llm.call_compatible(
    system_prompt="你是一个 OCR 助手。",
    user_prompt="读取这些图片中的文字。",
    temperature=0.0,
    file_bins=[image_bytes_1, image_bytes_2],
)
```

list 形式没有文件名，因此只能依赖 binary MIME 判断，并自动生成类似 `1.png`、`2.jpeg` 的名称。

### 上传限制和失败策略

| 行为 | 说明 |
| --- | --- |
| 总大小限制 | 所有上传图片总大小不能超过 50 MiB。 |
| MIME 限制 | 只允许 `image/*`，例如 `image/png`、`image/jpeg`、`image/webp`。 |
| 非图片 | 抛出 `ValueError`，不会静默忽略。 |
| 参数类型错误 | 抛出 `TypeError`，例如 `file_paths="a.png"` 或 `file_bins={"a.png": "not bytes"}`。 |
| 空响应 | 如果模型返回空字符串或只有空白字符，会抛出 `RuntimeError`。 |

## 输出解析工具

### parse_json

`parse_json` 用来从模型输出中解析 JSON object。它会处理常见的 fenced JSON 代码块包装，也会尝试截取第一个 `{...}` 区间。

```python
raw = await llm.call_responses(
    system_prompt="你只能输出 JSON。",
    user_prompt="返回一个包含 name 和 score 的对象。",
    temperature=0.0,
)

data = llm.parse_json(raw)
```

要求最终结果必须是 JSON object，也就是 Python `dict`。如果模型返回数组、字符串、非法 JSON，会抛出 `RuntimeError`。

### parse_codeblock

`parse_codeblock` 用来提取 Markdown 代码围栏中的指定语言代码：

```python
codes = llm.parse_codeblock(markdown_text, "python")
```

例如下面的文本会提取出 Python 代码：

````markdown
```python
print("hello")
```
````

## MCP stdio 封装

`agent_utils.mcpwrap` 提供 `StdioMCPSession` 和 `call_stdio_tool`，用于调用 stdio MCP server。

```python
import asyncio

from agent_utils.mcpwrap import StdioMCPSession


async def main() -> None:
    config = {
        "command": "python",
        "args": ["server.py"],
    }

    async with StdioMCPSession(config, initialize_timeout=30) as mcp:
        tools = await mcp.list_tools(timeout=10)
        print([tool.name for tool in tools.tools])

        result = await mcp.call_tool(
            "tool_name",
            {"key": "value"},
            timeout=60,
        )
        print(result)


asyncio.run(main())
```

如果只需要调用一次工具，可以使用 `call_stdio_tool`：

```python
from agent_utils.mcpwrap import call_stdio_tool

result = await call_stdio_tool(
    {"command": "python", "args": ["server.py"]},
    "tool_name",
    {"key": "value"},
    initialize_timeout=30,
    timeout=60,
)
```

配置可以是 `mcp.StdioServerParameters`，也可以是能传给 `StdioServerParameters(**config)` 的 dict。常见 key 包括：

| key | 说明 |
| --- | --- |
| `command` | MCP server 启动命令。 |
| `args` | 命令参数列表。 |
| `env` | 传给子进程的环境变量。 |
| `cwd` | 子进程工作目录。 |
| `encoding` | stdio 编码。 |
| `encoding_error_handler` | 编码错误处理策略。 |

注意：stdio MCP server 必须只把协议消息写到 stdout；日志和调试信息应该写到 stderr，否则会污染 MCP 协议流。

## PDF 转 Markdown

`src/agent_utils/pdf2markdown.py` 是一个面向教材、书籍、长 PDF 的实验性命令行工具。它的基本思路是：

1. 用 PyMuPDF 把 PDF 每页渲染成 PNG。
2. 调用视觉模型判断目录页范围。
3. 从目录页中提取结构化目录 JSON。
4. 根据目录把正文拆成若干章节区间。
5. 分节上传 PNG，让模型提取 Markdown 文本。
6. 支持失败后按章节重试，并最终导出多个 `.md` 文件。

这个工具更适合交互式、本地使用，而不是无人工参与的生产流水线。它会在关键步骤要求人工确认，例如检查目录 JSON、输入正文第一页偏移量等。

### 命令格式

建议从仓库根目录运行：

```bash
python -m agent_utils.pdf2markdown <cmd> <target> --output <output-dir> [options]
```

支持的命令：

| 命令 | 作用 |
| --- | --- |
| `extract` | 从 PDF 开始提取，生成目录 JSON、提取状态 pkl 和最终 JSON。 |
| `print` | 打印已提取 JSON 的章节树和每节文本长度。 |
| `retry` | 从 `.pkl` 恢复，手动选择若干章节重新提取。 |
| `export` | 从最终 JSON 导出分章节 Markdown 文件。 |

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output` | 无 | 输出目录。当前代码假设目录已存在。 |
| `--workers` | `20` | PDF 渲染为 PNG 时使用的进程数。 |
| `--dpi` | `144` | PDF 页面渲染 DPI。 |
| `--logdir` | `/tmp/agent_utils/pdf2markdown` | simpidlog 日志目录。 |
| `--restore` | 无 | `extract` 时从已有 `.pkl` 恢复。 |

### 示例流程

创建输出目录：

```bash
mkdir -p output/book
```

首次提取：

```bash
python -m agent_utils.pdf2markdown extract ./book.pdf --output ./output/book
```

程序会先渲染 PDF，然后尝试识别目录范围并生成：

```text
output/book/book.content.json
output/book/book.pkl
output/book/book.json
```

查看章节树：

```bash
python -m agent_utils.pdf2markdown print ./output/book/book.json
```

重试部分章节：

```bash
python -m agent_utils.pdf2markdown retry ./output/book/book.pkl --output ./output/book
```

进入交互后输入要重试的章节编号，例如：

```text
1.2
3.4.1
ok
```

导出 Markdown：

```bash
mkdir -p output/book/md
python -m agent_utils.pdf2markdown export ./output/book/book.json --output ./output/book/md
```

### PDF 工具注意事项

- 需要配置可用的视觉模型，且模型必须支持图片输入。
- 目录识别和内容提取依赖模型能力，复杂排版、扫描质量差、目录页不规范时可能需要人工修正或重试。
- 上传图片总大小受 `OnlineLLM` 的 50 MiB 限制影响；页数过多或 DPI 过高时可能失败。
- `--dpi` 越高，OCR/视觉效果可能越好，但图片更大、成本更高、速度更慢。
- 当前实现会并发提取多个章节，请注意模型网关的速率限制。

## 异常与日志

项目使用 `simpidlog` 记录错误和调试信息。`llmapi.py` 中的基础库逻辑避免在异常路径直接 `print` 或输出 traceback，便于在上层应用中控制日志流。

常见异常：

| 异常 | 场景 |
| --- | --- |
| `RuntimeError` | LLM 配置缺失、API 调用失败、模型返回空内容、JSON 解析失败。 |
| `TypeError` | `file_paths` / `file_bins` / messages 等调用参数类型错误。 |
| `ValueError` | 上传内容不是 `image/*`。 |
| `FileSizeExceededError` | 上传图片总大小超过 50 MiB。 |

## 测试

当前主要测试集中在 `llmapi.py`：

```bash
python -m py_compile src/agent_utils/llmapi.py src/agent_utils/test.llmapi.py/main.py
python src/agent_utils/test.llmapi.py/main.py
```

也可以只跑部分 unittest class：

```bash
python src/agent_utils/test.llmapi.py/main.py FileEncodingTests CompatibleMessageTests ResponsesMessageTests
```

如果修改 README 或 Markdown 内容，可以额外检查 diff 中是否有多余空白：

```bash
git diff --check
```

## 项目结构

```text
agent_utils/
├── README.md
├── pyproject.toml
└── src/
    └── agent_utils/
        ├── __init__.py
        ├── llmapi.py
        ├── mcpwrap.py
        ├── pdf2markdown.py
        ├── deai.py
        ├── test.llmapi.py/
        │   └── main.py
        ├── test_llmapi.py
        └── test.py
```

主要文件说明：

| 文件 | 说明 |
| --- | --- |
| `src/agent_utils/llmapi.py` | LLM API 封装、图片上传、流式响应解析、JSON/代码块解析。 |
| `src/agent_utils/mcpwrap.py` | stdio MCP server 的异步 session 封装。 |
| `src/agent_utils/pdf2markdown.py` | PDF 渲染、目录识别、分节提取、Markdown 导出工具。 |
| `src/agent_utils/__init__.py` | 导出 `OnlineLLM` 和 `StdioMCPSession`。 |
| `src/agent_utils/test.llmapi.py/main.py` | `llmapi.py` 的主要 unittest 测试。 |
| `src/agent_utils/deai.py` | 早期草稿代码，当前不建议作为稳定 API 使用。 |

## 开发说明

- 优先保持工具函数小而明确，不把项目扩展成重型框架。
- 对基础库代码，非法输入应早失败，而不是静默忽略。
- 对 LLM 返回内容，日志里尽量不要记录完整正文，避免泄露敏感数据。
- 对上传文件，当前策略是 image-only；不要重新引入通用 file payload，除非上层有明确需求并补齐测试。
- 修改 `llmapi.py` 后，至少运行完整 `src/agent_utils/test.llmapi.py/main.py`。

## License

项目在 `pyproject.toml` 中声明为 MIT License。
