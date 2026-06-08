import os
import re
import json
import base64
import typing
import uuid
import inspect
from magic import from_buffer as analyse_mime_from_bin
import mimetypes
import httpx

from typing import Annotated, Literal
from markdown import markdown
from bs4 import BeautifulSoup

from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import OpenAIError

import simpidlog

load_dotenv(Path(__file__).with_name(".env"))

_ERROR_PREFIX = '@Simpidbit/agent_utils/llmapi.py\n'

_FILE_PATHS_ANNOTATION = Annotated[
    list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...] | None,
    'Image file full path'
]

_FILE_BINS_ANNOTATION = Annotated[
    dict[str | os.PathLike[str], bytes] | list[bytes] | None,
    'dict[str | Path, bytes]: key is the image file path/name, value is the binary image bytes.\n'
    'list[bytes]: List of image bytes, while filename from python-magic guessing.'
]

_NAME_MIME_B64_ANNOTATION = list[
    tuple[
        Annotated[str, 'Image name'],
        Annotated[str, 'Image MIME'],
        Annotated[str, 'Base64 string encoded by UTF-8 of the image']
    ]
]

_ReasoningEffort = Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh']


def _get_attr_or_key(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_output_text_from_response_payload(response: object) -> str:
    chunks: list[str] = []
    output = _get_attr_or_key(response, 'output', [])

    if not isinstance(output, list):
        return ''

    for item in output:
        content = _get_attr_or_key(item, 'content', [])
        if not isinstance(content, list):
            continue

        for content_item in content:
            content_type = _get_attr_or_key(content_item, 'type')
            if content_type not in ('output_text', 'text'):
                continue

            text = _get_attr_or_key(content_item, 'text', '')
            if isinstance(text, str):
                chunks.append(text)

    return ''.join(chunks)

class FileSizeExceededError(Exception):
    MAX_SIZE: Literal[52428800,] = 50 * 1024 * 1024
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class LLMResponseTypeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def _extract_text_from_raw_sse(raw: str) -> str:
    chunks: list[str] = []

    for block in raw.split("\n\n"):
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            continue

        data = "\n".join(data_lines)
        if data == "[DONE]":
            continue

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    chunks.append(delta["content"])
                elif isinstance(message, dict) and isinstance(message.get("content"), str):
                    chunks.append(message["content"])
            continue

        typ = payload.get("type")

        if typ == "response.output_text.delta":
            delta = payload.get("delta", "")
            if isinstance(delta, str):
                chunks.append(delta)

        elif typ == "response.output_text.done":
            # 有些实现会在 done 里带完整 text
            text = payload.get("text")
            if isinstance(text, str) and text and not chunks:
                chunks.append(text)

        elif typ in ("response.completed", "response.done") and not chunks:
            # 兜底：从最终 response.output 里抽文本
            chunks.append(_extract_output_text_from_response_payload(payload.get("response") or {}))

    return "".join(chunks).strip()

class OnlineLLM:
    DEFAULT_MIME = 'application/octet-stream'
    DEFAULT_TIMEOUT = httpx.Timeout(
        connect = 60.0,
        read = 1200.0,
        write = 600.0,
        pool = 60.0
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        client: AsyncOpenAI | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        self.base_url: str | None = base_url or os.getenv("BASEURL") or os.getenv("OPENAI_BASE_URL")
        self.api_key: str | None = api_key or os.getenv("APIKEY") or os.getenv("OPENAI_API_KEY")
        self.model_id: str | None = model_id or os.getenv("MODEL") or os.getenv("OPENAI_MODEL")

        missing = []
        if client is None:
            if not self.base_url:
                missing.append("BASEURL")
            if not self.api_key:
                missing.append("APIKEY")
        if not self.model_id:
            missing.append("MODEL")

        if missing:
            names = ", ".join(missing)
            errmsg = simpidlog.error(
                _ERROR_PREFIX + 'OnlineLLM.__init__(): ' +
                f'Lacking LLM API config: {names}. Please config at `.env`, for instance,\n'
                '    BASEURL=\"https://api.openai.com/v1\"\n'
                '    APIKEY=\"sk-...\"\n'
                '    MODEL=\"Your model ID\"\n'
                'Then try again.'
            )
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)

        self.client: AsyncOpenAI = \
            client if client is not None else \
            AsyncOpenAI(
                api_key = self.api_key,
                base_url = self.base_url,
                timeout = self.DEFAULT_TIMEOUT if timeout is None else timeout
            )

    async def close(self) -> None:
        close = getattr(self.client, 'close', None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def __aenter__(self) -> typing.Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def _require_model_id(self) -> str:
        if self.model_id:
            return self.model_id

        errmsg = _ERROR_PREFIX + 'OnlineLLM._require_model_id(): MODEL is not configured.'
        simpidlog.error(errmsg)
        raise RuntimeError(errmsg)

    def _raise_invalid_argument(self, message: str) -> typing.NoReturn:
        errmsg = _ERROR_PREFIX + message
        simpidlog.error(errmsg)
        raise TypeError(errmsg)

    def _normalize_file_paths(
        self,
        file_paths: _FILE_PATHS_ANNOTATION
    ) -> list[Path]:
        if file_paths is None:
            return []

        if isinstance(file_paths, (str, bytes)) or not isinstance(file_paths, (list, tuple)):
            self._raise_invalid_argument(
                'OnlineLLM._normalize_file_paths(): file_paths must be a list/tuple of str or Path, or None.'
            )

        paths: list[Path] = []
        for index, path in enumerate(file_paths):
            if not isinstance(path, (str, os.PathLike)):
                self._raise_invalid_argument(
                    'OnlineLLM._normalize_file_paths(): ' +
                    f'file_paths[{index}] must be str or Path, got {type(path).__name__}.'
                )
            paths.append(Path(path))

        return paths

    def _validate_file_bin(self, name: str, file_bin: object) -> bytes:
        if not isinstance(file_bin, bytes):
            self._raise_invalid_argument(
                f'OnlineLLM._validate_file_bin(): {name} must be bytes, got {type(file_bin).__name__}.'
            )
        return file_bin

    def _has_uploads(
        self,
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> bool:
        if self._normalize_file_paths(file_paths):
            return True

        if file_bins is None:
            return False

        if isinstance(file_bins, dict):
            for path, file_bin in file_bins.items():
                if not isinstance(path, (str, os.PathLike)):
                    self._raise_invalid_argument(
                        'OnlineLLM._has_uploads(): file_bins keys must be str or Path.'
                    )
                self._validate_file_bin(f'file_bins[{path!r}]', file_bin)
            return bool(file_bins)

        if isinstance(file_bins, list):
            for index, file_bin in enumerate(file_bins):
                self._validate_file_bin(f'file_bins[{index}]', file_bin)
            return bool(file_bins)

        self._raise_invalid_argument(
            'OnlineLLM._has_uploads(): file_bins must be dict[str | Path, bytes], list[bytes], or None.'
        )

    def _raise_if_file_size_exceeded(self, total_size: int) -> None:
        if total_size > FileSizeExceededError.MAX_SIZE:
            errmsg = _ERROR_PREFIX + 'OnlineLLM._raise_if_file_size_exceeded(): ' + \
                     f'Files total size: {(total_size / 1024) / 1024} MB, ' + \
                     f'exceeded {(FileSizeExceededError.MAX_SIZE / 1024) / 1024} MB.'
            simpidlog.error(errmsg)
            raise FileSizeExceededError(errmsg)

    def _mime_from_bin(self, file_bin: bytes) -> str:
        try:
            mime = analyse_mime_from_bin(file_bin, mime = True)
        except Exception:
            return self.DEFAULT_MIME
        return mime if isinstance(mime, str) and mime else self.DEFAULT_MIME

    def _mime_from_path(self, path: Path, file_bin: bytes) -> str:
        bin_mime = self._mime_from_bin(file_bin)
        if bin_mime[:6] == 'image/':
            return bin_mime

        path_mime, _ = mimetypes.guess_type(path)
        if bin_mime == self.DEFAULT_MIME and path_mime and path_mime[:6] == 'image/':
            return path_mime

        return bin_mime

    def _raise_if_not_image_mime(self, name: str, mime: str) -> None:
        if mime[:6] == 'image/':
            return

        errmsg = _ERROR_PREFIX + 'OnlineLLM._raise_if_not_image_mime(): ' + \
                 f'Unsupported upload type for {name}: {mime}. Only image/* uploads are supported.'
        simpidlog.error(errmsg)
        raise ValueError(errmsg)

    def _build_name_mime_b64(
        self,
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> tuple[_NAME_MIME_B64_ANNOTATION, Annotated[int, 'Total size of the files']]:
        name_mime_b64: _NAME_MIME_B64_ANNOTATION = []
        total_size: int = 0

        for path in self._normalize_file_paths(file_paths):
            name = path.name
            file_size = path.stat().st_size
            total_size += file_size
            self._raise_if_file_size_exceeded(total_size)

            file_bin = path.read_bytes()
            mime = self._mime_from_path(path, file_bin)
            self._raise_if_not_image_mime(name, mime)
            b64: str = base64.b64encode(file_bin).decode('utf-8')
            name_mime_b64.append((name, mime, b64))

        if file_bins is None:
            return name_mime_b64, total_size

        if isinstance(file_bins, dict):
            for raw_path, raw_file_bin in file_bins.items():
                if not isinstance(raw_path, (str, os.PathLike)):
                    self._raise_invalid_argument(
                        'OnlineLLM._build_name_mime_b64(): file_bins keys must be str or Path.'
                    )

                path = Path(raw_path)
                file_bin = self._validate_file_bin(f'file_bins[{raw_path!r}]', raw_file_bin)
                name = path.name
                total_size += len(file_bin)
                self._raise_if_file_size_exceeded(total_size)

                mime = self._mime_from_path(path, file_bin)
                self._raise_if_not_image_mime(name, mime)
                b64: str = base64.b64encode(file_bin).decode('utf-8')

                name_mime_b64.append((name, mime, b64))

        elif isinstance(file_bins, list):
            for index, raw_file_bin in enumerate(file_bins, start = 1):
                file_bin = self._validate_file_bin(f'file_bins[{index - 1}]', raw_file_bin)
                total_size += len(file_bin)
                self._raise_if_file_size_exceeded(total_size)

                mime = self._mime_from_bin(file_bin)
                extension = mimetypes.guess_extension(mime) or '.bin'
                name = f'{index}{extension}'
                self._raise_if_not_image_mime(name, mime)
                b64: str = base64.b64encode(file_bin).decode('utf-8')

                name_mime_b64.append((name, mime, b64))
        else:
            self._raise_invalid_argument(
                'OnlineLLM._build_name_mime_b64(): file_bins must be dict[str | Path, bytes], list[bytes], or None.'
            )

        return name_mime_b64, total_size

    def _user_message_indices(self, messages: list[dict]) -> list[int]:
        indices: list[int] = []

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                self._raise_invalid_argument(
                    f'OnlineLLM._user_message_indices(): messages[{index}] must be dict.'
                )
            if message.get('role') == 'user':
                indices.append(index)

        return indices

    def _compatible_content_list(self, messages: list[dict], user_i: int) -> list:
        content = messages[user_i].get('content')
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            content_list = [{ 'type': 'text', 'text': content }]
            messages[user_i]['content'] = content_list
            return content_list

        self._raise_invalid_argument(
            'OnlineLLM._compatible_content_list(): user message content must be str or list.'
        )

    def _responses_content_list(self, messages: list[dict], user_i: int) -> list:
        content = messages[user_i].get('content')
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            content_list = [{ 'type': 'input_text', 'text': content }]
            messages[user_i]['content'] = content_list
            return content_list

        self._raise_invalid_argument(
            'OnlineLLM._responses_content_list(): user message content must be str or list.'
        )

    def _insert_bins_to_messages_compatible(
        self,
        messages: list[dict],
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> None:
        if not self._has_uploads(file_paths, file_bins):
            return

        name_mime_b64, total_size = self._build_name_mime_b64(
            file_paths = file_paths,
            file_bins = file_bins
        )

        if total_size > FileSizeExceededError.MAX_SIZE:
            errmsg = _ERROR_PREFIX + 'OnlineLLM._insert_bins_to_messages_compatible(): ' + \
                     f'Files total size: {(total_size / 1024) / 1024} MB, ' + \
                     f'exceeded {(FileSizeExceededError.MAX_SIZE / 1024) / 1024} MB.'
            simpidlog.error(errmsg)
            raise FileSizeExceededError(errmsg)

        user_indices = self._user_message_indices(messages)
        if not user_indices:
            self._raise_invalid_argument(
                'OnlineLLM._insert_bins_to_messages_compatible(): messages must contain at least one user message.'
            )

        if len(user_indices) == 1:
            content = self._compatible_content_list(messages, user_indices[0])
            for (name, mime, b64) in name_mime_b64:
                self._raise_if_not_image_mime(name, mime)
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime};base64,{b64}'
                    }
                })
        else:
            # messages 中有多个 user
            for (name, mime, _) in name_mime_b64:
                self._raise_if_not_image_mime(name, mime)

            messages.append({
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime};base64,{b64}'
                        }
                    }
                    for (name, mime, b64) in name_mime_b64
                ]
            })


    def _build_msgs_compatible(
        self,
        system_prompt: str | None,
        user_prompt: str,
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins:  _FILE_BINS_ANNOTATION
    ) -> list:
        messages: list[dict] = []
        if system_prompt is not None:
            messages = [{'role': 'system', 'content': system_prompt}]

        if self._has_uploads(file_paths, file_bins):
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_prompt},
                ],
            })
        else:
            messages.append({'role': 'user', 'content': user_prompt})

        self._insert_bins_to_messages_compatible(
            messages = messages,
            file_paths = file_paths,
            file_bins = file_bins
        )

        return messages
 
    def _stream_text_delta(self, event: object) -> str:
        event_type = _get_attr_or_key(event, 'type')
        if event_type == 'response.output_text.delta':
            delta = _get_attr_or_key(event, 'delta', '')
            return delta if isinstance(delta, str) else ''

        choices = _get_attr_or_key(event, 'choices', [])
        if not isinstance(choices, list):
            return ''

        chunks: list[str] = []
        for choice in choices:
            delta = _get_attr_or_key(choice, 'delta')
            content = _get_attr_or_key(delta, 'content', '')
            if isinstance(content, str):
                chunks.append(content)

        return ''.join(chunks)

    async def _parse_stream_response(self, stream: typing.AsyncIterable[object], request_id: str) -> str:
        text: str = ''
        index: int = 0
        async for event in stream:
            delta = self._stream_text_delta(event)
            if delta:
                text += delta
                index += 1
            elif not text and _get_attr_or_key(event, 'type') == 'response.output_text.done':
                done_text = _get_attr_or_key(event, 'text', '')
                if isinstance(done_text, str):
                    text = done_text
            if index % 1000 == 0 and index != 0:
                simpidlog.debug(_ERROR_PREFIX + f'OnlineLLM._parse_stream_response(): {request_id} has got {index} deltas...')
        simpidlog.debug(_ERROR_PREFIX + f'OnlineLLM._parse_stream_response(): ' + \
                        f'{request_id} done with {index} deltas, {len(text)} bytes.')
        return text

    def _content_from_compatible_response(self, response: object) -> str:
        if isinstance(response, str):
            return _extract_text_from_raw_sse(response) or response

        choices = _get_attr_or_key(response, 'choices', [])
        if not isinstance(choices, list) or not choices:
            return ''

        message = _get_attr_or_key(choices[0], 'message')
        content = _get_attr_or_key(message, 'content', '')
        return content if isinstance(content, str) else ''

    def _content_from_responses_response(self, response: object) -> str:
        if isinstance(response, str):
            return _extract_text_from_raw_sse(response) or response

        output_text = _get_attr_or_key(response, 'output_text', '')
        if isinstance(output_text, str) and output_text:
            return output_text

        return _extract_output_text_from_response_payload(response)

    async def call_compatible(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float,
        effort: _ReasoningEffort = 'medium',
        file_paths: _FILE_PATHS_ANNOTATION = None,
        file_bins: _FILE_BINS_ANNOTATION = None,
        stream: bool = False
     ) -> str:
        model_id = self._require_model_id()
        messages = self._build_msgs_compatible(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            file_paths = file_paths,
            file_bins = file_bins
        )

        try:
            request_id: str = str(uuid.uuid4())
            response = await self.client.chat.completions.create(
                model = model_id,
                temperature = temperature,
                messages = messages,
                stream = stream,
                reasoning_effort = effort
            )
        except OpenAIError as exc:
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_compatible(): Failed to call LLM API.\n' + str(exc)
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg) from exc

        if stream:
            content = await self._parse_stream_response(stream = response, request_id = request_id)
        else:
            content = self._content_from_compatible_response(response)


        content = content.strip()
        if not content:
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_compatible(): LLM API calling returned empty content.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)
        return content

    def _insert_bins_to_messages_responses(
        self,
        messages: list[dict],
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> None:
        if not self._has_uploads(file_paths, file_bins):
            return

        name_mime_b64, total_size = self._build_name_mime_b64(
            file_paths = file_paths,
            file_bins = file_bins
        )

        if total_size > FileSizeExceededError.MAX_SIZE:
            errmsg = _ERROR_PREFIX + 'OnlineLLM._insert_bins_to_messages_responses(): ' + \
                     f'Files total size: {(total_size / 1024) / 1024} MB, ' + \
                     f'exceeded {(FileSizeExceededError.MAX_SIZE / 1024) / 1024} MB.'
            simpidlog.error(errmsg)
            raise FileSizeExceededError(errmsg)

        user_indices = self._user_message_indices(messages)
        if not user_indices:
            self._raise_invalid_argument(
                'OnlineLLM._insert_bins_to_messages_responses(): messages must contain at least one user message.'
            )
        content = self._responses_content_list(messages, user_indices[0])

        for (name, mime, b64) in name_mime_b64:
            self._raise_if_not_image_mime(name, mime)
            content.append({
                'type': 'input_image',
                'image_url': f'data:{mime};base64,{b64}'
            })

    def _build_msgs_responses(
        self,
        system_prompt: str | None,
        user_prompt: str,
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins:  _FILE_BINS_ANNOTATION
    ) -> list:
        messages: list[dict] = []
        if system_prompt is not None:
            messages = [{'role': 'system', 'content': system_prompt}]

        messages.append({
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': user_prompt},
            ],
        })

        self._insert_bins_to_messages_responses(
            messages = messages,
            file_paths = file_paths,
            file_bins = file_bins
        )

        return messages

    async def call_responses(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float,
        effort: _ReasoningEffort = 'medium',
        web_search: Literal['none', 'low', 'medium', 'high'] = 'none',
        tools_required: bool = False,
        file_paths: _FILE_PATHS_ANNOTATION = None,
        file_bins: _FILE_BINS_ANNOTATION = None,
        stream: bool = False
     ) -> str:
        model_id = self._require_model_id()
        messages = self._build_msgs_responses(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            file_paths = file_paths,
            file_bins = file_bins
        )

        try:
            request_id: str = str(uuid.uuid4())
            if web_search == 'none':
                response = await self.client.responses.create(
                    model = model_id,
                    temperature = temperature,
                    input = messages,
                    stream = stream,
                    reasoning = { 'effort': effort },
                )
            else:
                response = await self.client.responses.create(
                    model = model_id,
                    temperature = temperature,
                    input = messages,
                    stream = stream,
                    reasoning = { 'effort': effort },
                    tools = [{
                        'type': 'web_search',
                        'search_context_size': web_search
                    }],
                    tool_choice = 'required' if tools_required else 'auto'
                )


        except OpenAIError as exc:
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_responses(): Failed to call LLM API.\n' + str(exc)
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg) from exc

        if stream:
            content = await self._parse_stream_response(stream = response, request_id = request_id)
        else:
            content = self._content_from_responses_response(response)

        content = content.strip()
        if not content:
            errmsg = _ERROR_PREFIX + 'LLM API calling returned empty content.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)
        return content

    def parse_json(
        self,
        text: str
    ) -> dict:
        """从 LLM 输出中解析 JSON 对象。

        很多模型会把 JSON 包在 ```json 代码块中。本函数只做格式清理和解析，
        不会用本地规则替代 LLM 的结果；解析失败说明 prompt 或模型输出需要调整。
        """

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 如果模型在 JSON 前后添加解释文字，尝试截取第一个 {...} 区间。
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            errmsg = _ERROR_PREFIX + f'Invalid JSON response. Original length: {len(text)} chars.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg) from exc

        if not isinstance(value, dict):
            errmsg = _ERROR_PREFIX + f'Invalid JSON response: expected object, got {type(value).__name__}.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)

        return value

    def parse_codeblock(
        self,
        text: str,
        language: str
    ) -> list[str]:
        '''从 LLM 输出中解析 <language> 代码围栏围起来的 <language> 代码。'''
        soup = BeautifulSoup(markdown(text, extensions = ['fenced_code', 'tables']), 'html.parser')
        codes = soup.find_all('code', attrs = { 'class': f'language-{language}' })
        results: list[str] = []
        for code in codes:
            results.append(code.text)
        return results
