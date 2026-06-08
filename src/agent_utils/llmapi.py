import os
import re
import json
import base64
import typing
import uuid
from magic import from_buffer as analyse_mime_from_bin
import mimetypes
import httpx

import traceback

from typing import Annotated, Literal
from markdown import markdown
from bs4 import BeautifulSoup

from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import OpenAIError
from openai import AsyncStream

import simpidlog

load_dotenv(Path(__file__).with_name(".env"))

_ERROR_PREFIX = '@Simpidbit/agent_utils/llmapi.py\n'

_FILE_PATHS_ANNOTATION = Annotated[
    list[str | Path] | None,
    'File\'s full path'
]

_FILE_BINS_ANNOTATION = Annotated[
    dict[str | Path, bytes] | list[bytes] | None,
    'dict[str | Path, bytes]: key is the full file path，value is the binary bytes data of the file.\n'
    'list[bytes]: List of bytes, while filename from python-magic guessing.'
]

_NAME_MIME_B64_ANNOTATION = list[
    tuple[
        Annotated[str, 'File name'],
        Annotated[str, 'File MIME'],
        Annotated[str, 'Base64 string encoded by UTF-8 of the file']
    ]
]

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

        typ = payload.get("type")

        if typ == "response.output_text.delta":
            chunks.append(payload.get("delta", ""))

        elif typ == "response.output_text.done":
            # 有些实现会在 done 里带完整 text
            text = payload.get("text")
            if text and not chunks:
                chunks.append(text)

        elif typ in ("response.completed", "response.done"):
            # 兜底：从最终 response.output 里抽文本
            resp = payload.get("response") or {}
            for item in resp.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        chunks.append(c.get("text", ""))

    return "".join(chunks).strip()

class OnlineLLM:
    DEFAULT_MIME = 'application/octet-stream'

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        client: AsyncOpenAI | None = None,
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
            AsyncOpenAI(api_key = self.api_key, base_url = self.base_url, timeout = httpx.Timeout(
                connect = 60.0,
                read = 1200.0,
                write = 600.0,
                pool = 60.0
            ))

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
        mime, _ = mimetypes.guess_type(path)
        return mime if mime else self._mime_from_bin(file_bin)

    def _build_name_mime_b64(
        self,
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> tuple[_NAME_MIME_B64_ANNOTATION, Annotated[int, 'Total size of the files']]:
        name_mime_b64: _NAME_MIME_B64_ANNOTATION = []
        total_size: int = 0

        if isinstance(file_paths, list):
            for path in file_paths:
                if isinstance(path, str):
                    path = Path(path)
                name = path.name
                file_size = path.stat().st_size
                total_size += file_size
                self._raise_if_file_size_exceeded(total_size)

                file_bin = path.read_bytes()
                mime = self._mime_from_path(path, file_bin)
                b64: str = base64.b64encode(file_bin).decode('utf-8')
                name_mime_b64.append((name, mime, b64))

        if isinstance(file_bins, dict):
            for path in file_bins.keys():
                file_bin = file_bins[path]
                if isinstance(path, str):
                    path = Path(path)
                name = path.name
                total_size += len(file_bin)
                self._raise_if_file_size_exceeded(total_size)

                mime = self._mime_from_bin(file_bin)
                b64: str = base64.b64encode(file_bin).decode('utf-8')

                name_mime_b64.append((name, mime, b64))

        elif isinstance(file_bins, list):
            index = 1
            for file_bin in file_bins:
                total_size += len(file_bin)
                self._raise_if_file_size_exceeded(total_size)

                mime = self._mime_from_bin(file_bin)
                extension = mimetypes.guess_extension(mime) or '.bin'
                name = f'{index}{extension}'
                b64: str = base64.b64encode(file_bin).decode('utf-8')

                name_mime_b64.append((name, mime, b64))
                index += 1

        return name_mime_b64, total_size

    def _insert_bins_to_messages_compatible(
        self,
        messages: list[dict],
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> None:
        if not (isinstance(file_paths, list) and file_paths) and file_bins is None:
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

        # 判断原 messages 结构中是否存在多个 user
        user_i: int | None = None
        insert_mode: Literal['single_user', 'multi_users'] = 'single_user'
        for i in range(len(messages)):
            if messages[i]['role'] == 'user':
                if user_i is None:
                    user_i = i
                    user_i = typing.cast(int, user_i)
                else:
                    insert_mode = 'multi_users'
                    user_i = None
                    break

        if insert_mode == 'single_user':
            assert isinstance(user_i, int)
            for (name, mime, b64) in name_mime_b64:
                if mime[:6] == 'image/':
                    typing.cast(list, messages[user_i]['content']).append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime};base64,{b64}'
                        }
                    })
                else:
                    typing.cast(list, messages[user_i]['content']).append({
                        'type': 'file',
                        'file': {
                            'filename': name,
                            'file_data': f'data:{mime};base64,{b64}'
                        }
                    })
        else:
            # messages 中有多个 user
            assert user_i is None

            messages.append({
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime};base64,{b64}'
                        }
                    } if mime[:6] == 'image/' else
                    {
                        'type': 'file',
                        'file': {
                            'filename': name,
                            'file_data': f'data:{mime};base64,{b64}'
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

        if file_paths or file_bins:
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
 
    async def _parse_stream_response(self, stream: AsyncStream, request_id: str) -> str:
        text: str = ''
        index: int = 0
        async for event in stream:
            match event.type:
                case 'response.output_text.delta':
                    text += event.delta
                    index += 1
                    simpidlog.debug(f'[{event.delta}]')
            if index % 1000 == 0 and index != 0:
                simpidlog.debug(_ERROR_PREFIX + f'OnlineLLM._parse_stream_response(): {request_id} has got {index} deltas...')
        simpidlog.debug(_ERROR_PREFIX + f'OnlineLLM._parse_stream_response(): ' + \
                        f'{request_id} done with {index} deltas, {len(text)} bytes.\n' + \
                        f'{text[:100]}...')
        return text

    async def call_compatible(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float,
        effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        file_paths: _FILE_PATHS_ANNOTATION = None,
        file_bins: _FILE_BINS_ANNOTATION = None,
        stream: bool = False
     ) -> str:
        assert self.model_id is not None
        messages = self._build_msgs_compatible(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            file_paths = file_paths,
            file_bins = file_bins
        )

        try:
            request_id: str = str(uuid.uuid4())
            response = await self.client.chat.completions.create(
                model = self.model_id,
                temperature = temperature,
                messages = messages,
                stream = stream,
                reasoning_effort = effort
            )
        except OpenAIError as exc:
            traceback.print_exc()
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_compatible(): Failed to call LLM API.\n' + str(exc)
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            print(str(exc))
            raise RuntimeError(errmsg) from exc

        if stream:
            assert(isinstance(response, AsyncStream))
            content = await self._parse_stream_response(stream = response, request_id = request_id)
        else:
            if isinstance(response, str):
                content = response
            else:
                content = response.choices[0].message.content


        if not content:
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_compatible(): LLM API calling returned empty content.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)
        return content.strip()

    def _insert_bins_to_messages_responses(
        self,
        messages: list[dict],
        file_paths: _FILE_PATHS_ANNOTATION,
        file_bins: _FILE_BINS_ANNOTATION
    ) -> None:
        if not (isinstance(file_paths, list) and file_paths) and file_bins is None:
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

        user_i: int = 0
        for i in range(len(messages)):
            if messages[i]['role'] == 'user':
                user_i = i
                break

        for (name, mime, b64) in name_mime_b64:
            content = typing.cast(list, messages[user_i]['content'])
            if mime[:6] == 'image/':
                content.append({
                    'type': 'input_image',
                    'image_url': f'data:{mime};base64,{b64}'
                })
            else:
                content.append({
                    'type': 'input_file',
                    'filename': name,
                    'file_data': f'data:{mime};base64,{b64}'
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
        effort: Literal['none', 'low', 'medium', 'high', 'xhigh'] = 'medium',
        web_search: Literal['none', 'low', 'medium', 'high'] = 'none',
        tools_required: bool = False,
        file_paths: _FILE_PATHS_ANNOTATION = None,
        file_bins: _FILE_BINS_ANNOTATION = None,
        stream: bool = False
     ) -> str:
        assert self.model_id is not None
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
                    model = self.model_id,
                    temperature = temperature,
                    input = messages,
                    stream = stream,
                    reasoning = { 'effort': effort },
                )
            else:
                response = await self.client.responses.create(
                    model = self.model_id,
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
            traceback.print_exc()
            errmsg = _ERROR_PREFIX + 'OnlineLLM.call_responses(): Failed to call LLM API.\n' + str(exc)
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg) from exc

        if stream:
            assert(isinstance(response, AsyncStream))
            content = await self._parse_stream_response(stream = response, request_id = request_id)
        else:
            if isinstance(response, str):
                '''
                errmsg = _ERROR_PREFIX + f'OnlineLLM.call_responses(): The type of response is str.'
                simpidlog.error(errmsg)
                simpidlog.error(_ERROR_PREFIX + response)
                raise LLMResponseTypeError(errmsg)
                '''
                content = _extract_text_from_raw_sse(response)
            else:
                content = response.output_text

        if not content:
            errmsg = _ERROR_PREFIX + 'LLM API calling returned empty content.'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg)
        return content.strip()

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
            errmsg = _ERROR_PREFIX + f'Invalid JSON: {text}'
            simpidlog.error(errmsg)
            simpidlog.wait_for_log_io()
            raise RuntimeError(errmsg) from exc

        if not isinstance(value, dict):
            errmsg = _ERROR_PREFIX + f'Invalid JSON: {text}'
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


if __name__ == '__main__':
    llm = OnlineLLM()

    import asyncio
    res = asyncio.run(llm.call_responses(
        system_prompt = None,
        user_prompt = 'How are you?',
        temperature = 0.0
    ))
    print(res)
