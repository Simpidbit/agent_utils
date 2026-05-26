import base64
import os
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


THIS_DIR = Path(__file__).resolve().parent
LLMAPI_DIR = THIS_DIR.parent
if str(LLMAPI_DIR) not in sys.path:
    sys.path.insert(0, str(LLMAPI_DIR))

import llmapi  # noqa: E402


class AsyncCreateRecorder:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class DummyClient:
    def __init__(
        self,
        chat_response=None,
        responses_response=None,
        chat_exc=None,
        responses_exc=None,
    ):
        self.chat_create = AsyncCreateRecorder(chat_response, chat_exc)
        self.responses_create = AsyncCreateRecorder(responses_response, responses_exc)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.chat_create.create)
        )
        self.responses = SimpleNamespace(create=self.responses_create.create)


def make_llm(client=None, model_id="test-model"):
    llm = llmapi.OnlineLLM.__new__(llmapi.OnlineLLM)
    llm.base_url = "https://example.test/v1"
    llm.api_key = "test-key"
    llm.model_id = model_id
    llm.client = client if client is not None else DummyClient()
    return llm


def chat_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def responses_response(content):
    return SimpleNamespace(output_text=content)


class OnlineLLMInitTests(unittest.TestCase):
    def test_init_reads_primary_environment_names_and_builds_client(self):
        fake_client = object()
        env = {
            "BASEURL": "https://primary.example/v1",
            "APIKEY": "primary-key",
            "MODEL": "primary-model",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            llmapi, "AsyncOpenAI", autospec=True, return_value=fake_client
        ) as async_openai:
            llm = llmapi.OnlineLLM()

        self.assertEqual(llm.base_url, "https://primary.example/v1")
        self.assertEqual(llm.api_key, "primary-key")
        self.assertEqual(llm.model_id, "primary-model")
        self.assertIs(llm.client, fake_client)
        async_openai.assert_called_once_with(
            api_key="primary-key", base_url="https://primary.example/v1"
        )

    def test_init_reads_openai_environment_fallback_names(self):
        fake_client = object()
        env = {
            "OPENAI_BASE_URL": "https://fallback.example/v1",
            "OPENAI_API_KEY": "fallback-key",
            "OPENAI_MODEL": "fallback-model",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            llmapi, "AsyncOpenAI", autospec=True, return_value=fake_client
        ) as async_openai:
            llm = llmapi.OnlineLLM()

        self.assertEqual(llm.base_url, "https://fallback.example/v1")
        self.assertEqual(llm.api_key, "fallback-key")
        self.assertEqual(llm.model_id, "fallback-model")
        self.assertIs(llm.client, fake_client)
        async_openai.assert_called_once_with(
            api_key="fallback-key", base_url="https://fallback.example/v1"
        )

    def test_init_prefers_primary_environment_names_over_fallback_names(self):
        fake_client = object()
        env = {
            "BASEURL": "https://primary.example/v1",
            "APIKEY": "primary-key",
            "MODEL": "primary-model",
            "OPENAI_BASE_URL": "https://fallback.example/v1",
            "OPENAI_API_KEY": "fallback-key",
            "OPENAI_MODEL": "fallback-model",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            llmapi, "AsyncOpenAI", autospec=True, return_value=fake_client
        ):
            llm = llmapi.OnlineLLM()

        self.assertEqual(llm.base_url, "https://primary.example/v1")
        self.assertEqual(llm.api_key, "primary-key")
        self.assertEqual(llm.model_id, "primary-model")

    def test_init_reports_all_missing_required_configuration_names(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            llmapi.simpidlog, "error", autospec=True, side_effect=lambda msg: msg
        ) as log_error, patch.object(
            llmapi.simpidlog, "wait_for_log_io", autospec=True
        ) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                llmapi.OnlineLLM()

        message = str(ctx.exception)
        self.assertIn("Lacking LLM API config", message)
        self.assertIn("BASEURL", message)
        self.assertIn("APIKEY", message)
        self.assertIn("MODEL", message)
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()

    def test_init_reports_only_missing_configuration_names(self):
        env = {
            "BASEURL": "https://example.test/v1",
            "MODEL": "model-present",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            llmapi.simpidlog, "error", autospec=True, side_effect=lambda msg: msg
        ):
            with self.assertRaises(RuntimeError) as ctx:
                llmapi.OnlineLLM()

        message = str(ctx.exception)
        self.assertIn("APIKEY", message)
        self.assertNotIn("BASEURL, APIKEY", message)

    def test_init_accepts_explicit_configuration_and_builds_client(self):
        fake_client = object()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            llmapi, "AsyncOpenAI", autospec=True, return_value=fake_client
        ) as async_openai:
            llm = llmapi.OnlineLLM(
                base_url="https://explicit.example/v1",
                api_key="explicit-key",
                model_id="explicit-model",
            )

        self.assertEqual(llm.base_url, "https://explicit.example/v1")
        self.assertEqual(llm.api_key, "explicit-key")
        self.assertEqual(llm.model_id, "explicit-model")
        self.assertIs(llm.client, fake_client)
        async_openai.assert_called_once_with(
            api_key="explicit-key", base_url="https://explicit.example/v1"
        )

    def test_init_accepts_injected_client_without_api_configuration(self):
        fake_client = object()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            llmapi, "AsyncOpenAI", autospec=True
        ) as async_openai:
            llm = llmapi.OnlineLLM(model_id="injected-model", client=fake_client)

        self.assertIsNone(llm.base_url)
        self.assertIsNone(llm.api_key)
        self.assertEqual(llm.model_id, "injected-model")
        self.assertIs(llm.client, fake_client)
        async_openai.assert_not_called()


class FileEncodingTests(unittest.TestCase):
    def test_file_size_exceeded_error_exposes_50mb_limit(self):
        self.assertEqual(llmapi.FileSizeExceededError.MAX_SIZE, 50 * 1024 * 1024)
        self.assertEqual(str(llmapi.FileSizeExceededError("too large")), "too large")

    def test_build_name_mime_b64_encodes_file_paths_and_dict_file_bins(self):
        llm = make_llm()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            text_path = tmp / "note.txt"
            image_path = tmp / "photo.png"
            binary_path = tmp / "manual.bin"
            text_path.write_bytes(b"hello")
            image_path.write_bytes(b"PNGDATA")

            with patch.object(
                llmapi, "analyse_mime_from_bin", autospec=True, return_value="application/pdf"
            ) as analyse_mime:
                name_mime_b64, total_size = llm._build_name_mime_b64(
                    file_paths=[str(text_path), image_path],
                    file_bins={str(binary_path): b"%PDF-1.4"},
                )

        self.assertEqual(total_size, len(b"hello") + len(b"PNGDATA") + len(b"%PDF-1.4"))
        self.assertEqual(
            name_mime_b64,
            [
                ("note.txt", "text/plain", base64.b64encode(b"hello").decode("utf-8")),
                ("photo.png", "image/png", base64.b64encode(b"PNGDATA").decode("utf-8")),
                (
                    "manual.bin",
                    "application/pdf",
                    base64.b64encode(b"%PDF-1.4").decode("utf-8"),
                ),
            ],
        )
        analyse_mime.assert_called_once_with(b"%PDF-1.4", mime=True)

    def test_build_name_mime_b64_encodes_list_file_bins_with_guessed_names(self):
        llm = make_llm()

        with patch.object(
            llmapi,
            "analyse_mime_from_bin",
            autospec=True,
            side_effect=["text/plain", "image/png"],
        ) as analyse_mime:
            name_mime_b64, total_size = llm._build_name_mime_b64(
                file_paths=None,
                file_bins=[b"plain text", b"png bytes"],
            )

        self.assertEqual(total_size, len(b"plain text") + len(b"png bytes"))
        self.assertEqual(
            name_mime_b64,
            [
                ("1.txt", "text/plain", base64.b64encode(b"plain text").decode("utf-8")),
                ("2.png", "image/png", base64.b64encode(b"png bytes").decode("utf-8")),
            ],
        )
        self.assertEqual(analyse_mime.call_count, 2)

    def test_build_name_mime_b64_ignores_non_list_file_paths_and_none_bins(self):
        llm = make_llm()

        name_mime_b64, total_size = llm._build_name_mime_b64(
            file_paths="not-a-list",
            file_bins=None,
        )

        self.assertEqual(name_mime_b64, [])
        self.assertEqual(total_size, 0)

    def test_build_name_mime_b64_uses_magic_when_path_mime_is_unknown(self):
        llm = make_llm()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "payload"
            path.write_bytes(b"binary payload")

            with patch.object(
                llmapi.mimetypes, "guess_type", autospec=True, return_value=(None, None)
            ) as guess_type, patch.object(
                llmapi, "analyse_mime_from_bin", autospec=True, return_value="application/pdf"
            ) as analyse_mime:
                name_mime_b64, total_size = llm._build_name_mime_b64(
                    file_paths=[path],
                    file_bins=None,
                )

        self.assertEqual(total_size, len(b"binary payload"))
        self.assertEqual(
            name_mime_b64,
            [("payload", "application/pdf", base64.b64encode(b"binary payload").decode("utf-8"))],
        )
        guess_type.assert_called_once_with(path)
        analyse_mime.assert_called_once_with(b"binary payload", mime=True)

    def test_build_name_mime_b64_uses_default_mime_when_magic_fails(self):
        llm = make_llm()

        with patch.object(
            llmapi, "analyse_mime_from_bin", autospec=True, side_effect=RuntimeError("boom")
        ):
            name_mime_b64, total_size = llm._build_name_mime_b64(
                file_paths=None,
                file_bins={"payload.bin": b"payload"},
            )

        self.assertEqual(total_size, len(b"payload"))
        self.assertEqual(
            name_mime_b64,
            [("payload.bin", llm.DEFAULT_MIME, base64.b64encode(b"payload").decode("utf-8"))],
        )

    def test_build_name_mime_b64_uses_bin_extension_when_mime_extension_is_unknown(self):
        llm = make_llm()

        with patch.object(
            llmapi, "analyse_mime_from_bin", autospec=True, return_value="application/x-custom"
        ), patch.object(llmapi.mimetypes, "guess_extension", autospec=True, return_value=None):
            name_mime_b64, total_size = llm._build_name_mime_b64(
                file_paths=None,
                file_bins=[b"payload"],
            )

        self.assertEqual(total_size, len(b"payload"))
        self.assertEqual(
            name_mime_b64,
            [("1.bin", "application/x-custom", base64.b64encode(b"payload").decode("utf-8"))],
        )

    def test_build_name_mime_b64_checks_path_size_before_reading(self):
        llm = make_llm()

        class FakeStat:
            st_size = llmapi.FileSizeExceededError.MAX_SIZE + 1

        class FakePath:
            name = "huge.bin"

            def __init__(self):
                self.read_bytes = Mock(return_value=b"should not be read")

            def stat(self):
                return FakeStat()

        fake_path = FakePath()

        with patch.object(llmapi.simpidlog, "error", autospec=True):
            with self.assertRaises(llmapi.FileSizeExceededError):
                llm._build_name_mime_b64(file_paths=[fake_path], file_bins=None)

        fake_path.read_bytes.assert_not_called()


class CompatibleMessageTests(unittest.TestCase):
    def test_insert_bins_to_messages_compatible_appends_file_paths_without_file_bins(self):
        llm = make_llm()
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        encoded = [("note.txt", "text/plain", "AAA=")]

        with patch.object(llm, "_build_name_mime_b64", autospec=True, return_value=(encoded, 3)) as builder:
            llm._insert_bins_to_messages_compatible(
                messages=messages,
                file_paths=["note.txt"],
                file_bins=None,
            )

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "note.txt",
                                "file_data": "data:text/plain;base64,AAA=",
                            },
                        },
                    ],
                }
            ],
        )
        builder.assert_called_once_with(file_paths=["note.txt"], file_bins=None)

    def test_insert_bins_to_messages_compatible_appends_image_and_file_to_single_user(self):
        llm = make_llm()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        encoded = [
            ("photo.png", "image/png", "AAA="),
            ("doc.pdf", "application/pdf", "BBB="),
        ]

        with patch.object(llm, "_build_name_mime_b64", autospec=True, return_value=(encoded, 20)):
            llm._insert_bins_to_messages_compatible(
                messages=messages,
                file_paths=None,
                file_bins=[b"data"],
            )

        self.assertEqual(
            messages[1]["content"],
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
                {
                    "type": "file",
                    "file": {
                        "filename": "doc.pdf",
                        "file_data": "data:application/pdf;base64,BBB=",
                    },
                },
            ],
        )

    def test_insert_bins_to_messages_compatible_appends_new_user_when_multiple_users_exist(self):
        llm = make_llm()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        encoded = [
            ("photo.png", "image/png", "AAA="),
            ("doc.txt", "text/plain", "BBB="),
        ]

        with patch.object(llm, "_build_name_mime_b64", autospec=True, return_value=(encoded, 20)):
            llm._insert_bins_to_messages_compatible(
                messages=messages,
                file_paths=None,
                file_bins=[b"data"],
            )

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            messages[-1]["content"],
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
                {
                    "type": "file",
                    "file": {
                        "filename": "doc.txt",
                        "file_data": "data:text/plain;base64,BBB=",
                    },
                },
            ],
        )

    def test_insert_bins_to_messages_compatible_raises_when_total_size_is_too_large(self):
        llm = make_llm()
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        with patch.object(
            llm, "_build_name_mime_b64", autospec=True, return_value=([], llmapi.FileSizeExceededError.MAX_SIZE + 1)
        ), patch.object(llmapi.simpidlog, "error", autospec=True) as log_error:
            with self.assertRaises(llmapi.FileSizeExceededError) as ctx:
                llm._insert_bins_to_messages_compatible(
                    messages=messages,
                    file_paths=None,
                    file_bins=[b"x"],
                )

        self.assertIn("exceeded", str(ctx.exception))
        log_error.assert_called_once()

    def test_build_msgs_compatible_uses_plain_string_content_without_files(self):
        llm = make_llm()

        messages = llm._build_msgs_compatible(
            system_prompt="system",
            user_prompt="hello",
            file_paths=None,
            file_bins=None,
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_build_msgs_compatible_uses_typed_content_and_delegates_file_insertion(self):
        llm = make_llm()

        with patch.object(llm, "_insert_bins_to_messages_compatible", autospec=True) as inserter:
            messages = llm._build_msgs_compatible(
                system_prompt=None,
                user_prompt="hello",
                file_paths=["a.txt"],
                file_bins=[b"a"],
            )

        self.assertEqual(
            messages,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        inserter.assert_called_once_with(
            messages=messages,
            file_paths=["a.txt"],
            file_bins=[b"a"],
        )


class ResponsesMessageTests(unittest.TestCase):
    def test_insert_bins_to_messages_responses_appends_file_paths_without_file_bins(self):
        llm = make_llm()
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
        encoded = [("note.txt", "text/plain", "AAA=")]

        with patch.object(llm, "_build_name_mime_b64", autospec=True, return_value=(encoded, 3)) as builder:
            llm._insert_bins_to_messages_responses(
                messages=messages,
                file_paths=["note.txt"],
                file_bins=None,
            )

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {
                            "type": "input_file",
                            "filename": "note.txt",
                            "file_data": "data:text/plain;base64,AAA=",
                        },
                    ],
                }
            ],
        )
        builder.assert_called_once_with(file_paths=["note.txt"], file_bins=None)

    def test_insert_bins_to_messages_responses_appends_image_and_file_to_first_user(self):
        llm = make_llm()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
        ]
        encoded = [
            ("photo.png", "image/png", "AAA="),
            ("doc.pdf", "application/pdf", "BBB="),
        ]

        with patch.object(llm, "_build_name_mime_b64", autospec=True, return_value=(encoded, 20)):
            llm._insert_bins_to_messages_responses(
                messages=messages,
                file_paths=None,
                file_bins=[b"data"],
            )

        self.assertEqual(
            messages[1]["content"],
            [
                {"type": "input_text", "text": "first"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAA="},
                {
                    "type": "input_file",
                    "filename": "doc.pdf",
                    "file_data": "data:application/pdf;base64,BBB=",
                },
            ],
        )
        self.assertEqual(messages[2]["content"], [{"type": "input_text", "text": "second"}])

    def test_insert_bins_to_messages_responses_raises_when_total_size_is_too_large(self):
        llm = make_llm()
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]

        with patch.object(
            llm, "_build_name_mime_b64", autospec=True, return_value=([], llmapi.FileSizeExceededError.MAX_SIZE + 1)
        ), patch.object(llmapi.simpidlog, "error", autospec=True) as log_error:
            with self.assertRaises(llmapi.FileSizeExceededError) as ctx:
                llm._insert_bins_to_messages_responses(
                    messages=messages,
                    file_paths=None,
                    file_bins=[b"x"],
                )

        self.assertIn("exceeded", str(ctx.exception))
        log_error.assert_called_once()

    def test_build_msgs_responses_always_uses_responses_typed_user_content(self):
        llm = make_llm()

        messages = llm._build_msgs_responses(
            system_prompt="system",
            user_prompt="hello",
            file_paths=None,
            file_bins=None,
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
        )

    def test_build_msgs_responses_delegates_file_insertion(self):
        llm = make_llm()

        with patch.object(llm, "_insert_bins_to_messages_responses", autospec=True) as inserter:
            messages = llm._build_msgs_responses(
                system_prompt=None,
                user_prompt="hello",
                file_paths=["a.txt"],
                file_bins=[b"a"],
            )

        self.assertEqual(
            messages,
            [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        )
        inserter.assert_called_once_with(
            messages=messages,
            file_paths=["a.txt"],
            file_bins=[b"a"],
        )


class CompatibleCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_compatible_sends_chat_completion_request_and_strips_content(self):
        messages = [{"role": "user", "content": "hello"}]
        client = DummyClient(chat_response=chat_response("  answer  \n"))
        llm = make_llm(client=client, model_id="chat-model")

        with patch.object(llm, "_build_msgs_compatible", autospec=True, return_value=messages) as builder:
            result = await llm.call_compatible(
                system_prompt="system",
                user_prompt="hello",
                temperature=0.25,
                file_paths=["a.txt"],
                file_bins=[b"a"],
            )

        self.assertEqual(result, "answer")
        builder.assert_called_once_with(
            system_prompt="system",
            user_prompt="hello",
            file_paths=["a.txt"],
            file_bins=[b"a"],
        )
        self.assertEqual(
            client.chat_create.calls,
            [
                {
                    "model": "chat-model",
                    "temperature": 0.25,
                    "messages": messages,
                }
            ],
        )

    async def test_call_compatible_accepts_string_response(self):
        client = DummyClient(chat_response="  direct answer\n")
        llm = make_llm(client=client)

        with patch.object(llm, "_build_msgs_compatible", autospec=True, return_value=[]):
            result = await llm.call_compatible(
                system_prompt=None,
                user_prompt="hello",
                temperature=0,
            )

        self.assertEqual(result, "direct answer")

    async def test_call_compatible_wraps_openai_errors(self):
        class DummyOpenAIError(Exception):
            pass

        client = DummyClient(chat_exc=DummyOpenAIError("boom"))
        llm = make_llm(client=client)

        with patch.object(llmapi, "OpenAIError", DummyOpenAIError), patch.object(
            llm, "_build_msgs_compatible", autospec=True, return_value=[]
        ), patch.object(llmapi.simpidlog, "error", autospec=True) as log_error, patch.object(
            llmapi.simpidlog, "wait_for_log_io", autospec=True
        ) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                await llm.call_compatible(
                    system_prompt=None,
                    user_prompt="hello",
                    temperature=0,
                )

        self.assertIn("Failed to call LLM API", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()

    async def test_call_compatible_rejects_empty_content(self):
        client = DummyClient(chat_response=chat_response(None))
        llm = make_llm(client=client)

        with patch.object(llm, "_build_msgs_compatible", autospec=True, return_value=[]), patch.object(
            llmapi.simpidlog, "error", autospec=True
        ) as log_error, patch.object(llmapi.simpidlog, "wait_for_log_io", autospec=True) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                await llm.call_compatible(
                    system_prompt=None,
                    user_prompt="hello",
                    temperature=0,
                )

        self.assertIn("empty content", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()


class ResponsesCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_responses_sends_request_without_tools_when_web_search_is_none(self):
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
        client = DummyClient(responses_response=responses_response("  response answer\n"))
        llm = make_llm(client=client, model_id="responses-model")

        with patch.object(llm, "_build_msgs_responses", autospec=True, return_value=messages) as builder:
            result = await llm.call_responses(
                system_prompt="system",
                user_prompt="hello",
                temperature=0.75,
                web_search="none",
                tools_required=False,
                file_paths=["a.txt"],
                file_bins=[b"a"],
            )

        self.assertEqual(result, "response answer")
        builder.assert_called_once_with(
            system_prompt="system",
            user_prompt="hello",
            file_paths=["a.txt"],
            file_bins=[b"a"],
        )
        self.assertEqual(
            client.responses_create.calls,
            [
                {
                    "model": "responses-model",
                    "temperature": 0.75,
                    "input": messages,
                }
            ],
        )

    async def test_call_responses_adds_web_search_tool_and_required_choice(self):
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
        client = DummyClient(responses_response="  direct response\n")
        llm = make_llm(client=client, model_id="responses-model")

        with patch.object(llm, "_build_msgs_responses", autospec=True, return_value=messages):
            result = await llm.call_responses(
                system_prompt=None,
                user_prompt="hello",
                temperature=1.0,
                web_search="high",
                tools_required=True,
            )

        self.assertEqual(result, "direct response")
        self.assertEqual(
            client.responses_create.calls,
            [
                {
                    "model": "responses-model",
                    "temperature": 1.0,
                    "input": messages,
                    "tools": [
                        {"type": "web_search", "search_context_size": "high"}
                    ],
                    "tool_choice": "required",
                }
            ],
        )

    async def test_call_responses_uses_auto_tool_choice_when_tools_are_not_required(self):
        client = DummyClient(responses_response="answer")
        llm = make_llm(client=client)

        with patch.object(llm, "_build_msgs_responses", autospec=True, return_value=[]):
            await llm.call_responses(
                system_prompt=None,
                user_prompt="hello",
                temperature=0,
                web_search="low",
                tools_required=False,
            )

        self.assertEqual(client.responses_create.calls[0]["tool_choice"], "auto")

    async def test_call_responses_wraps_openai_errors(self):
        class DummyOpenAIError(Exception):
            pass

        client = DummyClient(responses_exc=DummyOpenAIError("boom"))
        llm = make_llm(client=client)

        with patch.object(llmapi, "OpenAIError", DummyOpenAIError), patch.object(
            llm, "_build_msgs_responses", autospec=True, return_value=[]
        ), patch.object(llmapi.simpidlog, "error", autospec=True) as log_error, patch.object(
            llmapi.simpidlog, "wait_for_log_io", autospec=True
        ) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                await llm.call_responses(
                    system_prompt=None,
                    user_prompt="hello",
                    temperature=0,
                )

        self.assertIn("Failed to call LLM API", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()

    async def test_call_responses_rejects_empty_output_text(self):
        client = DummyClient(responses_response=responses_response(""))
        llm = make_llm(client=client)

        with patch.object(llm, "_build_msgs_responses", autospec=True, return_value=[]), patch.object(
            llmapi.simpidlog, "error", autospec=True
        ) as log_error, patch.object(llmapi.simpidlog, "wait_for_log_io", autospec=True) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                await llm.call_responses(
                    system_prompt=None,
                    user_prompt="hello",
                    temperature=0,
                )

        self.assertIn("empty content", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()


class ParseTests(unittest.TestCase):
    def test_parse_json_accepts_plain_json_object(self):
        llm = make_llm()

        self.assertEqual(llm.parse_json('{"ok": true, "n": 3}'), {"ok": True, "n": 3})

    def test_parse_json_accepts_fenced_json_object(self):
        llm = make_llm()

        self.assertEqual(
            llm.parse_json('```json\n{"items": [1, 2], "name": "demo"}\n```'),
            {"items": [1, 2], "name": "demo"},
        )

    def test_parse_json_extracts_object_from_surrounding_text(self):
        llm = make_llm()

        self.assertEqual(
            llm.parse_json('Before answer: {"value": {"nested": true}} after answer.'),
            {"value": {"nested": True}},
        )

    def test_parse_json_rejects_invalid_json(self):
        llm = make_llm()

        with patch.object(llmapi.simpidlog, "error", autospec=True) as log_error, patch.object(
            llmapi.simpidlog, "wait_for_log_io", autospec=True
        ) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                llm.parse_json("not json")

        self.assertIn("Invalid JSON", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()

    def test_parse_json_rejects_valid_non_object_json(self):
        llm = make_llm()

        with patch.object(llmapi.simpidlog, "error", autospec=True) as log_error, patch.object(
            llmapi.simpidlog, "wait_for_log_io", autospec=True
        ) as wait_for_log_io:
            with self.assertRaises(RuntimeError) as ctx:
                llm.parse_json('["not", "an", "object"]')

        self.assertIn("Invalid JSON", str(ctx.exception))
        log_error.assert_called_once()
        wait_for_log_io.assert_called_once()

    def test_parse_codeblock_returns_matching_language_blocks_only(self):
        llm = make_llm()
        text = """
Intro.

```python
print("one")
```

```json
{"ignored": true}
```

```python
x = 2
```
"""

        blocks = llm.parse_codeblock(text, "python")

        self.assertEqual([block.strip() for block in blocks], ['print("one")', "x = 2"])

    def test_parse_codeblock_returns_empty_list_when_language_is_absent(self):
        llm = make_llm()

        self.assertEqual(llm.parse_codeblock("```json\n{}\n```", "python"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
